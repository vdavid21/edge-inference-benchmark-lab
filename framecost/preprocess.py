"""ImageNet preprocessing — the single source of truth for every arm of the benchmark.

Why this module exists
----------------------
Calibration, TensorRT inference, and accuracy evaluation must apply *bit-identical*
preprocessing. If they don't, the accuracy delta you measure between FP32 and INT8
is contaminated by a preprocessing difference and the whole comparison is void.

So: nothing in this project resizes or normalises an image except through here.

Pipeline (torchvision-standard, which is what the ONNX model zoo MobileNetV2 expects):
    1. Resize shorter side to 256, preserving aspect ratio, bilinear + antialias
    2. Center-crop 224x224
    3. Convert to float32 in [0, 1]
    4. Normalise per-channel with ImageNet mean/std
    5. Transpose HWC -> CHW

Deliberate choice: PIL, not OpenCV. cv2.resize and PIL's resize produce different
pixel values (different kernel support and antialiasing behaviour). torchvision's
reference numbers were produced with the PIL path, so PIL is what reproduces them.
Mixing the two is a classic source of 1-2% phantom "quantisation loss".

Usage
-----
    from preprocess import preprocess_image, list_imagenetv2, stratified_split

    samples = list_imagenetv2("data/imagenetv2-matched-frequency-format-val")
    calib, evalset = stratified_split(samples, n_per_class=1, seed=0)
    x = preprocess_image(calib[0][0])          # (3, 224, 224) float32
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# --- Constants. Change these and every downstream number changes. -------------

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RESIZE_SHORT_SIDE = 256
CROP_SIZE = 224

# Recorded in the result row so a CSV can be traced back to a pipeline version.
PREPROCESS_VERSION = "imagenet-torchvision-v1"


def preprocess_image(source) -> np.ndarray:
    """Load and preprocess one image.

    Args:
        source: a path (str or Path) or an already-open PIL.Image.

    Returns:
        float32 array of shape (3, 224, 224), normalised. No batch dimension —
        the caller adds it, because batching policy differs between the
        calibration loader and the benchmark loop.
    """
    img = Image.open(source) if not isinstance(source, Image.Image) else source

    # Grayscale, CMYK and palette images exist in ImageNet. Force 3-channel RGB
    # or the channel count silently varies and downstream shapes break.
    img = img.convert("RGB")

    # 1. Resize shorter side to 256, preserving aspect ratio.
    w, h = img.size
    if w < h:
        new_w = RESIZE_SHORT_SIDE
        new_h = int(round(h * RESIZE_SHORT_SIDE / w))
    else:
        new_h = RESIZE_SHORT_SIDE
        new_w = int(round(w * RESIZE_SHORT_SIDE / h))
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # 2. Center crop.
    left = (new_w - CROP_SIZE) // 2
    top = (new_h - CROP_SIZE) // 2
    img = img.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))

    # 3. To float32 [0, 1].
    arr = np.asarray(img, dtype=np.float32) / 255.0      # (H, W, 3)

    # 4. Normalise.
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

    # 5. HWC -> CHW, contiguous (TensorRT wants a contiguous host buffer).
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def list_imagenetv2(root) -> list[tuple[Path, int]]:
    """Enumerate an extracted ImageNetV2 directory.

    Layout is  <root>/<class_index>/<image>.jpeg  where class_index is the
    integer ImageNet-1k label 0..999 — so the label is the directory name and
    no WNID mapping file is needed.

    Returns a list of (path, label), sorted for determinism.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory")

    samples: list[tuple[Path, int]] = []
    for class_dir in sorted(root.iterdir(), key=lambda p: p.name):
        if not class_dir.is_dir():
            continue
        try:
            label = int(class_dir.name)
        except ValueError:
            continue  # skip anything that isn't a class folder
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                samples.append((img_path, label))

    if not samples:
        raise RuntimeError(f"no images found under {root}")
    return samples


def stratified_split(
    samples: list[tuple[Path, int]],
    n_per_class: int = 1,
    seed: int = 0,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Split into a calibration set and a DISJOINT evaluation set.

    Stratified: takes n_per_class images from each class, so the calibration
    set covers the full label distribution rather than whatever a random draw
    happened to hit. With ImageNetV2's 10 images per class, n_per_class=1 gives
    a 1000-image calibration set and leaves 9000 for evaluation.

    Disjointness is the point. Calibrating on data you later evaluate on is
    test-set leakage and inflates the reported accuracy.
    """
    by_class: dict[int, list[tuple[Path, int]]] = defaultdict(list)
    for path, label in samples:
        by_class[label].append((path, label))

    rng = random.Random(seed)
    calib: list[tuple[Path, int]] = []
    evalset: list[tuple[Path, int]] = []

    for label in sorted(by_class):
        items = sorted(by_class[label], key=lambda t: str(t[0]))
        rng.shuffle(items)
        calib.extend(items[:n_per_class])
        evalset.extend(items[n_per_class:])

    return calib, evalset


def split_fingerprint(split: list[tuple[Path, int]]) -> str:
    """SHA256 over the sorted file names in a split.

    Put this in the result row. It proves two runs used the same images, and
    catches the case where a re-download or a different seed silently changed
    the split underneath you.
    """
    h = hashlib.sha256()
    for path, label in sorted(split, key=lambda t: str(t[0])):
        h.update(f"{path.name}:{label}\n".encode())
    return h.hexdigest()[:16]


def save_split(calib, evalset, out_path) -> None:
    """Persist the split so it survives a re-download or a machine change."""
    payload = {
        "preprocess_version": PREPROCESS_VERSION,
        "resize_short_side": RESIZE_SHORT_SIDE,
        "crop_size": CROP_SIZE,
        "mean": IMAGENET_MEAN.tolist(),
        "std": IMAGENET_STD.tolist(),
        "n_calib": len(calib),
        "n_eval": len(evalset),
        "calib_fingerprint": split_fingerprint(calib),
        "eval_fingerprint": split_fingerprint(evalset),
        "calib": [[str(p), l] for p, l in calib],
        "eval": [[str(p), l] for p, l in evalset],
    }
    Path(out_path).write_text(json.dumps(payload, indent=2))


def calibration_batches(calib, batch_size: int = 8):
    """Yield (N, 3, 224, 224) float32 batches. Used by the INT8 calibrator."""
    batch: list[np.ndarray] = []
    for path, _label in calib:
        batch.append(preprocess_image(path))
        if len(batch) == batch_size:
            yield np.stack(batch)
            batch = []
    if batch:
        yield np.stack(batch)


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "data/imagenetv2-matched-frequency-format-val"
    samples = list_imagenetv2(root)
    calib, evalset = stratified_split(samples, n_per_class=1, seed=0)

    print(f"total images     : {len(samples)}")
    print(f"classes          : {len({l for _, l in samples})}")
    print(f"calibration set  : {len(calib)}  fingerprint {split_fingerprint(calib)}")
    print(f"evaluation set   : {len(evalset)}  fingerprint {split_fingerprint(evalset)}")

    overlap = {str(p) for p, _ in calib} & {str(p) for p, _ in evalset}
    print(f"overlap          : {len(overlap)}  (must be 0)")
    assert not overlap, "calibration and evaluation sets overlap"

    x = preprocess_image(calib[0][0])
    print(f"\nsample           : {calib[0][0].name}  label {calib[0][1]}")
    print(f"shape / dtype    : {x.shape} {x.dtype}")
    print(f"min / max / mean : {x.min():.4f} {x.max():.4f} {x.mean():.4f}")
    print(f"array sha256[:16]: {hashlib.sha256(x.tobytes()).hexdigest()[:16]}")
    print("\nRe-run this on any machine; the array hash must match.")
