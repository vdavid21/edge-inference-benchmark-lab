#!/usr/bin/env python3
"""Top-1/top-5 accuracy of a TensorRT engine on the ImageNetV2 eval split.

Deliberately mirrors scripts/eval_onnx.py line for line: same split file, same
preprocessing, same shuffle seed, same metric code. The only thing that differs
is the inference backend. That is what makes the comparison meaningful — any
accuracy gap is attributable to TensorRT, not to an incidental difference in
how the two scripts were written.

Usage:
    python scripts/eval_trt.py models/mobilenetv2_fp32.plan          # full 9000
    python scripts/eval_trt.py models/mobilenetv2_fp16.plan 2000     # subset
"""

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from framecost.preprocess import preprocess_image
from framecost.trt_runner import TRTRunner

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "data" / "split_v1.json"

engine_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "models" / "mobilenetv2_fp32.plan")
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0

split = json.loads(SPLIT.read_text())

# Same shuffle and seed as eval_onnx.py, so a subset here is the identical
# subset there. Shuffle unconditionally: the split is ordered by class index,
# so truncating an unshuffled list samples only the low-numbered (animal)
# classes and inflates accuracy by ~10 points.
items = list(split["eval"])
random.Random(1234).shuffle(items)
if LIMIT:
    items = items[:LIMIT]

with TRTRunner(engine_path) as runner:
    meta = runner.describe()
    print(f"engine      {Path(engine_path).name}")
    print(f"engine sha  {meta['engine_sha256_16']}   {meta['engine_size_bytes']/1e6:.2f} MB")
    print(f"device mem  {meta['device_memory_bytes']/1e6:.2f} MB   layers {meta['num_layers']}")
    print(f"input       {meta['inputs'][0]['shape']} {meta['inputs'][0]['dtype']}")
    print(f"eval n={len(items)}  fingerprint {split['eval_fingerprint']}")
    print()

    top1 = top5 = 0
    t0 = time.time()
    for i, (path, label) in enumerate(items, 1):
        x = preprocess_image(path)[None]
        logits = runner.infer(x).reshape(-1)
        order = np.argsort(-logits)[:5]
        top1 += int(order[0] == label)
        top5 += int(label in order)
        if i % 500 == 0:
            print(f"  {i:5d}  top1 {100*top1/i:5.2f}%  top5 {100*top5/i:5.2f}%")

n = len(items)
p = top1 / n
ci = 1.96 * (p * (1 - p) / n) ** 0.5
print(f"\ntop-1 {100*p:.2f}% ± {100*ci:.2f}   top-5 {100*top5/n:.2f}%")
print(f"n={n}  elapsed {time.time()-t0:.0f}s")
