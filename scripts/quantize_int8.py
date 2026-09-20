#!/usr/bin/env python3
"""Insert explicit INT8 Q/DQ nodes into an ONNX graph via ModelOpt PTQ.

Stage 1 of 2. This produces a quantised ONNX file; TensorRT then compiles it
into an engine. Splitting the two matters because it makes the quantisation
decisions inspectable: you can open the output graph and see exactly where the
QuantizeLinear/DequantizeLinear pairs landed and what scales they carry.

Explicit vs implicit
--------------------
This is the explicit path. The scales live in the graph, so TensorRT has no
discretion about which layers run in INT8. The implicit path
(IInt8EntropyCalibrator2) is deprecated as of TRT 10.1 and lets TensorRT
silently keep layers in FP16 when that is faster — fine for shipping, useless
for a benchmark where you need to know what actually executed.

Calibration hygiene
-------------------
Images come from split_v1.json's calibration set, which is stratified across
all 1000 classes and disjoint from the evaluation set. Preprocessing goes
through framecost.preprocess, the same code path as evaluation. Both matter:
calibrating on eval data leaks the test set, and calibrating with different
preprocessing measures the wrong activation ranges.

Usage:
    python scripts/quantize_int8.py                          # 512 imgs, entropy
    python scripts/quantize_int8.py --n 128 --method max
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from framecost.preprocess import preprocess_image

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(ROOT / "models" / "mobilenetv2-12.onnx"))
    ap.add_argument("--split", default=str(ROOT / "data" / "split_v1.json"))
    ap.add_argument("--n", type=int, default=512,
                    help="calibration images (64/128/512 are the sweep points)")
    ap.add_argument("--method", default="entropy",
                    help="entropy | max  (see ModelOpt docs for the full list)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or str(
        ROOT / "models" / f"mobilenetv2_int8_{args.method}_n{args.n}.onnx"
    )

    # --- calibration data -----------------------------------------------
    split = json.loads(Path(args.split).read_text())
    calib_items = split["calib"][: args.n]

    print(f"calibration images : {len(calib_items)}")
    print(f"calib fingerprint  : {split['calib_fingerprint']}  (full set)")
    print(f"method             : {args.method}")

    t0 = time.time()
    batch = np.stack([preprocess_image(p) for p, _ in calib_items])
    print(f"preprocessed       : {batch.shape} {batch.dtype} "
          f"({batch.nbytes/1e6:.0f} MB) in {time.time()-t0:.1f}s")

    # Sanity: calibration data must look like the eval data. If these stats
    # are wildly different from what evaluation sees, the scales will be wrong.
    print(f"value range        : {batch.min():.3f} .. {batch.max():.3f}  "
          f"mean {batch.mean():.3f}  std {batch.std():.3f}")

    # --- quantise --------------------------------------------------------
    from modelopt.onnx.quantization import quantize

    print("\nquantising (CPU calibration — this takes a few minutes)...")
    t0 = time.time()
    quantize(
        onnx_path=args.onnx,
        calibration_data=batch,
        calibration_method=args.method,
        # onnxruntime-gpu has no prebuilt sm_87 wheel, so calibration must run
        # on the CPU execution provider. Leaving this at the default would try
        # CUDA/TRT providers that are not available here.
        calibration_eps=["cpu"],
        output_path=out_path,
        quantize_mode="int8",
    )
    print(f"done in {time.time()-t0:.0f}s -> {out_path}")

    # --- report what landed in the graph ---------------------------------
    import collections

    import onnx

    m = onnx.load(out_path)
    counts = collections.Counter(n.op_type for n in m.graph.node)
    print(f"\nnodes              : {len(m.graph.node)}")
    print(f"QuantizeLinear     : {counts.get('QuantizeLinear', 0)}")
    print(f"DequantizeLinear   : {counts.get('DequantizeLinear', 0)}")
    print(f"Conv               : {counts.get('Conv', 0)}")
    print(f"Gemm               : {counts.get('Gemm', 0)}")
    print("\nA Conv with no Q/DQ pair on its inputs will run in FP16/FP32, "
          "not INT8.")


if __name__ == "__main__":
    main()
