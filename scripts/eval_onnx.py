#!/usr/bin/env python3
"""FP32 top-1/top-5 baseline via ONNX Runtime (CPU). Validates preprocessing."""
import json, sys, time, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, onnxruntime as ort
from framecost.preprocess import preprocess_image

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "mobilenetv2-12.onnx"
SPLIT = ROOT / "data" / "split_v1.json"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0   # 0 = all

split = json.loads(SPLIT.read_text())
items = split["eval"]
random.Random(1234).shuffle(items)
if LIMIT:
    items = items[:LIMIT]
print(f"model {MODEL.name}  eval n={len(items)}  fingerprint {split['eval_fingerprint']}")

sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name

top1 = top5 = 0
t0 = time.time()
for i, (path, label) in enumerate(items, 1):
    x = preprocess_image(path)[None]                    # (1,3,224,224)
    logits = sess.run(None, {iname: x})[0][0]
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
