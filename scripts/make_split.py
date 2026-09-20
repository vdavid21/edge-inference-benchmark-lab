#!/usr/bin/env python3
"""Generate and persist the calibration/evaluation split. Run once; commit the output."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from framecost.preprocess import (
    list_imagenetv2, stratified_split, save_split, split_fingerprint
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "imagenetv2-matched-frequency-format-val"
OUT  = ROOT / "data" / "split_v1.json"

if OUT.exists():
    sys.exit(f"{OUT} already exists — delete it deliberately if you mean to regenerate")

samples = list_imagenetv2(DATA)
calib, evalset = stratified_split(samples, n_per_class=1, seed=0)
save_split(calib, evalset, OUT)

print(f"calib {len(calib):5d}  {split_fingerprint(calib)}")
print(f"eval  {len(evalset):5d}  {split_fingerprint(evalset)}")
print(f"wrote {OUT}")
