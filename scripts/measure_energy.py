#!/usr/bin/env python3
"""Measure energy per inference for a TensorRT engine.

Runs trtexec under tegrastats sampling, then divides integrated energy by the
number of inferences trtexec reports. Captures an idle baseline first so both
total and marginal energy can be reported.

    python scripts/measure_energy.py models/mobilenetv2_fp16.plan
    python scripts/measure_energy.py models/mobilenetv2_int8.plan --duration 120

Close everything else on the board first — a GUI monitor or a browser will
show up in VDD_IN and in the latency tail.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from framecost.env import capture, warn_if_unsuitable
from framecost.telemetry import TegraSampler

ROOT = Path(__file__).resolve().parents[1]
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"


def parse_trtexec(log: str) -> dict:
    """Pull the numbers we need out of trtexec's summary block."""
    out = {}
    if (m := re.search(r"Timing trace has (\d+) queries over ([\d.]+) s", log)):
        out["n_inferences"] = int(m.group(1))
        out["trt_duration_s"] = float(m.group(2))
    if (m := re.search(r"Throughput: ([\d.]+) qps", log)):
        out["throughput_qps"] = float(m.group(1))
    for label, key in [("Latency", "latency"), ("GPU Compute Time", "gpu_compute"),
                       ("Enqueue Time", "enqueue")]:
        m = re.search(
            rf"{re.escape(label)}: min = ([\d.]+) ms, max = ([\d.]+) ms, "
            rf"mean = ([\d.]+) ms, median = ([\d.]+) ms, "
            rf"percentile\(90%\) = ([\d.]+) ms, percentile\(95%\) = ([\d.]+) ms, "
            rf"percentile\(99%\) = ([\d.]+) ms", log)
        if m:
            for i, stat in enumerate(
                    ["min", "max", "mean", "p50", "p90", "p95", "p99"], start=1):
                out[f"{key}_{stat}_ms"] = float(m.group(i))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("--duration", type=int, default=120,
                    help="seconds of timed inference (longer = steadier power)")
    ap.add_argument("--idle-seconds", type=float, default=30.0)
    ap.add_argument("--interval-ms", type=int, default=100)
    ap.add_argument("--shapes", default="input:1x3x224x224")
    ap.add_argument("--no-cuda-graph", action="store_true")
    ap.add_argument("--skip-idle", action="store_true",
                    help="reuse a previous idle baseline instead of remeasuring")
    ap.add_argument("--idle-mw", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    engine = Path(args.engine)
    if not engine.exists():
        sys.exit(f"no such engine: {engine}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    env = capture(ROOT)
    sha = env["git"]["sha_short"] or "nogit"
    run_dir = Path(args.out) if args.out else ROOT / "results" / "energy" / f"{ts}_{sha}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for w in warn_if_unsuitable(env):
        print(f"  WARNING: {w}")
    print(f"\nengine    {engine.name}")
    print(f"power     {env['power_mode']['name']}  "
          f"gpu {env['clocks']['gpu_cur_hz']} Hz  emc {env['clocks']['emc_cur_hz']} Hz")
    print(f"run dir   {run_dir}\n")

    # --- idle baseline ---------------------------------------------------
    idle_mw = args.idle_mw
    idle_summary = None
    if not args.skip_idle and idle_mw is None:
        print(f"idle baseline, {args.idle_seconds:.0f}s — leave the board alone...")
        with TegraSampler(interval_ms=args.interval_ms) as t:
            time.sleep(args.idle_seconds)
        idle_summary = t.summary()
        idle_mw = idle_summary["vdd_in_mean_mw"]
        t.to_csv(run_dir / "idle_telemetry.csv")
        print(f"  idle VDD_IN {idle_mw:.0f} mW   "
              f"tj {idle_summary.get('temp_tj_max_c')} C\n")

    # --- the measured run ------------------------------------------------
    cmd = [TRTEXEC, f"--loadEngine={engine}", f"--shapes={args.shapes}",
           "--warmUp=2000", f"--duration={args.duration}"]
    if not args.no_cuda_graph:
        cmd.append("--useCudaGraph")

    print(f"running trtexec for ~{args.duration}s under telemetry...")
    with TegraSampler(interval_ms=args.interval_ms) as t:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    log = proc.stdout + proc.stderr

    (run_dir / "bench.log").write_text(log)
    t.to_csv(run_dir / "telemetry.csv")

    if "PASSED" not in log:
        print(log[-2000:])
        sys.exit("trtexec did not report PASSED — see bench.log")

    trt = parse_trtexec(log)
    telem = t.summary()
    energy = t.energy(trt.get("n_inferences", 0), idle_mw=idle_mw)

    # The sampler covers process startup, engine deserialisation and warm-up as
    # well as the timed region. At duration >= 120s that overhead is a few
    # percent; it is recorded rather than corrected for.
    result = {
        "captured_utc": ts,
        "engine": str(engine),
        "engine_sha256_16": None,
        "cuda_graph": not args.no_cuda_graph,
        "trtexec": trt,
        "telemetry": telem,
        "energy": energy,
        "idle": idle_summary,
        "env": env,
    }
    import hashlib
    result["engine_sha256_16"] = hashlib.sha256(engine.read_bytes()).hexdigest()[:16]

    (run_dir / "result.json").write_text(json.dumps(result, indent=2))

    # --- report ----------------------------------------------------------
    print(f"\n{'':<24}{'value'}")
    print(f"{'latency p50':<24}{trt.get('latency_p50_ms')} ms")
    print(f"{'latency p99':<24}{trt.get('latency_p99_ms')} ms")
    print(f"{'throughput':<24}{trt.get('throughput_qps')} qps")
    print(f"{'inferences':<24}{trt.get('n_inferences')}")
    print()
    print(f"{'VDD_IN mean':<24}{telem.get('vdd_in_mean_mw')} mW")
    print(f"{'VDD_CPU_GPU_CV mean':<24}{telem.get('vdd_cpu_gpu_cv_mean_mw')} mW")
    print(f"{'VDD_SOC mean':<24}{telem.get('vdd_soc_mean_mw')} mW")
    print(f"{'idle VDD_IN':<24}{idle_mw} mW")
    print()
    print(f"{'energy total':<24}{energy.get('mj_per_inference')} mJ/frame")
    print(f"{'energy marginal':<24}{energy.get('mj_per_inference_marginal')} mJ/frame")
    print()
    print(f"{'tj start / max':<24}{telem.get('temp_start_c')} / "
          f"{telem.get('temp_tj_max_c')} C")
    print(f"{'gpu clock min':<24}{telem.get('gpu_mhz_min')} MHz  "
          f"(throttling if below nominal)")
    print(f"\nwrote {run_dir}/result.json")
    print("\nRead the wall meter now and note it — VDD_IN excludes PSU loss.")


if __name__ == "__main__":
    main()
