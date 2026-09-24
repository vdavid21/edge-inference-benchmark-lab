#!/usr/bin/env python3
"""Profile a TensorRT engine and produce its roofline analysis.

    sudo jetson_clocks
    python scripts/profile_model.py models/mobilenetv2_fp16.plan \
        --onnx models/mobilenetv2-12.onnx --precision fp16

--dumpProfile inflates per-layer times, so they need an uninstrumented
reference to scale against. --separateProfileRun supplies one from the same
process: a Performance summary from the unprofiled benchmark run alongside the
per-layer table from the profiled one, with no drift between two commands.

Flags follow NVIDIA's recommended profiling invocation:
  --noDataTransfers   GPU Compute Time then covers only the kernels that the
                      per-layer sum also covers
  --useCudaGraph      how the engine is actually deployed
  --useSpinWait       steadier synchronisation for per-layer timing. On this
                      board it made no measurable difference to benchmark
                      latency — median and p99 both moved 0.5%, inside noise —
                      so it is used for profiling only.

Clocks must be pinned: a roofline assumes fixed ceilings, and under DVFS the
GPU and EMC rates move with demand. Rates are read from tegrastats during the
run rather than from the configured ceiling, because the two can disagree.

Output, mirroring results/energy/:

    results/profiles/<ts>_<model>_<precision>_<mode>_<pinned|dvfs>_<sha>/
        result.json    env, both runs, measured clocks, roofline summary
        profile.json   raw trtexec export
        trtexec.log    full output: Performance summary + per-layer table
        telemetry.csv  clocks and power during the run
        roofline.csv   per-layer time, FLOPs, bytes, AI, attained throughput
        roofline.png
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from framecost.env import capture, warn_if_unsuitable
from framecost.roofline import (analyse, build_graph_info, load_profile, plot,
                                summarise, write_csv)

ROOT = Path(__file__).resolve().parents[1]
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"

# Fallback EMC ceilings, used only when tegrastats is unavailable. Nominal per
# power mode; verify with `sudo jetson_clocks --show | grep EMC`.
EMC_HZ_BY_MODE = {"15W": 2133e6, "25W": 3199e6, "MAXN_SUPER": 3199e6}


def parse_median(log: str, label: str) -> float | None:
    m = re.search(rf"{re.escape(label)}:.*?median = ([\d.]+) ms", log)
    return float(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("--onnx", required=True,
                    help="source ONNX, for layer geometry (shapes and groups)")
    ap.add_argument("--precision", required=True,
                    choices=["fp32", "tf32", "fp16", "int8"])
    ap.add_argument("--shapes", default="input:1x3x224x224")
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--sms", type=int, default=8)
    ap.add_argument("--emc-hz", type=float, default=None,
                    help="state the EMC rate explicitly; also permits an "
                         "unpinned run")
    ap.add_argument("--gpu-hz", type=float, default=None)
    ap.add_argument("--allow-dvfs", action="store_true",
                    help="profile without pinned clocks; roofline ceilings will "
                         "not correspond to a single machine state")
    ap.add_argument("--no-cuda-graph", action="store_true",
                    help="measure without CUDA graphs, exposing per-kernel "
                         "launch overhead in the per-layer times")
    ap.add_argument("--no-spin-wait", action="store_true")
    ap.add_argument("--no-telemetry", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    engine, onnx_path = Path(args.engine), Path(args.onnx)
    for p in (engine, onnx_path):
        if not p.exists():
            sys.exit(f"no such file: {p}")

    env = capture(ROOT)
    pinned = bool(env["clocks"]["gpu_pinned"])

    # A roofline drawn against ceilings the run never reached is worse than no
    # roofline, so an unpinned run must be opted into deliberately.
    if not pinned and not (args.allow_dvfs or args.emc_hz):
        sys.exit(
            "clocks are not pinned — run `sudo jetson_clocks` first.\n"
            "A roofline assumes fixed ceilings; under DVFS the GPU and EMC "
            "rates move with demand and no single roofline applies.\n"
            "To proceed anyway: --allow-dvfs (and say so in the write-up), or "
            "state the rates with --emc-hz / --gpu-hz.")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = env["git"]["sha_short"] or "nogit"
    mode = env["power_mode"]["name"] or "unknown"
    clk_tag = "pinned" if pinned else "dvfs"
    run_dir = (Path(args.out) if args.out else ROOT / "results" / "profiles" /
               f"{ts}_{engine.stem}_{args.precision}_{mode}_{clk_tag}_{sha}")
    run_dir.mkdir(parents=True, exist_ok=True)

    for w in warn_if_unsuitable(env):
        print(f"  WARNING: {w}")
    if not pinned:
        print("  WARNING: running under DVFS — roofline ceilings are nominal")

    print(f"engine     {engine.name}  ({args.precision})")
    print(f"board      {mode}, {clk_tag}")
    print(f"run dir    {run_dir}\n")

    cmd = [TRTEXEC, f"--loadEngine={engine}", f"--shapes={args.shapes}",
           "--warmUp=2000", f"--duration={args.duration}",
           "--noDataTransfers", "--profilingVerbosity=detailed",
           "--dumpProfile", "--separateProfileRun",
           f"--exportProfile={run_dir/'profile.json'}"]
    if not args.no_cuda_graph:
        cmd.append("--useCudaGraph")
    if not args.no_spin_wait:
        cmd.append("--useSpinWait")

    # --- run, sampled -----------------------------------------------------
    print("running trtexec (benchmark run + separate profile run)...")
    sampler = None
    if not args.no_telemetry:
        try:
            from framecost.telemetry import TegraSampler
            sampler = TegraSampler(interval_ms=200).start()
        except Exception as e:
            print(f"           telemetry unavailable ({e}); "
                  f"falling back to configured ceilings")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    log = proc.stdout + proc.stderr

    measured = None
    if sampler is not None:
        sampler.stop()
        measured = sampler.summary()
        sampler.to_csv(run_dir / "telemetry.csv")

    (run_dir / "trtexec.log").write_text(log)
    if "PASSED" not in log:
        print(log[-1500:])
        sys.exit("trtexec did not report PASSED — see trtexec.log")

    # The Performance summary comes from the benchmark run, which
    # --separateProfileRun leaves unprofiled. That is the clean reference.
    true_gpu_ms = parse_median(log, "GPU Compute Time")
    true_lat_ms = parse_median(log, "Latency")
    if true_gpu_ms is None:
        sys.exit("no Performance summary in the log — is --separateProfileRun "
                 "supported by this trtexec build?")
    print(f"           benchmark run: GPU compute {true_gpu_ms:.4f} ms, "
          f"latency {true_lat_ms:.4f} ms")

    # --- clocks: measured where possible ----------------------------------
    gpu_hz_cfg = env["clocks"]["gpu_max_hz"]
    emc_hz_cfg = EMC_HZ_BY_MODE.get(mode)
    gpu_hz_meas = emc_hz_meas = None
    if measured and measured.get("n_samples"):
        gpu_hz_meas = (measured.get("gpu_mhz_mean") or 0) * 1e6 or None
        emc_hz_meas = (measured.get("emc_mhz_mean") or 0) * 1e6 or None

    gpu_hz = args.gpu_hz or gpu_hz_meas or gpu_hz_cfg
    emc_hz = args.emc_hz or emc_hz_meas or emc_hz_cfg
    if not gpu_hz or not emc_hz:
        sys.exit(f"could not determine clocks (gpu={gpu_hz}, emc={emc_hz}); "
                 f"pass --gpu-hz / --emc-hz")

    src = "measured" if (gpu_hz_meas and not args.gpu_hz) else "configured"
    print(f"           clocks ({src}): gpu {gpu_hz/1e6:.0f} MHz, "
          f"emc {emc_hz/1e6:.0f} MHz")
    for label, meas, cfg in (("gpu", gpu_hz_meas, gpu_hz_cfg),
                             ("emc", emc_hz_meas, emc_hz_cfg)):
        if meas and cfg and abs(meas - cfg) / cfg > 0.05:
            print(f"           NOTE: {label} configured {cfg/1e6:.0f} MHz but "
                  f"measured {meas/1e6:.0f} MHz")

    # --- scale ------------------------------------------------------------
    profile = load_profile(run_dir / "profile.json")
    instrumented_ms = sum(e["medianMs"] for e in profile)
    scale = true_gpu_ms / instrumented_ms
    overhead_pct = (1 / scale - 1) * 100
    print(f"           profile run:   {instrumented_ms:.4f} ms -> "
          f"{true_gpu_ms:.4f} ms  (scale {scale:.3f}, "
          f"{overhead_pct:.0f}% instrumentation overhead)")
    # Both runs share launch behaviour, so this figure is instrumentation
    # alone. A value far above ~15% would suggest the profile run is not
    # using CUDA graphs while the benchmark run is.
    if not args.no_cuda_graph and overhead_pct > 25:
        print(f"           NOTE: overhead above 25% suggests the profile run "
              f"may not be using CUDA graphs; per-layer times then include "
              f"launch gaps that the benchmark run does not have")
    print()

    # --- analysis ---------------------------------------------------------
    info = build_graph_info(onnx_path)
    rows, diag = analyse(profile, info, args.precision, scale)
    summary = summarise(rows, diag, args.precision, gpu_hz, emc_hz, args.sms)

    write_csv(rows, run_dir / "roofline.csv")
    plot(rows, summary, run_dir / "roofline.png",
         f"{engine.stem} {args.precision.upper()} — {mode}, "
         f"{gpu_hz/1e6:.0f} MHz{'' if pinned else ' (DVFS)'}")

    result = {
        "captured_utc": ts,
        "engine": str(engine),
        "engine_sha256_16": hashlib.sha256(engine.read_bytes()).hexdigest()[:16],
        "onnx": str(onnx_path),
        "onnx_sha256_16": hashlib.sha256(onnx_path.read_bytes()).hexdigest()[:16],
        "precision": args.precision,
        "trtexec_cmd": " ".join(cmd),
        "clocks_pinned": pinned,
        "clocks": {"gpu_hz_used": gpu_hz, "emc_hz_used": emc_hz,
                   "gpu_hz_configured": gpu_hz_cfg, "emc_hz_nominal": emc_hz_cfg,
                   "gpu_hz_measured": gpu_hz_meas, "emc_hz_measured": emc_hz_meas,
                   "source": src},
        "benchmark_run": {"gpu_compute_p50_ms": true_gpu_ms,
                          "latency_p50_ms": true_lat_ms,
                          "cuda_graph": not args.no_cuda_graph,
                          "data_transfers": False},
        "profile_run": {"total_ms": round(instrumented_ms, 4),
                        "scale_factor": round(scale, 4),
                        "instrumentation_overhead_pct": round(overhead_pct, 1)},
        "roofline": summary,
        "telemetry": measured,
        "env": env,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))

    print(f"{'peak':<22}{summary['peak_gflops']/1000:.2f} T/s")
    print(f"{'bandwidth':<22}{summary['bandwidth_gbs']:.1f} GB/s")
    print(f"{'ridge point':<22}{summary['ridge_flop_per_byte']:.0f} FLOP/byte")
    print(f"{'compute layers':<22}{summary['n_compute_layers']}")
    print(f"{'memory-bound':<22}{summary['n_memory_bound']}/"
          f"{summary['n_compute_layers']} ({summary['pct_memory_bound']:.0f}%)")
    print(f"{'AI min/med/max':<22}{summary['ai_min']} / {summary['ai_median']} / "
          f"{summary['ai_max']}")
    print(f"{'peak util med/best':<22}{summary['peak_util_median_pct']}% / "
          f"{summary['peak_util_best_pct']}%")
    print(f"{'overhead time':<22}{summary['overhead_pct_of_time']}%")
    print(f"{'unattributed time':<22}{summary['unattributed_pct_of_time']}%")

    if diag["unmatched_node_indices"]:
        frac = 100 * diag["unmatched_flops"] / max(1, diag["total_onnx_flops"])
        print(f"\n  {len(diag['unmatched_node_indices'])} ONNX compute nodes "
              f"unmatched ({frac:.1f}% of graph FLOPs) — likely fused into "
              f"myelin kernels whose names carry no node reference")

    print("\ntop 5 by time:")
    for r in sorted(rows, key=lambda r: -r["time_ms"])[:5]:
        ai = f"{r['ai']:.1f}" if r["ai"] else "—"
        gf = f"{r['gflops']:.0f}" if r["gflops"] else "—"
        print(f"  {r['pct']:5.1f}%  {r['time_ms']*1000:7.1f} us  "
              f"AI {ai:>7}  {gf:>6} G/s  {r['kind']:<13} {r['layer'][:46]}")

    print(f"\nwrote {run_dir}")


if __name__ == "__main__":
    main()
