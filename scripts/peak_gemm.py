#!/usr/bin/env python3
"""Measure the achievable arithmetic ceiling with large square GEMMs.

    sudo jetson_clocks
    python scripts/peak_gemm.py --sizes 2048

Why
---
The roofline ceilings are derived, not measured. INT8 is anchored to the Orin
datasheet (67 sparse TOPS at 1020 MHz back-solves to 4096 OP/clk/SM, consistent
across Orin Nano, Orin Nano Super and AGX Orin). FP16 is taken as half of INT8,
which is correct for FP16 accumulation.

But consumer GA10x halves FP16 throughput when accumulating in FP32 — an RTX
3080 does 119 TFLOP/s with FP16 accumulate and 59.5 with FP32 accumulate, while
A100 has no such penalty. Whether Orin's GA10B inherits the GeForce behaviour,
and which mode TensorRT selects, is unverified. If the penalty applies, the FP16
ceiling is 7.4 TFLOP/s rather than 14.8, and every FP16 "% of peak" figure is
wrong by 2x.

Method
------
A square N x N matmul has FLOPs = 2N^3 and traffic = 3N^2 elements, so its
arithmetic intensity is N / (3 * element_bytes / 2). At N = 2048 in FP16 that is
~683 FLOP/byte against a ridge of 145 — deeply compute-bound, the opposite
regime from MobileNetV2, so attained throughput is limited by arithmetic alone.

Three things make the result trustworthy rather than merely suggestive:

  FP32 IS THE CONTROL. It runs on the CUDA cores — 128 lanes x 2 FLOP x 8 SMs x
  clock — with no Tensor Core and no accumulate ambiguity. If FP32 reaches ~85%
  of that, the harness is sound. If it does not, the benchmark is at fault and
  the FP16 number says nothing about the hardware.

  THE VERDICT IS FALSIFICATION, NOT NEAREST-FIT. Any harness inefficiency biases
  attained throughput downward, so "closer to 7.4 than to 14.8" can be produced
  by a bad measurement. Exceeding 7.4 cannot: a roof cannot be beaten. So the
  only affirmative conclusion available is "no halving", and failing to reach it
  is reported as inconclusive rather than as evidence of a penalty.

  EACH ARM IS TYPED AT ITS OWN PRECISION. An FP32-typed ONNX built with --fp16
  makes TensorRT insert cast kernels at the network boundary; at N = 4096 that
  is ~200 MB of extra traffic, roughly 18% of runtime, biasing the FP16 arm
  downward — precisely the direction that would fake a penalty.

No timing cache
---------------
Tactic selection is empirical and its tails are stochastic: the same ONNX can
get different kernels on different builds. A timing cache freezes whichever
tactic won when the cache was first populated — possibly at a different thermal
state, possibly before clocks were pinned. A suboptimal frozen tactic biases the
peak downward, the same direction as the cast overhead above, so both push
toward falsely concluding a penalty exists.

It is also a reproducibility hole: with a cache, two identical invocations give
different numbers depending on whether a gitignored file happens to exist, and
the output would not record which happened.

So builds run uncached by default. --timing-cache exists for iterating on the
script itself; it prints a warning, and records timing_cache_used in the JSON so
a cached number can never be mistaken for a reported one. If builds are slow,
reduce N rather than reaching for the cache.
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

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from framecost.env import capture

ROOT = Path(__file__).resolve().parents[1]
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"

# FLOP (or OP) per SM per clock, dense. FP32 on CUDA cores; the rest on
# Tensor Cores, assuming no FP32-accumulate penalty.
PER_SM_CLK = {"fp32": 256, "tf32": 1024, "fp16": 2048}
ONNX_DTYPE = {"fp32": TensorProto.FLOAT, "tf32": TensorProto.FLOAT,
              "fp16": TensorProto.FLOAT16}
NP_DTYPE = {"fp32": np.float32, "tf32": np.float32, "fp16": np.float16}
ELEM_BYTES = {"fp32": 4, "tf32": 4, "fp16": 2}
SEED = 20260926


def make_matmul(n: int, precision: str, path: Path) -> None:
    """A single N x N matmul with a constant right-hand side, typed natively.

    Typing matters: an FP32 graph built with --fp16 gets cast kernels at the
    boundary, and their DRAM traffic lands inside GPU Compute Time.
    """
    dt, npdt = ONNX_DTYPE[precision], NP_DTYPE[precision]
    a = helper.make_tensor_value_info("A", dt, [n, n])
    y = helper.make_tensor_value_info("Y", dt, [n, n])
    # Seeded, so the generated ONNX hashes are stable across runs. Small
    # magnitudes keep FP16 well inside range — no overflow, no denormals.
    rng = np.random.default_rng(SEED)
    b = numpy_helper.from_array((rng.standard_normal((n, n)) * 0.01).astype(npdt), "B")
    node = helper.make_node("MatMul", ["A", "B"], ["Y"])
    graph = helper.make_graph([node], f"matmul_{n}_{precision}", [a], [y], [b])
    onnx.save(helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)]), str(path))


def compute_phase_stats(sampler, gpu_hz: float, duration: int,
                        util_threshold: int = 50) -> dict | None:
    """Power, clocks and temperature during the timed region only.

    The sampler spans the whole trtexec call, and that call is GPU-busy almost
    throughout: TensorRT autotunes by executing candidate kernels on the device,
    so the build phase is not idle and cannot be separated by utilisation alone.
    Autotuning runs hundreds of short kernels at varying efficiency, so a clock
    dip there says nothing about the steady-state measurement — including those
    samples would raise false throttle flags and skew load power.

    So the window is anchored by time: the timed region is the last `duration`
    seconds before the process exits. The utilisation filter is kept only to
    drop teardown samples at the very end.

    `throttled` compares the minimum clock inside that window against the pinned
    ceiling. With clocks pinned, a meaningful dip is the board pulling back —
    thermal, or the current limiter that a dense GEMM can trip.
    """
    if not sampler.samples:
        return None

    t_end = sampler.samples[-1]["t"]
    window_start = t_end - duration
    busy = [s for s in sampler.samples
            if s["t"] >= window_start
            and (s.get("gpu_util_pct") or 0) >= util_threshold]
    if not busy:
        return None

    vdd = np.array([s["power_mw"].get("VDD_IN", np.nan) for s in busy], dtype=float)
    vdd = vdd[~np.isnan(vdd)]
    gpu = [s["gpu_mhz"] for s in busy if s.get("gpu_mhz")]
    tj = [s["temp_c"]["tj"] for s in busy if s.get("temp_c", {}).get("tj")]

    ceiling_mhz = gpu_hz / 1e6
    gpu_min = min(gpu) if gpu else None
    return {
        "window": "last_timed_region",
        "window_seconds": duration,
        "n_samples_total": len(sampler.samples),
        "n_samples_in_window": len(busy),
        "vdd_in_mean_mw": round(float(vdd.mean()), 1) if vdd.size else None,
        "vdd_in_max_mw": round(float(vdd.max()), 1) if vdd.size else None,
        "gpu_mhz_mean": round(float(np.mean(gpu)), 1) if gpu else None,
        "gpu_mhz_min": gpu_min,
        "gpu_mhz_ceiling": round(ceiling_mhz, 1),
        "temp_tj_max_c": round(max(tj), 2) if tj else None,
        "throttled": bool(gpu_min is not None and gpu_min < 0.95 * ceiling_mhz),
    }


def run_trtexec(onnx_path: Path, n: int, precision: str, duration: int,
                out_dir: Path, cache_path: Path | None, obey: bool,
                workspace_mb: int, telemetry: bool, gpu_hz: float) -> dict:
    layer_info = out_dir / f"layers_{n}_{precision}.json"
    # No --shapes/--minShapes/--optShapes/--maxShapes: the generated ONNX is
    # fully static [n, n], so there are no dynamic dimensions to build an
    # optimization profile over, and passing them fails config setup.
    cmd = [TRTEXEC, f"--onnx={onnx_path}",
           "--noDataTransfers", "--useCudaGraph", "--useSpinWait",
           "--warmUp=2000", f"--duration={duration}",
           f"--memPoolSize=workspace:{workspace_mb}M",
           "--profilingVerbosity=detailed",
           f"--exportLayerInfo={layer_info}"]
    if cache_path is not None:
        # Opt-in only, and never for a number that will be reported. See the
        # note in the module docstring.
        cmd.append(f"--timingCacheFile={cache_path}")
    if precision == "fp32":
        cmd.append("--noTF32")          # true FP32 on the CUDA cores
    elif precision == "fp16":
        cmd.append("--fp16")
        if obey:
            # --fp16 only *permits* FP16; TensorRT may still pick an FP32
            # tactic. Constraining it makes a wrong-precision run fail loudly
            # instead of quietly measuring the wrong thing.
            cmd += ["--precisionConstraints=obey", "--layerPrecisions=*:fp16"]
    # tf32: TensorRT's Ampere default, no flag

    # A dense GEMM is the most power-hungry workload this board runs, so it can
    # trip the current limit that a memory-bound network never approaches.
    # Sampling turns "I saw a warning" into a recorded observation.
    sampler = None
    if telemetry:
        try:
            from framecost.telemetry import TegraSampler
            sampler = TegraSampler(interval_ms=200).start()
        except Exception as e:
            # Never silent: without telemetry a throttled run looks identical
            # to a clean one, and the number would be reported as if
            # unconstrained.
            print(f"         telemetry unavailable ({type(e).__name__}: {e}); "
                  f"throttling will go undetected — try `sudo -v` first")
            sampler = None

    p = subprocess.run(cmd, capture_output=True, text=True)

    phase = None
    if sampler is not None:
        sampler.stop()
        phase = compute_phase_stats(sampler, gpu_hz, duration)
        sampler.to_csv(out_dir / f"telemetry_{n}_{precision}.csv")

    log = p.stdout + p.stderr
    log_path = out_dir / f"trtexec_{n}_{precision}.log"
    log_path.write_text(log)

    def diagnose() -> dict:
        # The first error line is the cause; a blind tail of the log usually
        # catches only the cascade of consequences after it.
        errs = [l.strip() for l in log.splitlines() if "[E]" in l or "[W]" in l]
        return {"ok": False, "errors": errs[:4], "log_path": str(log_path)}

    if "PASSED" not in log:
        return diagnose()
    m = re.search(r"GPU Compute Time:.*?median = ([\d.]+) ms", log)
    if not m:
        return diagnose()

    # Evidence, not assumption: what precision did the kernel actually use?
    kernel_dtypes = []
    if layer_info.exists():
        try:
            li = json.loads(layer_info.read_text())
            layers = li.get("Layers", li) if isinstance(li, dict) else li
            for lyr in layers:
                if isinstance(lyr, dict):
                    for o in lyr.get("Outputs", []):
                        if isinstance(o, dict):
                            kernel_dtypes.append(
                                o.get("Format/Datatype", o.get("Datatype", "?")))
        except Exception:
            pass

    return {"ok": True, "gpu_ms": float(m.group(1)),
            "kernel_dtypes": kernel_dtypes, "telemetry": phase}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[2048])
    ap.add_argument("--precisions", nargs="+", default=["fp32", "tf32", "fp16"],
                    choices=list(PER_SM_CLK),
                    help="int8 is not available here: a bare MatMul graph has "
                         "no Q/DQ nodes, so TensorRT has no scales to quantise "
                         "with. Measuring the INT8 ceiling needs a quantised "
                         "ONNX (see scripts/quantize_int8.py).")
    ap.add_argument("--duration", type=int, default=15)
    ap.add_argument("--sms", type=int, default=8)
    ap.add_argument("--workspace-mb", type=int, default=1024)
    ap.add_argument("--gpu-hz", type=float, default=None,
                    help="override the clock used for the derived ceilings; "
                         "pass the value roofline runs used if they differ")
    ap.add_argument("--no-obey", action="store_true",
                    help="drop --precisionConstraints=obey if it makes the "
                         "build fail for unrelated reasons")
    ap.add_argument("--timing-cache", action="store_true",
                    help="reuse cached tactic choices. For iterating on this "
                         "script only — it freezes tactic selection and biases "
                         "the measured peak downward. Never for a reported "
                         "number.")
    ap.add_argument("--no-telemetry", action="store_true",
                    help="skip tegrastats sampling (needs sudo); throttling "
                         "will then go undetected")
    args = ap.parse_args()

    env = capture(ROOT)
    if not env["clocks"]["gpu_pinned"]:
        sys.exit("clocks are not pinned — run `sudo jetson_clocks` first.\n"
                 "A peak measurement against a moving clock is meaningless.")

    # Default to the pinned ceiling rather than a telemetry mean: with clocks
    # pinned the GPU runs at the ceiling during compute, and a sampled mean is
    # pulled down by idle gaps between runs.
    gpu_hz = args.gpu_hz or env["clocks"]["gpu_max_hz"]
    print(f"board      {env['power_mode']['name']}, pinned, "
          f"gpu {gpu_hz/1e6:.0f} MHz, {args.sms} SMs")
    if args.gpu_hz:
        print(f"           (clock overridden; configured ceiling is "
              f"{env['clocks']['gpu_max_hz']/1e6:.0f} MHz)")
    if args.timing_cache:
        print("           TIMING CACHE ENABLED — tactic selection is frozen, "
              "so the measured\n           peak is a lower bound at best. "
              "Do not report this number.")
    print()

    theory = {p: PER_SM_CLK[p] * args.sms * gpu_hz / 1e12 for p in args.precisions}
    for p in args.precisions:
        note = (f"   (halves to {theory[p]/2:.2f} if FP32-accumulate penalised)"
                if p == "fp16" else "")
        print(f"  derived {p:>5} ceiling: {theory[p]:6.2f} T/s{note}")
    print()

    work = ROOT / "results" / "peak_gemm"
    # Generated ONNX files are seeded and therefore identical across runs, so
    # they live in a shared directory rather than being regenerated per run.
    # Their hashes are recorded in each result.json, which is what ties a run
    # to its inputs.
    onnx_dir = work / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = env["git"]["sha_short"] or "nogit"
    mode = env["power_mode"]["name"] or "unknown"
    sizes_tag = "N" + "-".join(str(s) for s in args.sizes)
    run_dir = work / f"{ts}_{sizes_tag}_{mode}_pinned_{sha}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (work / "gemm.cache") if args.timing_cache else None
    print(f"run dir    {run_dir}\n")

    rows = []
    for n in args.sizes:
        flops = 2 * n ** 3
        print(f"N = {n}   ({flops/1e9:.1f} GFLOP per call)")

        for prec in args.precisions:
            onnx_path = onnx_dir / f"matmul_{n}_{prec}.onnx"
            if not onnx_path.exists():
                make_matmul(n, prec, onnx_path)
            onnx_sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest()[:12]
            ai = n / (3 * ELEM_BYTES[prec] / 2)

            r = run_trtexec(onnx_path, n, prec, args.duration, run_dir,
                            cache_path, obey=not args.no_obey,
                            workspace_mb=args.workspace_mb,
                            telemetry=not args.no_telemetry, gpu_hz=gpu_hz)
            if not r["ok"]:
                print(f"  {prec:>5}  FAILED  (AI ~{ai:.0f})")
                for line in r["errors"]:
                    print(f"         {line[:150]}")
                print(f"         full log: {r['log_path']}")
                continue

            tflops = flops / (r["gpu_ms"] * 1e-3) / 1e12
            frac = 100 * tflops / theory[prec]
            dts = sorted(set(r["kernel_dtypes"])) or ["unknown"]
            rows.append({"n": n, "precision": prec, "onnx_sha256_12": onnx_sha,
                         "arithmetic_intensity": round(ai, 1),
                         "gpu_ms": r["gpu_ms"], "tflops": round(tflops, 3),
                         "pct_of_derived": round(frac, 1),
                         "kernel_dtypes": dts, "telemetry": r["telemetry"]})
            print(f"  {prec:>5}  {r['gpu_ms']:8.3f} ms   {tflops:6.2f} T/s   "
                  f"{frac:5.1f}% of derived   AI ~{ai:.0f}")
            print(f"         kernel formats: {', '.join(d[:46] for d in dts)}")

            t = r["telemetry"]
            if t:
                flag = "  THROTTLED" if t["throttled"] else ""
                print(f"         timed region ({t['n_samples_in_window']}/"
                      f"{t['n_samples_total']} samples): "
                      f"{t['vdd_in_mean_mw']/1000:.2f} W mean, "
                      f"{t['vdd_in_max_mw']/1000:.2f} W peak   "
                      f"gpu {t['gpu_mhz_mean']:.0f}/{t['gpu_mhz_min']:.0f} MHz "
                      f"(mean/min of {t['gpu_mhz_ceiling']:.0f})   "
                      f"tj {t['temp_tj_max_c']:.0f}C{flag}")
        print()

    best = {}
    for r in rows:
        best[r["precision"]] = max(best.get(r["precision"], 0), r["tflops"])

    print("best attained:")
    for p in args.precisions:
        if p in best:
            print(f"  {p:>5}  {best[p]:6.2f} T/s   "
                  f"{100*best[p]/theory[p]:5.1f}% of derived")
    print()

    verdict = None
    if "fp32" in best:
        ctrl = 100 * best["fp32"] / theory["fp32"]
        if ctrl < 70:
            verdict = "control_failed"
            print(f"  CONTROL FAILED: FP32 reached only {ctrl:.0f}% of its "
                  f"CUDA-core ceiling. The harness is suspect, not the "
                  f"hardware — draw no conclusion about FP16 from this run.")
        else:
            print(f"  control ok: FP32 at {ctrl:.0f}% of its CUDA-core ceiling")

    if "fp16" in best and verdict != "control_failed":
        halved = theory["fp16"] / 2
        attained = best["fp16"]
        # One-sided: exceeding the halved ceiling falsifies it outright.
        # Falling short does not establish it — any inefficiency does that too.
        if attained > halved * 1.05:
            verdict = "no_penalty"
            print(f"\n  FP16 reached {attained:.2f} T/s, above the halved "
                  f"ceiling of {halved:.2f} T/s.\n"
                  f"  A roof cannot be exceeded, so the FP32-accumulate "
                  f"penalty does NOT apply.\n"
                  f"  Keep fp16 = 2048 FLOP/clk/SM in framecost/roofline.py.")
        else:
            verdict = "inconclusive"
            gap = ""
            if "fp32" in best:
                e16 = 100 * attained / theory["fp16"]
                e32 = 100 * best["fp32"] / theory["fp32"]
                gap = (f"\n  FP16 efficiency {e16:.0f}% vs FP32 control "
                       f"{e32:.0f}%: "
                       + ("a large shortfall here is consistent with a penalty, "
                          "but does not prove one."
                          if e16 < e32 * 0.7 else
                          "comparable, so a penalty is not indicated."))
            print(f"\n  INCONCLUSIVE: FP16 reached {attained:.2f} T/s, at or "
                  f"below the halved ceiling of {halved:.2f} T/s.\n"
                  f"  This is consistent with a penalty AND with a merely "
                  f"inefficient kernel, so it settles nothing.{gap}\n"
                  f"  Try a larger N, check the kernel formats above, or state "
                  f"the uncertainty in the write-up.")

    sampled = [r for r in rows if r.get("telemetry")]
    any_throttled = any(r["telemetry"]["throttled"] for r in sampled)
    # Tri-state: with no telemetry, throttle status is unknown, not clean.
    # Reporting False there would be absence of evidence read as evidence of
    # absence, on exactly the workload most likely to trip the current limit.
    throttle_status = ("unknown" if not sampled
                       else "throttled" if any_throttled else "clean")

    if not sampled and rows:
        print("\n  NOTE: no telemetry captured, so throttle status is UNKNOWN. "
              "A dense GEMM is\n"
              "  the most power-hungry workload this board runs; if it was "
              "current-limited,\n"
              "  these figures are sustained rather than unconstrained and "
              "there is no way\n"
              "  to tell from this run. Re-run after `sudo -v` to find out.")
    elif any_throttled:
        print("\n  NOTE: at least one arm ran with the GPU clock below its "
              "pinned ceiling.\n"
              "  A dense GEMM is the most power-hungry workload this board "
              "runs and can trip\n"
              "  the current limit that a memory-bound network never "
              "approaches. This biases\n"
              "  attained throughput DOWNWARD, so a ceiling exceeded despite "
              "it is exceeded\n"
              "  a fortiori — but report these figures as sustained rather "
              "than unconstrained.")

    out = run_dir / "result.json"
    out.write_text(json.dumps(
        {"captured_utc": ts, "env": env, "gpu_hz_used": gpu_hz,
         "derived_ceilings_tflops": theory,
         "timing_cache_used": bool(args.timing_cache),
         "precision_constraints_obey": not args.no_obey,
         "throttle_status": throttle_status,
         "telemetry_coverage": f"{len(sampled)}/{len(rows)}",
         "runs": rows, "best_attained_tflops": best, "verdict": verdict},
        indent=2))
    print(f"\nwrote {run_dir}")


if __name__ == "__main__":
    main()
