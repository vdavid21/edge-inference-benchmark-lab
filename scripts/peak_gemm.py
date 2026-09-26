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


def run_trtexec(onnx_path: Path, n: int, precision: str, duration: int,
                work: Path, obey: bool, workspace_mb: int,
                timing_cache: bool) -> dict:
    layer_info = work / f"layers_{n}_{precision}.json"
    # No --shapes/--minShapes/--optShapes/--maxShapes: the generated ONNX is
    # fully static [n, n], so there are no dynamic dimensions to build an
    # optimization profile over, and passing them fails config setup.
    cmd = [TRTEXEC, f"--onnx={onnx_path}",
           "--noDataTransfers", "--useCudaGraph", "--useSpinWait",
           "--warmUp=2000", f"--duration={duration}",
           f"--memPoolSize=workspace:{workspace_mb}M",
           "--profilingVerbosity=detailed",
           f"--exportLayerInfo={layer_info}"]
    if timing_cache:
        # Opt-in only, and never for a number that will be reported. See the
        # note in the module docstring.
        cmd.append(f"--timingCacheFile={work/'gemm.cache'}")
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

    p = subprocess.run(cmd, capture_output=True, text=True)
    log = p.stdout + p.stderr
    log_path = work / f"trtexec_{n}_{precision}.log"
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

    return {"ok": True, "gpu_ms": float(m.group(1)), "kernel_dtypes": kernel_dtypes}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[2048])
    ap.add_argument("--precisions", nargs="+", default=["fp32", "tf32", "fp16"])
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
    work.mkdir(parents=True, exist_ok=True)

    rows = []
    for n in args.sizes:
        flops = 2 * n ** 3
        print(f"N = {n}   ({flops/1e9:.1f} GFLOP per call)")

        for prec in args.precisions:
            onnx_path = work / f"matmul_{n}_{prec}.onnx"
            if not onnx_path.exists():
                make_matmul(n, prec, onnx_path)
            onnx_sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest()[:12]
            ai = n / (3 * ELEM_BYTES[prec] / 2)

            r = run_trtexec(onnx_path, n, prec, args.duration, work,
                            obey=not args.no_obey,
                            workspace_mb=args.workspace_mb,
                            timing_cache=args.timing_cache)
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
                         "kernel_dtypes": dts})
            print(f"  {prec:>5}  {r['gpu_ms']:8.3f} ms   {tflops:6.2f} T/s   "
                  f"{frac:5.1f}% of derived   AI ~{ai:.0f}")
            print(f"         kernel formats: {', '.join(d[:46] for d in dts)}")
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

    out = work / "peak_gemm.json"
    out.write_text(json.dumps(
        {"env": env, "gpu_hz_used": gpu_hz, "derived_ceilings_tflops": theory,
         "timing_cache_used": bool(args.timing_cache),
         "precision_constraints_obey": not args.no_obey,
         "runs": rows, "best_attained_tflops": best, "verdict": verdict},
        indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
