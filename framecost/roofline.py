"""Per-layer roofline analysis for TensorRT engines.

Library half of the profiling workflow. scripts/profile_model.py orchestrates
the measurement; this module does the arithmetic and draws the figure. Kept
importable so a figure can be re-derived from a stored profile without
re-running the board.

What it has to get right, in order of how easy it is to get wrong:

GROUPED CONVOLUTIONS. MobileNetV2's depthwise convs have groups == C_in.
Ignoring the group attribute overstates their FLOPs by up to 32x and moves them
to the wrong side of the ridge — which would invert the central finding.

FUSED BYTE COUNTS. A TensorRT layer named "Conv_5 + Clip_6" ran as one kernel,
so the intermediate never reaches DRAM. The group's traffic is the first node's
input, the last node's output, and all weights — not the sum of each node's
traffic. Getting this wrong makes fusion invisible.

NON-COMPUTE LAYERS. Reformats, quantise and dequantise nodes take real time and
perform no useful arithmetic. They must not be plotted on the roofline: an
earlier version matched "Conv_0" inside "Reformatting CopyNode ... to Conv_0"
and plotted a memcpy at 572 GFLOP/s.

UNATTRIBUTED LAYERS. TensorRT's myelin backend rewrites some subgraphs into
kernels whose names carry no ONNX node reference (e.g. __myl_FcCast_myl56_0).
These are reported separately rather than silently dropped, because on the INT8
engine the largest single layer is one of them.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import onnx
from onnx import shape_inference

# Ampere third-gen Tensor Core throughput, FLOP (or OP) per SM per clock, dense.
# FP32 runs on the CUDA cores: 128 lanes x 2 FLOP.
PEAK_PER_SM_CLK = {"fp32": 256, "tf32": 1024, "fp16": 2048, "int8": 4096}
ELEM_BYTES = {"fp32": 4, "tf32": 4, "fp16": 2, "int8": 1}

_RE_NODE = re.compile(r"\b([A-Za-z][A-Za-z0-9]*_\d+)\b")
_RE_NONCOMPUTE = re.compile(
    r"reformat|copynode|quantizelinear\s*$|dequantizelinear\s*$|"
    r"^\s*\w+_(De)?QuantizeLinear\s*$", re.I)


def peak_flops(precision: str, gpu_hz: float, sms: int = 8) -> float:
    return PEAK_PER_SM_CLK[precision] * sms * gpu_hz


def bandwidth_bps(emc_hz: float, bus_bits: int = 128) -> float:
    """LPDDR5: double data rate, bus_bits wide."""
    return emc_hz * 2 * (bus_bits / 8)


# --- ONNX geometry ----------------------------------------------------------

def build_graph_info(onnx_path: str | Path) -> dict:
    """node name -> {flops, in_elems, out_elems, weight_elems, kind, index}.

    TensorRT names a layer after the ONNX node name when present, and otherwise
    generates "<OpType>_<global node index>". Both keys are registered.
    """
    model = shape_inference.infer_shapes(onnx.load(str(onnx_path)))
    g = model.graph

    shapes: dict[str, tuple] = {}
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        dims = vi.type.tensor_type.shape.dim
        shapes[vi.name] = tuple(d.dim_value if d.dim_value > 0 else 1 for d in dims)
    inits = {t.name: tuple(t.dims) for t in g.initializer}

    def numel(s) -> int:
        return int(np.prod(s)) if s else 0

    info: dict[str, dict] = {}
    for idx, node in enumerate(g.node):
        keys = [f"{node.op_type}_{idx}"] + ([node.name] if node.name else [])
        in_shape = shapes.get(node.input[0], ()) if node.input else ()
        out_shape = shapes.get(node.output[0], ()) if node.output else ()

        flops = w_elems = 0
        kind = node.op_type

        if node.op_type == "Conv" and len(node.input) >= 2:
            w = inits.get(node.input[1], shapes.get(node.input[1], ()))
            if w and out_shape:
                groups = next((a.i for a in node.attribute if a.name == "group"), 1)
                c_out, c_in_pg = w[0], w[1]
                k = int(np.prod(w[2:])) if len(w) > 2 else 1
                spatial = int(np.prod(out_shape[2:])) if len(out_shape) > 2 else 1
                flops = 2 * spatial * c_out * c_in_pg * k
                w_elems = numel(w) + (numel(inits.get(node.input[2], ()))
                                      if len(node.input) > 2 else 0)
                kind = ("depthwise" if groups > 1 and c_in_pg == 1
                        else "pointwise" if k == 1 else "conv")

        elif node.op_type in ("Gemm", "MatMul") and len(node.input) >= 2:
            w = inits.get(node.input[1], shapes.get(node.input[1], ()))
            if w:
                flops = 2 * int(np.prod(w))
                w_elems = numel(w) + (numel(inits.get(node.input[2], ()))
                                      if len(node.input) > 2 else 0)
            kind = "gemm"

        entry = {"flops": flops, "in_elems": numel(in_shape),
                 "out_elems": numel(out_shape), "weight_elems": w_elems,
                 "kind": kind, "index": idx}
        for k in keys:
            info.setdefault(k, entry)

    return info


# --- profile join -----------------------------------------------------------

def load_profile(path: str | Path) -> list[dict]:
    raw = json.loads(Path(path).read_text())
    return [e for e in raw if isinstance(e, dict) and "name" in e]


def analyse(profile: list[dict], info: dict, precision: str,
            scale: float = 1.0) -> tuple[list[dict], dict]:
    """Returns (rows, diagnostics)."""
    eb = ELEM_BYTES[precision]
    rows: list[dict] = []
    matched_indices: set[int] = set()

    for entry in profile:
        name = entry["name"]
        t_ms = entry["medianMs"] * scale
        base = {"layer": name, "time_ms": t_ms, "pct": entry["percentage"]}

        # Non-compute layers are excluded from FLOP attribution entirely,
        # before any node-name matching, or a reformat named after the layer it
        # feeds would inherit that layer's FLOPs.
        if _RE_NONCOMPUTE.search(name):
            rows.append({**base, "kind": "overhead", "flops": 0, "bytes": 0,
                         "ai": None, "gflops": None})
            continue

        compute = [info[t] for t in _RE_NODE.findall(name)
                   if t in info and info[t]["flops"] > 0]

        if not compute:
            rows.append({**base, "kind": "unattributed", "flops": 0, "bytes": 0,
                         "ai": None, "gflops": None})
            continue

        compute.sort(key=lambda m: m["index"])
        matched_indices.update(m["index"] for m in compute)
        flops = sum(m["flops"] for m in compute)
        byts = eb * (compute[0]["in_elems"] + compute[-1]["out_elems"]
                     + sum(m["weight_elems"] for m in compute))

        rows.append({**base,
                     "kind": compute[0]["kind"] if len(compute) == 1 else "fused",
                     "flops": flops, "bytes": byts,
                     "ai": flops / byts if byts else None,
                     "gflops": flops / (t_ms * 1e-3) / 1e9 if t_ms > 0 else None})

    # Which ONNX compute nodes never appeared in any TensorRT layer name.
    all_compute = {v["index"]: v for v in info.values() if v["flops"] > 0}
    missing = sorted(set(all_compute) - matched_indices)
    diagnostics = {
        "onnx_compute_nodes": len(all_compute),
        "matched": len(matched_indices),
        "unmatched_node_indices": missing,
        "unmatched_flops": sum(all_compute[i]["flops"] for i in missing),
        "total_onnx_flops": sum(v["flops"] for v in all_compute.values()),
    }
    return rows, diagnostics


def summarise(rows: list[dict], diagnostics: dict, precision: str,
              gpu_hz: float, emc_hz: float, sms: int = 8) -> dict:
    peak = peak_flops(precision, gpu_hz, sms) / 1e9      # GFLOP/s
    bw = bandwidth_bps(emc_hz) / 1e9                     # GB/s
    ridge = peak / bw

    pts = [r for r in rows if r["ai"]]
    bound = [r for r in pts if r["ai"] < ridge]
    eff = [r["gflops"] / peak * 100 for r in pts if r["gflops"]]
    ais = [r["ai"] for r in pts]

    def pct_of(kind):
        return sum(r["pct"] for r in rows if r["kind"] == kind)

    return {
        "precision": precision,
        "gpu_mhz": gpu_hz / 1e6,
        "peak_gflops": round(peak, 1),
        "bandwidth_gbs": round(bw, 1),
        "ridge_flop_per_byte": round(ridge, 1),
        "n_compute_layers": len(pts),
        "n_memory_bound": len(bound),
        "pct_memory_bound": round(100 * len(bound) / max(1, len(pts)), 1),
        "ai_min": round(min(ais), 2) if ais else None,
        "ai_median": round(float(np.median(ais)), 2) if ais else None,
        "ai_max": round(max(ais), 2) if ais else None,
        "peak_util_median_pct": round(float(np.median(eff)), 2) if eff else None,
        "peak_util_best_pct": round(max(eff), 2) if eff else None,
        "overhead_pct_of_time": round(pct_of("overhead"), 2),
        "unattributed_pct_of_time": round(pct_of("unattributed"), 2),
        **diagnostics,
    }


# --- output -----------------------------------------------------------------

def write_csv(rows: list[dict], path: str | Path) -> None:
    cols = ["layer", "kind", "time_ms", "pct", "flops", "bytes", "ai", "gflops"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


COLOURS = {"depthwise": "#d62728", "pointwise": "#1f77b4", "conv": "#2ca02c",
           "gemm": "#9467bd", "fused": "#ff7f0e"}


def plot(rows: list[dict], summary: dict, path: str | Path,
         title: str | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    peak, bw = summary["peak_gflops"], summary["bandwidth_gbs"]
    ridge, prec = summary["ridge_flop_per_byte"], summary["precision"]
    unit = "TOP/s" if prec == "int8" else "TFLOP/s"

    pts = [r for r in rows if r["ai"] and r["gflops"]]
    fig, ax = plt.subplots(figsize=(9, 6))

    x = np.logspace(-1, 3.5, 400)
    ax.plot(x, np.minimum(peak, bw * x), "k-", lw=2,
            label=f"{peak/1000:.2f} {unit} · {bw:.0f} GB/s")
    ax.axvline(ridge, color="grey", ls=":", lw=1)
    ax.text(ridge * 1.06, peak * 0.03, f"ridge {ridge:.0f}", color="grey",
            fontsize=9, rotation=90, va="bottom")

    for kind in sorted({r["kind"] for r in pts}):
        sel = [r for r in pts if r["kind"] == kind]
        ax.scatter([r["ai"] for r in sel], [r["gflops"] for r in sel],
                   s=[max(15, r["pct"] * 40) for r in sel], alpha=0.75,
                   color=COLOURS.get(kind, "#7f7f7f"), edgecolor="white",
                   linewidth=0.5, label=f"{kind} ({len(sel)})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity  (FLOP / byte)")
    ax.set_ylabel(f"attained  (G{'OP' if prec == 'int8' else 'FLOP'}/s)")
    ax.set_title(title or f"{prec.upper()} — {summary['gpu_mhz']:.0f} MHz, "
                          f"{summary['pct_memory_bound']:.0f}% memory-bound")
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
