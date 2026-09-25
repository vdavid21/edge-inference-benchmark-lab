"""Per-run environment capture.

Writes env.json into each run directory. Distinct from docs/env/*.md, which is a
human-readable full system dump captured occasionally; this is machine-readable,
captured every run, and holds only what could plausibly change the numbers.

Design principle: record MEASURED state, not REQUESTED state.

    nvpmodel -q  ->  "NV Power Mode: 25W"
    actual GPU clock ->  624 MHz

Those disagreed on this board for an entire evening because the bootloader had
not granted the higher ceiling. Anything derived from the requested mode would
have been silently wrong. So every field that can be read back from the hardware
is read back, and the requested value is stored alongside it for comparison.

Every probe is wrapped: a missing sysfs path records null rather than raising.
A benchmark must never fail because telemetry was unavailable.

Two runs of the same configuration should produce near-identical env.json apart
from timestamps and temperatures, which makes `diff` a useful tool for answering
"what changed between these two runs?"
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _run(cmd: list[str], timeout: float = 5.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip() or None
    except Exception:
        return None


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip().rstrip("\x00")
    except Exception:
        return None


def _int(path: str) -> int | None:
    v = _read(path)
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# --- identity ---------------------------------------------------------------

# Paths holding measurement output rather than code or configuration.
_RESULT_PREFIXES = ("results/",)


def _status_paths(status: str) -> list[str]:
    """Paths out of `git status --porcelain` v1, renames resolved to destination."""
    paths = []
    for line in status.splitlines():
        if not line.strip():
            continue
        p = line[3:]                     # two status characters, then a space
        if " -> " in p:                  # rename or copy: the destination is the file
            p = p.split(" -> ", 1)[1]
        paths.append(p.strip().strip('"'))
    return paths


def git_state(repo_root: str | Path = ".") -> dict:
    """SHA and cleanliness of the working tree.

    `dirty` is the important field, and it has three values:

        False           the tree matches the commit
        "only results"  the sole changes are under results/, i.e. measurement
                        output that this run or an earlier one wrote
        True            something outside results/ differs from the commit

    Only True disqualifies a result from publication, because the SHA then
    fails to describe the code that ran. "only results" is the normal state
    from the second run onward: a run writes its output into the repo and so
    dirties the tree simply by existing. A plain boolean could not tell those
    apart, and reported every run after the first as untraceable.
    """
    cwd = str(repo_root)
    sha = _run(["git", "-C", cwd, "rev-parse", "HEAD"])
    status = _run(["git", "-C", cwd, "status", "--porcelain"])

    paths = _status_paths(status) if status else []
    outside = [p for p in paths if not p.startswith(_RESULT_PREFIXES)]
    dirty: bool | str = bool(paths)
    if paths and not outside:
        dirty = "only results"

    return {
        "sha": sha,
        "sha_short": sha[:12] if sha else None,
        "dirty": dirty,
        "branch": _run(["git", "-C", cwd, "rev-parse", "--abbrev-ref", "HEAD"]),
    }


def board() -> dict:
    l4t = _read("/etc/nv_tegra_release")
    rel = None
    if l4t:
        m = re.search(r"R(\d+).*REVISION:\s*([\d.]+)", l4t)
        if m:
            rel = f"r{m.group(1)}.{m.group(2)}"
    return {
        "model": _read("/proc/device-tree/model"),
        "l4t": rel,
        "l4t_raw": l4t,
        "compatible_spec": next(
            (l for l in (_read("/etc/nv_boot_control.conf") or "").splitlines()
             if l.startswith("COMPATIBLE_SPEC")), None),
        "kernel": platform.release(),
        "arch": platform.machine(),
    }


def software() -> dict:
    """Versions of everything between the model and the silicon."""
    ver = {"python": platform.python_version()}

    for mod, key in [("tensorrt", "tensorrt"), ("numpy", "numpy"),
                     ("onnx", "onnx"), ("onnxruntime", "onnxruntime"),
                     ("modelopt", "nvidia_modelopt"), ("torch", "torch")]:
        try:
            ver[key] = __import__(mod).__version__
        except Exception:
            ver[key] = None

    nvcc = _run(["nvcc", "--version"])
    if nvcc:
        m = re.search(r"release ([\d.]+), V([\d.]+)", nvcc)
        ver["cuda"] = m.group(2) if m else None
    else:
        ver["cuda"] = None

    pkgs = _run(["dpkg-query", "-W", "-f=${Package} ${Version}\n",
                 "libcudnn9-cuda-13", "libnvinfer-bin", "nvidia-l4t-core"])
    ver["apt"] = dict(
        line.split(" ", 1) for line in (pkgs or "").splitlines() if " " in line
    ) or None

    return ver


# --- board state at run time -------------------------------------------------

GPU_DEVFREQ = "/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu"


def clocks() -> dict:
    """Actual clock rates, read from the hardware.

    min == max means the clocks are pinned (jetson_clocks applied). If min < max
    the governor is free to move them and the run is subject to DVFS.
    """
    gpu_min = _int(f"{GPU_DEVFREQ}/min_freq")
    gpu_max = _int(f"{GPU_DEVFREQ}/max_freq")
    gpu_cur = _int(f"{GPU_DEVFREQ}/cur_freq")

    # EMC has no stable unprivileged sysfs path across JetPack versions, so
    # fall back to parsing a tegrastats sample. Stored as raw text too, because
    # a string that is never wrong beats a parse that might be.
    emc_hz = None
    show = _run(["jetson_clocks", "--show"], timeout=10)
    if show:
        m = re.search(r"EMC .*?CurrentFreq=(\d+)", show)
        if m:
            emc_hz = int(m.group(1))

    return {
        "gpu_min_hz": gpu_min,
        "gpu_max_hz": gpu_max,
        "gpu_cur_hz": gpu_cur,
        "gpu_pinned": (gpu_min == gpu_max) if gpu_min and gpu_max else None,
        "emc_cur_hz": emc_hz,
        "cpu_governor": _read(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "cpu_max_khz": _int("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"),
        "jetson_clocks_raw": show,
    }


def power_mode() -> dict:
    """Requested power mode, stored next to the clocks it was supposed to set.

    Comparing `name` against clocks()['gpu_max_hz'] is what catches a mode that
    the bootloader never actually granted.
    """
    q = _run(["nvpmodel", "-q"], timeout=10)
    mode_id = name = None
    if q:
        lines = [l.strip() for l in q.splitlines() if l.strip()]
        for l in lines:
            m = re.search(r"NV Power Mode:\s*(\S+)", l)
            if m:
                name = m.group(1)
        if lines and lines[-1].isdigit():
            mode_id = int(lines[-1])
    return {"mode_id": mode_id, "name": name, "raw": q}


def thermal() -> dict:
    zones = {}
    for z in sorted(Path("/sys/devices/virtual/thermal").glob("thermal_zone*")):
        t = _read(str(z / "type"))
        v = _int(str(z / "temp"))
        if t and v is not None:
            zones[t] = round(v / 1000.0, 2)
    return zones


def load() -> dict:
    """System load before the run.

    A non-trivial load average is the signature of the contamination that
    inflates p99 while leaving the median untouched. Record it so a suspicious
    tail can be checked against it afterwards rather than guessed at.
    """
    try:
        one, five, fifteen = os.getloadavg()
    except OSError:
        one = five = fifteen = None
    mem = _read("/proc/meminfo") or ""
    total = re.search(r"MemTotal:\s+(\d+)", mem)
    avail = re.search(r"MemAvailable:\s+(\d+)", mem)
    return {
        "loadavg_1m": one,
        "loadavg_5m": five,
        "loadavg_15m": fifteen,
        "mem_total_kb": int(total.group(1)) if total else None,
        "mem_available_kb": int(avail.group(1)) if avail else None,
    }


# --- assembly ----------------------------------------------------------------

def capture(repo_root: str | Path = ".", note: str | None = None) -> dict:
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": note,
        "git": git_state(repo_root),
        "board": board(),
        "software": software(),
        "clocks": clocks(),
        "power_mode": power_mode(),
        "thermal_c": thermal(),
        "load": load(),
    }


def write(out_path: str | Path, repo_root: str | Path = ".",
          note: str | None = None) -> dict:
    env = capture(repo_root, note)
    Path(out_path).write_text(json.dumps(env, indent=2))
    return env


def warn_if_unsuitable(env: dict) -> list[str]:
    """Conditions that make a run unfit for publication. Check before measuring."""
    w = []
    if env["git"]["dirty"] is True:
        w.append("working tree is dirty — result not traceable to a commit")
    if env["clocks"]["gpu_pinned"] is False:
        w.append("GPU clocks not pinned — run `sudo jetson_clocks`")
    la = env["load"]["loadavg_1m"]
    if la is not None and la > 0.5:
        w.append(f"load average {la:.2f} — close other processes")
    tj = env["thermal_c"].get("tj-thermal")
    if tj is not None and tj > 70:
        w.append(f"junction temperature {tj}C — board is hot, expect throttling")
    return w


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "."
    env = capture(root)
    # jetson_clocks_raw and l4t_raw are long; keep them in the file, not the terminal
    printable = json.loads(json.dumps(env))
    printable["clocks"].pop("jetson_clocks_raw", None)
    printable["board"].pop("l4t_raw", None)
    print(json.dumps(printable, indent=2))

    for msg in warn_if_unsuitable(env):
        print(f"\n  WARNING: {msg}")
