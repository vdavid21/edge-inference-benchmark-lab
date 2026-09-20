"""tegrastats sampling and energy accounting.

Wraps `tegrastats` as a background sampler so that power, thermal and clock
telemetry is collected alongside a measurement rather than as a separate step
someone has to remember. Used as a context manager for exactly that reason:
a telemetry step you can forget is one that eventually gets skipped on the run
that mattered.

    with TegraSampler(interval_ms=100) as t:
        run_the_benchmark()
    print(t.summary())
    print(t.energy(n_inferences=39248, idle_mw=3700))

Rails on the Orin Nano devkit (INA3221, reported by tegrastats):
    VDD_IN           total board input — the number that matters for energy
    VDD_CPU_GPU_CV   combined CPU, GPU and CV accelerator rail
    VDD_SOC          memory controller, interconnect, I/O

Output format differs by JetPack version:
    JP6:  VDD_IN 4832mW/4680mW           instantaneous / running average
    JP7:  VDD_IN 4832mW/4680mW/5012mW    instantaneous / average / max
Only the first value is used. The running average is cumulative from process
start, which is not what you want when the run has a warm-up phase — compute
your own statistics from the samples instead.

Caveats worth stating in any write-up that uses this:
  - tegrastats costs CPU to sample. At 100 ms on six A78AE cores the effect is
    small but not zero. Run one configuration with telemetry off to quantify it.
  - VDD_IN is measured on-board, after the barrel jack. It does not include the
    PSU's conversion loss. A wall meter reads higher; the ratio is worth
    reporting.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field

import numpy as np

# --- parsing -----------------------------------------------------------------

_RE_POWER = re.compile(r"(VDD_[A-Z0-9_]+)\s+(\d+)mW")
_RE_THERM = re.compile(r"([a-z]+\d*)@([\d.]+)C")
_RE_RAM = re.compile(r"RAM (\d+)/(\d+)MB")
_RE_EMC = re.compile(r"EMC_FREQ (\d+)%@(\d+)")
_RE_GR3D = re.compile(r"GR3D_FREQ (\d+)%@?\[?(\d+)?\]?")
_RE_CPU = re.compile(r"CPU \[([^\]]+)\]")


def parse_line(line: str) -> dict | None:
    """Parse one tegrastats line into a flat dict. Returns None if unparseable."""
    power = {m.group(1): int(m.group(2)) for m in _RE_POWER.finditer(line)}
    if not power:
        return None  # not a telemetry line

    sample: dict = {"t": time.time(), "power_mw": power}

    sample["temp_c"] = {m.group(1): float(m.group(2))
                        for m in _RE_THERM.finditer(line)}

    if (m := _RE_RAM.search(line)):
        sample["ram_used_mb"] = int(m.group(1))
        sample["ram_total_mb"] = int(m.group(2))

    if (m := _RE_EMC.search(line)):
        sample["emc_util_pct"] = int(m.group(1))
        sample["emc_mhz"] = int(m.group(2))

    if (m := _RE_GR3D.search(line)):
        sample["gpu_util_pct"] = int(m.group(1))
        sample["gpu_mhz"] = int(m.group(2)) if m.group(2) else None

    if (m := _RE_CPU.search(line)):
        cores = []
        for c in m.group(1).split(","):
            if "@" in c:
                util, freq = c.split("@")
                cores.append((int(util.rstrip("%")), int(freq)))
        sample["cpu_cores"] = cores

    return sample


# --- sampler -----------------------------------------------------------------

@dataclass
class TegraSampler:
    interval_ms: int = 100
    use_sudo: bool = True
    samples: list[dict] = field(default_factory=list)
    _proc: subprocess.Popen | None = None
    _thread: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    started_at: float | None = None
    stopped_at: float | None = None

    def start(self) -> "TegraSampler":
        if shutil.which("tegrastats") is None:
            raise RuntimeError("tegrastats not found on PATH")

        cmd = ["tegrastats", "--interval", str(self.interval_ms)]
        if self.use_sudo:
            cmd = ["sudo", "-n"] + cmd          # -n: fail rather than prompt

        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            bufsize=1)
        self.started_at = time.time()

        def reader():
            assert self._proc and self._proc.stdout
            for line in self._proc.stdout:
                if self._stop.is_set():
                    break
                s = parse_line(line)
                if s:
                    self.samples.append(s)

        self._thread = threading.Thread(target=reader, daemon=True)
        self._thread.start()

        # Give it a moment to produce a first sample, so a permission failure
        # surfaces here rather than as an empty summary at the end.
        time.sleep(max(0.5, self.interval_ms / 1000 * 3))
        if not self.samples:
            err = ""
            if self._proc.poll() is not None and self._proc.stderr:
                err = self._proc.stderr.read().strip()
            self.stop()
            raise RuntimeError(
                f"tegrastats produced no samples. {err or 'Needs sudo?'}\n"
                "Try: sudo -v   (or add a NOPASSWD sudoers rule for tegrastats)")
        return self

    def stop(self) -> None:
        self._stop.set()
        self.stopped_at = time.time()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=2)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # -- analysis ------------------------------------------------------------

    @property
    def duration_s(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.stopped_at or time.time()
        return end - self.started_at

    def rail(self, name: str = "VDD_IN") -> np.ndarray:
        return np.array([s["power_mw"].get(name, np.nan) for s in self.samples],
                        dtype=float)

    def summary(self) -> dict:
        if not self.samples:
            return {"n_samples": 0}

        out: dict = {
            "n_samples": len(self.samples),
            "duration_s": round(self.duration_s, 3),
            "interval_ms": self.interval_ms,
        }

        for name in sorted({k for s in self.samples for k in s["power_mw"]}):
            v = self.rail(name)
            v = v[~np.isnan(v)]
            out[f"{name.lower()}_mean_mw"] = round(float(v.mean()), 1)
            out[f"{name.lower()}_p95_mw"] = round(float(np.percentile(v, 95)), 1)
            out[f"{name.lower()}_max_mw"] = round(float(v.max()), 1)

        temps: dict[str, list[float]] = {}
        for s in self.samples:
            for z, c in s.get("temp_c", {}).items():
                temps.setdefault(z, []).append(c)
        for z, vals in temps.items():
            out[f"temp_{z}_max_c"] = round(max(vals), 2)
        out["temp_start_c"] = self.samples[0].get("temp_c", {}).get("tj")
        out["temp_end_c"] = self.samples[-1].get("temp_c", {}).get("tj")

        for key, label in [("gpu_util_pct", "gpu_util"), ("emc_util_pct", "emc_util"),
                           ("gpu_mhz", "gpu_mhz"), ("emc_mhz", "emc_mhz")]:
            vals = [s[key] for s in self.samples if s.get(key) is not None]
            if vals:
                out[f"{label}_mean"] = round(float(np.mean(vals)), 1)
                out[f"{label}_min"] = min(vals)

        ram = [s["ram_used_mb"] for s in self.samples if "ram_used_mb" in s]
        if ram:
            out["ram_peak_mb"] = max(ram)

        return out

    def energy(self, n_inferences: int, idle_mw: float | None = None,
               rail: str = "VDD_IN") -> dict:
        """Energy per inference.

        `total` sizes a battery: everything the board drew while working.
        `marginal` is the incremental cost of running this model on a system
        that is already powered — subtract the idle draw to get it. Report both;
        they answer different design questions.
        """
        v = self.rail(rail)
        v = v[~np.isnan(v)]
        if v.size == 0 or n_inferences <= 0:
            return {}

        mean_w = float(v.mean()) / 1000.0
        total_j = mean_w * self.duration_s
        out = {
            "rail": rail,
            "mean_power_w": round(mean_w, 3),
            "duration_s": round(self.duration_s, 3),
            "n_inferences": n_inferences,
            "energy_total_j": round(total_j, 3),
            "mj_per_inference": round(total_j / n_inferences * 1000, 4),
        }
        if idle_mw is not None:
            marg_w = max(0.0, mean_w - idle_mw / 1000.0)
            out["idle_power_w"] = round(idle_mw / 1000.0, 3)
            out["marginal_power_w"] = round(marg_w, 3)
            out["mj_per_inference_marginal"] = round(
                marg_w * self.duration_s / n_inferences * 1000, 4)
        return out

    def to_csv(self, path) -> None:
        """Write the raw time series. Keep it: percentiles can be recomputed,
        a discarded trace cannot."""
        import csv

        rails = sorted({k for s in self.samples for k in s["power_mw"]})
        zones = sorted({z for s in self.samples for z in s.get("temp_c", {})})
        cols = (["t_rel_s"] + [r.lower() + "_mw" for r in rails]
                + ["temp_" + z + "_c" for z in zones]
                + ["gpu_util_pct", "gpu_mhz", "emc_util_pct", "emc_mhz",
                   "ram_used_mb"])

        t0 = self.samples[0]["t"] if self.samples else 0
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for s in self.samples:
                row = [round(s["t"] - t0, 3)]
                row += [s["power_mw"].get(r, "") for r in rails]
                row += [s.get("temp_c", {}).get(z, "") for z in zones]
                row += [s.get("gpu_util_pct", ""), s.get("gpu_mhz", ""),
                        s.get("emc_util_pct", ""), s.get("emc_mhz", ""),
                        s.get("ram_used_mb", "")]
                w.writerow(row)


def measure_idle(seconds: float = 30.0, interval_ms: int = 100) -> dict:
    """Idle baseline. Run with nothing else on the board.

    Needed for marginal energy, and worth recapturing per session — idle draw
    shifts with ambient temperature and whatever the OS has running.
    """
    with TegraSampler(interval_ms=interval_ms) as t:
        time.sleep(seconds)
    return t.summary()


if __name__ == "__main__":
    import json
    import sys

    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    print(f"sampling {secs:.0f}s — leave the board idle...\n")
    with TegraSampler() as t:
        time.sleep(secs)
    print(json.dumps(t.summary(), indent=2))
