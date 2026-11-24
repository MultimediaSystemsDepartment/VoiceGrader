#!/usr/bin/env python3
"""
stress_tests/load/gpu_monitor.py — NVML-based GPU monitor for RLHF training jobs.

Backend priority:
    1. pynvml   (from nvidia-ml-py)   — low-overhead in-process NVML bindings
  2. nvidia-smi subprocess          — fallback if pynvml is not installed

Per-container VRAM filtering:
  Works on Linux (container PIDs == host PIDs visible to NVML).
  On Windows + Docker Desktop (WSL2), PID namespaces differ so per-container
  filtering is skipped; overall GPU metrics are still collected.

Tracked metrics per sample:
  gpu_util_pct      — SM utilisation reported by NVML / nvidia-smi
  vram_used_mib     — total device VRAM in use
  vram_total_mib    — total device VRAM capacity
  vram_free_mib     — available VRAM
  temp_c            — GPU core temperature (Celsius)
  power_w           — current power draw (Watts); -1 if unsupported
  container_vram_mib — VRAM used by processes belonging to the watched container
                       (Linux only; 0 on Windows / when no container specified)

Usage — standalone CLI:
    python stress_tests/load/gpu_monitor.py --duration 300 --out logs/gpu.csv
    python stress_tests/load/gpu_monitor.py --container rlhf-asr-coordinator-1 \\
                                     --duration 600 --interval 10

Usage — as a library (e.g. from a training script):
    from tests.load.gpu_monitor import GPUMonitor
    from pathlib import Path

    mon = GPUMonitor(csv_path=Path("run_gpu.csv"),
                     container_name="rlhf-asr-coordinator-1")
    mon.start()
    # ... kick off or wait for training ...
    summary = mon.stop()
    print(f"GPU peak util: {summary['gpu_util_max']:.1f}%  "
          f"VRAM peak: {summary['vram_used_max_mib']:.0f} MiB")

    # Context-manager form:
    with GPUMonitor(Path("run_gpu.csv")) as mon:
        run_training()
    print(mon.summary)
"""

from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# pynvml — optional import
# ---------------------------------------------------------------------------

try:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress pynvml deprecation (use nvidia-ml-py)
        import pynvml  # type: ignore
    _PYNVML_OK = True
except ImportError:
    _PYNVML_OK = False

_IS_LINUX = platform.system() == "Linux"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _container_pids(container_name: str) -> set[int]:
    """Return host-namespace PIDs belonging to *container_name* via docker top."""
    try:
        r = subprocess.run(
            ["docker", "top", container_name, "-eo", "pid"],
            capture_output=True, text=True, timeout=6,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return set()
        lines = r.stdout.strip().splitlines()[1:]   # skip "PID" header
        return {int(ln.strip()) for ln in lines if ln.strip().isdigit()}
    except Exception:
        return set()


def _nvml_power(handle) -> float:
    """Return power draw in Watts, or -1.0 if the sensor is not supported."""
    try:
        return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    except pynvml.NVMLError:
        return -1.0


# ---------------------------------------------------------------------------
# nvidia-smi fallback polling
# ---------------------------------------------------------------------------

_SMI_QUERY = (
    "utilization.gpu,utilization.memory,"
    "memory.used,memory.total,memory.free,"
    "temperature.gpu,power.draw"
)

def _poll_smi(gpu_index: int) -> dict | None:
    """Single nvidia-smi snapshot.  Returns None if the call fails."""
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                f"--query-gpu={_SMI_QUERY}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return None
        line = r.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            return None

        def _f(v: str) -> float | None:
            try:
                return float(v)
            except ValueError:
                return None

        return {
            "ts":              datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "gpu_util_pct":    _f(parts[0]),
            "mem_util_pct":    _f(parts[1]),
            "vram_used_mib":   _f(parts[2]),
            "vram_total_mib":  _f(parts[3]),
            "vram_free_mib":   _f(parts[4]),
            "temp_c":          _f(parts[5]),
            "power_w":         _f(parts[6]),
            "container_vram_mib": 0.0,
        }
    except FileNotFoundError:
        return None   # nvidia-smi not on PATH
    except Exception:
        return None


# ---------------------------------------------------------------------------
# GPUMonitor
# ---------------------------------------------------------------------------

class GPUMonitor:
    """
    Thread-based GPU monitor.  Polls every *interval_s* seconds and appends
    rows to a CSV file.  Call .stop() to join the thread and get a summary.

    Parameters
    ----------
    csv_path        : where to write per-sample data
    container_name  : Docker container to track per-process VRAM for (optional)
    gpu_index       : NVML/nvidia-smi device index (default 0)
    interval_s      : seconds between samples (default 5)
    """

    CSV_COLUMNS = [
        "ts", "backend",
        "gpu_util_pct", "mem_util_pct",
        "vram_used_mib", "vram_total_mib", "vram_free_mib",
        "temp_c", "power_w",
        "container_vram_mib",
    ]

    def __init__(
        self,
        csv_path: Path,
        container_name: Optional[str] = None,
        gpu_index: int = 0,
        interval_s: int = 5,
        append: bool = False,
    ) -> None:
        self._csv_path      = Path(csv_path)
        self._container     = container_name
        self._gpu_index     = gpu_index
        self._interval      = interval_s
        self._append        = append
        self._stop_event    = threading.Event()
        self._thread        = threading.Thread(target=self._loop, daemon=True)
        self._samples: list[dict] = []
        self._backend       = "none"
        self.summary: dict  = {}

        # Decide backend once at init time so the loop has no branching.
        if _PYNVML_OK:
            try:
                pynvml.nvmlInit()
                pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
                self._backend = "pynvml"
            except Exception:
                self._backend = "smi"
        else:
            self._backend = "smi"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> "GPUMonitor":
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop_event.set()
        self._thread.join(timeout=self._interval + 5)
        if _PYNVML_OK and self._backend == "pynvml":
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        self.summary = self._summarize()
        return self.summary

    @property
    def latest_sample(self) -> dict | None:
        """Return a copy of the newest sample, if monitoring has started."""
        return dict(self._samples[-1]) if self._samples else None

    # Context-manager support ──────────────────────────────────────────
    def __enter__(self) -> "GPUMonitor":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _poll_nvml(self) -> dict | None:
        try:
            handle   = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
            util     = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem      = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp     = pynvml.nvmlDeviceGetTemperature(
                           handle, pynvml.NVML_TEMPERATURE_GPU)

            used_mib  = mem.used  / (1024 ** 2)
            total_mib = mem.total / (1024 ** 2)
            free_mib  = mem.free  / (1024 ** 2)

            container_vram = 0.0
            # Per-container VRAM: only reliable on Linux (matching PID namespaces).
            if self._container and _IS_LINUX:
                c_pids = _container_pids(self._container)
                if c_pids:
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    container_vram = sum(
                        p.usedGpuMemory for p in procs if p.pid in c_pids
                    ) / (1024 ** 2)

            return {
                "ts":                   datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "backend":              "pynvml",
                "gpu_util_pct":         float(util.gpu),
                "mem_util_pct":         float(util.memory),
                "vram_used_mib":        used_mib,
                "vram_total_mib":       total_mib,
                "vram_free_mib":        free_mib,
                "temp_c":               float(temp),
                "power_w":              _nvml_power(handle),
                "container_vram_mib":   container_vram,
            }
        except Exception:
            return None

    def _poll(self) -> dict | None:
        if self._backend == "pynvml":
            sample = self._poll_nvml()
        else:
            sample = _poll_smi(self._gpu_index)
        if sample:
            sample.setdefault("backend", self._backend)
        return sample

    def _loop(self) -> None:
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self._append or not self._csv_path.exists()
        mode = "a" if self._append else "w"
        with open(self._csv_path, mode, newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.CSV_COLUMNS,
                                    extrasaction="ignore")
            if write_header:
                writer.writeheader()
            while True:
                sample = self._poll()
                if sample:
                    self._samples.append(sample)
                    writer.writerow(sample)
                    fh.flush()
                if self._stop_event.wait(timeout=self._interval):
                    break

    def _summarize(self) -> dict:
        n = len(self._samples)
        base = {
            "backend":  self._backend,
            "samples":  n,
            "available": self._backend != "none",
        }
        if n == 0:
            return base

        def _values(key: str) -> list[float]:
            return [float(s[key]) for s in self._samples if s.get(key) is not None]

        def _add_summary(summary: dict, key: str, average_key: str, max_key: str) -> None:
            values = _values(key)
            if not values:
                summary["unavailable_metrics"].append(key)
                return
            summary[average_key] = sum(values) / len(values)
            summary[max_key] = max(values)

        summary = {
            **base,
            "unavailable_metrics": [],
        }
        _add_summary(summary, "gpu_util_pct", "gpu_util_avg", "gpu_util_max")
        _add_summary(
            summary,
            "vram_used_mib",
            "vram_used_avg_mib",
            "vram_used_max_mib",
        )
        _add_summary(summary, "temp_c", "temp_avg_c", "temp_max_c")

        total_vram = _values("vram_total_mib")
        if total_vram:
            summary["vram_total_mib"] = total_vram[-1]
        else:
            summary["unavailable_metrics"].append("vram_total_mib")
        free_vram = _values("vram_free_mib")
        if free_vram:
            summary["vram_free_min_mib"] = min(free_vram)
        else:
            summary["unavailable_metrics"].append("vram_free_mib")

        # Power may be unsupported or reported as -1 by NVML.
        power_samples = [value for value in _values("power_w") if value >= 0]
        if power_samples:
            summary["power_avg_w"] = sum(power_samples) / len(power_samples)
            summary["power_max_w"] = max(power_samples)
        else:
            summary["unavailable_metrics"].append("power_w")

        # Per-container VRAM (Linux + container_name only)
        if self._container and _IS_LINUX:
            c_vrams = [s["container_vram_mib"] for s in self._samples]
            summary["container_vram_avg_mib"] = sum(c_vrams) / n
            summary["container_vram_max_mib"] = max(c_vrams)

        return summary

    # ------------------------------------------------------------------
    # Convenience: pretty one-liner for log output
    # ------------------------------------------------------------------

    def format_summary(self, summary: dict | None = None) -> str:
        s = summary or self.summary
        if not s.get("available") or s.get("samples", 0) == 0:
            return f"GPU: N/A (backend={s.get('backend', 'none')})"

        parts = []
        if "gpu_util_avg" in s:
            parts.append(
                f"util avg={s['gpu_util_avg']:.1f}% max={s['gpu_util_max']:.1f}%"
            )
        if "vram_used_avg_mib" in s and "vram_total_mib" in s:
            parts.append(
                f"VRAM avg={s['vram_used_avg_mib']:.0f}MiB"
                f" max={s['vram_used_max_mib']:.0f}MiB"
                f" / {s['vram_total_mib']:.0f}MiB"
            )
        if "temp_avg_c" in s:
            parts.append(
                f"T avg={s['temp_avg_c']:.0f}C max={s['temp_max_c']:.0f}C"
            )
        if "power_avg_w" in s:
            parts.append(f"P avg={s['power_avg_w']:.0f}W max={s['power_max_w']:.0f}W")
        if "container_vram_max_mib" in s:
            parts.append(
                f"container_VRAM max={s['container_vram_max_mib']:.0f}MiB"
            )
        if s.get("unavailable_metrics"):
            parts.append("unavailable=" + ",".join(s["unavailable_metrics"]))
        return "GPU: " + "  ".join(parts) + f"  (n={s['samples']} backend={s['backend']})"


# ---------------------------------------------------------------------------
# CLI — standalone usage
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor GPU utilisation and VRAM. "
                    "Writes per-sample CSV and prints a summary on exit."
    )
    parser.add_argument("--container", default=None,
                        help="Docker container name to track per-process VRAM for "
                             "(Linux only; ignored on Windows).")
    parser.add_argument("--gpu",  type=int, default=0,
                        help="GPU device index (default: 0).")
    parser.add_argument("--interval", type=int, default=5,
                        help="Seconds between samples (default: 5).")
    parser.add_argument("--duration", type=int, default=60,
                        help="How long to monitor in seconds (default: 60). "
                             "Use 0 to run until Ctrl-C.")
    parser.add_argument("--out", default="gpu_monitor.csv",
                        help="Output CSV path (default: gpu_monitor.csv).")
    args = parser.parse_args()

    out_path = Path(args.out)
    mon = GPUMonitor(
        csv_path=out_path,
        container_name=args.container,
        gpu_index=args.gpu,
        interval_s=args.interval,
    )

    print(f"[gpu_monitor] backend={mon._backend}  "
          f"gpu_index={args.gpu}  interval={args.interval}s  "
          f"out={out_path}")
    if args.container:
        print(f"[gpu_monitor] watching container: {args.container}"
              + ("" if _IS_LINUX else " (per-process VRAM: Linux only)"))

    mon.start()
    try:
        if args.duration > 0:
            time.sleep(args.duration)
        else:
            print("[gpu_monitor] Press Ctrl-C to stop.")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print()
    finally:
        summary = mon.stop()
        print(mon.format_summary(summary))
        print(f"[gpu_monitor] CSV written to {out_path}  ({summary.get('samples', 0)} rows)")


if __name__ == "__main__":
    _cli()
