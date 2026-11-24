#!/usr/bin/env python3
"""
stress_tests/load/simulate_month.py — Month-long realistic load simulation.

Generates a Poisson-arrival schedule of k6 bursts spread over N simulation
days following three intensity zones: night / work / evening.  Executes the
schedule in real time (or compressed time via --speed), logging each burst to
a per-day directory and writing a daily summary at midnight boundaries.

Usage
-----
    # Remote 30-day run (Docker k6; target owns refill and monitoring)
    python stress_tests/load/simulate_month.py --remote --base-url https://host.example

    # Real 30-day simulation (leave running in screen / tmux)
    python stress_tests/load/simulate_month.py

    # Quick smoke test — 1 simulated day compressed to ~3 real minutes
    python stress_tests/load/simulate_month.py --smoke

    # 7-day run, 1 real minute = 1 simulated hour (good for demo/CI)
    python stress_tests/load/simulate_month.py --days 7 --speed 60

    # Preview the schedule without running anything
    python stress_tests/load/simulate_month.py --dry-run --days 3

    # Resume after an interruption
    python stress_tests/load/simulate_month.py --resume

Environment variables
---------------------
    BASE_URL    target server          (default: http://localhost:8080)
    K6_IMAGE    Docker image for remote mode (default: grafana/k6:latest)
    RESEED_CMD  shell command to run when pending_pairs falls below threshold
                (default: docker compose exec web python
                stress_tests/fixtures/seed_db.py --pairs 500)

Intensity zones
---------------
    Night   00:00–09:00  VU  3–15   ~1.5 bursts / 8 h
    Work    09:00–17:00  VU 15–50   ~5.0 bursts / 8 h   (two daily peaks: 10:30 + 14:30)
    Evening 17:00–24:00  VU  5–25   ~2.5 bursts / 7 h
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE      = Path(__file__).parent
K6_SCRIPT  = _HERE / "sim_burst.js"
LOG_DIR    = _HERE / "logs"
CHECKPOINT = LOG_DIR / "checkpoint.json"
K6_IMAGE   = os.environ.get("K6_IMAGE", "grafana/k6:latest")

# ---------------------------------------------------------------------------
# Defaults (override via env or CLI)
# ---------------------------------------------------------------------------

BASE_URL         = os.environ.get("BASE_URL", "http://localhost:8080")
RESEED_CMD       = os.environ.get(
    "RESEED_CMD",
    "docker compose exec web python stress_tests/fixtures/seed_db.py --pairs 500",
)
RESEED_THRESHOLD  = 200       # auto-reseed when pending_pairs < this
MIN_REAL_BURST_S  = 10        # minimum real-clock k6 duration in speed>1 mode
CONTAINER_NAME    = os.environ.get("CONTAINER_NAME", "rlhf-asr-web-1")
MONITOR_INTERVAL  = 5         # seconds between docker-stats polls during a burst

# ---------------------------------------------------------------------------
# Zone table: (name, hour_start, hour_end, vus_min, vus_max, mean_bursts/hour)
# ---------------------------------------------------------------------------

ZONES = [
    #  name       h0   h1  vmin  vmax   λ/h
    ("night",      0,   9,    3,   15,  0.19),   # ~1.5 events in 8 h
    ("work",       9,  17,   15,   50,  0.62),   # ~5.0 events in 8 h
    ("evening",   17,  24,    5,   25,  0.36),   # ~2.5 events in 7 h
]

# Simulated burst duration range [s] per zone
BURST_RANGE_S = {
    "night":   (30,  90),
    "work":    (60, 300),
    "evening": (45, 180),
}

# Work-zone intra-day peaks: Gaussian bumps at 10:30 and 14:30.
# Applied as a VU multiplier in [1.0, 1.5].
_PEAK_CENTERS = [10.5, 14.5]
_PEAK_SIGMA   = 0.8
_PEAK_SCALE   = 0.5


def _work_vu_multiplier(hour: float) -> float:
    bump = max(
        math.exp(-0.5 * ((hour - c) / _PEAK_SIGMA) ** 2)
        for c in _PEAK_CENTERS
    )
    return 1.0 + _PEAK_SCALE * bump


# ---------------------------------------------------------------------------
# Poisson sampler (no numpy/scipy required)
# ---------------------------------------------------------------------------

def _poisson(rng: random.Random, lam: float) -> int:
    """Sample a Poisson count, splitting large means into stable chunks."""
    if lam <= 0:
        return 0
    if lam > 30:
        chunks, remainder = divmod(lam, 30)
        return sum(_poisson(rng, 30) for _ in range(int(chunks))) + _poisson(
            rng, remainder
        )
    L = math.exp(-lam)
    k, p = 0, 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BurstEvent:
    day:           int   # 0-indexed simulation day
    day_offset_s:  int   # seconds from midnight of that day
    zone:          str
    n_vus:         int
    sim_duration_s: int  # simulated burst length in seconds

    @property
    def sim_ts(self) -> int:
        """Absolute seconds from simulation epoch (day 0 00:00)."""
        return self.day * 86400 + self.day_offset_s


# ---------------------------------------------------------------------------
# Schedule generation
# ---------------------------------------------------------------------------

def _allocate_zone_counts(total: int) -> dict[str, int]:
    weights = {
        name: (hour_end - hour_start) * rate
        for name, hour_start, hour_end, _, _, rate in ZONES
    }
    weight_total = sum(weights.values())
    raw_counts = {name: total * weight / weight_total for name, weight in weights.items()}
    counts = {name: int(count) for name, count in raw_counts.items()}
    remaining = total - sum(counts.values())
    for name in sorted(
        weights,
        key=lambda zone: raw_counts[zone] - counts[zone],
        reverse=True,
    )[:remaining]:
        counts[name] += 1
    return counts


def generate_schedule(
    days: int,
    rng: random.Random,
    bursts_per_day: int | None = None,
    average_bursts_per_day: float | None = None,
) -> list[BurstEvent]:
    """
    Return a sorted list of BurstEvents for `days` simulated days.
    Uses the normal Poisson rates by default. `bursts_per_day` allocates an
    exact daily count; `average_bursts_per_day` scales the Poisson means.
    """
    if bursts_per_day is not None and average_bursts_per_day is not None:
        raise ValueError("exact and average daily burst counts are mutually exclusive")
    events: list[BurstEvent] = []
    fixed_zone_counts = (
        _allocate_zone_counts(bursts_per_day)
        if bursts_per_day is not None
        else None
    )
    normal_daily_mean = sum((h1 - h0) * rate for _, h0, h1, _, _, rate in ZONES)
    average_scale = (
        average_bursts_per_day / normal_daily_mean
        if average_bursts_per_day is not None
        else 1.0
    )

    for day in range(days):
        day_events: list[BurstEvent] = []

        for zone_name, h0, h1, vmin, vmax, lam_per_h in ZONES:
            zone_hours = h1 - h0
            n = (
                fixed_zone_counts[zone_name]
                if fixed_zone_counts is not None
                else _poisson(rng, lam_per_h * zone_hours * average_scale)
            )

            for _ in range(n):
                hour_in_zone = rng.uniform(0, zone_hours)
                abs_hour     = h0 + hour_in_zone
                offset_s     = int(abs_hour * 3600)

                # VU count — work zone gets Gaussian peaks at 10:30 and 14:30
                if zone_name == "work":
                    mult  = _work_vu_multiplier(abs_hour)
                    n_vus = min(vmax, int(rng.randint(vmin, vmax) * mult))
                else:
                    n_vus = rng.randint(vmin, vmax)

                dur_s = rng.randint(*BURST_RANGE_S[zone_name])
                day_events.append(BurstEvent(
                    day=day,
                    day_offset_s=offset_s,
                    zone=zone_name,
                    n_vus=n_vus,
                    sim_duration_s=dur_s,
                ))

        day_events.sort(key=lambda e: e.day_offset_s)
        events.extend(day_events)

    return events


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _make_day_logger(sim_date: datetime, log_dir: Path) -> logging.Logger:
    """Return a logger that writes to logs/YYYY-MM-DD/orchestrator.log + stdout."""
    day_dir = log_dir / sim_date.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)

    name   = f"sim.{sim_date.strftime('%Y-%m-%d')}"
    logger = logging.getLogger(name)

    if logger.handlers:          # already configured (same day called twice)
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                             datefmt="%H:%M:%S")

    fh = logging.FileHandler(day_dir / "orchestrator.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# k6 execution
# ---------------------------------------------------------------------------

def _assert_k6() -> None:
    if not shutil.which("k6"):
        sys.exit(
            "ERROR: k6 not found in PATH.\n"
            "Install from https://k6.io/docs/get-started/installation/\n"
            "or run via Docker: docker run --rm grafana/k6 ..."
        )


def _assert_docker() -> None:
    if not shutil.which("docker"):
        sys.exit("ERROR: docker not found in PATH.")
    result = subprocess.run(
        ["docker", "info"], capture_output=True, text=True, timeout=20
    )
    if result.returncode != 0:
        sys.exit(f"ERROR: Docker daemon is unavailable:\n{result.stderr.strip()}")


def _preflight(base_url: str, min_pending_pairs: int = 1) -> dict:
    target = base_url.rstrip("/")
    try:
        with urllib.request.urlopen(f"{target}/setup", timeout=10) as response:
            if response.status != 200:
                raise RuntimeError(f"GET /setup returned HTTP {response.status}")
        with urllib.request.urlopen(f"{target}/api/stats", timeout=10) as response:
            stats = json.loads(response.read())
    except Exception as exc:
        sys.exit(f"ERROR: deployment preflight failed for {target}: {exc}")
    try:
        pending_pairs = int(stats["pending_pairs"])
    except (KeyError, TypeError, ValueError):
        sys.exit("ERROR: deployment preflight returned no valid pending_pairs count")
    if pending_pairs < min_pending_pairs:
        sys.exit(
            "ERROR: deployment has "
            f"{pending_pairs} pending pairs; at least {min_pending_pairs} are required"
        )
    print(
        "[preflight] deployment reachable; "
        f"pending_pairs={pending_pairs} (required={min_pending_pairs})"
    )
    return stats


def _validate_remote_url(base_url: str) -> str:
    target = base_url.rstrip("/")
    parsed = urlparse(target)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "remote --base-url must be an HTTPS origin without a path or trailing slash"
        )
    return target


def _validate_resume_identity(
    checkpoint: dict,
    base_url: str,
    remote: bool,
    k6_runtime: str,
) -> None:
    checkpoint_target = checkpoint.get("base_url")
    if checkpoint_target and checkpoint_target != base_url:
        raise ValueError(
            "--base-url differs from the resumed checkpoint "
            f"({checkpoint_target})"
        )
    checkpoint_remote = checkpoint.get("remote")
    if checkpoint_remote is not None and bool(checkpoint_remote) != remote:
        raise ValueError("--remote mode differs from the resumed checkpoint")
    checkpoint_runtime = checkpoint.get("k6_runtime")
    if checkpoint_runtime and checkpoint_runtime != k6_runtime:
        raise ValueError("--k6-runtime differs from the resumed checkpoint")


# ---------------------------------------------------------------------------
# Resource monitoring (docker stats polling in a background thread)
# ---------------------------------------------------------------------------

def _parse_mem_mib(s: str) -> float:
    """Convert '95.7MiB', '1.23GiB', '512kB' -> float MiB."""
    s = s.strip()
    try:
        if s.endswith("GiB"):   return float(s[:-3]) * 1024.0
        if s.endswith("MiB"):   return float(s[:-3])
        if s.endswith("KiB"):   return float(s[:-3]) / 1024.0
        if s.endswith("kB"):    return float(s[:-2]) / 1024.0
        if s.endswith("MB"):    return float(s[:-2])
        if s.endswith("GB"):    return float(s[:-2]) * 1024.0
        if s.endswith("B"):     return float(s[:-1]) / (1024.0 ** 2)
    except ValueError:
        pass
    return 0.0


class ResourceMonitor:
    """
    Polls `docker stats` for a container in a background thread.

    Writes one CSV row per sample to `csv_path` (columns: ts, cpu_pct,
    mem_mib, mem_pct).  Call .start() before k6, .stop() after — stop()
    returns a summary dict with avg/max CPU and memory.

    GPU note: the web service uses CPU-only PyTorch (python:3.12-slim base
    image, no CUDA driver inside container).  GPU monitoring is N/A for this
    service; the training container would need a separate monitor.
    """

    def __init__(self, container: str, interval_s: int, csv_path: Path) -> None:
        self._container  = container
        self._interval   = interval_s
        self._csv_path   = csv_path
        self._stop_event = threading.Event()
        self._thread     = threading.Thread(target=self._loop, daemon=True)
        self._samples: list[dict] = []

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict:
        """Signal stop, wait for thread, return summary."""
        self._stop_event.set()
        self._thread.join(timeout=self._interval + 3)
        return self._summarize()

    # ── internal ─────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        with open(self._csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["ts", "cpu_pct", "mem_mib", "mem_pct"])
            # first sample immediately, then every interval
            while True:
                sample = self._poll()
                if sample:
                    self._samples.append(sample)
                    writer.writerow([
                        sample["ts"],
                        f"{sample['cpu_pct']:.2f}",
                        f"{sample['mem_mib']:.1f}",
                        f"{sample['mem_pct']:.2f}",
                    ])
                    fh.flush()
                if self._stop_event.wait(timeout=self._interval):
                    break   # stop was set

    def _poll(self) -> dict | None:
        try:
            r = subprocess.run(
                [
                    "docker", "stats", "--no-stream",
                    "--format",
                    "{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}",
                    self._container,
                ],
                capture_output=True, text=True, timeout=8,
            )
            if r.returncode != 0 or not r.stdout.strip():
                return None
            line = r.stdout.strip().splitlines()[0]
            parts = line.split("\t")
            if len(parts) < 3:
                return None
            cpu_str, mem_str, mem_pct_str = parts[:3]
            mem_used_str = mem_str.split("/")[0].strip()
            return {
                "ts":      time.strftime("%H:%M:%S"),
                "cpu_pct": float(cpu_str.strip().rstrip("%") or 0),
                "mem_mib": _parse_mem_mib(mem_used_str),
                "mem_pct": float(mem_pct_str.strip().rstrip("%") or 0),
            }
        except Exception:
            return None

    def _summarize(self) -> dict:
        n = len(self._samples)
        if n == 0:
            return {"samples": 0, "cpu_avg": 0.0, "cpu_max": 0.0,
                    "mem_avg_mib": 0.0, "mem_max_mib": 0.0, "mem_pct_max": 0.0}
        cpus     = [s["cpu_pct"] for s in self._samples]
        mems     = [s["mem_mib"] for s in self._samples]
        mem_pcts = [s["mem_pct"] for s in self._samples]
        return {
            "samples":     n,
            "cpu_avg":     sum(cpus)     / n,
            "cpu_max":     max(cpus),
            "mem_avg_mib": sum(mems)     / n,
            "mem_max_mib": max(mems),
            "mem_pct_max": max(mem_pcts),
        }


def run_burst(
    event: BurstEvent,
    speed: int,
    sim_start: datetime,
    log_dir: Path,
    base_url: str,
    container: str = CONTAINER_NAME,
    monitor_interval: int = MONITOR_INTERVAL,
    k6_runtime: str = "native",
    monitor_container: bool = True,
    fetch_audio: bool = False,
    run_id: str = "",
) -> dict:
    """Execute one k6 burst, monitor container resources, return result dict."""
    sim_dt    = sim_start + timedelta(seconds=event.sim_ts)
    day_dir   = log_dir / sim_dt.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)

    # In speed mode, compress burst duration (floor at MIN_REAL_BURST_S)
    real_dur_s   = max(MIN_REAL_BURST_S, event.sim_duration_s // speed)
    duration_arg = f"{real_dur_s}s"
    ts_str       = sim_dt.strftime("%H-%M-%S")

    artifact_stem = f"{ts_str}_{event.n_vus:02d}vus_{event.zone}"
    burst_log = day_dir / f"{artifact_stem}.txt"
    summary_json = day_dir / f"{artifact_stem}_summary.json"
    res_csv = day_dir / f"{artifact_stem}_resources.csv"

    log = _make_day_logger(sim_dt, log_dir)
    monitor_label = container if monitor_container else "disabled"
    log.info(
        "BURST >> zone=%-8s  vus=%2d  sim=%3ds  real=%3ds  monitor=%s",
        event.zone, event.n_vus, event.sim_duration_s, real_dur_s, monitor_label,
    )

    summary_target = (
        f"/results/{summary_json.name}"
        if k6_runtime == "docker"
        else str(summary_json)
    )
    k6_args = [
        "run",
        "--no-color",
        "--env", f"BASE_URL={base_url}",
        "--env", f"SIM_VUS={event.n_vus}",
        "--env", f"SIM_DURATION={duration_arg}",
        "--env", f"SIM_FETCH_AUDIO={'1' if fetch_audio else '0'}",
        "--env", f"DIAGNOSTICS_RUN_ID={run_id}",
        "--summary-export", summary_target,
        "--summary-trend-stats", "p(50),p(90),p(95),p(99)",
    ]
    if k6_runtime == "docker":
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{_HERE.resolve()}:/scripts:ro",
            "-v", f"{day_dir.resolve()}:/results",
            K6_IMAGE,
            *k6_args,
            "/scripts/sim_burst.js",
        ]
    else:
        cmd = ["k6", *k6_args, str(K6_SCRIPT)]

    monitor = None
    if monitor_container:
        monitor = ResourceMonitor(container, monitor_interval, res_csv)
        monitor.start()

    wall0 = time.monotonic()
    with open(burst_log, "w", encoding="utf-8") as fh:
        fh.write(
            f"# sim_time   : {sim_dt.isoformat()}\n"
            f"# zone       : {event.zone}\n"
            f"# vus        : {event.n_vus}\n"
            f"# sim_dur    : {event.sim_duration_s}s\n"
            f"# real_dur   : {real_dur_s}s  (speed={speed}x)\n"
            f"# monitor    : {monitor_label}\n"
            f"# k6_runtime : {k6_runtime}\n"
            f"# fetch_audio: {fetch_audio}\n"
            f"# run_id     : {run_id}\n"
            f"# cmd        : {' '.join(cmd)}\n\n"
        )
        proc = subprocess.run(cmd, stdout=fh, stderr=fh, text=True)

    res = (
        monitor.stop()
        if monitor is not None
        else {
            "samples": 0,
            "cpu_avg": 0.0,
            "cpu_max": 0.0,
            "mem_avg_mib": 0.0,
            "mem_max_mib": 0.0,
            "mem_pct_max": 0.0,
        }
    )

    elapsed = time.monotonic() - wall0
    rc      = proc.returncode
    status  = "OK" if rc == 0 else f"FAIL(rc={rc})"
    log.info(
        "BURST ## %s  wall=%.1fs  "
        "CPU avg=%.1f%% max=%.1f%%  Mem avg=%.0fMiB max=%.0fMiB  (n=%d samples)",
        status, elapsed,
        res["cpu_avg"], res["cpu_max"],
        res["mem_avg_mib"], res["mem_max_mib"],
        res["samples"],
    )

    return {
        "sim_ts":          sim_dt.isoformat(),
        "zone":            event.zone,
        "n_vus":           event.n_vus,
        "sim_duration_s":  event.sim_duration_s,
        "real_duration_s": real_dur_s,
        "exit_code":       rc,
        "log":             str(burst_log),
        "summary":         str(summary_json),
        "resources":       res,
        "run_id":          run_id,
    }


# ---------------------------------------------------------------------------
# Auto-reseed
# ---------------------------------------------------------------------------

def _maybe_reseed(base_url: str, log: logging.Logger) -> None:
    try:
        with urllib.request.urlopen(f"{base_url}/api/stats", timeout=5) as r:
            stats = json.loads(r.read())
        pending = int(stats.get("pending_pairs", RESEED_THRESHOLD + 1))
        if pending < RESEED_THRESHOLD:
            log.warning("pending_pairs=%d < %d — triggering reseed …", pending, RESEED_THRESHOLD)
            result = subprocess.run(
                RESEED_CMD.split(), capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                log.info("Reseed OK: %s", result.stdout.strip().splitlines()[-1])
            else:
                log.error("Reseed FAILED:\n%s", result.stderr.strip())
    except Exception as exc:
        log.warning("Stats/reseed check failed: %s", exc)


# ---------------------------------------------------------------------------
# Daily summary
# ---------------------------------------------------------------------------

def _write_daily_summary(
    results: list[dict], sim_date: datetime, log_dir: Path
) -> None:
    day_dir = log_dir / sim_date.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)

    total    = len(results)
    ok_count = sum(1 for r in results if r["exit_code"] == 0)
    by_zone: dict[str, list[dict]] = {}
    for r in results:
        by_zone.setdefault(r["zone"], []).append(r)

    # Aggregate resource samples across all bursts of the day
    all_res = [r.get("resources", {}) for r in results if r.get("resources", {}).get("samples", 0) > 0]
    cpu_maxes = [r["cpu_max"]     for r in all_res]
    cpu_avgs  = [r["cpu_avg"]     for r in all_res]
    mem_maxes = [r["mem_max_mib"] for r in all_res]
    mem_avgs  = [r["mem_avg_mib"] for r in all_res]

    lines = [
        f"# Daily summary — {sim_date.strftime('%Y-%m-%d')}",
        f"Bursts total : {total}",
        f"Passed (rc=0): {ok_count}",
        f"Failed       : {total - ok_count}",
        "",
        "Zone breakdown:",
    ]
    for zone_name in ("night", "work", "evening"):
        zone_rs = by_zone.get(zone_name, [])
        if not zone_rs:
            continue
        vus  = [r["n_vus"] for r in zone_rs]
        ok_z = sum(1 for r in zone_rs if r["exit_code"] == 0)
        lines.append(
            f"  {zone_name:<8s}: {len(zone_rs):3d} bursts  "
            f"vus {min(vus):2d}-{max(vus):2d}  "
            f"ok={ok_z}/{len(zone_rs)}"
        )

    if all_res:
        lines += [
            "",
            "Resource usage (web container — CPU-only, GPU: N/A):",
            f"  CPU  : avg={sum(cpu_avgs)/len(cpu_avgs):.1f}%   max={max(cpu_maxes):.1f}%",
            f"  Mem  : avg={sum(mem_avgs)/len(mem_avgs):.0f} MiB  max={max(mem_maxes):.0f} MiB",
            f"  Samples collected: {sum(r['samples'] for r in all_res)}",
        ]
    else:
        lines += ["", "Resource usage: no samples collected (docker stats unavailable?)"]

    path = day_dir / "summary.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def _save_checkpoint(state: dict, checkpoint: Path = CHECKPOINT) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    tmp = checkpoint.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(checkpoint)


def _load_checkpoint(checkpoint: Path = CHECKPOINT) -> Optional[dict]:
    if checkpoint.exists():
        return json.loads(checkpoint.read_text(encoding="utf-8"))
    return None


# ---------------------------------------------------------------------------
# Monthly report
# ---------------------------------------------------------------------------

def _write_monthly_report(
    results: list[dict],
    sim_start: datetime,
    days: int,
    log_dir: Path,
    run_id: str = "",
) -> None:
    total    = len(results)
    ok_count = sum(1 for r in results if r["exit_code"] == 0)
    by_zone: dict[str, list[dict]] = {}
    for r in results:
        by_zone.setdefault(r["zone"], []).append(r)

    lines = [
        "# Monthly Load Simulation Report",
        f"# Generated : {datetime.now().isoformat()}",
        f"# sim_start : {sim_start.isoformat()}",
        f"# sim_days  : {days}",
        f"# run_id    : {run_id}",
        "",
        f"Total bursts   : {total}",
        f"Passed (rc=0)  : {ok_count}",
        f"Failed         : {total - ok_count}",
        f"Pass rate      : {100*ok_count/max(total,1):.1f} %",
        "",
        "Zone breakdown:",
    ]
    for zone_name in ("night", "work", "evening"):
        rs = by_zone.get(zone_name, [])
        if not rs:
            continue
        vus  = [r["n_vus"] for r in rs]
        durs = [r["sim_duration_s"] for r in rs]
        ok_z = sum(1 for r in rs if r["exit_code"] == 0)
        lines.append(
            f"  {zone_name:<8s}: {len(rs):4d} bursts  "
            f"avg_vus={sum(vus)/len(vus):.0f}  "
            f"avg_sim_dur={sum(durs)/len(durs):.0f}s  "
            f"ok={ok_z}/{len(rs)}"
        )

    # Resource peak across entire simulation
    all_res = [r.get("resources", {}) for r in results if r.get("resources", {}).get("samples", 0) > 0]
    lines.append("")
    if all_res:
        total_samples = sum(r["samples"] for r in all_res)
        overall_cpu_avg = sum(r["cpu_avg"] * r["samples"] for r in all_res) / total_samples
        overall_mem_avg = sum(r["mem_avg_mib"] * r["samples"] for r in all_res) / total_samples
        peak_cpu = max(r["cpu_max"]     for r in all_res)
        peak_mem = max(r["mem_max_mib"] for r in all_res)

        # Find which burst hit peak CPU / memory
        peak_cpu_burst = max(results, key=lambda r: r.get("resources", {}).get("cpu_max", 0))
        peak_mem_burst = max(results, key=lambda r: r.get("resources", {}).get("mem_max_mib", 0))

        lines += [
            "Resource usage (web container — CPU-only, GPU: N/A):",
            f"  Total samples    : {total_samples}",
            f"  CPU  overall avg : {overall_cpu_avg:.1f}%",
            f"  CPU  peak        : {peak_cpu:.1f}%"
            f"  (burst {peak_cpu_burst['sim_ts']}  zone={peak_cpu_burst['zone']}"
            f"  vus={peak_cpu_burst['n_vus']})",
            f"  Mem  overall avg : {overall_mem_avg:.0f} MiB",
            f"  Mem  peak        : {peak_mem:.0f} MiB"
            f"  (burst {peak_mem_burst['sim_ts']}  zone={peak_mem_burst['zone']}"
            f"  vus={peak_mem_burst['n_vus']})",
        ]
    else:
        lines.append("Resource usage: no samples collected.")

    path = log_dir / "monthly_report.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nMonthly report -> {path}")


# ---------------------------------------------------------------------------
# Schedule preview (--dry-run)
# ---------------------------------------------------------------------------

def _print_schedule(schedule: list[BurstEvent], sim_start: datetime, speed: int) -> None:
    n_by_zone: dict[str, int] = {}
    for e in schedule:
        n_by_zone[e.zone] = n_by_zone.get(e.zone, 0) + 1

    print(f"\nSchedule: {len(schedule)} bursts over {schedule[-1].day + 1} day(s)")
    for z, n in sorted(n_by_zone.items()):
        print(f"  {z:<8s}: {n} bursts")
    print()
    print(f"{'Day':>4}  {'Sim time':>8}  {'Zone':<8}  {'VUs':>4}  {'Sim dur':>7}  {'Real dur':>8}")
    print("-" * 55)
    for e in schedule[:80]:   # show first 80 lines to avoid flooding terminal
        # Display logical simulation time-of-day, not real wall-clock time
        h = e.day_offset_s // 3600
        m = (e.day_offset_s % 3600) // 60
        s = e.day_offset_s % 60
        time_str = f"{h:02d}:{m:02d}:{s:02d}"
        real_dur = max(MIN_REAL_BURST_S, e.sim_duration_s // speed)
        print(
            f"{e.day:>4}  {time_str:>8}  {e.zone:<8}  "
            f"{e.n_vus:>4}  {e.sim_duration_s:>5}s  {real_dur:>6}s"
        )
    if len(schedule) > 80:
        print(f"  ... {len(schedule) - 80} more events not shown (use --dry-run with fewer --days)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Month-long RLHF load simulation orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--days",     type=int,   default=30,      help="Simulated days (default: 30)")
    parser.add_argument("--speed",    type=int,   default=1,       help="Time compression 1x = real time, 1440x = 1 min/day (default: 1)")
    parser.add_argument("--seed",     type=int,   default=None,    help="Random seed for schedule reproducibility")
    burst_density = parser.add_mutually_exclusive_group()
    burst_density.add_argument("--bursts-per-day", type=int, default=None, help="Exact daily burst count allocated across normal night/work/evening proportions")
    burst_density.add_argument("--average-bursts-per-day", type=float, default=None, help="Average daily burst count with Poisson variation and normal zone proportions")
    parser.add_argument("--base-url",             default=BASE_URL, help="Target server URL")
    parser.add_argument("--min-pending-pairs", type=int, default=1, help="Local-mode queue floor checked before starting (default: 1; ignored in remote mode)")
    parser.add_argument("--log-dir",  type=Path,  default=LOG_DIR, help="Root log directory")
    parser.add_argument("--run-id", default=None, help="UUID4 from the server Admin diagnostics capture")
    parser.add_argument("--resume",           action="store_true",        help="Resume from checkpoint.json")
    parser.add_argument("--smoke",            action="store_true",        help="Quick validation: --days 1 --speed 1440 (~3 min total)")
    parser.add_argument("--dry-run",          action="store_true",        help="Preview schedule then exit (no k6 run)")
    parser.add_argument("--container",        default=CONTAINER_NAME,     help=f"Docker container to monitor (default: {CONTAINER_NAME})")
    parser.add_argument("--monitor-interval", type=int, default=MONITOR_INTERVAL, help=f"Seconds between docker-stats polls (default: {MONITOR_INTERVAL})")
    parser.add_argument("--remote", action="store_true", help="Remote deployment mode: Docker k6, audio enabled, no local monitor or reseed")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed k6 burst instead of recording it and continuing")
    parser.add_argument("--k6-runtime", choices=("native", "docker"), default=None, help="k6 execution method (remote default: docker; local default: native)")
    parser.add_argument("--fetch-audio", dest="fetch_audio", action="store_true", help="Download each pair's audio during bursts")
    parser.add_argument("--no-fetch-audio", dest="fetch_audio", action="store_false", help="Skip audio downloads")
    parser.set_defaults(fetch_audio=None)
    parser.add_argument("--no-container-monitor", action="store_true", help="Do not poll local docker stats")
    parser.add_argument("--no-reseed", action="store_true", help="Do not execute the local RESEED_CMD")
    args = parser.parse_args()

    k6_runtime = args.k6_runtime or ("docker" if args.remote else "native")
    fetch_audio = args.fetch_audio if args.fetch_audio is not None else args.remote
    monitor_container = not (args.remote or args.no_container_monitor)
    reseed = not (args.remote or args.no_reseed)
    checkpoint = args.log_dir / "checkpoint.json"
    if args.min_pending_pairs < 1:
        parser.error("--min-pending-pairs must be at least 1")
    if args.bursts_per_day is not None and args.bursts_per_day < 1:
        parser.error("--bursts-per-day must be at least 1")
    if (
        args.average_bursts_per_day is not None
        and args.average_bursts_per_day <= 0
    ):
        parser.error("--average-bursts-per-day must be greater than 0")

    args.base_url = args.base_url.rstrip("/")
    if args.remote:
        try:
            args.base_url = _validate_remote_url(args.base_url)
        except ValueError as exc:
            parser.error(str(exc))

    # --smoke implies --days 1 --speed 1440
    if args.smoke:
        args.days  = 1
        args.speed = 1440

    if args.run_id:
        try:
            requested_run_id = uuid.UUID(args.run_id)
        except ValueError:
            parser.error("--run-id must be a canonical UUID4")
        if requested_run_id.version != 4 or str(requested_run_id) != args.run_id:
            parser.error("--run-id must be a canonical UUID4")
    resume_checkpoint = _load_checkpoint(checkpoint) if args.resume else None
    if args.resume and resume_checkpoint is None:
        parser.error(f"--resume requires an existing checkpoint: {checkpoint}")
    if resume_checkpoint is not None:
        try:
            _validate_resume_identity(
                resume_checkpoint,
                args.base_url,
                args.remote,
                k6_runtime,
            )
        except ValueError as exc:
            parser.error(str(exc))
        checkpoint_run_id = resume_checkpoint.get("run_id")
        if args.run_id and checkpoint_run_id and args.run_id != checkpoint_run_id:
            parser.error("--run-id differs from the resumed checkpoint")

    if not args.dry_run:
        _assert_docker() if k6_runtime == "docker" else _assert_k6()
        if not args.remote:
            _preflight(args.base_url, args.min_pending_pairs)

    args.log_dir.mkdir(parents=True, exist_ok=True)

    # ── Build or load schedule ────────────────────────────────────────────────
    if resume_checkpoint is not None:
        cp = resume_checkpoint
        print(f"[resume] checkpoint: {checkpoint}")
        schedule   = [BurstEvent(**e) for e in cp["schedule"]]
        start_idx  = cp["next_event_idx"]
        sim_start  = datetime.fromisoformat(cp["sim_start"])
        rng_seed   = cp["seed"]
        results    = cp.get("results", [])
        speed      = cp["speed"]
        days       = cp["days"]
        bursts_per_day = cp.get("bursts_per_day")
        average_bursts_per_day = cp.get("average_bursts_per_day")
        run_id     = cp.get("run_id") or args.run_id or str(uuid.uuid4())
        print(f"[resume] {start_idx}/{len(schedule)} events already done")
    else:
        run_id = args.run_id or str(uuid.uuid4())
        rng_seed  = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
        rng       = random.Random(rng_seed)
        schedule  = generate_schedule(
            args.days,
            rng,
            args.bursts_per_day,
            args.average_bursts_per_day,
        )
        sim_start = datetime.now().replace(microsecond=0)
        start_idx = 0
        results   = []
        speed     = args.speed
        days      = args.days
        bursts_per_day = args.bursts_per_day
        average_bursts_per_day = args.average_bursts_per_day
        if not args.dry_run:
            _save_checkpoint({
                "sim_start":       sim_start.isoformat(),
                "days":            days,
                "bursts_per_day":  bursts_per_day,
                "average_bursts_per_day": average_bursts_per_day,
                "speed":           speed,
                "seed":            rng_seed,
                "next_event_idx":  0,
                "schedule":        [asdict(e) for e in schedule],
                "results":         [],
                "run_id":          run_id,
                "base_url":        args.base_url,
                "remote":          args.remote,
                "k6_runtime":      k6_runtime,
                "fetch_audio":     fetch_audio,
            }, checkpoint)

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        _print_schedule(schedule, sim_start, speed)
        return

    # ── Banner ────────────────────────────────────────────────────────────────
    total_events = len(schedule)
    remaining    = total_events - start_idx
    print(f"\n{'='*62}")
    print(f"  RLHF Monthly Simulation")
    print(f"  sim days     : {days}")
    print(f"  speed        : {speed}x {'(real time)' if speed == 1 else f'(1 min = {speed//60}h sim)'}")
    print(f"  sim_start    : {sim_start}")
    print(f"  total bursts : {total_events}  ({remaining} remaining)")
    density = (
        f"exactly {bursts_per_day}"
        if bursts_per_day is not None
        else f"Poisson average {average_bursts_per_day:g}"
        if average_bursts_per_day is not None
        else "Poisson default"
    )
    print(f"  bursts/day   : {density}")
    print(f"  random seed  : {rng_seed}")
    print(f"  run ID       : {run_id}")
    print(f"  log dir      : {args.log_dir}")
    print(f"  target       : {args.base_url}")
    print(f"  k6 runtime   : {k6_runtime}")
    print(f"  fetch audio  : {fetch_audio}")
    print(f"  failures     : {'stop immediately' if args.fail_fast else 'record and continue'}")
    print(f"  reseed       : {reseed}")
    print(f"  monitor      : {args.container if monitor_container else 'disabled'}")
    print(f"  GPU          : N/A (web service is CPU-only)")
    print(f"{'='*62}\n")

    # ── Real-time origin for timing calculations ──────────────────────────────
    # sim_origin: simulated seconds already elapsed (0 for new runs, >0 for resumes)
    sim_origin  = schedule[start_idx].sim_ts if start_idx < len(schedule) else 0
    real_origin = time.monotonic()

    current_day = -1
    day_results: list[dict] = []
    log: logging.Logger | None = None

    for idx, event in enumerate(schedule[start_idx:], start=start_idx):

        # ── Day boundary ──────────────────────────────────────────────────────
        if event.day != current_day:
            if current_day >= 0 and day_results:
                _write_daily_summary(
                    day_results,
                    sim_start + timedelta(days=current_day),
                    args.log_dir,
                )
            day_results  = []
            current_day  = event.day
            sim_day_dt   = sim_start + timedelta(days=current_day)
            log          = _make_day_logger(sim_day_dt, args.log_dir)
            log.info("=== Simulation day %d / %d  (%s) ===",
                     current_day + 1, days, sim_day_dt.strftime("%Y-%m-%d"))

        # ── Sleep until event's simulated time ────────────────────────────────
        target_real = (event.sim_ts - sim_origin) / speed
        wait        = target_real - (time.monotonic() - real_origin)
        if wait > 1.0:
            sim_hm = f"{event.day_offset_s // 3600:02d}:{(event.day_offset_s % 3600) // 60:02d}"
            log.info("  sleep %.1f s  ->  next burst @ sim %s  zone=%s  vus=%d",
                     wait, sim_hm, event.zone, event.n_vus)
            time.sleep(wait)

        # ── Auto-reseed if queue running low ──────────────────────────────────
        if reseed:
            _maybe_reseed(args.base_url, log)

        # ── Run k6 + collect resources ────────────────────────────────────────
        result = run_burst(
            event, speed, sim_start, args.log_dir, args.base_url,
            container=args.container,
            monitor_interval=args.monitor_interval,
            k6_runtime=k6_runtime,
            monitor_container=monitor_container,
            fetch_audio=fetch_audio,
            run_id=run_id,
        )
        day_results.append(result)
        results.append(result)

        failed_burst = result["exit_code"] != 0

        # ── Checkpoint (atomic write) ─────────────────────────────────────────
        _save_checkpoint({
            "sim_start":      sim_start.isoformat(),
            "days":           days,
            "bursts_per_day": bursts_per_day,
            "average_bursts_per_day": average_bursts_per_day,
            "speed":          speed,
            "seed":           rng_seed,
            "next_event_idx": idx + 1,
            "schedule":       [asdict(e) for e in schedule],
            "results":        results,
            "run_id":         run_id,
            "base_url":       args.base_url,
            "remote":         args.remote,
            "k6_runtime":     k6_runtime,
            "fetch_audio":    fetch_audio,
        }, checkpoint)
        if failed_burst and args.fail_fast:
            sys.exit(
                "ERROR: burst failed; inspect its k6 log before resuming: "
                f"{result['log']}"
            )

    # ── Final day summary + monthly report ───────────────────────────────────
    if day_results and current_day >= 0:
        _write_daily_summary(
            day_results,
            sim_start + timedelta(days=current_day),
            args.log_dir,
        )

    _write_monthly_report(results, sim_start, days, args.log_dir, run_id)
    print("\nSimulation complete.")


if __name__ == "__main__":
    main()
