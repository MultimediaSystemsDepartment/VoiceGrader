from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_key_values(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, raw = line.partition(":")
        if not separator:
            continue
        token = raw.strip().split()[0]
        try:
            values[key] = int(token)
        except (IndexError, ValueError):
            continue
    return values


def _read_cpu(path: Path) -> tuple[int, int]:
    first = path.read_text(encoding="utf-8").splitlines()[0].split()
    if not first or first[0] != "cpu":
        raise ValueError(f"Missing aggregate CPU row in {path}")
    counters = [int(value) for value in first[1:]]
    idle = counters[3] + (counters[4] if len(counters) > 4 else 0)
    return sum(counters), idle


def _read_network(path: Path) -> tuple[int, int]:
    received = sent = 0
    for line in path.read_text(encoding="utf-8").splitlines()[2:]:
        if ":" not in line:
            continue
        interface, raw = line.split(":", 1)
        if interface.strip() == "lo":
            continue
        fields = raw.split()
        if len(fields) >= 9:
            received += int(fields[0])
            sent += int(fields[8])
    return received, sent


def _read_disk_io(path: Path) -> tuple[int, int]:
    sectors_read = sectors_written = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 14:
            continue
        name = fields[2]
        if name.startswith(("loop", "ram", "fd")):
            continue
        sectors_read += int(fields[5])
        sectors_written += int(fields[9])
    return sectors_read * 512, sectors_written * 512


class HostMetricsSampler:
    def __init__(
        self,
        proc_root: Path = Path("/host/proc"),
        data_paths: tuple[Path, ...] = (),
    ) -> None:
        self.proc_root = Path(proc_root)
        self.data_paths = tuple(Path(path) for path in data_paths)
        self._previous: dict[str, float] | None = None

    def sample(self, monotonic_time: float) -> dict[str, object]:
        total, idle = _read_cpu(self.proc_root / "stat")
        memory = _read_key_values(self.proc_root / "meminfo")
        load_fields = (self.proc_root / "loadavg").read_text(encoding="utf-8").split()
        received, sent = _read_network(self.proc_root / "net" / "dev")
        disk_read, disk_written = _read_disk_io(self.proc_root / "diskstats")
        current = {
            "time": monotonic_time,
            "cpu_total": float(total),
            "cpu_idle": float(idle),
            "net_rx": float(received),
            "net_tx": float(sent),
            "disk_read": float(disk_read),
            "disk_written": float(disk_written),
        }
        result: dict[str, object] = {
            "ts": utc_now(),
            "load_1m": float(load_fields[0]),
            "load_5m": float(load_fields[1]),
            "load_15m": float(load_fields[2]),
            "memory_total_mib": memory.get("MemTotal", 0) / 1024,
            "memory_available_mib": memory.get("MemAvailable", 0) / 1024,
            "swap_total_mib": memory.get("SwapTotal", 0) / 1024,
            "swap_free_mib": memory.get("SwapFree", 0) / 1024,
        }
        if self._previous is not None:
            elapsed = current["time"] - self._previous["time"]
            cpu_delta = current["cpu_total"] - self._previous["cpu_total"]
            idle_delta = current["cpu_idle"] - self._previous["cpu_idle"]
            if elapsed > 0 and cpu_delta > 0 and 0 <= idle_delta <= cpu_delta:
                result["cpu_util_pct"] = 100 * (cpu_delta - idle_delta) / cpu_delta
                result["net_rx_bytes_per_s"] = max(
                    0.0, (current["net_rx"] - self._previous["net_rx"]) / elapsed
                )
                result["net_tx_bytes_per_s"] = max(
                    0.0, (current["net_tx"] - self._previous["net_tx"]) / elapsed
                )
                result["disk_read_bytes_per_s"] = max(
                    0.0,
                    (current["disk_read"] - self._previous["disk_read"]) / elapsed,
                )
                result["disk_write_bytes_per_s"] = max(
                    0.0,
                    (current["disk_written"] - self._previous["disk_written"])
                    / elapsed,
                )
        self._previous = current

        disks = {}
        for path in self.data_paths:
            try:
                usage = shutil.disk_usage(path)
            except OSError:
                continue
            disks[str(path)] = {
                "total_mib": usage.total / (1024**2),
                "used_mib": usage.used / (1024**2),
                "free_mib": usage.free / (1024**2),
            }
        result["disks"] = disks
        return result

