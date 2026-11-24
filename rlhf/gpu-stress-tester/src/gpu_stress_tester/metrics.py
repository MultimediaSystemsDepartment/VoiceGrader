from __future__ import annotations

import subprocess


def gpu_snapshot() -> dict[str, float | str] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
        fields = [field.strip() for field in result.stdout.splitlines()[0].split(",")]
        return {
            "name": fields[0],
            "utilization_percent": float(fields[1]),
            "memory_used_mib": float(fields[2]),
            "memory_total_mib": float(fields[3]),
            "temperature_c": float(fields[4]),
            "power_w": float(fields[5]),
        }
    except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError):
        return None
