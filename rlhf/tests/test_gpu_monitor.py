from pathlib import Path
from types import SimpleNamespace

from stress_tests.load import gpu_monitor


def test_poll_smi_preserves_unavailable_fields(monkeypatch) -> None:
    result = SimpleNamespace(
        returncode=0,
        stdout="[N/A], [N/A], [N/A], [N/A], [N/A], 30, 118.18\n",
    )
    monkeypatch.setattr(gpu_monitor.subprocess, "run", lambda *args, **kwargs: result)

    sample = gpu_monitor._poll_smi(0)

    assert sample is not None
    assert sample["gpu_util_pct"] is None
    assert sample["vram_total_mib"] is None
    assert sample["temp_c"] == 30.0
    assert sample["power_w"] == 118.18


def test_summary_reports_partially_unavailable_smi_metrics(tmp_path: Path) -> None:
    monitor = gpu_monitor.GPUMonitor(tmp_path / "gpu.csv")
    monitor._backend = "smi"
    monitor._samples = [
        {
            "gpu_util_pct": None,
            "vram_used_mib": None,
            "vram_total_mib": None,
            "vram_free_mib": None,
            "temp_c": 30.0,
            "power_w": 118.18,
        }
    ]

    summary = monitor._summarize()

    assert "gpu_util_avg" not in summary
    assert "vram_total_mib" not in summary
    assert summary["temp_avg_c"] == 30.0
    assert summary["power_avg_w"] == 118.18
    assert summary["unavailable_metrics"] == [
        "gpu_util_pct",
        "vram_used_mib",
        "vram_total_mib",
        "vram_free_mib",
    ]
    assert "unavailable=gpu_util_pct" in monitor.format_summary(summary)