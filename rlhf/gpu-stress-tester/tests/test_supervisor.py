from pathlib import Path

import pytest

from gpu_stress_tester.state import StateStore
from gpu_stress_tester.supervisor import RunManager, Settings


def test_start_requires_admedvoice_archive(tmp_path: Path) -> None:
    manager = RunManager(Settings(tmp_path / "missing.zip", tmp_path / "outputs"))

    with pytest.raises(FileNotFoundError, match="ADMEDVOICE archive"):
        manager.start("smoke", {})


def test_start_rejects_empty_archive(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    archive.touch()
    manager = RunManager(Settings(archive, tmp_path / "outputs"))

    with pytest.raises(FileNotFoundError, match="archive is empty"):
        manager.start("smoke", {})


def test_log_tail_is_bounded_to_current_run(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    run_dir = output_dir / "runs" / "run-id"
    run_dir.mkdir(parents=True)
    (run_dir / "run.log").write_text("one\ntwo\nthree\n", encoding="utf-8")
    manager = RunManager(Settings(tmp_path / "archive.zip", output_dir))
    StateStore(output_dir / "current.json").replace(
        {"status": "completed", "run_dir": str(run_dir)}
    )

    assert manager.log_tail(2) == ["two\n", "three\n"]


def test_restart_preserves_terminal_worker_state(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    run_dir = output_dir / "runs" / "run-id"
    StateStore(run_dir / "state.json").replace(
        {"status": "completed", "run_dir": str(run_dir), "stage": "complete"}
    )
    StateStore(output_dir / "current.json").replace(
        {"status": "running", "run_dir": str(run_dir)}
    )

    manager = RunManager(Settings(tmp_path / "archive.zip", output_dir))

    assert manager.status()["status"] == "completed"