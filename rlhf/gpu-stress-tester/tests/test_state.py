from pathlib import Path

import pytest

from gpu_stress_tester.state import StateStore, StopRequested, check_stop


def test_state_updates_atomically(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.json")

    store.replace({"status": "running", "round": 1})
    state = store.update(stage="sft")

    assert state["status"] == "running"
    assert state["stage"] == "sft"
    assert store.read() == state


def test_stop_marker_interrupts_worker(tmp_path: Path) -> None:
    marker = tmp_path / "stop"
    marker.touch()

    with pytest.raises(StopRequested):
        check_stop(marker)