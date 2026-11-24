from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import resolve_config
from .metrics import gpu_snapshot
from .state import StateStore, utc_now


@dataclass(frozen=True)
class Settings:
    archive: Path
    output_dir: Path

    @classmethod
    def from_environment(cls) -> Settings:
        return cls(
            archive=Path(os.environ.get("ADMEDVOICE_ARCHIVE", "/data/archive.zip")),
            output_dir=Path(os.environ.get("STRESS_OUTPUT_DIR", "/outputs")),
        )


class RunManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        self.pointer = StateStore(self.settings.output_dir / "current.json")
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        current = self.pointer.read()
        if current.get("status") in {"starting", "running", "stopping"}:
            run_dir = Path(current["run_dir"])
            run_state = StateStore(run_dir / "state.json")
            persisted = run_state.read()
            if persisted.get("status") in {"completed", "failed", "stopped", "interrupted"}:
                self.pointer.replace({**current, **persisted})
            else:
                changes = {
                    "status": "interrupted",
                    "stage": "service-restarted",
                    "completed_at": utc_now(),
                }
                self.pointer.update(**changes)
                run_state.update(**changes)

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_process()
            current = self.pointer.read()
            run_dir_value = current.get("run_dir")
            if run_dir_value:
                run_state = StateStore(Path(run_dir_value) / "state.json").read()
                current.update(run_state)
            dataset_ready, dataset_error = self._dataset_status()
            current["dataset_ready"] = dataset_ready
            current["dataset_error"] = dataset_error
            current["archive"] = str(self.settings.archive)
            if current.get("gpu") is None:
                current["gpu"] = gpu_snapshot()
            return current

    def start(self, profile: str, overrides: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._refresh_process()
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("A training run is already active")
            dataset_ready, dataset_error = self._dataset_status()
            if not dataset_ready:
                raise FileNotFoundError(dataset_error)
            config = resolve_config(profile, **overrides)
            run_id = str(uuid.uuid4())
            run_dir = self.settings.output_dir / "runs" / run_id
            run_dir.mkdir(parents=True)
            config_path = run_dir / "config.json"
            config.to_json(config_path)
            state = {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "status": "starting",
                "stage": "startup",
                "started_at": utc_now(),
                "profile": profile,
            }
            self.pointer.replace(state)
            StateStore(run_dir / "state.json").replace(state)
            log_handle = (run_dir / "run.log").open("a", encoding="utf-8", buffering=1)
            command = [
                sys.executable,
                "-m",
                "gpu_stress_tester.runner",
                "--config",
                str(config_path),
                "--archive",
                str(self.settings.archive),
                "--run-dir",
                str(run_dir),
            ]
            try:
                self._process = subprocess.Popen(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            finally:
                log_handle.close()
            return self.pointer.update(pid=self._process.pid)

    def _dataset_status(self) -> tuple[bool, str | None]:
        if not self.settings.archive.is_file():
            return False, f"Start unavailable: ADMEDVOICE archive not found at {self.settings.archive}."
        if self.settings.archive.stat().st_size == 0:
            return False, f"Start unavailable: ADMEDVOICE archive is empty at {self.settings.archive}."
        return True, None

    def request_stop(self) -> dict[str, Any]:
        with self._lock:
            current = self.pointer.read()
            if not self._is_running():
                raise RuntimeError("No training run is active")
            Path(current["run_dir"], "stop.requested").touch()
            StateStore(Path(current["run_dir"]) / "state.json").update(status="stopping")
            return self.pointer.update(status="stopping")

    def force_stop(self) -> dict[str, Any]:
        with self._lock:
            if not self._is_running() or self._process is None:
                raise RuntimeError("No training run is active")
            if os.name == "nt":
                self._process.terminate()
            else:
                os.killpg(self._process.pid, signal.SIGTERM)
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    self._process.kill()
                else:
                    os.killpg(self._process.pid, signal.SIGKILL)
                self._process.wait(timeout=5)
            self._process = None
            changes = {
                "status": "interrupted",
                "stage": "force-stopped",
                "completed_at": utc_now(),
            }
            current = self.pointer.update(**changes)
            StateStore(Path(current["run_dir"]) / "state.json").update(**changes)
            return current

    def log_tail(self, lines: int = 200) -> list[str]:
        current = self.pointer.read()
        if not current.get("run_dir"):
            return []
        path = Path(current["run_dir"]) / "run.log"
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.readlines()[-lines:]

    def _is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _refresh_process(self) -> None:
        if self._process is None or self._process.poll() is None:
            return
        return_code = self._process.returncode
        self._process = None
        current = self.pointer.read()
        if current.get("status") in {"starting", "running", "stopping"}:
            self.pointer.update(
                status="failed" if return_code else "completed",
                error=(f"Worker exited with code {return_code}" if return_code else None),
                completed_at=utc_now(),
            )
