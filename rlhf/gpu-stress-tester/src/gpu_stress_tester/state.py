from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def read(self) -> dict[str, Any]:
        with self._lock:
            if not self.path.exists():
                return {"status": "idle", "updated_at": utc_now()}
            return json.loads(self.path.read_text(encoding="utf-8"))

    def replace(self, value: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            payload = {**value, "updated_at": utc_now()}
            self._write(payload)
            return payload

    def update(self, **changes: Any) -> dict[str, Any]:
        with self._lock:
            current = (
                json.loads(self.path.read_text(encoding="utf-8"))
                if self.path.exists()
                else {"status": "idle"}
            )
            current.update(changes)
            current["updated_at"] = utc_now()
            self._write(current)
            return current

    def _write(self, value: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        temporary.replace(self.path)


class StopRequested(RuntimeError):
    pass


def check_stop(stop_file: Path) -> None:
    if stop_file.exists():
        raise StopRequested("Stop requested by operator")
