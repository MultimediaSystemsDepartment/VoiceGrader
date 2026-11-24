from __future__ import annotations

import json
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ALLOWED_FIELDS = {
    "severity",
    "event",
    "route",
    "method",
    "status",
    "duration_ms",
    "run_id",
    "error_type",
    "detail",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class StructuredEventWriter:
    def __init__(self, path: Path, capacity: int = 10_000) -> None:
        self.path = Path(path)
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(capacity)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.dropped = 0
        self._thread.start()

    def emit(self, event: str, **fields: object) -> None:
        record = {"ts": _utc_now(), "event": str(event)[:80]}
        record.update(
            {
                key: value
                for key, value in fields.items()
                if key in _ALLOWED_FIELDS and value is not None
            }
        )
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1

    def close(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            self._queue.get_nowait()
            self._queue.put_nowait(None)
            self.dropped += 1
        self._thread.join(timeout=5)

    def _run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            while True:
                record = self._queue.get()
                if record is None:
                    break
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()

