"""Admission and drain coordination shared by HTTP, live sessions and workers."""

from __future__ import annotations

import asyncio
import threading
import time


class RuntimeControl:
    def __init__(self):
        self._lock = threading.Lock()
        self.active = 0
        self.draining = False
        self.available = True
        self.state = "idle"
        self.error = None

    def enter(self, *, admitted=False):
        with self._lock:
            if not admitted and (self.draining or not self.available):
                return False
            self.active += 1
            return True

    def leave(self):
        with self._lock:
            if self.active <= 0:
                raise RuntimeError("Unbalanced runtime admission")
            self.active -= 1

    def snapshot(self):
        with self._lock:
            return {"state": self.state, "active_operations": self.active,
                    "draining": self.draining, "available": self.available,
                    "error": self.error}

    async def reload(self, operation, *, timeout=300):
        with self._lock:
            if self.draining:
                raise RuntimeError("A model reload is already running")
            self.draining = True
            self.state = "draining"
            self.error = None
        deadline = time.monotonic() + timeout
        try:
            while self.snapshot()["active_operations"]:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Drain timed out; existing sessions were not interrupted")
                await asyncio.sleep(0.05)
            self.state = "reloading"
            await operation()
            self.state = "idle"
        except Exception as exc:
            self.state = "failed"
            self.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            with self._lock:
                self.draining = False


class AdmissionMiddleware:
    def __init__(self, app, control):
        self.app = app
        self.control = control

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        guarded = path in {"/process", "/api/quality", "/ws/live_stream"} or path.startswith("/stream/")
        if scope["type"] not in {"http", "websocket"} or not guarded:
            return await self.app(scope, receive, send)
        if not self.control.enter():
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1013,
                            "reason": "Model reload in progress; retry shortly"})
            else:
                await send({"type": "http.response.start", "status": 503,
                            "headers": [(b"content-type", b"application/json"),
                                        (b"retry-after", b"5")]})
                await send({"type": "http.response.body",
                            "body": b'{"detail":"ASR draining/reloading or unavailable; retry shortly"}'})
            return
        try:
            await self.app(scope, receive, send)
        finally:
            self.control.leave()
