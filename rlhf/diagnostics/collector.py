from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from paths import PathsConfig
from storage.feedback_store import FeedbackStore
from stress_tests.load.gpu_monitor import GPUMonitor

from .artifacts import build_bundle, delete_run
from .host_metrics import HostMetricsSampler, utc_now

_REQUEST_KEY = "diagnostics_request"
_STATUS_KEY = "diagnostics_status"


class DiagnosticsCollector:
    HOST_COLUMNS = (
        "ts",
        "cpu_util_pct",
        "load_1m",
        "load_5m",
        "load_15m",
        "memory_total_mib",
        "memory_available_mib",
        "swap_total_mib",
        "swap_free_mib",
        "net_rx_bytes_per_s",
        "net_tx_bytes_per_s",
        "disk_read_bytes_per_s",
        "disk_write_bytes_per_s",
        "disks_json",
    )

    def __init__(
        self,
        store: FeedbackStore,
        paths: PathsConfig,
        interval_s: float = 5.0,
        proc_root: Path = Path("/proc"),
        clock=time.monotonic,
    ) -> None:
        self.store = store
        self.paths = paths
        self.interval_s = interval_s
        self.clock = clock
        self.root = Path(paths.output_dir) / "diagnostics" / "runs"
        self.sampler = HostMetricsSampler(
            proc_root,
            (Path(paths.output_dir), Path(paths.admedvoice_audio_dir or paths.output_dir)),
        )
        self.run_id: str | None = None
        self.run_root: Path | None = None
        self.manifest: dict = {}
        self._host_handle = None
        self._host_writer = None
        self._platform_handle = None
        self._gpu_monitor: GPUMonitor | None = None
        self._last_request_id = self._stored_last_request_id()
        self._last_partial_bundle_at: float | None = None
        self._samples = 0
        self._open_issues: set[str] = set()
        self._last_error = ""
        self._recover_active_run()

    def serve_forever(self) -> None:
        while True:
            self.tick()
            time.sleep(self.interval_s)

    def tick(self) -> None:
        self._handle_request()
        if self.run_id is not None:
            self._sample()
        self._publish_status()

    def start(self, run_id: str, name: str = "") -> None:
        if self.run_id is not None:
            raise RuntimeError("A diagnostics capture is already active")
        parsed = uuid.UUID(run_id)
        if parsed.version != 4 or str(parsed) != run_id:
            raise ValueError("Diagnostics run ID must be a canonical UUID4")
        self.root.mkdir(parents=True, exist_ok=True)
        run_root = self.root / run_id
        if run_root.exists():
            raise FileExistsError(f"Diagnostics run already exists: {run_id}")
        run_root.mkdir()
        self.run_id = run_id
        self.run_root = run_root
        self._samples = 0
        self.manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "name": name[:120],
            "state": "running",
            "started_at": utc_now(),
            "completed_at": None,
            "sample_interval_s": self.interval_s,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "git_commit": self._git_commit(),
            "collector_pid": os.getpid(),
            "telemetry_scope": "runtime",
            "telemetry_sources": ["procfs", "gpu", "application", "http"],
            "partial_bundle_interval_s": float(
                os.environ.get("DIAGNOSTICS_PARTIAL_BUNDLE_SECONDS", "60")
            ),
            "thresholds": {
                "memory_available_pct": float(
                    os.environ.get("DIAGNOSTICS_MIN_MEMORY_AVAILABLE_PCT", "10")
                ),
                "disk_free_pct": float(
                    os.environ.get("DIAGNOSTICS_MIN_DISK_FREE_PCT", "10")
                ),
                "gpu_temperature_c": float(
                    os.environ.get("DIAGNOSTICS_MAX_GPU_TEMP_C", "85")
                ),
                "gpu_vram_free_pct": float(
                    os.environ.get("DIAGNOSTICS_MIN_VRAM_FREE_PCT", "5")
                ),
                "web_latency_ms": float(
                    os.environ.get("DIAGNOSTICS_MAX_WEB_LATENCY_MS", "2000")
                ),
                "coordinator_stale_s": float(
                    os.environ.get("DIAGNOSTICS_COORDINATOR_STALE_SECONDS", "120")
                ),
            },
        }
        self._write_manifest()
        self._open_run_files(append=False)

    def _open_run_files(self, append: bool) -> None:
        assert self.run_root is not None
        host_path = self.run_root / "host_metrics.csv"
        write_header = not append or not host_path.exists()
        self._host_handle = host_path.open(
            "a" if append else "w", newline="", encoding="utf-8"
        )
        self._host_writer = csv.DictWriter(
            self._host_handle, fieldnames=self.HOST_COLUMNS, extrasaction="ignore"
        )
        if write_header:
            self._host_writer.writeheader()
        self._host_handle.flush()
        self._platform_handle = (self.run_root / "platform_metrics.jsonl").open(
            "a", encoding="utf-8"
        )
        self._gpu_monitor = GPUMonitor(
            self.run_root / "gpu_metrics.csv",
            gpu_index=int(os.environ.get("DIAGNOSTICS_GPU_INDEX", "0")),
            interval_s=max(1, int(self.interval_s)),
            append=append,
        ).start()

    def _recover_active_run(self) -> None:
        if not self.root.exists():
            return
        active = []
        for manifest_path in self.root.glob("*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("state") == "running":
                active.append((manifest.get("started_at", ""), manifest_path, manifest))
        if not active:
            return
        active.sort()
        for _, stale_path, stale_manifest in active[:-1]:
            stale_manifest.update(
                state="interrupted",
                interrupted_at=utc_now(),
                interruption_reason="superseded during collector recovery",
            )
            self._write_manifest_path(stale_path, stale_manifest)
        _, manifest_path, manifest = active[-1]
        self.run_id = str(manifest["run_id"])
        self.run_root = manifest_path.parent
        self.manifest = manifest
        self.manifest.setdefault(
            "partial_bundle_interval_s",
            float(os.environ.get("DIAGNOSTICS_PARTIAL_BUNDLE_SECONDS", "60")),
        )
        self._samples = int(manifest.get("samples", 0))
        self.manifest.setdefault("collector_restarts", []).append(utc_now())
        self.manifest["collector_pid"] = os.getpid()
        self._write_manifest()
        self._open_run_files(append=True)
        self._set_issue(
            "collector_restarted",
            True,
            "warning",
            "collector resumed an active capture after restart",
        )

    def stop(self) -> None:
        if self.run_id is None:
            raise RuntimeError("No diagnostics capture is active")
        if self._gpu_monitor is not None:
            self.manifest["gpu_summary"] = self._gpu_monitor.stop()
        for handle in (
            self._host_handle,
            self._platform_handle,
        ):
            if handle is not None:
                handle.close()
        self.manifest.update(
            state="completed",
            completed_at=utc_now(),
            samples=self._samples,
        )
        self._write_manifest()
        completed_run_id = self.run_id
        try:
            build_bundle(self.store, self.paths, completed_run_id, "sanitized")
        except Exception as exc:
            self._last_error = f"Final sanitized bundle failed: {exc}"
        self.run_id = None
        self.run_root = None
        self._host_handle = None
        self._host_writer = None
        self._platform_handle = None
        self._gpu_monitor = None
        self._last_partial_bundle_at = None

    def _handle_request(self) -> None:
        raw = self.store.get_setting(_REQUEST_KEY, "")
        if not raw:
            return
        request: object = None
        try:
            request = json.loads(raw)
            if not isinstance(request, dict):
                raise ValueError("Diagnostics request must be a JSON object")
            request_id = str(request["request_id"])
            if request_id == self._last_request_id:
                return
            action = request["action"]
            if action == "start":
                self.start(str(request["run_id"]), str(request.get("name", "")))
            elif action == "stop":
                self.stop()
            elif action == "bundle":
                build_bundle(
                    self.store,
                    self.paths,
                    str(request["run_id"]),
                    str(request["kind"]),
                )
            elif action == "delete":
                delete_run(
                    self.paths,
                    str(request["run_id"]),
                    self.run_id,
                )
            else:
                raise ValueError(f"Unsupported diagnostics action: {action}")
            self._last_request_id = request_id
            self._last_error = ""
        except Exception as exc:
            self._last_request_id = (
                str(request.get("request_id", ""))
                if isinstance(request, dict)
                else raw
            )
            self._last_error = str(exc)

    def _sample(self) -> None:
        assert self._host_writer is not None
        assert self._host_handle is not None
        assert self._platform_handle is not None
        try:
            sampled_at = self.clock()
            sample = self.sampler.sample(sampled_at)
            row = {
                **sample,
                "disks_json": json.dumps(sample.get("disks", {}), sort_keys=True),
            }
            self._host_writer.writerow(row)
            self._host_handle.flush()
            platform_sample = {
                "ts": utc_now(),
                "stats": self.store.get_stats(),
                "training_status": self.store.get_setting("training_status", "idle"),
                "training_step": self.store.get_setting("training_step", ""),
                "queue_refill_status": self.store.get_setting("queue_refill_status", "idle"),
                "promotion_gate_status": self.store.get_setting("promotion_gate_status", ""),
                "coordinator_heartbeat_at": self.store.get_setting(
                    "coordinator_heartbeat_at", ""
                ),
                "web": self._probe_web(),
            }
            self._platform_handle.write(json.dumps(platform_sample, sort_keys=True) + "\n")
            self._platform_handle.flush()
            self._detect_issues(sample, platform_sample)
            self._samples += 1
            self.manifest["samples"] = self._samples
            self.manifest["last_sample_at"] = utc_now()
            self._write_manifest()
            self._refresh_partial_bundle(sampled_at)
        except Exception as exc:
            self._set_issue("sample_failed", True, "error", str(exc))

    def _refresh_partial_bundle(self, sampled_at: float) -> None:
        assert self.run_id is not None
        interval = float(self.manifest.get("partial_bundle_interval_s", 60))
        if (
            self._last_partial_bundle_at is not None
            and sampled_at - self._last_partial_bundle_at < interval
        ):
            return
        try:
            build_bundle(self.store, self.paths, self.run_id, "sanitized")
            self._last_partial_bundle_at = sampled_at
            self._set_issue(
                "partial_bundle_failed", False, "error", "partial bundle recovered"
            )
        except Exception as exc:
            self._set_issue("partial_bundle_failed", True, "error", str(exc))

    def _detect_issues(self, host: dict, platform_sample: dict) -> None:
        thresholds = self.manifest["thresholds"]
        memory_total = float(host.get("memory_total_mib", 0))
        memory_available = float(host.get("memory_available_mib", 0))
        memory_pct = 100 * memory_available / memory_total if memory_total else 100
        self._set_issue(
            "low_memory",
            memory_pct < thresholds["memory_available_pct"],
            "warning",
            f"available={memory_pct:.1f}%",
        )
        disks = host.get("disks", {})
        low_disks = []
        if isinstance(disks, dict):
            for path, usage in disks.items():
                total = float(usage.get("total_mib", 0))
                free_pct = 100 * float(usage.get("free_mib", 0)) / total if total else 100
                if free_pct < thresholds["disk_free_pct"]:
                    low_disks.append(f"{path}={free_pct:.1f}%")
        self._set_issue("low_disk", bool(low_disks), "warning", ", ".join(low_disks))
        self._set_issue(
            "training_failed",
            platform_sample["training_status"] == "failed",
            "error",
            "training_status=failed",
        )
        self._set_issue(
            "queue_refill_failed",
            platform_sample["queue_refill_status"] == "failed",
            "error",
            "queue_refill_status=failed",
        )
        web = platform_sample["web"]
        self._set_issue(
            "web_unreachable",
            not web["reachable"],
            "error",
            str(web.get("error", "unreachable")),
        )
        self._set_issue(
            "web_high_latency",
            bool(web["reachable"])
            and float(web["latency_ms"]) >= thresholds["web_latency_ms"],
            "warning",
            f"latency={float(web.get('latency_ms', 0)):.1f}ms",
        )
        heartbeat = platform_sample["coordinator_heartbeat_at"]
        heartbeat_age = None
        if heartbeat:
            try:
                heartbeat_age = (
                    datetime.now(timezone.utc)
                    - datetime.fromisoformat(heartbeat.replace("Z", "+00:00"))
                ).total_seconds()
            except ValueError:
                heartbeat_age = float("inf")
        self._set_issue(
            "coordinator_stale",
            heartbeat_age is None
            or heartbeat_age >= thresholds["coordinator_stale_s"],
            "error",
            "heartbeat missing"
            if heartbeat_age is None
            else f"heartbeat_age={heartbeat_age:.1f}s",
        )
        gpu = (
            getattr(self._gpu_monitor, "latest_sample", None)
            if self._gpu_monitor
            else None
        )
        if gpu:
            total_vram_value = gpu.get("vram_total_mib")
            free_vram_value = gpu.get("vram_free_mib")
            total_vram = float(total_vram_value) if total_vram_value is not None else 0
            free_vram_pct = (
                100 * float(free_vram_value) / total_vram
                if total_vram and free_vram_value is not None
                else 100
            )
            temperature = gpu.get("temp_c")
            self._set_issue(
                "gpu_temperature",
                temperature is not None
                and float(temperature) >= thresholds["gpu_temperature_c"],
                "warning",
                "temperature unavailable"
                if temperature is None
                else f"temperature={float(temperature):.1f}C",
            )
            self._set_issue(
                "gpu_vram_pressure",
                free_vram_pct < thresholds["gpu_vram_free_pct"],
                "warning",
                f"free={free_vram_pct:.1f}%",
            )
        self._set_issue("sample_failed", False, "error", "sampling recovered")
        self._set_issue(
            "collector_restarted",
            False,
            "warning",
            "sampling resumed after collector restart",
        )

    def _probe_web(self) -> dict[str, object]:
        url = os.environ.get("DIAGNOSTICS_WEB_URL", "http://web:8080/api/stats")
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                response.read(1)
                return {
                    "reachable": response.status == 200,
                    "status": response.status,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                }
        except Exception as exc:
            return {
                "reachable": False,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "error": type(exc).__name__,
            }

    def _set_issue(
        self,
        issue: str,
        active: bool,
        severity: str,
        detail: str,
    ) -> None:
        was_active = issue in self._open_issues
        if active == was_active:
            return
        if active:
            self._open_issues.add(issue)
        else:
            self._open_issues.discard(issue)
        if self.run_root is None:
            return
        with (self.run_root / "issues.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "ts": utc_now(),
                        "event": "issue_opened" if active else "issue_resolved",
                        "issue": issue,
                        "severity": severity,
                        "detail": detail[:300],
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    def _publish_status(self) -> None:
        self.store.set_setting(
            _STATUS_KEY,
            json.dumps(
                {
                    "state": "running" if self.run_id else "idle",
                    "run_id": self.run_id,
                    "samples": self._samples,
                    "heartbeat_at": utc_now(),
                    "error": self._last_error,
                    "last_request_id": self._last_request_id,
                    "open_issues": sorted(self._open_issues),
                }
            ),
        )

    def _write_manifest(self) -> None:
        assert self.run_root is not None
        self._write_manifest_path(self.run_root / "manifest.json", self.manifest)

    @staticmethod
    def _write_manifest_path(path: Path, manifest: dict) -> None:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _stored_last_request_id(self) -> str:
        raw = self.store.get_setting(_STATUS_KEY, "")
        try:
            status = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return ""
        return str(status.get("last_request_id", ""))

    @staticmethod
    def _git_commit() -> str:
        try:
            return subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
        except Exception:
            return "unknown"
