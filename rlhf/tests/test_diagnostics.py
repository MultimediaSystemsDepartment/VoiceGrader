from __future__ import annotations

import csv
import json
import uuid
import zipfile
from pathlib import Path

from diagnostics.artifacts import build_bundle, list_runs
from diagnostics.collector import DiagnosticsCollector
from diagnostics.host_metrics import HostMetricsSampler
from paths import PathsConfig
from storage.feedback_store import FeedbackStore
from utils.structured_events import StructuredEventWriter


def _proc_fixture(root: Path, *, idle: int = 40, received: int = 1000) -> None:
    (root / "net").mkdir(parents=True, exist_ok=True)
    (root / "stat").write_text(
        f"cpu  10 20 30 {idle} 5 0 0 0 0 0\n", encoding="utf-8"
    )
    (root / "meminfo").write_text(
        "MemTotal: 2048000 kB\nMemAvailable: 1024000 kB\n"
        "SwapTotal: 512000 kB\nSwapFree: 256000 kB\n",
        encoding="utf-8",
    )
    (root / "loadavg").write_text("1.00 0.50 0.25 1/100 42\n", encoding="utf-8")
    (root / "diskstats").write_text(
        "8 0 sda 1 0 100 0 1 0 200 0 0 0 0 0 0 0\n",
        encoding="utf-8",
    )
    (root / "net" / "dev").write_text(
        "Inter-| Receive | Transmit\n face |bytes |bytes\n"
        f"  eth0: {received} 0 0 0 0 0 0 0 2000 0 0 0 0 0 0 0\n",
        encoding="utf-8",
    )


def test_host_sampler_calculates_rates_from_proc_snapshots(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    _proc_fixture(proc)
    sampler = HostMetricsSampler(proc)

    first = sampler.sample(10.0)
    _proc_fixture(proc, idle=50, received=3000)
    second = sampler.sample(12.0)

    assert first["memory_total_mib"] == 2000
    assert second["cpu_util_pct"] == 0
    assert second["net_rx_bytes_per_s"] == 1000
    assert second["disk_read_bytes_per_s"] == 0
    assert second["load_15m"] == 0.25


def test_collector_start_sample_stop_is_atomic(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "outputs"
    proc = tmp_path / "proc"
    output.mkdir()
    _proc_fixture(proc)
    paths = PathsConfig(output_dir=str(output), db_path=str(output / "feedback.sqlite"))
    store = FeedbackStore(paths.db_path)

    class FakeGPU:
        summary = {}

        def start(self):
            return self

        def stop(self):
            return {"available": False, "samples": 0}

    monkeypatch.setattr("diagnostics.collector.GPUMonitor", lambda *args, **kwargs: FakeGPU())
    collector = DiagnosticsCollector(store, paths, interval_s=1, proc_root=proc)
    run_id = str(uuid.uuid4())

    collector.start(run_id, "smoke")
    collector.tick()
    collector.stop()

    run_root = output / "diagnostics" / "runs" / run_id
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((run_root / "host_metrics.csv").open(encoding="utf-8")))
    assert manifest["state"] == "completed"
    assert manifest["samples"] == 1
    assert manifest["telemetry_scope"] == "runtime"
    assert manifest["telemetry_sources"] == [
        "procfs",
        "gpu",
        "application",
        "http",
    ]
    assert len(rows) == 1
    assert not (run_root / "container_metrics.jsonl").exists()
    assert not (run_root / "container_events.jsonl").exists()
    assert not (run_root / "manifest.json.tmp").exists()
    with zipfile.ZipFile(
        run_root / f"diagnostics-{run_id}-sanitized.zip"
    ) as archive:
        assert json.loads(archive.read("bundle.json"))["partial"] is False


def test_collector_reports_malformed_admin_request(tmp_path: Path) -> None:
    output = tmp_path / "outputs"
    output.mkdir()
    paths = PathsConfig(output_dir=str(output), db_path=str(output / "feedback.sqlite"))
    store = FeedbackStore(paths.db_path)
    collector = DiagnosticsCollector(store, paths, proc_root=tmp_path / "missing")
    store.set_setting("diagnostics_request", "not-json")

    collector.tick()

    status = json.loads(store.get_setting("diagnostics_status"))
    assert status["state"] == "idle"
    assert status["error"]
    assert collector.run_id is None


def test_collector_acknowledges_start_request_and_creates_partial_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "outputs"
    proc = tmp_path / "proc"
    output.mkdir()
    _proc_fixture(proc)
    paths = PathsConfig(
        output_dir=str(output),
        logs_dir=str(output / "logs"),
        db_path=str(output / "feedback.sqlite"),
    )
    store = FeedbackStore(paths.db_path)
    request_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    store.set_setting(
        "diagnostics_request",
        json.dumps(
            {
                "request_id": request_id,
                "action": "start",
                "run_id": run_id,
                "name": "ack-test",
            }
        ),
    )

    class FakeGPU:
        def start(self):
            return self

        def stop(self):
            return {"available": False, "samples": 0}

    monkeypatch.setattr("diagnostics.collector.GPUMonitor", lambda *args, **kwargs: FakeGPU())
    collector = DiagnosticsCollector(store, paths, interval_s=1, proc_root=proc)
    monkeypatch.setattr(
        collector,
        "_probe_web",
        lambda: {"reachable": True, "status": 200, "latency_ms": 1.0},
    )

    collector.tick()

    status = json.loads(store.get_setting("diagnostics_status"))
    assert status["state"] == "running"
    assert status["last_request_id"] == request_id
    bundle = output / "diagnostics" / "runs" / run_id / f"diagnostics-{run_id}-sanitized.zip"
    with zipfile.ZipFile(bundle) as archive:
        assert json.loads(archive.read("bundle.json"))["partial"] is True
    collector.stop()


def test_recovery_resumes_latest_run_and_marks_older_run_interrupted(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "outputs"
    runs_root = output / "diagnostics" / "runs"
    older_id = str(uuid.uuid4())
    latest_id = str(uuid.uuid4())
    for run_id, started_at in (
        (older_id, "2026-08-03T10:00:00Z"),
        (latest_id, "2026-08-03T11:00:00Z"),
    ):
        root = runs_root / run_id
        root.mkdir(parents=True)
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "state": "running",
                    "started_at": started_at,
                    "samples": 2,
                }
            ),
            encoding="utf-8",
        )

    class FakeGPU:
        def start(self):
            return self

        def stop(self):
            return {"available": False, "samples": 0}

    monkeypatch.setattr("diagnostics.collector.GPUMonitor", lambda *args, **kwargs: FakeGPU())
    paths = PathsConfig(
        output_dir=str(output),
        logs_dir=str(output / "logs"),
        db_path=str(output / "feedback.sqlite"),
    )

    collector = DiagnosticsCollector(FeedbackStore(paths.db_path), paths)

    assert collector.run_id == latest_id
    older = json.loads(
        (runs_root / older_id / "manifest.json").read_text(encoding="utf-8")
    )
    assert older["state"] == "interrupted"
    assert older["interruption_reason"] == "superseded during collector recovery"
    collector.stop()


def test_structured_event_writer_allows_only_safe_fields(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    writer = StructuredEventWriter(path)
    writer.emit(
        "http_request",
        route="/api/submit",
        method="POST",
        status=200,
        cookie="secret",
        correction_text="private transcript",
    )
    writer.close()

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["route"] == "/api/submit"
    assert "cookie" not in record
    assert "correction_text" not in record


def test_issue_events_are_deduplicated_and_resolved(tmp_path: Path) -> None:
    output = tmp_path / "outputs"
    run_id = str(uuid.uuid4())
    run_root = output / "diagnostics" / "runs" / run_id
    run_root.mkdir(parents=True)
    paths = PathsConfig(output_dir=str(output), db_path=str(output / "feedback.sqlite"))
    collector = DiagnosticsCollector(FeedbackStore(paths.db_path), paths)
    collector.run_id = run_id
    collector.run_root = run_root

    collector._set_issue("low_disk", True, "warning", "free=2%")
    collector._set_issue("low_disk", True, "warning", "free=1%")
    collector._set_issue("low_disk", False, "warning", "recovered")

    events = [
        json.loads(line)
        for line in (run_root / "issues.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == ["issue_opened", "issue_resolved"]


def test_bundle_kinds_keep_sensitive_artifacts_separate(tmp_path: Path) -> None:
    output = tmp_path / "outputs"
    logs = output / "logs"
    run_id = str(uuid.uuid4())
    run_root = output / "diagnostics" / "runs" / run_id
    logs.mkdir(parents=True)
    run_root.mkdir(parents=True)
    (run_root / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "state": "completed"}), encoding="utf-8"
    )
    (run_root / "host_metrics.csv").write_text("ts,cpu\nnow,10\n", encoding="utf-8")
    (logs / "2026-08-02.log").write_text("raw private detail\n", encoding="utf-8")
    (logs / "web_events.jsonl").write_text(
        json.dumps({"event": "http_request", "run_id": run_id, "status": 200}) + "\n",
        encoding="utf-8",
    )
    paths = PathsConfig(
        output_dir=str(output),
        logs_dir=str(logs),
        db_path=str(output / "feedback.sqlite"),
    )
    store = FeedbackStore(paths.db_path)
    store.set_setting("private_test_value", "transcript-like private value")

    sanitized = build_bundle(store, paths, run_id, "sanitized")
    sensitive = build_bundle(store, paths, run_id, "sensitive")

    with zipfile.ZipFile(sanitized) as archive:
        names = set(archive.namelist())
        assert "web_events.jsonl" in names
        assert not any(name.startswith("sensitive/") for name in names)
        assert b"private_test_value" not in b"".join(
            archive.read(name) for name in names
        )
    with zipfile.ZipFile(sensitive) as archive:
        names = set(archive.namelist())
        assert "sensitive/feedback.sqlite" in names
        assert "sensitive/logs/2026-08-02.log" in names


def test_running_bundle_is_partial_and_drops_torn_metric_line(tmp_path: Path) -> None:
    output = tmp_path / "outputs"
    run_id = str(uuid.uuid4())
    run_root = output / "diagnostics" / "runs" / run_id
    run_root.mkdir(parents=True)
    (run_root / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "state": "running", "samples": 3}),
        encoding="utf-8",
    )
    (run_root / "host_metrics.csv").write_bytes(b"ts,cpu\nnow,10\ntorn,9")
    (run_root / "platform_metrics.jsonl").write_bytes(
        b'{"training_status":"idle"}\n{"training_status":"torn"'
    )
    paths = PathsConfig(
        output_dir=str(output),
        logs_dir=str(output / "logs"),
        db_path=str(output / "feedback.sqlite"),
    )

    bundle = build_bundle(FeedbackStore(paths.db_path), paths, run_id, "sanitized")

    with zipfile.ZipFile(bundle) as archive:
        metadata = json.loads(archive.read("bundle.json"))
        assert metadata["partial"] is True
        assert metadata["run_state"] == "running"
        assert metadata["samples"] == 3
        assert archive.read("host_metrics.csv") == b"ts,cpu\nnow,10\n"
        assert archive.read("platform_metrics.jsonl") == (
            b'{"training_status":"idle"}\n'
        )
    listed = list_runs(paths)[0]["bundles"]["sanitized"]
    assert listed["partial"] is True
    assert listed["samples"] == 3


def test_legacy_bundle_uses_completed_manifest_metadata(tmp_path: Path) -> None:
    output = tmp_path / "outputs"
    run_id = str(uuid.uuid4())
    run_root = output / "diagnostics" / "runs" / run_id
    run_root.mkdir(parents=True)
    (run_root / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "state": "completed",
                "samples": 7,
                "completed_at": "2026-08-03T12:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    paths = PathsConfig(output_dir=str(output))
    legacy = run_root / f"diagnostics-{run_id}-sanitized.zip"
    with zipfile.ZipFile(legacy, "w") as archive:
        archive.writestr("bundle.json", json.dumps({"schema_version": 1}))

    metadata = list_runs(paths)[0]["bundles"]["sanitized"]

    assert metadata["partial"] is False
    assert metadata["run_state"] == "completed"
    assert metadata["samples"] == 7
    assert metadata["captured_at"] == "2026-08-03T12:00:00Z"


def test_k6_scripts_attach_diagnostics_run_header() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("sim_burst.js", "loadtest.js"):
        source = (root / "stress_tests" / "load" / name).read_text(encoding="utf-8")
        assert '"X-Diagnostics-Run-ID": DIAGNOSTICS_RUN_ID' in source
        assert source.count("requestParams(") >= 8
        declaration = source.index("const DIAGNOSTICS_RUN_ID")
        assert declaration > source.index("*/")
        assert declaration < source.index("export const options")


def test_endurance_burst_requires_completed_annotations() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "stress_tests" / "load" / "sim_burst.js").read_text(
        encoding="utf-8"
    )

    assert 'gracefulStop: "60s"' in source
    assert 'capacity_errors:  ["count<1"]' in source
    assert 'pair_queue_empty: ["count<1"]' in source
    assert 'successful_submissions: ["count>0"]' in source
    assert 'audio_duration:   ["p(95)<8000"]' in source
    assert "audioDuration.add(Date.now() - audioStarted)" in source
    assert "successfulSubmissions.add(1)" in source
    assert 'typeof body === "string"' in source
    assert 'requestParams("setup", { redirects: 0 })' in source
    assert 'jar.set(baseUrl, "session_id"' in source
    assert 'session_errors:   ["count<1"]' in source
    assert "noCookiesReset: true" in source
    assert "Audio failed pair=${pairData.pair_id}" in source
    assert "status: String(audioRes.status)" in source
