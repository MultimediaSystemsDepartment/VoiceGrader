import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from stress_analytics.repository import ArtifactRepository


def _write_artifact(path: Path) -> None:
    manifest = {
        "run_id": "11111111-1111-4111-8111-111111111111",
        "state": "completed",
        "started_at": "2026-08-18T10:00:00Z",
        "completed_at": "2026-08-18T10:00:10Z",
        "last_sample_at": "2026-08-18T10:00:10Z",
        "sample_interval_s": 5,
        "samples": 3,
        "platform": "test-platform",
        "python": "3.12",
        "git_commit": "abc123",
    }
    host_csv = "\n".join(
        [
            "ts,cpu_util_pct,load_1m,load_5m,load_15m,memory_total_mib,memory_available_mib,swap_total_mib,swap_free_mib,net_rx_bytes_per_s,net_tx_bytes_per_s,disk_read_bytes_per_s,disk_write_bytes_per_s,disks_json",
            '2026-08-18T10:00:00Z,10,1,1,1,1000,800,0,0,100,50,10,20,"{""/app/_outputs"": {""free_mib"": 900, ""total_mib"": 1000}}"',
            '2026-08-18T10:00:05Z,30,2,2,2,1000,700,0,0,200,60,30,40,"{""/app/_outputs"": {""free_mib"": 800, ""total_mib"": 1000}}"',
            '2026-08-18T10:00:10Z,20,1,1,1,1000,750,0,0,150,40,20,30,"{""/app/_outputs"": {""free_mib"": 850, ""total_mib"": 1000}}"',
        ]
    )
    platform = [
        {
            "ts": f"2026-08-18T10:00:{second:02d}Z",
            "stats": {
                "pending_pairs": 10 - index,
                "annotated_pairs": index,
                "actionable_annotations": index,
                "human_annotations": index,
                "skipped_pairs": 0,
                "training_runs": 1,
            },
            "web": {"reachable": True, "latency_ms": 2 + index},
            "training_status": "idle",
            "queue_refill_status": "idle",
            "promotion_gate_status": "idle",
        }
        for index, second in enumerate((0, 5, 10))
    ]
    web_events = [
        {"ts": "2026-08-18T10:00:01Z", "event": "http_request", "route": "/api/pair", "method": "GET", "status": 200, "duration_ms": 10},
        {"ts": "2026-08-18T10:00:06Z", "event": "http_request", "route": "/api/pair", "method": "GET", "status": 500, "duration_ms": 100},
        {"ts": "2026-08-18T10:00:09Z", "event": "http_request", "route": "/api/stats", "method": "GET", "status": 200, "duration_ms": 5},
    ]
    issues = [
        {"ts": "2026-08-18T10:00:06Z", "event": "issue_opened", "issue": "web_high_latency", "severity": "warning", "detail": "latency=100ms"},
        {"ts": "2026-08-18T10:00:09Z", "event": "issue_resolved", "issue": "web_high_latency", "severity": "warning", "detail": "latency recovered"},
    ]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("bundle.json", json.dumps({"run_id": manifest["run_id"], "run_state": "completed", "partial": False}))
        archive.writestr("host_metrics.csv", host_csv + "\n")
        archive.writestr("platform_metrics.jsonl", "\n".join(map(json.dumps, platform)) + "\n")
        archive.writestr("web_events.jsonl", "\n".join(map(json.dumps, web_events)) + "\n")
        archive.writestr("issues.jsonl", "\n".join(map(json.dumps, issues)) + "\n")
        archive.writestr("gpu_metrics.csv", "ts,gpu_util_pct\n")


def test_repository_indexes_artifact_without_mutating_input(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    artifact = data_dir / "diagnostics.zip"
    _write_artifact(artifact)
    original_digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    repository = ArtifactRepository(data_dir, tmp_path / "cache")

    assert repository.refresh() == {"indexed": 1, "reused": 0, "failed": 0}
    assert repository.refresh() == {"indexed": 0, "reused": 1, "failed": 0}
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == original_digest

    run = repository.list_runs()[0]
    overview = repository.overview(run["id"])
    timeline = repository.timeline(run["id"], points=100)

    assert run["host_rows"] == 3
    assert run["platform_rows"] == 3
    assert run["web_rows"] == 3
    assert overview["http"]["requests"] == 3
    assert overview["http"]["errors"] == 1
    assert overview["host"]["cpu_max"] == 30
    assert len(timeline["host"]) == 3
    assert repository.route_summary(run["id"])[0]["route"] == "/api/pair"
    assert len(repository.issue_events(run["id"])) == 2


def test_repository_rejects_unknown_artifact(tmp_path: Path) -> None:
    repository = ArtifactRepository(tmp_path / "data", tmp_path / "cache")

    with pytest.raises(KeyError):
        repository.overview(999)
    with pytest.raises(KeyError):
        repository.timeline(999)
    with pytest.raises(KeyError):
        repository.route_summary(999)
    with pytest.raises(KeyError):
        repository.issue_events(999)


def test_dashboard_contract_contains_local_charts_and_filters() -> None:
    root = Path(__file__).parents[1]
    template = (root / "stress_analytics" / "templates" / "index.html").read_text(encoding="utf-8")
    script = (root / "stress_analytics" / "static" / "dashboard.js").read_text(encoding="utf-8")

    assert "vendor/echarts.min.js" in template
    assert 'id="rangePresets"' in template
    assert 'id="routeFilter"' in template
    assert 'data-chart="hostChart"' in template
    assert 'class="measure-info"' in template
    assert 'data-tooltip="95th-percentile request duration' in template
    assert 'renderer: "canvas"' in script
    assert 'axisDefaults("Utilization (%)")' in script
    assert 'name: "Memory used (%)"' in script
    assert 'name: "Average latency (ms)"' in script
    assert "escapeHTML" in script
    assert "getDataURL" in script