from __future__ import annotations

import csv
import io
import json
import math
import sqlite3
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def _epoch(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _iso(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")


def _number(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)
    return ordered[max(0, index)]


class ArtifactRepository:
    def __init__(self, data_dir: Path, cache_dir: Path) -> None:
        self.data_dir = data_dir.resolve()
        self.cache_dir = cache_dir.resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "index.sqlite"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY,
                    path TEXT NOT NULL UNIQUE,
                    file_name TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    run_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    partial INTEGER NOT NULL,
                    started_at REAL,
                    completed_at REAL,
                    last_sample_at REAL,
                    samples INTEGER NOT NULL,
                    sample_interval_s REAL,
                    platform TEXT,
                    python_version TEXT,
                    git_commit TEXT,
                    gpu_rows INTEGER NOT NULL DEFAULT 0,
                    manifest_json TEXT NOT NULL,
                    bundle_json TEXT NOT NULL,
                    indexed_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS host_metrics (
                    artifact_id INTEGER NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
                    ts REAL NOT NULL,
                    cpu_util_pct REAL,
                    load_1m REAL,
                    load_5m REAL,
                    memory_total_mib REAL,
                    memory_available_mib REAL,
                    net_rx_bps REAL,
                    net_tx_bps REAL,
                    disk_read_bps REAL,
                    disk_write_bps REAL,
                    disk_free_mib REAL,
                    disk_total_mib REAL
                );

                CREATE TABLE IF NOT EXISTS platform_metrics (
                    artifact_id INTEGER NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
                    ts REAL NOT NULL,
                    pending_pairs REAL,
                    annotated_pairs REAL,
                    actionable_annotations REAL,
                    human_annotations REAL,
                    skipped_pairs REAL,
                    training_runs REAL,
                    web_reachable INTEGER,
                    web_latency_ms REAL,
                    training_status TEXT,
                    training_step TEXT,
                    queue_refill_status TEXT,
                    promotion_gate_status TEXT
                );

                CREATE TABLE IF NOT EXISTS web_events (
                    artifact_id INTEGER NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
                    ts REAL NOT NULL,
                    route TEXT NOT NULL,
                    method TEXT NOT NULL,
                    status INTEGER,
                    duration_ms REAL,
                    severity TEXT,
                    error_type TEXT
                );

                CREATE TABLE IF NOT EXISTS issues (
                    artifact_id INTEGER NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
                    ts REAL NOT NULL,
                    event TEXT NOT NULL,
                    issue TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    detail TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_host_artifact_ts
                    ON host_metrics(artifact_id, ts);
                CREATE INDEX IF NOT EXISTS idx_platform_artifact_ts
                    ON platform_metrics(artifact_id, ts);
                CREATE INDEX IF NOT EXISTS idx_web_artifact_ts
                    ON web_events(artifact_id, ts);
                CREATE INDEX IF NOT EXISTS idx_web_artifact_route
                    ON web_events(artifact_id, route);
                CREATE INDEX IF NOT EXISTS idx_issues_artifact_ts
                    ON issues(artifact_id, ts);
                """
            )

    def refresh(self) -> dict[str, int]:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        archives = sorted(self.data_dir.glob("*.zip"))
        indexed = 0
        reused = 0
        failed = 0
        known_paths = {str(path.resolve()) for path in archives}

        with self._connect() as connection:
            stale = connection.execute("SELECT id, path FROM artifacts").fetchall()
            for row in stale:
                if row["path"] not in known_paths:
                    connection.execute("DELETE FROM artifacts WHERE id = ?", (row["id"],))

        for archive_path in archives:
            stat = archive_path.stat()
            with self._connect() as connection:
                current = connection.execute(
                    "SELECT id, size_bytes, mtime_ns FROM artifacts WHERE path = ?",
                    (str(archive_path.resolve()),),
                ).fetchone()
            if (
                current
                and current["size_bytes"] == stat.st_size
                and current["mtime_ns"] == stat.st_mtime_ns
            ):
                reused += 1
                continue
            try:
                self._index_archive(archive_path)
                indexed += 1
            except (OSError, ValueError, KeyError, json.JSONDecodeError, zipfile.BadZipFile):
                failed += 1
        return {"indexed": indexed, "reused": reused, "failed": failed}

    def _index_archive(self, archive_path: Path) -> None:
        stat = archive_path.stat()
        with zipfile.ZipFile(archive_path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            bundle = json.loads(archive.read("bundle.json"))
            run_id = str(manifest.get("run_id") or bundle.get("run_id") or "")
            if not run_id:
                raise ValueError("Diagnostics artifact has no run ID")

            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "DELETE FROM artifacts WHERE path = ?",
                    (str(archive_path.resolve()),),
                )
                cursor = connection.execute(
                    """INSERT INTO artifacts (
                           path, file_name, size_bytes, mtime_ns, run_id, state,
                           partial, started_at, completed_at, last_sample_at,
                           samples, sample_interval_s, platform, python_version,
                           git_commit, gpu_rows, manifest_json, bundle_json, indexed_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)""",
                    (
                        str(archive_path.resolve()),
                        archive_path.name,
                        stat.st_size,
                        stat.st_mtime_ns,
                        run_id,
                        str(bundle.get("run_state") or manifest.get("state") or "unknown"),
                        int(bool(bundle.get("partial", manifest.get("state") != "completed"))),
                        _epoch(manifest.get("started_at")),
                        _epoch(manifest.get("completed_at")),
                        _epoch(manifest.get("last_sample_at")),
                        int(manifest.get("samples", 0)),
                        _number(manifest.get("sample_interval_s")),
                        str(manifest.get("platform", "")),
                        str(manifest.get("python", "")),
                        str(manifest.get("git_commit", "")),
                        json.dumps(manifest, sort_keys=True),
                        json.dumps(bundle, sort_keys=True),
                        datetime.now(timezone.utc).timestamp(),
                    ),
                )
                artifact_id = int(cursor.lastrowid)
                self._index_host(connection, archive, artifact_id)
                self._index_platform(connection, archive, artifact_id)
                self._index_web_events(connection, archive, artifact_id)
                self._index_issues(connection, archive, artifact_id)
                gpu_rows = self._count_csv_rows(archive, "gpu_metrics.csv")
                connection.execute(
                    "UPDATE artifacts SET gpu_rows = ? WHERE id = ?",
                    (gpu_rows, artifact_id),
                )

    @staticmethod
    def _insert_batches(
        connection: sqlite3.Connection,
        statement: str,
        rows: Iterable[tuple],
        batch_size: int = 5000,
    ) -> None:
        batch: list[tuple] = []
        for row in rows:
            batch.append(row)
            if len(batch) >= batch_size:
                connection.executemany(statement, batch)
                batch.clear()
        if batch:
            connection.executemany(statement, batch)

    def _index_host(
        self, connection: sqlite3.Connection, archive: zipfile.ZipFile, artifact_id: int
    ) -> None:
        if "host_metrics.csv" not in archive.namelist():
            return

        def rows() -> Iterable[tuple]:
            with archive.open("host_metrics.csv") as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))
                for row in reader:
                    timestamp = _epoch(row.get("ts"))
                    if timestamp is None:
                        continue
                    disk_free = None
                    disk_total = None
                    try:
                        disks = json.loads(row.get("disks_json", "{}"))
                        disk = disks.get("/app/_outputs") or next(iter(disks.values()), {})
                        disk_free = _number(disk.get("free_mib"))
                        disk_total = _number(disk.get("total_mib"))
                    except (json.JSONDecodeError, AttributeError):
                        pass
                    yield (
                        artifact_id,
                        timestamp,
                        _number(row.get("cpu_util_pct")),
                        _number(row.get("load_1m")),
                        _number(row.get("load_5m")),
                        _number(row.get("memory_total_mib")),
                        _number(row.get("memory_available_mib")),
                        _number(row.get("net_rx_bytes_per_s")),
                        _number(row.get("net_tx_bytes_per_s")),
                        _number(row.get("disk_read_bytes_per_s")),
                        _number(row.get("disk_write_bytes_per_s")),
                        disk_free,
                        disk_total,
                    )

        self._insert_batches(
            connection,
            "INSERT INTO host_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows(),
        )

    def _index_platform(
        self, connection: sqlite3.Connection, archive: zipfile.ZipFile, artifact_id: int
    ) -> None:
        if "platform_metrics.jsonl" not in archive.namelist():
            return

        def rows() -> Iterable[tuple]:
            with archive.open("platform_metrics.jsonl") as raw:
                for raw_line in raw:
                    try:
                        item = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    timestamp = _epoch(item.get("ts"))
                    if timestamp is None:
                        continue
                    stats = item.get("stats") or {}
                    web = item.get("web") or {}
                    reachable = web.get("reachable")
                    yield (
                        artifact_id,
                        timestamp,
                        _number(stats.get("pending_pairs")),
                        _number(stats.get("annotated_pairs")),
                        _number(stats.get("actionable_annotations")),
                        _number(stats.get("human_annotations")),
                        _number(stats.get("skipped_pairs")),
                        _number(stats.get("training_runs")),
                        None if reachable is None else int(bool(reachable)),
                        _number(web.get("latency_ms")),
                        str(item.get("training_status", "")),
                        str(item.get("training_step", "")),
                        str(item.get("queue_refill_status", "")),
                        str(item.get("promotion_gate_status", "")),
                    )

        self._insert_batches(
            connection,
            "INSERT INTO platform_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows(),
        )

    def _index_web_events(
        self, connection: sqlite3.Connection, archive: zipfile.ZipFile, artifact_id: int
    ) -> None:
        if "web_events.jsonl" not in archive.namelist():
            return

        def rows() -> Iterable[tuple]:
            with archive.open("web_events.jsonl") as raw:
                for raw_line in raw:
                    try:
                        item = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    timestamp = _epoch(item.get("ts"))
                    if timestamp is None or item.get("event") != "http_request":
                        continue
                    status = item.get("status")
                    yield (
                        artifact_id,
                        timestamp,
                        str(item.get("route", "unknown")),
                        str(item.get("method", "")),
                        int(status) if isinstance(status, (int, float)) else None,
                        _number(item.get("duration_ms")),
                        str(item.get("severity", "")),
                        str(item.get("error_type") or ""),
                    )

        self._insert_batches(
            connection,
            "INSERT INTO web_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows(),
        )

    def _index_issues(
        self, connection: sqlite3.Connection, archive: zipfile.ZipFile, artifact_id: int
    ) -> None:
        if "issues.jsonl" not in archive.namelist():
            return
        rows = []
        with archive.open("issues.jsonl") as raw:
            for raw_line in raw:
                try:
                    item = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                timestamp = _epoch(item.get("ts"))
                if timestamp is None:
                    continue
                rows.append(
                    (
                        artifact_id,
                        timestamp,
                        str(item.get("event", "")),
                        str(item.get("issue", "")),
                        str(item.get("severity", "")),
                        str(item.get("detail", "")),
                    )
                )
        connection.executemany("INSERT INTO issues VALUES (?, ?, ?, ?, ?, ?)", rows)

    @staticmethod
    def _count_csv_rows(archive: zipfile.ZipFile, name: str) -> int:
        if name not in archive.namelist():
            return 0
        with archive.open(name) as raw:
            return max(0, sum(1 for line in raw if line.strip()) - 1)

    def list_runs(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT a.*,
                          (SELECT COUNT(*) FROM host_metrics h WHERE h.artifact_id = a.id) AS host_rows,
                          (SELECT COUNT(*) FROM platform_metrics p WHERE p.artifact_id = a.id) AS platform_rows,
                          (SELECT COUNT(*) FROM web_events w WHERE w.artifact_id = a.id) AS web_rows,
                          (SELECT COUNT(*) FROM issues i WHERE i.artifact_id = a.id) AS issue_rows
                   FROM artifacts a ORDER BY COALESCE(a.started_at, 0) DESC"""
            ).fetchall()
        return [self._run_dict(row) for row in rows]

    def get_run(self, artifact_id: int) -> dict:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM artifacts WHERE id = ?", (artifact_id,)
            ).fetchone()
        if row is None:
            raise KeyError(artifact_id)
        return self._run_dict(row)

    @staticmethod
    def _run_dict(row: sqlite3.Row) -> dict:
        result = dict(row)
        for key in ("started_at", "completed_at", "last_sample_at", "indexed_at"):
            if key in result:
                result[key] = _iso(result[key])
        result["partial"] = bool(result["partial"])
        result.pop("manifest_json", None)
        result.pop("bundle_json", None)
        return result

    def overview(self, artifact_id: int) -> dict:
        run = self.get_run(artifact_id)
        with self._connect() as connection:
            host = connection.execute(
                """SELECT MIN(ts) AS first_ts, MAX(ts) AS last_ts,
                          AVG(cpu_util_pct) AS cpu_avg, MAX(cpu_util_pct) AS cpu_max,
                          MIN(100.0 * memory_available_mib / NULLIF(memory_total_mib, 0)) AS memory_available_min_pct,
                          MIN(100.0 * disk_free_mib / NULLIF(disk_total_mib, 0)) AS disk_free_min_pct
                   FROM host_metrics WHERE artifact_id = ?""",
                (artifact_id,),
            ).fetchone()
            latest_platform = connection.execute(
                """SELECT * FROM platform_metrics WHERE artifact_id = ?
                   ORDER BY ts DESC LIMIT 1""",
                (artifact_id,),
            ).fetchone()
            web = connection.execute(
                """SELECT COUNT(*) AS requests,
                          SUM(CASE WHEN status >= 400 OR status IS NULL THEN 1 ELSE 0 END) AS errors,
                          AVG(duration_ms) AS latency_avg, MAX(duration_ms) AS latency_max
                   FROM web_events WHERE artifact_id = ?""",
                (artifact_id,),
            ).fetchone()
            durations = [
                float(row[0])
                for row in connection.execute(
                    "SELECT duration_ms FROM web_events WHERE artifact_id = ? AND duration_ms IS NOT NULL",
                    (artifact_id,),
                )
            ]
            issues = connection.execute(
                """SELECT COUNT(*) AS events,
                          SUM(CASE WHEN event = 'issue_opened' THEN 1 ELSE 0 END) AS opened,
                          SUM(CASE WHEN event = 'issue_opened' AND severity = 'error' THEN 1 ELSE 0 END) AS errors
                   FROM issues WHERE artifact_id = ?""",
                (artifact_id,),
            ).fetchone()
        request_count = int(web["requests"] or 0)
        error_count = int(web["errors"] or 0)
        platform = dict(latest_platform) if latest_platform else {}
        platform.pop("artifact_id", None)
        if "ts" in platform:
            platform["ts"] = _iso(platform["ts"])
        return {
            "run": run,
            "bounds": {
                "start": _iso(host["first_ts"]),
                "end": _iso(host["last_ts"]),
                "start_epoch": host["first_ts"],
                "end_epoch": host["last_ts"],
            },
            "host": {
                "cpu_avg": host["cpu_avg"],
                "cpu_max": host["cpu_max"],
                "memory_available_min_pct": host["memory_available_min_pct"],
                "disk_free_min_pct": host["disk_free_min_pct"],
            },
            "platform": platform,
            "http": {
                "requests": request_count,
                "errors": error_count,
                "error_rate": error_count / request_count if request_count else 0,
                "latency_avg": web["latency_avg"],
                "latency_p95": _percentile(durations, 0.95),
                "latency_max": web["latency_max"],
            },
            "issues": dict(issues),
        }

    def timeline(
        self,
        artifact_id: int,
        start: float | None = None,
        end: float | None = None,
        points: int = 900,
    ) -> dict:
        run = self.get_run(artifact_id)
        start = start if start is not None else _epoch(run.get("started_at"))
        end = end if end is not None else _epoch(run.get("last_sample_at"))
        if start is None or end is None or end <= start:
            return {"host": [], "platform": [], "http": []}
        bucket = max(1.0, (end - start) / max(50, min(points, 2000)))
        parameters = (start, bucket, artifact_id, start, end)
        with self._connect() as connection:
            host_rows = connection.execute(
                """SELECT MIN(ts) AS ts,
                          AVG(cpu_util_pct) AS cpu_avg, MAX(cpu_util_pct) AS cpu_max,
                          AVG(load_1m) AS load_avg,
                          AVG(100.0 * (memory_total_mib - memory_available_mib) / NULLIF(memory_total_mib, 0)) AS memory_used_pct,
                          AVG(net_rx_bps + net_tx_bps) AS network_bps,
                          AVG(disk_read_bps + disk_write_bps) AS disk_bps,
                          MIN(100.0 * disk_free_mib / NULLIF(disk_total_mib, 0)) AS disk_free_pct
                   FROM host_metrics
                   WHERE artifact_id = ?3 AND ts BETWEEN ?4 AND ?5
                   GROUP BY CAST((ts - ?1) / ?2 AS INTEGER) ORDER BY ts""",
                parameters,
            ).fetchall()
            platform_rows = connection.execute(
                """SELECT MIN(ts) AS ts,
                          AVG(pending_pairs) AS pending_pairs,
                          MAX(annotated_pairs) AS annotated_pairs,
                          MAX(actionable_annotations) AS actionable_annotations,
                          AVG(web_latency_ms) AS probe_latency_ms,
                          AVG(web_reachable) AS reachable_rate
                   FROM platform_metrics
                   WHERE artifact_id = ?3 AND ts BETWEEN ?4 AND ?5
                   GROUP BY CAST((ts - ?1) / ?2 AS INTEGER) ORDER BY ts""",
                parameters,
            ).fetchall()
            http_rows = connection.execute(
                """SELECT MIN(ts) AS ts, COUNT(*) AS requests,
                          SUM(CASE WHEN status >= 400 OR status IS NULL THEN 1 ELSE 0 END) AS errors,
                          AVG(duration_ms) AS latency_avg, MAX(duration_ms) AS latency_max
                   FROM web_events
                   WHERE artifact_id = ?3 AND ts BETWEEN ?4 AND ?5
                   GROUP BY CAST((ts - ?1) / ?2 AS INTEGER) ORDER BY ts""",
                parameters,
            ).fetchall()
        return {
            "bucket_seconds": bucket,
            "host": [self._timeline_row(row) for row in host_rows],
            "platform": [self._timeline_row(row) for row in platform_rows],
            "http": [
                {
                    **self._timeline_row(row),
                    "error_rate": (row["errors"] or 0) / row["requests"] if row["requests"] else 0,
                }
                for row in http_rows
            ],
        }

    @staticmethod
    def _timeline_row(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["ts"] = _iso(result["ts"])
        return result

    def route_summary(
        self, artifact_id: int, start: float | None = None, end: float | None = None
    ) -> list[dict]:
        self.get_run(artifact_id)
        conditions = ["artifact_id = ?"]
        parameters: list[object] = [artifact_id]
        if start is not None:
            conditions.append("ts >= ?")
            parameters.append(start)
        if end is not None:
            conditions.append("ts <= ?")
            parameters.append(end)
        where = " AND ".join(conditions)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT route, method, status, duration_ms FROM web_events WHERE {where}",
                parameters,
            ).fetchall()
        grouped: dict[tuple[str, str], dict] = {}
        for row in rows:
            key = (row["route"], row["method"])
            item = grouped.setdefault(
                key,
                {"route": key[0], "method": key[1], "requests": 0, "errors": 0, "durations": []},
            )
            item["requests"] += 1
            if row["status"] is None or row["status"] >= 400:
                item["errors"] += 1
            if row["duration_ms"] is not None:
                item["durations"].append(float(row["duration_ms"]))
        result = []
        for item in grouped.values():
            durations = item.pop("durations")
            requests = item["requests"]
            result.append(
                {
                    **item,
                    "error_rate": item["errors"] / requests if requests else 0,
                    "latency_avg": sum(durations) / len(durations) if durations else None,
                    "latency_p95": _percentile(durations, 0.95),
                    "latency_max": max(durations) if durations else None,
                }
            )
        return sorted(result, key=lambda item: (-item["requests"], item["route"]))

    def issue_events(self, artifact_id: int) -> list[dict]:
        self.get_run(artifact_id)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT ts, event, issue, severity, detail FROM issues WHERE artifact_id = ? ORDER BY ts DESC",
                (artifact_id,),
            ).fetchall()
        return [{**dict(row), "ts": _iso(row["ts"])} for row in rows]