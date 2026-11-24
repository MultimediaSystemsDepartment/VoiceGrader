from __future__ import annotations

import hashlib
import json
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from paths import PathsConfig
from storage.feedback_store import FeedbackStore

_BUNDLE_KINDS = {"sanitized", "sensitive"}
_SAFE_RUN_FILES = {
    "manifest.json",
    "host_metrics.csv",
    "gpu_metrics.csv",
    "platform_metrics.jsonl",
    "issues.jsonl",
}


def validate_run_id(run_id: str) -> str:
    parsed = uuid.UUID(run_id)
    if parsed.version != 4 or str(parsed) != run_id:
        raise ValueError("Diagnostics run ID must be a canonical UUID4")
    return run_id


def diagnostics_root(paths: PathsConfig) -> Path:
    return Path(paths.output_dir) / "diagnostics" / "runs"


def run_root(paths: PathsConfig, run_id: str) -> Path:
    validate_run_id(run_id)
    root = diagnostics_root(paths).resolve()
    candidate = (root / run_id).resolve()
    if candidate.parent != root:
        raise ValueError("Diagnostics run escapes the artifact root")
    return candidate


def bundle_path(paths: PathsConfig, run_id: str, kind: str) -> Path:
    if kind not in _BUNDLE_KINDS:
        raise ValueError("Unknown diagnostics bundle kind")
    return run_root(paths, run_id) / f"diagnostics-{run_id}-{kind}.zip"


def list_runs(paths: PathsConfig) -> list[dict]:
    root = diagnostics_root(paths)
    if not root.exists():
        return []
    runs = []
    for child in root.iterdir():
        if not child.is_dir() or child.is_symlink():
            continue
        try:
            validate_run_id(child.name)
            manifest = json.loads((child / "manifest.json").read_text(encoding="utf-8"))
        except (ValueError, OSError, json.JSONDecodeError):
            continue
        bundles = {}
        for kind in sorted(_BUNDLE_KINDS):
            path = bundle_path(paths, child.name, kind)
            if path.is_file() and not path.is_symlink():
                bundles[kind] = {
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                    **_bundle_metadata(path, manifest),
                }
        runs.append({**manifest, "bundles": bundles})
    return sorted(runs, key=lambda item: item.get("started_at", ""), reverse=True)


def build_bundle(
    store: FeedbackStore,
    paths: PathsConfig,
    run_id: str,
    kind: str,
) -> Path:
    destination = bundle_path(paths, run_id, kind)
    root = destination.parent
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    run_state = str(manifest.get("state", "unknown"))
    if run_state not in {"running", "completed", "interrupted"}:
        raise RuntimeError(f"Cannot bundle diagnostics run in state: {run_state}")
    captured_at = _utc_now()
    partial = run_state != "completed"
    temporary = destination.with_suffix(".zip.tmp")
    temporary.unlink(missing_ok=True)
    snapshot = root / "feedback.snapshot.sqlite"
    snapshot.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(_SAFE_RUN_FILES):
                source = root / name
                if source.is_file() and not source.is_symlink():
                    if name == "manifest.json":
                        archive.writestr(name, json.dumps(manifest, indent=2))
                    else:
                        archive.writestr(name, _complete_lines(source))
            _write_filtered_events(
                archive,
                Path(paths.logs_dir) / "web_events.jsonl",
                run_id,
                manifest.get("started_at", ""),
                manifest.get("completed_at", ""),
            )
            metadata = {
                "schema_version": 1,
                "run_id": run_id,
                "kind": kind,
                "sensitive": kind == "sensitive",
                "partial": partial,
                "run_state": run_state,
                "captured_at": captured_at,
                "samples": int(manifest.get("samples", 0)),
                "excludes": [
                    "audio",
                    "model checkpoints",
                    "password configuration",
                    "environment dumps",
                ],
            }
            archive.writestr("bundle.json", json.dumps(metadata, indent=2))
            if kind == "sensitive":
                store.clone_to(snapshot)
                archive.write(snapshot, "sensitive/feedback.sqlite")
                logs_root = Path(paths.logs_dir)
                if logs_root.exists():
                    for log_path in sorted(logs_root.glob("*.log")):
                        if log_path.is_file() and not log_path.is_symlink():
                            archive.writestr(
                                f"sensitive/logs/{log_path.name}",
                                _complete_lines(log_path),
                            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
        snapshot.unlink(missing_ok=True)
    return destination


def delete_run(paths: PathsConfig, run_id: str, active_run_id: str | None) -> None:
    if run_id == active_run_id:
        raise RuntimeError("Cannot delete the active diagnostics run")
    target = run_root(paths, run_id)
    if not target.is_dir() or target.is_symlink():
        raise FileNotFoundError(f"Diagnostics run does not exist: {run_id}")
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("state") == "running":
        raise RuntimeError("Cannot delete a running diagnostics capture")
    shutil.rmtree(target)


def _write_filtered_events(
    archive: zipfile.ZipFile,
    source: Path,
    run_id: str,
    started_at: str,
    completed_at: str,
) -> None:
    if not source.is_file() or source.is_symlink():
        return
    selected = []
    for line in source.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        timestamp = str(record.get("ts", ""))
        in_window = bool(
            started_at
            and timestamp >= started_at
            and (not completed_at or timestamp <= completed_at)
        )
        if record.get("run_id") == run_id or in_window:
            selected.append(json.dumps(record, sort_keys=True))
    if selected:
        archive.writestr("web_events.jsonl", "\n".join(selected) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _complete_lines(path: Path) -> bytes:
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return data
    complete, separator, _ = data.rpartition(b"\n")
    return complete + separator if separator else b""


def _bundle_metadata(path: Path, manifest: dict) -> dict:
    fallback = {
        "partial": manifest.get("state") != "completed",
        "run_state": str(manifest.get("state", "unknown")),
        "captured_at": str(
            manifest.get("completed_at") or manifest.get("last_sample_at") or ""
        ),
        "samples": int(manifest.get("samples", 0)),
    }
    try:
        with zipfile.ZipFile(path) as archive:
            metadata = json.loads(archive.read("bundle.json"))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return fallback
    return {
        "partial": bool(metadata.get("partial", fallback["partial"])),
        "run_state": str(metadata.get("run_state", fallback["run_state"])),
        "captured_at": str(metadata.get("captured_at", fallback["captured_at"])),
        "samples": int(metadata.get("samples", fallback["samples"])),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

