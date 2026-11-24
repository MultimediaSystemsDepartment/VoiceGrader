from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from utils.policy_checkpoint import validate_policy_checkpoint

PUBLICATION_MANIFEST = "publication.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != PUBLICATION_MANIFEST
    }


def publish_latest_model(
    source: Path,
    destination: Path,
    *,
    model_key: str,
    checkpoint_id: str,
    base_model_id: str,
    monitor_id: str,
    monitor_fingerprint: str,
    wer: float,
) -> dict[str, Any]:
    source = source.resolve()
    destination = destination.resolve()
    validate_policy_checkpoint(source, base_model_id)
    if source == destination or source in destination.parents:
        raise ValueError("Latest-model destination cannot be inside the checkpoint")

    destination.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    staging = destination.parent / f".{destination.name}.staging-{token}"
    backup = destination.parent / f".{destination.name}.backup-{token}"
    manifest: dict[str, Any]
    try:
        shutil.copytree(source, staging)
        files = _file_manifest(staging)
        if not files:
            raise ValueError("Checkpoint contains no publishable files")
        manifest = {
            "schema_version": 1,
            "published_at": datetime.now(timezone.utc).isoformat(),
            "base_model_id": base_model_id,
            "checkpoint_id": checkpoint_id,
            "model_key": model_key,
            "source_checkpoint": str(source),
            "monitor_id": monitor_id,
            "monitor_fingerprint": monitor_fingerprint,
            "wer": wer,
            "files": files,
        }
        (staging / PUBLICATION_MANIFEST).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        had_previous = destination.exists()
        if had_previous:
            destination.rename(backup)
        try:
            staging.rename(destination)
        except Exception:
            if had_previous and backup.exists():
                backup.rename(destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup.exists() and not destination.exists():
            backup.rename(destination)

    return manifest


def read_publication(destination: Path) -> dict[str, Any] | None:
    manifest_path = destination / PUBLICATION_MANIFEST
    if not manifest_path.is_file():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
