from __future__ import annotations

import json
import logging
import re
import secrets
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import FileResponse

from paths import DEFAULT_PATHS
from storage.feedback_store import FeedbackStore
from utils.asr_publication import (
    approved_policy, checked_relative_path, publication_root,
    read_current_publication, validate_manifest,
)
from utils.bridge_auth import read_bridge_token

router = APIRouter(prefix="/api/model-publication", tags=["ASR publication"])
log = logging.getLogger(__name__)


def require_bridge_token(authorization: str = Header(default="")) -> None:
    try:
        expected = read_bridge_token()
    except (OSError, ValueError):
        log.exception("Model-bridge credential is unavailable")
        raise HTTPException(503, "Model-bridge credential is unavailable")
    if (not re.fullmatch(r"Bearer [0-9a-f]{64}", authorization)
            or not secrets.compare_digest(authorization, f"Bearer {expected}")):
        raise HTTPException(401, "Model-bridge authentication required")


@router.get("/latest", dependencies=[Depends(require_bridge_token)])
def latest_publication():
    manifest = read_current_publication(DEFAULT_PATHS)
    if manifest:
        try:
            checkpoint, monitor, _ = approved_policy(
                FeedbackStore(DEFAULT_PATHS.db_path), DEFAULT_PATHS,
            )
        except ValueError as exc:
            log.info("ASR publication unavailable: %s", exc)
        else:
            if (manifest["checkpoint_id"] == checkpoint["id"]
                    and manifest["monitor_fingerprint"] == monitor["fingerprint"]):
                return manifest
    raise HTTPException(404, detail={
        "code": "publication_not_ready", "message": "No approved ASR publication is ready",
    })


@router.get("/{publication_id}/files/{filename:path}",
            dependencies=[Depends(require_bridge_token)])
def publication_file(publication_id: str, filename: str):
    try:
        if str(UUID(publication_id)) != publication_id:
            raise ValueError("Invalid publication identity")
        checked_relative_path(filename)
    except (ValueError, TypeError):
        raise HTTPException(404, "Unknown publication file")
    root = publication_root(DEFAULT_PATHS) / publication_id
    if root.is_symlink():
        raise HTTPException(404, "Unknown publication")
    path = root / "publication.json"
    if not path.is_file():
        raise HTTPException(404, "Unknown publication")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    if filename not in manifest["files"]:
        raise HTTPException(404, "Unknown publication file")
    target = root / filename
    if target.is_symlink() or not target.is_file() or not target.resolve().is_relative_to(root.resolve()):
        raise HTTPException(404, "Publication file is unavailable")
    return FileResponse(target, media_type="application/octet-stream")
