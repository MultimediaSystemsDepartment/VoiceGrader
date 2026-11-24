"""Verified model delivery and pinned runtime publication identity."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import secrets
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from uuid import UUID, uuid4

log = logging.getLogger(__name__)
MODEL_ID = "openai/whisper-large-v3"
REQUIRED_FILES = {"model.bin", "config.json", "tokenizer.json", "preprocessor_config.json"}
MAX_TOTAL_BYTES = 12 * 1024**3


def model_root() -> Path:
    return Path(os.environ.get("VOICEGRADER_MODEL_ROOT", "./data/models/voicegrader"))


def _json(data: bytes) -> dict:
    def reject_constant(value: str):
        raise ValueError(f"Invalid JSON constant: {value}")
    value = json.loads(data, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError("Model publication must be a JSON object")
    return value


def _validate_manifest(manifest: dict) -> None:
    if (manifest.get("schema_version") != 1
            or manifest.get("format") != "ctranslate2-whisper"
            or manifest.get("quantization") != "float16"
            or manifest.get("base_model_id") != MODEL_ID):
        raise ValueError("Publication does not match the ASR model/format")
    identity = manifest["publication_id"]
    if not isinstance(identity, str) or str(UUID(identity)) != identity:
        raise ValueError("Invalid publication identity")
    files = manifest["files"]
    if not isinstance(files, dict) or not REQUIRED_FILES <= files.keys():
        raise ValueError("Publication is missing required model files")
    total = 0
    for name, info in files.items():
        path = PurePosixPath(name)
        if (not name or name in {"publication.json", "current.json"}
                or "\\" in name or ":" in name or path.is_absolute()
                or ".." in path.parts or path.as_posix() != name):
            raise ValueError("Unsafe publication file path")
        if (not isinstance(info, dict) or type(info.get("size")) is not int
                or info["size"] <= 0
                or not isinstance(info.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", info["sha256"])):
            raise ValueError("Invalid publication checksums/sizes")
        total += info["size"]
    if total > MAX_TOTAL_BYTES:
        raise ValueError("Publication exceeds the download size limit")


def installed_publication(root: Path | None = None) -> tuple[dict, Path] | None:
    root = root or model_root()
    pointer = root / "current.json"
    if not pointer.exists():
        return None
    record = _json(pointer.read_bytes())
    identity = record["publication_id"]
    if not isinstance(identity, str) or str(UUID(identity)) != identity:
        raise ValueError("Invalid installed publication pointer")
    directory = root / identity
    manifest = _json((directory / "publication.json").read_bytes())
    _validate_manifest(manifest)
    if manifest["publication_id"] != identity:
        raise ValueError("Installed model identity mismatch")
    for filename in manifest["files"]:
        path = directory / filename
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Installed model files are missing or unsafe")
    return manifest, directory


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Model delivery redirects are not permitted")


def _open(url: str, token: str):
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    return urllib.request.build_opener(_NoRedirect()).open(request, timeout=60)


def service_authorized(authorization: str | None) -> bool:
    token_file = os.environ.get("VOICEGRADER_BRIDGE_TOKEN_FILE", "")
    if not token_file:
        return False
    token = Path(token_file).read_text(encoding="ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", token):
        raise ValueError("Invalid model-bridge credential")
    return secrets.compare_digest(authorization or "", "Bearer " + token)


def latest_publication() -> dict | None:
    base = os.environ.get("VOICEGRADER_RLHF_URL", "").rstrip("/")
    if not base:
        return None
    parsed = urllib.parse.urlsplit(base)
    if (parsed.scheme not in {"http", "https"} or not parsed.netloc
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Invalid model-bridge base URL")
    token = Path(os.environ["VOICEGRADER_BRIDGE_TOKEN_FILE"]).read_text(encoding="ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", token):
        raise ValueError("Invalid model-bridge credential")
    try:
        response = _open(f"{base}/api/model-publication/latest", token)
    except urllib.error.HTTPError as exc:
        if exc.code == 404 and exc.headers.get_content_type() == "application/json":
            error = _json(exc.read(65536))
            if isinstance(error.get("detail"), dict) and error["detail"].get("code") == "publication_not_ready":
                return None
        raise
    with response:
        if response.headers.get_content_type() != "application/json":
            raise ValueError("Model metadata endpoint did not return JSON")
        data = response.read(1024 * 1024 + 1)
        if len(data) > 1024 * 1024:
            raise ValueError("Model metadata exceeds size limit")
        manifest = _json(data)
    _validate_manifest(manifest)
    return manifest


def _loadable(directory: Path) -> None:
    from faster_whisper import WhisperModel
    model = WhisperModel(str(directory), device="cuda", compute_type="float16",
                         local_files_only=True)
    del model


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pin_installed_publication(identity: str | None, root: Path | None = None) -> None:
    root = root or model_root()
    if identity is None:
        (root / "current.json").unlink(missing_ok=True)
        return
    if str(UUID(identity)) != identity:
        raise ValueError("Invalid publication pointer identity")
    manifest = _json((root / identity / "publication.json").read_bytes())
    _validate_manifest(manifest)
    if manifest["publication_id"] != identity:
        raise ValueError("Publication pointer identity mismatch")
    pointer = root / f".current-{uuid4()}"
    try:
        with pointer.open("w", encoding="utf-8") as output:
            json.dump({"publication_id": identity}, output)
            output.flush()
            os.fsync(output.fileno())
        pointer.replace(root / "current.json")
    finally:
        pointer.unlink(missing_ok=True)


def refresh_model(*, root: Path | None = None, validate_model=_loadable) -> dict | None:
    root = root or model_root()
    installed = installed_publication(root)
    base = os.environ.get("VOICEGRADER_RLHF_URL", "").rstrip("/")
    if not base:
        return installed[0] if installed else None
    token_file = os.environ.get("VOICEGRADER_BRIDGE_TOKEN_FILE", "")
    if not token_file:
        raise ValueError("Model bridge requires its service credential file")
    token = Path(token_file).read_text(encoding="ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", token):
        raise ValueError("Invalid model-bridge credential")
    manifest = latest_publication()
    if manifest is None:
        log.info("No approved publication; retaining installed model or stock Large-v3")
        return installed[0] if installed else None
    if installed and installed[0]["publication_id"] == manifest["publication_id"]:
        if installed[0] != manifest:
            raise ValueError("Immutable publication metadata changed")
        return manifest
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < sum(info["size"] for info in manifest["files"].values()):
        raise OSError("Insufficient free space for staged model update")
    staging = root / f".staging-{uuid4()}"
    staging.mkdir()
    try:
        for filename, info in manifest["files"].items():
            target = staging / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            url = (f"{base}/api/model-publication/{manifest['publication_id']}/files/"
                   f"{urllib.parse.quote(filename, safe='/')}")
            digest = hashlib.sha256()
            received = 0
            with _open(url, token) as source, target.open("wb") as output:
                while block := source.read(1024 * 1024):
                    received += len(block)
                    if received > info["size"]:
                        raise ValueError("Downloaded model file exceeds declared size")
                    digest.update(block)
                    output.write(block)
            if received != info["size"] or digest.hexdigest() != info["sha256"]:
                raise ValueError("Downloaded model file failed checksum verification")
        (staging / "publication.json").write_text(
            json.dumps(manifest, allow_nan=False), encoding="utf-8",
        )
        validate_model(staging)
        destination = root / manifest["publication_id"]
        if destination.exists():
            # Rollback may select a retained immutable version; verify its bytes.
            for filename, info in manifest["files"].items():
                path = destination / filename
                if (path.is_symlink() or not path.is_file()
                        or not path.resolve().is_relative_to(destination.resolve())
                        or _file_digest(path) != info["sha256"]):
                    raise ValueError("Retained publication has changed")
            existing = _json((destination / "publication.json").read_bytes())
            if existing != manifest:
                raise ValueError("Retained publication metadata has changed")
        else:
            staging.rename(destination)
        pin_installed_publication(manifest["publication_id"], root)
        log.info("Verified publication %s downloaded; awaiting runtime activation",
                 manifest["publication_id"])
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def startup_model_update() -> None:
    try:
        refresh_model()
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        log.exception("Model update failed; retaining the installed version")
        os.environ["VOICEGRADER_MODEL_UPDATE_ERROR"] = str(exc)
    installed = installed_publication()
    if installed:
        manifest, directory = installed
        os.environ["VOICEGRADER_ACTIVE_MODEL_DIR"] = str(directory.resolve())
        os.environ["VOICEGRADER_ACTIVE_PUBLICATION_ID"] = manifest["publication_id"]
    else:
        os.environ.pop("VOICEGRADER_ACTIVE_MODEL_DIR", None)
        os.environ["VOICEGRADER_ACTIVE_PUBLICATION_ID"] = ""
        log.info("ASR starts with stock Large-v3; no installed approved publication")


def loaded_model_status() -> dict:
    installed = installed_publication()
    downloaded = installed[0]["publication_id"] if installed else None
    active = os.environ.get("VOICEGRADER_ACTIVE_PUBLICATION_ID") or None
    return {
        "base_model_id": MODEL_ID, "loaded_publication_id": active,
        "downloaded_publication_id": downloaded,
        "restart_required": downloaded != active,
        "state": "published" if active else "stock",
        "update_error": os.environ.get("VOICEGRADER_MODEL_UPDATE_ERROR") or None,
    }


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        manifest = refresh_model()
    except (OSError, ValueError, KeyError, RuntimeError):
        log.exception("Model refresh failed; installed version unchanged")
        return 1
    print(json.dumps({"downloaded_publication_id": manifest["publication_id"] if manifest else None,
                      "restart_required": bool(manifest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
