from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import types
from pathlib import Path
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "voicegrader_model_update", ROOT / "asr" / "asr_processor" / "model_update.py",
)
update = importlib.util.module_from_spec(spec)
spec.loader.exec_module(update)


class Response(io.BytesIO):
    class Headers:
        def get_content_type(self):
            return "application/json"
    headers = Headers()


def artifact(payload=b"model"):
    files = {name: payload for name in update.REQUIRED_FILES}
    manifest = {
        "schema_version": 1, "format": "ctranslate2-whisper",
        "quantization": "float16", "base_model_id": update.MODEL_ID,
        "publication_id": str(uuid4()),
        "files": {name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                  for name, data in files.items()},
    }
    return manifest, files


@pytest.fixture
def delivery(tmp_path, monkeypatch):
    manifest, files = artifact()
    token_file = tmp_path / "credential"
    token_file.write_text("a" * 64)
    monkeypatch.setenv("VOICEGRADER_RLHF_URL", "http://rlhf:8080")
    monkeypatch.setenv("VOICEGRADER_BRIDGE_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("VOICEGRADER_MODEL_ROOT", str(tmp_path / "models"))

    def serve(url, token):
        assert token == "a" * 64
        if url.endswith("/latest"):
            return Response(json.dumps(manifest).encode())
        return Response(files[url.rsplit("/", 1)[1]])

    monkeypatch.setattr(update, "_open", serve)
    return manifest, files


def test_verified_download_and_rollback_are_pinned_until_restart(delivery, monkeypatch):
    remote, _ = delivery
    first = json.loads(json.dumps(remote))
    refresh = update.refresh_model

    def restart():
        monkeypatch.setattr(update, "refresh_model",
                            lambda: update.installed_publication()[0])
        update.startup_model_update()
        monkeypatch.setattr(update, "refresh_model", refresh)

    refresh(validate_model=lambda path: None)
    restart()
    assert update.loaded_model_status()["loaded_publication_id"] == first["publication_id"]
    second, _ = artifact()
    remote.clear()
    remote.update(second)
    refresh(validate_model=lambda path: None)
    status = update.loaded_model_status()
    assert status["loaded_publication_id"] == first["publication_id"]
    assert status["downloaded_publication_id"] == second["publication_id"]
    assert status["restart_required"] is True
    restart()
    remote.clear()
    remote.update(first)
    refresh(validate_model=lambda path: None)
    status = update.loaded_model_status()
    assert status["loaded_publication_id"] == second["publication_id"]
    assert status["downloaded_publication_id"] == first["publication_id"]
    assert status["restart_required"] is True


def test_checksum_and_load_failure_keep_installed_pointer(delivery):
    manifest, files = delivery
    update.refresh_model(validate_model=lambda path: None)
    before = (update.model_root() / "current.json").read_bytes()
    manifest["publication_id"] = str(uuid4())
    files["model.bin"] = b"wrong"
    with pytest.raises(ValueError, match="checksum"):
        update.refresh_model(validate_model=lambda path: None)
    assert (update.model_root() / "current.json").read_bytes() == before
    files["model.bin"] = b"model"

    def fail(path):
        raise RuntimeError("not loadable")

    with pytest.raises(RuntimeError, match="not loadable"):
        update.refresh_model(validate_model=fail)
    assert (update.model_root() / "current.json").read_bytes() == before
    assert not list(update.model_root().glob(".staging-*"))


def test_gpu_activation_failure_keeps_previous_publication(delivery, monkeypatch):
    manifest, _ = delivery
    previous = manifest["publication_id"]
    update.refresh_model(validate_model=lambda path: None)
    manifest["publication_id"] = str(uuid4())

    def load_model(path, *, device, **kwargs):
        if device == "cuda":
            raise RuntimeError("GPU allocation failed")
        return object()

    monkeypatch.setitem(sys.modules, "faster_whisper",
                        types.SimpleNamespace(WhisperModel=load_model))
    with pytest.raises(RuntimeError, match="GPU allocation failed"):
        update.refresh_model()
    assert update.installed_publication()[0]["publication_id"] == previous


def test_path_traversal_and_wrong_model_are_rejected_before_download(delivery):
    manifest, _ = delivery
    for filename in ("../secret", "publication.json", "current.json"):
        manifest["files"][filename] = next(iter(manifest["files"].values()))
        with pytest.raises(ValueError, match="Unsafe"):
            update.refresh_model(validate_model=lambda path: None)
        assert not update.model_root().exists()
        del manifest["files"][filename]
    manifest["base_model_id"] = "openai/whisper-medium"
    with pytest.raises(ValueError, match="model/format"):
        update.refresh_model(validate_model=lambda path: None)


def test_startup_reports_unavailable_bridge_without_losing_installed(delivery, monkeypatch):
    manifest, _ = delivery
    update.refresh_model(validate_model=lambda path: None)

    def offline():
        raise OSError("bridge offline")

    monkeypatch.setattr(update, "refresh_model", offline)
    update.startup_model_update()
    status = update.loaded_model_status()
    assert status["loaded_publication_id"] == manifest["publication_id"]
    assert status["update_error"] == "bridge offline"


def test_failed_runtime_can_repin_previous_publication_or_stock(delivery):
    manifest, _ = delivery
    previous = manifest["publication_id"]
    update.refresh_model(validate_model=lambda path: None)
    manifest["publication_id"] = str(uuid4())
    update.refresh_model(validate_model=lambda path: None)
    update.pin_installed_publication(previous)
    assert update.installed_publication()[0]["publication_id"] == previous
    update.pin_installed_publication(None)
    assert update.installed_publication() is None
    assert (update.model_root() / previous).is_dir()
