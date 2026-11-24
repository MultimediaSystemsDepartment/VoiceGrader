from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from evaluation.canary import evaluation_profile_hash
from paths import PathsConfig
from utils import asr_publication
from utils.bridge_auth import ensure_bridge_token, read_bridge_token
from web import publication_api


class Store:
    def __init__(self, source: Path):
        self.source = source
        self.approved = True
        self.healthy = True

    def get_setting(self, key, default=""):
        return str(self.source) if key == "active_policy_checkpoint" else default

    def get_model_checkpoint_by_path(self, path):
        return {"id": self.source.name, "checkpoint_path": path, "available": True}

    def get_policy_activations(self):
        return [{"new_checkpoint_id": self.source.name,
                 "reason": "automatic" if self.approved else "manual"}]

    def get_promotion_gate_canary_set(self):
        return {"id": "monitor", "status": "ready", "fingerprint": "frozen"}

    def get_canary_evaluations(self, identity):
        return [{
            "id": "evaluation", "checkpoint_id": self.source.name,
            "status": "completed", "health_status": "healthy" if self.healthy else "warning",
            "profile_hash": evaluation_profile_hash(self.get_promotion_gate_canary_set()),
        }]


def checkpoint(path: Path) -> Path:
    path.mkdir()
    (path / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "openai/whisper-large-v3",
    }))
    (path / "adapter_model.safetensors").write_bytes(path.name.encode())
    return path


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source = checkpoint(tmp_path / "first")
    store = Store(source)
    paths = PathsConfig(output_dir=str(tmp_path / "outputs"))

    def convert(source, base, merged, converted):
        converted.mkdir(parents=True)
        for name in asr_publication.REQUIRED_FILES:
            (converted / name).write_bytes(source.name.encode())

    monkeypatch.setattr(asr_publication, "convert_adapter", convert)
    monkeypatch.setattr(asr_publication, "validate_conversion", lambda *args: {"ci_high": 0})
    return store, paths


def test_publication_requires_automatic_approval_and_healthy_monitor(setup):
    store, paths = setup
    store.approved = False
    with pytest.raises(ValueError, match="gate-approved"):
        asr_publication.publish_approved_model(store, paths)
    store.approved = True
    store.healthy = False
    with pytest.raises(ValueError, match="healthy evaluation"):
        asr_publication.publish_approved_model(store, paths)
    assert not asr_publication.publication_root(paths).exists()


def test_publication_is_immutable_and_rollback_has_new_identity(setup, tmp_path):
    store, paths = setup
    first = asr_publication.publish_approved_model(store, paths)
    assert asr_publication.publish_approved_model(store, paths) == first
    store.source = checkpoint(tmp_path / "second")
    second = asr_publication.publish_approved_model(store, paths)
    store.source = tmp_path / "first"
    rollback = asr_publication.publish_approved_model(store, paths)
    assert len({row["publication_id"] for row in (first, second, rollback)}) == 3
    assert rollback["checkpoint_id"] == first["checkpoint_id"]
    root = asr_publication.publication_root(paths)
    assert (root / first["publication_id"] / "model.bin").read_bytes() == b"first"
    assert asr_publication.read_current_publication(paths) == rollback
    assert not list(root.glob(".staging-*"))


def test_failed_conversion_retains_current_publication(setup, monkeypatch):
    store, paths = setup
    first = asr_publication.publish_approved_model(store, paths)
    store.source = checkpoint(store.source.parent / "second")

    def fail(*args):
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(asr_publication, "convert_adapter", fail)
    with pytest.raises(RuntimeError, match="conversion failed"):
        asr_publication.publish_approved_model(store, paths)
    assert asr_publication.read_current_publication(paths) == first


def test_published_checkpoint_mutation_is_not_silently_republished(setup):
    store, paths = setup
    first = asr_publication.publish_approved_model(store, paths)
    (store.source / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="weights changed"):
        asr_publication.publish_approved_model(store, paths)
    assert asr_publication.read_current_publication(paths) == first


def test_authenticated_metadata_and_allowlisted_files(setup, tmp_path, monkeypatch):
    store, paths = setup
    manifest = asr_publication.publish_approved_model(store, paths)
    monkeypatch.setenv("VOICEGRADER_BRIDGE_TOKEN_FILE", str(tmp_path / "credential"))
    ensure_bridge_token()
    token = read_bridge_token()
    monkeypatch.setattr(publication_api, "DEFAULT_PATHS", paths)
    monkeypatch.setattr(publication_api, "FeedbackStore", lambda path: store)
    app = FastAPI()
    app.include_router(publication_api.router)
    client = TestClient(app)
    assert client.get("/api/model-publication/latest").status_code == 401
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/api/model-publication/latest", headers=headers).json() == manifest
    base = f"/api/model-publication/{manifest['publication_id']}/files"
    assert client.get(f"{base}/model.bin", headers=headers).content == b"first"
    assert client.get(f"{base}/publication.json", headers=headers).status_code == 404
    assert client.get(f"{base}/%2e%2e/secret", headers=headers).status_code == 404
    store.approved = False
    response = client.get("/api/model-publication/latest", headers=headers)
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "publication_not_ready"
