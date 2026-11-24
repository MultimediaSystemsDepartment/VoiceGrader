from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from utils.model_publisher import publish_latest_model, read_publication


def _checkpoint(path: Path, payload: bytes) -> Path:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "openai/whisper-medium"}', encoding="utf-8"
    )
    (path / "adapter_model.safetensors").write_bytes(payload)
    return path


def test_publish_latest_replaces_previous_export_with_manifest(tmp_path: Path) -> None:
    first = _checkpoint(tmp_path / "checkpoint-first", b"first")
    second = _checkpoint(tmp_path / "checkpoint-second", b"second")
    latest = tmp_path / "shared" / "latest"

    publish_latest_model(
        first,
        latest,
        model_key="dpo:first",
        checkpoint_id="first-id",
        base_model_id="openai/whisper-medium",
        monitor_id="monitor-id",
        monitor_fingerprint="fingerprint",
        wer=0.2,
    )
    manifest = publish_latest_model(
        second,
        latest,
        model_key="dpo:second",
        checkpoint_id="second-id",
        base_model_id="openai/whisper-medium",
        monitor_id="monitor-id",
        monitor_fingerprint="fingerprint",
        wer=0.19,
    )

    assert (latest / "adapter_model.safetensors").read_bytes() == b"second"
    assert manifest == read_publication(latest)
    assert manifest["checkpoint_id"] == "second-id"
    assert (
        manifest["files"]["adapter_model.safetensors"]
        == hashlib.sha256(b"second").hexdigest()
    )
    assert not list(latest.parent.glob(".latest.*-*"))


def test_publish_latest_rejects_invalid_source_without_touching_current(
    tmp_path: Path,
) -> None:
    valid = _checkpoint(tmp_path / "valid", b"current")
    latest = tmp_path / "latest"
    publish_latest_model(
        valid,
        latest,
        model_key="dpo:valid",
        checkpoint_id="valid-id",
        base_model_id="openai/whisper-medium",
        monitor_id="monitor-id",
        monitor_fingerprint="fingerprint",
        wer=0.2,
    )
    invalid = tmp_path / "invalid"
    invalid.mkdir()

    with pytest.raises(ValueError, match="Invalid policy checkpoint"):
        publish_latest_model(
            invalid,
            latest,
            model_key="dpo:invalid",
            checkpoint_id="invalid-id",
            base_model_id="openai/whisper-medium",
            monitor_id="monitor-id",
            monitor_fingerprint="fingerprint",
            wer=0.3,
        )

    assert (
        json.loads((latest / "publication.json").read_text(encoding="utf-8"))[
            "checkpoint_id"
        ]
        == "valid-id"
    )


def test_publish_rejects_mislabeled_adapter_without_creating_destination(tmp_path: Path) -> None:
    medium = _checkpoint(tmp_path / "medium", b"weights")
    latest = tmp_path / "latest"
    with pytest.raises(ValueError, match="does not match"):
        publish_latest_model(
            medium, latest, model_key="wrong-base", checkpoint_id="id",
            base_model_id="openai/whisper-large-v3", monitor_id="monitor",
            monitor_fingerprint="fingerprint", wer=0.1,
        )
    assert not latest.exists()


def test_publish_latest_restores_previous_export_when_swap_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    previous = _checkpoint(tmp_path / "previous", b"previous")
    candidate = _checkpoint(tmp_path / "candidate", b"candidate")
    latest = tmp_path / "latest"
    publish_latest_model(
        previous,
        latest,
        model_key="dpo:previous",
        checkpoint_id="previous-id",
        base_model_id="openai/whisper-medium",
        monitor_id="monitor-id",
        monitor_fingerprint="fingerprint",
        wer=0.2,
    )
    original_rename = Path.rename

    def fail_staging_swap(path: Path, target: Path) -> Path:
        if ".staging-" in path.name:
            raise OSError("simulated swap failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_staging_swap)

    with pytest.raises(OSError, match="simulated swap failure"):
        publish_latest_model(
            candidate,
            latest,
            model_key="dpo:candidate",
            checkpoint_id="candidate-id",
            base_model_id="openai/whisper-medium",
            monitor_id="monitor-id",
            monitor_fingerprint="fingerprint",
            wer=0.19,
        )

    assert (latest / "adapter_model.safetensors").read_bytes() == b"previous"
    assert read_publication(latest)["checkpoint_id"] == "previous-id"
