from __future__ import annotations

import os
import json
from pathlib import Path
import pytest

from run_evaluate_canary import _intermediate_sft_checkpoint_id, _select_round_output
from utils.policy_checkpoint import (
    ensure_active_policy_model_version,
    find_latest_policy_checkpoint,
    policy_model_version,
    record_active_policy_checkpoint,
    register_policy_candidate,
    resolve_active_policy_checkpoint,
    rollback_active_policy_checkpoint,
    validate_policy_checkpoint,
)
from tests.conftest import make_audio, make_pair


class FakeStore:
    def __init__(self, value: str = "") -> None:
        self.settings = {"active_policy_checkpoint": value}

    def get_setting(self, key: str, default: str = "") -> str:
        return self.settings.get(key, default)

    def set_setting(self, key: str, value: str) -> None:
        self.settings[key] = value


def _checkpoint(path: Path, mtime: int) -> Path:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "openai/whisper-medium"}),
        encoding="utf-8",
    )
    os.utime(path, (mtime, mtime))
    return path


def _predictions(prefix: str, wers: list[float]) -> list[dict]:
    import json

    return [
        {
            "canary_item_id": str(index),
            "hypothesis": f"{prefix}-{index}",
            "metrics_json": json.dumps({"wer": wer}),
        }
        for index, wer in enumerate(wers)
    ]


def test_round_output_selection_promotes_better_sft_when_dpo_fails() -> None:
    active = _predictions("active", [0.2] * 10)
    sft = {"id": "sft", "stage": "sft"}
    dpo = {"id": "dpo", "stage": "dpo"}

    selected, decisions = _select_round_output(
        active,
        [
            (sft, _predictions("sft", [0.1] * 10)),
            (dpo, _predictions("dpo", [0.3] * 10)),
        ],
    )

    assert selected == sft
    assert decisions["sft"]["promote"] is True
    assert decisions["dpo"]["promote"] is False


def test_round_output_selection_chooses_best_eligible_output() -> None:
    active = _predictions("active", [0.2] * 10)
    sft = {"id": "sft", "stage": "sft"}
    dpo = {"id": "dpo", "stage": "dpo"}

    selected, _ = _select_round_output(
        active,
        [
            (sft, _predictions("sft", [0.15] * 10)),
            (dpo, _predictions("dpo", [0.1] * 10)),
        ],
    )

    assert selected == dpo


def test_round_output_selection_requires_strict_wer_improvement() -> None:
    active = _predictions("active", [0.2] * 10)
    sft = {"id": "sft", "stage": "sft"}
    dpo = {"id": "dpo", "stage": "dpo"}

    selected, decisions = _select_round_output(
        active,
        [
            (sft, _predictions("sft", [0.205] * 10)),
            (dpo, _predictions("dpo", [0.3] * 10)),
        ],
    )

    assert decisions["sft"]["promote"] is True
    assert selected is None


def test_explicit_active_policy_wins_over_newer_checkpoint(tmp_path: Path) -> None:
    checkpoints = tmp_path / "_outputs" / "checkpoints"
    active = _checkpoint(checkpoints / "whisper_dpo_old", 1)
    _checkpoint(checkpoints / "whisper_dpo_new", 2)
    store = FakeStore(str(active.relative_to(tmp_path)))

    assert resolve_active_policy_checkpoint(store, checkpoints, tmp_path) == active


def test_stale_active_policy_falls_back_to_latest_valid_adapter(tmp_path: Path) -> None:
    checkpoints = tmp_path / "_outputs" / "checkpoints"
    _checkpoint(checkpoints / "sft", 1)
    latest = _checkpoint(checkpoints / "whisper_dpo_latest", 2)
    (checkpoints / "whisper_dpo_incomplete").mkdir(parents=True)
    store = FakeStore("D:/old-machine/missing-checkpoint")

    assert find_latest_policy_checkpoint(checkpoints) == latest
    assert resolve_active_policy_checkpoint(store, checkpoints, tmp_path) == latest


def test_cold_start_initializes_base_policy_model_version() -> None:
    store = FakeStore()

    model_version = ensure_active_policy_model_version(
        store, "openai/whisper-medium", None
    )

    assert model_version == "openai/whisper-medium"
    assert store.settings["active_policy_model_version"] == model_version


def test_initialized_base_policy_does_not_recover_unactivated_adapter(tmp_path: Path) -> None:
    pending = _checkpoint(tmp_path / "whisper_dpo_pending", 1)
    store = FakeStore()
    ensure_active_policy_model_version(store, "openai/whisper-medium", None)
    store.settings["pending_policy_checkpoint_id"] = "pending-candidate"

    assert find_latest_policy_checkpoint(tmp_path) == pending
    assert resolve_active_policy_checkpoint(
        store, tmp_path, tmp_path, "openai/whisper-medium"
    ) is None
    assert store.settings["active_policy_model_version"] == "openai/whisper-medium"
    assert store.settings["active_policy_checkpoint"] == ""


def test_policy_model_version_includes_adapter_name(tmp_path: Path) -> None:
    checkpoint = tmp_path / "whisper_dpo_active"

    assert policy_model_version("openai/whisper-medium", checkpoint) == (
        "openai/whisper-medium+lora:whisper_dpo_active"
    )


def test_policy_identity_initialization_does_not_overwrite_activation() -> None:
    store = FakeStore()
    store.settings["active_policy_model_version"] = "openai/whisper-medium+lora:deployed-policy"

    model_version = ensure_active_policy_model_version(
        store, "openai/whisper-medium", None
    )

    assert model_version == "openai/whisper-medium+lora:deployed-policy"
    assert store.settings["active_policy_model_version"] == model_version


def test_recovery_ignores_newer_incompatible_adapter(tmp_path: Path) -> None:
    compatible = _checkpoint(tmp_path / "whisper_dpo_compatible", 1)
    (compatible / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "openai/whisper-large-v3"}', encoding="utf-8"
    )
    _checkpoint(tmp_path / "whisper_dpo_medium", 2)
    assert resolve_active_policy_checkpoint(
        FakeStore(), tmp_path, tmp_path, "openai/whisper-large-v3"
    ) == compatible


def test_explicit_incompatible_policy_fails_without_changing_state(tmp_path: Path) -> None:
    medium = _checkpoint(tmp_path / "whisper_dpo_medium", 1)
    store = FakeStore(str(medium))
    with pytest.raises(ValueError, match="does not match"):
        resolve_active_policy_checkpoint(store, tmp_path, tmp_path, "openai/whisper-large-v3")
    assert store.settings["active_policy_checkpoint"] == str(medium)
    store.settings["active_policy_model_version"] = "openai/whisper-medium"
    with pytest.raises(ValueError, match="fresh policy database"):
        ensure_active_policy_model_version(store, "openai/whisper-large-v3", None)
    assert store.settings["active_policy_model_version"] == "openai/whisper-medium"


def test_activation_rejects_incompatible_adapter_before_audit_changes(tmp_path: Path) -> None:
    medium = _checkpoint(tmp_path / "whisper_dpo_medium", 1)
    store = FakeStore()
    with pytest.raises(ValueError, match="does not match"):
        record_active_policy_checkpoint(
            store, medium, tmp_path, base_model_id="openai/whisper-large-v3"
        )
    assert store.settings == {"active_policy_checkpoint": ""}


@pytest.mark.parametrize("payload", ["{}", "[]", "{"])
def test_unverifiable_adapter_is_rejected(tmp_path: Path, payload: str) -> None:
    checkpoint = _checkpoint(tmp_path / "whisper_dpo_invalid", 1)
    (checkpoint / "adapter_config.json").write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        validate_policy_checkpoint(checkpoint, "openai/whisper-large-v3")


def test_record_active_policy_uses_portable_relative_path(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "_outputs" / "checkpoints" / "sft", 1)
    store = FakeStore()

    record_active_policy_checkpoint(store, checkpoint, tmp_path)

    assert store.settings["active_policy_checkpoint"] == str(
        Path("_outputs") / "checkpoints" / "sft"
    )


def test_publication_and_rollback_are_audited(store, tmp_path: Path) -> None:
    checkpoints = tmp_path / "_outputs" / "checkpoints"
    first = _checkpoint(checkpoints / "whisper_dpo_first", 1)
    second = _checkpoint(checkpoints / "whisper_dpo_second", 2)

    record_active_policy_checkpoint(store, first, tmp_path)
    record_active_policy_checkpoint(store, second, tmp_path)
    first_row = store.get_model_checkpoint_by_path(str(first.resolve()))
    rollback_active_policy_checkpoint(store, first_row["id"], tmp_path)

    activations = list(reversed(store.get_policy_activations()))
    assert [row["reason"] for row in activations] == [
        "automatic",
        "automatic",
        "rollback",
    ]
    assert activations[1]["previous_model_key"] == activations[0]["new_model_key"]
    assert activations[2]["new_model_key"] == activations[0]["new_model_key"]
    assert resolve_active_policy_checkpoint(store, checkpoints, tmp_path) == first


def test_candidate_registration_does_not_activate_policy(store, tmp_path: Path) -> None:
    checkpoints = tmp_path / "_outputs" / "checkpoints"
    active = _checkpoint(checkpoints / "whisper_dpo_active", 1)
    candidate = _checkpoint(checkpoints / "whisper_dpo_candidate", 2)
    record_active_policy_checkpoint(store, active, tmp_path)
    active_setting = store.get_setting("active_policy_checkpoint")
    activation_count = len(store.get_policy_activations())

    registered, previous, _ = register_policy_candidate(
        store, candidate, tmp_path, parent_checkpoint=active
    )

    assert registered["checkpoint_path"] == str(candidate.resolve())
    assert previous["checkpoint_path"] == str(active.resolve())
    assert store.get_setting("active_policy_checkpoint") == active_setting
    assert len(store.get_policy_activations()) == activation_count


def test_automatic_gated_publication_uses_supported_audit_reason(
    store, tmp_path: Path
) -> None:
    checkpoint = _checkpoint(
        tmp_path / "_outputs" / "checkpoints" / "whisper_dpo_candidate", 1
    )

    record_active_policy_checkpoint(
        store,
        checkpoint,
        tmp_path,
        reason="automatic",
        model_version="candidate-policy",
    )

    assert store.get_policy_activations()[0]["reason"] == "automatic"


def test_dpo_publication_records_sft_parent_and_rolls_queue_atomically(
    store, tmp_path: Path
) -> None:
    checkpoints = tmp_path / "_outputs" / "checkpoints"
    previous = _checkpoint(checkpoints / "whisper_dpo_previous", 1)
    sft = _checkpoint(checkpoints / "sft_vround", 2)
    dpo = _checkpoint(checkpoints / "whisper_dpo_new", 3)
    audio = make_audio(store, "audio/stale.wav")
    make_pair(store, audio.id)
    record_active_policy_checkpoint(store, previous, tmp_path)

    retired = record_active_policy_checkpoint(
        store,
        dpo,
        tmp_path,
        parent_checkpoint=sft,
        model_version="new-policy",
    )

    previous_row = store.get_model_checkpoint_by_path(str(previous.resolve()))
    sft_row = store.get_model_checkpoint_by_path(str(sft.resolve()))
    dpo_row = store.get_model_checkpoint_by_path(str(dpo.resolve()))
    assert sft_row["parent_checkpoint_id"] == previous_row["id"]
    assert dpo_row["parent_checkpoint_id"] == sft_row["id"]
    assert (
        _intermediate_sft_checkpoint_id(store, dpo_row, previous_row["id"])
        == sft_row["id"]
    )
    latest_activation = store.get_policy_activations()[0]
    assert latest_activation["previous_checkpoint_id"] == previous_row["id"]
    assert latest_activation["new_checkpoint_id"] == dpo_row["id"]
    assert retired == 1
    assert store.count_pending_pairs() == 0
    assert store.get_setting("queue_refill_requested") == "true"
    assert store.get_setting("queue_refill_status") == "queued"
