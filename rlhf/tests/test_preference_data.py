from __future__ import annotations

import json

import pytest

from rlhf.preference_data import build_dpo_rows, human_preference_rows, load_synthetic_rows


class FakeStore:
    def get_preference_dataset(self):
        return [{
            "annotation_id": "annotation-1",
            "pair_id": "pair-1",
            "preference": "a",
            "hyp_a": "human winner",
            "hyp_b": "reward model winner",
            "file_path": "audio.wav",
        }]


def _synthetic_row(index: int, round_id: str = "round-1") -> dict:
    return {
        "artifact_version": 1,
        "round_id": round_id,
        "policy_version": "policy-1",
        "reward_checkpoint": "rm-1",
        "file_path": f"audio-{index}.wav",
        "chosen_hyp": f"chosen-{index}",
        "rejected_hyp": f"rejected-{index}",
        "reward_gap": 0.5,
    }


def test_human_choice_is_authoritative() -> None:
    rows = human_preference_rows(FakeStore())

    assert rows[0]["chosen_hyp"] == "human winner"
    assert rows[0]["rejected_hyp"] == "reward model winner"


def test_synthetic_artifact_rejects_mixed_rounds(tmp_path) -> None:
    artifact = tmp_path / "synthetic.jsonl"
    artifact.write_text(
        "\n".join(json.dumps(row) for row in [
            _synthetic_row(1, "round-1"),
            _synthetic_row(2, "round-2"),
        ]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mixes multiple round identities"):
        load_synthetic_rows(str(artifact))


def test_synthetic_artifact_rejects_wrong_policy(tmp_path) -> None:
    artifact = tmp_path / "synthetic.jsonl"
    artifact.write_text(json.dumps(_synthetic_row(1)), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match DPO start policy"):
        load_synthetic_rows(str(artifact), "another-policy")


def test_synthetic_mix_is_capped_and_deterministic(tmp_path) -> None:
    artifact = tmp_path / "synthetic.jsonl"
    artifact.write_text(
        "\n".join(json.dumps(_synthetic_row(index)) for index in range(20)),
        encoding="utf-8",
    )

    first = build_dpo_rows(FakeStore(), str(artifact), 3.0, seed=7)
    second = build_dpo_rows(FakeStore(), str(artifact), 3.0, seed=7)

    assert first == second
    assert first[1:] == (1, 3)