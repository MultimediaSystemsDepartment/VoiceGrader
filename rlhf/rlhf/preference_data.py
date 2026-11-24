"""Build DPO rows from durable human choices and one ephemeral synthetic pool."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional


def human_preference_rows(store) -> list[dict]:
    rows = []
    preference_rows = (
        store.get_current_policy_preference_dataset()
        if hasattr(store, "get_current_policy_preference_dataset")
        else store.get_preference_dataset()
    )
    for row in preference_rows:
        if row["preference"] == "a":
            chosen, rejected = row["hyp_a"], row["hyp_b"]
        elif row["preference"] == "b":
            chosen, rejected = row["hyp_b"], row["hyp_a"]
        else:
            continue
        rows.append({
            "file_path": row["file_path"],
            "chosen_hyp": chosen,
            "rejected_hyp": rejected,
            "source": "human",
            "pair_id": row["pair_id"],
            "annotation_id": row["annotation_id"],
        })
    return rows


def load_synthetic_rows(
    path: Optional[str], expected_policy_version: Optional[str] = None
) -> list[dict]:
    if not path:
        return []
    artifact = Path(path)
    if not artifact.is_file():
        raise ValueError(f"Synthetic DPO artifact not found: {artifact}")

    rows = []
    required = {
        "artifact_version", "round_id", "policy_version", "reward_checkpoint",
        "file_path", "chosen_hyp", "rejected_hyp",
    }
    round_id = policy_version = reward_checkpoint = None
    with artifact.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            missing = required.difference(row)
            if missing:
                raise ValueError(
                    f"Synthetic artifact line {line_number} missing: "
                    f"{', '.join(sorted(missing))}"
                )
            if row["artifact_version"] != 1:
                raise ValueError(
                    f"Unsupported synthetic artifact version: {row['artifact_version']}"
                )
            identity = (
                row["round_id"], row["policy_version"], row["reward_checkpoint"]
            )
            if round_id is None:
                round_id, policy_version, reward_checkpoint = identity
                if (
                    expected_policy_version is not None
                    and policy_version != expected_policy_version
                ):
                    raise ValueError(
                        "Synthetic artifact policy does not match DPO start policy: "
                        f"{policy_version} != {expected_policy_version}"
                    )
            elif identity != (round_id, policy_version, reward_checkpoint):
                raise ValueError("Synthetic artifact mixes multiple round identities")
            rows.append({
                "file_path": row["file_path"],
                "chosen_hyp": row["chosen_hyp"],
                "rejected_hyp": row["rejected_hyp"],
                "source": "synthetic",
                "round_id": row["round_id"],
                "reward_gap": row.get("reward_gap"),
            })
    return rows


def build_dpo_rows(
    store,
    synthetic_path: Optional[str],
    max_synthetic_to_human_ratio: float,
    seed: int,
    expected_policy_version: Optional[str] = None,
) -> tuple[list[dict], int, int]:
    human = human_preference_rows(store)
    synthetic = load_synthetic_rows(synthetic_path, expected_policy_version)

    max_synthetic = int(len(human) * max(0.0, max_synthetic_to_human_ratio))
    if len(synthetic) > max_synthetic:
        synthetic = random.Random(seed).sample(synthetic, max_synthetic)

    combined = human + synthetic
    random.Random(seed).shuffle(combined)
    return combined, len(human), len(synthetic)