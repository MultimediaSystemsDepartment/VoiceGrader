from __future__ import annotations

from typing import Sequence


def prediction_difference(
    previous_predictions: Sequence[dict],
    current_predictions: Sequence[dict],
) -> dict[str, int | set[str]]:
    previous = {
        row["canary_item_id"]: row["hypothesis"] for row in previous_predictions
    }
    current = {row["canary_item_id"]: row["hypothesis"] for row in current_predictions}
    shared_ids = previous.keys() & current.keys()
    changed_ids = {
        item_id for item_id in shared_ids if previous[item_id] != current[item_id]
    }
    return {
        "changed": len(changed_ids),
        "total": len(shared_ids),
        "changed_ids": changed_ids,
    }
