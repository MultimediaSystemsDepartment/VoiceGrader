from __future__ import annotations

import json
import random
from pathlib import Path

from .config import ExperimentConfig


def build_partition_schedule(
    audio_ids: list[str],
    config: ExperimentConfig,
    seed: int,
    audio_groups: dict[str, str] | None = None,
) -> dict:
    unique_ids = list(dict.fromkeys(audio_ids))
    required = (
        config.interaction_samples
        + config.monitor_samples
        + config.holdout_samples
    )
    if len(unique_ids) < required:
        raise ValueError(
            f"Only {len(unique_ids)} GT audio files available; {required} required"
        )
    rng = random.Random(seed)
    if audio_groups is None:
        audio_groups = {audio_id: audio_id for audio_id in unique_ids}
    missing_groups = [audio_id for audio_id in unique_ids if audio_id not in audio_groups]
    if missing_groups:
        raise ValueError(f"Missing partition groups for {len(missing_groups)} audio files")
    grouped: dict[str, list[str]] = {}
    for audio_id in unique_ids:
        grouped.setdefault(audio_groups[audio_id], []).append(audio_id)
    group_names = list(grouped)
    rng.shuffle(group_names)
    for values in grouped.values():
        rng.shuffle(values)

    def take_partition(size: int, name: str) -> tuple[list[str], list[str]]:
        selected: list[str] = []
        selected_groups: list[str] = []
        while len(selected) < size and group_names:
            group_name = group_names.pop()
            selected_groups.append(group_name)
            selected.extend(grouped[group_name][: size - len(selected)])
        if len(selected) != size:
            raise ValueError(
                f"Session-disjoint {name} partition requires {size} samples; "
                f"only {len(selected)} fit after assigning earlier partitions"
            )
        return selected, selected_groups

    interaction, interaction_groups = take_partition(
        config.interaction_samples, "interaction"
    )
    monitor, monitor_groups = take_partition(config.monitor_samples, "monitor")
    holdout, holdout_groups = take_partition(config.holdout_samples, "holdout")
    round_size = config.annotations_per_round
    rounds = [
        interaction[index * round_size : (index + 1) * round_size]
        for index in range(config.rounds)
    ]
    return {
        "schema_version": 1,
        "seed": seed,
        "interaction_audio_ids": interaction,
        "promotion_monitor_audio_ids": monitor,
        "final_holdout_audio_ids": holdout,
        "partition_groups": {
            "interaction": interaction_groups,
            "promotion_monitor": monitor_groups,
            "final_holdout": holdout_groups,
        },
        "rounds": rounds,
    }


def write_partition_schedule(path: Path, schedule: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(schedule, indent=2), encoding="utf-8")
    temporary.replace(path)
