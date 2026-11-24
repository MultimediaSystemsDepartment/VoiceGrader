from __future__ import annotations

import json
from pathlib import Path

from storage.feedback_store import FeedbackStore

from .config import ExperimentConfig
from .schedule import build_partition_schedule, write_partition_schedule


def plan_partitions(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = ExperimentConfig(**manifest["config"])
    config.validate()
    initial_paths = json.loads(
        Path(manifest["initial"]["paths_file"]).read_text(encoding="utf-8")
    )
    store = FeedbackStore(initial_paths["db_path"])
    audio_rows = store.get_ground_truth_audio_files()
    audio_ids = [audio.id for audio in audio_rows]
    audio_groups = {audio.id: _parent_group(audio.file_path) for audio in audio_rows}
    experiment_root = manifest_path.parent
    replicates = []
    for replicate, seed in enumerate(config.seeds, 1):
        replicate_root = experiment_root / f"replicate-{replicate:02d}"
        schedule = build_partition_schedule(
            audio_ids, config, seed, audio_groups=audio_groups
        )
        schedule_path = replicate_root / "partitions.json"
        write_partition_schedule(schedule_path, schedule)
        round_files = []
        for round_number, round_ids in enumerate(schedule["rounds"], 1):
            round_path = replicate_root / f"round-{round_number:02d}-audio-ids.json"
            write_partition_schedule(round_path, round_ids)
            round_files.append(str(round_path))
        replicates.append(
            {
                "replicate": replicate,
                "seed": seed,
                "partitions_file": str(schedule_path),
                "round_audio_id_files": round_files,
            }
        )
    manifest["status"] = "partitions-planned"
    manifest["replicates"] = replicates
    _write_manifest(manifest_path, manifest)
    return manifest


def _parent_group(file_path: str) -> str:
    normalized = file_path.replace("\\", "/")
    member = normalized.split("!", 1)[1] if "!" in normalized else normalized
    parent, separator, _ = member.rpartition("/")
    if not separator or not parent:
        raise ValueError(f"Cannot derive recording-session group from {file_path!r}")
    return parent


def _write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(path)
