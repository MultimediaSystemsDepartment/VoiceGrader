from __future__ import annotations

import json
from pathlib import Path

from storage.feedback_store import FeedbackStore
from storage.schema import CanarySet
from utils.policy_checkpoint import record_active_policy_checkpoint


def materialize_arm_states(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "partitions-planned":
        raise ValueError("Partitions must be planned before arm materialization")

    initial_paths = _load_paths(manifest["initial"]["paths_file"])
    initial_store = FeedbackStore(initial_paths["db_path"])
    initial_policy = manifest["config"].get("initial_policy", "base")
    schedules = {
        item["replicate"]: json.loads(
            Path(item["partitions_file"]).read_text(encoding="utf-8")
        )
        for item in manifest["replicates"]
    }
    materialized_arms = []
    fingerprints_by_replicate: dict[int, set[tuple[str, str]]] = {}
    for arm in manifest["arms"]:
        schedule = schedules[arm["replicate"]]
        arm_paths = _load_paths(arm["paths_file"])
        target_db = Path(arm_paths["db_path"])
        staging_db = target_db.with_suffix(target_db.suffix + ".materializing")
        if staging_db.exists():
            raise FileExistsError(
                f"Incomplete arm materialization exists: {staging_db}"
            )
        initial_store.clone_to(staging_db)
        try:
            staging_store = FeedbackStore(staging_db)
            if initial_policy != "base":
                checkpoint = Path(initial_policy)
                if not (checkpoint / "adapter_config.json").is_file():
                    raise FileNotFoundError(
                        f"Invalid initial policy checkpoint: {checkpoint}"
                    )
                record_active_policy_checkpoint(
                    staging_store,
                    checkpoint,
                    Path(manifest["project_root"]),
                    reason="manual",
                    actor="rlhf-loop-experiment",
                    base_model_id=arm_paths["whisper_model_id"],
                )
            else:
                staging_store.set_setting(
                    "active_policy_model_version", arm_paths["whisper_model_id"]
                )
            holdout = _create_frozen_set(
                staging_store,
                name="Experiment final holdout",
                seed=arm["seed"],
                selection_method="experiment_explicit_final_holdout",
                audio_ids=schedule["final_holdout_audio_ids"],
            )
            promotion = _create_frozen_set(
                staging_store,
                name="Experiment promotion monitor",
                seed=arm["seed"],
                selection_method="experiment_explicit_promotion_monitor",
                audio_ids=schedule["promotion_monitor_audio_ids"],
            )
            staging_store.set_promotion_gate_canary_set(promotion["id"])
            staging_store.clone_to(target_db)
        finally:
            staging_db.unlink(missing_ok=True)

        fingerprints = (promotion["fingerprint"], holdout["fingerprint"])
        fingerprints_by_replicate.setdefault(arm["replicate"], set()).add(
            fingerprints
        )
        materialized_arms.append(
            {
                "replicate": arm["replicate"],
                "arm": arm["arm"],
                "db_path": str(target_db),
                "promotion_monitor": promotion,
                "final_holdout": holdout,
            }
        )

    mismatched = [
        replicate
        for replicate, fingerprints in fingerprints_by_replicate.items()
        if len(fingerprints) != 1
    ]
    if mismatched:
        raise RuntimeError(f"Arm monitor fingerprints differ: {mismatched}")
    manifest["status"] = "arms-materialized"
    manifest["materialized_arms"] = materialized_arms
    _write_manifest(manifest_path, manifest)
    return manifest


def _create_frozen_set(
    store: FeedbackStore,
    name: str,
    seed: int,
    selection_method: str,
    audio_ids: list[str],
) -> dict[str, str]:
    canary_set = CanarySet.new(
        name=name,
        selection_seed=seed,
        selection_method=selection_method,
        target_size=len(audio_ids),
    )
    store.create_canary_set(canary_set, audio_ids=audio_ids)
    copied = store.bulk_copy_canary_ground_truth(canary_set.id)
    if copied["copied"] != len(audio_ids) or copied["missing_gt"]:
        raise ValueError(f"Ground truth is incomplete for {name}")
    fingerprint = store.freeze_canary_set(canary_set.id)
    return {"id": canary_set.id, "fingerprint": fingerprint}


def _load_paths(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(path)
