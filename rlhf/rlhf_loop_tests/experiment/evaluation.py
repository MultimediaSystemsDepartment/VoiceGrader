from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from evaluation.canary import discover_checkpoints
from evaluation.model_health import paired_bootstrap_corpus_wer
from paths import PathsConfig
from storage.feedback_store import FeedbackStore
from utils.policy_checkpoint import resolve_active_policy_checkpoint

from .runtime import subprocess_environment


def evaluate_experiment_round(
    manifest_path: Path,
    replicate: int,
    round_number: int,
    command_runner: Callable[..., object] = subprocess.run,
) -> dict:
    manifest = _read_json(manifest_path)
    if not any(
        run["replicate"] == replicate and run["round"] == round_number
        for run in manifest.get("training_runs", [])
    ):
        raise ValueError("Both training arms must complete before holdout evaluation")
    if any(
        run["replicate"] == replicate and run["round"] == round_number
        for run in manifest.get("evaluation_runs", [])
    ):
        raise ValueError("The requested holdout evaluation is already complete")

    project_root = Path(manifest["project_root"])
    config = manifest["config"]
    materialized = {
        item["arm"]: item
        for item in manifest["materialized_arms"]
        if item["replicate"] == replicate
    }
    arm_results = {}
    for arm in (
        item for item in manifest["arms"] if item["replicate"] == replicate
    ):
        holdout_id = materialized[arm["arm"]]["final_holdout"]["id"]
        arm_results[arm["arm"]] = _evaluate_arm(
            arm,
            holdout_id,
            config,
            project_root,
            command_runner,
        )

    seed = next(
        item["seed"]
        for item in manifest["replicates"]
        if item["replicate"] == replicate
    )
    comparisons = {
        "rlhf_minus_sft": compare_holdout_predictions(
            arm_results["sft"]["active_predictions"],
            arm_results["rlhf"]["active_predictions"],
            seed=seed,
        ),
        "rlhf_noisy_minus_sft": compare_holdout_predictions(
            arm_results["sft"]["active_predictions"],
            arm_results["rlhf_noisy"]["active_predictions"],
            seed=seed,
        ),
        "rlhf_noisy_minus_rlhf": compare_holdout_predictions(
            arm_results["rlhf"]["active_predictions"],
            arm_results["rlhf_noisy"]["active_predictions"],
            seed=seed,
        ),
    }
    initial_signatures = {
        _prediction_signature(result["initial_predictions"])
        for result in arm_results.values()
    }
    if len(initial_signatures) != 1:
        raise RuntimeError("Arm initial-policy holdout predictions differ")

    report = {
        "schema_version": 1,
        "replicate": replicate,
        "round": round_number,
        "arms": {
            name: {
                "initial_checkpoint_id": result["initial_checkpoint_id"],
                "active_checkpoint_id": result["active_checkpoint_id"],
                "initial_metrics": result["initial_metrics"],
                "active_metrics": result["active_metrics"],
                "wer_improvement_from_initial": (
                    result["active_metrics"]["wer"]
                    - result["initial_metrics"]["wer"]
                ),
            }
            for name, result in arm_results.items()
        },
        **comparisons,
    }
    report_path = manifest_path.parent / f"replicate-{replicate:02d}" / (
        f"round-{round_number:02d}-holdout.json"
    )
    _write_json_atomic(report_path, report)
    evaluation_run = {
        "replicate": replicate,
        "round": round_number,
        "report_file": str(report_path),
    }
    manifest.setdefault("evaluation_runs", []).append(evaluation_run)
    manifest["status"] = "evaluation-round-complete"
    _write_json_atomic(manifest_path, manifest)
    return report


def _evaluate_arm(
    arm: dict,
    holdout_id: str,
    config: dict,
    project_root: Path,
    command_runner: Callable[..., object],
) -> dict:
    raw_paths = _read_json(Path(arm["paths_file"]))
    paths = PathsConfig(
        **{
            key: value
            for key, value in raw_paths.items()
            if key in PathsConfig.__dataclass_fields__
        }
    )
    store = FeedbackStore(paths.db_path)
    checkpoints = discover_checkpoints(store, paths)
    active_path = resolve_active_policy_checkpoint(
        store, Path(paths.checkpoints_dir), project_root,
        base_model_id=paths.whisper_model_id,
    )
    active = _checkpoint_for_path(store, checkpoints, active_path)
    initial = _initial_checkpoint(
        store, checkpoints, config.get("initial_policy", "base")
    )
    environment = subprocess_environment(arm["paths_file"], config["gpu"])
    evaluations = {}
    for label, checkpoint in (("initial", initial), ("active", active)):
        command = [
            sys.executable,
            str(project_root / "run_evaluate_canary.py"),
            "--checkpoint-id",
            checkpoint["id"],
            "--canary-set-id",
            holdout_id,
        ]
        command_runner(command, cwd=project_root, env=environment, check=True)
        evaluation = next(
            (
                row
                for row in store.get_canary_evaluations(holdout_id)
                if row["checkpoint_id"] == checkpoint["id"]
                and row["status"] == "completed"
            ),
            None,
        )
        if evaluation is None:
            raise RuntimeError(f"{arm['arm']} {label} holdout evaluation did not complete")
        evaluations[label] = {
            "checkpoint_id": checkpoint["id"],
            "metrics": json.loads(evaluation["aggregate_json"]),
            "predictions": store.get_canary_predictions(evaluation["id"]),
        }
    return {
        "initial_checkpoint_id": evaluations["initial"]["checkpoint_id"],
        "active_checkpoint_id": evaluations["active"]["checkpoint_id"],
        "initial_metrics": evaluations["initial"]["metrics"],
        "active_metrics": evaluations["active"]["metrics"],
        "initial_predictions": evaluations["initial"]["predictions"],
        "active_predictions": evaluations["active"]["predictions"],
    }


def compare_holdout_predictions(
    sft_predictions: list[dict],
    rlhf_predictions: list[dict],
    seed: int,
) -> dict:
    sft = {row["audio_id"]: row for row in sft_predictions}
    rlhf = {row["audio_id"]: row for row in rlhf_predictions}
    if not sft or sft.keys() != rlhf.keys():
        raise ValueError("SFT and RLHF holdout predictions must contain the same audio")
    audio_ids = sorted(sft)
    references = [sft[audio_id]["transcript"] for audio_id in audio_ids]
    if references != [rlhf[audio_id]["transcript"] for audio_id in audio_ids]:
        raise ValueError("SFT and RLHF holdout references differ")
    comparison = paired_bootstrap_corpus_wer(
        references,
        [sft[audio_id]["hypothesis"] for audio_id in audio_ids],
        [rlhf[audio_id]["hypothesis"] for audio_id in audio_ids],
        seed=seed,
    )
    return {**comparison, "audio_ids": audio_ids}


def _checkpoint_for_path(
    store: FeedbackStore,
    checkpoints: list[dict],
    path: Path | None,
) -> dict:
    if path is None:
        return next(row for row in checkpoints if row["stage"] == "base")
    resolved = str(path.resolve())
    registered = store.get_model_checkpoint_by_path(resolved)
    if registered is not None:
        return registered
    return next(row for row in checkpoints if row.get("checkpoint_path") == resolved)


def _initial_checkpoint(
    store: FeedbackStore,
    checkpoints: list[dict],
    initial_policy: str,
) -> dict:
    if initial_policy == "base":
        return next(row for row in checkpoints if row["stage"] == "base")
    return _checkpoint_for_path(store, checkpoints, Path(initial_policy))


def _prediction_signature(predictions: list[dict]) -> list[tuple[str, str, str]]:
    return sorted(
        (row["audio_id"], row["transcript"], row["hypothesis"])
        for row in predictions
    )


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)
