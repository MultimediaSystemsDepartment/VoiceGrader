from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from storage.feedback_store import FeedbackStore
from utils.policy_checkpoint import (
    record_active_policy_checkpoint,
    resolve_active_policy_checkpoint,
)
from utils.training_tuning import TrainingTuning

from .runtime import subprocess_environment


def train_experiment_round(
    manifest_path: Path,
    replicate: int,
    round_number: int,
    command_runner: Callable[..., object] = subprocess.run,
) -> dict:
    manifest = _read_json(manifest_path)
    annotation_run = next(
        (
            run
            for run in manifest.get("round_runs", [])
            if run["replicate"] == replicate and run["round"] == round_number
        ),
        None,
    )
    if annotation_run is None:
        raise ValueError("Matched annotations must complete before training")
    if any(
        run["replicate"] == replicate and run["round"] == round_number
        for run in manifest.get("training_runs", [])
    ):
        raise ValueError("The requested training round is already complete")

    project_root = Path(manifest["project_root"])
    arms = [arm for arm in manifest["arms"] if arm["replicate"] == replicate]
    results = []
    for arm in arms:
        if arm["arm"] == "sft":
            result = _train_sft_control(
                arm,
                project_root,
                round_number,
                manifest["config"],
                command_runner,
            )
        elif arm["arm"] in {"rlhf", "rlhf_noisy"}:
            result = _train_rlhf_proposal(
                arm,
                project_root,
                round_number,
                manifest["config"],
                command_runner,
            )
        else:
            raise ValueError(f"Unknown experiment arm: {arm['arm']}")
        results.append(result)

    training_run = {
        "replicate": replicate,
        "round": round_number,
        "annotation_parity_file": annotation_run["parity_file"],
        "arms": results,
    }
    manifest.setdefault("training_runs", []).append(training_run)
    manifest["status"] = "training-round-complete"
    _write_json_atomic(manifest_path, manifest)
    return training_run


def _train_sft_control(
    arm: dict,
    project_root: Path,
    round_number: int,
    config: dict,
    command_runner: Callable[..., object],
) -> dict:
    state_path = Path(arm["root"]) / f"round-{round_number:02d}" / "training.json"
    state = _load_training_state(state_path, arm, round_number)
    if state.get("status") == "completed":
        return state["result"]
    if state.get("status") == "running":
        raise RuntimeError(f"Incomplete SFT training stage: {state_path}")

    paths = _read_json(Path(arm["paths_file"]))
    store = FeedbackStore(paths["db_path"])
    tuning = TrainingTuning.from_store(store)
    starting_policy = resolve_active_policy_checkpoint(
        store, Path(paths["checkpoints_dir"]), project_root,
        base_model_id=paths["whisper_model_id"],
    )
    output_dir = (
        Path(paths["checkpoints_dir"])
        / f"sft_control_r{arm['replicate']:02d}_round{round_number:02d}"
    )
    command = [
        sys.executable,
        str(project_root / "run_sft.py"),
        "--epochs",
        str(tuning.sft_epochs),
        "--learning_rate",
        str(tuning.sft_learning_rate),
        "--warmup_steps",
        str(tuning.sft_warmup_steps),
        "--max_grad_norm",
        str(tuning.max_grad_norm),
        "--early_stopping_patience",
        str(tuning.early_stopping_patience),
        "--output_dir",
        str(output_dir),
    ]
    if starting_policy is not None:
        command.extend(("--resume", str(starting_policy)))
    elif config.get("initial_policy", "base") != "base":
        command.extend(("--resume", config["initial_policy"]))

    _write_json_atomic(state_path, {**state, "status": "running", "command": command})
    environment = subprocess_environment(arm["paths_file"], config["gpu"])
    command_runner(command, cwd=project_root, env=environment, check=True)
    if not (output_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"SFT control checkpoint is incomplete: {output_dir}")
    record_active_policy_checkpoint(
        store,
        output_dir,
        project_root,
        reason="manual",
        actor="rlhf-loop-experiment",
        parent_checkpoint=starting_policy,
        base_model_id=paths["whisper_model_id"],
    )
    result = {
        "arm": "sft",
        "outcome": "activated",
        "starting_checkpoint": str(starting_policy) if starting_policy else None,
        "active_checkpoint": str(output_dir.resolve()),
        "diagnostics": _setting_json(store, "training_sft_diagnostics"),
    }
    _write_json_atomic(state_path, {**state, "status": "completed", "result": result})
    return result


def _train_rlhf_proposal(
    arm: dict,
    project_root: Path,
    round_number: int,
    config: dict,
    command_runner: Callable[..., object],
) -> dict:
    state_path = Path(arm["root"]) / f"round-{round_number:02d}" / "training.json"
    state = _load_training_state(state_path, arm, round_number)
    if state.get("status") == "completed":
        return state["result"]
    if state.get("status") == "running":
        raise RuntimeError(f"Incomplete RLHF training stage: {state_path}")

    paths = _read_json(Path(arm["paths_file"]))
    store = FeedbackStore(paths["db_path"])
    starting_policy = resolve_active_policy_checkpoint(
        store, Path(paths["checkpoints_dir"]), project_root,
        base_model_id=paths["whisper_model_id"],
    )
    command = [
        sys.executable,
        str(project_root / "scripts" / "coordinator.py"),
        "--once",
    ]
    _write_json_atomic(state_path, {**state, "status": "running", "command": command})
    environment = subprocess_environment(arm["paths_file"], config["gpu"])
    command_runner(command, cwd=project_root, env=environment, check=True)
    training_status = store.get_setting("training_status", "")
    if training_status not in {"idle", "guarded"}:
        raise RuntimeError(
            f"Canonical RLHF pipeline ended with status {training_status!r}"
        )
    active_policy = resolve_active_policy_checkpoint(
        store, Path(paths["checkpoints_dir"]), project_root,
        base_model_id=paths["whisper_model_id"],
    )
    result = {
        "arm": arm["arm"],
        "outcome": "guarded" if training_status == "guarded" else "activated",
        "starting_checkpoint": str(starting_policy) if starting_policy else None,
        "active_checkpoint": str(active_policy) if active_policy else None,
        "sft_diagnostics": _setting_json(store, "training_sft_diagnostics"),
        "dpo_diagnostics": _setting_json(store, "training_dpo_diagnostics"),
        "promotion_gate_status": store.get_setting("promotion_gate_status", ""),
    }
    _write_json_atomic(state_path, {**state, "status": "completed", "result": result})
    return result


def _load_training_state(path: Path, arm: dict, round_number: int) -> dict:
    expected = {
        "schema_version": 1,
        "replicate": arm["replicate"],
        "arm": arm["arm"],
        "round": round_number,
    }
    state = _read_json(path) if path.exists() else {**expected, "status": "pending"}
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"Training state mismatch for {arm['arm']}: {key}")
    return state


def _setting_json(store: FeedbackStore, key: str):
    value = store.get_setting(key, "")
    return json.loads(value) if value else None


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)
