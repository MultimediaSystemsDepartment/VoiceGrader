from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from .evaluation import evaluate_experiment_round
from .initialization import initialize_experiment
from .materialization import materialize_arm_states
from .reporting import build_experiment_report
from .rounds import run_annotation_round
from .training import train_experiment_round


def run_experiment_to_completion(
    manifest_path: Path,
    initialize: Callable[[Path], dict] = initialize_experiment,
    materialize: Callable[[Path], dict] = materialize_arm_states,
    annotate: Callable[[Path, int, int], dict] = run_annotation_round,
    train: Callable[[Path, int, int], dict] = train_experiment_round,
    evaluate: Callable[[Path, int, int], dict] = evaluate_experiment_round,
    report: Callable[[Path], dict] = build_experiment_report,
) -> dict:
    manifest = _read_json(manifest_path)
    if manifest.get("status") == "complete" and manifest.get("report"):
        return _read_json(Path(manifest["report"]["json_file"]))
    if manifest.get("status") == "prepared":
        manifest = initialize(manifest_path)
    elif manifest.get("status") == "partitions-planned":
        manifest = materialize(manifest_path)
    if not manifest.get("materialized_arms"):
        raise ValueError("Experiment initialization is incomplete")

    config = manifest["config"]
    for replicate in range(1, len(config["seeds"]) + 1):
        for round_number in range(1, config["rounds"] + 1):
            manifest = _read_json(manifest_path)
            if not _completed(manifest, "round_runs", replicate, round_number):
                annotate(manifest_path, replicate, round_number)
            manifest = _read_json(manifest_path)
            if not _completed(manifest, "training_runs", replicate, round_number):
                train(manifest_path, replicate, round_number)
            manifest = _read_json(manifest_path)
            if not _completed(manifest, "evaluation_runs", replicate, round_number):
                evaluate(manifest_path, replicate, round_number)
    return report(manifest_path)


def _completed(
    manifest: dict, collection: str, replicate: int, round_number: int
) -> bool:
    return any(
        item["replicate"] == replicate and item["round"] == round_number
        for item in manifest.get(collection, [])
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))