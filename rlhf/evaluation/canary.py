"""Checkpoint discovery and resumable deterministic canary evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from evaluation.evaluator import WhisperEvaluator
from evaluation.model_health import (
    CANARY_DECODE_PROFILE,
    aggregate_metrics,
    classify_regression,
    compute_clip_metrics,
    paired_bootstrap_delta,
    paired_corpus_wer_from_predictions,
    profile_hash,
)
from storage.schema import CanaryEvaluation, CanaryPrediction, ModelCheckpoint
from utils.log import get_logger

log = get_logger(__name__)


def evaluation_profile_hash(canary_set: dict) -> str:
    """Bind decode settings to the exact frozen transcript revision."""
    return profile_hash(
        {
            "decode": CANARY_DECODE_PROFILE,
            "monitor_fingerprint": canary_set["fingerprint"],
        }
    )


def discover_checkpoints(store, paths) -> list[dict]:
    """Register the base model and recognizable local adapter checkpoints."""
    discovered = [
        store.register_model_checkpoint(
            ModelCheckpoint.new(
                model_key=f"base:{paths.whisper_model_id}",
                stage="base",
                provenance="discovered",
                metadata_json=json.dumps({"model_id": paths.whisper_model_id}),
            )
        )
    ]
    checkpoints_dir = Path(paths.checkpoints_dir)
    if not checkpoints_dir.exists():
        return discovered

    for checkpoint_path in sorted(checkpoints_dir.iterdir()):
        if not checkpoint_path.is_dir():
            continue
        if not (checkpoint_path / "adapter_config.json").exists():
            continue
        if checkpoint_path.name == "sft" or checkpoint_path.name.startswith("sft_"):
            stage = "sft"
        elif checkpoint_path.name.startswith("whisper_dpo_"):
            stage = "dpo"
        else:
            continue
        resolved_path = str(checkpoint_path.resolve())
        discovered.append(
            store.register_model_checkpoint(
                ModelCheckpoint.new(
                    model_key=f"{stage}:{resolved_path}",
                    stage=stage,
                    checkpoint_path=resolved_path,
                    provenance="discovered",
                )
            )
        )
    return discovered


class CanaryBackfill:
    def __init__(
        self,
        store,
        paths,
        device: str = "cuda",
        torch_dtype: str = "float16",
        batch_size: int = 1,
        evaluator_factory: Callable[..., WhisperEvaluator] = WhisperEvaluator,
    ) -> None:
        self.store = store
        self.paths = paths
        self.device = device
        self.torch_dtype = torch_dtype
        self.batch_size = batch_size
        self.evaluator_factory = evaluator_factory

    def run(
        self,
        checkpoint_id: Optional[str] = None,
        canary_set_id: Optional[str] = None,
    ) -> list[dict]:
        canary_set = (
            self.store.get_canary_set(canary_set_id)
            if canary_set_id
            else self.store.get_promotion_gate_canary_set()
        )
        if not canary_set or canary_set["status"] != "ready":
            raise ValueError("A frozen canary set is required before evaluation")
        items = self.store.get_canary_items(canary_set["id"])
        if not items or any(not item["transcript"] for item in items):
            raise ValueError("The frozen canary set has incomplete transcripts")

        checkpoints = discover_checkpoints(self.store, self.paths)
        if checkpoint_id:
            checkpoints = [row for row in checkpoints if row["id"] == checkpoint_id]
            if not checkpoints:
                raise ValueError(f"Unknown or unavailable checkpoint: {checkpoint_id}")
        checkpoints.sort(key=lambda row: (row["stage"] != "base", row["created_at"]))

        current_profile_hash = evaluation_profile_hash(canary_set)
        baseline = self._completed_base_context(canary_set["id"], current_profile_hash)
        results = []
        for checkpoint in checkpoints:
            evaluation = self.store.get_or_create_canary_evaluation(
                CanaryEvaluation.new(
                    canary_set_id=canary_set["id"],
                    checkpoint_id=checkpoint["id"],
                    profile_hash=current_profile_hash,
                )
            )
            if evaluation["status"] == "completed":
                aggregate = json.loads(evaluation["aggregate_json"])
                if checkpoint["stage"] == "base":
                    baseline = self._baseline_context(evaluation["id"], aggregate)
                results.append(evaluation)
                continue
            result, aggregate = self._evaluate(checkpoint, evaluation, items, baseline)
            if result is None:
                results.append(evaluation)
                break
            if checkpoint["stage"] == "base" and aggregate is not None:
                baseline = self._baseline_context(result["id"], aggregate)
            results.append(result)
        return results

    def _completed_base_context(
        self, canary_set_id: str, current_profile_hash: str
    ) -> Optional[dict]:
        for evaluation in self.store.get_canary_evaluations(canary_set_id):
            if (
                evaluation["stage"] == "base"
                and evaluation["status"] == "completed"
                and evaluation["profile_hash"] == current_profile_hash
            ):
                return self._baseline_context(
                    evaluation["id"], json.loads(evaluation["aggregate_json"])
                )
        return None

    def _baseline_context(self, evaluation_id: str, aggregate: dict) -> dict:
        predictions = self.store.get_canary_predictions(evaluation_id)
        return {
            "aggregate": aggregate,
            "clips": {
                row["canary_item_id"]: json.loads(row["metrics_json"])
                for row in predictions
            },
            "predictions": predictions,
        }

    def _evaluate(self, checkpoint, evaluation, items, baseline):
        evaluation_id = evaluation["id"]
        if not self.store.start_canary_evaluation(evaluation_id):
            return None, None
        references = [item["transcript"] for item in items]
        hypotheses: list[str] = []
        clip_metrics: dict[str, dict] = {}

        def persist_batch(start: int, batch_hypotheses: list[str]) -> None:
            for offset, hypothesis in enumerate(batch_hypotheses):
                item = items[start + offset]
                metrics = compute_clip_metrics(item["transcript"], hypothesis)
                clip_metrics[item["id"]] = metrics
                prediction = CanaryPrediction.new(
                    evaluation_id=evaluation_id,
                    canary_item_id=item["id"],
                    hypothesis=hypothesis,
                    metrics_json=json.dumps(metrics, sort_keys=True),
                )
                self.store.insert_canary_prediction(
                    prediction.id,
                    prediction.evaluation_id,
                    prediction.canary_item_id,
                    prediction.hypothesis,
                    prediction.metrics_json,
                )

        try:
            evaluator = self.evaluator_factory(
                model_id=self.paths.whisper_model_id,
                paths=self.paths,
                store=self.store,
                device=self.device,
                torch_dtype=self.torch_dtype,
                lora_checkpoint=checkpoint["checkpoint_path"],
                decode_num_beams=CANARY_DECODE_PROFILE["num_beams"],
            )
            hypotheses = evaluator.transcribe_files(
                [item["file_path"] for item in items],
                batch_size=self.batch_size,
                on_batch=persist_batch,
            )
            aggregate = aggregate_metrics(references, hypotheses)
            if baseline is not None and checkpoint["stage"] != "base":
                paired_ids = [
                    item["id"] for item in items if item["id"] in baseline["clips"]
                ]
                aggregate["vs_base"] = {
                    "wer": paired_corpus_wer_from_predictions(
                        baseline["predictions"],
                        self.store.get_canary_predictions(evaluation_id),
                    ),
                    "cer": paired_bootstrap_delta(
                        [baseline["clips"][item_id]["cer"] for item_id in paired_ids],
                        [clip_metrics[item_id]["cer"] for item_id in paired_ids],
                    ),
                }
            warnings = (
                classify_regression(baseline["aggregate"], aggregate)
                if baseline is not None and checkpoint["stage"] != "base"
                else []
            )
            health_status = "warning" if warnings else "healthy"
            warning_json = json.dumps(warnings) if warnings else None
            self.store.complete_canary_evaluation(
                evaluation_id,
                json.dumps(aggregate, sort_keys=True),
                health_status,
                warning_json,
            )
            self.store.update_checkpoint_health(checkpoint["id"], health_status)
            completed = self.store.get_or_create_canary_evaluation(
                CanaryEvaluation.new(
                    canary_set_id=evaluation["canary_set_id"],
                    checkpoint_id=checkpoint["id"],
                    profile_hash=evaluation["profile_hash"],
                )
            )
            return completed, aggregate
        except Exception as exc:
            log.exception("Canary evaluation failed for %s", checkpoint["model_key"])
            self.store.fail_canary_evaluation(evaluation_id, str(exc)[:1000])
            failed = self.store.get_or_create_canary_evaluation(
                CanaryEvaluation.new(
                    canary_set_id=evaluation["canary_set_id"],
                    checkpoint_id=checkpoint["id"],
                    profile_hash=evaluation["profile_hash"],
                )
            )
            return failed, None
