#!/usr/bin/env python
"""
run_rlhf.py -- DPO fine-tuning entry point.

Steps:
  1. Load RLHFConfig from JSON (or defaults)
  2. Evaluate Whisper baseline (WER before)
  3. Run WhisperRLHFTrainer.train()
  4. Evaluate fine-tuned model (WER after)
  5. Record TrainingRun in DB

Usage:
  py run_rlhf.py                                       # uses config/rlhf_config.json
  py run_rlhf.py --resume _outputs/checkpoints/sft
  py run_rlhf.py --skip_eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _parse_args():
    p = argparse.ArgumentParser(description="Run Whisper RLHF fine-tuning")
    p.add_argument(
        "--config", default="config/rlhf_config.json", help="Path to RLHFConfig JSON"
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Directory for LoRA checkpoints (default: configured checkpoints_dir)",
    )
    p.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip WER evaluation before/after training",
    )
    p.add_argument(
        "--max_eval_samples",
        type=int,
        default=int(os.environ.get("EVAL_MAX_SAMPLES", 0)) or None,
        help="Max test samples per split (default: EVAL_MAX_SAMPLES env var or None=full set)",
    )
    p.add_argument(
        "--resume",
        default=None,
        help="Path to LoRA checkpoint to resume from (overrides config)",
    )
    p.add_argument(
        "--reference",
        default=None,
        help="Frozen LoRA reference policy (default: same as --resume)",
    )
    p.add_argument(
        "--synthetic_data",
        default=None,
        help="Current-round ephemeral synthetic DPO JSONL artifact",
    )
    p.add_argument(
        "--epochs", type=int, default=None, help="Override DPO num_epochs from config"
    )
    p.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override DPO learning_rate from config",
    )
    p.add_argument(
        "--dpo_beta", type=float, default=None, help="Override DPO beta from config"
    )
    p.add_argument("--max_synthetic_ratio", type=float, default=None)
    p.add_argument("--max_grad_norm", type=float, default=None)
    p.add_argument("--early_stopping_patience", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    # -- Imports ----------------------------------------------------------------
    from paths import DEFAULT_PATHS
    from storage.feedback_store import FeedbackStore
    from storage.schema import TrainingRun
    from rlhf.config import RLHFConfig
    from rlhf.trainer import WhisperRLHFTrainer
    from evaluation.evaluator import WhisperEvaluator
    from utils.log import setup_logging
    from utils.policy_checkpoint import register_policy_candidate

    setup_logging(logs_dir=DEFAULT_PATHS.logs_dir)

    # -- Config -----------------------------------------------------------------
    cfg_path = Path(args.config)
    if cfg_path.exists():
        config = RLHFConfig.from_json(cfg_path)
    else:
        print(f"[warn] Config not found: {cfg_path}  -- using defaults")
        config = RLHFConfig()

    if args.resume:
        config.resume_from_checkpoint = args.resume
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.dpo_beta is not None:
        config.dpo_beta = args.dpo_beta
    if args.max_synthetic_ratio is not None:
        config.max_synthetic_to_human_ratio = args.max_synthetic_ratio
    if args.max_grad_norm is not None:
        config.max_grad_norm = args.max_grad_norm
    if args.early_stopping_patience is not None:
        config.early_stopping_patience = args.early_stopping_patience
    # -- Storage ----------------------------------------------------------------
    from utils.policy_checkpoint import validate_policy_model_id

    validate_policy_model_id(config.whisper_model_id, DEFAULT_PATHS.whisper_model_id)
    db_path = Path(DEFAULT_PATHS.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = FeedbackStore(str(db_path))
    if not config.resume_from_checkpoint:
        from utils.policy_checkpoint import resolve_active_policy_checkpoint

        active_policy = resolve_active_policy_checkpoint(
            store,
            Path(DEFAULT_PATHS.checkpoints_dir),
            Path(__file__).resolve().parent,
            base_model_id=config.whisper_model_id,
        )
        if active_policy is not None:
            config.resume_from_checkpoint = str(active_policy)
    if args.reference:
        config.reference_checkpoint = args.reference
    elif not config.reference_checkpoint:
        config.reference_checkpoint = config.resume_from_checkpoint

    policy_before = config.resume_from_checkpoint or config.whisper_model_id
    if config.reference_checkpoint != config.resume_from_checkpoint:
        raise ValueError("DPO reference must equal the round-start policy checkpoint")

    # -- Evaluate baseline ------------------------------------------------------
    wer_before_admed = wer_before_cv = None
    if not args.skip_eval:
        evaluator = WhisperEvaluator(
            config.whisper_model_id,
            DEFAULT_PATHS,
            store,
            device=config.device,
            torch_dtype=config.torch_dtype,
            lora_checkpoint=config.resume_from_checkpoint or None,
            max_samples=args.max_eval_samples,
            use_monitor=True,
        )
        _CACHE_KEY = "baseline_monitor_wer_cache" if evaluator.monitor else "baseline_wer_cache"
        subset_hash = evaluator.compute_subset_hash()
        cached_raw = store.get_setting(_CACHE_KEY)
        cached = json.loads(cached_raw) if cached_raw else {}
        if (
            cached.get("hash") == subset_hash
            and cached.get("model_id") == policy_before
        ):
            wer_before_admed = cached.get("wer_admed")
            wer_before_cv = cached.get("wer_cv")
            import logging as _logging

            _logging.getLogger(__name__).info(
                "Baseline WER reused from cache (subset hash unchanged)  "
                "Primary WER=%s",
                f"{wer_before_admed * 100:.2f}%" if wer_before_admed is not None else "N/A",
            )
        else:
            print("Evaluating baseline Whisper (before fine-tuning)...")
            before = evaluator.evaluate()
            wer_before_admed = before.wer_admed
            wer_before_cv = before.wer_cv
            print(before)
            store.set_setting(
                _CACHE_KEY,
                json.dumps(
                    {
                        "hash": subset_hash,
                        "model_id": policy_before,
                        "wer_admed": wer_before_admed,
                        "wer_cv": wer_before_cv,
                        "n_admed": before.n_admed,
                        "n_cv": before.n_cv,
                    }
                ),
            )

    # -- Record training run start ----------------------------------------------
    run = TrainingRun.new(
        algorithm=config.algorithm,
        whisper_version_before=policy_before,
        n_human_annotations=store.count_human_annotations(),
        n_synthetic_annotations=0,
        config_json=config.to_json_str(),
    )
    store.start_training_run(run)
    if not args.skip_eval:
        store.update_training_run_wer_before(
            run.id,
            wer_admed_before=wer_before_admed,
            wer_cv_before=wer_before_cv,
        )

    # -- Fine-tune --------------------------------------------------------------
    output_dir = Path(args.output_dir or DEFAULT_PATHS.checkpoints_dir)
    trainer = WhisperRLHFTrainer(config, store, DEFAULT_PATHS)
    try:
        ckpt_dir = trainer.train(output_dir, synthetic_data=args.synthetic_data)
    except ValueError as exc:
        import logging as _log

        _log.getLogger(__name__).warning(
            "DPO training skipped (cold start / insufficient data): %s", exc
        )
        store.complete_training_run(
            run.id,
            whisper_version_after=None,
            wer_admed_after=None,
            wer_cv_after=None,
        )
        print(f"[SKIP] DPO: {exc}")
        return 2

    store.update_training_run_pair_counts(
        run.id, trainer.n_human_pairs, trainer.n_synthetic_pairs
    )
    print(f"Checkpoint saved: {ckpt_dir}")

    # -- Evaluate fine-tuned model ----------------------------------------------
    wer_after_admed = wer_after_cv = None
    if not args.skip_eval:
        print("Evaluating fine-tuned model (after training)...")
        evaluator_after = WhisperEvaluator(
            config.whisper_model_id,
            DEFAULT_PATHS,
            store,
            device=config.device,
            torch_dtype=config.torch_dtype,
            lora_checkpoint=str(ckpt_dir),
            max_samples=args.max_eval_samples,
            use_monitor=True,
        )
        # Keep the exact frozen reference snapshot used before training.
        evaluator_after.monitor = evaluator.monitor
        after = evaluator_after.evaluate()
        wer_after_admed = after.wer_admed
        wer_after_cv = after.wer_cv
        print(after)

    # -- Finalise training run --------------------------------------------------
    store.complete_training_run(
        run.id,
        whisper_version_after=str(ckpt_dir),
        wer_admed_after=wer_after_admed,
        wer_cv_after=wer_after_cv,
    )
    model_version = f"{config.whisper_model_id}+lora:{ckpt_dir.name}"
    candidate, previous, _ = register_policy_candidate(
        store,
        ckpt_dir,
        Path(__file__).resolve().parent,
        parent_checkpoint=(
            Path(config.resume_from_checkpoint)
            if config.resume_from_checkpoint
            else None
        ),
    )
    if candidate is None:
        raise RuntimeError("Persistent checkpoint registry is required for promotion")
    store.set_setting("pending_policy_checkpoint_id", candidate["id"])
    store.set_setting(
        "pending_policy_previous_checkpoint_id", previous["id"] if previous else ""
    )
    store.set_setting("pending_policy_model_version", model_version)
    store.set_setting("promotion_gate_status", "pending")
    print("Training run and pending policy candidate recorded in DB.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
