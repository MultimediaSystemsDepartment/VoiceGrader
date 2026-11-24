#!/usr/bin/env python
"""
run_sft.py -- Supervised fine-tuning of Whisper on human-corrected transcripts.

Pulls all annotations where the annotator typed a corrected transcript, and
trains Whisper directly on those (audio, correction) pairs using standard
seq2seq cross-entropy loss (teacher forcing).

Run this before DPO to update the active policy from human corrections. Without
--resume, the latest active LoRA policy is selected automatically.

Usage:
    py run_sft.py                          # uses config/rlhf_config.json
    py run_sft.py --config my_cfg.json
    py run_sft.py --output_dir _outputs/checkpoints/sft_round1
    py run_sft.py --epochs 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _parse_args():
    p = argparse.ArgumentParser(description="SFT fine-tuning on human corrections")
    p.add_argument(
        "--config",
        default="config/rlhf_config.json",
        help="Path to RLHFConfig JSON (default: config/rlhf_config.json)",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Directory for LoRA SFT checkpoint",
    )
    p.add_argument(
        "--epochs", type=int, default=None, help="Override num_epochs from config"
    )
    p.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning_rate from config",
    )
    p.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="Override the adaptive warmup cap from config",
    )
    p.add_argument("--max_grad_norm", type=float, default=None)
    p.add_argument("--early_stopping_patience", type=int, default=None)
    p.add_argument(
        "--resume", default=None, help="Path to existing LoRA checkpoint to resume from"
    )
    return p.parse_args()


def main():
    args = _parse_args()

    from paths import DEFAULT_PATHS
    from storage.feedback_store import FeedbackStore
    from rlhf.config import RLHFConfig
    from rlhf.sft_trainer import WhisperSFTTrainer
    from utils.log import setup_logging, get_logger
    from utils.policy_checkpoint import resolve_active_policy_checkpoint, validate_policy_model_id

    setup_logging(logs_dir=DEFAULT_PATHS.logs_dir)
    log = get_logger(__name__)
    log.info("=== run_sft ===")

    # -- Config -----------------------------------------------------------------
    cfg_path = Path(args.config)
    config = RLHFConfig.from_json(cfg_path) if cfg_path.exists() else RLHFConfig()

    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.warmup_steps is not None:
        config.warmup_steps = args.warmup_steps
    if args.max_grad_norm is not None:
        config.max_grad_norm = args.max_grad_norm
    if args.early_stopping_patience is not None:
        config.early_stopping_patience = args.early_stopping_patience
    if args.resume:
        config.resume_from_checkpoint = args.resume

    # -- Storage ----------------------------------------------------------------
    validate_policy_model_id(config.whisper_model_id, DEFAULT_PATHS.whisper_model_id)
    store = FeedbackStore(DEFAULT_PATHS.db_path)
    if not config.resume_from_checkpoint:
        active_policy = resolve_active_policy_checkpoint(
            store,
            Path(DEFAULT_PATHS.checkpoints_dir),
            Path(__file__).resolve().parent,
            base_model_id=config.whisper_model_id,
        )
        if active_policy is not None:
            config.resume_from_checkpoint = str(active_policy)
            log.info("Continuing SFT from active policy: %s", active_policy)

    summary = store.annotation_summary()
    corrections = store.get_sft_training_rows()
    log.info(
        "Annotations -- total: %d  actionable: %d  equal: %d  skip: %d  "
        "with correction text: %d",
        summary["total"],
        summary["actionable"],
        summary["equal"],
        summary["skip"],
        len(corrections),
    )

    if not corrections:
        log.info(
            "No human corrections found in DB. Type corrected transcripts "
            "in the annotation UI (correction box), then re-run this script."
        )
        sys.exit(0)

    log.info("Starting SFT on %d permitted correction/TTS examples...", len(corrections))

    # -- Train ------------------------------------------------------------------
    trainer = WhisperSFTTrainer(config, store, DEFAULT_PATHS)
    ckpt_dir = trainer.train(args.output_dir or Path(DEFAULT_PATHS.checkpoints_dir) / "sft")
    config.to_json(ckpt_dir / "sft_config.json")

    log.info("SFT complete. Checkpoint: %s", ckpt_dir)
    log.info("Next step: py run_rlhf.py --resume %s", ckpt_dir)


if __name__ == "__main__":
    main()
