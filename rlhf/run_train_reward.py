"""
run_train_reward.py -- train the AudioTextRewardModel on human preference annotations.

Trains the MLP head of AudioTextRewardModel (Whisper encoder + Polish BERT + MLP)
using Bradley-Terry pairwise loss on human-annotated preference pairs.

Requires config.min_annotations_for_training human preference annotations
(default: 50).  Will refuse to train if this threshold is not met.

After training, the checkpoint is saved to:
    _outputs/reward_model/reward_model_v{N}/

Bump config.reward_model_version (v1 -> v2 -> ...) before each new training run
to keep checkpoints versioned alongside reward_scores.source in the DB.

Usage:
    py run_train_reward.py
    py run_train_reward.py --config config/reward_model_config.json
    py run_train_reward.py --force    # bypass annotation count guard
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import DEFAULT_PATHS
from reward_model import RewardModelConfig, RewardModelTrainer
from storage import FeedbackStore
from utils.log import get_logger, setup_logging

log = get_logger(__name__)

_DEFAULT_CONFIG = Path("config") / "reward_model_config.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AudioTextRewardModel on human preferences")
    p.add_argument("--config", type=Path, default=None,
                   help="Path to reward_model_config.json")
    p.add_argument("--force", action="store_true",
                   help="Skip the minimum-annotations guard")
    p.add_argument("--version", default=None,
                   help="Immutable checkpoint version assigned by the coordinator")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(logs_dir=DEFAULT_PATHS.logs_dir)
    log.info("=== run_train_reward ===")

    cfg_path = args.config or (_DEFAULT_CONFIG if _DEFAULT_CONFIG.exists() else None)
    config = RewardModelConfig.from_json(cfg_path) if cfg_path else RewardModelConfig()
    if args.version:
        config.reward_model_version = args.version
        target = Path(DEFAULT_PATHS.reward_model_dir) / f"reward_model_{args.version}"
        if target.exists():
            raise FileExistsError(
                f"Immutable reward checkpoint already exists: {target}"
            )

    if args.force:
        config.min_annotations_for_training = 0
        log.warning("--force: annotation count guard bypassed")

    store = FeedbackStore(DEFAULT_PATHS.db_path)

    summary = store.annotation_summary()
    log.info(
        "Annotations -- total: %d  actionable (a/b): %d  equal: %d  skip: %d  "
        "(need %d actionable to train)",
        summary["total"], summary["actionable"], summary["equal"], summary["skip"],
        config.min_annotations_for_training,
    )
    if summary["actionable"] < config.min_annotations_for_training:
        log.info(
            "Not enough clear-preference annotations (%d / %d required) — skipping. "
            "Keep annotating, or use --force to override.",
            summary["actionable"], config.min_annotations_for_training,
        )
        return

    trainer = RewardModelTrainer(
        config=config,
        store=store,
        output_dir=DEFAULT_PATHS.reward_model_dir,
    )
    checkpoint = trainer.train()
    log.info("Training complete. Checkpoint: %s", checkpoint)
    log.info("Next step: py run_score_rewards.py --checkpoint %s", checkpoint)


if __name__ == "__main__":
    main()
