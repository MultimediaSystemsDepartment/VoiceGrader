"""
run_score_rewards.py -- score unscored annotation pairs with the reward model.

Uses AudioTextRewardModel (Whisper encoder + Polish BERT + MLP) to compute
reward scores for all pairs that have been annotated but not yet scored.

Requires a trained reward model checkpoint (run run_train_reward.py first).
If no checkpoint is specified, auto-detects the latest in the reward model dir.

Results are written to the reward_scores table so run_rlhf.py can read them
via RewardBuffer without holding the reward model in GPU memory during training.

Usage:
    py run_score_rewards.py                         # auto-detect checkpoint
    py run_score_rewards.py --checkpoint path/to/checkpoint
    py run_score_rewards.py --config config/reward_model_config.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import DEFAULT_PATHS
from reward_model import AudioTextRewardModel, RewardModelConfig
from storage import FeedbackStore
from storage.schema import RewardScore
from utils.log import get_logger, setup_logging

log = get_logger(__name__)

_DEFAULT_RM_CONFIG = Path("config") / "reward_model_config.json"


def _find_latest_checkpoint(base_dir: Path) -> Optional[Path]:
    """Return the lexicographically latest reward_model_vN directory."""
    if not base_dir.exists():
        return None
    checkpoints = sorted(base_dir.glob("reward_model_v*"), key=lambda p: p.name)
    return checkpoints[-1] if checkpoints else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score annotation pairs with the audio-text reward model"
    )
    p.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Path to trained reward model checkpoint directory "
             "(default: auto-detect latest in reward_model_dir)",
    )
    p.add_argument(
        "--config", type=Path, default=None,
        help="Path to reward_model_config.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(logs_dir=DEFAULT_PATHS.logs_dir)
    log.info("=== run_score_rewards ===")

    cfg_path = args.config or (_DEFAULT_RM_CONFIG if _DEFAULT_RM_CONFIG.exists() else None)
    config = RewardModelConfig.from_json(cfg_path) if cfg_path else RewardModelConfig()

    store = FeedbackStore(DEFAULT_PATHS.db_path)

    # Pairs that have been annotated but not yet scored
    pairs = store.get_unscored_annotated_pairs()
    if not pairs:
        log.info("No unscored annotated pairs found -- nothing to do")
        return

    # Resolve checkpoint (only needed if there are pairs to score)
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = _find_latest_checkpoint(Path(DEFAULT_PATHS.reward_model_dir))
    if checkpoint is None:
        base_dir = Path(DEFAULT_PATHS.reward_model_dir)
        if not base_dir.exists() or not list(base_dir.glob("reward_model_v*")):
            # Reward model has never been trained yet — soft-skip this cycle.
            # This is expected on cold start before enough actionable annotations
            # have accumulated. Pipeline will retry on the next cycle.
            log.warning(
                "Reward model not yet bootstrapped (no checkpoint in %s). "
                "Skipping scoring this cycle -- collect more a/b annotations "
                "and re-run run_train_reward.py.",
                DEFAULT_PATHS.reward_model_dir,
            )
            return  # exit 0 -- pipeline continues
        # Directory exists but checkpoint is missing: unexpected, likely corrupted.
        log.error(
            "Reward model checkpoint missing from %s (directory exists but "
            "no reward_model_v* checkpoint found). "
            "Run run_train_reward.py to rebuild.",
            DEFAULT_PATHS.reward_model_dir,
        )
        sys.exit(1)
    log.info("Using checkpoint: %s", checkpoint)
    log.info("Pairs to score: %d", len(pairs))

    # Build audio_id -> file_path mapping for all relevant pairs
    audio_ids = list({p.audio_id for p in pairs})
    audio_paths = {}
    for aid in audio_ids:
        af = store.get_audio_file(aid)
        if af is not None:
            audio_paths[aid] = af.file_path

    log.info("Unique audio files resolved: %d / %d", len(audio_paths), len(audio_ids))

    source = f"reward_model_{config.reward_model_version}"
    inserted = 0

    with AudioTextRewardModel(config, checkpoint_path=checkpoint) as rm:
        results = rm.score_pairs(pairs, audio_paths)

    for pair_id, r_a, r_b in results:
        score = RewardScore.new(
            pair_id=pair_id,
            reward_a=r_a,
            reward_b=r_b,
            source=source,
            gemma_version=str(checkpoint),   # field repurposed: stores RM checkpoint path
        )
        store.insert_reward_score(score)
        inserted += 1

    log.info(
        "Inserted %d reward scores  (source=%s  checkpoint=%s)",
        inserted, source, checkpoint,
    )
    log.info("Next step: py run_rlhf.py")


if __name__ == "__main__":
    main()
