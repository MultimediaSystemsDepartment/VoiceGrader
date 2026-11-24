"""Score one queued Reward Audit job with its selected checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from paths import DEFAULT_PATHS
from reward_model import AudioTextRewardModel, RewardModelConfig
from storage import FeedbackStore
from utils.log import get_logger, setup_logging

log = get_logger(__name__)
_DEFAULT_RM_CONFIG = Path("config") / "reward_model_config.json"


def main() -> None:
    setup_logging(logs_dir=DEFAULT_PATHS.logs_dir)
    store = FeedbackStore(DEFAULT_PATHS.db_path)
    job = store.claim_next_reward_audit_job()
    if job is None:
        log.info("No queued Reward Audit job")
        return

    try:
        checkpoint = Path(job["checkpoint_path"])
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Reward checkpoint not found: {checkpoint}")
        config = (
            RewardModelConfig.from_json(_DEFAULT_RM_CONFIG)
            if _DEFAULT_RM_CONFIG.exists()
            else RewardModelConfig()
        )
        rows = store.get_reward_audit_items(job["id"])
        pairs = [
            SimpleNamespace(
                id=row["pair_id"],
                audio_id=row["audio_id"],
                hyp_a=row["hyp_a"],
                hyp_b=row["hyp_b"],
            )
            for row in rows
        ]
        audio_paths = {row["audio_id"]: row["file_path"] for row in rows}
        log.info(
            "Scoring Reward Audit job %s: %d pairs with %s",
            job["id"],
            len(pairs),
            checkpoint,
        )
        with AudioTextRewardModel(config, checkpoint_path=checkpoint) as model:
            scores = model.score_pairs(pairs, audio_paths)
        if len(scores) != len(pairs):
            raise RuntimeError(f"Scored only {len(scores)} of {len(pairs)} pairs")
        store.complete_reward_audit_job(job["id"], scores)
        log.info("Reward Audit job %s completed", job["id"])
    except Exception as exc:
        store.fail_reward_audit_job(job["id"], str(exc))
        log.error("Reward Audit job %s failed: %s", job["id"], exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
