"""
RewardBuffer -- in-memory lookup of pre-computed reward scores.

Decouples the AudioTextRewardModel (GPU-heavy) from the DPO training loop
(Whisper policy, also GPU-heavy). Rewards are scored offline by
run_score_rewards.py and stored in the DB; this class loads them into RAM
for fast access during training without holding the reward model in VRAM.

Usage:
    buf = RewardBuffer(store)
    n   = buf.load()                          # load all scored pairs
    r_a, r_b = buf.get_rewards(pair_id)       # (None, None) if not found
    chosen_hyp, reward = buf.get_winner(row)  # higher-reward hypothesis
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

log = logging.getLogger(__name__)


class RewardBuffer:
    """
    Dict-backed store: pair_id -> (reward_a, reward_b).
    Not thread-safe for writes, but safe for concurrent reads after load().
    """

    def __init__(self, store) -> None:      # store: FeedbackStore
        self._store = store
        self._rewards: Dict[str, Tuple[float, float]] = {}
        self._rows: list = []               # full dataset rows for convenience

    def load(self) -> int:
        """
        Load all scored pairs from DB into RAM.
        Returns the number of pairs loaded.
        """
        self._rows = self._store.get_reward_dataset()
        self._rewards = {
            r["pair_id"]: (r["reward_a"], r["reward_b"])
            for r in self._rows
        }
        log.info("RewardBuffer loaded %d scored pairs", len(self._rewards))
        return len(self._rewards)

    def __len__(self) -> int:
        return len(self._rewards)

    def get_rewards(self, pair_id: str) -> Tuple[Optional[float], Optional[float]]:
        """Return (reward_a, reward_b) or (None, None) if pair not scored."""
        r = self._rewards.get(pair_id)
        return r if r is not None else (None, None)

    def get_winner(self, row: dict) -> Tuple[str, float]:
        """
        Given a reward dataset row, return (winning_hypothesis_text, reward).
        Ties (reward_a == reward_b) resolve to hyp_a.
        """
        if row["reward_a"] >= row["reward_b"]:
            return row["hyp_a"], row["reward_a"]
        return row["hyp_b"], row["reward_b"]

    def get_dpo_pair(self, row: dict) -> Optional[Tuple[str, str]]:
        """
        Return (chosen_text, rejected_text) for DPO.
        Returns None for equal-reward pairs (no preference signal).
        """
        if row["reward_a"] > row["reward_b"]:
            return row["hyp_a"], row["hyp_b"]
        if row["reward_b"] > row["reward_a"]:
            return row["hyp_b"], row["hyp_a"]
        return None

    @property
    def rows(self) -> list:
        """All loaded dataset rows (for building PyTorch Dataset)."""
        return self._rows
