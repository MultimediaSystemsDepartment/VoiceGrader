"""
Tests for RewardBuffer and WhisperRLHFDataset.

No GPU, no Whisper, no audio. The reward buffer and dataset operate on plain
Python dicts (DB row dicts), so they can be tested entirely in CPU.

Critical coverage:
- get_dpo_pair() correct chosen/rejected assignment
  Risk: if hyp_a and hyp_b are swapped, DPO trains in the WRONG direction.
  The model would learn to produce worse transcriptions — a silent catastrophic failure.
- WhisperRLHFDataset min_reward_delta filtering
  Risk: if delta filter is inverted or off-by-one, either all pairs are kept
  (noisy training on near-tie pairs) or all pairs are dropped (training loop
  processes zero batches with no error raised).
"""

from __future__ import annotations

import pytest

from rlhf.dataset import WhisperRLHFDataset
from rlhf.reward_buffer import RewardBuffer
from tests.conftest import annotate, make_audio, make_pair, make_session
from storage.schema import RewardScore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(hyp_a: str, hyp_b: str, reward_a: float, reward_b: float) -> dict:
    """Minimal reward-dataset-style row for unit testing."""
    return {
        "pair_id":    "test-pair-id",
        "audio_id":   "test-audio-id",
        "file_path":  "audio/test.wav",
        "hyp_a":      hyp_a,
        "hyp_b":      hyp_b,
        "reward_a":   reward_a,
        "reward_b":   reward_b,
        "strategy_a": "beam_5",
        "strategy_b": "temp_0.8",
        "source":     "reward_model_v1",
        "reference_transcript": None,
    }


# ---------------------------------------------------------------------------
# RewardBuffer.get_dpo_pair()
# ---------------------------------------------------------------------------

class TestRewardBufferGetDpoPair:
    """
    Critical: if chosen/rejected are swapped, DPO optimises in the wrong direction.
    The model would learn to produce worse transcriptions without raising any error.
    """

    def test_a_is_chosen_when_reward_a_higher(self):
        row = _row("good transcript", "bad transcript", reward_a=0.9, reward_b=0.3)
        store_mock = type("S", (), {"get_reward_dataset": lambda self: [row]})()
        buf = RewardBuffer(store_mock)
        buf.load()

        chosen, rejected = buf.get_dpo_pair(row)
        assert chosen  == "good transcript"
        assert rejected == "bad transcript"

    def test_b_is_chosen_when_reward_b_higher(self):
        row = _row("bad transcript", "good transcript", reward_a=0.2, reward_b=0.8)
        store_mock = type("S", (), {"get_reward_dataset": lambda self: [row]})()
        buf = RewardBuffer(store_mock)
        buf.load()

        chosen, rejected = buf.get_dpo_pair(row)
        assert chosen  == "good transcript"
        assert rejected == "bad transcript"

    def test_returns_none_for_equal_rewards(self):
        """Tied pairs have no preference signal — must be excluded from training."""
        row = _row("hyp a", "hyp b", reward_a=0.5, reward_b=0.5)
        store_mock = type("S", (), {"get_reward_dataset": lambda self: [row]})()
        buf = RewardBuffer(store_mock)
        buf.load()

        assert buf.get_dpo_pair(row) is None

    def test_get_winner_resolves_tie_to_a(self):
        """Ties in get_winner() resolve to hyp_a by convention."""
        row = _row("hyp a", "hyp b", reward_a=0.5, reward_b=0.5)
        store_mock = type("S", (), {"get_reward_dataset": lambda self: [row]})()
        buf = RewardBuffer(store_mock)
        buf.load()

        winner, reward = buf.get_winner(row)
        assert winner == "hyp a"
        assert reward == 0.5


# ---------------------------------------------------------------------------
# RewardBuffer.load() against real DB
# ---------------------------------------------------------------------------

class TestRewardBufferLoad:
    def test_load_returns_count(self, store):
        af = make_audio(store)
        session = make_session(store)
        pair = make_pair(store, af.id)
        annotate(store, session.id, pair.id, "a")
        store.insert_reward_score(
            RewardScore.new(pair.id, reward_a=0.8, reward_b=0.3,
                            source="reward_model_v1", gemma_version="v1")
        )

        buf = RewardBuffer(store)
        n = buf.load()
        assert n == 1

    def test_get_rewards_returns_none_none_for_unknown_pair(self, store):
        buf = RewardBuffer(store)
        buf.load()
        assert buf.get_rewards("nonexistent-id") == (None, None)

    def test_get_rewards_returns_correct_tuple(self, store):
        af = make_audio(store)
        session = make_session(store)
        pair = make_pair(store, af.id)
        annotate(store, session.id, pair.id, "b")
        store.insert_reward_score(
            RewardScore.new(pair.id, reward_a=0.2, reward_b=0.9,
                            source="reward_model_v1", gemma_version="v1")
        )

        buf = RewardBuffer(store)
        buf.load()
        r_a, r_b = buf.get_rewards(pair.id)
        assert r_a == pytest.approx(0.2)
        assert r_b == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# WhisperRLHFDataset — min_reward_delta filtering
# ---------------------------------------------------------------------------

class TestWhisperRLHFDataset:
    def test_explicit_human_preference_needs_no_reward_scores(self):
        row = {
            "file_path": "audio.wav",
            "chosen_hyp": "human winner",
            "rejected_hyp": "human loser",
            "source": "human",
        }

        ds = WhisperRLHFDataset([row])

        assert len(ds) == 1
        assert ds[0]["chosen_hyp"] == "human winner"

    def test_keeps_pair_with_clear_delta(self):
        rows = [_row("a", "b", 0.9, 0.1)]  # delta = 0.8
        ds = WhisperRLHFDataset(rows, min_reward_delta=0.0)
        assert len(ds) == 1

    def test_excludes_equal_reward_pair_with_default_delta(self):
        rows = [_row("a", "b", 0.5, 0.5)]  # delta = 0.0
        ds = WhisperRLHFDataset(rows, min_reward_delta=0.0)
        assert len(ds) == 0

    def test_excludes_pair_below_min_delta(self):
        rows = [_row("a", "b", 0.55, 0.5)]  # delta = 0.05
        ds = WhisperRLHFDataset(rows, min_reward_delta=0.1)
        assert len(ds) == 0

    def test_keeps_pair_at_or_above_min_delta(self):
        rows = [
            _row("a", "b", 0.6, 0.5),   # delta = 0.1 — exactly at threshold, excluded (<=)
            _row("c", "d", 0.7, 0.5),   # delta = 0.2 — above threshold, kept
        ]
        ds = WhisperRLHFDataset(rows, min_reward_delta=0.1)
        # delta == threshold is EXCLUDED (filter uses strict <=)
        assert len(ds) == 1
        assert ds[0]["hyp_a"] == "c"

    def test_empty_rows_produces_empty_dataset(self):
        ds = WhisperRLHFDataset([], min_reward_delta=0.0)
        assert len(ds) == 0

    def test_all_pairs_filtered_produces_empty_dataset(self):
        """
        Critical: if delta threshold is too high, all pairs are filtered.
        Training loop should receive an empty dataset — not a hidden bug.
        """
        rows = [_row("a", "b", 0.51, 0.5) for _ in range(10)]  # all delta=0.01
        ds = WhisperRLHFDataset(rows, min_reward_delta=0.5)
        assert len(ds) == 0
