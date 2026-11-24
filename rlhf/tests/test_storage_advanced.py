"""
Tests for FeedbackStore operations not covered in test_storage.py:

- get_reward_dataset() most-recent-score-wins query
  Risk: stale/older scores used for DPO training if subquery is wrong.
- TrainingRun DB operations (start, update WER before, complete)
  Risk: TypeError crash in run_rlhf.py if parameter names don't match.
  Bug caught: update_training_run_wer_before() was called with wer_admed=/wer_cv=
  but the method signature requires wer_admed_before=/wer_cv_before=.
- annotation_summary() auto-session exclusion
  Risk: auto-labeled annotations inflate training threshold count.
- export_to_json() / merge_from_export() roundtrip
  Risk: data loss or duplication on multi-machine annotation merges.
"""

from __future__ import annotations

import json

import pytest

from storage.feedback_store import FeedbackStore, _AUTO_SESSION_ID
from storage.schema import (
    Annotation,
    AnnotationSession,
    AudioFile,
    HypothesisPair,
    RewardScore,
    TrainingRun,
)
from tests.conftest import annotate, make_audio, make_pair, make_session

# ---------------------------------------------------------------------------
# get_reward_dataset() — most-recent-score-wins
# ---------------------------------------------------------------------------


class TestGetRewardDataset:
    def test_returns_empty_with_no_scores(self, store):
        assert store.get_reward_dataset() == []

    def test_returns_one_row_per_pair(self, store):
        af = make_audio(store)
        session = make_session(store)
        pair = make_pair(store, af.id)
        annotate(store, session.id, pair.id, "a")
        store.insert_reward_score(
            RewardScore.new(pair.id, 0.8, 0.3, "reward_model_v1", "v1")
        )

        rows = store.get_reward_dataset()
        assert len(rows) == 1
        assert rows[0]["pair_id"] == pair.id

    def test_most_recent_score_wins(self, store):
        """
        When a pair is scored twice (e.g., after reward model retrain),
        only the most recent score should appear in the training dataset.
        """
        import time

        af = make_audio(store)
        session = make_session(store)
        pair = make_pair(store, af.id)
        annotate(store, session.id, pair.id, "a")

        # Insert old score
        score_old = RewardScore.new(pair.id, 0.4, 0.9, "reward_model_v1", "v1")
        store.insert_reward_score(score_old)

        # Insert newer score (reward model retrained — scores differ)
        score_new = RewardScore.new(pair.id, 0.85, 0.2, "reward_model_v2", "v2")
        store.insert_reward_score(score_new)

        rows = store.get_reward_dataset()
        # Only one row per pair
        assert len(rows) == 1
        # Must be the newer score
        assert rows[0]["reward_a"] == pytest.approx(0.85)
        assert rows[0]["reward_b"] == pytest.approx(0.2)

    def test_row_contains_required_dpo_fields(self, store):
        af = make_audio(store)
        session = make_session(store)
        pair = make_pair(store, af.id)
        annotate(store, session.id, pair.id, "b")
        store.insert_reward_score(
            RewardScore.new(pair.id, 0.3, 0.7, "reward_model_v1", "v1")
        )

        row = store.get_reward_dataset()[0]
        required_keys = {
            "pair_id",
            "hyp_a",
            "hyp_b",
            "reward_a",
            "reward_b",
            "file_path",
            "strategy_a",
            "strategy_b",
        }
        assert required_keys.issubset(row.keys())


# ---------------------------------------------------------------------------
# TrainingRun DB operations
# ---------------------------------------------------------------------------


class TestTrainingRunOperations:
    def _make_run(self) -> TrainingRun:
        return TrainingRun.new(
            algorithm="dpo",
            whisper_version_before="openai/whisper-medium",
            n_human_annotations=55,
            n_synthetic_annotations=0,
            config_json="{}",
        )

    def test_start_training_run(self, store):
        run = self._make_run()
        store.start_training_run(run)
        runs = store.get_training_runs()
        assert len(runs) == 1
        assert runs[0]["algorithm"] == "dpo"

    def test_update_wer_before_correct_parameter_names(self, store):
        """
        Regression: run_rlhf.py called update_training_run_wer_before() with
        wer_admed=/wer_cv= but the method signature uses wer_admed_before=/wer_cv_before=.
        This would raise TypeError after baseline WER evaluation, crashing the pipeline
        before any training happened.
        """
        run = self._make_run()
        store.start_training_run(run)
        # Must not raise TypeError
        store.update_training_run_wer_before(
            run.id,
            wer_admed_before=0.15,
            wer_cv_before=None,
        )
        runs = store.get_training_runs()
        assert runs[0]["wer_before_admed"] == pytest.approx(0.15)

    def test_complete_training_run(self, store):
        run = self._make_run()
        store.start_training_run(run)
        store.complete_training_run(
            run.id,
            whisper_version_after="_outputs/checkpoints/dpo_v1",
            wer_admed_after=0.12,
            wer_cv_after=None,
        )
        runs = store.get_training_runs()
        assert runs[0]["wer_after_admed"] == pytest.approx(0.12)
        assert runs[0]["whisper_version_after"] == "_outputs/checkpoints/dpo_v1"
        assert runs[0]["completed_at"] is not None


# ---------------------------------------------------------------------------
# annotation_summary() auto-session exclusion
# ---------------------------------------------------------------------------


class TestAutoSessionExclusion:
    def test_auto_labeled_annotations_excluded_from_summary(self, store):
        """
        Auto-labeled annotations (WER-based system labels) must NOT count toward
        the reward model training threshold — only human preference labels count.
        """
        af = make_audio(store)

        # Create a real auto-label session using the fixed AUTO session ID
        auto_session = AnnotationSession(
            id=_AUTO_SESSION_ID,
            participant_id="system",
            machine_id="system",
        )
        store.open_session(auto_session)

        # Insert an auto-labeled pair
        pair = make_pair(store, af.id)
        auto_ann = Annotation.new(
            session_id=_AUTO_SESSION_ID,
            pair_id=pair.id,
            preference="a",
        )
        store.insert_annotation(auto_ann)

        summary = store.annotation_summary()
        # Auto-labels must be invisible to the training threshold
        assert summary["total"] == 0
        assert summary["actionable"] == 0

    def test_auto_labels_excluded_but_human_labels_counted(self, store):
        af = make_audio(store)

        # Auto session
        auto_session = AnnotationSession(
            id=_AUTO_SESSION_ID, participant_id="system", machine_id="system"
        )
        store.open_session(auto_session)
        auto_pair = make_pair(store, af.id)
        store.insert_annotation(Annotation.new(_AUTO_SESSION_ID, auto_pair.id, "a"))

        # Human session
        human_session = make_session(store)
        human_pair = make_pair(store, af.id)
        annotate(store, human_session.id, human_pair.id, "b")

        summary = store.annotation_summary()
        assert summary["total"] == 1  # only the human annotation
        assert summary["actionable"] == 1


class TestAnnotationModeration:
    def test_annotation_mute_is_reversible_and_preserves_audit_data(self, store):
        audio = make_audio(store)
        session = make_session(store, participant="review-me")
        pair = make_pair(store, audio.id)
        annotation = annotate(
            store, session.id, pair.id, "a", correction_text="trusted correction"
        )
        store.insert_reward_score(
            RewardScore.new(pair.id, 0.9, 0.2, "reward_model_v1", "v1")
        )

        store.set_annotation_muted(annotation.id, True, reason="quality review")

        assert store.count_actionable_annotations() == 0
        assert store.get_preference_dataset() == []
        assert store.get_sft_corrections() == []
        assert store.get_reward_dataset() == []
        audit = store.get_annotation_audit()
        assert audit[0]["annotation_id"] == annotation.id
        assert audit[0]["effective_muted"] == 1
        assert audit[0]["reward_a"] == pytest.approx(0.9)
        assert audit[0]["reward_b"] == pytest.approx(0.2)
        assert store.get_annotation_moderation_events()[0]["action"] == "mute"

        store.set_annotation_muted(annotation.id, False, reason="review passed")

        assert store.count_actionable_annotations() == 1
        assert len(store.get_preference_dataset()) == 1
        assert len(store.get_sft_corrections()) == 1
        assert len(store.get_reward_dataset()) == 1
        assert store.get_annotation_moderation_events()[0]["action"] == "unmute"

    def test_annotator_mute_covers_all_sessions_and_future_annotations(self, store):
        first_audio = make_audio(store, "audio/first.wav")
        first_session = make_session(store, participant="shared-id")
        first_pair = make_pair(store, first_audio.id)
        annotate(store, first_session.id, first_pair.id, "a")

        store.set_annotator_muted("shared-id", True, reason="audit sample")

        second_audio = make_audio(store, "audio/second.wav")
        second_session = make_session(store, participant="shared-id")
        second_pair = make_pair(store, second_audio.id)
        annotate(store, second_session.id, second_pair.id, "b")

        assert store.count_human_annotations() == 0
        assert store.annotation_summary()["total"] == 0
        assert store.get_preference_dataset() == []
        assert len(store.get_annotation_audit(state="muted")) == 2
        annotator = store.get_annotator_audit()[0]
        assert annotator["participant_id"] == "shared-id"
        assert annotator["is_muted"] == 1
        assert annotator["session_count"] == 2

        store.set_annotator_muted("shared-id", False, reason="review passed")

        assert store.count_human_annotations() == 2
        assert len(store.get_preference_dataset()) == 2
        assert len(store.get_annotation_audit(state="active")) == 2


def test_excluded_audio_audit_supports_curation_filters(store):
    excluded = make_audio(store, "audio/noisy-recording.wav")
    retained = make_audio(store, "audio/retained.wav")
    pair = make_pair(store, excluded.id)
    store.set_setting("active_policy_model_version", pair.model_version)
    store.exclude_audio_file(excluded.id, "background noise", actor="AK")

    rows = store.get_excluded_audio_audit(query="noise")

    assert [row["audio_id"] for row in rows] == [excluded.id]
    assert rows[0]["exclusion_reason"] == "background noise"
    assert rows[0]["excluded_by"] == "AK"
    assert rows[0]["pair_count"] == 1
    assert retained.id not in {row["audio_id"] for row in rows}
    assert store.get_excluded_audio_datasets() == ["admedvoice"]
    assert store.is_audio_ever_excluded(excluded.id) is True
    assert store.is_audio_ever_excluded(retained.id) is False

    store.set_audio_excluded(
        excluded.id, False, actor="admin", reason="manual review passed"
    )

    restored = store.get_excluded_audio_audit(state="restored")
    assert [row["audio_id"] for row in restored] == [excluded.id]
    assert restored[0]["latest_actor"] == "admin"
    assert restored[0]["latest_action"] == "restore"
    assert store.get_pair_by_id(pair.id).status == "pending"
    assert store.get_setting("queue_refill_requested") == "true"

    store.set_audio_excluded(
        excluded.id, True, actor="admin", reason="second review failed"
    )

    assert store.get_excluded_audio_audit(state="restored") == []
    assert store.get_excluded_audio_audit(state="excluded")[0]["audio_id"] == excluded.id
    assert [event["action"] for event in store.get_audio_curation_events(excluded.id)] == [
        "exclude",
        "restore",
        "exclude",
    ]


# ---------------------------------------------------------------------------
# export_to_json() / merge_from_export() roundtrip
# ---------------------------------------------------------------------------


class TestExportMergeRoundtrip:
    def _populate(self, store: FeedbackStore) -> str:
        """Insert audio + pair + annotation + score. Return pair_id."""
        af = make_audio(store, "audio/export_test.wav")
        session = make_session(store, participant="exporter")
        pair = make_pair(store, af.id)
        annotate(store, session.id, pair.id, "a")
        store.insert_reward_score(
            RewardScore.new(pair.id, 0.9, 0.2, "reward_model_v1", "v1")
        )
        return pair.id

    def test_moderation_state_and_history_roundtrip(self, store, tmp_path):
        audio = make_audio(store, "audio/moderated-export.wav")
        session = make_session(store, participant="portable-review")
        pair = make_pair(store, audio.id)
        annotation = annotate(store, session.id, pair.id, "a")
        store.set_annotation_muted(annotation.id, True, reason="pair quality")
        store.set_annotator_muted("portable-review", True, reason="review all")
        export_path = tmp_path / "moderated.json"
        store.export_to_json(export_path)

        imported = FeedbackStore(tmp_path / "imported.sqlite")
        counts = imported.merge_from_export(export_path)

        assert counts["annotator_moderation"] == 1
        assert counts["annotation_moderation_events"] == 2
        assert imported.get_preference_dataset() == []
        assert imported.get_annotation_audit()[0]["effective_muted"] == 1
        assert len(imported.get_annotation_moderation_events()) == 2

    def test_export_creates_file(self, store, tmp_path):
        self._populate(store)
        out = tmp_path / "export.json"
        store.export_to_json(out)
        assert out.exists()

    def test_exported_json_contains_all_tables(self, store, tmp_path):
        self._populate(store)
        out = tmp_path / "export.json"
        store.export_to_json(out)
        data = json.loads(out.read_text())
        for table in (
            "audio_files",
            "hypothesis_pairs",
            "annotation_sessions",
            "annotations",
            "reward_scores",
        ):
            assert table in data
            assert len(data[table]) > 0

    def test_merge_into_empty_master(self, store, tmp_path):
        pair_id = self._populate(store)
        out = tmp_path / "export.json"
        store.export_to_json(out)

        master = FeedbackStore(tmp_path / "master.sqlite")
        counts = master.merge_from_export(out)

        assert counts["annotations"] >= 1
        assert counts["reward_scores"] >= 1

        # Data should be queryable in master
        rows = master.get_reward_dataset()
        assert any(r["pair_id"] == pair_id for r in rows)

    def test_double_merge_inserts_zero_on_second_pass(self, store, tmp_path):
        """INSERT OR IGNORE must make repeated merges idempotent."""
        self._populate(store)
        out = tmp_path / "export.json"
        store.export_to_json(out)

        master = FeedbackStore(tmp_path / "master.sqlite")
        first = master.merge_from_export(out)
        second = master.merge_from_export(out)

        # Second merge must insert nothing
        assert all(
            v == 0 for v in second.values()
        ), f"Second merge inserted rows: {second}"

    def test_merge_preserves_annotation_count(self, store, tmp_path):
        self._populate(store)
        out = tmp_path / "export.json"
        store.export_to_json(out)

        master = FeedbackStore(tmp_path / "master.sqlite")
        master.merge_from_export(out)

        assert (
            master.get_stats()["human_annotations"]
            == store.get_stats()["human_annotations"]
        )


# ---------------------------------------------------------------------------
# run_score_rewards cold-start soft-skip (never bootstrapped)
# ---------------------------------------------------------------------------


class TestScoreRewardsColdStart:
    """
    Regression: when the reward model has NEVER been trained (no reward_model_dir),
    run_score_rewards.py was exiting 1 (hard fail) even though it's expected that
    the reward model hasn't been bootstrapped yet on cold start.

    The fix: distinguish "never trained" (dir absent → soft-skip, exit 0)
             from "trained but checkpoint missing" (dir exists, no v* → exit 1).
    """

    def test_find_latest_checkpoint_returns_none_for_missing_dir(self, tmp_path):
        from run_score_rewards import _find_latest_checkpoint

        assert _find_latest_checkpoint(tmp_path / "nonexistent") is None

    def test_find_latest_checkpoint_returns_none_for_empty_dir(self, tmp_path):
        from run_score_rewards import _find_latest_checkpoint

        reward_dir = tmp_path / "reward_model"
        reward_dir.mkdir()
        assert _find_latest_checkpoint(reward_dir) is None

    def test_find_latest_checkpoint_returns_versioned_dir(self, tmp_path):
        from run_score_rewards import _find_latest_checkpoint

        reward_dir = tmp_path / "reward_model"
        v1 = reward_dir / "reward_model_v1"
        v2 = reward_dir / "reward_model_v2"
        v1.mkdir(parents=True)
        v2.mkdir(parents=True)
        result = _find_latest_checkpoint(reward_dir)
        assert result == v2  # lexicographically latest

    def test_never_bootstrapped_is_distinguishable_from_missing_checkpoint(
        self, tmp_path
    ):
        """
        Soft-skip condition: reward_model_dir does NOT exist.
        Hard-fail condition: reward_model_dir EXISTS but contains no versioned checkpoint.
        Both return None from _find_latest_checkpoint, but the caller checks differently.
        """
        from run_score_rewards import _find_latest_checkpoint
        from pathlib import Path

        # Case 1: Never bootstrapped — directory doesn't exist
        never_trained = tmp_path / "reward_model_never"
        assert not never_trained.exists()
        assert _find_latest_checkpoint(never_trained) is None

        # Case 2: Was bootstrapped, checkpoint now missing
        corrupted = tmp_path / "reward_model_corrupted"
        corrupted.mkdir()
        # No reward_model_v* subdirs — checkpoint deleted
        assert corrupted.exists()
        assert not list(corrupted.glob("reward_model_v*"))
        assert _find_latest_checkpoint(corrupted) is None
