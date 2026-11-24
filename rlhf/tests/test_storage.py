"""
Tests for FeedbackStore — the SQLite persistence layer.

All tests run against an in-memory temp DB via the `store` fixture (conftest.py).
No GPU, no Whisper, no real audio files required.

Regression coverage for:
- get_stats() returning actionable_annotations (a+b only) vs human_annotations (a+b+equal)
  Bug: threshold mismatch — coordinator was triggering on human_annotations but reward
  model training required actionable annotations. This caused premature pipeline triggers.
- get_unscored_annotated_pairs() returning pairs without reward scores
  Bug: run_score_rewards.py checked for a checkpoint before calling this method,
  causing exit(1) even when there was nothing to score.
- annotation_summary() correctly categorising all preference types
"""

from __future__ import annotations

import pytest

from storage.schema import HypothesisPair, RewardAuditItem, RewardAuditJob, RewardScore
from tests.conftest import annotate, make_audio, make_pair, make_session


class TestPolicySpecificQueue:
    def test_pair_reservation_breaks_timestamp_ties_by_insertion_order(self, store):
        audio = make_audio(store, path="audio/ordered.wav")
        first = make_pair(store, audio.id)
        second = HypothesisPair.new(
            audio_id=audio.id,
            hyp_a="second A",
            hyp_b="second B",
            strategy_a="beam_5",
            strategy_b="temp_0.8",
            model_version="second-policy",
        )
        store.insert_hypothesis_pair(second)

        assert store.get_and_reserve_pair().id == first.id

    def test_random_audio_sample_is_seeded(self, store):
        for index in range(10):
            make_audio(store, path=f"audio/random-{index}.wav")

        first = store.get_random_audio_files(4, seed=17)
        second = store.get_random_audio_files(4, seed=17)

        assert [item.id for item in first] == [item.id for item in second]
        assert len(first) == 4

    def test_explicit_audio_selection_preserves_order_and_eligibility(self, store):
        first = make_audio(store, path="audio/first.wav")
        second = make_audio(store, path="audio/second.wav")
        already_paired = make_audio(store, path="audio/paired.wav")
        excluded = make_audio(store, path="audio/excluded.wav")
        make_pair(store, already_paired.id)
        store.set_audio_excluded(excluded.id, True, actor="test", reason="fixture")

        selected = store.get_eligible_audio_files_by_ids(
            [second.id, "missing", excluded.id, already_paired.id, first.id, second.id],
            "openai/whisper-medium",
        )

        assert [audio.id for audio in selected] == [second.id, first.id]

    def test_ground_truth_candidates_exclude_missing_gt_and_curated_audio(self, store):
        eligible = make_audio(store, path="audio/eligible.wav")
        missing_gt = make_audio(store, path="audio/missing.wav")
        excluded = make_audio(store, path="audio/excluded-gt.wav")
        with store._conn() as conn:
            conn.execute(
                "UPDATE audio_files SET reference_transcript = NULL WHERE id = ?",
                (missing_gt.id,),
            )
        store.set_audio_excluded(excluded.id, True, actor="test", reason="fixture")

        candidates = store.get_ground_truth_audio_files()

        assert [audio.id for audio in candidates] == [eligible.id]

    def test_database_clone_is_consistent_and_refuses_overwrite(self, store, tmp_path):
        audio = make_audio(store, path="audio/cloned.wav")
        destination = tmp_path / "clone.sqlite"

        store.clone_to(destination)

        clone = type(store)(destination)
        assert [item.id for item in clone.get_ground_truth_audio_files()] == [audio.id]
        with pytest.raises(FileExistsError, match="destination exists"):
            store.clone_to(destination)

    def test_eligible_audio_selection_is_randomized(self, store):
        for index in range(8):
            make_audio(store, path=f"audio/{index}.wav")

        selections = {
            tuple(
                item.id
                for item in store.get_unqueued_audio_files(
                    3, model_version="current-policy"
                )
            )
            for _ in range(12)
        }

        assert len(selections) > 1

    def test_new_policy_can_regenerate_audio_without_deleting_history(self, store):
        audio = make_audio(store)
        old_pair = make_pair(store, audio.id)

        available = store.get_unqueued_audio_files(
            10, model_version="openai/whisper-medium+lora:new-policy"
        )

        assert [item.id for item in available] == [audio.id]
        assert store.get_pair_by_id(old_pair.id) is not None

    def test_same_policy_cannot_queue_audio_twice(self, store):
        audio = make_audio(store)
        make_pair(store, audio.id)

        assert store.get_unqueued_audio_files(
            10, model_version="openai/whisper-medium"
        ) == []

    def test_rollover_retires_only_stale_pending_pairs(self, store):
        audio = make_audio(store)
        stale = make_pair(store, audio.id)
        current = make_pair(store, audio.id)
        with store._conn() as conn:
            conn.execute(
                "UPDATE hypothesis_pairs SET model_version = ? WHERE id = ?",
                ("current-policy", current.id),
            )

        retired = store.retire_pending_pairs_except_model("current-policy")

        assert retired == 1
        assert store.get_pair_by_id(stale.id).status == "skipped"
        assert store.get_pair_by_id(current.id).status == "pending"


class TestRewardAudit:
    def test_search_job_scores_without_touching_training_scores(self, store):
        audio = make_audio(store, path="audio/patient-alpha.wav")
        pair = make_pair(store, audio.id)
        session = make_session(store)
        annotate(store, session.id, pair.id, "a", correction_text="poprawiona")

        selected = store.select_reward_audit_pairs(
            "search", limit=10, query="patient-alpha"
        )
        assert [row["pair_id"] for row in selected] == [pair.id]

        job = RewardAuditJob.new(
            checkpoint_path="checkpoints/reward_model_v1",
            checkpoint_name="reward_model_v1",
            selection_mode="search",
            sample_size=10,
            query="patient-alpha",
        )
        store.create_reward_audit_job(
            job, [RewardAuditItem.new(job.id, pair.id)]
        )

        claimed = store.claim_next_reward_audit_job()
        assert claimed["id"] == job.id
        assert store.claim_next_reward_audit_job() is None

        store.complete_reward_audit_job(job.id, [(pair.id, 1.5, -0.25)])
        item = store.get_reward_audit_items(job.id)[0]
        assert item["reward_a"] == pytest.approx(1.5)
        assert item["reward_b"] == pytest.approx(-0.25)
        assert item["preference"] == "a"
        assert item["correction_text"] == "poprawiona"
        assert store.get_reward_dataset() == []

    def test_running_job_is_failed_after_coordinator_restart(self, store):
        audio = make_audio(store, path="audio/restart.wav")
        pair = make_pair(store, audio.id)
        job = RewardAuditJob.new("rm/v1", "v1", "random", 1)
        store.create_reward_audit_job(job, [RewardAuditItem.new(job.id, pair.id)])
        store.claim_next_reward_audit_job()

        assert store.fail_running_reward_audit_jobs("restarted") == 1
        failed = store.get_reward_audit_job(job.id)
        assert failed["status"] == "failed"
        assert failed["error_summary"] == "restarted"

    def test_random_selection_is_bounded(self, store):
        for index in range(4):
            audio = make_audio(store, path=f"audio/audit-{index}.wav")
            make_pair(store, audio.id)

        selected = store.select_reward_audit_pairs("random", limit=2)

        assert len(selected) == 2
        assert len({row["pair_id"] for row in selected}) == 2


# ---------------------------------------------------------------------------
# get_stats()
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_empty_db_returns_zeros(self, store):
        stats = store.get_stats()
        assert stats["audio_files"] == 0
        assert stats["human_annotations"] == 0
        assert stats["actionable_annotations"] == 0
        assert stats["reward_scores"] == 0

    def test_actionable_annotations_counts_only_a_and_b(self, store):
        """Regression: actionable_annotations must NOT include 'equal' preferences."""
        af = make_audio(store)
        session = make_session(store)

        # Three annotated pairs: a, b, equal
        for i, pref in enumerate(["a", "b", "equal"]):
            pair = make_pair(store, af.id)
            annotate(store, session.id, pair.id, pref)

        stats = store.get_stats()
        assert stats["actionable_annotations"] == 2   # a + b only
        assert stats["human_annotations"] == 3        # a + b + equal (no skip)

    def test_skip_excluded_from_human_annotations(self, store):
        af = make_audio(store)
        session = make_session(store)
        pair = make_pair(store, af.id)
        annotate(store, session.id, pair.id, "skip")

        stats = store.get_stats()
        assert stats["human_annotations"] == 0
        assert stats["actionable_annotations"] == 0

    def test_auto_session_excluded_from_human_annotations(self, store):
        """Regression: auto-labeled annotations inflated human_annotations count
        (e.g. 231 shown instead of genuine 153 in the admin panel)."""
        from storage.feedback_store import _AUTO_SESSION_ID
        from storage.schema import AnnotationSession, Annotation

        af = make_audio(store)
        auto_session = AnnotationSession(
            id=_AUTO_SESSION_ID, participant_id="system", machine_id="system"
        )
        store.open_session(auto_session)
        auto_pair = make_pair(store, af.id)
        store.insert_annotation(Annotation.new(_AUTO_SESSION_ID, auto_pair.id, "a"))

        # Human annotator
        session = make_session(store)
        human_pair = make_pair(store, af.id)
        annotate(store, session.id, human_pair.id, "b")

        stats = store.get_stats()
        assert stats["human_annotations"] == 1    # auto-label excluded
        assert stats["actionable_annotations"] == 1

    def test_actionable_not_inflated_by_equal(self, store):
        """50 equal annotations should not trigger reward model training threshold."""
        af = make_audio(store)
        session = make_session(store)
        for i in range(10):
            pair = make_pair(store, af.id)
            annotate(store, session.id, pair.id, "equal")

        stats = store.get_stats()
        assert stats["actionable_annotations"] == 0
        assert stats["human_annotations"] == 10

    def test_reward_scores_counted(self, store):
        af = make_audio(store)
        session = make_session(store)
        pair = make_pair(store, af.id)
        annotate(store, session.id, pair.id, "a")

        score = RewardScore.new(pair_id=pair.id, reward_a=0.8, reward_b=0.3,
                                source="reward_model_v1", gemma_version="v1")
        store.insert_reward_score(score)

        assert store.get_stats()["reward_scores"] == 1


# ---------------------------------------------------------------------------
# annotation_summary()
# ---------------------------------------------------------------------------

class TestAnnotationSummary:
    def test_empty_returns_zeros(self, store):
        summary = store.annotation_summary()
        assert summary == {"total": 0, "actionable": 0, "equal": 0, "skip": 0}

    def test_counts_all_preference_types(self, store):
        af = make_audio(store)
        session = make_session(store)
        for pref in ["a", "a", "b", "equal", "skip"]:
            pair = make_pair(store, af.id)
            annotate(store, session.id, pair.id, pref)

        summary = store.annotation_summary()
        # skip is excluded from annotation_summary (it filters by session_id != AUTO)
        # but skip IS stored — let's verify total excludes it:
        assert summary["actionable"] == 3   # 2x a + 1x b
        assert summary["equal"] == 1
        # total includes all preferences: a + b + equal + skip
        assert summary["total"] == summary["actionable"] + summary["equal"] + summary["skip"]


# ---------------------------------------------------------------------------
# get_unscored_annotated_pairs()
# ---------------------------------------------------------------------------

class TestUnscoredAnnotatedPairs:
    def test_empty_when_no_pairs(self, store):
        """Regression: run_score_rewards must exit 0 when this returns empty."""
        assert store.get_unscored_annotated_pairs() == []

    def test_returns_annotated_pair_without_score(self, store):
        af = make_audio(store)
        session = make_session(store)
        pair = make_pair(store, af.id)
        annotate(store, session.id, pair.id, "a")

        unscored = store.get_unscored_annotated_pairs()
        assert len(unscored) == 1
        assert unscored[0].id == pair.id

    def test_excludes_pair_that_already_has_score(self, store):
        af = make_audio(store)
        session = make_session(store)
        pair = make_pair(store, af.id)
        annotate(store, session.id, pair.id, "b")

        score = RewardScore.new(pair_id=pair.id, reward_a=0.2, reward_b=0.9,
                                source="reward_model_v1", gemma_version="v1")
        store.insert_reward_score(score)

        assert store.get_unscored_annotated_pairs() == []

    def test_excludes_pending_and_skipped_pairs(self, store):
        af = make_audio(store)
        session = make_session(store)

        pending_pair = make_pair(store, af.id)        # never annotated → status='pending'
        skipped_pair = make_pair(store, af.id)
        annotate(store, session.id, skipped_pair.id, "skip")  # status='skipped'

        assert store.get_unscored_annotated_pairs() == []

    def test_partial_scoring(self, store):
        """Only the unscored pair in a mixed set should be returned."""
        af = make_audio(store)
        session = make_session(store)

        scored_pair = make_pair(store, af.id)
        annotate(store, session.id, scored_pair.id, "a")
        store.insert_reward_score(
            RewardScore.new(scored_pair.id, 0.7, 0.3, "reward_model_v1", "v1")
        )

        unscored_pair = make_pair(store, af.id)
        annotate(store, session.id, unscored_pair.id, "b")

        result = store.get_unscored_annotated_pairs()
        assert len(result) == 1
        assert result[0].id == unscored_pair.id


# ---------------------------------------------------------------------------
# get_sft_corrections()
# ---------------------------------------------------------------------------

class TestSftCorrections:
    def test_empty_when_no_corrections(self, store):
        af = make_audio(store)
        session = make_session(store)
        pair = make_pair(store, af.id)
        annotate(store, session.id, pair.id, "a", correction_text=None)

        assert store.get_sft_corrections() == []

    def test_returns_row_with_correction_text(self, store):
        af = make_audio(store)
        session = make_session(store)
        pair = make_pair(store, af.id)
        annotate(store, session.id, pair.id, "a", correction_text="corrected transcript")

        corrections = store.get_sft_corrections()
        assert len(corrections) == 1
        assert corrections[0]["correction_text"] == "corrected transcript"

    def test_excludes_empty_correction_text(self, store):
        af = make_audio(store)
        session = make_session(store)
        pair = make_pair(store, af.id)
        annotate(store, session.id, pair.id, "a", correction_text="   ")

        assert store.get_sft_corrections() == []

    def test_current_policy_feedback_excludes_historical_policy(self, store):
        audio = make_audio(store)
        session = make_session(store)
        versions = ["whisper+lora:old", "whisper+lora:active"]
        for version in versions:
            pair = HypothesisPair.new(
                audio_id=audio.id,
                hyp_a="a",
                hyp_b="b",
                strategy_a="beam",
                strategy_b="sample",
                model_version=version,
            )
            store.insert_hypothesis_pair(pair)
            annotate(
                store,
                session.id,
                pair.id,
                "a",
                correction_text=f"correction-{version}",
            )
        store.set_setting("active_policy_model_version", versions[1])

        assert len(store.get_preference_dataset()) == 2
        assert len(store.get_sft_corrections()) == 2
        assert {
            row["model_version"]
            for row in store.get_current_policy_preference_dataset()
        } == {versions[1]}
        assert {
            row["correction_text"]
            for row in store.get_current_policy_sft_corrections()
        } == {f"correction-{versions[1]}"}

    def test_active_policy_version_is_recovered_for_existing_deployment(self, store):
        audio = make_audio(store)
        active_version = "openai/whisper-medium+lora:whisper_dpo_active"
        pair = HypothesisPair.new(
            audio_id=audio.id,
            hyp_a="a",
            hyp_b="b",
            strategy_a="beam",
            strategy_b="sample",
            model_version=active_version,
        )
        store.insert_hypothesis_pair(pair)
        store.set_setting(
            "active_policy_checkpoint",
            "_outputs/checkpoints/whisper_dpo_active",
        )

        assert store.get_active_policy_model_version() == active_version

    def test_initialized_base_policy_feedback_is_current(self, store):
        audio = make_audio(store)
        session = make_session(store)
        pair = HypothesisPair.new(
            audio_id=audio.id,
            hyp_a="a",
            hyp_b="b",
            strategy_a="beam",
            strategy_b="sample",
            model_version="openai/whisper-medium",
        )
        store.insert_hypothesis_pair(pair)
        annotate(store, session.id, pair.id, "a", correction_text="corrected")
        store.set_setting("active_policy_model_version", "openai/whisper-medium")

        assert len(store.get_current_policy_preference_dataset()) == 1
        assert len(store.get_current_policy_sft_corrections()) == 1
