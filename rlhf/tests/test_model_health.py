from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlhf.preference_data import human_preference_rows
from evaluation.canary import CanaryBackfill
from storage.feedback_store import FeedbackStore
from storage.schema import AudioFile, CanaryEvaluation, CanarySet, ModelCheckpoint
from tests.conftest import annotate, make_audio, make_pair, make_session


def test_existing_001_database_upgrades_once(tmp_path: Path) -> None:
    db_path = tmp_path / "existing.sqlite"
    migration = Path("storage/migrations/001_initial.sql")
    with sqlite3.connect(db_path) as conn:
        conn.executescript(migration.read_text(encoding="utf-8"))
        conn.execute("""INSERT INTO audio_files (id, file_path, dataset)
               VALUES ('existing-audio', 'existing.wav', 'admedvoice')""")

    store = FeedbackStore(db_path)
    FeedbackStore(db_path)

    assert store.get_audio_file("existing-audio") is not None
    with sqlite3.connect(db_path) as conn:
        versions = conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

    assert versions == [
        ("001_initial.sql",),
        ("002_model_health.sql",),
        ("003_reward_audit.sql",),
        ("004_promotion_gate_monitor.sql",),
        ("005_annotation_moderation.sql",),
        ("006_audio_curation.sql",),
        ("007_catalogue.sql",),
        ("008_tts_campaigns.sql",),
        ("009_corpus_purpose.sql",),
        ("010_audio_quality.sql",),
    ]
    assert {
        "canary_sets",
        "canary_items",
        "model_checkpoints",
        "reward_audit_jobs",
        "reward_audit_items",
    }.issubset(tables)


def test_legacy_inline_columns_upgrade_once(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy-inline.sqlite"
    migration = Path("storage/migrations/001_initial.sql")
    with sqlite3.connect(db_path) as conn:
        conn.executescript(migration.read_text(encoding="utf-8"))
        conn.execute(
            "ALTER TABLE audio_files ADD COLUMN excluded INTEGER NOT NULL DEFAULT 0"
        )
        conn.execute("ALTER TABLE audio_files ADD COLUMN exclusion_reason TEXT")
        conn.execute(
            "ALTER TABLE hypothesis_pairs ADD COLUMN reserved_until TEXT"
        )
        conn.execute("""INSERT INTO audio_files (id, file_path, dataset, excluded)
               VALUES ('legacy-audio', 'legacy.wav', 'admedvoice', 1)""")

    store = FeedbackStore(db_path)
    FeedbackStore(db_path)

    assert store.get_audio_file("legacy-audio") is not None
    with sqlite3.connect(db_path) as conn:
        versions = conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        audio = conn.execute(
            "SELECT excluded, ever_excluded, excluded_by FROM audio_files WHERE id = ?",
            ("legacy-audio",),
        ).fetchone()

    assert versions == [
        ("001_initial.sql",),
        ("002_model_health.sql",),
        ("003_reward_audit.sql",),
        ("004_promotion_gate_monitor.sql",),
        ("005_annotation_moderation.sql",),
        ("006_audio_curation.sql",),
        ("007_catalogue.sql",),
        ("008_tts_campaigns.sql",),
        ("009_corpus_purpose.sql",),
        ("010_audio_quality.sql",),
    ]
    assert audio == (1, 1, "AK")


def test_canary_is_excluded_from_every_learning_path(store: FeedbackStore) -> None:
    selected_candidate = make_audio(store, "audio/canary.wav")
    training_audio = make_audio(store, "audio/training.wav")
    pending_pair = make_pair(store, selected_candidate.id)

    canary_set = CanarySet.new(
        name="Regression canary",
        selection_seed=7,
        selection_method="random_unannotated",
        target_size=1,
    )
    items = store.create_canary_set(canary_set)
    assert len(items) == 1
    selected_id = items[0].audio_id

    selected_pair = pending_pair
    if selected_id != selected_candidate.id:
        selected_pair = make_pair(store, selected_id)
    session = make_session(store)
    annotate(
        store,
        session.id,
        selected_pair.id,
        "a",
        correction_text="holdout transcript",
    )

    training_pair = make_pair(store, training_audio.id)
    annotate(
        store,
        session.id,
        training_pair.id,
        "b",
        correction_text="training transcript",
    )

    assert selected_id not in {
        audio.id for audio in store.get_unqueued_audio_files(100, "new-policy")
    }
    assert selected_id not in {
        audio.id for audio in store.get_random_audio_files(100, seed=1)
    }
    assert selected_id not in {
        row["audio_id"] for row in store.get_preference_dataset()
    }
    assert "holdout transcript" not in {
        row["correction_text"] for row in store.get_sft_corrections()
    }
    assert selected_pair.id not in {
        row["pair_id"] for row in human_preference_rows(store)
    }


def test_canary_can_use_exact_explicit_audio_ids(store: FeedbackStore) -> None:
    first = make_audio(store, "audio/first.wav")
    second = make_audio(store, "audio/second.wav")
    canary_set = CanarySet.new(
        name="Exact monitor",
        selection_seed=7,
        selection_method="experiment_explicit",
        target_size=2,
    )

    items = store.create_canary_set(canary_set, audio_ids=[second.id, first.id])

    assert [item.audio_id for item in items] == [second.id, first.id]
    with pytest.raises(ValueError, match="target size"):
        store.create_canary_set(canary_set, audio_ids=[first.id])


def test_frozen_canary_transcript_is_immutable(store: FeedbackStore) -> None:
    make_audio(store, "audio/canary.wav")
    canary_set = CanarySet.new("Canary", 3, "random_unannotated", target_size=1)
    item = store.create_canary_set(canary_set)[0]

    store.save_canary_transcript(item.id, "tekst wzorcowy")
    fingerprint = store.freeze_canary_set(canary_set.id)
    assert len(fingerprint) == 64
    assert store.get_setting("wer_monitor_evaluation_requested") == "true"
    assert store.get_setting("wer_monitor_evaluation_status") == "queued"
    assert store.get_setting("wer_monitor_evaluation_canary_set_id") == canary_set.id

    with pytest.raises(ValueError, match="immutable"):
        store.save_canary_transcript(item.id, "zmieniony")
    with pytest.raises(ValueError, match="immutable"):
        store.bulk_copy_canary_ground_truth(canary_set.id)

    store.set_setting("wer_monitor_evaluation_status", "running")
    with pytest.raises(ValueError, match="running"):
        store.unfreeze_canary_set(canary_set.id)
    store.set_setting("wer_monitor_evaluation_status", "queued")

    revision_id = store.unfreeze_canary_set(canary_set.id)
    reopened = store.get_canary_sets()[0]
    assert reopened["status"] == "collecting"
    assert reopened["fingerprint"] is None
    assert store.get_canary_sets()[1]["status"] == "ready"
    assert store.get_canary_items(revision_id)[0]["transcript"] == ("tekst wzorcowy")
    assert store.get_setting("wer_monitor_evaluation_requested") == "true"


def test_canary_item_reroll_excludes_faulty_audio_and_preserves_history(
    store: FeedbackStore,
) -> None:
    candidates = [
        make_audio(store, "audio/silence.wav"),
        make_audio(store, "audio/replacement.wav"),
    ]
    canary_set = CanarySet.new("Monitor", 3, "random_unannotated", target_size=1)
    original_item = store.create_canary_set(canary_set)[0]
    faulty = next(audio for audio in candidates if audio.id == original_item.audio_id)
    replacement = next(
        audio for audio in candidates if audio.id != original_item.audio_id
    )
    store.save_canary_transcript(original_item.id, "tekst wzorcowy")
    store.freeze_canary_set(canary_set.id)
    revision_id = store.unfreeze_canary_set(canary_set.id)
    draft_item = store.get_canary_items(revision_id)[0]

    replacement_item_id = store.reroll_canary_item(draft_item["id"])

    assert store.get_canary_items(canary_set.id)[0]["audio_id"] == faulty.id
    revised_item = store.get_canary_items(revision_id)[0]
    assert revised_item["id"] == replacement_item_id
    assert revised_item["audio_id"] == replacement.id
    assert revised_item["verified_at"] is None
    assert faulty.id not in {
        audio.id for audio in store.get_random_audio_files(100, seed=1)
    }


def test_monitor_removal_deletes_drafts_but_archives_locked_history(
    store: FeedbackStore,
) -> None:
    draft_audio = make_audio(store, "audio/draft.wav")
    draft = CanarySet.new("Draft", 1, "random_unannotated", target_size=1)
    store.create_canary_set(draft)

    assert store.remove_canary_set(draft.id) == "deleted"
    assert store.get_canary_set(draft.id) is None
    assert draft_audio.id in {
        audio.id for audio in store.get_random_audio_files(100, seed=1)
    }

    locked = CanarySet.new("Locked", 2, "random_unannotated", target_size=1)
    locked_item = store.create_canary_set(locked)[0]
    store.save_canary_transcript(locked_item.id, "tekst wzorcowy")
    store.freeze_canary_set(locked.id)
    with pytest.raises(ValueError, match="evaluation"):
        store.remove_canary_set(locked.id)
    store.set_setting("wer_monitor_evaluation_status", "completed")

    with pytest.raises(ValueError, match="another promotion gate"):
        store.remove_canary_set(locked.id)
    make_audio(store, "audio/replacement-gate.wav")
    replacement = CanarySet.new("Replacement", 3, "random_unannotated", target_size=1)
    replacement_item = store.create_canary_set(replacement)[0]
    store.save_canary_transcript(replacement_item.id, "tekst wzorcowy")
    store.freeze_canary_set(replacement.id)
    store.set_promotion_gate_canary_set(replacement.id)

    assert store.remove_canary_set(locked.id) == "archived"
    assert store.get_canary_set(locked.id)["status"] == "retired"
    assert store.get_canary_items(locked.id)[0]["id"] == locked_item.id


def test_canary_evaluation_claim_is_atomic_and_preserves_running_predictions(
    store: FeedbackStore,
) -> None:
    make_audio(store, "audio/claim.wav")
    canary_set = CanarySet.new("Claim", 4, "random_unannotated", target_size=1)
    item = store.create_canary_set(canary_set)[0]
    store.save_canary_transcript(item.id, "tekst wzorcowy")
    store.freeze_canary_set(canary_set.id)
    checkpoint = store.register_model_checkpoint(
        ModelCheckpoint.new("base:test", "base", provenance="discovered")
    )
    evaluation = store.get_or_create_canary_evaluation(
        CanaryEvaluation.new(canary_set.id, checkpoint["id"], "profile")
    )

    assert store.start_canary_evaluation(evaluation["id"]) is True
    store.insert_canary_prediction(
        "prediction", evaluation["id"], item.id, "hipoteza", "{}"
    )
    assert store.start_canary_evaluation(evaluation["id"]) is False
    assert len(store.get_canary_predictions(evaluation["id"])) == 1


def test_manual_evaluation_request_keeps_selected_monitor_id(
    store: FeedbackStore,
) -> None:
    for index in range(2):
        make_audio(store, f"audio/request-{index}.wav")
    monitors = []
    for index in range(2):
        monitor = CanarySet.new(
            f"Monitor {index}", index, "random_unannotated", target_size=1
        )
        item = store.create_canary_set(monitor)[0]
        store.save_canary_transcript(item.id, "tekst wzorcowy")
        store.freeze_canary_set(monitor.id)
        monitors.append(monitor)

    store.request_wer_monitor_evaluation(monitors[0].id)

    assert store.get_setting("wer_monitor_evaluation_canary_set_id") == monitors[0].id


def test_promotion_gate_is_explicit_and_does_not_follow_newest_monitor(
    store: FeedbackStore,
) -> None:
    for index in range(2):
        make_audio(store, f"audio/gate-{index}.wav")
    monitors = []
    for index in range(2):
        monitor = CanarySet.new(
            f"Gate {index}", index, "random_unannotated", target_size=1
        )
        item = store.create_canary_set(monitor)[0]
        store.save_canary_transcript(item.id, "tekst wzorcowy")
        store.freeze_canary_set(monitor.id)
        monitors.append(monitor)

    assert store.get_ready_canary_set()["id"] == monitors[1].id
    assert store.get_promotion_gate_canary_set()["id"] == monitors[0].id

    store.set_promotion_gate_canary_set(monitors[1].id)

    assert store.get_promotion_gate_canary_set()["id"] == monitors[1].id
    assert store.get_canary_set(monitors[0].id)["is_promotion_gate"] == 0
    assert store.get_setting("wer_monitor_evaluation_canary_set_id") == monitors[1].id
    with pytest.raises(ValueError, match="another promotion gate"):
        store.remove_canary_set(monitors[1].id)


def test_canary_ground_truth_is_hidden_and_provenance_is_audited(
    store: FeedbackStore,
) -> None:
    make_audio(store, "audio/canary.wav")
    canary_set = CanarySet.new("Canary", 3, "random_unannotated", target_size=1)
    item = store.create_canary_set(canary_set)[0]

    visible_item = store.get_canary_items(canary_set.id)[0]
    assert "reference_transcript" not in visible_item
    assert store.reveal_canary_ground_truth(item.id) == "test ref audio/canary.wav"

    store.save_canary_transcript(item.id, "tekst ręczny")
    assert store.get_canary_items(canary_set.id)[0]["transcript_provenance"] == (
        "manual_after_reveal"
    )
    with pytest.raises(ValueError, match="match"):
        store.save_canary_transcript(item.id, "nie wzorzec", copied_gt=True)

    store.save_canary_transcript(item.id, "test ref audio/canary.wav", copied_gt=True)
    assert store.get_canary_items(canary_set.id)[0]["transcript_provenance"] == (
        "copied_gt"
    )


def test_bulk_gt_copy_preserves_completed_items_and_reports_missing_gt(
    store: FeedbackStore,
) -> None:
    make_audio(store, "audio/manual.wav")
    make_audio(store, "audio/copy.wav")
    missing = AudioFile.new("audio/missing.wav", dataset="admedvoice")
    store.insert_audio_file(missing)
    canary_set = CanarySet.new("WER Monitor", 3, "random_unannotated", target_size=3)
    store.create_canary_set(canary_set)
    items = {item["file_path"]: item for item in store.get_canary_items(canary_set.id)}
    store.save_canary_transcript(items["audio/manual.wav"]["id"], "manual text")

    result = store.bulk_copy_canary_ground_truth(canary_set.id)
    updated = {
        item["file_path"]: item for item in store.get_canary_items(canary_set.id)
    }

    assert result == {"copied": 1, "already_completed": 1, "missing_gt": 1}
    assert updated["audio/manual.wav"]["transcript"] == "manual text"
    assert updated["audio/manual.wav"]["transcript_provenance"] == "manual"
    assert updated["audio/copy.wav"]["transcript"] == "test ref audio/copy.wav"
    assert updated["audio/copy.wav"]["transcript_provenance"] == "copied_gt"
    assert updated["audio/copy.wav"]["gt_revealed"] == 1
    assert updated["audio/missing.wav"]["verified_at"] is None


def test_canary_backfill_discovers_persists_warns_and_reuses_cache(
    store: FeedbackStore, tmp_path: Path
) -> None:
    for index in range(2):
        make_audio(store, f"audio/canary-{index}.wav")
    canary_set = CanarySet.new("Canary", 3, "random_unannotated", target_size=2)
    items = store.create_canary_set(canary_set)
    for item in items:
        store.save_canary_transcript(item.id, "tekst wzorcowy")
    store.freeze_canary_set(canary_set.id)

    dpo_path = tmp_path / "checkpoints" / "whisper_dpo_20260101_120000"
    dpo_path.mkdir(parents=True)
    (dpo_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    paths = SimpleNamespace(
        whisper_model_id="openai/whisper-medium",
        checkpoints_dir=str(tmp_path / "checkpoints"),
    )
    decoder_calls = []

    class FakeEvaluator:
        def __init__(self, **kwargs):
            self.checkpoint = kwargs["lora_checkpoint"]

        def transcribe_files(self, file_paths, batch_size, on_batch):
            decoder_calls.append(self.checkpoint)
            hypothesis = (
                "te-k-s-t wz-o-r-c-o-w-y" if self.checkpoint else "tekst wzorcowy"
            )
            predictions = [hypothesis] * len(file_paths)
            for start in range(0, len(predictions), batch_size):
                on_batch(start, predictions[start : start + batch_size])
            return predictions

    backfill = CanaryBackfill(
        store,
        paths,
        batch_size=1,
        evaluator_factory=FakeEvaluator,
    )
    first_results = backfill.run()
    second_results = backfill.run()

    first_profile_hashes = {
        row["profile_hash"] for row in store.get_canary_evaluations(canary_set.id)
    }
    revision_id = store.unfreeze_canary_set(canary_set.id)
    revised_items = store.get_canary_items(revision_id)
    store.save_canary_transcript(revised_items[0]["id"], "zmieniony tekst wzorcowy")
    store.freeze_canary_set(revision_id)
    store.set_promotion_gate_canary_set(revision_id)
    third_results = backfill.run()

    assert len(first_results) == 2
    assert len(second_results) == 2
    assert len(third_results) == 2
    assert len(decoder_calls) == 4
    evaluations = [
        *store.get_canary_evaluations(canary_set.id),
        *store.get_canary_evaluations(revision_id),
    ]
    assert len(evaluations) == 4
    assert all(row["status"] == "completed" for row in evaluations)
    assert len({row["profile_hash"] for row in evaluations}) == 2
    assert first_profile_hashes < {row["profile_hash"] for row in evaluations}
    dpo_evaluations = [row for row in evaluations if row["stage"] == "dpo"]
    assert len(dpo_evaluations) == 2
    assert all(
        "vs_base" in __import__("json").loads(row["aggregate_json"])
        for row in dpo_evaluations
    )
    assert all(
        __import__("json").loads(row["aggregate_json"])["vs_base"]["wer"]["metric"]
        == "corpus_wer"
        for row in dpo_evaluations
    )
    assert len(store.get_canary_predictions(evaluations[0]["id"])) == 2
    assert len(store.get_canary_predictions(evaluations[1]["id"])) == 2
