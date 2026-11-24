from __future__ import annotations

import json
import sqlite3
import threading
import wave
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import pytest

from storage.feedback_store import FeedbackStore
from storage.schema import (
    Annotation, AnnotationSession, AudioCurationEvent, AudioFile, CanaryItem,
    CanarySet, Corpus, HypothesisPair, RewardScore,
)
from tests.conftest import annotate, make_pair, make_session
from run_import_corpus import main as import_main
from run_import_corpus import validate_manifest


def _write_import_audio(path: Path, duration: float = 0.2, sample_rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\x00\x00" * int(duration * sample_rate))


def test_evaluation_import_is_atomic_inactive_and_requires_review(store, tmp_path):
    _write_import_audio(tmp_path / "held.wav")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("file_path;transcript\nheld.wav;Verified reference.\n", encoding="utf-8")
    corpus = Corpus.new("external-test", purpose="evaluation")
    files, report = validate_manifest(manifest, corpus.name)
    assert not report.errors
    result = store.import_uploaded_corpus(corpus, files)
    monitor = store.get_canary_set(result["monitor_id"])
    assert monitor["status"] == "collecting"
    assert monitor["verified_count"] == 0
    assert not store.get_random_audio_files(10, 1)
    with pytest.raises(ValueError, match="cannot be activated"):
        store.set_corpus_active(corpus.name, True)
    item = store.get_canary_items(monitor["id"])[0]
    assert store.reveal_canary_ground_truth(item["id"]) == "Verified reference."
    store.save_canary_transcript(item["id"], "Verified reference.", copied_gt=True)
    store.freeze_canary_set(monitor["id"])
    assert store.get_training_monitor_rows()["rows"][0]["reference"] == "Verified reference."


@pytest.mark.parametrize("overlap", ["audio", "text", "missing"])
def test_evaluation_import_rejects_overlap_or_missing_text_without_partial_writes(
    store, tmp_path, overlap,
):
    _write_import_audio(tmp_path / "learning.wav")
    store.insert_audio_file(AudioFile.new(str(tmp_path / "learning.wav"), "learning", "Same text."))
    _write_import_audio(tmp_path / "test.wav", duration=0.3 if overlap != "audio" else 0.2)
    transcript = {"audio": "Different text.", "text": "Same  text.", "missing": ""}[overlap]
    corpus = Corpus.new("test-upload", purpose="evaluation")
    corpus.active = False
    audio = AudioFile.new(str(tmp_path / "test.wav"), corpus.name, transcript)
    with pytest.raises(ValueError):
        store.import_uploaded_corpus(corpus, [audio])
    assert len(store.get_corpora()) == 1
    assert store.get_canary_sets() == []
    assert store.count_audio_files() == 1


def test_training_upload_cannot_reimport_held_audio_under_another_name(store, tmp_path):
    _write_import_audio(tmp_path / "held.wav")
    held = AudioFile.new(str(tmp_path / "held.wav"), "existing", "Held text.")
    store.insert_audio_file(held)
    store.create_canary_set(CanarySet.new("monitor", 1, "explicit", 1), [held.id])
    _write_import_audio(tmp_path / "renamed.wav")
    corpus = Corpus.new("new-training")
    corpus.active = False
    with pytest.raises(ValueError, match="Audio content overlaps"):
        store.import_uploaded_corpus(corpus, [
            AudioFile.new(str(tmp_path / "renamed.wav"), corpus.name, "Different text.")
        ])
    assert len(store.get_corpora()) == 1


def test_frozen_monitor_training_evaluation_uses_full_snapshot_without_csv(store, tmp_path, monkeypatch):
    from evaluation.evaluator import WhisperEvaluator
    from paths import PathsConfig

    _write_import_audio(tmp_path / "held.wav")
    _write_import_audio(tmp_path / "second.wav", duration=0.3)
    held = AudioFile.new(str(tmp_path / "held.wav"), "source", "Correct words.")
    second = AudioFile.new(str(tmp_path / "second.wav"), "source", "Correct words.")
    store.insert_audio_file(held)
    store.insert_audio_file(second)
    monitor = CanarySet.new("monitor", 1, "explicit", 2)
    store.create_canary_set(monitor, [held.id, second.id])
    for item in store.get_canary_items(monitor.id):
        store.save_canary_transcript(item["id"], "Correct words.")
    store.freeze_canary_set(monitor.id)
    evaluator = WhisperEvaluator("base", PathsConfig(), store, use_monitor=True, max_samples=1)
    original_hash = evaluator.compute_subset_hash()
    monkeypatch.setattr(evaluator, "_load_model", lambda: None)
    monkeypatch.setattr(evaluator, "_unload_model", lambda: None)
    monkeypatch.setattr(evaluator, "_transcribe_batch", lambda paths: ["Correct words."] * len(paths))
    result = evaluator.evaluate()
    assert (result.wer_admed, result.n_admed, result.wer_cv) == (0, 2, None)
    assert evaluator.decode_num_beams == 5
    assert result.primary_set == "WER Monitor"
    draft_id = store.unfreeze_canary_set(monitor.id)
    draft_item = store.get_canary_items(draft_id)[0]
    store.save_canary_transcript(draft_item["id"], "Changed words.")
    store.freeze_canary_set(draft_id)
    store.set_promotion_gate_canary_set(draft_id)
    revised = WhisperEvaluator("base", PathsConfig(), store, use_monitor=True)
    assert revised.compute_subset_hash() != original_hash
    assert evaluator.monitor["rows"][0]["reference"] == "Correct words."
    benchmark = tmp_path / "benchmark.csv"
    evaluator.paths.commonvoice_test_csv = str(benchmark)
    benchmark.write_text("file_path;transcript\n", encoding="utf-8")
    benchmark_hash = evaluator.compute_subset_hash()
    assert benchmark_hash != original_hash
    benchmark.write_text("file_path;transcript\nsecond.wav;Changed reference.\n", encoding="utf-8")
    assert evaluator.compute_subset_hash() != benchmark_hash


def test_manifest_cli_stages_imports_and_requires_explicit_activation(tmp_path, capsys) -> None:
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    _write_import_audio(first, sample_rate=24_000)
    _write_import_audio(second)
    original_bytes = first.read_bytes()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "file_path;transcript\nfirst.wav;A sentence.\nsecond.wav;\n./first.wav;A  sentence.\n",
        encoding="utf-8",
    )
    database = tmp_path / "catalogue.sqlite"
    importing = ["--db", str(database), "import", "--name", "new-corpus",
                 "--manifest", str(manifest)]
    assert import_main(importing + ["--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "validated"
    assert not database.exists()
    assert import_main(importing) == 0
    report = json.loads(capsys.readouterr().out)
    assert (report["rows"], report["inserted"], report["duplicates"]) == (3, 2, 1)
    assert report["active"] is False
    store = FeedbackStore(database)
    assert store.get_random_audio_files(10, 1) == []
    assert import_main(importing) == 0
    assert json.loads(capsys.readouterr().out)["existing"] == 2
    assert import_main(["--db", str(database), "activate", "--name", "new-corpus"]) == 0
    capsys.readouterr()
    eligible = store.get_random_audio_files(10, 1)
    assert len(eligible) == 2
    imported_first = next(audio for audio in eligible if audio.reference_transcript)
    assert imported_first.sample_rate == 24_000
    assert imported_first.duration_s == pytest.approx(0.2)
    assert json.loads(imported_first.provenance_json)["manifest_row"] == 2
    assert first.read_bytes() == original_bytes
    assert import_main(["--db", str(database), "list"]) == 0
    assert json.loads(capsys.readouterr().out)["corpora"][0]["audio_files"] == 2
    pair = make_pair(store, imported_first.id)
    assert import_main(["--db", str(database), "deactivate", "--name", "new-corpus"]) == 0
    capsys.readouterr()
    assert store.get_and_reserve_pair() is None
    assert store.get_random_audio_files(10, 1) == []
    assert store.get_pair_by_id(pair.id).id == pair.id


@pytest.mark.parametrize("invalid", ["short", "long", "zip-long", "corrupt"])
def test_invalid_audio_rejects_entire_import_without_creating_database(
    tmp_path, capsys, invalid,
) -> None:
    _write_import_audio(tmp_path / "good.wav")
    bad = tmp_path / "bad.wav"
    if invalid == "corrupt":
        bad.write_bytes(b"not audio")
    else:
        _write_import_audio(bad, duration=0.05 if invalid == "short" else 30.1)
    bad_path = "bad.wav"
    if invalid == "zip-long":
        with zipfile.ZipFile(tmp_path / "clips.zip", "w") as archive:
            archive.write(bad, "nested/bad.wav")
        bad_path = "zip:clips.zip!nested/bad.wav"
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(f"file_path\ngood.wav\n{bad_path}\n", encoding="utf-8")
    database = tmp_path / "catalogue.sqlite"
    assert import_main(["--db", str(database), "import", "--name", "new",
                        "--manifest", str(manifest)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "rejected"
    assert len(report["errors"]) == 1
    assert report["errors"][0].startswith("Row 3:")
    assert not database.exists()


@pytest.mark.parametrize("frames, accepted", [(1_600, True), (480_000, True),
                                            (1_599, False), (480_001, False)])
def test_import_duration_limits_are_inclusive_and_checked_to_one_sample(
    tmp_path, capsys, frames, accepted,
) -> None:
    _write_import_audio(tmp_path / "clip.wav", duration=frames / 16_000)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("file_path\nclip.wav\n", encoding="utf-8")
    database = tmp_path / "catalogue.sqlite"
    result = import_main(["--db", str(database), "import", "--name", "new",
                          "--manifest", str(manifest), "--dry-run"])
    assert result == (0 if accepted else 1)
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == ("validated" if accepted else "rejected")
    assert not database.exists()


def test_zip_manifest_links_holdout_text_and_activation_preserves_exclusion(
    store, tmp_path, capsys, monkeypatch,
) -> None:
    monkeypatch.setattr("run_import_corpus.PROJECT_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    held = AudioFile.new("held.wav", "legacy", "Held sentence.")
    store.insert_audio_file(held)
    store.create_canary_set(CanarySet.new("monitor", 1, "explicit", target_size=1), [held.id])
    _write_import_audio(tmp_path / "clip.wav")
    with zipfile.ZipFile(tmp_path / "clips.zip", "w") as archive:
        archive.write(tmp_path / "clip.wav", "nested/clip.wav")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "file_path;transcript\nzip:clips.zip!nested/clip.wav;Held  sentence.\n",
        encoding="utf-8",
    )
    assert import_main(["--db", str(store.db_path), "import", "--name", "imported",
                        "--manifest", str(manifest)]) == 0
    capsys.readouterr()
    assert import_main(["--db", str(store.db_path), "activate", "--name", "imported"]) == 0
    capsys.readouterr()
    assert store.count_audio_files("imported") == 1
    assert "zip:clips.zip!nested/clip.wav" in store.get_all_audio_paths()
    from utils.audio import load_audio
    waveform, sample_rate = load_audio("zip:clips.zip!nested/clip.wav")
    assert len(waveform) / sample_rate == pytest.approx(0.2)
    assert store.get_random_audio_files(10, 1) == []
    assert store.get_synthesis_source_utterances() == []


def test_corpus_import_rolls_back_cross_corpus_path_conflict(store) -> None:
    existing = AudioFile.new("existing.wav", "original", "Existing.")
    store.insert_audio_file(existing)
    corpus = Corpus.new("new-corpus")
    corpus.active = False
    with pytest.raises(ValueError, match="Conflicting existing audio"):
        store.import_corpus_audio(corpus, [
            AudioFile.new("new.wav", corpus.name, "New."),
            AudioFile.new(existing.file_path, corpus.name, "Existing."),
        ])
    assert store.count_audio_files("new-corpus") == 0
    assert [row["name"] for row in store.get_corpora()] == ["original"]
    assert store.get_audio_file(existing.id).dataset == "original"


def test_active_corpus_requires_deactivation_before_append(store) -> None:
    corpus = Corpus.new("new-corpus")
    corpus.active = False
    first = AudioFile.new("first.wav", corpus.name, "First.")
    store.import_corpus_audio(corpus, [first])
    store.set_corpus_active(corpus.name, True)
    assert store.import_corpus_audio(corpus, [first])["existing"] == 1
    with pytest.raises(ValueError, match="Deactivate"):
        store.import_corpus_audio(corpus, [AudioFile.new("second.wav", corpus.name)])
    with pytest.raises(ValueError, match="Conflicting existing audio"):
        store.import_corpus_audio(corpus, [AudioFile.new("first.wav", corpus.name, "Changed.")])
    assert store.count_audio_files(corpus.name) == 1
    store.set_corpus_active(corpus.name, False)
    assert store.import_corpus_audio(corpus, [AudioFile.new("second.wav", corpus.name)])["inserted"] == 1
    assert store.get_random_audio_files(10, 1) == []
    with pytest.raises(ValueError, match="Unknown corpus"):
        store.set_corpus_active("missing", True)


def test_conflicting_duplicate_manifest_transcripts_are_rejected(tmp_path, capsys) -> None:
    _write_import_audio(tmp_path / "clip.wav")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "file_path;transcript\nclip.wav;Original.\n./clip.wav;Changed.\n", encoding="utf-8",
    )
    database = tmp_path / "catalogue.sqlite"
    assert import_main(["--db", str(database), "import", "--name", "new",
                        "--manifest", str(manifest)]) == 1
    assert "conflicting transcripts" in json.loads(capsys.readouterr().out)["errors"][0]
    assert not database.exists()


@pytest.mark.parametrize("content", ["file_path\n", "path;transcript\nclip.wav;Text\n",
                                    "file_path;transcript\nclip.wav;Text;unexpected\n"])
def test_invalid_manifest_structure_produces_failure_report_without_database(
    tmp_path, capsys, content,
) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(content, encoding="utf-8")
    database = tmp_path / "catalogue.sqlite"
    assert import_main(["--db", str(database), "import", "--name", "new",
                        "--manifest", str(manifest)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] in {"failed", "rejected"}
    assert report["errors"]
    assert not database.exists()


def _legacy_database(path: Path) -> AudioFile:
    audio = AudioFile.new("legacy.wav", "admedvoice", "Existing sentence.")
    migrations = Path(__file__).resolve().parents[1] / "storage" / "migrations"
    with sqlite3.connect(path) as conn:
        conn.executescript((migrations / "001_initial.sql").read_text(encoding="utf-8"))
        conn.execute("ALTER TABLE audio_files ADD COLUMN excluded INTEGER NOT NULL DEFAULT 0")
        conn.execute("ALTER TABLE audio_files ADD COLUMN exclusion_reason TEXT")
        conn.execute("ALTER TABLE hypothesis_pairs ADD COLUMN reserved_until TEXT")
        for migration in sorted(migrations.glob("00[2-6]_*.sql")):
            conn.executescript(migration.read_text(encoding="utf-8"))
        conn.execute("CREATE TABLE schema_migrations (version TEXT PRIMARY KEY)")
        conn.executemany(
            "INSERT INTO schema_migrations VALUES (?)",
            [(migration.name,) for migration in sorted(migrations.glob("00[1-6]_*.sql"))],
        )
        conn.execute(
            """INSERT INTO audio_files
               (id, file_path, dataset, reference_transcript, excluded, ever_excluded, excluded_by)
               VALUES (?, ?, ?, ?, 1, 1, 'reviewer')""",
            (audio.id, audio.file_path, audio.dataset, audio.reference_transcript),
        )
        pair = HypothesisPair.new(audio.id, "A", "B", "beam", "sample", "old-policy")
        conn.execute(
            """INSERT INTO hypothesis_pairs
               (id, audio_id, hyp_a, hyp_b, strategy_a, strategy_b, model_version)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (pair.id, pair.audio_id, pair.hyp_a, pair.hyp_b, pair.strategy_a,
             pair.strategy_b, pair.model_version),
        )
        session = AnnotationSession.new("legacy reviewer", "fixture")
        monitor = CanarySet.new("legacy monitor", 1, "explicit", target_size=1)
        records = [
            ("annotation_sessions", session),
            ("annotations", Annotation.new(session.id, pair.id, "a")),
            ("reward_scores", RewardScore.new(pair.id, 0.9, 0.1, "fixture", "fixture")),
            ("audio_curation_events", AudioCurationEvent.new(
                audio.id, "exclude", "reviewer", "legacy reason", pair_id=pair.id
            )),
            ("canary_sets", monitor),
            ("canary_items", CanaryItem.new(monitor.id, audio.id)),
        ]
        for table, record in records:
            values = {key: value for key, value in asdict(record).items() if value is not None}
            conn.execute(
                f"INSERT INTO {table} ({', '.join(values)}) "
                f"VALUES ({', '.join('?' for _ in values)})",
                list(values.values()),
            )
    return audio


def test_legacy_migration_preserves_ids_exclusions_and_links(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite"
    audio = _legacy_database(path)
    tables = (
        "hypothesis_pairs", "annotation_sessions", "annotations", "reward_scores",
        "audio_curation_events", "canary_sets", "canary_items",
    )
    with sqlite3.connect(path) as conn:
        before = {table: conn.execute(f"SELECT * FROM {table}").fetchall() for table in tables}
    store = FeedbackStore(path)
    migrated = store.get_audio_file(audio.id)
    assert migrated.id == audio.id
    assert migrated.reference_transcript == audio.reference_transcript
    assert migrated.corpus_id and migrated.utterance_id
    assert store.get_synthesis_source_utterances() == []
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        for table in tables:
            assert conn.execute(f"SELECT * FROM {table}").fetchall() == before[table]
        assert conn.execute("SELECT audio_id FROM hypothesis_pairs").fetchone()[0] == audio.id
        assert conn.execute(
            "SELECT excluded, ever_excluded, excluded_by FROM audio_files"
        ).fetchone() == (1, 1, "reviewer")
    FeedbackStore(path)
    assert store.get_audio_file(audio.id).utterance_id == migrated.utterance_id


def test_failed_catalogue_migration_rolls_back_rebuild(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "legacy.sqlite"
    audio = _legacy_database(path)

    def fail_backfill(self, conn):
        raise RuntimeError("backfill interrupted")

    monkeypatch.setattr(FeedbackStore, "_backfill_catalogue", fail_backfill)
    with pytest.raises(RuntimeError, match="backfill interrupted"):
        FeedbackStore(path)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT id FROM audio_files").fetchone()[0] == audio.id
        columns = {row[1] for row in conn.execute("PRAGMA table_info(audio_files)")}
        assert "corpus_id" not in columns
        assert conn.execute(
            "SELECT COUNT(*) FROM schema_migrations WHERE version = '007_catalogue.sql'"
        ).fetchone()[0] == 0
    monkeypatch.undo()
    assert FeedbackStore(path).get_audio_file(audio.id).corpus_id


def test_concurrent_catalogue_migration_is_applied_once(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "legacy.sqlite"
    audio = _legacy_database(path)
    barrier = threading.Barrier(2)
    migrate = FeedbackStore._migrate_catalogue

    def synchronize(self, conn, migration):
        barrier.wait(timeout=10)
        migrate(self, conn, migration)

    monkeypatch.setattr(FeedbackStore, "_migrate_catalogue", synchronize)
    with ThreadPoolExecutor(max_workers=2) as executor:
        stores = list(executor.map(FeedbackStore, [path, path]))
    ids = [store.get_audio_file(audio.id).utterance_id for store in stores]
    assert ids[0] == ids[1]
    with sqlite3.connect(path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM schema_migrations WHERE version = '007_catalogue.sql'"
        ).fetchone()[0] == 1
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_generic_corpora_share_conservative_text_identity(store) -> None:
    corpus = store.register_corpus(Corpus.new("another-polish-corpus"))
    first = AudioFile.new("first.wav", corpus["name"], "A  sentence.", corpus_id=corpus["id"])
    second = AudioFile.new("second.wav", "commonvoice", " A\nsentence. ")
    different = AudioFile.new("different.wav", "commonvoice", "a sentence.")
    assert store.bulk_insert_audio_files([first, second, different, first]) == 3
    assert store.get_audio_file(first.id).utterance_id == store.get_audio_file(second.id).utterance_id
    assert store.get_audio_file(first.id).utterance_id != store.get_audio_file(different.id).utterance_id
    assert store.count_audio_files(corpus["name"]) == 1


def test_unicode_whitespace_normalization_preserves_numbers_and_punctuation(store) -> None:
    first = AudioFile.new("first.wav", "generic", "Caf\u00e9  1.5 mg.")
    same = AudioFile.new("same.wav", "generic", "Cafe\u0301\n1.5 mg.")
    different = AudioFile.new("different.wav", "generic", "Caf\u00e9 15 mg.")
    store.bulk_insert_audio_files([first, same, different])
    assert store.get_audio_file(first.id).utterance_id == store.get_audio_file(same.id).utterance_id
    assert store.get_audio_file(first.id).utterance_id != store.get_audio_file(different.id).utterance_id


def test_holdout_text_blocks_all_learning_paths_and_synthesis_sources(store) -> None:
    held = AudioFile.new("held.wav", "admedvoice", "A sentence.")
    related = AudioFile.new("related.wav", "commonvoice", " A  sentence. ")
    unrelated = AudioFile.new("unrelated.wav", "new-corpus", "Another sentence.")
    store.bulk_insert_audio_files([held, related, unrelated])
    pair = make_pair(store, related.id)
    session = make_session(store)
    annotate(store, session.id, pair.id, "a", correction_text="A sentence.")
    pending = make_pair(store, unrelated.id)
    monitor = CanarySet.new("holdout", 1, "explicit", target_size=1)
    # A related utterance with human feedback cannot become a prospective holdout.
    with pytest.raises(ValueError, match="ineligible"):
        store.create_canary_set(monitor, [held.id])
    # Simulate a previously reserved holdout, then imported feedback on a sibling.
    with store._conn() as conn:
        conn.execute(
            "INSERT INTO canary_sets (id, name, selection_seed, selection_method, target_size) "
            "VALUES (?, 'legacy monitor', 1, 'explicit', 1)", (monitor.id,),
        )
        item = CanaryItem.new(monitor.id, held.id)
        conn.execute(
            "INSERT INTO canary_items (id, canary_set_id, audio_id) VALUES (?, ?, ?)",
            (item.id, monitor.id, held.id),
        )
    assert [af.id for af in store.get_unqueued_audio_files(10, "new-policy")] == [unrelated.id]
    assert [af.id for af in store.get_random_audio_files(10, 1)] == [unrelated.id]
    assert store.get_eligible_audio_files_by_ids([held.id, related.id], "new-policy") == []
    assert store.get_preference_dataset() == []
    assert store.get_sft_corrections() == []
    assert [row["text"] for row in store.get_synthesis_source_utterances()] == ["Another sentence."]
    assert store.get_and_reserve_pair().id == pending.id


def test_verified_monitor_text_and_synthetic_origin_are_excluded(store) -> None:
    held = AudioFile.new("held.wav", "admedvoice", "Original text.")
    corrected = AudioFile.new("corrected.wav", "another-corpus", "Verified text.")
    synthetic = AudioFile.new(
        "tts.wav", "tts-corpus", "Training text.", origin="synthesized",
        provenance_json='{"provider": "fixture", "engine": "fixture-engine"}',
    )
    store.bulk_insert_audio_files([held, corrected, synthetic])
    items = store.create_canary_set(
        CanarySet.new("monitor", 1, "explicit", target_size=1), [held.id]
    )
    store.save_canary_transcript(items[0].id, " Verified  text. ")
    assert store.get_random_audio_files(10, 1) == []
    assert store.get_synthesis_source_utterances() == []
    assert store.get_audio_file(synthetic.id).origin == "synthesized"


def test_catalogue_export_merge_remaps_natural_identities(store, tmp_path: Path) -> None:
    incoming = AudioFile.new("incoming.wav", "custom-corpus", "Same text.")
    store.insert_audio_file(incoming)
    export = tmp_path / "export.json"
    store.export_to_json(export)
    target = FeedbackStore(tmp_path / "target.sqlite")
    existing = AudioFile.new("existing.wav", "custom-corpus", " Same  text. ")
    target.insert_audio_file(existing)
    target.merge_from_export(export)
    assert target.get_audio_file(incoming.id).corpus_id == target.get_audio_file(existing.id).corpus_id
    assert target.get_audio_file(incoming.id).utterance_id == target.get_audio_file(existing.id).utterance_id
    assert all(value == 0 for value in target.merge_from_export(export).values())
    legacy = json.loads(export.read_text(encoding="utf-8"))
    legacy.pop("corpora")
    legacy.pop("utterances")
    for row in legacy["audio_files"]:
        for key in ("corpus_id", "utterance_id", "origin", "provenance_json"):
            row.pop(key)
    export.write_text(json.dumps(legacy), encoding="utf-8")
    legacy_target = FeedbackStore(tmp_path / "legacy-import.sqlite")
    legacy_target.merge_from_export(export)
    assert legacy_target.get_audio_file(incoming.id).utterance_id
