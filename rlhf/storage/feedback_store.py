"""
FeedbackStore -- SQLite-backed persistence layer for all annotation and training data.

Single point of contact for the database. Never construct raw SQL outside this
module. All primary keys are UUID4 strings (cross-machine merge safe).

Usage:
    from storage.feedback_store import FeedbackStore
    store = FeedbackStore(paths.db_path)
    store.insert_audio_file(AudioFile.new("path/to/file.wav", "admedvoice"))
"""

from __future__ import annotations

import json
import hashlib
import math
import random
import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Dict, Generator, List, Optional, Union

from .schema import (
    Annotation,
    AnnotationModerationEvent,
    AnnotationSession,
    AudioCurationEvent,
    AudioFile,
    AudioQualityReview,
    CanaryEvaluation,
    CanaryItem,
    CanarySet,
    Corpus,
    HypothesisPair,
    ModelCheckpoint,
    PolicyActivation,
    RewardAuditItem,
    RewardAuditJob,
    RewardScore,
    TrainingRun,
    TTSCampaign,
    TTSItem,
    Utterance,
    normalize_utterance_text,
)

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"

# Fixed session ID used for all WER-based auto-labeled annotations.
_AUTO_SESSION_ID = "system-auto-label"


class FeedbackStore:
    """
    Thread-safe-ish SQLite store (WAL mode, new connection per call).
    Each public method opens and closes its own connection, making it safe
    to call from background threads (e.g. queue refill thread in the GUI).
    """

    def __init__(self, db_path: Union[str, Path]) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def clone_to(self, destination: Union[str, Path]) -> None:
        """Create a transactionally consistent copy of this database."""
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError(f"Database clone destination exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        if temporary.exists():
            raise FileExistsError(f"Database clone staging file exists: {temporary}")
        try:
            with self._conn() as source:
                target = sqlite3.connect(str(temporary))
                try:
                    source.backup(target)
                finally:
                    target.close()
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    # -- Internal connection helper ---------------------------------------------

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path), timeout=15)
        conn.row_factory = sqlite3.Row
        conn.create_function(
            "normalize_utterance_text", 1, normalize_utterance_text, deterministic=True
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Apply ordered SQL migrations and create runtime coordination tables."""
        with self._conn() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS schema_migrations (
                       version    TEXT PRIMARY KEY,
                       applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
                   )""")
            applied = {
                row[0]
                for row in conn.execute(
                    "SELECT version FROM schema_migrations"
                ).fetchall()
            }
            for migration in sorted(_MIGRATIONS_DIR.glob("[0-9][0-9][0-9]_*.sql")):
                if migration.name in applied:
                    continue
                if migration.name == "007_catalogue.sql":
                    self._migrate_catalogue(conn, migration)
                    continue
                if migration.name in {
                    "008_tts_campaigns.sql", "009_corpus_purpose.sql", "010_audio_quality.sql",
                }:
                    self._migrate_atomic(conn, migration)
                    continue
                conn.executescript(migration.read_text(encoding="utf-8"))
                conn.execute(
                    "INSERT INTO schema_migrations (version) VALUES (?)",
                    (migration.name,),
                )
                if migration.name == "001_initial.sql":
                    audio_columns = {
                        row[1]
                        for row in conn.execute("PRAGMA table_info(audio_files)").fetchall()
                    }
                    if "excluded" not in audio_columns:
                        conn.execute(
                            "ALTER TABLE audio_files ADD COLUMN excluded INTEGER NOT NULL DEFAULT 0"
                        )
                    if "exclusion_reason" not in audio_columns:
                        conn.execute(
                            "ALTER TABLE audio_files ADD COLUMN exclusion_reason TEXT"
                        )
            # Incremental schema additions (ALTER TABLE is idempotent via try/except)
            for stmt in [
                "ALTER TABLE audio_files ADD COLUMN excluded INTEGER NOT NULL DEFAULT 0",
                "ALTER TABLE audio_files ADD COLUMN exclusion_reason TEXT",
                "ALTER TABLE hypothesis_pairs ADD COLUMN reserved_until TEXT",
            ]:
                try:
                    conn.execute(stmt)
                except Exception:
                    pass  # column already exists
            # Admin / coordination tables (idempotent)
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS settings (
                    key        TEXT PRIMARY KEY,
                    value      TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                INSERT OR IGNORE INTO settings (key, value) VALUES
                    ('max_concurrent_sessions', '10'),
                    ('training_threshold',      '50'),
                    ('training_status',         'idle'),
                    ('training_requested',      'false'),
                    ('training_error',          ''),
                    ('training_sft_included',   'true'),
                    ('annotations_at_last_training', '0'),
                    ('admin_session_id',        ''),
                    ('queue_min_size',          '20'),
                    ('queue_refill_batch',      '50'),
                    ('queue_refill_status',     'idle'),
                    ('queue_refill_log',        ''),
                    ('queue_refill_requested',  'false'),
                    ('show_reference',          'true'),
                    ('show_wer',                'true'),
                    ('tts_sft_share_percent',   '0'),
                    ('tts_refill_share_percent','0');

                CREATE TABLE IF NOT EXISTS active_sessions (
                    session_id     TEXT PRIMARY KEY,
                    participant_id TEXT NOT NULL,
                    machine_id     TEXT NOT NULL DEFAULT '',
                    last_seen      TEXT NOT NULL,
                    started_at     TEXT NOT NULL DEFAULT (datetime('now'))
                );
            """)

    def _migrate_catalogue(self, conn: sqlite3.Connection, migration: Path) -> None:
        conn.commit()
        conn.execute("PRAGMA foreign_keys=OFF")
        try:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute(
                "SELECT 1 FROM schema_migrations WHERE version = ?", (migration.name,)
            ).fetchone():
                conn.commit()
                return
            statement = ""
            for line in migration.read_text(encoding="utf-8").splitlines(keepends=True):
                statement += line
                if sqlite3.complete_statement(statement):
                    conn.execute(statement)
                    statement = ""
            if statement.strip():
                raise ValueError(f"Incomplete migration statement: {migration.name}")
            self._backfill_catalogue(conn)
            violations = conn.execute("PRAGMA foreign_key_check").fetchall()
            if violations:
                raise ValueError(f"Catalogue migration has broken references: {violations}")
            conn.execute(
                "INSERT INTO schema_migrations (version) VALUES (?)", (migration.name,)
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.execute("PRAGMA foreign_keys=ON")

    # -- Catalogue --------------------------------------------------------------

    def _migrate_atomic(self, conn, migration):
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        try:
            if conn.execute("SELECT 1 FROM schema_migrations WHERE version=?",
                            (migration.name,)).fetchone():
                conn.commit()
                return
            statement = ""
            for line in migration.read_text(encoding="utf-8").splitlines(keepends=True):
                statement += line
                if sqlite3.complete_statement(statement):
                    conn.execute(statement)
                    statement = ""
            if statement.strip():
                raise ValueError(f"Incomplete migration: {migration.name}")
            if conn.execute("PRAGMA foreign_key_check").fetchall():
                raise ValueError(f"Migration has broken references: {migration.name}")
            conn.execute("INSERT INTO schema_migrations(version) VALUES (?)", (migration.name,))
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def register_corpus(self, corpus: Corpus) -> dict:
        with self._conn() as conn:
            return self._register_corpus(conn, corpus)

    @staticmethod
    def _register_corpus(conn: sqlite3.Connection, corpus: Corpus) -> dict:
        # Migration 007 backfills corpora before migration 009 adds their purpose.
        has_purpose = any(row[1] == "purpose" for row in conn.execute("PRAGMA table_info(corpora)"))
        if not has_purpose and corpus.purpose != "training":
            raise ValueError("Evaluation corpora require the purpose migration")
        existing = conn.execute(
            "SELECT * FROM corpora WHERE name = ?", (corpus.name,)
        ).fetchone()
        if existing is not None:
            if any(
                existing[key] != getattr(corpus, key)
                for key in ("origin", "language", "provider") + (("purpose",) if has_purpose else ())
            ):
                raise ValueError(f"Conflicting corpus metadata: {corpus.name}")
            return dict(existing)
        values = (corpus.id, corpus.name, corpus.origin, corpus.language,
                  corpus.provider, int(corpus.active))
        if has_purpose:
            conn.execute(
                """INSERT INTO corpora (id, name, origin, language, provider, active, purpose)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""", values + (corpus.purpose,),
            )
        else:
            conn.execute(
                """INSERT INTO corpora (id, name, origin, language, provider, active)
                   VALUES (?, ?, ?, ?, ?, ?)""", values,
            )
        return dict(conn.execute("SELECT * FROM corpora WHERE id = ?", (corpus.id,)).fetchone())

    def get_corpora(self) -> List[Dict]:
        with self._conn() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM corpora ORDER BY name")]

    def get_corpus_summaries(self) -> List[Dict]:
        with self._conn() as conn:
            return [dict(row) for row in conn.execute(
                """SELECT c.*, COUNT(af.id) AS audio_count,
                          COUNT(DISTINCT af.utterance_id) AS utterance_count,
                          COUNT(eligible.id) AS eligible_count
                   FROM corpora c
                   LEFT JOIN audio_files af ON af.corpus_id = c.id
                   LEFT JOIN learning_audio_files eligible ON eligible.id = af.id
                   GROUP BY c.id ORDER BY c.name"""
            )]

    def get_corpus_audio_files(
        self, corpus_id: str, limit: int = 25, offset: int = 0,
    ) -> List[Dict]:
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("Corpus preview requires limit 1-100 and nonnegative offset")
        with self._conn() as conn:
            return [dict(row) for row in conn.execute(
                """SELECT af.*, (eligible.id IS NOT NULL) AS learning_eligible
                   FROM audio_files af
                   LEFT JOIN learning_audio_files eligible ON eligible.id = af.id
                   WHERE af.corpus_id = ?
                   ORDER BY af.created_at, af.id LIMIT ? OFFSET ?""",
                (corpus_id, limit, offset),
            )]

    def import_corpus_audio(self, corpus: Corpus, audio_files: List[AudioFile]) -> Dict:
        """Atomically import validated assets without activating a new corpus."""
        if not audio_files:
            raise ValueError("A corpus import must contain audio files")
        if corpus.origin != "recorded":
            raise ValueError("This importer supports recorded corpora only")
        if corpus.active:
            raise ValueError("New corpus imports must be inactive")
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            registered = self._register_corpus(conn, corpus)
            inserted = 0
            for audio in audio_files:
                if audio.dataset != registered["name"] or audio.origin != corpus.origin:
                    raise ValueError("Imported audio must match the recorded corpus")
                existing = conn.execute(
                    "SELECT * FROM audio_files WHERE file_path = ?", (audio.file_path,)
                ).fetchone()
                if existing is not None:
                    if (
                        existing["corpus_id"] != registered["id"]
                        or existing["origin"] != audio.origin
                        or normalize_utterance_text(existing["reference_transcript"])
                        != normalize_utterance_text(audio.reference_transcript)
                    ):
                        raise ValueError(f"Conflicting existing audio: {audio.file_path}")
                    continue
                if registered["active"]:
                    raise ValueError("Deactivate the corpus before appending new audio")
                inserted += self._insert_audio_file(
                    conn, replace(audio, corpus_id=registered["id"])
                )
            return {
                "corpus_id": registered["id"],
                "active": bool(registered["active"]),
                "inserted": inserted,
                "existing": len(audio_files) - inserted,
            }

    def set_corpus_active(self, name: str, active: bool) -> Dict:
        with self._conn() as conn:
            corpus = conn.execute("SELECT purpose FROM corpora WHERE name = ?", (name,)).fetchone()
            if corpus and corpus["purpose"] == "evaluation":
                raise ValueError("Evaluation corpora cannot be activated for learning")
            cursor = conn.execute(
                "UPDATE corpora SET active = ? WHERE name = ?", (int(active), name)
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Unknown corpus: {name}")
            return dict(conn.execute("SELECT * FROM corpora WHERE name = ?", (name,)).fetchone())

    @staticmethod
    def _audio_fingerprint(file_path: str) -> str:
        from utils.audio import load_audio

        waveform, _ = load_audio(file_path)
        return hashlib.sha256(waveform.tobytes()).hexdigest()

    def _validate_uploaded_corpus(
        self, conn: sqlite3.Connection, corpus: Corpus, audio_files: List[AudioFile],
    ) -> None:
        if corpus.active or corpus.origin != "recorded" or corpus.language != "pl":
            raise ValueError("Uploads require an inactive Polish recorded corpus")
        if not audio_files:
            raise ValueError("No recordings to import")
        if conn.execute("SELECT 1 FROM corpora WHERE name = ?", (corpus.name,)).fetchone():
            raise ValueError("Corpus name already exists; choose a new name")
        references = set()
        fingerprints = set()
        for index, audio in enumerate(audio_files, 2):
            reference = normalize_utterance_text(audio.reference_transcript)
            if corpus.purpose == "evaluation" and not reference:
                raise ValueError(f"Row {index}: evaluation transcript is required")
            fingerprint = self._audio_fingerprint(audio.file_path)
            if fingerprint in fingerprints:
                raise ValueError(f"Row {index}: duplicate audio content")
            fingerprints.add(fingerprint)
            if reference:
                references.add(reference)
        existing = conn.execute(
            """SELECT af.file_path, af.reference_transcript, ci.transcript,
                      (ci.id IS NOT NULL OR c.purpose = 'evaluation') AS held
               FROM audio_files af JOIN corpora c ON c.id = af.corpus_id
               LEFT JOIN canary_items ci ON ci.audio_id = af.id"""
        ).fetchall()
        if corpus.purpose == "evaluation":
            corrections = conn.execute(
                "SELECT correction_text FROM annotations WHERE correction_text IS NOT NULL"
            ).fetchall()
            if any(normalize_utterance_text(row["correction_text"]) in references for row in corrections):
                raise ValueError("Transcript overlaps a learning correction")
        checked_paths = set()
        for row in existing:
            texts = {normalize_utterance_text(row["reference_transcript"]),
                     normalize_utterance_text(row["transcript"])}
            if (corpus.purpose == "evaluation" or row["held"]) and references.intersection(texts):
                raise ValueError("Transcript overlaps existing evaluation/learning data")
            if row["file_path"] not in checked_paths:
                checked_paths.add(row["file_path"])
                if self._audio_fingerprint(row["file_path"]) in fingerprints:
                    raise ValueError("Audio content overlaps existing evaluation/learning data")

    def validate_uploaded_corpus(self, corpus: Corpus, audio_files: List[AudioFile]) -> None:
        with self._conn() as conn:
            self._validate_uploaded_corpus(conn, corpus, audio_files)

    def import_uploaded_corpus(self, corpus: Corpus, audio_files: List[AudioFile]) -> Dict:
        """Commit reviewed uploads and reserve evaluation items in one transaction."""
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._validate_uploaded_corpus(conn, corpus, audio_files)
            registered = self._register_corpus(conn, corpus)
            for audio in audio_files:
                if audio.dataset != corpus.name or audio.origin != "recorded":
                    raise ValueError("Uploaded audio must match the recorded corpus")
                if not self._insert_audio_file(conn, replace(audio, corpus_id=registered["id"])):
                    raise ValueError(f"Recording path already exists: {audio.file_path}")
            monitor_id = None
            if corpus.purpose == "evaluation":
                monitor = CanarySet.new(corpus.name, 0, "imported_manifest",
                                        target_size=len(audio_files))
                monitor_id = monitor.id
                conn.execute(
                    """INSERT INTO canary_sets
                       (id, name, selection_seed, selection_method, target_size, status)
                       VALUES (?, ?, ?, 'imported_manifest', ?, 'collecting')""",
                    (monitor.id, monitor.name, monitor.selection_seed, monitor.target_size),
                )
                for audio in audio_files:
                    item = CanaryItem.new(monitor.id, audio.id, gt_available=True)
                    conn.execute(
                        """INSERT INTO canary_items
                           (id, canary_set_id, audio_id, gt_available, gt_revealed)
                           VALUES (?, ?, ?, 1, 0)""",
                        (item.id, monitor.id, audio.id),
                    )
            return {"corpus_id": registered["id"], "monitor_id": monitor_id}

    def get_training_monitor_rows(self) -> Optional[Dict]:
        """Snapshot the designated frozen monitor and its verified references."""
        with self._conn() as conn:
            monitor = conn.execute(
                "SELECT * FROM canary_sets WHERE is_promotion_gate = 1 AND status = 'ready'"
            ).fetchone()
            if monitor is None:
                return None
            rows = [dict(row) for row in conn.execute(
                """SELECT af.file_path, ci.transcript AS reference, ci.verified_at
                   FROM canary_items ci JOIN audio_files af ON af.id = ci.audio_id
                   WHERE ci.canary_set_id = ? ORDER BY ci.id""", (monitor["id"],)
            )]
            if not rows or any(not row["reference"] or not row["verified_at"] for row in rows):
                raise ValueError("Frozen training evaluation monitor has invalid references")
            return {"id": monitor["id"], "fingerprint": monitor["fingerprint"], "rows": rows}

    @staticmethod
    def _register_utterance(conn: sqlite3.Connection, utterance: Utterance) -> str:
        conn.execute(
            """INSERT OR IGNORE INTO utterances (id, text, normalized_text, language)
               VALUES (?, ?, ?, ?)""",
            (utterance.id, utterance.text, utterance.normalized_text, utterance.language),
        )
        row = conn.execute(
            "SELECT id FROM utterances WHERE language = ? AND normalized_text = ?",
            (utterance.language, utterance.normalized_text),
        ).fetchone()
        if row is None:
            raise ValueError(f"Conflicting utterance ID: {utterance.id}")
        return row["id"]

    def _catalogue_links(self, conn: sqlite3.Connection, af: AudioFile) -> tuple[str, Optional[str]]:
        if af.corpus_id is None:
            corpus = conn.execute(
                "SELECT * FROM corpora WHERE name = ?", (af.dataset,)
            ).fetchone()
            if corpus is None:
                corpus = self._register_corpus(conn, Corpus.new(af.dataset, af.origin))
        else:
            corpus = conn.execute(
                "SELECT * FROM corpora WHERE id = ?", (af.corpus_id,)
            ).fetchone()
        if corpus is None or corpus["name"] != af.dataset:
            raise ValueError(f"Audio has an unknown or mismatched corpus: {af.dataset}")
        if af.origin not in {"recorded", "synthesized"} or corpus["origin"] not in {
            af.origin, "mixed"
        }:
            raise ValueError(f"Audio origin does not match corpus: {af.dataset}")
        if af.provenance_json is not None:
            provenance = json.loads(af.provenance_json)
            if not isinstance(provenance, dict):
                raise ValueError("Audio provenance must be a JSON object")
        normalized = normalize_utterance_text(af.reference_transcript)
        if af.utterance_id is not None:
            utterance = conn.execute(
                "SELECT * FROM utterances WHERE id = ?", (af.utterance_id,)
            ).fetchone()
            if utterance is None or utterance["language"] != corpus["language"]:
                raise ValueError(f"Unknown or mismatched utterance: {af.utterance_id}")
            if normalized != utterance["normalized_text"]:
                raise ValueError("Audio transcript does not match its utterance")
            utterance_id = af.utterance_id
        elif af.reference_transcript is not None and normalized is not None:
            utterance_id = self._register_utterance(
                conn, Utterance.new(af.reference_transcript, corpus["language"])
            )
        else:
            utterance_id = None
        return corpus["id"], utterance_id

    def _backfill_catalogue(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute("SELECT * FROM audio_files WHERE corpus_id IS NULL").fetchall()
        for row in rows:
            corpus_id, utterance_id = self._catalogue_links(conn, _row_to_audio_file(row))
            conn.execute(
                "UPDATE audio_files SET corpus_id = ?, utterance_id = ? WHERE id = ?",
                (corpus_id, utterance_id, row["id"]),
            )

    def get_synthesis_source_utterances(self, corpus_id: Optional[str] = None) -> List[Dict]:
        """Return unique eligible recorded texts; synthesized audio is not a source."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT DISTINCT u.* FROM utterances u
                   JOIN learning_audio_files af ON af.utterance_id = u.id
                   WHERE af.origin = 'recorded' AND (? IS NULL OR af.corpus_id = ?)
                   ORDER BY u.id""",
                (corpus_id, corpus_id),
            ).fetchall()
        return [dict(row) for row in rows]

    # -- Persistent TTS campaigns ----------------------------------------------

    def _tts_plan(self, conn, corpus_id, profiles, mode, value, provider_url):
        corpus = conn.execute("SELECT * FROM corpora WHERE id = ?", (corpus_id,)).fetchone()
        if not corpus or not corpus["active"] or corpus["language"] != "pl":
            raise ValueError("Choose an active Polish source corpus")
        if mode not in {"count", "coverage", "all"} or not math.isfinite(float(value)):
            raise ValueError("Invalid selection mode/value")
        if mode == "count" and (value < 1 or int(value) != value):
            raise ValueError("Count must be a positive integer")
        if mode == "coverage" and not 0 < value <= 100:
            raise ValueError("Coverage must be in (0, 100]")
        if not profiles or len({p["id"] for p in profiles}) != len(profiles):
            raise ValueError("Select distinct engine/voice profiles")
        source = [dict(r) for r in conn.execute(
            """SELECT DISTINCT u.* FROM utterances u
               JOIN eligible_catalogue_audio af ON af.utterance_id = u.id
               WHERE af.corpus_id = ? AND af.origin = 'recorded' ORDER BY u.id""",
            (corpus_id,))]
        from tts.provider import is_local_provider
        excluded_long = 0
        if is_local_provider(provider_url):
            excluded_long = sum(len(utterance["text"]) > 280 for utterance in source)
            source = [utterance for utterance in source if len(utterance["text"]) <= 280]
        existing = {(r["utterance_id"], r["profile_id"]): dict(r) for r in conn.execute(
            "SELECT * FROM tts_items WHERE provider_url = ?", (provider_url,))}
        if mode == "count":
            selected = [u for u in source if any(
                (u["id"], p["id"]) not in existing for p in profiles)][:int(value)]
        else:
            selected = source[:math.ceil(len(source) * value / 100)] if mode == "coverage" else source
        work = [(u, p, existing.get((u["id"], p["id"]))) for u in selected for p in profiles]
        return {"eligible_utterances": len(source), "selected_utterances": len(selected),
                "combinations": len(work), "missing": sum(old is None for _, _, old in work),
                "excluded_texts_over_limit": excluded_long}, work

    def preview_tts_campaign(self, corpus_id, profiles, mode="all", value=100, provider_url=""):
        with self._conn() as conn:
            return self._tts_plan(conn, corpus_id, profiles, mode, value, provider_url)[0]

    def create_tts_campaign(self, corpus_id, profiles, intended_use="both",
                            mode="all", value=100, provider_url=""):
        from tts.provider import validate_provider_url, validate_profiles

        provider_url = validate_provider_url(provider_url)
        profiles = validate_profiles(profiles, provider_url=provider_url)
        if intended_use not in {"sft", "annotation", "both"}:
            raise ValueError("Invalid campaign intended use")
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            summary, work = self._tts_plan(conn, corpus_id, profiles, mode, value, provider_url)
            campaign = TTSCampaign.new(corpus_id, intended_use, mode, value,
                                       json.dumps(profiles), provider_url)
            conn.execute("""INSERT INTO tts_campaigns
                (id,source_corpus_id,intended_use,selection_mode,selection_value,profiles_json,provider_url)
                VALUES (?,?,?,?,?,?,?)""",
                (campaign.id, corpus_id, intended_use, mode, value, campaign.profiles_json, provider_url))
            for utterance, profile, existing in work:
                if existing:
                    if json.loads(existing["profile_json"]) != profile:
                        raise ValueError("Profile contract changed; use a new profile ID")
                    item_id = existing["id"]
                else:
                    item = TTSItem.new(utterance["id"], profile["id"], json.dumps(profile), provider_url)
                    item_id = item.id
                    conn.execute("""INSERT INTO tts_items
                        (id,utterance_id,profile_id,profile_json,provider_url) VALUES (?,?,?,?,?)""",
                        (item.id, item.utterance_id, item.profile_id, item.profile_json, item.provider_url))
                conn.execute("INSERT INTO tts_campaign_members VALUES (?,?)", (campaign.id, item_id))
            return dict(summary, id=campaign.id)

    def get_tts_campaigns(self):
        with self._conn() as conn:
            campaigns = [dict(r) for r in conn.execute(
                """SELECT tc.*, c.name AS source_name FROM tts_campaigns tc
                   JOIN corpora c ON c.id = tc.source_corpus_id ORDER BY tc.created_at DESC, tc.id""")]
            for campaign in campaigns:
                campaign["counts"] = {r["state"]: r["n"] for r in conn.execute(
                    """SELECT ti.state, COUNT(*) n FROM tts_items ti
                       JOIN tts_campaign_members tm ON tm.item_id = ti.id
                       WHERE tm.campaign_id = ? GROUP BY ti.state""", (campaign["id"],))}
            return campaigns

    def get_tts_campaign_items(self, campaign_id, limit=100, offset=0):
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("Invalid item page")
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(
                """SELECT ti.*, u.text FROM tts_items ti
                   JOIN utterances u ON u.id = ti.utterance_id
                   JOIN tts_campaign_members tm ON tm.item_id = ti.id
                   WHERE tm.campaign_id = ? ORDER BY ti.created_at, ti.id LIMIT ? OFFSET ?""",
                (campaign_id, limit, offset))]

    def request_audio_quality(self, audio_id: str) -> str:
        review = AudioQualityReview.new(audio_id)
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            audio = conn.execute("SELECT origin FROM audio_files WHERE id=?", (audio_id,)).fetchone()
            if audio is None or audio["origin"] != "synthesized":
                raise ValueError("Quality review requires imported synthesized audio")
            if conn.execute(
                "SELECT 1 FROM audio_quality_reviews WHERE audio_id=? AND state IN ('queued','running')",
                (audio_id,),
            ).fetchone():
                raise ValueError("Quality assessment already queued or running")
            conn.execute(
                "INSERT INTO audio_quality_reviews(id,audio_id) VALUES (?,?)",
                (review.id, audio_id),
            )
        return review.id

    def claim_audio_quality(self) -> Optional[Dict]:
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """SELECT q.*, af.file_path FROM audio_quality_reviews q
                   JOIN audio_files af ON af.id=q.audio_id
                   WHERE q.state='queued' ORDER BY q.created_at,q.id LIMIT 1"""
            ).fetchone()
            if row:
                conn.execute("UPDATE audio_quality_reviews SET state='running' WHERE id=?", (row["id"],))
        return dict(row) if row else None

    def finish_audio_quality(
        self, review_id: str, *, profile: Optional[str] = None,
        metrics: Optional[Dict] = None, error: Optional[str] = None,
    ) -> None:
        if error is None and (not profile or not metrics):
            raise ValueError("Completed quality assessment requires profile and metrics")
        serialized = json.dumps(metrics, allow_nan=False) if metrics is not None else None
        with self._conn() as conn:
            if conn.execute(
                """UPDATE audio_quality_reviews SET state=?,profile=?,metrics_json=?,error=?,
                   completed_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')
                   WHERE id=? AND state='running'""",
                ("failed" if error else "completed", profile, serialized, error, review_id),
            ).rowcount != 1:
                raise ValueError("Quality assessment is no longer running")

    def fail_interrupted_audio_quality(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE audio_quality_reviews SET state='failed',
                   error='Quality dispatcher restarted during assessment; retry explicitly'
                   WHERE state='running'"""
            )

    def get_audio_quality(self, audio_ids: List[str]) -> Dict:
        if not audio_ids:
            return {}
        if len(audio_ids) > 100:
            raise ValueError("Quality preview is limited to 100 recordings")
        with self._conn() as conn:
            rows = conn.execute(
                f"""SELECT * FROM audio_quality_reviews WHERE audio_id IN
                    ({','.join('?' for _ in audio_ids)}) ORDER BY created_at DESC, rowid DESC""",
                audio_ids,
            ).fetchall()
        result = {}
        for row in rows:
            result.setdefault(row["audio_id"], dict(row))
        return result

    def review_audio_quality(self, review_id: str, notes: str, actor: str = "admin") -> None:
        with self._conn() as conn:
            if conn.execute(
                """UPDATE audio_quality_reviews
                   SET reviewed_at=strftime('%Y-%m-%dT%H:%M:%fZ','now'),reviewer=?,notes=?
                   WHERE id=? AND state='completed' AND reviewed_at IS NULL""",
                (actor, notes.strip()[:2000], review_id),
            ).rowcount != 1:
                raise ValueError("Only completed, unreviewed scores can be marked reviewed")

    def cancel_tts_campaign(self, campaign_id):
        """Cancel local unsubmitted work only; accepted jobs still get collected."""
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("UPDATE tts_campaigns SET status='cancelled' WHERE id=?",
                            (campaign_id,)).rowcount != 1:
                raise ValueError("Unknown campaign")
            conn.execute("""UPDATE tts_items SET state='cancelled', error='Local cancellation'
                WHERE state='pending' AND id IN
                    (SELECT item_id FROM tts_campaign_members WHERE campaign_id=?)
                AND NOT EXISTS (
                    SELECT 1 FROM tts_campaign_members tm JOIN tts_campaigns tc ON tc.id=tm.campaign_id
                    WHERE tm.item_id=tts_items.id AND tc.status='active')""", (campaign_id,))

    def review_tts_item(self, item_id, action, actor="admin", job_id=None, confirm_duplicate=False):
        if action not in {"attach_job", "retry", "cancel"}:
            raise ValueError("Invalid review action")
        if action == "retry" and not confirm_duplicate:
            raise ValueError("Explicit duplicate-risk acknowledgement is required")
        if action == "attach_job" and (not job_id or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", job_id)):
            raise ValueError("Invalid provider job ID")
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            item = conn.execute("SELECT * FROM tts_items WHERE id=?", (item_id,)).fetchone()
            if not item or item["state"] not in {"review_required", "failed", "cancelled"}:
                raise ValueError("Only failed, cancelled or uncertain items can be reviewed")
            if action == "attach_job" and conn.execute(
                "SELECT 1 FROM tts_items WHERE provider_url=? AND job_id=? AND id!=?",
                (item["provider_url"], job_id, item_id)).fetchone():
                raise ValueError("Provider job ID already belongs to a different utterance/profile item")
            if action == "retry" and not conn.execute(
                """SELECT 1 FROM tts_campaign_members tm JOIN tts_campaigns tc ON tc.id=tm.campaign_id
                   WHERE tm.item_id=? AND tc.status='active'""", (item_id,)).fetchone():
                raise ValueError("Create an active coverage/all campaign including this item before retrying")
            history = json.loads(item["review_log_json"])
            history.append({"actor": actor, "action": action, "previous_job_id": item["job_id"],
                            "previous_error": item["error"], "at": time.time()})
            state = {"attach_job": "accepted", "retry": "pending", "cancel": "cancelled"}[action]
            conn.execute("""UPDATE tts_items SET state=?, job_id=?, attempts=0, get_failures=0,
                lease_until=0, next_attempt_at=0, error=NULL, review_log_json=?,
                updated_at=datetime('now') WHERE id=?""",
                (state, job_id if action == "attach_job" else None, json.dumps(history), item_id))

    def claim_tts_item(self, now=None, lease_seconds=360):
        now = time.time() if now is None else now
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("""UPDATE tts_items SET state='review_required', lease_until=0,
                error='Interrupted submission: outcome unknown; operator review required'
                WHERE state='submitting' AND lease_until <= ?""", (now,))
            conn.execute("""UPDATE tts_items SET state='accepted', lease_until=0
                WHERE state='polling' AND lease_until <= ?""", (now,))
            # Exclusions are rechecked immediately before external submission.
            conn.execute("""UPDATE tts_items SET state='cancelled', error='Source no longer eligible'
                WHERE state='pending' AND NOT EXISTS (
                    SELECT 1 FROM tts_campaign_members tm JOIN tts_campaigns tc ON tc.id=tm.campaign_id
                    JOIN eligible_catalogue_audio af ON af.utterance_id=tts_items.utterance_id
                    WHERE tm.item_id=tts_items.id AND tc.status='active'
                      AND af.corpus_id=tc.source_corpus_id AND af.origin='recorded')""")
            row = conn.execute("""SELECT ti.*, u.text FROM tts_items ti
                JOIN utterances u ON u.id=ti.utterance_id
                WHERE ti.state IN ('pending','accepted') AND ti.next_attempt_at <= ?
                  AND (ti.state='accepted' OR EXISTS (
                    SELECT 1 FROM tts_campaign_members tm JOIN tts_campaigns tc ON tc.id=tm.campaign_id
                    WHERE tm.item_id=ti.id AND tc.status='active'))
                ORDER BY CASE ti.state WHEN 'accepted' THEN 0 ELSE 1 END, ti.created_at, ti.id LIMIT 1""",
                (now,)).fetchone()
            if not row:
                return None
            state = "submitting" if row["state"] == "pending" else "polling"
            conn.execute("UPDATE tts_items SET state=?, lease_until=?, updated_at=datetime('now') WHERE id=?",
                         (state, now + lease_seconds, row["id"]))
            campaigns = [dict(c) for c in conn.execute(
                """SELECT tc.id, tc.source_corpus_id, tc.intended_use FROM tts_campaigns tc
                   JOIN tts_campaign_members tm ON tm.campaign_id=tc.id WHERE tm.item_id=?""",
                (row["id"],))]
            return dict(row, claimed_state=state, lease_until=now + lease_seconds, campaigns=campaigns)

    def update_tts_item(self, item_id, state, *, job_id=None, error=None, attempts=0,
                        next_attempt_at=0, lease_until=None, get_failures=0):
        with self._conn() as conn:
            if job_id is not None and conn.execute(
                """SELECT 1 FROM tts_items existing JOIN tts_items target
                   ON existing.provider_url=target.provider_url
                   WHERE target.id=? AND existing.job_id=? AND existing.id!=target.id""",
                (item_id, job_id)).fetchone():
                raise ValueError("Provider job identity conflicts with another item")
            params = [state, job_id, error, attempts, get_failures, next_attempt_at, item_id]
            predicate = ""
            if lease_until is not None:
                predicate = " AND lease_until=? AND state IN ('submitting','polling')"
                params.append(lease_until)
            changed = conn.execute("""UPDATE tts_items SET state=?, job_id=COALESCE(?,job_id),
                error=?, attempts=?, get_failures=?, next_attempt_at=?, lease_until=0,
                updated_at=datetime('now') WHERE id=?""" + predicate, params).rowcount
            if changed != 1:
                raise ValueError("TTS item lease lost or unknown item")

    def import_tts_audio(self, item_id, file_path, duration_s, sample_rate, provenance, lease_until=None):
        """Register validated, durably staged audio and completion atomically."""
        if not math.isfinite(duration_s) or not 0.1 <= duration_s <= 30 or sample_rate <= 0:
            raise ValueError("Invalid validated TTS audio properties")
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            item = conn.execute("""SELECT ti.*, u.text FROM tts_items ti
                JOIN utterances u ON u.id=ti.utterance_id WHERE ti.id=?""", (item_id,)).fetchone()
            if not item:
                raise ValueError("Unknown TTS item")
            if item["audio_id"]:
                return item["audio_id"]
            if item["state"] != "polling" or (lease_until is not None and item["lease_until"] != lease_until):
                raise ValueError("TTS import requires the current accepted-job lease")
            if (provenance.get("tts_item_id") != item_id
                    or provenance.get("job_id") != item["job_id"]
                    or provenance.get("utterance_id") != item["utterance_id"]
                    or provenance.get("profile") != json.loads(item["profile_json"])):
                raise ValueError("TTS import provenance does not match the accepted item")
            corpus = self._register_corpus(conn, Corpus.new("TTSCompare", "synthesized", provider="ttscompare"))
            audio = AudioFile.new(str(file_path), corpus["name"], item["text"], duration_s, sample_rate,
                                  corpus["id"], item["utterance_id"], "synthesized", json.dumps(provenance))
            if not self._insert_audio_file(conn, audio):
                raise ValueError("TTS staging path belongs to another imported asset")
            conn.execute("""UPDATE tts_items SET audio_id=?, state='completed', error=NULL,
                lease_until=0, updated_at=datetime('now') WHERE id=?""", (audio.id, item_id))
            return audio.id

    def set_tts_shares(self, sft_percent, refill_percent):
        from utils.origin_sampling import select_origin_capped
        for value in (sft_percent, refill_percent):
            select_origin_capped([], value)
        with self._conn() as conn:
            for key, value in (("tts_sft_share_percent", sft_percent),
                               ("tts_refill_share_percent", refill_percent)):
                conn.execute("""INSERT INTO settings(key,value) VALUES (?,?)
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=datetime('now')""",
                    (key, str(value)))

    def get_sft_training_candidates(self):
        rows = self.get_current_policy_sft_corrections()
        with self._conn() as conn:
            rows.extend(dict(r) for r in conn.execute("""SELECT id AS audio_id, file_path,
                dataset, origin, utterance_id, provenance_json, reference_transcript AS correction_text
                FROM sft_audio_files WHERE origin='synthesized' ORDER BY id"""))
        return rows

    def get_sft_training_rows(self, seed=0):
        from utils.origin_sampling import select_origin_capped
        return select_origin_capped(self.get_sft_training_candidates(), self.get_setting("tts_sft_share_percent", "0"),
                                    seed=seed, one_tts_per_utterance=True)

    # -- Audio Files ------------------------------------------------------------

    def insert_audio_file(self, af: AudioFile) -> None:
        """Insert audio file record. Silently ignores duplicate file_path (INSERT OR IGNORE)."""
        with self._conn() as conn:
            self._insert_audio_file(conn, af)

    def _insert_audio_file(self, conn: sqlite3.Connection, af: AudioFile) -> int:
        if conn.execute(
            "SELECT 1 FROM audio_files WHERE file_path = ?", (af.file_path,)
        ).fetchone():
            return 0
        corpus_id, utterance_id = self._catalogue_links(conn, af)
        cursor = conn.execute(
            """INSERT INTO audio_files
               (id, file_path, dataset, reference_transcript, duration_s, sample_rate,
                corpus_id, utterance_id, origin, provenance_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (af.id, af.file_path, af.dataset, af.reference_transcript, af.duration_s,
             af.sample_rate, corpus_id, utterance_id, af.origin, af.provenance_json),
        )
        return cursor.rowcount

    def bulk_insert_audio_files(self, afs: list) -> int:
        """Insert a list of AudioFile records in a single transaction.

        Links corpus/text identities and inserts assets in one transaction.
        Returns the number of newly inserted paths.
        """
        if not afs:
            return 0
        with self._conn() as conn:
            return sum(self._insert_audio_file(conn, af) for af in afs)

    def exclude_audio_file(
        self, audio_id: str, reason: str = "noise", actor: str = "unknown"
    ) -> None:
        self.set_audio_excluded(audio_id, True, actor=actor, reason=reason)

    def set_audio_excluded(
        self, audio_id: str, excluded: bool, actor: str, reason: str
    ) -> None:
        actor = actor.strip()
        reason = reason.strip()
        if not actor:
            raise ValueError("An actor is required for audio curation")
        if not reason:
            raise ValueError("A reason is required for audio curation")
        with self._conn() as conn:
            audio = conn.execute(
                "SELECT excluded FROM audio_files WHERE id = ?", (audio_id,)
            ).fetchone()
            if audio is None:
                raise ValueError(f"Unknown audio file: {audio_id}")
            if bool(audio["excluded"]) == excluded:
                return
            pair_id = None
            if excluded:
                pair = conn.execute(
                    """SELECT id FROM hypothesis_pairs
                       WHERE audio_id = ? AND status = 'pending'
                       ORDER BY generated_at DESC, rowid DESC LIMIT 1""",
                    (audio_id,),
                ).fetchone()
                pair_id = pair["id"] if pair else None
            else:
                latest = conn.execute(
                    """SELECT pair_id FROM audio_curation_events
                       WHERE audio_id = ? AND action = 'exclude'
                       ORDER BY created_at DESC, rowid DESC LIMIT 1""",
                    (audio_id,),
                ).fetchone()
                pair_id = latest["pair_id"] if latest else None
            event = AudioCurationEvent.new(
                audio_id,
                "exclude" if excluded else "restore",
                actor,
                reason,
                pair_id=pair_id,
            )
            conn.execute(
                """UPDATE audio_files
                   SET excluded = ?, ever_excluded = 1,
                       excluded_by = COALESCE(excluded_by, ?),
                       exclusion_reason = CASE WHEN ? = 1 THEN ? ELSE exclusion_reason END
                   WHERE id = ?""",
                (int(excluded), actor, int(excluded), reason, audio_id),
            )
            if excluded:
                conn.execute(
                    """UPDATE hypothesis_pairs SET status = 'skipped', reserved_until = NULL
                       WHERE audio_id = ? AND status = 'pending'""",
                    (audio_id,),
                )
            elif pair_id:
                active_version = conn.execute(
                    "SELECT value FROM settings WHERE key = 'active_policy_model_version'"
                ).fetchone()
                if active_version:
                    conn.execute(
                        """UPDATE hypothesis_pairs SET status = 'pending'
                           WHERE id = ? AND status = 'skipped' AND model_version = ?""",
                        (pair_id, active_version["value"]),
                    )
            conn.execute(
                """INSERT INTO audio_curation_events
                   (id, audio_id, pair_id, action, actor, reason)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    event.id,
                    event.audio_id,
                    event.pair_id,
                    event.action,
                    event.actor,
                    event.reason,
                ),
            )
            if not excluded:
                conn.execute(
                    """INSERT INTO settings (key, value, updated_at)
                       VALUES ('queue_refill_requested', 'true', datetime('now'))
                       ON CONFLICT(key) DO UPDATE
                       SET value = excluded.value, updated_at = excluded.updated_at"""
                )
                conn.execute(
                    """INSERT INTO settings (key, value, updated_at)
                       VALUES ('queue_refill_status', 'queued', datetime('now'))
                       ON CONFLICT(key) DO UPDATE
                       SET value = excluded.value, updated_at = excluded.updated_at"""
                )

    def get_audio_id_for_pair(self, pair_id: str) -> Optional[str]:
        """Return the audio_id for a given hypothesis pair."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT audio_id FROM hypothesis_pairs WHERE id = ?", (pair_id,)
            ).fetchone()
        return row["audio_id"] if row else None

    def get_pair_by_id(self, pair_id: str) -> Optional[HypothesisPair]:
        """Return a HypothesisPair by its primary key, or None if not found."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM hypothesis_pairs WHERE id = ?", (pair_id,)
            ).fetchone()
        return _row_to_pair(row) if row else None

    def audio_file_exists(self, file_path: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM audio_files WHERE file_path = ?", (file_path,)
            ).fetchone()
        return row is not None

    def get_all_audio_paths(self) -> set:
        """Return the set of every registered file_path in one query.
        Use this for bulk deduplication during catalogue ingestion instead of
        calling audio_file_exists() per row."""
        with self._conn() as conn:
            rows = conn.execute("SELECT file_path FROM audio_files").fetchall()
        return {r[0] for r in rows}

    def get_audio_file(self, af_id: str) -> Optional[AudioFile]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM audio_files WHERE id = ?", (af_id,)
            ).fetchone()
        return _row_to_audio_file(row) if row else None

    def count_audio_files(self, dataset: Optional[str] = None) -> int:
        with self._conn() as conn:
            if dataset:
                return conn.execute(
                    "SELECT COUNT(*) FROM audio_files WHERE dataset = ?", (dataset,)
                ).fetchone()[0]
            return conn.execute("SELECT COUNT(*) FROM audio_files").fetchone()[0]

    # -- Hypothesis Pairs -------------------------------------------------------

    def insert_hypothesis_pair(self, pair: HypothesisPair) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO hypothesis_pairs
                   (id, audio_id, hyp_a, hyp_b, strategy_a, strategy_b,
                    model_version, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    pair.id,
                    pair.audio_id,
                    pair.hyp_a,
                    pair.hyp_b,
                    pair.strategy_a,
                    pair.strategy_b,
                    pair.model_version,
                    pair.status,
                ),
            )

    def count_pending_pairs(self) -> int:
        with self._conn() as conn:
            return conn.execute(
                """SELECT COUNT(*) FROM hypothesis_pairs
                   WHERE status = 'pending'
                     AND audio_id IN (SELECT id FROM annotation_audio_files)"""
            ).fetchone()[0]

    def retire_pending_pairs_except_model(self, model_version: str) -> int:
        with self._conn() as conn:
            cursor = conn.execute(
                """UPDATE hypothesis_pairs SET status = 'skipped'
                   WHERE status = 'pending' AND model_version != ?""",
                (model_version,),
            )
            return cursor.rowcount

    def get_pending_pairs(self, limit: int = 50) -> List[HypothesisPair]:
        """Return up to `limit` pending pairs, oldest first."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM hypothesis_pairs
                   WHERE status = 'pending'
                     AND audio_id IN (SELECT id FROM annotation_audio_files)
                         ORDER BY generated_at ASC, rowid ASC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [_row_to_pair(r) for r in rows]

    def get_and_reserve_pair(self, duration_s: int = 300) -> Optional[HypothesisPair]:
        """Atomically fetch and reserve the next available pending pair.

        Uses BEGIN IMMEDIATE so that concurrent callers cannot both receive
        the same pair.  The reservation expires after `duration_s` seconds;
        call release_expired_reservations() periodically to return timed-out
        pairs to the queue.

        Returns None when no unreserved pending pairs are available.
        """
        # Manual transaction control: we need BEGIN IMMEDIATE, which the
        # standard _conn() context manager does not issue.
        conn = sqlite3.connect(str(self.db_path), timeout=15, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.create_function(
            "normalize_utterance_text", 1, normalize_utterance_text, deterministic=True
        )
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("""SELECT * FROM hypothesis_pairs
                   WHERE status = 'pending'
                   AND audio_id IN (SELECT id FROM annotation_audio_files)
                   AND (reserved_until IS NULL
                        OR reserved_until < datetime('now'))
                   ORDER BY generated_at ASC, rowid ASC
                   LIMIT 1""").fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            pair = _row_to_pair(row)
            conn.execute(
                """UPDATE hypothesis_pairs
                   SET reserved_until = datetime('now', '+' || ? || ' seconds')
                   WHERE id = ?""",
                (duration_s, pair.id),
            )
            conn.execute("COMMIT")
            return pair
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            conn.close()

    def release_expired_reservations(self) -> int:
        """Clear reserved_until for pairs whose reservation window has passed.

        Returns the number of pairs returned to the available queue.
        Call this periodically (e.g. every 60 s) from a background task.
        """
        with self._conn() as conn:
            cursor = conn.execute("""UPDATE hypothesis_pairs
                   SET reserved_until = NULL
                   WHERE status = 'pending'
                   AND reserved_until IS NOT NULL
                   AND reserved_until < datetime('now')""")
            return cursor.rowcount

    def mark_pair_status(self, pair_id: str, status: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE hypothesis_pairs SET status = ? WHERE id = ?",
                (status, pair_id),
            )

    def audio_id_has_pending_pair(self, audio_id: str) -> bool:
        """True if there is already a pending pair for this audio file."""
        with self._conn() as conn:
            row = conn.execute(
                """SELECT 1 FROM hypothesis_pairs
                   WHERE audio_id = ? AND status = 'pending'""",
                (audio_id,),
            ).fetchone()
        return row is not None

    def get_unqueued_audio_files(
        self, limit: int, model_version: Optional[str] = None
    ) -> List[AudioFile]:
        """
        Return non-excluded audio without a pair for the requested policy.

        With no model_version, preserve the legacy one-pair-ever behavior.
        """
        from utils.origin_sampling import composition, select_origin_capped
        rows = []
        with self._conn() as conn:
            for origin in ("recorded", "synthesized"):
                rows.extend(dict(r) for r in conn.execute(
                    """SELECT af.* FROM annotation_audio_files af
                       WHERE af.origin=? AND NOT EXISTS (
                           SELECT 1 FROM hypothesis_pairs hp WHERE hp.audio_id=af.id
                           AND (? IS NULL OR hp.model_version=?))
                       ORDER BY RANDOM() LIMIT ?""", (origin, model_version, model_version, limit)))
        selected = select_origin_capped(rows, self.get_setting("tts_refill_share_percent", "0"), limit)
        self.set_setting("queue_refill_composition", json.dumps(composition(selected)))
        return [_row_to_audio_file(r) for r in selected]

    def get_eligible_audio_files_by_ids(
        self, audio_ids: List[str], model_version: str
    ) -> List[AudioFile]:
        """Return scheduled audio in caller order when eligible for this policy."""
        if not audio_ids:
            return []
        unique_ids = list(dict.fromkeys(audio_ids))
        placeholders = ",".join("?" for _ in unique_ids)
        with self._conn() as conn:
            from utils.origin_sampling import select_origin_capped
            scheduled = [dict(r) for r in conn.execute(
                f"SELECT * FROM annotation_audio_files WHERE id IN ({placeholders})", unique_ids)]
            allowed = select_origin_capped(scheduled, self.get_setting("tts_refill_share_percent", "0"),
                                           limit=len(unique_ids))
            if len(allowed) != len(scheduled):
                raise ValueError("Scheduled audio exceeds the configured TTS refill share cap")
            rows = conn.execute(
                f"""SELECT af.*
                    FROM annotation_audio_files af
                    WHERE af.id IN ({placeholders})
                      AND af.excluded = 0
                      AND NOT EXISTS (
                          SELECT 1 FROM canary_items ci WHERE ci.audio_id = af.id
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM hypothesis_pairs hp
                          WHERE hp.audio_id = af.id AND hp.model_version = ?
                      )""",
                (*unique_ids, model_version),
            ).fetchall()
        by_id = {row["id"]: _row_to_audio_file(row) for row in rows}
        return [by_id[audio_id] for audio_id in unique_ids if audio_id in by_id]

    def get_paired_audio_ids_by_ids(
        self, audio_ids: List[str], model_version: str
    ) -> set[str]:
        """Return scheduled audio IDs already paired for the requested policy."""
        if not audio_ids:
            return set()
        unique_ids = list(dict.fromkeys(audio_ids))
        placeholders = ",".join("?" for _ in unique_ids)
        with self._conn() as conn:
            rows = conn.execute(
                f"""SELECT DISTINCT audio_id
                    FROM hypothesis_pairs
                    WHERE audio_id IN ({placeholders}) AND model_version = ?""",
                (*unique_ids, model_version),
            ).fetchall()
        return {row["audio_id"] for row in rows}

    def get_random_audio_files(self, limit: int, seed: int) -> List[AudioFile]:
        """Return a deterministic random sample of non-excluded catalogue audio."""
        with self._conn() as conn:
            rows = conn.execute("""SELECT af.* FROM learning_audio_files af
                                     WHERE af.excluded = 0 AND af.origin = 'recorded'
                                         AND NOT EXISTS (
                                             SELECT 1 FROM canary_items ci WHERE ci.audio_id = af.id
                                         )""").fetchall()
        if len(rows) > limit:
            rows = random.Random(seed).sample(rows, limit)
        return [_row_to_audio_file(row) for row in rows]

    def get_ground_truth_audio_files(self) -> List[AudioFile]:
        """Return stable experiment candidates with GT and no monitor membership."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT af.*
                   FROM learning_audio_files af
                   WHERE af.excluded = 0
                     AND af.origin = 'recorded'
                     AND trim(COALESCE(af.reference_transcript, '')) != ''
                     AND NOT EXISTS (
                         SELECT 1 FROM canary_items ci WHERE ci.audio_id = af.id
                     )
                   ORDER BY af.id"""
            ).fetchall()
        return [_row_to_audio_file(row) for row in rows]

    def get_or_create_auto_session(self) -> str:
        """Return the stable session ID used for WER-based auto-labeled annotations."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM annotation_sessions WHERE id = ?",
                (_AUTO_SESSION_ID,),
            ).fetchone()
            if not row:
                conn.execute(
                    """INSERT INTO annotation_sessions (id, participant_id, machine_id, notes)
                       VALUES (?, ?, ?, ?)""",
                    (
                        _AUTO_SESSION_ID,
                        "auto-label",
                        "system",
                        "Automatically labeled by WER filter in HypothesisGenerator",
                    ),
                )
        return _AUTO_SESSION_ID

    # -- Annotation Sessions ----------------------------------------------------

    def open_session(self, session: AnnotationSession) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO annotation_sessions
                   (id, participant_id, machine_id, notes)
                   VALUES (?, ?, ?, ?)""",
                (session.id, session.participant_id, session.machine_id, session.notes),
            )

    def session_exists(self, session_id: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM annotation_sessions WHERE id = ? LIMIT 1",
                (session_id,),
            ).fetchone()
        return row is not None

    def close_session(self, session_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE annotation_sessions
                   SET ended_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                   WHERE id = ?""",
                (session_id,),
            )

    # -- Annotations ------------------------------------------------------------

    def insert_annotation(self, annotation: Annotation) -> None:
        """
        Persist an annotation and update the parent pair status atomically.
        preference='skip' -> pair status becomes 'skipped'.
        All other preferences -> pair status becomes 'annotated'.
        """
        pair_status = "skipped" if annotation.preference == "skip" else "annotated"
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO annotations
                   (id, session_id, pair_id, preference, preference_strength,
                    correction_text, notes, replay_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    annotation.id,
                    annotation.session_id,
                    annotation.pair_id,
                    annotation.preference,
                    annotation.preference_strength,
                    annotation.correction_text,
                    annotation.notes,
                    annotation.replay_count,
                ),
            )
            conn.execute(
                "UPDATE hypothesis_pairs SET status = ? WHERE id = ?",
                (pair_status, annotation.pair_id),
            )

    def count_human_annotations(self) -> int:
        """Count non-skip human annotations (excludes auto-labeled system session)."""
        with self._conn() as conn:
            return conn.execute(
                """SELECT COUNT(*) FROM annotations a
                                     JOIN annotation_sessions s ON s.id = a.session_id
                                     LEFT JOIN annotator_moderation am
                                         ON am.participant_id = s.participant_id
                                     WHERE a.preference != 'skip' AND a.session_id != ?
                                         AND a.is_muted = 0 AND COALESCE(am.is_muted, 0) = 0""",
                (_AUTO_SESSION_ID,),
            ).fetchone()[0]

    def count_actionable_annotations(self) -> int:
        """Count clear-preference (a/b only) human annotations -- the usable subset for RM training."""
        with self._conn() as conn:
            return conn.execute(
                """SELECT COUNT(*) FROM annotations a
                                     JOIN annotation_sessions s ON s.id = a.session_id
                                     LEFT JOIN annotator_moderation am
                                         ON am.participant_id = s.participant_id
                                     WHERE a.preference IN ('a', 'b') AND a.session_id != ?
                                         AND a.is_muted = 0 AND COALESCE(am.is_muted, 0) = 0""",
                (_AUTO_SESSION_ID,),
            ).fetchone()[0]

    def annotation_summary(self) -> dict:
        """Return a breakdown dict: total / actionable (a+b) / equal / skip."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT preference, COUNT(*) as n
                                     FROM annotations a
                                     JOIN annotation_sessions s ON s.id = a.session_id
                                     LEFT JOIN annotator_moderation am
                                         ON am.participant_id = s.participant_id
                                     WHERE a.session_id != ? AND a.is_muted = 0
                                         AND COALESCE(am.is_muted, 0) = 0
                   GROUP BY preference""",
                (_AUTO_SESSION_ID,),
            ).fetchall()
        counts = {r["preference"]: r["n"] for r in rows}
        actionable = counts.get("a", 0) + counts.get("b", 0)
        return {
            "total": sum(counts.values()),
            "actionable": actionable,
            "equal": counts.get("equal", 0),
            "skip": counts.get("skip", 0),
        }

    def get_preference_dataset(self, model_version: Optional[str] = None) -> List[Dict]:
        """
        Join annotations + hypothesis_pairs + audio_files for reward model training.
        Returns only clear-preference rows (preference IN ('a', 'b')) ordered oldest first.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT
                       a.id                AS annotation_id,
                       a.preference,
                       a.preference_strength,
                       a.correction_text,
                       hp.id               AS pair_id,
                       hp.audio_id,
                       hp.hyp_a,
                       hp.hyp_b,
                       hp.strategy_a,
                       hp.strategy_b,
                       hp.model_version,
                       af.file_path,
                       af.dataset,
                       af.reference_transcript,
                       af.origin, af.utterance_id, af.provenance_json
                   FROM annotations a
                                     JOIN annotation_sessions s ON s.id = a.session_id
                                     LEFT JOIN annotator_moderation am
                                         ON am.participant_id = s.participant_id
                   JOIN hypothesis_pairs hp ON hp.id = a.pair_id
                   JOIN annotation_audio_files af ON af.id = hp.audio_id
                   WHERE a.preference IN ('a', 'b')
                                         AND a.session_id != ?
                                         AND a.is_muted = 0
                                         AND COALESCE(am.is_muted, 0) = 0
                                         AND (? IS NULL OR hp.model_version = ?)
                                         AND NOT EXISTS (
                                             SELECT 1 FROM canary_items ci WHERE ci.audio_id = af.id
                                         )
                   ORDER BY a.annotated_at ASC""",
                (_AUTO_SESSION_ID, model_version, model_version),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_sft_corrections(self, model_version: Optional[str] = None) -> List[Dict]:
        """
        Return all human-provided correction_text entries joined with their audio path.
        Used by run_sft.py for supervised fine-tuning on gold-standard transcripts.
        Excludes auto-labeled annotations (system-auto-label session).
        """
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT
                       af.id AS audio_id, af.dataset, af.origin, af.utterance_id, af.provenance_json,
                       af.file_path,
                       a.correction_text,
                       a.annotated_at
                   FROM annotations a
                                     JOIN annotation_sessions s ON s.id = a.session_id
                                     LEFT JOIN annotator_moderation am
                                         ON am.participant_id = s.participant_id
                   JOIN hypothesis_pairs hp ON hp.id = a.pair_id
                   JOIN sft_audio_files af ON af.id = hp.audio_id
                   WHERE a.correction_text IS NOT NULL
                     AND trim(a.correction_text) != ''
                     AND a.session_id != ?
                     AND a.is_muted = 0
                     AND COALESCE(am.is_muted, 0) = 0
                     AND (? IS NULL OR hp.model_version = ?)
                     AND af.origin = 'recorded'
                     AND NOT EXISTS (
                       SELECT 1 FROM canary_items ci WHERE ci.audio_id = af.id
                     )
                   ORDER BY a.annotated_at ASC""",
                (_AUTO_SESSION_ID, model_version, model_version),
            ).fetchall()
        return [dict(r) for r in rows]

    def set_annotation_muted(
        self,
        annotation_id: str,
        muted: bool,
        actor: str = "admin",
        reason: Optional[str] = None,
    ) -> None:
        event = AnnotationModerationEvent.new(
            "annotation", annotation_id, "mute" if muted else "unmute", actor, reason
        )
        with self._conn() as conn:
            annotation = conn.execute(
                "SELECT is_muted FROM annotations WHERE id = ?", (annotation_id,)
            ).fetchone()
            if annotation is None:
                raise ValueError(f"Unknown annotation: {annotation_id}")
            if bool(annotation["is_muted"]) == muted:
                return
            conn.execute(
                "UPDATE annotations SET is_muted = ? WHERE id = ?",
                (int(muted), annotation_id),
            )
            self._insert_annotation_moderation_event(conn, event)

    def set_annotator_muted(
        self,
        participant_id: str,
        muted: bool,
        actor: str = "admin",
        reason: Optional[str] = None,
    ) -> None:
        participant_id = participant_id.strip()
        if not participant_id or participant_id == "auto-label":
            raise ValueError("A human annotator ID is required")
        event = AnnotationModerationEvent.new(
            "annotator", participant_id, "mute" if muted else "unmute", actor, reason
        )
        with self._conn() as conn:
            exists = conn.execute(
                "SELECT 1 FROM annotation_sessions WHERE participant_id = ? LIMIT 1",
                (participant_id,),
            ).fetchone()
            if exists is None:
                raise ValueError(f"Unknown annotator: {participant_id}")
            current = conn.execute(
                "SELECT is_muted FROM annotator_moderation WHERE participant_id = ?",
                (participant_id,),
            ).fetchone()
            if (bool(current["is_muted"]) if current else False) == muted:
                return
            conn.execute(
                """INSERT INTO annotator_moderation
                       (id, participant_id, is_muted, updated_by)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(participant_id) DO UPDATE
                   SET is_muted = excluded.is_muted,
                       updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'),
                       updated_by = excluded.updated_by""",
                (event.id, participant_id, int(muted), actor),
            )
            self._insert_annotation_moderation_event(conn, event)

    @staticmethod
    def _insert_annotation_moderation_event(
        conn: sqlite3.Connection, event: AnnotationModerationEvent
    ) -> None:
        conn.execute(
            """INSERT INTO annotation_moderation_events
               (id, target_type, target_id, action, actor, reason)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                event.id,
                event.target_type,
                event.target_id,
                event.action,
                event.actor,
                event.reason,
            ),
        )

    def get_annotation_audit(
        self,
        limit: int = 100,
        offset: int = 0,
        participant_id: Optional[str] = None,
        state: str = "all",
    ) -> List[Dict]:
        """Return annotation history with effective mute and latest stored rewards."""
        if state not in {"all", "active", "muted"}:
            raise ValueError(f"Unknown annotation audit state: {state}")
        conditions = ["a.session_id != ?"]
        params: list[object] = [_AUTO_SESSION_ID]
        if participant_id:
            conditions.append("s.participant_id = ?")
            params.append(participant_id)
        effective_muted = "(a.is_muted = 1 OR COALESCE(am.is_muted, 0) = 1)"
        if state == "active":
            conditions.append(f"NOT {effective_muted}")
        elif state == "muted":
            conditions.append(effective_muted)
        params.extend((max(1, min(limit, 500)), max(0, offset)))
        with self._conn() as conn:
            rows = conn.execute(
                f"""SELECT a.id AS annotation_id, a.preference,
                           a.preference_strength, a.correction_text, a.notes,
                           a.replay_count, a.annotated_at, a.is_muted,
                           s.participant_id, s.machine_id, hp.id AS pair_id,
                           hp.hyp_a, hp.hyp_b, hp.strategy_a, hp.strategy_b,
                           hp.model_version, hp.generated_at, af.id AS audio_id,
                           af.file_path, af.dataset, rs.reward_a, rs.reward_b,
                           rs.source AS reward_source, rs.gemma_version,
                           rs.scored_at, COALESCE(am.is_muted, 0) AS annotator_muted,
                           CASE WHEN {effective_muted} THEN 1 ELSE 0 END
                             AS effective_muted
                    FROM annotations a
                    JOIN annotation_sessions s ON s.id = a.session_id
                    JOIN hypothesis_pairs hp ON hp.id = a.pair_id
                    JOIN audio_files af ON af.id = hp.audio_id
                    LEFT JOIN annotator_moderation am
                      ON am.participant_id = s.participant_id
                    LEFT JOIN reward_scores rs ON rs.id = (
                        SELECT latest.id FROM reward_scores latest
                        WHERE latest.pair_id = hp.id
                        ORDER BY latest.scored_at DESC, latest.rowid DESC LIMIT 1
                    )
                    WHERE {" AND ".join(conditions)}
                    ORDER BY a.annotated_at DESC, a.rowid DESC
                    LIMIT ? OFFSET ?""",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def get_excluded_audio_audit(
        self,
        limit: int = 100,
        offset: int = 0,
        dataset: Optional[str] = None,
        query: Optional[str] = None,
        state: str = "all",
    ) -> List[Dict]:
        if state not in {"all", "excluded", "restored"}:
            raise ValueError(f"Unknown audio curation state: {state}")
        conditions = ["af.ever_excluded = 1"]
        params: list[object] = []
        if state == "excluded":
            conditions.append("af.excluded = 1")
        elif state == "restored":
            conditions.append("af.excluded = 0")
        if dataset:
            conditions.append("af.dataset = ?")
            params.append(dataset)
        if query:
            pattern = f"%{query}%"
            conditions.append(
                "(af.id LIKE ? OR af.file_path LIKE ? "
                "OR COALESCE(af.exclusion_reason, '') LIKE ? "
                "OR COALESCE(af.reference_transcript, '') LIKE ?)"
            )
            params.extend((pattern, pattern, pattern, pattern))
        params.extend((max(1, min(limit, 500)), max(0, offset)))
        with self._conn() as conn:
            rows = conn.execute(
                f"""SELECT af.id AS audio_id, af.file_path, af.dataset,
                           af.reference_transcript, af.duration_s, af.sample_rate,
                           af.created_at, af.excluded, af.excluded_by,
                           af.exclusion_reason,
                           latest.action AS latest_action,
                           latest.actor AS latest_actor,
                           latest.reason AS latest_reason,
                           latest.created_at AS latest_action_at,
                           (SELECT COUNT(*) FROM hypothesis_pairs hp
                            WHERE hp.audio_id = af.id) AS pair_count,
                           (SELECT COUNT(*) FROM annotations a
                            JOIN hypothesis_pairs hp ON hp.id = a.pair_id
                            WHERE hp.audio_id = af.id) AS annotation_count
                    FROM audio_files af
                    LEFT JOIN audio_curation_events latest ON latest.id = (
                        SELECT ace.id FROM audio_curation_events ace
                        WHERE ace.audio_id = af.id
                        ORDER BY ace.created_at DESC, ace.rowid DESC LIMIT 1
                    )
                    WHERE {" AND ".join(conditions)}
                    ORDER BY af.dataset, af.file_path, af.id
                    LIMIT ? OFFSET ?""",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def get_excluded_audio_datasets(self) -> List[str]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT DISTINCT dataset FROM audio_files
                   WHERE ever_excluded = 1 ORDER BY dataset"""
            ).fetchall()
        return [row["dataset"] for row in rows]

    def is_audio_ever_excluded(self, audio_id: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT ever_excluded FROM audio_files WHERE id = ?", (audio_id,)
            ).fetchone()
        return bool(row and row["ever_excluded"])

    def get_audio_curation_events(self, audio_id: str) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM audio_curation_events WHERE audio_id = ?
                   ORDER BY created_at DESC, rowid DESC""",
                (audio_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_annotator_audit(self) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT s.participant_id,
                          COUNT(DISTINCT s.id) AS session_count,
                          COUNT(a.id) AS annotation_count,
                          SUM(CASE WHEN a.preference IN ('a', 'b') THEN 1 ELSE 0 END)
                            AS actionable_count,
                          MIN(s.started_at) AS first_seen,
                          MAX(COALESCE(a.annotated_at, s.started_at)) AS last_seen,
                          COALESCE(am.is_muted, 0) AS is_muted,
                          am.updated_at AS moderation_updated_at
                   FROM annotation_sessions s
                   LEFT JOIN annotations a ON a.session_id = s.id
                   LEFT JOIN annotator_moderation am
                     ON am.participant_id = s.participant_id
                   WHERE s.id != ?
                   GROUP BY s.participant_id
                   ORDER BY last_seen DESC, s.participant_id""",
                (_AUTO_SESSION_ID,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_annotation_moderation_events(self, limit: int = 100) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM annotation_moderation_events
                   ORDER BY created_at DESC, rowid DESC LIMIT ?""",
                (max(1, min(limit, 500)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_active_policy_model_version(self) -> Optional[str]:
        configured = self.get_setting("active_policy_model_version", "").strip()
        if configured:
            return configured
        checkpoint = self.get_setting("active_policy_checkpoint", "").strip()
        if not checkpoint:
            return None
        checkpoint_name = Path(checkpoint).name
        with self._conn() as conn:
            row = conn.execute(
                """SELECT model_version FROM hypothesis_pairs
                   WHERE model_version LIKE ?
                         ORDER BY generated_at DESC, rowid DESC LIMIT 1""",
                (f"%+lora:{checkpoint_name}",),
            ).fetchone()
        return row["model_version"] if row else None

    def get_current_policy_preference_dataset(self) -> List[Dict]:
        model_version = self.get_active_policy_model_version()
        return self.get_preference_dataset(model_version) if model_version else []

    def get_current_policy_sft_corrections(self) -> List[Dict]:
        model_version = self.get_active_policy_model_version()
        return self.get_sft_corrections(model_version) if model_version else []

    # -- Canary benchmark ------------------------------------------------------

    def create_canary_set(
        self,
        canary_set: CanarySet,
        audio_ids: Optional[List[str]] = None,
    ) -> List[CanaryItem]:
        """Select and reserve a deterministic, previously unannotated audio holdout."""
        if audio_ids is not None:
            if len(audio_ids) != canary_set.target_size:
                raise ValueError("Explicit canary IDs must match the target size")
            if len(set(audio_ids)) != len(audio_ids):
                raise ValueError("Explicit canary IDs must be unique")
        with self._conn() as conn:
            candidates = conn.execute("""SELECT af.*
                   FROM learning_audio_files af
                   WHERE af.excluded = 0
                     AND af.origin = 'recorded'
                     AND NOT EXISTS (
                       SELECT 1 FROM canary_items ci WHERE ci.audio_id = af.id
                     )
                     AND NOT EXISTS (
                       SELECT 1 FROM hypothesis_pairs hp
                       JOIN annotations a ON a.pair_id = hp.id
                       WHERE hp.audio_id IN (
                           SELECT related.id FROM audio_files related
                           WHERE related.id = af.id OR related.utterance_id = af.utterance_id
                       )
                     )
                   ORDER BY af.id""").fetchall()
            if len(candidates) < canary_set.target_size:
                raise ValueError(
                    f"Only {len(candidates)} eligible canary clips available "
                    f"(need {canary_set.target_size})"
                )
            if audio_ids is None:
                selected = random.Random(canary_set.selection_seed).sample(
                    candidates, canary_set.target_size
                )
            else:
                candidates_by_id = {row["id"]: row for row in candidates}
                unavailable = [
                    audio_id
                    for audio_id in audio_ids
                    if audio_id not in candidates_by_id
                ]
                if unavailable:
                    raise ValueError(
                        "Explicit canary audio is missing or ineligible: "
                        + ", ".join(unavailable)
                    )
                selected = [candidates_by_id[audio_id] for audio_id in audio_ids]
            conn.execute(
                """INSERT INTO canary_sets
                   (id, name, selection_seed, selection_method, target_size, status)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    canary_set.id,
                    canary_set.name,
                    canary_set.selection_seed,
                    canary_set.selection_method,
                    canary_set.target_size,
                    canary_set.status,
                ),
            )
            items = [
                CanaryItem.new(
                    canary_set.id,
                    row["id"],
                    gt_available=bool(row["reference_transcript"]),
                )
                for row in selected
            ]
            conn.executemany(
                """INSERT INTO canary_items
                   (id, canary_set_id, audio_id, gt_available, gt_revealed)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (
                        item.id,
                        item.canary_set_id,
                        item.audio_id,
                        int(item.gt_available),
                        int(item.gt_revealed),
                    )
                    for item in items
                ],
            )
            conn.executemany(
                """UPDATE hypothesis_pairs SET status = 'skipped', reserved_until = NULL
                   WHERE audio_id = ? AND status = 'pending'""",
                [(item.audio_id,) for item in items],
            )
        return items

    def get_canary_sets(self) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute("""SELECT cs.*,
                          COUNT(ci.id) AS item_count,
                          SUM(CASE WHEN ci.verified_at IS NOT NULL THEN 1 ELSE 0 END) AS verified_count
                   FROM canary_sets cs
                   LEFT JOIN canary_items ci ON ci.canary_set_id = cs.id
                   GROUP BY cs.id
                     ORDER BY cs.created_at DESC, cs.rowid DESC""").fetchall()
        return [dict(row) for row in rows]

    def get_canary_set(self, canary_set_id: str) -> Optional[Dict]:
        return next(
            (row for row in self.get_canary_sets() if row["id"] == canary_set_id),
            None,
        )

    def get_canary_items(self, canary_set_id: str) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT ci.*, af.file_path, af.dataset
                   FROM canary_items ci
                   JOIN audio_files af ON af.id = ci.audio_id
                   WHERE ci.canary_set_id = ?
                   ORDER BY ci.created_at, ci.id""",
                (canary_set_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_canary_item(self, item_id: str) -> Optional[Dict]:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT ci.*, af.file_path, af.dataset
                   FROM canary_items ci
                   JOIN audio_files af ON af.id = ci.audio_id
                   WHERE ci.id = ?""",
                (item_id,),
            ).fetchone()
        return dict(row) if row else None

    def remove_canary_set(self, canary_set_id: str) -> str:
        """Delete an editable draft or archive a locked monitor."""
        with self._conn() as conn:
            canary_set = conn.execute(
                "SELECT status, is_promotion_gate FROM canary_sets WHERE id = ?",
                (canary_set_id,),
            ).fetchone()
            if canary_set is None:
                raise ValueError(f"Unknown WER Monitor: {canary_set_id}")
            if canary_set["status"] == "retired":
                raise ValueError("WER Monitor is already archived")
            if canary_set["is_promotion_gate"]:
                raise ValueError(
                    "Choose another promotion gate before archiving this evaluation set"
                )
            active_evaluation = conn.execute(
                """SELECT 1 FROM canary_evaluations
                   WHERE canary_set_id = ? AND status IN ('queued', 'running')
                   LIMIT 1""",
                (canary_set_id,),
            ).fetchone()
            requested_set = conn.execute("""SELECT value FROM settings
                   WHERE key = 'wer_monitor_evaluation_canary_set_id'""").fetchone()
            requested_status = conn.execute("""SELECT value FROM settings
                   WHERE key = 'wer_monitor_evaluation_status'""").fetchone()
            request_active = (
                requested_set
                and requested_set["value"] == canary_set_id
                and requested_status
                and requested_status["value"] in {"queued", "running"}
            )
            if canary_set["status"] == "ready" and (
                active_evaluation or request_active
            ):
                raise ValueError(
                    "Wait for the WER Monitor evaluation to finish before archiving"
                )
            if canary_set["status"] == "collecting":
                evaluation_count = conn.execute(
                    """SELECT COUNT(*) AS count FROM canary_evaluations
                       WHERE canary_set_id = ?""",
                    (canary_set_id,),
                ).fetchone()["count"]
                if evaluation_count:
                    raise ValueError(
                        "A monitor with evaluation history cannot be deleted"
                    )
                conn.execute("DELETE FROM canary_sets WHERE id = ?", (canary_set_id,))
                return "deleted"
            conn.execute(
                "UPDATE canary_sets SET status = 'retired' WHERE id = ?",
                (canary_set_id,),
            )
        return "archived"

    def reveal_canary_ground_truth(self, item_id: str) -> str:
        """Record an admin GT reveal and return the catalogue reference."""
        with self._conn() as conn:
            row = conn.execute(
                """SELECT af.reference_transcript
                   FROM canary_items ci
                   JOIN canary_sets cs ON cs.id = ci.canary_set_id
                   JOIN audio_files af ON af.id = ci.audio_id
                   WHERE ci.id = ? AND cs.status = 'collecting'""",
                (item_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Unknown or frozen canary item: {item_id}")
            ground_truth = (row["reference_transcript"] or "").strip()
            if not ground_truth:
                raise ValueError("Ground-truth transcript is unavailable")
            conn.execute(
                "UPDATE canary_items SET gt_revealed = 1 WHERE id = ?",
                (item_id,),
            )
        return ground_truth

    def save_canary_transcript(
        self,
        item_id: str,
        transcript: str,
        copied_gt: bool = False,
    ) -> None:
        transcript = transcript.strip()
        if not transcript:
            raise ValueError("Canary transcript cannot be empty")
        with self._conn() as conn:
            item = conn.execute(
                """SELECT ci.id, ci.gt_revealed, af.reference_transcript, cs.status
                   FROM canary_items ci
                   JOIN canary_sets cs ON cs.id = ci.canary_set_id
                   JOIN audio_files af ON af.id = ci.audio_id
                   WHERE ci.id = ?""",
                (item_id,),
            ).fetchone()
            if item is None:
                raise ValueError(f"Unknown canary item: {item_id}")
            if item["status"] != "collecting":
                raise ValueError("Frozen canary transcripts are immutable")
            ground_truth = (item["reference_transcript"] or "").strip()
            if copied_gt:
                if not item["gt_revealed"] or not ground_truth:
                    raise ValueError("Ground truth must be revealed before copying")
                if transcript != ground_truth:
                    raise ValueError(
                        "Copied ground truth must match the stored reference"
                    )
                provenance = "copied_gt"
            elif item["gt_revealed"]:
                provenance = "manual_after_reveal"
            else:
                provenance = "manual"
            conn.execute(
                """UPDATE canary_items
                   SET transcript = ?, transcript_provenance = ?,
                       verified_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                   WHERE id = ?""",
                (transcript, provenance, item_id),
            )

    def bulk_copy_canary_ground_truth(self, canary_set_id: str) -> Dict[str, int]:
        """Copy available GT into unfinished monitor items without overwriting work."""
        with self._conn() as conn:
            canary_set = conn.execute(
                "SELECT status FROM canary_sets WHERE id = ?",
                (canary_set_id,),
            ).fetchone()
            if canary_set is None:
                raise ValueError(f"Unknown WER Monitor: {canary_set_id}")
            if canary_set["status"] != "collecting":
                raise ValueError("Frozen WER Monitor transcripts are immutable")

            counts = conn.execute(
                """SELECT
                     SUM(CASE WHEN ci.verified_at IS NOT NULL THEN 1 ELSE 0 END)
                       AS already_completed,
                     SUM(CASE WHEN ci.verified_at IS NULL
                               AND trim(COALESCE(af.reference_transcript, '')) = ''
                              THEN 1 ELSE 0 END) AS missing_gt
                   FROM canary_items ci
                   JOIN audio_files af ON af.id = ci.audio_id
                   WHERE ci.canary_set_id = ?""",
                (canary_set_id,),
            ).fetchone()
            cursor = conn.execute(
                """UPDATE canary_items
                   SET transcript = (
                         SELECT trim(af.reference_transcript)
                         FROM audio_files af WHERE af.id = canary_items.audio_id
                       ),
                       transcript_provenance = 'copied_gt',
                       gt_revealed = 1,
                       verified_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                   WHERE canary_set_id = ?
                     AND verified_at IS NULL
                     AND EXISTS (
                       SELECT 1 FROM audio_files af
                       WHERE af.id = canary_items.audio_id
                         AND trim(COALESCE(af.reference_transcript, '')) != ''
                     )""",
                (canary_set_id,),
            )
        return {
            "copied": cursor.rowcount,
            "already_completed": int(counts["already_completed"] or 0),
            "missing_gt": int(counts["missing_gt"] or 0),
        }

    def freeze_canary_set(self, canary_set_id: str) -> str:
        """Freeze a fully transcribed canary set and return its stable fingerprint."""
        with self._conn() as conn:
            canary_set = conn.execute(
                "SELECT * FROM canary_sets WHERE id = ?",
                (canary_set_id,),
            ).fetchone()
            if canary_set is None:
                raise ValueError(f"Unknown canary set: {canary_set_id}")
            if canary_set["status"] != "collecting":
                raise ValueError("Canary set is already frozen or retired")
            items = conn.execute(
                """SELECT audio_id, transcript, transcript_provenance
                   FROM canary_items
                   WHERE canary_set_id = ?
                   ORDER BY audio_id""",
                (canary_set_id,),
            ).fetchall()
            if len(items) != canary_set["target_size"] or any(
                not row["transcript"] for row in items
            ):
                raise ValueError(
                    "Canary set can be frozen only after every transcript is verified"
                )
            payload = [
                (row["audio_id"], row["transcript"], row["transcript_provenance"])
                for row in items
            ]
            fingerprint = hashlib.sha256(
                json.dumps(payload, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            conn.execute(
                """UPDATE canary_sets
                   SET status = 'ready', fingerprint = ?,
                       frozen_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                   WHERE id = ?""",
                (fingerprint, canary_set_id),
            )
            promotion_gate = conn.execute("""SELECT id FROM canary_sets
                   WHERE status = 'ready' AND is_promotion_gate = 1""").fetchone()
            if promotion_gate is None:
                conn.execute(
                    "UPDATE canary_sets SET is_promotion_gate = 1 WHERE id = ?",
                    (canary_set_id,),
                )
            conn.execute(
                """INSERT INTO settings (key, value, updated_at)
                   VALUES ('wer_monitor_evaluation_canary_set_id', ?, datetime('now'))
                   ON CONFLICT(key) DO UPDATE
                   SET value = excluded.value, updated_at = excluded.updated_at""",
                (canary_set_id,),
            )
            conn.execute("""INSERT INTO settings (key, value, updated_at)
                   VALUES ('wer_monitor_evaluation_requested', 'true', datetime('now'))
                   ON CONFLICT(key) DO UPDATE
                   SET value = excluded.value, updated_at = excluded.updated_at""")
            conn.execute("""INSERT INTO settings (key, value, updated_at)
                   VALUES ('wer_monitor_evaluation_status', 'queued', datetime('now'))
                   ON CONFLICT(key) DO UPDATE
                   SET value = excluded.value, updated_at = excluded.updated_at""")
            conn.execute("""INSERT INTO settings (key, value, updated_at)
                   VALUES ('wer_monitor_evaluation_log', 'Waiting for coordinator', datetime('now'))
                   ON CONFLICT(key) DO UPDATE
                   SET value = excluded.value, updated_at = excluded.updated_at""")
        return fingerprint

    def unfreeze_canary_set(self, canary_set_id: str) -> str:
        """Create an editable revision while preserving the evaluated set."""
        with self._conn() as conn:
            canary_set = conn.execute(
                "SELECT * FROM canary_sets WHERE id = ?",
                (canary_set_id,),
            ).fetchone()
            if canary_set is None:
                raise ValueError(f"Unknown WER Monitor: {canary_set_id}")
            if canary_set["status"] != "ready":
                raise ValueError("Only a frozen WER Monitor can be unfrozen")
            active_evaluation = conn.execute(
                """SELECT 1 FROM canary_evaluations
                   WHERE canary_set_id = ? AND status = 'running' LIMIT 1""",
                (canary_set_id,),
            ).fetchone()
            requested_set = conn.execute("""SELECT value FROM settings
                   WHERE key = 'wer_monitor_evaluation_canary_set_id'""").fetchone()
            evaluation_status = conn.execute("""SELECT value FROM settings
                   WHERE key = 'wer_monitor_evaluation_status'""").fetchone()
            request_running = (
                requested_set
                and requested_set["value"] == canary_set_id
                and evaluation_status
                and evaluation_status["value"] == "running"
            )
            if active_evaluation or request_running:
                raise ValueError(
                    "Wait for the running WER Monitor evaluation to finish"
                )
            base_name = re.sub(r" \(revision \d+\)$", "", canary_set["name"])
            revision_count = conn.execute(
                """SELECT COUNT(*) AS count FROM canary_sets
                   WHERE name = ? OR name LIKE ?""",
                (base_name, f"{base_name} (revision %)"),
            ).fetchone()["count"]
            revision = CanarySet.new(
                name=f"{base_name} (revision {revision_count + 1})",
                selection_seed=canary_set["selection_seed"],
                selection_method=f'revision_of:{canary_set["id"]}',
                target_size=canary_set["target_size"],
            )
            conn.execute(
                """INSERT INTO canary_sets
                   (id, name, selection_seed, selection_method, target_size, status)
                   VALUES (?, ?, ?, ?, ?, 'collecting')""",
                (
                    revision.id,
                    revision.name,
                    revision.selection_seed,
                    revision.selection_method,
                    revision.target_size,
                ),
            )
            source_items = conn.execute(
                "SELECT * FROM canary_items WHERE canary_set_id = ?",
                (canary_set_id,),
            ).fetchall()
            revised_items = [
                CanaryItem.new(
                    revision.id,
                    row["audio_id"],
                    gt_available=bool(row["gt_available"]),
                )
                for row in source_items
            ]
            conn.executemany(
                """INSERT INTO canary_items
                   (id, canary_set_id, audio_id, transcript,
                    transcript_provenance, gt_available, gt_revealed, verified_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        item.id,
                        revision.id,
                        source["audio_id"],
                        source["transcript"],
                        source["transcript_provenance"],
                        int(source["gt_available"]),
                        int(source["gt_revealed"]),
                        source["verified_at"],
                    )
                    for item, source in zip(revised_items, source_items)
                ],
            )
        return revision.id

    def reroll_canary_item(self, item_id: str) -> str:
        """Replace a faulty clip in an editable monitor and exclude the old audio."""
        with self._conn() as conn:
            item = conn.execute(
                """SELECT ci.*, cs.status, cs.selection_seed
                   FROM canary_items ci
                   JOIN canary_sets cs ON cs.id = ci.canary_set_id
                   WHERE ci.id = ?""",
                (item_id,),
            ).fetchone()
            if item is None:
                raise ValueError(f"Unknown WER Monitor item: {item_id}")
            if item["status"] != "collecting":
                raise ValueError("Unlock the WER Monitor before replacing audio")
            candidates = conn.execute("""SELECT af.* FROM learning_audio_files af
                   WHERE af.excluded = 0
                     AND af.origin = 'recorded'
                     AND NOT EXISTS (
                       SELECT 1 FROM canary_items ci WHERE ci.audio_id = af.id
                     )
                     AND NOT EXISTS (
                       SELECT 1 FROM hypothesis_pairs hp
                       JOIN annotations a ON a.pair_id = hp.id
                       WHERE hp.audio_id IN (
                           SELECT related.id FROM audio_files related
                           WHERE related.id = af.id OR related.utterance_id = af.utterance_id
                       )
                     )
                   ORDER BY af.id""").fetchall()
            if not candidates:
                raise ValueError("No eligible replacement clips are available")
            replacement = random.Random(
                f'{item["canary_set_id"]}:{item_id}:{item["selection_seed"]}'
            ).choice(candidates)
            new_item = CanaryItem.new(
                item["canary_set_id"],
                replacement["id"],
                gt_available=bool(replacement["reference_transcript"]),
            )
            conn.execute(
                """UPDATE audio_files SET excluded = 1,
                   exclusion_reason = 'faulty_wer_monitor_audio' WHERE id = ?""",
                (item["audio_id"],),
            )
            conn.execute("DELETE FROM canary_items WHERE id = ?", (item_id,))
            conn.execute(
                """INSERT INTO canary_items
                   (id, canary_set_id, audio_id, gt_available, gt_revealed)
                   VALUES (?, ?, ?, ?, 0)""",
                (
                    new_item.id,
                    new_item.canary_set_id,
                    new_item.audio_id,
                    int(new_item.gt_available),
                ),
            )
            conn.execute(
                """UPDATE hypothesis_pairs SET status = 'skipped', reserved_until = NULL
                   WHERE audio_id = ? AND status = 'pending'""",
                (new_item.audio_id,),
            )
        return new_item.id

    def request_wer_monitor_evaluation(self, canary_set_id: str) -> None:
        with self._conn() as conn:
            canary_set = conn.execute(
                "SELECT status FROM canary_sets WHERE id = ?",
                (canary_set_id,),
            ).fetchone()
            if canary_set is None or canary_set["status"] != "ready":
                raise ValueError("Freeze the WER Monitor before evaluation")
            for key, value in (
                ("wer_monitor_evaluation_canary_set_id", canary_set_id),
                ("wer_monitor_evaluation_requested", "true"),
                ("wer_monitor_evaluation_status", "queued"),
                ("wer_monitor_evaluation_log", "Waiting for coordinator"),
            ):
                conn.execute(
                    """INSERT INTO settings (key, value, updated_at)
                       VALUES (?, ?, datetime('now'))
                       ON CONFLICT(key) DO UPDATE
                       SET value = excluded.value, updated_at = excluded.updated_at""",
                    (key, value),
                )

    # -- Model checkpoint registry and health evaluations ---------------------

    def register_model_checkpoint(self, checkpoint: ModelCheckpoint) -> Dict:
        """Register an immutable model identity, returning the canonical row."""
        with self._conn() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO model_checkpoints
                   (id, model_key, checkpoint_path, stage, parent_checkpoint_id,
                    round_id, provenance, health_status, available, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    checkpoint.id,
                    checkpoint.model_key,
                    checkpoint.checkpoint_path,
                    checkpoint.stage,
                    checkpoint.parent_checkpoint_id,
                    checkpoint.round_id,
                    checkpoint.provenance,
                    checkpoint.health_status,
                    int(checkpoint.available),
                    checkpoint.metadata_json,
                ),
            )
            row = conn.execute(
                "SELECT * FROM model_checkpoints WHERE model_key = ?",
                (checkpoint.model_key,),
            ).fetchone()
        return dict(row)

    def get_model_checkpoints(self) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute("""SELECT mc.*, parent.model_key AS parent_model_key
                   FROM model_checkpoints mc
                   LEFT JOIN model_checkpoints parent ON parent.id = mc.parent_checkpoint_id
                   ORDER BY mc.created_at, mc.model_key""").fetchall()
        return [dict(row) for row in rows]

    def get_model_checkpoint(self, checkpoint_id: str) -> Optional[Dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM model_checkpoints WHERE id = ?", (checkpoint_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_model_checkpoint_by_path(self, checkpoint_path: str) -> Optional[Dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM model_checkpoints WHERE checkpoint_path = ?",
                (checkpoint_path,),
            ).fetchone()
        return dict(row) if row else None

    def activate_model_checkpoint(
        self,
        activation: PolicyActivation,
        setting_value: str,
        model_version: Optional[str] = None,
        queue_log: Optional[str] = None,
    ) -> int:
        """Atomically activate a policy and optionally roll over its queue."""
        retired = 0
        with self._conn() as conn:
            target = conn.execute(
                "SELECT available FROM model_checkpoints WHERE id = ?",
                (activation.new_checkpoint_id,),
            ).fetchone()
            if target is None or not target["available"]:
                raise ValueError("Checkpoint is not available for activation")
            conn.execute(
                """INSERT INTO policy_activations
                   (id, previous_checkpoint_id, new_checkpoint_id, reason, actor,
                    warning_present)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    activation.id,
                    activation.previous_checkpoint_id,
                    activation.new_checkpoint_id,
                    activation.reason,
                    activation.actor,
                    int(activation.warning_present),
                ),
            )
            conn.execute(
                """INSERT INTO settings (key, value, updated_at)
                   VALUES ('active_policy_checkpoint', ?, datetime('now'))
                   ON CONFLICT(key) DO UPDATE
                   SET value = excluded.value, updated_at = excluded.updated_at""",
                (setting_value,),
            )
            if model_version is not None:
                conn.execute(
                    """INSERT INTO settings (key, value, updated_at)
                       VALUES ('active_policy_model_version', ?, datetime('now'))
                       ON CONFLICT(key) DO UPDATE
                       SET value = excluded.value, updated_at = excluded.updated_at""",
                    (model_version,),
                )
                cursor = conn.execute(
                    """UPDATE hypothesis_pairs
                       SET status = 'skipped', reserved_until = NULL
                       WHERE status = 'pending' AND model_version != ?""",
                    (model_version,),
                )
                retired = cursor.rowcount
                for key, value in (
                    ("queue_refill_requested", "true"),
                    ("queue_refill_status", "queued"),
                    (
                        "queue_refill_log",
                        queue_log
                        or f"Policy active; retired {retired} stale pending pairs",
                    ),
                ):
                    conn.execute(
                        """INSERT INTO settings (key, value, updated_at)
                           VALUES (?, ?, datetime('now'))
                           ON CONFLICT(key) DO UPDATE
                           SET value = excluded.value,
                               updated_at = excluded.updated_at""",
                        (key, value),
                    )
        return retired

    def get_policy_activations(self) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT pa.*, previous.model_key AS previous_model_key,
                          current.model_key AS new_model_key
                   FROM policy_activations pa
                   LEFT JOIN model_checkpoints previous
                     ON previous.id = pa.previous_checkpoint_id
                   JOIN model_checkpoints current ON current.id = pa.new_checkpoint_id
                     ORDER BY pa.activated_at DESC, pa.rowid DESC"""
            ).fetchall()
        return [dict(row) for row in rows]

    def get_ready_canary_set(self) -> Optional[Dict]:
        with self._conn() as conn:
            row = conn.execute("""SELECT * FROM canary_sets
                   WHERE status = 'ready'
                         ORDER BY frozen_at DESC, rowid DESC LIMIT 1""").fetchone()
        return dict(row) if row else None

    def get_promotion_gate_canary_set(self) -> Optional[Dict]:
        with self._conn() as conn:
            row = conn.execute("""SELECT * FROM canary_sets
                   WHERE status = 'ready' AND is_promotion_gate = 1
                   LIMIT 1""").fetchone()
        return dict(row) if row else None

    def set_promotion_gate_canary_set(self, canary_set_id: str) -> None:
        """Atomically designate one frozen set as the policy-promotion gate."""
        with self._conn() as conn:
            canary_set = conn.execute(
                "SELECT status, is_promotion_gate FROM canary_sets WHERE id = ?",
                (canary_set_id,),
            ).fetchone()
            if canary_set is None or canary_set["status"] != "ready":
                raise ValueError("The promotion gate must be a locked evaluation set")
            if canary_set["is_promotion_gate"]:
                return
            running = conn.execute("""SELECT value FROM settings
                   WHERE key = 'training_status'""").fetchone()
            if running and running["value"] == "running":
                raise ValueError("Wait for training to finish before changing the gate")
            evaluation = conn.execute("""SELECT value FROM settings
                   WHERE key = 'wer_monitor_evaluation_status'""").fetchone()
            if evaluation and evaluation["value"] == "running":
                raise ValueError(
                    "Wait for model evaluation to finish before changing the gate"
                )
            conn.execute(
                "UPDATE canary_sets SET is_promotion_gate = 0 WHERE is_promotion_gate = 1"
            )
            conn.execute(
                "UPDATE canary_sets SET is_promotion_gate = 1 WHERE id = ?",
                (canary_set_id,),
            )
            for key, value in (
                ("wer_monitor_evaluation_canary_set_id", canary_set_id),
                ("wer_monitor_evaluation_requested", "true"),
                ("wer_monitor_evaluation_status", "queued"),
                ("wer_monitor_evaluation_log", "Waiting for coordinator"),
            ):
                conn.execute(
                    """INSERT INTO settings (key, value, updated_at)
                       VALUES (?, ?, datetime('now'))
                       ON CONFLICT(key) DO UPDATE
                       SET value = excluded.value, updated_at = excluded.updated_at""",
                    (key, value),
                )

    def get_or_create_canary_evaluation(self, evaluation: CanaryEvaluation) -> Dict:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO canary_evaluations
                   (id, canary_set_id, checkpoint_id, profile_hash, status,
                    health_status)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    evaluation.id,
                    evaluation.canary_set_id,
                    evaluation.checkpoint_id,
                    evaluation.profile_hash,
                    evaluation.status,
                    evaluation.health_status,
                ),
            )
            row = conn.execute(
                """SELECT * FROM canary_evaluations
                   WHERE canary_set_id = ? AND checkpoint_id = ? AND profile_hash = ?""",
                (
                    evaluation.canary_set_id,
                    evaluation.checkpoint_id,
                    evaluation.profile_hash,
                ),
            ).fetchone()
        return dict(row)

    def start_canary_evaluation(self, evaluation_id: str) -> bool:
        with self._conn() as conn:
            cursor = conn.execute(
                """UPDATE canary_evaluations
                   SET status = 'running', error_summary = NULL,
                       started_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                   WHERE id = ? AND status IN ('queued', 'failed')""",
                (evaluation_id,),
            )
            if cursor.rowcount != 1:
                return False
            conn.execute(
                "DELETE FROM canary_predictions WHERE evaluation_id = ?",
                (evaluation_id,),
            )
        return True

    def insert_canary_prediction(
        self,
        prediction_id: str,
        evaluation_id: str,
        canary_item_id: str,
        hypothesis: str,
        metrics_json: str,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO canary_predictions
                   (id, evaluation_id, canary_item_id, hypothesis, metrics_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    prediction_id,
                    evaluation_id,
                    canary_item_id,
                    hypothesis,
                    metrics_json,
                ),
            )

    def complete_canary_evaluation(
        self,
        evaluation_id: str,
        aggregate_json: str,
        health_status: str = "unknown",
        warning_json: Optional[str] = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE canary_evaluations
                   SET status = 'completed', health_status = ?, aggregate_json = ?,
                       warning_json = ?,
                       completed_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'),
                       duration_s = (julianday('now') - julianday(started_at)) * 86400.0
                   WHERE id = ?""",
                (health_status, aggregate_json, warning_json, evaluation_id),
            )

    def update_checkpoint_health(self, checkpoint_id: str, health_status: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE model_checkpoints SET health_status = ? WHERE id = ?",
                (health_status, checkpoint_id),
            )

    def fail_canary_evaluation(self, evaluation_id: str, error_summary: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE canary_evaluations
                   SET status = 'failed', health_status = 'unknown', error_summary = ?,
                       completed_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'),
                       duration_s = (julianday('now') - julianday(started_at)) * 86400.0
                   WHERE id = ?""",
                (error_summary, evaluation_id),
            )

    def get_canary_evaluations(self, canary_set_id: str) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT ce.*, mc.model_key, mc.stage, mc.checkpoint_path
                   FROM canary_evaluations ce
                   JOIN model_checkpoints mc ON mc.id = ce.checkpoint_id
                   WHERE ce.canary_set_id = ?
                   ORDER BY mc.created_at, mc.model_key""",
                (canary_set_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_canary_predictions(self, evaluation_id: str) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT cp.*, ci.audio_id, ci.transcript, ci.transcript_provenance,
                          af.file_path, af.dataset
                   FROM canary_predictions cp
                   JOIN canary_items ci ON ci.id = cp.canary_item_id
                   JOIN audio_files af ON af.id = ci.audio_id
                   WHERE cp.evaluation_id = ?
                   ORDER BY ci.audio_id""",
                (evaluation_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    # -- Reward audit ----------------------------------------------------------

    def select_reward_audit_pairs(
        self,
        selection_mode: str,
        limit: int,
        query: str = "",
    ) -> List[Dict]:
        """Select immutable existing pairs for diagnostic reward scoring."""
        if selection_mode not in {"random", "search"}:
            raise ValueError("Reward audit selection must be random or search")
        limit = max(1, min(int(limit), 100))
        query = query.strip().casefold()
        if selection_mode == "search" and not query:
            raise ValueError("Enter an audio ID, pair ID, filename, or path")
        with self._conn() as conn:
            if selection_mode == "random":
                rows = conn.execute(
                    """SELECT hp.id AS pair_id, hp.audio_id, af.file_path
                       FROM hypothesis_pairs hp
                       JOIN audio_files af ON af.id = hp.audio_id
                       ORDER BY random() LIMIT ?""",
                    (limit,),
                ).fetchall()
            else:
                pattern = f"%{query}%"
                rows = conn.execute(
                    """SELECT hp.id AS pair_id, hp.audio_id, af.file_path
                       FROM hypothesis_pairs hp
                       JOIN audio_files af ON af.id = hp.audio_id
                       WHERE lower(hp.id) LIKE ?
                          OR lower(hp.audio_id) LIKE ?
                          OR lower(af.file_path) LIKE ?
                       ORDER BY hp.generated_at DESC, hp.rowid DESC
                       LIMIT ?""",
                    (pattern, pattern, pattern, limit),
                ).fetchall()
        return [dict(row) for row in rows]

    def create_reward_audit_job(
        self,
        job: RewardAuditJob,
        items: List[RewardAuditItem],
    ) -> None:
        if not items:
            raise ValueError("No hypothesis pairs matched the audit selection")
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO reward_audit_jobs
                   (id, checkpoint_path, checkpoint_name, selection_mode,
                    query, sample_size, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    job.id,
                    job.checkpoint_path,
                    job.checkpoint_name,
                    job.selection_mode,
                    job.query,
                    job.sample_size,
                    job.status,
                ),
            )
            conn.executemany(
                """INSERT INTO reward_audit_items (id, job_id, pair_id)
                   VALUES (?, ?, ?)""",
                [(item.id, item.job_id, item.pair_id) for item in items],
            )

    def get_reward_audit_jobs(self, limit: int = 20) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT raj.*, COUNT(rai.id) AS item_count,
                          SUM(CASE WHEN rai.reward_a IS NOT NULL
                                   AND rai.reward_b IS NOT NULL THEN 1 ELSE 0 END)
                              AS scored_count
                   FROM reward_audit_jobs raj
                   LEFT JOIN reward_audit_items rai ON rai.job_id = raj.id
                   GROUP BY raj.id
                   ORDER BY raj.created_at DESC, raj.rowid DESC
                   LIMIT ?""",
                (max(1, min(int(limit), 100)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_reward_audit_job(self, job_id: str) -> Optional[Dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM reward_audit_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return dict(row) if row else None

    def has_queued_reward_audit_job(self) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM reward_audit_jobs WHERE status = 'queued' LIMIT 1"
            ).fetchone()
        return row is not None

    def fail_running_reward_audit_jobs(self, reason: str) -> int:
        """Mark jobs abandoned by a previous coordinator process as failed."""
        with self._conn() as conn:
            cursor = conn.execute(
                """UPDATE reward_audit_jobs
                   SET status = 'failed', error_summary = ?,
                       completed_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                   WHERE status = 'running'""",
                (reason[:1000],),
            )
        return cursor.rowcount

    def get_reward_audit_items(self, job_id: str) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT rai.*, hp.audio_id, hp.hyp_a, hp.hyp_b,
                          hp.strategy_a, hp.strategy_b, hp.model_version,
                          hp.status AS pair_status, af.file_path, af.dataset,
                          a.preference, a.preference_strength, a.correction_text,
                          s.participant_id
                   FROM reward_audit_items rai
                   JOIN hypothesis_pairs hp ON hp.id = rai.pair_id
                   JOIN audio_files af ON af.id = hp.audio_id
                   LEFT JOIN annotations a ON a.id = (
                       SELECT latest.id FROM annotations latest
                       WHERE latest.pair_id = hp.id
                       ORDER BY latest.annotated_at DESC, latest.rowid DESC
                       LIMIT 1
                   )
                   LEFT JOIN annotation_sessions s ON s.id = a.session_id
                   WHERE rai.job_id = ?
                   ORDER BY rai.created_at, rai.rowid""",
                (job_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def claim_next_reward_audit_job(self) -> Optional[Dict]:
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("""SELECT * FROM reward_audit_jobs
                   WHERE status = 'queued'
                   ORDER BY created_at, rowid LIMIT 1""").fetchone()
            if row is None:
                return None
            cursor = conn.execute(
                """UPDATE reward_audit_jobs
                   SET status = 'running', error_summary = NULL,
                       started_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                   WHERE id = ? AND status = 'queued'""",
                (row["id"],),
            )
            if cursor.rowcount != 1:
                return None
            claimed = dict(row)
            claimed["status"] = "running"
        return claimed

    def complete_reward_audit_job(
        self,
        job_id: str,
        scores: List[tuple[str, float, float]],
    ) -> None:
        with self._conn() as conn:
            for pair_id, reward_a, reward_b in scores:
                conn.execute(
                    """UPDATE reward_audit_items SET reward_a = ?, reward_b = ?
                       WHERE job_id = ? AND pair_id = ?""",
                    (reward_a, reward_b, job_id, pair_id),
                )
            conn.execute(
                """UPDATE reward_audit_jobs
                   SET status = 'completed',
                       completed_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                   WHERE id = ? AND status = 'running'""",
                (job_id,),
            )

    def fail_reward_audit_job(self, job_id: str, error_summary: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE reward_audit_jobs
                   SET status = 'failed', error_summary = ?,
                       completed_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                   WHERE id = ?""",
                (error_summary[:1000], job_id),
            )

    # -- Reward Scores ----------------------------------------------------------

    def insert_reward_score(self, score: RewardScore) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO reward_scores
                   (id, pair_id, reward_a, reward_b, source, gemma_version)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    score.id,
                    score.pair_id,
                    score.reward_a,
                    score.reward_b,
                    score.source,
                    score.gemma_version,
                ),
            )

    def get_unscored_annotated_pairs(self) -> List[HypothesisPair]:
        """Return annotated pairs that have no reward score entry yet."""
        with self._conn() as conn:
            rows = conn.execute("""SELECT hp.*
                   FROM hypothesis_pairs hp
                   WHERE hp.status = 'annotated'
                     AND NOT EXISTS (
                         SELECT 1 FROM reward_scores rs WHERE rs.pair_id = hp.id
                     )
                   ORDER BY hp.generated_at ASC""").fetchall()
        return [_row_to_pair(r) for r in rows]

    def get_reward_dataset(self) -> List[Dict]:
        """
        Join most-recent reward scores + hypothesis_pairs + audio_files for PPO.
        Returns one row per pair (most recent score wins on tied pair_id).
        """
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT
                       rs.reward_a,
                       rs.reward_b,
                       rs.source,
                       hp.id               AS pair_id,
                       hp.audio_id,
                       hp.hyp_a,
                       hp.hyp_b,
                       hp.strategy_a,
                       hp.strategy_b,
                       af.file_path,
                       af.reference_transcript
                   FROM reward_scores rs
                   JOIN hypothesis_pairs hp ON hp.id = rs.pair_id
                   JOIN annotation_audio_files af ON af.id = hp.audio_id
                   WHERE rs.id = (
                       SELECT id FROM reward_scores
                       WHERE pair_id = rs.pair_id
                       ORDER BY scored_at DESC, rowid DESC
                       LIMIT 1
                   )
                                     AND EXISTS (
                                             SELECT 1 FROM annotations a
                                             JOIN annotation_sessions s ON s.id = a.session_id
                                             LEFT JOIN annotator_moderation am
                                                 ON am.participant_id = s.participant_id
                                             WHERE a.pair_id = hp.id
                                                 AND a.preference IN ('a', 'b')
                                                 AND a.session_id != ? AND a.is_muted = 0
                                                 AND COALESCE(am.is_muted, 0) = 0
                                     )
                   ORDER BY rs.scored_at ASC""",
                (_AUTO_SESSION_ID,),
            ).fetchall()
        return [dict(r) for r in rows]

    # -- Training Runs ----------------------------------------------------------

    def start_training_run(self, run: TrainingRun) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO training_runs
                   (id, algorithm, whisper_version_before,
                    n_human_annotations, n_synthetic_annotations, config_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    run.id,
                    run.algorithm,
                    run.whisper_version_before,
                    run.n_human_annotations,
                    run.n_synthetic_annotations,
                    run.config_json,
                ),
            )

    def update_training_run_pair_counts(
        self, run_id: str, human_pairs: int, synthetic_pairs: int
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE training_runs
                   SET n_human_annotations = ?, n_synthetic_annotations = ?
                   WHERE id = ?""",
                (human_pairs, synthetic_pairs, run_id),
            )

    def update_training_run_wer_before(
        self,
        run_id: str,
        wer_admed_before: Optional[float],
        wer_cv_before: Optional[float],
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE training_runs
                   SET wer_before_admed = ?, wer_before_cv = ?
                   WHERE id = ?""",
                (wer_admed_before, wer_cv_before, run_id),
            )

    def complete_training_run(
        self,
        run_id: str,
        whisper_version_after: str,
        wer_admed_after: Optional[float],
        wer_cv_after: Optional[float],
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE training_runs SET
                   whisper_version_after = ?,
                   wer_after_admed       = ?,
                   wer_after_cv          = ?,
                   completed_at          = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                   WHERE id = ?""",
                (whisper_version_after, wer_admed_after, wer_cv_after, run_id),
            )

    def get_training_runs(self) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM training_runs ORDER BY started_at ASC"
            ).fetchall()
        return [dict(r) for r in rows]

    # -- Import / Merge ---------------------------------------------------------

    def merge_from_export(self, json_path: Union[str, Path]) -> Dict[str, int]:
        """
        Import a JSON file produced by export_to_json() from another machine.
        Uses INSERT OR IGNORE on every row -- UUID PKs guarantee no collisions.

        Returns a dict of {table_name: rows_inserted} counts.
        """
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))

        _TABLE_ORDER = [
            "audio_files",
            "hypothesis_pairs",
            "audio_curation_events",
            "annotation_sessions",
            "annotations",
            "annotator_moderation",
            "annotation_moderation_events",
            "reward_scores",
            "training_runs",
        ]
        counts: Dict[str, int] = {}
        with self._conn() as conn:
            corpus_ids = {}
            utterance_ids = {}
            for row in data.get("corpora", []):
                corpus = Corpus(
                    id=row["id"], name=row["name"], origin=row["origin"],
                    language=row["language"], provider=row.get("provider"),
                    active=bool(row.get("active", True)),
                    purpose=row.get("purpose", "training"),
                )
                before = conn.total_changes
                registered = self._register_corpus(conn, corpus)
                counts["corpora"] = counts.get("corpora", 0) + conn.total_changes - before
                corpus_ids[row["id"]] = registered["id"]
            for row in data.get("utterances", []):
                utterance = Utterance(
                    row["id"], row["text"], row["normalized_text"], row["language"]
                )
                if utterance.normalized_text != normalize_utterance_text(utterance.text):
                    raise ValueError(f"Invalid exported utterance identity: {utterance.id}")
                before = conn.total_changes
                utterance_ids[row["id"]] = self._register_utterance(conn, utterance)
                counts["utterances"] = counts.get("utterances", 0) + conn.total_changes - before
            counts.setdefault("corpora", 0)
            counts.setdefault("utterances", 0)
            audio_ids = {}
            for table in _TABLE_ORDER:
                rows = data.get(table, [])
                if not rows:
                    counts[table] = 0
                    continue
                cols = list(rows[0].keys())
                col_sql = ", ".join(cols)
                placeholders = ", ".join(["?"] * len(cols))
                inserted = 0
                for original in rows:
                    row = dict(original)
                    if table == "audio_files":
                        if row.get("corpus_id"):
                            row["corpus_id"] = corpus_ids.get(row["corpus_id"], row["corpus_id"])
                        if row.get("utterance_id"):
                            row["utterance_id"] = utterance_ids.get(
                                row["utterance_id"], row["utterance_id"]
                            )
                    elif row.get("audio_id"):
                        row["audio_id"] = audio_ids.get(row["audio_id"], row["audio_id"])
                    cur = conn.execute(
                        f"INSERT OR IGNORE INTO {table} ({col_sql}) VALUES ({placeholders})",  # noqa: S608
                        [row.get(c) for c in cols],
                    )
                    inserted += cur.rowcount
                    if table == "audio_files":
                        stored = conn.execute(
                            "SELECT id FROM audio_files WHERE file_path = ?", (row["file_path"],)
                        ).fetchone()
                        if stored is None:
                            raise ValueError(f"Cannot merge audio: {row['file_path']}")
                        audio_ids[row["id"]] = stored["id"]
                counts[table] = inserted
                if table == "audio_files":
                    self._backfill_catalogue(conn)

        return counts

    # -- Export -----------------------------------------------------------------

    def export_to_json(self, output_path: Union[str, Path]) -> None:
        """
        Dump all tables to a single JSON file for cross-machine merge.
        Transfer via OneDrive / USB / network share, then run merger.py.
        """
        _TABLES = [
            "corpora",
            "utterances",
            "audio_files",
            "audio_curation_events",
            "hypothesis_pairs",
            "annotation_sessions",
            "annotations",
            "annotator_moderation",
            "annotation_moderation_events",
            "reward_scores",
            "training_runs",
        ]
        dump: Dict[str, list] = {}
        with self._conn() as conn:
            for table in _TABLES:
                rows = conn.execute(f"SELECT * FROM {table}").fetchall()  # noqa: S608
                dump[table] = [dict(r) for r in rows]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # -- Settings (key-value store for admin config) ---------------------------

    def get_setting(self, key: str, default: str = "") -> str:
        """Read a settings key, returning `default` when absent."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM settings WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        """Insert or update a settings key."""
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO settings (key, value, updated_at)
                   VALUES (?, ?, datetime('now'))
                   ON CONFLICT(key) DO UPDATE
                   SET value = excluded.value,
                       updated_at = excluded.updated_at""",
                (key, value),
            )

    # -- Active session tracking (heartbeat for concurrent session cap) ---------

    def upsert_session(
        self,
        session_id: str,
        participant_id: str,
        machine_id: str = "",
    ) -> None:
        """Insert or refresh the heartbeat timestamp for a session."""
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO active_sessions
                       (session_id, participant_id, machine_id, last_seen)
                   VALUES (?, ?, ?, datetime('now'))
                   ON CONFLICT(session_id) DO UPDATE
                   SET participant_id = excluded.participant_id,
                       machine_id     = excluded.machine_id,
                       last_seen      = datetime('now')""",
                (session_id, participant_id, machine_id),
            )

    def get_active_sessions(self, timeout_s: int = 120) -> List[Dict]:
        """Return sessions whose last heartbeat was within `timeout_s` seconds.

        Each entry includes an `annotation_count` key with the number of
        non-skip annotations submitted in that session.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT a.session_id, a.participant_id, a.machine_id,
                          a.last_seen, a.started_at,
                          COUNT(an.id) AS annotation_count
                   FROM active_sessions a
                   LEFT JOIN annotations an
                          ON an.session_id = a.session_id
                         AND an.preference != 'skip'
                   WHERE a.last_seen >= datetime('now', '-' || ? || ' seconds')
                   GROUP BY a.session_id
                   ORDER BY a.last_seen DESC""",
                (timeout_s,),
            ).fetchall()
        return [dict(r) for r in rows]

    def count_active_sessions(self, timeout_s: int = 120) -> int:
        """Count sessions with a heartbeat within `timeout_s` seconds."""
        with self._conn() as conn:
            return conn.execute(
                """SELECT COUNT(*) FROM active_sessions
                   WHERE last_seen >= datetime('now', '-' || ? || ' seconds')""",
                (timeout_s,),
            ).fetchone()[0]

    def remove_session(self, session_id: str) -> None:
        """Explicitly remove a session (clean logout)."""
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM active_sessions WHERE session_id = ?",
                (session_id,),
            )

    def purge_stale_sessions(self, timeout_s: int = 120) -> int:
        """Delete sessions silent for more than `timeout_s` seconds.

        Returns the number of sessions removed.
        """
        with self._conn() as conn:
            cursor = conn.execute(
                """DELETE FROM active_sessions
                   WHERE last_seen < datetime('now', '-' || ? || ' seconds')""",
                (timeout_s,),
            )
            return cursor.rowcount

    # -- Stats helpers (used by GUI status bar) ---------------------------------

    def get_stats(self) -> Dict[str, int]:
        """Return a snapshot of key counts for display in the GUI."""
        with self._conn() as conn:
            return {
                "audio_files": conn.execute(
                    "SELECT COUNT(*) FROM audio_files"
                ).fetchone()[0],
                "pending_pairs": conn.execute(
                    "SELECT COUNT(*) FROM hypothesis_pairs WHERE status='pending'"
                ).fetchone()[0],
                "annotated_pairs": conn.execute(
                    "SELECT COUNT(*) FROM hypothesis_pairs WHERE status='annotated'"
                ).fetchone()[0],
                "skipped_pairs": conn.execute(
                    "SELECT COUNT(*) FROM hypothesis_pairs WHERE status='skipped'"
                ).fetchone()[0],
                "human_annotations": conn.execute(
                    """SELECT COUNT(*) FROM annotations a
                                             JOIN annotation_sessions s ON s.id = a.session_id
                                             LEFT JOIN annotator_moderation am
                                                 ON am.participant_id = s.participant_id
                                             WHERE a.preference != 'skip' AND a.session_id != ?
                                                 AND a.is_muted = 0 AND COALESCE(am.is_muted, 0) = 0""",
                    (_AUTO_SESSION_ID,),
                ).fetchone()[0],
                "actionable_annotations": conn.execute(
                    """SELECT COUNT(*) FROM annotations a
                                             JOIN annotation_sessions s ON s.id = a.session_id
                                             LEFT JOIN annotator_moderation am
                                                 ON am.participant_id = s.participant_id
                                             WHERE a.preference IN ('a','b') AND a.session_id != ?
                                                 AND a.is_muted = 0 AND COALESCE(am.is_muted, 0) = 0""",
                    (_AUTO_SESSION_ID,),
                ).fetchone()[0],
                "reward_scores": conn.execute(
                    "SELECT COUNT(*) FROM reward_scores"
                ).fetchone()[0],
                "training_runs": conn.execute(
                    "SELECT COUNT(*) FROM training_runs"
                ).fetchone()[0],
            }


# -- Row -> dataclass converters (module-private) --------------------------------


def _row_to_audio_file(row: sqlite3.Row) -> AudioFile:
    return AudioFile(
        id=row["id"],
        file_path=row["file_path"],
        dataset=row["dataset"],
        reference_transcript=row["reference_transcript"],
        duration_s=row["duration_s"],
        sample_rate=row["sample_rate"],
        created_at=row["created_at"],
        corpus_id=row["corpus_id"],
        utterance_id=row["utterance_id"],
        origin=row["origin"],
        provenance_json=row["provenance_json"],
    )


def _row_to_pair(row: sqlite3.Row) -> HypothesisPair:
    return HypothesisPair(
        id=row["id"],
        audio_id=row["audio_id"],
        hyp_a=row["hyp_a"],
        hyp_b=row["hyp_b"],
        strategy_a=row["strategy_a"],
        strategy_b=row["strategy_b"],
        model_version=row["model_version"],
        generated_at=row["generated_at"],
        status=row["status"],
    )
