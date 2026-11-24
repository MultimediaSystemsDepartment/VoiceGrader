#!/usr/bin/env python3
"""
stress_tests/fixtures/seed_db.py — Seed synthetic test pairs into the app database.

Creates N hypothesis_pairs, each backed by its own unique audio_file entry
(virtual paths like /app/_outputs/test_fixtures/seed_{i}.wav — only seed_0.wav
is written to disk). One-pair-per-audio-file ensures that concurrent `exclude`
calls in tests don't cross-contaminate each other's pairs.

All test records carry model_version='test-seed' so they can be cleaned up
without touching real annotation data.

USAGE
-----
    docker compose exec web python stress_tests/fixtures/seed_db.py
    docker compose exec web python stress_tests/fixtures/seed_db.py --pairs 1000
    docker compose exec web python stress_tests/fixtures/seed_db.py --cleanup
"""

import argparse
import sqlite3
import struct
import sys
import wave
from pathlib import Path

sys.path.insert(0, "/app")
from paths import DEFAULT_PATHS
from storage.feedback_store import FeedbackStore
from storage.schema import AudioFile, HypothesisPair

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WAV_DIR = "/app/_outputs/test_fixtures"
WAV_REAL = f"{WAV_DIR}/seed_0.wav"      # only this one is written to disk
WAV_PATTERN = f"{WAV_DIR}/seed_"        # prefix for all virtual paths
TEST_MODEL_VERSION = "test-seed"
TEST_MARKER = "[TEST-SEED]"
N_DEFAULT = 500
MAX_SESSIONS_FOR_TEST = "300"


# ---------------------------------------------------------------------------
# WAV generation
# ---------------------------------------------------------------------------

def _generate_silence_wav(path: str, duration_s: float = 1.0, sample_rate: int = 16000) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(duration_s * sample_rate)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<" + "h" * n_samples, *([0] * n_samples)))
    print(f"[seed] WAV generated: {path}  ({n_samples} samples @ {sample_rate} Hz)")


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def seed(n_pairs: int) -> None:
    store = FeedbackStore(DEFAULT_PATHS.db_path)

    # 1. One real WAV on disk (only seed_0.wav needs to be physically present
    #    for GET /audio requests in E2E tests; DB tests don't read the file).
    if not Path(WAV_REAL).exists():
        _generate_silence_wav(WAV_REAL)
    else:
        print(f"[seed] WAV already exists: {WAV_REAL}")

    # 2. Insert N audio_file rows with unique virtual paths + 1 pair each.
    #    Only seed_0.wav is a real file; the rest are DB-only virtual paths.
    conn = sqlite3.connect(str(DEFAULT_PATHS.db_path))
    conn.row_factory = sqlite3.Row

    inserted_pairs = 0
    skipped = 0
    for i in range(n_pairs):
        vpath = f"{WAV_DIR}/seed_{i}.wav"

        # Find or create audio_file row
        row = conn.execute(
            "SELECT id FROM audio_files WHERE file_path = ?", (vpath,)
        ).fetchone()
        if row:
            audio_id = row["id"]
            skipped += 1
        else:
            af = AudioFile.new(
                file_path=vpath,
                dataset="admedvoice",
                reference_transcript=f"{TEST_MARKER} Testowe nagranie {i}.",
                duration_s=1.0,
                sample_rate=16000,
            )
            store.insert_audio_file(af)
            audio_id = af.id

        pair = HypothesisPair.new(
            audio_id=audio_id,
            hyp_a=f"{TEST_MARKER} Hipoteza A numer {i + 1}.",
            hyp_b=f"{TEST_MARKER} Hipoteza B numer {i + 1}.",
            strategy_a="beam_5",
            strategy_b="temp_0.8",
            model_version=TEST_MODEL_VERSION,
        )
        store.insert_hypothesis_pair(pair)
        inserted_pairs += 1

    conn.close()
    if skipped:
        print(f"[seed] {skipped} audio_file(s) already existed (reused).")
    print(f"[seed] Inserted {inserted_pairs} hypothesis_pairs (status=pending).")

    # 3. Raise session cap
    old_cap = store.get_setting("max_concurrent_sessions", "10")
    store.set_setting("max_concurrent_sessions", MAX_SESSIONS_FOR_TEST)
    print(
        f"[seed] max_concurrent_sessions: {old_cap} -> {MAX_SESSIONS_FOR_TEST}"
    )

    print(f"[seed] DB stats: {store.get_stats()}")


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup() -> None:
    conn = sqlite3.connect(str(DEFAULT_PATHS.db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    cur = conn.execute(
        """DELETE FROM annotations
           WHERE pair_id IN (
               SELECT id FROM hypothesis_pairs WHERE model_version = ?
           )""",
        (TEST_MODEL_VERSION,),
    )
    print(f"[cleanup] Deleted {cur.rowcount} annotation(s).")

    cur = conn.execute(
        """DELETE FROM reward_scores
           WHERE pair_id IN (
               SELECT id FROM hypothesis_pairs WHERE model_version = ?
           )""",
        (TEST_MODEL_VERSION,),
    )
    print(f"[cleanup] Deleted {cur.rowcount} reward_score(s).")

    cur = conn.execute(
        "DELETE FROM hypothesis_pairs WHERE model_version = ?",
        (TEST_MODEL_VERSION,),
    )
    print(f"[cleanup] Deleted {cur.rowcount} hypothesis_pair(s).")

    cur = conn.execute(
        "DELETE FROM audio_files WHERE file_path LIKE ?",
        (f"{WAV_PATTERN}%",),
    )
    print(f"[cleanup] Deleted {cur.rowcount} audio_file(s).")

    conn.commit()
    conn.close()

    # Remove real WAV and directory if empty
    wav = Path(WAV_REAL)
    if wav.exists():
        wav.unlink()
        print(f"[cleanup] Deleted WAV: {WAV_REAL}")
        try:
            wav.parent.rmdir()
            print(f"[cleanup] Removed dir: {wav.parent}")
        except OSError:
            pass

    print("[cleanup] Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Seed or clean up test data in the RLHF-ASR database."
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=N_DEFAULT,
        help=f"Number of hypothesis pairs to insert (default: {N_DEFAULT})",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove all test data (model_version='test-seed') instead of seeding.",
    )
    args = parser.parse_args()

    if args.cleanup:
        cleanup()
    else:
        seed(args.pairs)
