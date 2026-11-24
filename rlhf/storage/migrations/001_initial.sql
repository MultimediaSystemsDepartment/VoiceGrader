-- RLHF-ASR Training — initial schema
-- Run via FeedbackStore._init_db() on every startup (all statements are idempotent).
-- All primary keys are UUID4 TEXT — safe for cross-machine JSON merge.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ── Audio catalogue ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audio_files (
    id                   TEXT PRIMARY KEY,
    file_path            TEXT NOT NULL UNIQUE,
    dataset              TEXT NOT NULL CHECK (dataset IN ('admedvoice', 'commonvoice')),
    reference_transcript TEXT,
    duration_s           REAL,
    sample_rate          INTEGER,
    created_at           TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- ── Generated hypothesis pairs (the annotation unit) ──────────────────────────
CREATE TABLE IF NOT EXISTS hypothesis_pairs (
    id            TEXT PRIMARY KEY,
    audio_id      TEXT NOT NULL REFERENCES audio_files(id),
    hyp_a         TEXT NOT NULL,
    hyp_b         TEXT NOT NULL,
    strategy_a    TEXT NOT NULL,   -- e.g. 'beam_5', 'temp_0.8', 'ckpt_v1'
    strategy_b    TEXT NOT NULL,
    model_version TEXT NOT NULL,   -- Whisper checkpoint identifier or HF model ID
    generated_at  TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    status        TEXT NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending', 'annotated', 'skipped'))
);

-- ── One row per annotator sitting ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS annotation_sessions (
    id             TEXT PRIMARY KEY,
    participant_id TEXT NOT NULL,
    machine_id     TEXT NOT NULL,   -- socket.gethostname() of the annotating machine
    started_at     TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    ended_at       TEXT,
    notes          TEXT
);

-- ── One row per annotated pair ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS annotations (
    id                  TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL REFERENCES annotation_sessions(id),
    pair_id             TEXT NOT NULL REFERENCES hypothesis_pairs(id),
    preference          TEXT NOT NULL
                        CHECK (preference IN ('a', 'b', 'equal', 'skip')),
    preference_strength REAL CHECK (
                            preference_strength IS NULL OR
                            (preference_strength >= 0.0 AND preference_strength <= 10.0)
                        ),
    correction_text     TEXT,      -- optional free-text correction by annotator
    notes               TEXT,
    replay_count        INTEGER NOT NULL DEFAULT 0,
    annotated_at        TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- ── Gemma reward scores (offline pass, decoupled from PPO step) ────────────────
CREATE TABLE IF NOT EXISTS reward_scores (
    id            TEXT PRIMARY KEY,
    pair_id       TEXT NOT NULL REFERENCES hypothesis_pairs(id),
    reward_a      REAL NOT NULL,
    reward_b      REAL NOT NULL,
    source        TEXT NOT NULL,   -- 'rlaif_zeroshot' | 'rlaif_fewshot' | 'reward_model_vN'
    gemma_version TEXT NOT NULL,
    scored_at     TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- ── Log of every PPO / REINFORCE / DPO training run ───────────────────────────
CREATE TABLE IF NOT EXISTS training_runs (
    id                      TEXT PRIMARY KEY,
    algorithm               TEXT NOT NULL CHECK (algorithm IN ('ppo', 'reinforce', 'dpo')),
    whisper_version_before  TEXT NOT NULL,
    whisper_version_after   TEXT,
    n_human_annotations     INTEGER,
    n_synthetic_annotations INTEGER,
    wer_before_admed        REAL,
    wer_after_admed         REAL,
    wer_before_cv           REAL,
    wer_after_cv            REAL,
    config_json             TEXT,
    started_at              TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    completed_at            TEXT
);

-- ── Indices for common access patterns ────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_pairs_status        ON hypothesis_pairs(status);
CREATE INDEX IF NOT EXISTS idx_pairs_audio         ON hypothesis_pairs(audio_id);
CREATE INDEX IF NOT EXISTS idx_annotations_pair    ON annotations(pair_id);
CREATE INDEX IF NOT EXISTS idx_annotations_session ON annotations(session_id);
CREATE INDEX IF NOT EXISTS idx_reward_pair         ON reward_scores(pair_id);
CREATE INDEX IF NOT EXISTS idx_reward_scored_at    ON reward_scores(pair_id, scored_at DESC);
