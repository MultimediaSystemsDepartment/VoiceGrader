-- Durable model lineage, canary benchmark, and health evaluation state.
-- Additive and idempotent: safe for existing 001 databases.

CREATE TABLE IF NOT EXISTS training_rounds (
    id                         TEXT PRIMARY KEY,
    round_key                  TEXT NOT NULL UNIQUE,
    starting_checkpoint_id     TEXT REFERENCES model_checkpoints(id),
    reward_checkpoint_path     TEXT,
    sft_checkpoint_id          TEXT REFERENCES model_checkpoints(id),
    dpo_checkpoint_id          TEXT REFERENCES model_checkpoints(id),
    config_json                TEXT,
    n_human_annotations        INTEGER,
    n_synthetic_annotations    INTEGER,
    status                     TEXT NOT NULL DEFAULT 'running'
                               CHECK (status IN ('running', 'completed', 'failed')),
    health_status              TEXT NOT NULL DEFAULT 'unknown'
                               CHECK (health_status IN ('healthy', 'warning', 'unknown')),
    warning_json               TEXT,
    error_summary              TEXT,
    started_at                 TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    completed_at               TEXT
);

CREATE TABLE IF NOT EXISTS training_round_steps (
    id             TEXT PRIMARY KEY,
    round_id       TEXT NOT NULL REFERENCES training_rounds(id) ON DELETE CASCADE,
    stage          TEXT NOT NULL,
    ordinal        INTEGER NOT NULL,
    status         TEXT NOT NULL DEFAULT 'pending'
                   CHECK (status IN ('pending', 'running', 'completed', 'failed', 'skipped')),
    command_json   TEXT,
    error_summary  TEXT,
    started_at     TEXT,
    completed_at   TEXT,
    duration_s     REAL,
    UNIQUE (round_id, ordinal)
);

CREATE TABLE IF NOT EXISTS model_checkpoints (
    id                    TEXT PRIMARY KEY,
    model_key             TEXT NOT NULL UNIQUE,
    checkpoint_path       TEXT,
    stage                 TEXT NOT NULL CHECK (stage IN ('base', 'sft', 'dpo')),
    parent_checkpoint_id  TEXT REFERENCES model_checkpoints(id),
    round_id              TEXT REFERENCES training_rounds(id),
    provenance            TEXT NOT NULL DEFAULT 'pipeline'
                          CHECK (provenance IN ('pipeline', 'discovered')),
    health_status         TEXT NOT NULL DEFAULT 'unknown'
                          CHECK (health_status IN ('healthy', 'warning', 'unknown')),
    available             INTEGER NOT NULL DEFAULT 1 CHECK (available IN (0, 1)),
    metadata_json         TEXT,
    created_at            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS policy_activations (
    id                      TEXT PRIMARY KEY,
    previous_checkpoint_id  TEXT REFERENCES model_checkpoints(id),
    new_checkpoint_id       TEXT NOT NULL REFERENCES model_checkpoints(id),
    reason                  TEXT NOT NULL CHECK (reason IN ('automatic', 'manual', 'rollback', 'recovery')),
    actor                   TEXT NOT NULL DEFAULT 'system',
    warning_present         INTEGER NOT NULL DEFAULT 0 CHECK (warning_present IN (0, 1)),
    activated_at            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS canary_sets (
    id                 TEXT PRIMARY KEY,
    name               TEXT NOT NULL,
    selection_seed     INTEGER NOT NULL,
    selection_method   TEXT NOT NULL,
    target_size        INTEGER NOT NULL,
    status             TEXT NOT NULL DEFAULT 'collecting'
                       CHECK (status IN ('collecting', 'ready', 'retired')),
    fingerprint        TEXT,
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    frozen_at          TEXT
);

CREATE TABLE IF NOT EXISTS canary_items (
    id                    TEXT PRIMARY KEY,
    canary_set_id         TEXT NOT NULL REFERENCES canary_sets(id) ON DELETE CASCADE,
    audio_id              TEXT NOT NULL REFERENCES audio_files(id),
    transcript            TEXT,
    transcript_provenance TEXT CHECK (transcript_provenance IN ('manual', 'copied_gt', 'manual_after_reveal')),
    gt_available          INTEGER NOT NULL DEFAULT 0 CHECK (gt_available IN (0, 1)),
    gt_revealed           INTEGER NOT NULL DEFAULT 0 CHECK (gt_revealed IN (0, 1)),
    verified_at           TEXT,
    created_at            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE (canary_set_id, audio_id)
);

CREATE TABLE IF NOT EXISTS canary_evaluations (
    id                  TEXT PRIMARY KEY,
    canary_set_id       TEXT NOT NULL REFERENCES canary_sets(id),
    checkpoint_id       TEXT NOT NULL REFERENCES model_checkpoints(id),
    profile_hash        TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'queued'
                        CHECK (status IN ('queued', 'running', 'completed', 'failed')),
    health_status       TEXT NOT NULL DEFAULT 'unknown'
                        CHECK (health_status IN ('healthy', 'warning', 'unknown')),
    aggregate_json      TEXT,
    warning_json        TEXT,
    error_summary       TEXT,
    started_at          TEXT,
    completed_at        TEXT,
    duration_s          REAL,
    UNIQUE (canary_set_id, checkpoint_id, profile_hash)
);

CREATE TABLE IF NOT EXISTS canary_predictions (
    id                  TEXT PRIMARY KEY,
    evaluation_id       TEXT NOT NULL REFERENCES canary_evaluations(id) ON DELETE CASCADE,
    canary_item_id      TEXT NOT NULL REFERENCES canary_items(id),
    hypothesis          TEXT NOT NULL,
    metrics_json        TEXT NOT NULL,
    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE (evaluation_id, canary_item_id)
);

CREATE INDEX IF NOT EXISTS idx_round_steps_round ON training_round_steps(round_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_checkpoints_round ON model_checkpoints(round_id);
CREATE INDEX IF NOT EXISTS idx_activations_time ON policy_activations(activated_at DESC);
CREATE INDEX IF NOT EXISTS idx_canary_items_audio ON canary_items(audio_id);
CREATE INDEX IF NOT EXISTS idx_canary_eval_checkpoint ON canary_evaluations(checkpoint_id, status);
CREATE INDEX IF NOT EXISTS idx_canary_predictions_eval ON canary_predictions(evaluation_id);
