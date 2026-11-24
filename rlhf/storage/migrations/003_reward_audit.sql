-- Read-only reward-model audit jobs and results.
-- These tables are diagnostic only and are never consumed by training.

CREATE TABLE IF NOT EXISTS reward_audit_jobs (
    id              TEXT PRIMARY KEY,
    checkpoint_path TEXT NOT NULL,
    checkpoint_name TEXT NOT NULL,
    selection_mode  TEXT NOT NULL CHECK (selection_mode IN ('random', 'search')),
    query            TEXT,
    sample_size      INTEGER NOT NULL CHECK (sample_size BETWEEN 1 AND 100),
    status           TEXT NOT NULL DEFAULT 'queued'
                     CHECK (status IN ('queued', 'running', 'completed', 'failed')),
    error_summary    TEXT,
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    started_at       TEXT,
    completed_at     TEXT
);

CREATE TABLE IF NOT EXISTS reward_audit_items (
    id         TEXT PRIMARY KEY,
    job_id     TEXT NOT NULL REFERENCES reward_audit_jobs(id) ON DELETE CASCADE,
    pair_id    TEXT NOT NULL REFERENCES hypothesis_pairs(id),
    reward_a   REAL,
    reward_b   REAL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE (job_id, pair_id)
);

CREATE INDEX IF NOT EXISTS idx_reward_audit_jobs_status
    ON reward_audit_jobs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_reward_audit_items_job
    ON reward_audit_items(job_id);
