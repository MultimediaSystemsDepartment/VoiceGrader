CREATE TABLE audio_quality_reviews (
    id TEXT PRIMARY KEY,
    audio_id TEXT NOT NULL REFERENCES audio_files(id),
    state TEXT NOT NULL DEFAULT 'queued'
        CHECK (state IN ('queued', 'running', 'completed', 'failed')),
    profile TEXT,
    metrics_json TEXT,
    error TEXT,
    reviewed_at TEXT,
    reviewer TEXT,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    completed_at TEXT
);
CREATE UNIQUE INDEX one_pending_audio_quality ON audio_quality_reviews(audio_id)
    WHERE state IN ('queued', 'running');
