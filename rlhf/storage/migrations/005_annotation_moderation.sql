-- Reversible annotation and self-declared annotator muting with audit history.

ALTER TABLE annotations
    ADD COLUMN is_muted INTEGER NOT NULL DEFAULT 0
    CHECK (is_muted IN (0, 1));

CREATE TABLE IF NOT EXISTS annotator_moderation (
    id             TEXT PRIMARY KEY,
    participant_id TEXT NOT NULL UNIQUE,
    is_muted       INTEGER NOT NULL DEFAULT 0 CHECK (is_muted IN (0, 1)),
    updated_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_by     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS annotation_moderation_events (
    id             TEXT PRIMARY KEY,
    target_type    TEXT NOT NULL CHECK (target_type IN ('annotation', 'annotator')),
    target_id      TEXT NOT NULL,
    action         TEXT NOT NULL CHECK (action IN ('mute', 'unmute')),
    actor          TEXT NOT NULL,
    reason         TEXT,
    created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_annotations_muted
    ON annotations(is_muted, annotated_at DESC);
CREATE INDEX IF NOT EXISTS idx_moderation_events_target
    ON annotation_moderation_events(target_type, target_id, created_at DESC);