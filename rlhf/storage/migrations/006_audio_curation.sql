-- Reversible audio exclusion with append-only dataset-curation provenance.

ALTER TABLE audio_files
    ADD COLUMN ever_excluded INTEGER NOT NULL DEFAULT 0
    CHECK (ever_excluded IN (0, 1));

ALTER TABLE audio_files
    ADD COLUMN excluded_by TEXT;

CREATE TABLE IF NOT EXISTS audio_curation_events (
    id         TEXT PRIMARY KEY,
    audio_id   TEXT NOT NULL REFERENCES audio_files(id),
    pair_id    TEXT REFERENCES hypothesis_pairs(id),
    action     TEXT NOT NULL CHECK (action IN ('exclude', 'restore')),
    actor      TEXT NOT NULL,
    reason     TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_audio_curation_events_audio
    ON audio_curation_events(audio_id, created_at DESC);

UPDATE audio_files
SET ever_excluded = 1,
    excluded_by = 'AK'
WHERE excluded = 1;

INSERT INTO audio_curation_events (id, audio_id, pair_id, action, actor, reason)
SELECT lower(hex(randomblob(4))) || '-' ||
       lower(hex(randomblob(2))) || '-' ||
       '4' || substr(lower(hex(randomblob(2))), 2, 3) || '-' ||
       substr('89ab', abs(random()) % 4 + 1, 1) ||
       substr(lower(hex(randomblob(2))), 2, 3) || '-' ||
       lower(hex(randomblob(6))),
       id,
    (SELECT hp.id FROM hypothesis_pairs hp
     WHERE hp.audio_id = audio_files.id AND hp.status = 'skipped'
     ORDER BY hp.generated_at DESC, hp.rowid DESC LIMIT 1),
       'exclude',
       'AK',
       COALESCE(NULLIF(trim(exclusion_reason), ''), 'legacy exclusion')
FROM audio_files
WHERE excluded = 1;
