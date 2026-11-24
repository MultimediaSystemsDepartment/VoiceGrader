-- Preserve audio IDs and inbound references while removing the dataset enum.
-- FeedbackStore applies this rebuild and its UUID4 backfill atomically.

CREATE TABLE IF NOT EXISTS corpora (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    origin     TEXT NOT NULL CHECK (origin IN ('recorded', 'synthesized', 'mixed')),
    language   TEXT NOT NULL,
    provider   TEXT,
    active     INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS utterances (
    id              TEXT PRIMARY KEY,
    text            TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    language        TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE (language, normalized_text)
);

CREATE TABLE IF NOT EXISTS audio_files_catalogue (
    id                   TEXT PRIMARY KEY,
    file_path            TEXT NOT NULL UNIQUE,
    dataset              TEXT NOT NULL,
    reference_transcript TEXT,
    duration_s           REAL,
    sample_rate          INTEGER,
    created_at           TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    excluded             INTEGER NOT NULL DEFAULT 0,
    exclusion_reason     TEXT,
    ever_excluded        INTEGER NOT NULL DEFAULT 0 CHECK (ever_excluded IN (0, 1)),
    excluded_by          TEXT,
    corpus_id            TEXT REFERENCES corpora(id),
    utterance_id         TEXT REFERENCES utterances(id),
    origin               TEXT NOT NULL DEFAULT 'recorded'
                         CHECK (origin IN ('recorded', 'synthesized')),
    provenance_json      TEXT
);

INSERT INTO audio_files_catalogue
    (id, file_path, dataset, reference_transcript, duration_s, sample_rate,
     created_at, excluded, exclusion_reason, ever_excluded, excluded_by)
SELECT id, file_path, dataset, reference_transcript, duration_s, sample_rate,
       created_at, excluded, exclusion_reason, ever_excluded, excluded_by
FROM audio_files;

DROP TABLE audio_files;
ALTER TABLE audio_files_catalogue RENAME TO audio_files;
CREATE INDEX IF NOT EXISTS idx_audio_corpus ON audio_files(corpus_id);
CREATE INDEX IF NOT EXISTS idx_audio_utterance ON audio_files(utterance_id);

CREATE VIEW IF NOT EXISTS learning_audio_files AS
SELECT af.*
FROM audio_files af
JOIN corpora c ON c.id = af.corpus_id
LEFT JOIN utterances u ON u.id = af.utterance_id
WHERE af.excluded = 0 AND af.origin = 'recorded' AND c.active = 1 AND c.language = 'pl'
  AND NOT EXISTS (
      SELECT 1
      FROM canary_items ci
      JOIN audio_files held ON held.id = ci.audio_id
      JOIN corpora held_corpus ON held_corpus.id = held.corpus_id
      WHERE held.id = af.id
         OR held.utterance_id = af.utterance_id
         OR (held_corpus.language = c.language
             AND normalize_utterance_text(ci.transcript) = u.normalized_text)
  );
