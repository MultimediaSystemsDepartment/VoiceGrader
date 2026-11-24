CREATE TABLE IF NOT EXISTS tts_campaigns (
    id TEXT PRIMARY KEY,
    source_corpus_id TEXT NOT NULL REFERENCES corpora(id),
    intended_use TEXT NOT NULL CHECK (intended_use IN ('sft','annotation','both')),
    selection_mode TEXT NOT NULL CHECK (selection_mode IN ('count','coverage','all')),
    selection_value REAL NOT NULL,
    profiles_json TEXT NOT NULL,
    provider_url TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','cancelled')),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS tts_items (
    id TEXT PRIMARY KEY,
    utterance_id TEXT NOT NULL REFERENCES utterances(id),
    profile_id TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    provider_url TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending' CHECK (
        state IN ('pending','submitting','accepted','polling','completed',
                  'failed','review_required','cancelled')),
    job_id TEXT,
    audio_id TEXT REFERENCES audio_files(id),
    attempts INTEGER NOT NULL DEFAULT 0,
    get_failures INTEGER NOT NULL DEFAULT 0,
    next_attempt_at REAL NOT NULL DEFAULT 0,
    lease_until REAL NOT NULL DEFAULT 0,
    error TEXT,
    review_log_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (utterance_id, profile_id, provider_url)
);
CREATE TABLE IF NOT EXISTS tts_campaign_members (
    campaign_id TEXT NOT NULL REFERENCES tts_campaigns(id),
    item_id TEXT NOT NULL REFERENCES tts_items(id),
    PRIMARY KEY (campaign_id, item_id)
);
CREATE INDEX IF NOT EXISTS idx_tts_work ON tts_items(state, next_attempt_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_tts_provider_job
    ON tts_items(provider_url, job_id) WHERE job_id IS NOT NULL;

-- Shared exclusion predicate, irrespective of rendition or intended use.
CREATE VIEW IF NOT EXISTS eligible_catalogue_audio AS
SELECT af.* FROM audio_files af
JOIN corpora c ON c.id = af.corpus_id
LEFT JOIN utterances u ON u.id = af.utterance_id
WHERE af.excluded = 0 AND c.active = 1 AND c.language = 'pl'
  AND NOT EXISTS (
    SELECT 1 FROM canary_items ci
    JOIN audio_files held ON held.id = ci.audio_id
    JOIN corpora hc ON hc.id = held.corpus_id
    WHERE held.id = af.id OR held.utterance_id = af.utterance_id
       OR (hc.language = c.language
           AND normalize_utterance_text(ci.transcript) = u.normalized_text)
  );
CREATE VIEW IF NOT EXISTS permitted_tts_audio AS
SELECT DISTINCT af.*, tc.intended_use AS campaign_use
FROM eligible_catalogue_audio af
JOIN tts_items ti ON ti.audio_id = af.id AND ti.state = 'completed'
JOIN tts_campaign_members tm ON tm.item_id = ti.id
JOIN tts_campaigns tc ON tc.id = tm.campaign_id
JOIN corpora source ON source.id = tc.source_corpus_id
WHERE af.origin = 'synthesized' AND source.active = 1;
DROP VIEW IF EXISTS learning_audio_files;
CREATE VIEW learning_audio_files AS
SELECT af.* FROM eligible_catalogue_audio af
WHERE af.origin = 'recorded'
   OR EXISTS (SELECT 1 FROM permitted_tts_audio t WHERE t.id = af.id);
CREATE VIEW IF NOT EXISTS annotation_audio_files AS
SELECT af.* FROM eligible_catalogue_audio af
WHERE af.origin = 'recorded'
   OR EXISTS (SELECT 1 FROM permitted_tts_audio t
              WHERE t.id = af.id AND t.campaign_use IN ('annotation','both'));
CREATE VIEW IF NOT EXISTS sft_audio_files AS
SELECT af.* FROM eligible_catalogue_audio af
WHERE af.origin = 'recorded'
   OR EXISTS (SELECT 1 FROM permitted_tts_audio t
              WHERE t.id = af.id AND t.campaign_use IN ('sft','both'));
