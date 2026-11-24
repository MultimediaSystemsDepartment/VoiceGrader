ALTER TABLE corpora ADD COLUMN purpose TEXT NOT NULL DEFAULT 'training'
    CHECK (purpose IN ('training', 'evaluation'));

DROP VIEW eligible_catalogue_audio;
CREATE VIEW eligible_catalogue_audio AS
SELECT af.* FROM audio_files af
JOIN corpora c ON c.id = af.corpus_id
LEFT JOIN utterances u ON u.id = af.utterance_id
WHERE af.excluded = 0 AND c.active = 1 AND c.language = 'pl'
  AND c.purpose = 'training'
  AND NOT EXISTS (
    SELECT 1 FROM canary_items ci
    JOIN audio_files held ON held.id = ci.audio_id
    JOIN corpora hc ON hc.id = held.corpus_id
    WHERE held.id = af.id OR held.utterance_id = af.utterance_id
       OR (hc.language = c.language
           AND normalize_utterance_text(ci.transcript) = u.normalized_text)
  );
