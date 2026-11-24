-- Designate one frozen evaluation set as the authoritative promotion gate.

ALTER TABLE canary_sets
    ADD COLUMN is_promotion_gate INTEGER NOT NULL DEFAULT 0
    CHECK (is_promotion_gate IN (0, 1));

CREATE UNIQUE INDEX IF NOT EXISTS idx_canary_single_promotion_gate
    ON canary_sets(is_promotion_gate)
    WHERE is_promotion_gate = 1;

UPDATE canary_sets
SET is_promotion_gate = 1
WHERE id = (
    SELECT id
    FROM canary_sets
    WHERE status = 'ready'
    ORDER BY frozen_at DESC, rowid DESC
    LIMIT 1
);