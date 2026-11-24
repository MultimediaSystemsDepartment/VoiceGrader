# Data Schema Reference

Schema source: ordered files in `storage/migrations/`  
Python dataclasses: `storage/schema.py`  
All operations: `storage/feedback_store.py`

---

## Table Overview

```
audio_files          Audio catalogue (one row per file)
corpora              Generic dataset/source metadata and activation
utterances           Shared linguistic identity across audio renditions
hypothesis_pairs     Competing transcript pairs (one per annotation unit)
annotation_sessions  Annotator sitting (one per session)
annotations          Pairwise preference or skip (one per pair annotated)
reward_scores        AudioTextRewardModel scalar scores for a pair
reward_audit_jobs    Diagnostic reward-checkpoint scoring requests
reward_audit_items   Selected pairs and audit-only utility results
training_runs        Log of every RLHF fine-tuning run
```

---

## Table: `audio_files`

| Column | Type | Notes |
|--------|------|-------|
| `id` | TEXT PK | UUID4 |
| `file_path` | TEXT UNIQUE NOT NULL | Plain file path or `zip:archive!member`; project-relative paths are supported |
| `dataset` | TEXT NOT NULL | Compatibility name for the linked corpus; not a fixed enum |
| `reference_transcript` | TEXT | Ground-truth transcript; may be NULL |
| `duration_s` | REAL | Clip duration in seconds |
| `sample_rate` | INTEGER | Native sample rate (before resampling) |
| `created_at` | TEXT | ISO-8601 UTC, auto-set |
| `corpus_id` | TEXT FK | Linked `corpora` record |
| `utterance_id` | TEXT FK | Linked text identity; NULL without a transcript |
| `origin` | TEXT | `recorded` or `synthesized` |
| `provenance_json` | TEXT | Optional JSON object for source/provider metadata |

**Insert:** `FeedbackStore.insert_audio_file()` links corpus/utterance identities
and preserves idempotence by file path. Bulk registration returns the actual
new-asset count.
**Query:** `count_audio_files(dataset=None)`, `get_unqueued_audio_files(limit)`.

**Generic manifest import:** `import_corpus_audio()` registers a new inactive
corpus and its validated assets in one write-locked transaction. Existing paths
must match corpus/origin/text identity; conflicts roll back all new records.
It never changes an existing transcript or asset ID. `set_corpus_active()` toggles
future learning eligibility without deleting assets, pairs or feedback.
`get_corpus_summaries()` reports recording, distinct-utterance and learning-eligible
counts. `get_corpus_audio_files()` bounds preview queries; both reuse
`learning_audio_files` rather than implementing different browser exclusions.

Migration `007_catalogue.sql` preserves legacy audio IDs, annotations and monitor
relationships. Its table rebuild and UUID4 backfill are atomic and protected by
the migration ledger/write lock.

`corpora` has UUID4 ID, unique name, origin (`recorded`, `synthesized`, `mixed`),
language, optional provider and active state. Register explicit corpus metadata
through `FeedbackStore.register_corpus(Corpus.new(...))` before adding assets
where the defaults (recorded Polish) do not apply.

`utterances` stores original text, NFC/whitespace-normalized text and language.
Identity is unique by language and normalized text; case, punctuation and numbers
remain significant. Blank transcripts have no text identity.

The shared `learning_audio_files` view excludes inactive/non-Polish corpora,
synthesized assets, curated exclusions, monitor audio, related utterances and
verified monitor text. SQLite connections querying this view must register the
normalization function, as `FeedbackStore` does. Synthesis sources are exposed
through `get_synthesis_source_utterances()` and use the same exclusions.
Synthesized asset ingestion/sampling is not yet enabled.

JSON exports include corpora and utterances. Merge remaps natural-identity
conflicts before linking assets and supports legacy exports without these tables.

---

## Table: `hypothesis_pairs`

| Column | Type | Notes |
|--------|------|-------|
| `id` | TEXT PK | UUID4 |
| `audio_id` | TEXT FK → `audio_files.id` | |
| `hyp_a` | TEXT NOT NULL | First transcription hypothesis |
| `hyp_b` | TEXT NOT NULL | Second transcription hypothesis |
| `strategy_a` | TEXT NOT NULL | e.g. `'beam_5'`, `'temp_0.8'`, `'ckpt_v1'` |
| `strategy_b` | TEXT NOT NULL | |
| `model_version` | TEXT NOT NULL | Whisper checkpoint identifier used for generation |
| `generated_at` | TEXT | ISO-8601 UTC, auto-set |
| `status` | TEXT | `'pending'` / `'annotated'` / `'skipped'` (CHECK) |

**Insert:** `insert_hypothesis_pair()`.  
**Query:** `get_pending_pairs(limit)`, `count_pending_pairs()`, `mark_pair_status()`.

---

## Table: `annotation_sessions`

| Column | Type | Notes |
|--------|------|-------|
| `id` | TEXT PK | UUID4 |
| `participant_id` | TEXT NOT NULL | Annotator handle (entered in pre-session dialog) |
| `machine_id` | TEXT NOT NULL | Hostname (`socket.gethostname()`) |
| `started_at` | TEXT | ISO-8601 UTC, auto-set |
| `ended_at` | TEXT | Set on session close |
| `notes` | TEXT | Optional free-text |

**Insert:** `open_session(session)`.  
**Update:** `close_session(session_id)` sets `ended_at`.

---

## Table: `annotations`

| Column | Type | Notes |
|--------|------|-------|
| `id` | TEXT PK | UUID4 |
| `session_id` | TEXT FK → `annotation_sessions.id` | |
| `pair_id` | TEXT FK → `hypothesis_pairs.id` | |
| `preference` | TEXT NOT NULL | `'a'` / `'b'` / `'equal'` / `'skip'` (CHECK) |
| `preference_strength` | REAL | 0–10 slider; NULL if not used |
| `correction_text` | TEXT | Free-text correction (single mode only) |
| `notes` | TEXT | Optional |
| `replay_count` | INTEGER DEFAULT 0 | How many times audio was replayed |
| `annotated_at` | TEXT | ISO-8601 UTC, auto-set |

**Insert:** `insert_annotation()` — atomically updates `hypothesis_pairs.status`.  
**Query:** `count_human_annotations()` (excludes skips), `get_preference_dataset()` (only `'a'`/`'b'` rows, joined for reward training).

---

## Table: `reward_scores`

| Column | Type | Notes |
|--------|------|-------|
| `id` | TEXT PK | UUID4 |
| `pair_id` | TEXT FK → `hypothesis_pairs.id` | |
| `reward_a` | REAL NOT NULL | Unbounded utility logit for hyp_a |
| `reward_b` | REAL NOT NULL | Unbounded utility logit for hyp_b |
| `source` | TEXT NOT NULL | `'rlaif_zeroshot'` / `'reward_model_vN'` |
| `gemma_version` | TEXT NOT NULL | Gemma model ID or checkpoint path |
| `scored_at` | TEXT | ISO-8601 UTC, auto-set |

**Insert:** `insert_reward_score()`.  
**Query:** `get_unscored_annotated_pairs()`, `get_reward_dataset()` (most-recent score per pair, joined with `hypothesis_pairs` and `audio_files`).

Reward convention (from RLAIF):

| Outcome | reward_a | reward_b |
|---------|----------|----------|
| A is better | 1.0 | 0.0 |
| B is better | 0.0 | 1.0 |
| Equal | 0.5 | 0.5 |

---

## Tables: `reward_audit_jobs`, `reward_audit_items`

`reward_audit_jobs` records the selected reward checkpoint, random/search
selection parameters, status, timestamps, and failure summary. Its child
`reward_audit_items` rows reference immutable `hypothesis_pairs` and hold
nullable unbounded `reward_a` / `reward_b` utility logits until scoring finishes.

These tables are diagnostic only. They are not read by reward training, SFT,
human DPO, or synthetic DPO.

---

## Table: `training_runs`

| Column | Type | Notes |
|--------|------|-------|
| `id` | TEXT PK | UUID4 |
| `algorithm` | TEXT NOT NULL | `'ppo'` / `'reinforce'` / `'dpo'` (CHECK) |
| `whisper_version_before` | TEXT NOT NULL | Model ID or LoRA path |
| `whisper_version_after` | TEXT | LoRA checkpoint path post-training |
| `n_human_annotations` | INTEGER | Count at training time |
| `n_synthetic_annotations` | INTEGER | RLAIF annotations used |
| `wer_before_admed` | REAL | ADMEDVOICE WER before fine-tuning |
| `wer_after_admed` | REAL | ADMEDVOICE WER after fine-tuning |
| `wer_before_cv` | REAL | CommonVoice WER before fine-tuning |
| `wer_after_cv` | REAL | CommonVoice WER after fine-tuning |
| `config_json` | TEXT | Serialised `RLHFConfig` at training time |
| `started_at` | TEXT | ISO-8601 UTC, auto-set |
| `completed_at` | TEXT | Set by `complete_training_run()` |

**Insert:** `start_training_run()`.  
**Update:** `update_training_run_wer_before()`, `complete_training_run()`.  
**Query:** `get_training_runs()` → list of dicts used by `evaluation/report.py`.

---

## Indexes

```sql
idx_audio_files_dataset          ON audio_files(dataset)
idx_hypothesis_pairs_audio_id    ON hypothesis_pairs(audio_id)
idx_hypothesis_pairs_status      ON hypothesis_pairs(status)
idx_annotations_pair_id          ON annotations(pair_id)
idx_annotations_session_id       ON annotations(session_id)
idx_reward_scores_pair_id        ON reward_scores(pair_id)
```

---

## JSON Export Format

`FeedbackStore.export_to_json(path)` writes one object per table:

```json
{
  "audio_files":          [...],
  "hypothesis_pairs":     [...],
  "annotation_sessions":  [...],
  "annotations":          [...],
  "reward_scores":        [...],
  "training_runs":        [...]
}
```

Each array contains flat dicts matching the table column names. Import via `merge_from_export(json_path)` — all inserts are `INSERT OR IGNORE`.
