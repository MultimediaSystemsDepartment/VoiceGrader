# CONTRACT.md — Design Invariants and "Do Not" Rules

This document records the non-negotiable constraints of the RLHF for ASR system. Any future change — human or AI-driven — must respect these rules. Violations risk data loss, merge collisions, or training instability.

---

## 1. Database Invariants

### 1.1 Schema freeze policy
The SQL schema is defined by the ordered files in `storage/migrations/`.

- **Do not** add, rename, or drop columns without creating a new migration file (`002_*.sql`) and updating `storage/schema.py` dataclasses in sync.
- **Do not** change column types or CHECK constraints of existing tables without a migration.
- Migration application must be **idempotent** through the ordered migration
  ledger and appropriate statement guards. The catalogue rebuild/backfill runs
  in one write-locked transaction, rechecks the ledger after acquiring the lock,
  and verifies foreign keys before commit.

### 1.2 Primary keys are UUID4 — never auto-increment
All PKs are `TEXT UUID4` generated in Python (`uuid.uuid4()`). Auto-increment integers break cross-machine merges. This is the only collision-free strategy for the multi-annotator use case.

- **Do not** introduce integer PKs or `ROWID`-based keys.
- Always use `ClassName.new(...)` factory methods to create records (they call `_new_id()`).
- Direct `ClassName(id=..., ...)` construction is only for deserialising rows read from the DB.

### 1.3 Merge rules
- Same `pair_id` + same `participant_id` → keep most recent (`annotated_at`) — de-duplicate.
- Same `pair_id` + different `participant_id` → keep both rows — IAA data.
- All merge operations use `INSERT OR IGNORE` per table. Never use `INSERT OR REPLACE` (it would silently overwrite IAA rows).

### 1.4 Web authentication

- Normal deployments require password-backed admin sessions.
- Showcase mode is removed. Admin pages/actions always require an admin session;
  annotators retain their supplied participant identities.
- New installations use the public bootstrap password `change-me` only while
  no password configuration exists. Change it before access beyond localhost.
  Malformed or unreadable configuration fails closed, never restoring the default.
- Password hashes live in `_outputs/admin_password.json`, independent of the
  policy lineage, or an explicitly configured `ADMIN_PASSWORD_CFG`. Preserve this
  runtime file across restarts/rebuilds; never commit credential configuration.
- Password changes/reset invalidate existing admin sessions. Web changes require
  the current password. Local recovery uses the shared password utility.
- `GET /setup` must remain read-only and return a page without requiring
  cookies, so deployment probes cannot create sessions or enter redirect loops.
  Annotation sessions start through `POST /setup`.

---

## 2. VRAM Allocation Invariants

### 2.0 Diagnostics privilege and data boundary

- The web service remains CPU-only and must not receive the Docker socket,
  host process privileges, or permission to launch telemetry or ML processes.
- Report-only quality review may dispatch authenticated HTTP requests to the
  local ASR service from the CPU web, but never load its scorers or start its
  processes. Requests/results/errors and human notes persist; scores do not
  change audio eligibility, annotations or TTS participation caps.
- Admin diagnostics controls exchange validated requests through SQLite. The
  coordinator's isolated diagnostics thread performs runtime/GPU/application
  sampling and bundle work. Collector failures must never escape into the
  coordinator polling or training control flow.
- Diagnostics uses only its runtime-visible `/proc`, shared storage, GPU tools,
  HTTP, and SQLite. It must not mount a Docker or container-runtime socket,
  require Kubernetes API credentials, or require host PID/network namespaces.
- OS-level CPU, memory, load, network, and disk-I/O fields describe what the
  diagnostics runtime can observe. In Kubernetes this is pod/runtime telemetry,
  not guaranteed node-wide telemetry. Per-container lifecycle and orchestration
  events are intentionally outside the application diagnostics boundary.
- Sanitized bundles exclude the feedback database, raw logs, transcripts,
  participant IDs, audio, checkpoints, password configuration, tokens, and
  environment dumps.
- Sensitive bundles may add a consistent SQLite snapshot and selected raw
  application logs, but never audio, checkpoints, password configuration, or
  environment dumps.
- Runs use UUID4 identifiers, remain under `_outputs/diagnostics/runs`, and
  cannot delete an active capture or resolve arbitrary filesystem paths.
- Running captures flush records continuously and maintain an atomically
  replaced sanitized partial bundle. Partial bundles contain only complete
  line-oriented records and declare their run state, capture time, and sample
  count. Restart recovery appends to the newest active run and marks superseded
  active manifests interrupted without deleting their artifacts.

GPU model loads default to SDPA attention. Frozen reward-model encoders run in
the configured inference dtype (normally FP16), while the trainable reward MLP
stays FP32. Reward training must cache frozen encoder outputs once per round;
do not restore encoder forwards inside every MLP training epoch.

### 2.1 Reward model and Whisper training can coexist — but unload generation decoder first

The reward model is now a lightweight **AudioTextRewardModel** (Whisper encoder frozen + Polish BERT + small MLP). VRAM budget (RTX 3090, 24 GB):

The Medium generation estimates below are historical reference measurements,
not capacity guarantees for the new Large-v3 policy. The approved short-clip
Large-v3 GPU proof passed (see the root README); long-clip/full-campaign capacity
remains unmeasured.

| Stage | What is in GPU memory | Estimated VRAM |
|-------|-----------------------|---------------|
| Hypothesis generation | Whisper Medium full model (float16) | ~3.5 GB |
| N-best scoring | Whisper encoder (frozen float16) + Polish BERT + MLP | ~2.5 GB |
| RLHF training | Whisper policy (float16 LoRA) + exact float16 Whisper reference | hardware-dependent |
| Annotation GUI | CPU-only | 0 GB |

For the N-best RSO mode (`run_generate_pairs.py --n_candidates N`):
- Unload `HypothesisGenerator` (releases Whisper decoder, ~2 GB) before loading `AudioTextRewardModel`.
- Total peak: ~5.5 GB — well within budget.
- Do NOT keep `HypothesisGenerator` loaded during RM scoring unnecessarily.

### 2.2 Whisper encoder is ALWAYS frozen in the reward model

- `AudioTextRewardModel` loads the Whisper encoder from HuggingFace Hub and **never trains it**.
- Only the MLP head (`module.mlp`) has `requires_grad=True`.
- Checkpoint format: `mlp_head.pt` (MLP state dict) + `rm_config.json`. Encoders are **not saved** — always re-loaded from Hub.
- **Do not** add the Whisper encoder or BERT to the optimizer.

### 2.3 DPO reference must be numerically consistent
The frozen DPO reference uses the same checkpoint, dtype, attention backend, and
deterministic evaluation mode as the round-start policy. Quantizing only the
reference changes the likelihood baseline and is forbidden. DPO must abort
before its first optimizer step when the initial policy/reference preference
offset exceeds the configured invariant.
AMP overflow may skip an update while reducing its scale. Log those skips and
count only completed optimizer updates; gradient diagnostics exclude skipped
updates. Non-finite loss or an unscaled gradient failure aborts training rather
than registering a success-shaped candidate.
An initialized stock-model deployment with no active adapter must remain on the
stock model. Checkpoint discovery must not implicitly resume a newly saved,
unactivated candidate merely because it is the newest adapter on disk.

ASR publication is worker-only and requires the deployed adapter's automatic
activation audit and healthy evaluation on the current frozen monitor/profile.
The converted CUDA/float16 model must pass a paired WER preservation check
before publishing immutable UUID4 artifacts with checksums. Source adapters
must remain immutable. The CPU web only serves authenticated metadata/files
using an independent service credential; proof adapters are not promotion evidence.
ASR hot reload must drain existing file/live and submitted worker operations
before stopping workers. New admissions are explicitly rejected while draining.
Loaded publication identity changes only after readiness. Failed activation
restores the prior installed pointer and workers, or exposes unavailability if
recovery fails. No mid-session model replacement or simultaneous model preload
is required; automatic GPU workload scheduling remains separate.

---

## 3. Audio Invariants

The generic catalogue retains `audio_files` as the asset table and preserves its
IDs and inbound references. `dataset` is a corpus compatibility name, not a
two-dataset enum. Each registered asset links to a corpus and, when a transcript
is available, an utterance. Text identity uses Unicode NFC and collapsed
whitespace while preserving case, punctuation and numbers. Near-duplicate alias
management is not implemented.

Learning queries use `learning_audio_files` with origin-aware selection.
Synthesized assets require the campaign's permitted use and independently
configured SFT/refill hard caps, both defaulting to 0%. SFT selects at most one
TTS rendition per utterance per epoch; requested text never fabricates human
DPO labels. Campaign/job state persists, and ambiguous submissions require
operator review rather than automatic resubmission.
Resubmission acknowledgement may be conveyed by the explicit new-synthesis action
with a hover/focus duplicate-risk warning; it does not require a separate checkbox.
Retrying collection must reuse the existing provider job without another POST.
CPU-only web dispatch owns
campaign submission, polling and validated import, independently of GPU training;
the coordinator must not start a second campaign loop. No local synthesis model
or ML subprocess runs in the web service. Monitor membership excludes all renditions
of the same utterance and matches to verified monitor text, including synthesis
source selection. Retired evaluated monitors retain their exclusions.

New local campaigns use the authenticated REST v0.7 adapter and distinct local
profile IDs; existing external campaigns retain their frozen origin/profile and
legacy adapter. Local estimates report exclusion of texts over 280 characters,
never truncation. Known local 429 rejections may retry boundedly with the same
request ID; explicit audited new synthesis alone changes that ID. Archives must
pass identity, actual voice, real/non-demo and checksum validation. GPU outputs
require the actual reference checksum; OmniVoice also confirms reference text.
The separate CPU API/Piper services have no public ports or GPU access. GPU TTS
profiles require separately approved assets/terms and their own GPU window.

All audio is normalised to **16 kHz, mono, float32** before any processing. This is enforced in `utils/audio.py::load_audio()`.

- **Do not** bypass `load_audio()` and pass raw waveforms elsewhere.
- Generic manifest imports probe native duration and decode through `load_audio()`
  before any corpus/audio writes. Reject clips outside 0.1-30 s rather than
  truncating them and retaining a potentially mismatched reference transcript.
  A single invalid row rejects the entire import.
- Legacy named-dataset imports defer full decoding to inference. The shared
  decoder rejects clips shorter than 0.1 s and truncates clips longer than
  30 s with a warning; this legacy behavior is unchanged.
- New generic imports are inactive by default. Activation is explicit; appending
  assets requires deactivation first. Deactivation blocks new learning/queue
  selection, not a training round that already captured its inputs.
- Browser uploads reuse manifest/audio validation, remain staged until explicit
  acceptance, and reject duplicate audio and evaluation/learning overlap. Evaluation
  corpora have a persisted `purpose="evaluation"` and can never enter learning,
  including after removal of an editable monitor. Evaluation import atomically
  registers inactive assets and an unverified monitor draft; freezing stays explicit.

---

## 4. Algorithm

The project uses **DPO (Direct Preference Optimisation)** as the primary fine-tuning algorithm.

- REINFORCE was removed on 2026-07-15 (see `docs/MAINTENANCE_LOG.md`). Configs with `algorithm='reinforce'` raise `ValueError`.
- PPO is a **stub** (`NotImplementedError`) — requires online rollout collection.
- **Do not** claim PPO is functional until the online rollout loop is implemented.

For iterative (RSO) DPO:
1. Auto-refill generates the human annotation queue from the active Whisper policy.
2. `run_train_reward.py` writes an immutable, round-versioned reward checkpoint from human A/B choices.
3. `run_sft.py` optionally creates a round-scoped policy from human corrections.
	Its warmup is capped at 10% of planned optimizer updates so small rounds do
	not spend their entire update budget below the configured learning rate.
4. `run_generate_synthetic.py` generates a large randomized N-best pool from that exact DPO start policy, ranks it with the current reward checkpoint, and writes temporary JSONL.
5. `run_rlhf.py` mixes current-policy human choices with capped current-round synthetic choices and saves a pending candidate.
6. The frozen model evaluation set caches the intermediate SFT result, then compares both
	the SFT and DPO round outputs with the deployed policy. Each output requires at least
	one changed evaluation transcript and a paired 95% corpus-WER-delta CI upper bound
	at most +0.01. Among outputs with a strictly lower observed corpus WER, activate the
	lowest-WER checkpoint; otherwise retain the deployed policy. Bootstrap samples
	resample clips but recompute corpus WER from summed word errors and reference words;
	do not average per-clip WER for activation.

Repeat each round with the latest Whisper checkpoint as the new policy. See `docs/DPO_PRIMER.md` §11.

**Human queue refill and synthetic DPO generation are distinct.** Refill writes pending pairs for annotators. Synthetic generation is a training stage that must not insert pairs, annotations, or reward scores into SQLite. Its JSONL artifact is deleted after successful publication and retained only for diagnosis/retry after failure.

Human A/B choices are authoritative DPO labels and must never be reversed by reward-model scores. SFT and human DPO consume only feedback on pairs generated by the currently deployed policy; reward-model training may remain cumulative. Synthetic labels are valid only for the policy and reward checkpoint recorded in their artifact. DPO must reject a mismatched artifact and cap synthetic leverage to one pair per current-policy human pair by default.

Annotations are immutable audit records and must not be physically deleted.
Authenticated admins may reversibly mute an individual annotation or every
annotation associated with a self-declared participant ID. Muted annotations
remain visible with their stored reward scores and append-only moderation
events, but are excluded from annotation counts, reward-model training, SFT,
human DPO, and the legacy reward buffer. Annotator muting applies to past and
future sessions using that exact ID; it is not an identity ban, because
annotators have no accounts and may choose another ID. Moderation performed
during training affects future snapshots only and must not mutate the running
round's loaded data.

Audio exclusion is reversible dataset curation, not deletion. Every recording
that has ever been excluded remains visible in Annotation Audit with its
original excluding annotator and append-only exclude/restore decisions. Restore
makes the audio eligible for future learning and reactivates the pair skipped by
that exclusion only when it belongs to the currently deployed policy; otherwise
normal queue refill creates a current-policy pair. Historical exclusions created
before curation provenance was introduced are attributed to `AK`.

The default DPO data floor is one effective batch (32 combined human and
synthetic pairs with the default batch and accumulation settings). Do not reuse
the old 50-pair cumulative threshold after scoping feedback to the current
policy; deployment safety is enforced by the exact-reference checks and the
prospective model-evaluation promotion gate.

Admin training controls are snapshotted at training start and passed as
explicit command arguments. Browser tuning is limited to epochs, learning
rates, the adaptive SFT warmup cap, DPO beta, gradient clipping, epoch-level
early-stopping patience, and guarded synthetic data composition. Synthetic
controls may cap the current-round synthetic-to-human ratio, expand the
candidate pool, and filter by reward margin and token edit distance. They must
not expose reference precision, LoRA scope, batch geometry, or promotion-gate
bypasses. Human preferences remain authoritative regardless of the configured
synthetic ratio.

`active_policy_checkpoint` in DB settings is the deployed policy pointer. DPO training registers a pending candidate but cannot update this pointer; only a passing WER and observable-change gate or audited manual activation/rollback may do so. A behaviorally unchanged candidate must not roll the annotation queue. `run_generate_pairs.py`, `run_sft.py`, and `run_rlhf.py` resolve it automatically, with a valid-checkpoint scan only as recovery from missing or machine-stale settings. Do not confuse this with the reward-model checkpoint: the policy generates text; the reward model only ranks candidates.

VoiceGrader defaults to `openai/whisper-large-v3` in a fresh
`_outputs/large-v3/` state directory. Medium history is not relabeled or migrated
into that policy. Policy adapters must declare the exact matching
`base_model_name_or_path`; missing metadata, malformed configuration and explicit
incompatible checkpoints fail before model loading. Recovery scans log and skip
incompatible adapters. Policy processors determine mel-bin counts. The reward
model's Medium encoder is independent of the Large-v3 transcription policy.

External model publication is separate from deployment. The authenticated
Models & Publishing view may copy only the currently deployed policy with a
completed healthy model evaluation into `latest_model_dir`. Publication
uses a staged directory swap and writes `publication.json` with source,
monitor, base-model, WER, and file-checksum provenance. Downstream applications
consume this stable directory; they must not infer the latest model by scanning
training checkpoints.

When the active policy changes, pending pairs from older policies are marked skipped and refill is queued. Annotated rows remain immutable training/history data. An audio file may have one pair per `model_version`, not one pair forever.

Queue refill samples randomly from audio that is eligible for the active policy. Do not restore insertion-order selection: small annotation rounds would repeatedly overrepresent the beginning of the manifest. Pair reservation remains oldest-first within the generated queue for predictable concurrent assignment.

The DPO frozen reference must equal the policy at the start of that DPO round. `reference_checkpoint` therefore defaults to `resume_from_checkpoint`; using bare base Whisper as the reference while training an SFT/previous-DPO policy is incorrect.

**N-best generation uses a two-pass beam + temperature-sampling hybrid.**

Diverse beam search (`num_beam_groups`) was tried first but collapsed to identical outputs on high-confidence Polish medical speech regardless of `diversity_penalty`. The current strategy:

- **Pass 1 (beam):** `num_beams=5, do_sample=False, num_return_sequences=1` — deterministic best hypothesis.
- **Pass 2 (sample):** `do_sample=True, temperature=config.nbest_temperature (0.8), num_beams=1, num_return_sequences=n_candidates-1` — stochastic alternatives.
- Pairing: beam result is `strategy_a`; the sample with the highest `1 - SequenceMatcher.ratio()` edit distance from the beam result is `strategy_b`.

**Do not** revert to standard beam search (`num_beams=N, num_return_sequences=N`). It collapses all N sequences to the same output on confident domain speech, yielding 0 insertable pairs and triggering an infinite refill loop every 60 s.

**Do not** remove the `skipped_identical` counter and warning in `run_generate_pairs.py`. It is the only signal that temperature is too low.

If collapse rates rise: increase `nbest_temperature` in `inference_config.json` (0.9–1.1 range) before any structural change.

**Post-mortem (2026-07-30):** Diverse beam search produced identical outputs for all ~160 K audio catalogue files. All six beam groups converged because the domain (Polish medical) is narrow and Whisper confidence is very high. Two-pass hybrid now yields ~45–48 insertable pairs per 50-file batch.

---

## 5. CSV Format Invariants

- Delimiter: `;` (semicolon) — matches ADMEDVOICE and CommonVoice source formats.
- **Do not** change to comma without updating `PathsConfig.csv_delimiter` and all readers.
- Column detection is automatic (see `inference/hypothesis_generator.py::_detect_columns`).
- Audio paths in CSV may be relative to `audio_dir` or absolute — both are supported.

---

## 6. Reward Score Sources

The `reward_scores` table is retained for compatibility and analysis. Automated
DPO does not read it; human choices come directly from `annotations`, while
synthetic choices come from the current round's temporary JSONL artifact.

Reward Audit jobs and results live only in `reward_audit_jobs` and
`reward_audit_items`. They are diagnostic snapshots over existing hypothesis
pairs. No reward-training, SFT, human DPO, or synthetic DPO query may read these
tables, and audit scoring must run in the GPU coordinator rather than the
CPU-only web service.

The `reward_scores.source` column tracks provenance:

| Value | Meaning |
|-------|---------|
| `human_cold_start` | Human-annotated cold-start pairs (first ~100 pairs) |
| `reward_model_vN` | AudioTextRewardModel (Whisper enc + Polish BERT), version N |

RLAIF / Gemma-based source labels (`rlaif_zeroshot`, `rlaif_fewshot`) are no longer generated. Rows with those labels from earlier runs are left in place for historical completeness but are not used.

When both cold-start and RM scores exist for a pair, `get_reward_dataset()` returns the **most recent** score. This is intentional — the fine-tuned model supersedes earlier scores.

- **Do not** mix sources when computing baselines for a training run.

---

## 7. LoRA Target Modules

Whisper policy: every module whose name ends in `q_proj` or `v_proj`. PEFT's
suffix matching therefore applies LoRA to encoder self-attention and to decoder
self-attention and cross-attention. A verified Whisper Medium adapter contains
96 trainable LoRA tensors in the encoder and 192 in the decoder.

Do not describe the encoder as frozen during SFT or DPO: its base weights are
frozen, but its attention projections have trainable LoRA updates. Restricting
LoRA to decoder-only would change adapter scope and requires re-running all
training from scratch (existing checkpoints are not compatible with that policy).

---

## 8. Validation Must Be Fast

Any function named `validate_*`, `check_*`, or used during catalogue ingestion **must not** perform full audio decoding. Validation is called on every file in the dataset — tens of thousands of files.

- For plain files: use header-only reads (`sf.info()`, `zf.getinfo()`).
- For zip:-prefixed paths: check ZIP central directory only (existence + non-zero `file_size`). No `zf.read()`, no `load_audio()`.
- Full decode is only permitted inside `load_audio()` itself, which is called on-demand at inference/training time.

If a validation function is taking more than a few seconds on a full dataset, it is doing too much work.

---

## 9. Logging

All modules must use `utils/log.py::get_logger(__name__)`. Do not use `print()` for operational messages or `logging.getLogger()` directly — the central `setup_logging()` call configures file and console handlers, and silences noisy third-party loggers.

---

## 10. Container Responsibility Boundary

The web image (`Dockerfile.web`) is a **lightweight annotation-serving container** — Python slim, CPU-only torch, no CUDA, no `transformers`, no `peft`. It must remain deployable on machines without a GPU.

**Rule: the web process must never spawn ML subprocesses.**

Concretely:
- `web/app.py` and any code imported by it at module level must not import `torch`, `transformers`, `peft`, or any CUDA-dependent package as a top-level import.
- Background tasks in the web process (cleanup loop, queue-refill loop) must not `exec` / `subprocess` any script that loads a model.
- Pair generation (`run_generate_pairs.py`) and all training scripts require GPU; they run only in the coordinator container or in `INLINE_COORDINATOR=1` mode on a GPU host.

**Triggering mechanism for queue refill and training:**

Both features use the same DB-flag pattern to cross the container boundary:

| Flag (settings table) | Set by | Consumed by |
|-----------------------|--------|-------------|
| `training_requested` | web admin panel | coordinator / inline coordinator |
| `queue_refill_requested` | web refill loop or admin panel | coordinator / inline coordinator |
| `annotations_at_last_training` | coordinator (after training completes) | web admin panel ("new since last training" stat) |

The coordinator (`scripts/coordinator.py`) and inline coordinator loop (`web/app.py::_inline_coordinator_loop`) check these flags every poll cycle. Only those paths — running in the GPU environment — spawn `run_*.py` subprocesses.

**Admin panel template context rule:** Values computed from multiple DB settings must be pre-computed in Python (`admin_panel` in `web/app.py`) and passed as explicit keys. Do not perform arithmetic inside Jinja2 using `settings.*` keys that are not in the `settings` dict — the dict is a *curated subset* of DB settings, not the full table. The `new_actionable` counter (actionable annotations since last training) is computed in Python and passed directly; it is not derived in the template.

**Post-mortem (2026-07-30):** An initial implementation of queue auto-refill spawned `run_generate_pairs.py` directly from the web process. This failed in Docker with `ModuleNotFoundError: No module named 'tqdm'` (fixed), then `ModuleNotFoundError: No module named 'transformers'` (architectural). Adding ML dependencies to `requirements.web.txt` would have ballooned the web image and still crashed on `.to("cuda")`. The correct fix was the DB-flag delegation described above.

---

## 11. Machine-Local Data (`_data/`) and Auto-Prepare

`_data/` (source ZIP archives + generated manifest CSVs) is **entirely gitignored** — it never travels with `git pull`/`git clone`. Every machine (dev laptop, GPU host, CI) must independently place the source ZIP(s) in `_data/` and generate manifests before pair generation can find any audio.

- **Do not** assume manifests exist just because the code was updated on a machine. A fresh checkout / pull has an empty `_data/manifests/` until `prepare_data.py` runs there.
- If manifests are missing, `HypothesisGenerator.load_audio_catalogue()` logs a **warning** per missing CSV and silently catalogues 0 files — this is intentional (supports partial datasets) but means downstream steps see "0 unqueued audio files" with no obvious cause.
- **Auto-prepare:** `scripts/coordinator.py::main()` (and the inline coordinator in `web/app.py` under `INLINE_COORDINATOR=1`) call `_ensure_data_prepared()` on startup. If `admedvoice_train_csv` doesn't exist on disk but `_data/archive.zip` is present, `prepare_data.py` is run automatically before the poll loop starts. If `archive.zip` is also missing, a clear warning is logged and the coordinator continues idling rather than crashing.
- **Do not** remove `_ensure_data_prepared()` as "redundant" with the `prepare` Docker service — the Docker service requires a manual `docker compose run --rm prepare` step that is easy to forget on a new machine; auto-prepare is the safety net for the documented `docker compose up web coordinator` one-liner.
- This does **not** replace `docker compose run --rm prepare` as the documented step — it only prevents the silent 0-pairs failure mode when a user forgets it.

**Post-mortem (2026-07-31):** A user pulled code changes to a second machine and ran `docker compose up web coordinator --build` directly. Manifests didn't exist there (never committed, per the gitignore rule above). `run_generate_pairs.py` logged `CSV not found ... skipping` at WARNING level, catalogued 0 files, and the queue-refill loop repeated `+0 pairs` every 60 s indefinitely with no obvious error. Fixed by (a) auto-running `prepare_data.py` on coordinator startup when possible, and (b) escalating the empty-catalogue case in `run_generate_pairs.py` to an actionable ERROR log.

---

## 12. Polish Language Must Be Locked at Every Whisper Call Site

Whisper is multilingual. Without an explicit language/task lock, both `.generate()` calls and teacher-forced training can drift into another language's script (observed: Cyrillic tokens mixed into a Polish hypothesis) — especially under temperature sampling, which explores lower-probability decoding paths.

**Every** place that calls `.generate()` or tokenizes/label-encodes transcripts for teacher forcing must lock `language="pl", task="transcribe"`. Current call sites and how each locks it:

| Site | Mechanism |
|------|-----------|
| `inference/hypothesis_generator.py::_transcribe_files_nbest` (beam + sample passes) | `DecodingStrategy.to_generate_kwargs()` — `language`/`task` default to `"pl"`/`"transcribe"` on `BEAM_5` |
| `evaluation/evaluator.py::_transcribe_batch` | explicit `language="pl", task="transcribe"` kwargs on `.generate()` |
| `rlhf/sft_trainer.py` (SFT teacher forcing) | `processor.tokenizer.set_prefix_tokens(...)` before label tokenization + per-batch `forced_decoder_ids` |
| `rlhf/trainer.py` (DPO teacher forcing) | `processor.tokenizer.set_prefix_tokens(...)` + `base.config.forced_decoder_ids` in `_load_policy` |
| `reward_model/audio_text_reward.py` | N/A — Whisper **encoder only** (mean-pooled), no decoding |

**Do not** add a new `.generate()` call, or a new tokenizer instance used to encode transcript labels for teacher forcing, without also setting the language/task lock. A processor created via `WhisperProcessor.from_pretrained(...)` does **not** default its tokenizer to Polish — `set_prefix_tokens()` must be called explicitly before tokenizing any label text, and `.generate()` calls must pass `language=`/`task=` explicitly (relying on baked-in `forced_decoder_ids` alone is not sufficient since callers can override it).

**Post-mortem (2026-07-31):** `rlhf/trainer.py::_load_policy` created an unlocked processor, so DPO labels lacked the `<|pl|><|transcribe|>` prefix. This affected saved DPO policies and was fixed with `set_prefix_tokens()` plus `forced_decoder_ids`. It did **not** cause the observed mixed-script annotation pair: at that time refill still generated from base Whisper, whose generation calls were already Polish-locked. A language lock constrains Whisper's language prompt but cannot guarantee that stochastic decoding never emits a mixed-script token.

---

## 13. Model Evaluation Health and Policy Activation

The frozen model evaluation set is a prospective holdout selected from raw audio before
transcription. Evaluation-set audio IDs are excluded from queue refill, reward-model
training, SFT corrections, human DPO, and synthetic DPO sampling. Do not add a
learning-data query that omits this exclusion.

Evaluation-set transcripts are immutable while locked. An authenticated admin may
unlock a set for inspection and correction; this creates a new editable
revision rather than mutating the evaluated set. A faulty recording may be
rerolled only in that draft: the old audio is excluded from the catalogue and
the evaluated revision remains intact. Locking the revision computes a new
fingerprint, so old evaluations remain historical and cannot be reused against
changed references. Ground truth is hidden from the normal acquisition query; an
authenticated reveal is audited, and transcript provenance distinguishes
`manual`, `manual_after_reveal`, and `copied_gt`.

Removing an editable monitor draft deletes it and returns clips not held by any
other monitor to normal eligibility. Removing a locked monitor archives it as
`retired`; never delete its items, predictions, or evaluations. Archival is
blocked while monitor evaluation is queued or running.

Exactly one locked evaluation set is explicitly designated as the promotion
gate. The first locked set receives that designation; locking newer sets must
not silently replace it. Automatic policy promotion and stable publication use
only the designated set. Changing the gate is an authenticated, atomic admin
action that queues evaluation of the new gate and is blocked during training or
model evaluation. The designated set cannot be archived until another locked
set has been designated.

DPO before/after reporting uses the designated frozen monitor when available,
with its complete verified reference snapshot pinned across the round and
beam-5 Polish decoding. Its identity/fingerprint keys the baseline cache.
The existing primary-domain training-run WER columns retain their compatibility
names (`wer_*_admed`); runtime evaluator output identifies monitor scores.
Legacy CSV evaluation remains the fallback, not a substitute promotion gate.

Health evaluation is deterministic beam-5 Polish transcription
(`language="pl"`, `task="transcribe"`, `do_sample=False`) under a versioned
profile hash. Completed checkpoint/profile results are cached. Accuracy and
formatting warnings remain diagnostic, but automatic DPO activation is
separately blocked unless the candidate-versus-deployed paired corpus WER 95% CI upper
bound is at most +0.01. Freezing queues evaluation of every uncached checkpoint through the GPU
coordinator; coordinator startup also discovers a frozen monitor created before
this automation existed. Admins can explicitly retry a failed evaluation.

Every adapter activation, including rollback, must update the deployed setting
and insert `policy_activations` atomically through `FeedbackStore`. Rollback
retires stale pending pairs and queues refill, exactly like automatic
publication. Historical parentage that cannot be proven must remain NULL.

---

## Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-06-27 | Initial CONTRACT locked | design session |
| 2026-07-15 | Replaced Gemma 9B RM with AudioTextRewardModel (Whisper enc + Polish BERT); removed REINFORCE; updated VRAM, algorithm, and reward source invariants | architecture overhaul |
| 2026-07-30 | Added §10 container boundary rule after queue-refill post-mortem | architecture fix |
| 2026-07-30 | Replaced diverse beam search with two-pass beam+sample hybrid in §4; updated pipeline step count (5 steps, no generate); documented `annotations_at_last_training` flag and admin template context rule | post-mortem fixes |
| 2026-07-31 | Added §11: `_data/` is gitignored/machine-local; documented auto-prepare in coordinator + inline coordinator after silent 0-pairs post-mortem on a second machine | self-setup fix |
| 2026-07-31 | Added §12: Polish language-lock audit across all Whisper call sites; fixed missing `set_prefix_tokens`/`forced_decoder_ids` in `rlhf/trainer.py` DPO path (mixed-script hypothesis post-mortem) | bug fix |
| 2026-07-31 | Implemented active-policy refill and per-policy queue rollover; made SFT/DPO rounds cumulative; aligned DPO reference with the round-start policy; corrected §4 and §12 attribution | iterative RLHF fix |
| 2026-08-01 | Added §13 prospective Canary isolation, deterministic health profiles, warning-only publication, and audited activation/rollback | model health implementation |
