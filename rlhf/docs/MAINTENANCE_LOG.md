# Maintenance Log

Records of significant architectural decisions, refactors, and rationale.
Each entry is dated and describes *what* changed, *why*, and *what was removed or added*.
Intended audience: future developers and AI coding agents picking up this project.

---

## 2026-07-31 — Ephemeral Synthetic DPO Leverage

An architectural audit found that reward-ranked generation only populated the
human annotation queue; no large synthetic DPO pool existed. DPO also oriented
human-reviewed pairs by persisted reward scores, allowing an old reward model
to override the annotator and mixing stale scores across reward-model versions.

- Human A/B choices are now durable, authoritative DPO labels.
- Each round trains an immutable reward checkpoint, optionally runs SFT, then
  generates a randomized N-best synthetic pool from the exact DPO start policy.
- Synthetic best/worst labels live in round-scoped JSONL, never in SQLite, and
  are deleted after successful publication.
- DPO validates artifact policy identity and caps synthetic pairs relative to
  human pairs.
- Partial gradient-accumulation tails now produce an optimizer step.
- Failed automatic rounds wait for new feedback or a manual retry instead of
  restarting expensive GPU stages every poll.
- Reward-model output is an unbounded utility logit suitable for Bradley-Terry
  differences rather than a sigmoid-bounded score.

---

## 2026-07-31 — Active-Policy RLHF Handoff

An audit found that refill passed the latest reward-model checkpoint but always generated with base `openai/whisper-medium`; repeated training also restarted SFT from base, and DPO used base Whisper rather than the round-start policy as its frozen reference. Documentation described iterative DPO although those handoffs were absent.

- Added the portable `active_policy_checkpoint` DB setting and valid-LoRA fallback resolution.
- Pair generation now attaches the active PEFT adapter and records adapter-aware `model_version` provenance.
- New policy publication retires stale pending pairs, preserves annotations, and permits one pair per audio per policy version.
- SFT and DPO continue from the active policy; DPO reference defaults to that same starting adapter.
- Inline and Docker refill now both use the latest reward model and defer queued refill until training finishes.
- Removed report-only evaluation from the coordinator pipeline; `run_rlhf.py` already evaluates and records before/after WER.

---

## 2026-07-30 — Two-Pass Generation, Container Boundary Hardening, Auto-Refill, Admin Stats

### Context

First end-to-end training run completed. Debugging revealed four independent issues:
(1) 0 pairs inserted per refill cycle causing an infinite GPU-burning loop;
(2) web container unable to serve updated code because source was baked into the image;
(3) stale session cookies causing FK constraint errors on annotation submit;
(4) admin panel showing total actionable annotations instead of "new since last training".

### Changes

#### 1. `inference/hypothesis_generator.py` — Two-pass beam + temperature sampling

Diverse beam search (`num_beam_groups`) produced identical outputs for all audio files: high-confidence Polish medical speech collapsed all groups regardless of `diversity_penalty`. Replaced with a two-pass strategy:

- Pass 1: `num_beams=5, do_sample=False` — deterministic best hypothesis.
- Pass 2: `do_sample=True, temperature=0.8, num_beams=1, num_return_sequences=n-1` — stochastic alternatives.
- Pairing: beam result vs. the sample with the highest `difflib.SequenceMatcher` edit distance from it.

Result: ~45–48 insertable pairs per 50-file batch (was 0).

#### 2. `scripts/coordinator.py` — Generate step removed from training pipeline; auto-refill uses latest RM checkpoint

`_pipeline()` reduced from 6 to 5 steps — pair generation removed. Auto-refill is a separate DB-flag-triggered operation that runs independently. Coordinator now calls `_find_latest_rm_checkpoint()` and passes `--rm_checkpoint <path>` to `run_generate_pairs.py` so each refill batch is scored by the freshest reward model.

#### 3. `docker-compose.yml` — Live source volume mounts for both containers

Added `.:/app` bind mount to both `web` and `coordinator` services. Code changes now take effect with `docker compose restart <service>` — no image rebuild needed.

#### 4. `web/app.py` + `storage/feedback_store.py` — Stale session guard

Added `session_exists(session_id)` to `FeedbackStore`. The `/annotate`, `/submit`, and `/skip` endpoints now redirect to `/setup` (or return 401) if the session cookie references a row that no longer exists. Prevents FK constraint errors after container restarts.

#### 5. `web/app.py` + `web/templates/admin.html` — "New since last training" stat

Admin panel now shows actionable annotations accumulated *since the last training run* (not total). `annotations_at_last_training` is written to DB settings by the coordinator after each completed run. `admin_panel` reads it in Python and passes `new_actionable` directly to the template; Jinja2 does no arithmetic over the raw settings dict.

---

## 2026-07-24 — Hardcoded-Path Audit + Admin Console Enhancements

### Context

New-machine setup was broken in three places due to hardcoded paths left from the
Docker-first development workflow. Separately, the admin panel's "Trigger Training Now"
button only worked when the `coordinator` Docker service was running — silently doing
nothing in local development mode. Documentation was also extensively out of date
(Gemma references from the 2026-07-15 overhaul had not been propagated).

### Changes

#### 1. `default_paths.json` — Docker absolute paths replaced with repo-relative paths

Manifest CSV paths changed from `/app/_data/manifests/*.csv` (Docker container absolute)
to `_data/manifests/*.csv` (repo-relative). `paths.py::load_paths()` now resolves relative
path values against `PROJECT_ROOT` automatically, so the file works correctly regardless of
the working directory the caller was launched from.

**Before:** cloning the repo gave wrong manifest paths on any non-Docker machine.  
**After:** out-of-the-box correct on any machine where manifests have been generated.

#### 2. `scripts/coordinator.py` — Docker-only hardcodes removed

- `sys.path.insert(0, "/app")` → derived from `Path(__file__).resolve().parent.parent`
- `_PYTHON = "python3.12"` → `sys.executable` (correct on any machine)
- `_WORKDIR = "/app"` → derived project root

**After:** `python scripts/coordinator.py` works on Windows and Linux outside Docker.

#### 3. `prepare_data.py` — CommonVoice ZIP version no longer hardcoded

Added `--admed_zip PATH` and `--cv_zip PATH` CLI arguments. CommonVoice ZIP auto-detected
by `glob("cv-corpus-*.zip")` when `--cv_zip` is not supplied; the alphabetically-last
match is used (newest version). The `CV_PREFIX` is now inferred from the ZIP's top-level
directory, not hardcoded.

#### 4. `web/app.py` — Inline coordinator for local development

New env var `INLINE_COORDINATOR=1` activates an asyncio background task inside the
FastAPI process that mirrors `scripts/coordinator.py` behaviour. Training steps run as
subprocesses via `asyncio.create_subprocess_exec`. This makes the admin panel's
"Trigger Training Now" button functional without a separate coordinator process.

The external coordinator service remains the recommended path for production Docker
deployments (separation of concerns, GPU-process isolation). Inline mode is for local
development and single-machine setups.

#### 5. `web/app.py` + `storage/feedback_store.py` — Per-annotator annotation count

`get_active_sessions()` now joins the `annotations` table to include an
`annotation_count` key per session (non-skip annotations only). The admin panel active
sessions table shows this count as a new column.

#### 6. Documentation — stale Gemma references removed across all files

All documentation files that still referenced Gemma 9B (removed 2026-07-15) have been
updated:

| File | What changed |
|------|-------------|
| `README.md` | Intro, score step comment, reward config section, VRAM table, CLI table |
| `quick_start.md` | Intro loop, Step 0 (wrong JSON keys + `prepare_data.py` added), Step 4 (Gemma → AudioTextRewardModel), all `py -3.12` → `py` |
| `DESIGN.md` | Pipeline flowchart (Gemma 9B node → AudioTextRewardModel; PPO node → DPO) |
| `docs/ARCHITECTURE.md` | Flowchart, module map, data flow, key design decisions |
| `docs/TRAINING_GUIDE.md` | Step 3, Step 5, Round 1+ section, troubleshooting, config reference |
| `agent_onboarding_manual.md` | Reward model entry |

---

## 2026-07-15 — Reward Model Overhaul + DPO Pipeline Redesign

### Context

Design review session identified two fundamental weaknesses in the original architecture:

1. **Gemma reward model never processed audio.**
   The RLAIF scoring pipeline (`reward_model/rlaif_annotator.py`, `reward_model/gemma_reward.py`)
   passed only text hypotheses to Gemma 9B. Without audio, Gemma could only judge linguistic
   fluency and grammatical plausibility of Polish — not transcription accuracy. This meant the
   reward signal optimised for "sounds like natural Polish" rather than "matches what was said."
   For domain-specific speech (medical vocabulary, names, acronyms), fluency and accuracy can
   be inversely correlated.

2. **DPO degenerated to supervised fine-tuning.**
   Both hypotheses in each training pair (`hyp_a`, `hyp_b`) were generated by the same Whisper
   checkpoint using fixed decoding strategies (beam_5 vs temp_0.8). The reference model
   `π_ref` was the same checkpoint. The log-ratio signal in the DPO loss was therefore small
   from the start, and training converged to approximately cross-entropy on the winning
   hypothesis — i.e. supervised fine-tuning, not preference optimisation.
   See `docs/DPO_PRIMER.md §8` for the formal argument.

### Decisions made

#### 1. Drop Gemma entirely

The entire Gemma subsystem is removed — RLAIF annotator, reward scorer, Gemma fine-tuning,
and associated prompts. Rationale:

- Gemma (9B, ~6.5 GB VRAM) could not access audio and therefore could not produce a valid
  ASR reward signal.
- The cold-start problem (needing ~100 labelled pairs before any reward model works) is solved
  trivially: ~100 human pairwise annotations take approximately 20–30 minutes for one annotator
  and produce higher-quality signal than any number of text-only Gemma labels.
- Removing Gemma eliminates the primary VRAM bottleneck and the CONTRACT.md invariant
  "never load Gemma and Whisper training simultaneously."

#### 2. New reward model: Whisper encoder + Polish BERT + MLP

Replace Gemma with a small, audio-grounded reward model:

```
audio ──► Whisper encoder (frozen) ──► mean-pool ──► audio_vec  (1024-dim)
                                                           │
hypothesis ──► dkleczek/bert-base-polish-uncased-v1 ──► [CLS] ──► text_vec (768-dim)
                                                           │
                              MLP( cat([audio_vec, text_vec]) ) ──► scalar reward
```

Trained with **Bradley-Terry pairwise loss** on human preference annotations:

$$L = -\log \sigma(r_w - r_l)$$

- Whisper encoder is **frozen** (preserves acoustic features; avoids circular dependency
  with the policy model).
- BERT is **frozen** initially; only the MLP head is trained. With ~100–300 pairs this
  prevents overfitting.
- BERT model: `dkleczek/bert-base-polish-uncased-v1` (~110M params, Polish-specific).
- Total reward model VRAM at inference: ~2–3 GB — fits alongside Whisper training on RTX 3090.

#### 3. Drop REINFORCE; retain DPO with iterative synthetic pair generation

REINFORCE is removed from the active training pipeline. DPO is retained as the primary
fine-tuning algorithm, but upgraded with iterative synthetic pair generation:

```
Round 0 (bootstrap):
  - Collect ~100 human pairwise annotations (one annotator session, ~25 min)
  - Train reward model on those annotations

Round N ≥ 1 (iterative):
  - For each audio clip: sample N candidates from current Whisper policy
  - Score all candidates with reward model
  - y_w = highest-scored, y_l = lowest-scored
  - Discard pairs where |r_w - r_l| < τ  (near-ties, weak signal)
  - Run DPO on the resulting (audio, y_w, y_l) triples
  - Evaluate WER, save checkpoint, repeat
```

This is the **Rejection Sampling Optimisation (RSO)** pattern (Liu et al., 2023,
arXiv:2309.06657). Pairs are regenerated from the updated policy each round, avoiding
the distribution shift problem of the original static beam_5 vs temp_0.8 approach.

### Files removed

| File | Reason |
|------|--------|
| `reward_model/rlaif_annotator.py` | Gemma RLAIF — replaced by human cold-start |
| `reward_model/gemma_reward.py` | Gemma inference — replaced by AudioTextRewardModel |
| `reward_model/train_reward_model.py` | Gemma PEFT training — replaced by new RM trainer |
| `reward_model/prompts/rlaif_pairwise_pl.txt` | Gemma prompt — no longer needed |
| `reward_model/prompts/reward_scoring_pl.txt` | Gemma prompt — no longer needed |

### Files added / modified

| File | Change |
|------|--------|
| `reward_model/audio_text_reward.py` | New: AudioTextRewardModel (Whisper enc + BERT + MLP) |
| `reward_model/train_reward_model.py` | New: Bradley-Terry trainer for AudioTextRewardModel |
| `reward_model/config.py` | Updated: remove Gemma fields, add BERT + MLP config |
| `config/reward_model_config.json` | Updated: new config fields |
| `run_train_reward.py` | Updated: trains new RM instead of Gemma |
| `run_score_rewards.py` | Updated: uses AudioTextRewardModel instead of Gemma |
| `run_generate_pairs.py` | Updated: adds `--n_candidates` flag for N-best sampling |
| `rlhf/trainer.py` | Updated: remove REINFORCE; iterative pair generation for DPO |
| `CONTRACT.md` | Updated: remove Gemma VRAM invariant; add RM invariants |
| `DESIGN.md` | Updated: reward model section |
| `agent_onboarding_manual.md` | Updated: implementation status |
| `default_paths.json` | Updated: remove `gemma_model_id` |
| `paths.py` | Updated: remove `gemma_model_id` field |

### Implementation order

Changes are applied in the following sequence to allow rollback at each checkpoint:

1. **Checkpoint 1** — Remove Gemma files and references; verify project still imports cleanly -- COMPLETE
2. **Checkpoint 2** — New `AudioTextRewardModel` class + config; verify forward pass -- COMPLETE
3. **Checkpoint 3** — Bradley-Terry trainer + `run_train_reward.py`; verify training loop -- COMPLETE
4. **Checkpoint 4** — `run_score_rewards.py` updated to use new RM -- COMPLETE
5. **Checkpoint 5** — N-best pair generation in `run_generate_pairs.py` -- COMPLETE
6. **Checkpoint 6** — DPO trainer updated (remove REINFORCE, add iterative pair regen) -- COMPLETE
7. **Checkpoint 7** — Docs updated (CONTRACT.md, DESIGN.md, agent_onboarding_manual.md) -- COMPLETE

---
## 2026-07-15 (post-session) -- Annotation UX, Diverse Beam Search, WER Signal

### Changes

#### Annotation UI (web/templates/annotate.html, web/static/style.css)
- Hypothesis cards are now clickable to select preference (onclick handler + CSS cursor)
- Reference transcript hidden by default; click card title to reveal/hide (prevents anchoring bias)
- Live WER badges appear on each hypothesis card as annotator types a correction; green = better, red = worse; computed in-browser via word-level Levenshtein (no server round-trip)

#### WER auto-preference from correction (web/app.py)
- At submit time, if `preference == 'equal'` and `correction_text` is non-empty:
  - Compute WER(correction, hyp_a) and WER(correction, hyp_b) using `jiwer`
  - If |WER_a - WER_b| > 0.05, override preference to 'a' or 'b'
- Turns every informative correction into both an SFT example and a usable preference pair
- Explicit A/B clicks are never overridden

#### Diverse N-best beam search (inference/hypothesis_generator.py, inference/config.py)
- `generate_candidates()` now uses **diverse beam search** (`num_beam_groups=n_candidates`, `diversity_penalty=0.5`) instead of N separate temperature-sampling passes
- One forward pass per batch (faster); each beam group penalised for reusing tokens from earlier groups, producing genuinely different hypotheses rather than near-identical outputs
- `InferenceConfig.nbest_diversity_penalty = 0.5` (tunable in config/inference_config.json)

#### Annotation quality guard (storage/feedback_store.py, reward_model/train_reward_model.py, run_train_reward.py)
- Added `count_actionable_annotations()` -- counts only `preference IN ('a', 'b')`
- Added `annotation_summary()` -- returns `{total, actionable, equal, skip}` in one query
- `RewardModelTrainer.train()` guard now checks *actionable* count, not total (previously included 'equal' in the threshold, causing false-positive readiness checks)
- `run_train_reward.py` prints full breakdown before attempting training; exits early with a clear message when not enough clear-preference annotations exist
- `run_sft.py` also prints the breakdown and uses consistent logging (no more `print()`)

#### Naming cleanup
- `HITL` acronym removed from all code and docs; project is now consistently **RLHF-ASR**
- Docker compose project name changed from `hitl-asr` to `rlhf-asr`
- `HITL_DATA_DIR` env var renamed to `RLHF_DATA_DIR` in `web/app.py`

---