# Agent Guide

Use this file as the canonical route into the repository. Do not read every
design document before making a local change.

## Start Here

1. Read `CONTRACT.md` for invariants relevant to the task.
2. Open the owning file from the task router below.
3. Read one neighboring test or caller, form a local hypothesis, edit, and run
   the narrowest relevant test.
4. Run `py -3.12 -m pytest -q` before finishing.

When prose and runtime disagree, current code plus tests win. Update the active
guide that was wrong; keep dated `docs/MAINTENANCE_LOG.md` entries historical.

## Task Router

| Concern | Owning code | First test/check |
|---|---|---|
| Pair generation / Polish decoding | `inference/hypothesis_generator.py`, `inference/decoding_strategies.py` | `tests/test_policy_flow.py` |
| Active Whisper policy / LoRA handoff | `utils/policy_checkpoint.py`, `run_generate_pairs.py`, `run_rlhf.py` | `tests/test_policy_checkpoint.py`, `tests/test_policy_flow.py` |
| Queue refill / training orchestration | `scripts/coordinator.py`, `web/app.py` | `tests/test_coordinator.py` |
| DPO policy and frozen reference | `rlhf/trainer.py`, `rlhf/reference_model.py` | `tests/test_policy_flow.py`, `tests/test_reward_buffer.py` |
| SFT corrections | `run_sft.py`, `rlhf/sft_trainer.py` | full pytest suite |
| Reward model | `reward_model/audio_text_reward.py`, `reward_model/train_reward_model.py` | relevant config/storage tests |
| SQLite behavior / generic catalogue | `storage/feedback_store.py`, `storage/schema.py` | `tests/test_catalogue.py`, `tests/test_storage.py`, `tests/test_storage_advanced.py` |
| Browser UI / Polish text | `web/templates/`, `web/static/i18n.js`, `web/app.py` | compile all Jinja templates + full pytest |
| Data paths / manifests | `paths.py`, `prepare_data.py`, `default_paths.json` | `tests/test_config.py` |
| Generic recorded-audio import / corpus activation | `run_import_corpus.py`, `storage/feedback_store.py` | `tests/test_catalogue.py` |
| Browser corpus management / playback | `web/app.py`, `web/templates/corpora.html`, `utils/audio.py` | `tests/test_web_annotation.py`, `tests/test_audio.py` |
| Admin authentication / password changes | `web/app.py`, `utils/admin_password.py`, `set_admin_password.py` | `tests/test_web_auth.py` |
| Evaluation | `evaluation/evaluator.py`, `evaluation/report.py` | full pytest suite |
| WER Monitor health / rollback | `evaluation/canary.py`, `evaluation/model_health.py`, `utils/policy_checkpoint.py`, `web/templates/model_health.html` | `tests/test_model_health.py`, `tests/test_model_health_metrics.py`, `tests/test_policy_checkpoint.py` |

## Runtime Loop

1. Refill generates beam + sampled candidates from the active Whisper LoRA
   policy and optionally ranks them with the latest reward model.
2. Annotators compare pairs in the FastAPI browser UI.
3. The coordinator trains an immutable reward model, optionally runs SFT,
   generates ephemeral synthetic DPO data, then runs DPO on human plus capped
   current-round synthetic preferences with before/after WER evaluation.
4. DPO registers a pending candidate without changing `active_policy_checkpoint`.
5. The coordinator compares candidate and deployed policy on the frozen WER
   Monitor; only a passing paired WER gate activates it, retires stale pending
   pairs, and queues refill.

The reward-model checkpoint ranks candidates. It is not the Whisper generation
policy. The DPO frozen reference must equal the policy at the start of the DPO
round in checkpoint, precision, and deterministic model mode.

## Non-Negotiables

- Every Whisper decode call uses `language="pl", task="transcribe"`.
- Every SFT/DPO label tokenizer sets Polish/transcribe prefix tokens.
- Use `utils/audio.py::load_audio()` for audio decoding.
- Use UUID4 `.new()` factories for persisted records.
- DB access belongs in `storage/feedback_store.py`; schema changes need a migration.
- The web image stays CPU-only and never launches ML subprocesses unless
  `INLINE_COORDINATOR=1` is explicitly used for local development.
- `_data/` and `_outputs/` are machine-local and gitignored.
- WER Monitor audio is excluded from every learning path; unlocking creates an
   editable revision, and reroll excludes the faulty audio instead of mutating
   an evaluated monitor.
- Policy changes and rollback go through the audited activation helper.
- Do not restore PPO, REINFORCE, diverse-beam generation, or the removed Tkinter
  annotation path as if they were active features.

## Commands

```powershell
py -3.12 -m pytest -q
py -3.12 -m pytest -q tests/test_coordinator.py
py -3.12 -m pytest -q tests/test_policy_flow.py
py -3.12 run_evaluate_canary.py --if-ready
docker compose -f ..\compose.yaml up rlhf training
```

VoiceGrader copies source into images; Python/template changes require rebuilding
the affected image. Use the root Compose project rather than the imported
upstream deployment files. Do not run GPU training as a validation step
unless the task explicitly requires it and a CUDA environment is available.

## Documentation Roles

- `README.md`: operator setup and commands.
- `CONTRACT.md`: current invariants and post-mortem rules.
- `docs/ARCHITECTURE.md`: current module/data flow.
- `docs/TRAINING_GUIDE.md`: manual training workflow.
- `docs/MAINTENANCE_LOG.md`: dated history, not current instructions.
- `DESIGN.md`: concise original rationale; not a current specification.
- `quick_start.md`, `agent_onboarding_manual.md`: compatibility pointers only.