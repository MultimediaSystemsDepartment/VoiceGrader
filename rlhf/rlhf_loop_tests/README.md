# RLHF Loop Experiments

This directory contains local, isolated experiments comparing plain supervised
fine-tuning with the existing RLHF loop under perfect and noisy WER-derived
feedback. It is separate from
`stress_tests`: the objective here is learning convergence, not HTTP capacity.

## Safety

Experiment state belongs under `_outputs/rlhf_loop_tests/`. Every experiment
arm receives a separate database, checkpoint directory, reward-model directory,
published-model directory, and log directory. The workspace builder rejects
paths that alias the normal deployment outputs.

Do not point `RLHF_PATHS_FILE` at the normal `default_paths.json` while running
an experiment command. Do not use production annotation data as an experiment
workspace.

## Implemented Foundation

The current milestone provides:

- process-level path isolation through `RLHF_PATHS_FILE`;
- validated smoke and convergence configurations;
- guarded SFT, clean-RLHF, and noisy-RLHF arm workspace and manifest creation;
- explicit `CUDA_VISIBLE_DEVICES` environment mapping;
- a deterministic GT oracle using the project's lexical WER normalization;
- a surrogate client for the real setup, heartbeat, pair, submit, and skip APIs;
- JSONL provenance for every surrogate decision;
- deterministic recording-session-disjoint interaction, promotion-monitor, and final-holdout partitions;
- exact per-round audio-ID schedules shared by all arms;
- reproducibly seeded temperature sampling for experiment generation;
- transactionally consistent initial-database cloning;
- identical frozen monitor and holdout fingerprints across all arms;
- real HTTP annotation rounds with isolated arm web services and parity reports.

Matched GPU training, final-holdout evaluation, and multi-seed reporting are
available through one resumable command. Lower-level stage commands remain for
diagnosis. Interrupted GPU stages fail closed and are never silently repeated.

## Run the Complete Verification

First run the small mechanical smoke configuration:

```bash
python3 -m rlhf_loop_tests.run_experiment \
  --config rlhf_loop_tests/configs/smoke.json \
  --gpu 0
```

Then run the five-seed comparison on an L40:

```bash
python3 -m rlhf_loop_tests.run_experiment \
  --config rlhf_loop_tests/configs/convergence.json \
  --gpu 0
```

After an interruption, repeat the same command with `--resume`. Inspect and
resolve an arm whose `training.json` remains in `running` state before retrying;
the harness does not guess whether a GPU update completed.

## Prepare a Smoke Workspace

From the repository root:

```powershell
py -3.12 -m rlhf_loop_tests.run_experiment `
  --config rlhf_loop_tests/configs/smoke.json `
  --prepare-only `
  --gpu 0
```

Linux:

```bash
python3 -m rlhf_loop_tests.run_experiment \
  --config rlhf_loop_tests/configs/smoke.json \
  --prepare-only \
  --gpu 0
```

Preparation refuses a non-empty experiment directory. Use `--resume` only with
the exact same configuration.

## Initialize the Common State

After preparation, populate the isolated initial catalogue, persist every
replicate's partitions, and clone all three matched arm databases:

```powershell
py -3.12 -m rlhf_loop_tests.initialize_experiment `
  --manifest _outputs/rlhf_loop_tests/smoke/manifest.json
```

Linux:

```bash
python3 -m rlhf_loop_tests.initialize_experiment \
  --manifest _outputs/rlhf_loop_tests/smoke/manifest.json
```

This command invokes `run_generate_pairs.py --catalogue_only` with the initial
state's `RLHF_PATHS_FILE` and CPU visibility. It never ingests the live feedback
database. Initialization fails if the configured manifests do not provide
enough GT-bearing audio for all three partitions.

The lower-level `plan_partitions` and `materialize_arms` commands remain
available for diagnosis, but normal setup should use `initialize_experiment`.

Each round file can be consumed without random refill:

```bash
python3 run_generate_pairs.py \
  --audio_ids_file _outputs/rlhf_loop_tests/smoke/replicate-01/round-01-audio-ids.json
```

The command must run with the selected arm's `RLHF_PATHS_FILE`. Scheduled mode
fails if any requested ID is missing, held out, excluded, or already paired for
that policy. It also fails if candidate collapse prevents creation of one pair
per scheduled recording.

## Run a Matched Annotation Round

After initialization, generate from each arm on the configured GPU and submit
the scheduled GT annotations through the real FastAPI endpoints:

```powershell
py -3.12 -m rlhf_loop_tests.run_annotation_round `
  --manifest _outputs/rlhf_loop_tests/smoke/manifest.json `
  --replicate 1 `
  --round 1
```

The arms run sequentially so they do not compete for VRAM. Generation receives
the same replicate/round seed in all arms; ordinary queue refill remains
stochastic because it does not pass `--seed`. Each CPU-only web process starts
with `INLINE_COORDINATOR=0`, is checked through `/setup`, and is stopped after
the surrogate reaches the exact round budget.

Per-arm `state.json` and append-only `annotations.jsonl` files allow an
interrupted annotation stage to continue without regenerating completed pairs.
The round is committed to the manifest only after all arms match the scheduled
audio order and submitted GT corrections. When the arms use the same policy,
their seeded hypotheses and oracle preferences must match as well.

## Train a Matched Round

After the annotation round passes parity validation:

```powershell
py -3.12 -m rlhf_loop_tests.run_training_round `
  --manifest _outputs/rlhf_loop_tests/smoke/manifest.json `
  --replicate 1 `
  --round 1
```

The SFT arm runs ordinary `run_sft.py` with the same SFT tuning snapshot used by
the canonical pipeline, then explicitly activates that arm-local adapter. The
RLHF arm invokes `scripts/coordinator.py --once`, preserving reward training,
SFT, synthetic generation, DPO, and the frozen-monitor promotion gate. A
guarded candidate rejection is recorded as a valid RLHF outcome rather than a
harness failure.

All arms run sequentially under their own `RLHF_PATHS_FILE` and the configured
`CUDA_VISIBLE_DEVICES`. Training state is written before subprocess startup;
an interrupted `running` stage fails closed and must be inspected rather than
silently repeated.

The smoke configuration uses 64 annotations because the unchanged reward model
requires 50 actionable A/B preferences. Ties can reduce the actionable count;
the convergence configuration's 100-annotation rounds provide more margin.

## Evaluate the Final Holdout

After both training arms complete:

```powershell
py -3.12 -m rlhf_loop_tests.evaluate_round `
  --manifest _outputs/rlhf_loop_tests/smoke/manifest.json `
  --replicate 1 `
  --round 1
```

This reuses deterministic beam-5 canary backfill with explicit final-holdout
and checkpoint IDs. It evaluates the common initial policy and each arm's
active policy without changing the promotion gate. The round report contains
corpus WER/CER, improvement from initial, and a clip-resampled paired
RLHF-minus-SFT corpus-WER interval.

After every configured replicate and round has an evaluation:

```powershell
py -3.12 -m rlhf_loop_tests.build_report `
  --manifest _outputs/rlhf_loop_tests/convergence-v1/manifest.json
```

`report.json` and `report.md` retain each seed's learning curve, final metrics,
normalized WER-versus-annotation AUC, round-level paired comparisons, and the
paired across-seed final WER interval. Runs with fewer than five seeds are
marked non-reportable even though their mechanics and metrics remain useful.

## Surrogate Annotator

When an isolated arm web process has a real audio-backed queue:

```bash
python3 -m rlhf_loop_tests.surrogate_annotator \
  --base-url http://localhost:8180 \
  --participant-id surrogate-gt-r01-sft \
  --budget 32 \
  --tie-threshold 0.05 \
  --output _outputs/rlhf_loop_tests/smoke/replicate-01/sft/round-01/annotations.jsonl
```

The oracle compares each hypothesis with the API-provided ground truth. It
chooses A or B by lower normalized WER, chooses `equal` within the configured
gap, and submits the GT transcript as correction text. Missing GT fails by
default; `--missing-gt skip` explicitly records that operating choice through
the skipped database pair.

## GPU Selection

`--gpu 1` will make physical GPU 1 visible to experiment subprocesses as their
logical `cuda:0`. Comma-separated visibility such as `--gpu 1,2` is accepted
for future multi-GPU stages. `--gpu cpu` supports orchestration tests only; the
real Whisper training pipeline requires CUDA.

GPU selection is applied before ML subprocess startup. Application configs
continue to use `device="cuda"`; model code does not hard-code physical GPU
indices.

## Artifacts

```text
_outputs/rlhf_loop_tests/<experiment-id>/
├── manifest.json
├── initial/
│   ├── paths.json
│   └── outputs/feedback.sqlite
├── replicate-01/
│   ├── partitions.json
│   ├── round-01-audio-ids.json
│   ├── round-01-parity.json
│   ├── round-01-holdout.json
│   ├── sft/
│   │   ├── paths.json
│   │   ├── outputs/feedback.sqlite
│   │   └── round-01/
│   │       ├── state.json
│   │       ├── training.json
│   │       ├── annotations.jsonl
│   │       └── web.log
│   └── rlhf/
│       ├── paths.json
│       ├── outputs/feedback.sqlite
│       └── round-01/
│           ├── state.json
│           ├── training.json
│           ├── annotations.jsonl
│           └── web.log
├── report.json
└── report.md
```

The manifest records the exact configuration, Git commit, Python runtime,
replicate seeds, arm ports, paths, partition artifacts, monitor IDs, and stable
partition fingerprints. Later milestones will append training stages,
checkpoints, metrics, and resume state.

See [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) for the comparison protocol.
