# RLHF ASR Training System

Fine-tune **OpenAI Whisper Large-v3** on Polish speech using pairwise human preference feedback. Human annotators compare transcription hypotheses in a browser UI; an **AudioTextRewardModel** (independent frozen Medium encoder + Polish BERT + MLP) turns those preferences into a reward signal; **DPO** fine-tunes the Whisper policy via LoRA.

VoiceGrader deployment uses the [joint README](../README.md) and root Compose
configuration. Defaults now use a fresh `_outputs/large-v3/` policy state.
Medium adapters cannot be reused; actual Large-v3 GPU readiness is unverified.
Upstream deployment examples elsewhere below are references, not the active
VoiceGrader service configuration.

---

> **TL;DR — run the whole system:**
> ```bash
> docker compose -f ..\compose.yaml up --build rlhf training
> ```
> Annotation UI → http://localhost:8080 · Admin panel → http://localhost:8080/admin  
> The coordinator triggers guarded training automatically (or from Admin), while
> its isolated diagnostics thread enables Admin-controlled performance captures.<br>
> First time? Complete [steps 1–2 below](#1-place-raw-data-archives) before running this.

---

> **SIGNAL ROOM — open downloaded diagnostics:**
>
> Place diagnostic ZIP files in `_stress_test_data/`, then run from the repository root:
> ```powershell
> .\.venv\Scripts\python.exe -m stress_analytics.app
> ```
> Open **http://127.0.0.1:8090** in your browser. See
> [Offline diagnostics explorer](#offline-diagnostics-explorer) for storage details.

---

> **RLHF CONVERGENCE VERIFICATION — run on an L40:**
> ```bash
> python3 -m rlhf_loop_tests.run_experiment --config rlhf_loop_tests/configs/convergence.json --gpu 0
> ```
> This runs isolated SFT, perfect-feedback RLHF, and 5%-error RLHF arms and
> writes `report.md`, `convergence.csv`, and `convergence.svg` under
> `_outputs/rlhf_loop_tests/convergence-v1/`. Add `--resume` after an interruption.

---

## Quick Start (Docker)

Docker is the primary way to run this system. The annotation web UI needs no GPU and runs in a lightweight container; all ML pipeline steps run in a separate GPU container.

### Prerequisites

- [Docker Desktop](https://docs.docker.com/get-docker/) (or Docker Engine + Compose plugin)
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) — required for GPU services (`generate`, `score`, `sft`, `dpo`, `evaluate`, `coordinator`). The `web` service works without it.

Diagnostics runs in an isolated background thread inside the coordinator. It
needs no additional service, Docker socket, host namespace mounts, or Kubernetes
API permissions. OS metrics reflect the coordinator container or pod's
runtime-visible `/proc`; use infrastructure monitoring separately for node-wide
and per-container telemetry.

### Offline diagnostics explorer

Downloaded sanitized diagnostics bundles can be explored without server access
using the localhost-only **Signal Room** dashboard. Place ZIP artifacts under
`_stress_test_data/`, then run:

```powershell
.\.venv\Scripts\python.exe -m stress_analytics.app
```

Open **http://127.0.0.1:8090**. The dashboard reads input archives without
extracting or modifying them and stores its derived SQLite index under
`_outputs/stress_analytics/`.

### 1. Place raw data archives

Put the source ZIP archives in `_data/` before first use:

```
_data/archive.zip                       ← ADMEDVOICE
```

### 2. Prepare manifests (once)

```
docker compose run --rm prepare
```

Reads the ZIPs, writes manifest CSVs into `_data/manifests/`, and updates `default_paths.json` automatically. No GPU needed. Re-run if you add new data.

### 3. Start the web UI and coordinator

```bash
docker compose up web coordinator
```

- Annotators open **http://localhost:8080** in any browser.
- The coordinator polls the DB in the background and automatically triggers training when enough annotations accumulate (configurable threshold — see [Admin panel](#admin-panel)).
- Administrators control runtime/GPU/application diagnostics and bundle downloads at **http://localhost:8080/admin/diagnostics**.
- Annotation continues uninterrupted during training (SQLite WAL + separate containers).

To run the web UI without the coordinator (annotation only, no auto-training):
```
docker compose up web
```

### 4. Run pipeline steps manually (GPU)

Each command runs in a fresh container and exits when done:

```
docker compose run --rm generate    # transcribe audio → pending pairs
docker compose run --rm score       # AudioTextRewardModel scoring
docker compose run --rm sft         # SFT fine-tuning on corrections
docker compose run --rm dpo         # DPO preference fine-tuning
docker compose run --rm evaluate    # WER/CER report
```

Override defaults inline, e.g.:
```
docker compose run --rm generate python3 run_generate_pairs.py --n 500 --datasets admedvoice
docker compose run --rm dpo python3 run_rlhf.py --resume _outputs/checkpoints/my_ckpt
```

---

## Web Annotation UI

The web UI (`web/app.py`, port 8080) is the main annotation interface.

### Annotator flow

1. Open **http://localhost:8080** (or the host's IP if running remotely).
2. Enter your participant ID on the setup page and click **Start Annotating**.
3. Each trial shows an audio player, the reference transcript, and two competing hypotheses side by side.
4. Pick **A**, **Equal**, or **B** — optionally rate confidence with the slider and type a correction — then click **Submit**.
5. The next pair loads automatically. Close the tab at any time — all submitted annotations are already saved.

### Annotation setup

Showcase functionality is removed. `/` redirects to `/setup`, where annotators
enter their own participant ID. `GET /setup` is read-only and remains suitable
for probes without cookies. Sessions are created only by `POST /setup`.

### Admin panel

The admin panel is at **http://localhost:8080/admin**.

Admin login is always required. The public initial password is `change-me` when
no configuration exists. Change it before exposing the service beyond localhost,
using **Change Password** in the admin navigation, or the desktop utility:

```powershell
py -3.12 set_admin_password.py
```

For a headless terminal, use `py -3.12 -m utils.admin_password`.
Both tools write a salted PBKDF2-HMAC-SHA256 hash to `_outputs/admin_password.json`;
the plaintext password is never stored. This runtime file is independent of the
model lineage and must not be committed. Changes apply without restarting web
and invalidate existing admin sessions. For a custom persisted file location,
set `ADMIN_PASSWORD_CFG` consistently in web and password tools.

**What the admin panel shows and controls:**

| Area | What you can do |
|------|----------------|
| Live Stats | Active annotators, pending pairs, annotation count, training runs |
| Training | Trigger or cancel a run, see current status, and tune guarded SFT/DPO convergence settings for the next run |
| Model Evaluation | Create and switch evaluation sets, unlock revisions, reroll faulty clips, delete drafts, and archive evaluated sets |
| Models & Publishing | Compare checkpoint WER/CER, inspect outputs, perform audited rollback, and publish the deployed healthy model as `latest` |
| Reward Audit | Score searched or random historical pairs with a selected reward checkpoint and compare its ranking with human choices |
| Diagnostics | Start or stop server performance captures and download sanitized or sensitive forensic bundles |
| Settings | Max concurrent annotators, auto-training threshold (annotations since last run) |

---

## Configuration

### Web service environment variables

Set in `docker-compose.yml` under the `web` service, or as host environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ADMIN_PASSWORD_CFG` | `_outputs/admin_password.json` | Persisted PBKDF2 admin password configuration; keep outside Git. |
| `PAIR_RESERVATION_SECONDS` | `300` | How long a pair is reserved for one annotator before being released to others. |

The endurance deployment starts `web` and `coordinator`. The coordinator's
diagnostics thread writes under `_outputs/diagnostics` and the configured
application log directory. See
[stress_tests/DEPLOYMENT_OWNER_GUIDE.md](stress_tests/DEPLOYMENT_OWNER_GUIDE.md)
for the Admin-only capture and download workflow.

### Coordinator environment variables

Set in `docker-compose.yml` under the `coordinator` service:

| Variable | Default | Description |
|----------|---------|-------------|
| `COORDINATOR_POLL_INTERVAL` | `30` | Seconds between DB polls for training triggers. |
| `EVAL_MAX_SAMPLES` | `40` | Max test utterances per evaluation split (reproducible random subset; `0` = full set). |

### `config/inference_config.json` — Hypothesis generation

| Key | Default | Description |
|-----|---------|-------------|
| `batch_size` | `8` | Audio clips processed per Whisper forward pass. |
| `target_batch_size` | `50` | Pairs to generate per `run_generate_pairs.py` call. |
| `nbest_temperature` | `0.8` | Temperature for the sampling pass of two-pass N-best generation. Higher = more diverse alternatives. |
| `datasets` | `["admedvoice"]` | Datasets to draw audio from. |
| `splits` | `["train"]` | CSV splits to use. |

### `config/rlhf_config.json` — RLHF fine-tuning

| Key | Default | Description |
|-----|---------|-------------|
| `algorithm` | `dpo` | Training algorithm. Only DPO is supported. |
| `lora_r` | `16` | LoRA rank. |
| `lora_alpha` | `32` | LoRA scaling factor. |
| `learning_rate` | `1e-5` | |
| `batch_size` | `4` | |
| `gradient_accumulation_steps` | `8` | Effective batch = `batch_size × grad_accum`. |
| `num_epochs` | `2` | DPO passes per round; coordinator SFT remains three epochs. |
| `dpo_beta` | `0.1` | DPO temperature (DPO only). |
| `synthetic_pool_size` | `500` | Random audio clips used for current-round synthetic DPO generation. |
| `synthetic_n_candidates` | `6` | Whisper candidates ranked per synthetic audio clip. |
| `max_synthetic_to_human_ratio` | `1.0` | Maximum current-round synthetic pairs per current-policy human pair. |
| `min_reward_scores` | `32` | Minimum combined DPO pairs: one default effective batch. |
| `resume_from_checkpoint` | `""` | Path to a LoRA checkpoint to resume from. |

### `config/reward_model_config.json` — AudioTextRewardModel

| Key | Default | Description |
|-----|---------|-------------|
| `whisper_model_id` | `openai/whisper-medium` | Whisper checkpoint used for the frozen audio encoder. |
| `bert_model_id` | `dkleczek/bert-base-polish-uncased-v1` | Polish BERT for hypothesis text encoding. |
| `min_annotations_for_training` | `50` | Actionable (a/b) annotations required before reward model training starts. |
| `learning_rate` | `2e-4` | |
| `num_epochs` | `5` | |
| `reward_model_version` | `1` | Bump before each new training run to version checkpoints. |

---

## Project Structure

```
├── paths.py                     # centralised path config
├── default_paths.json           # machine-specific paths (edit once)
├── prepare_data.py              # one-time data prep: ZIP → manifest CSVs
├── config/
│   ├── inference_config.json    # hypothesis generation settings
│   ├── reward_model_config.json # AudioTextRewardModel settings
│   └── rlhf_config.json         # RLHF training settings
├── storage/                     # SQLite layer (feedback_store, schema, merger)
├── utils/                       # logging, audio loading/validation
├── inference/                   # Whisper hypothesis generation
├── diagnostics/                 # runtime/GPU/application telemetry and bundles
├── web/                         # FastAPI web annotation UI (Docker / browser)
│   ├── app.py
│   ├── templates/               # annotation and Admin pages
│   └── static/                  # style.css
├── scripts/
│   └── coordinator.py           # training orchestrator + isolated diagnostics
├── reward_model/                # AudioTextRewardModel (Whisper enc + Polish BERT + MLP)
├── rlhf/                        # SFT/DPO training and preference data
├── evaluation/                  # WER/CER evaluator + report generator
├── stress_tests/                # k6 load/endurance tooling and operator guides
├── rlhf_loop_tests/             # isolated RLHF-vs-SFT convergence experiment
├── run_*.py                     # entry points (one per pipeline stage)
├── Dockerfile                   # GPU image (ML pipeline steps)
├── Dockerfile.web               # lightweight web-only image (no GPU)
├── docker-compose.yml           # all services declared here
├── DESIGN.md                    # historical rationale; not current specification
├── CONTRACT.md                  # design invariants and "Do Not" rules
└── docs/                        # detailed guides
    ├── ARCHITECTURE.md          # current module and data flow
    ├── DATA_SCHEMA.md
    ├── MULTI_MACHINE.md
    └── TRAINING_GUIDE.md
```

---

## Outputs

All outputs land in `_outputs/` (auto-created):

| Path | Contents |
|------|---------|
| `_outputs/feedback.sqlite` | SQLite database |
| `_outputs/logs/` | Timestamped log files |
| `_outputs/checkpoints/` | LoRA fine-tuned Whisper checkpoints |
| `_outputs/latest/` | Explicitly published LoRA model and checksummed `publication.json` provenance |
| `_outputs/reward_model/` | AudioTextRewardModel checkpoints (`mlp_head.pt`) |
| `_outputs/diagnostics/runs/` | Host/GPU/container metrics, lifecycle events, issue timelines, manifests, and downloadable bundles |

Diagnostics retention is manual. Sanitized bundles exclude the database, raw
logs, transcripts, participant IDs, audio, checkpoints, password configuration,
tokens, and environment dumps. Sensitive bundles additionally contain a
consistent SQLite snapshot and selected application logs.

While a capture is running, the coordinator diagnostics thread atomically refreshes a
sanitized partial ZIP every 60 seconds and Admin can request a fresh sanitized
or sensitive partial bundle at any time. Metric/event files are flushed after
each sample or record, and incomplete trailing records are excluded from ZIP
snapshots. If the coordinator process or container restarts, diagnostics resumes the
latest running capture in append mode and records the restart. Superseded
orphan captures are marked `interrupted` and remain bundleable. These guarantees
require the host `_outputs/` storage to remain available; they cannot survive
host disk loss or deletion of the run directory.

---

## Testing

Install the test runner after the project dependencies, then run the complete
CPU-safe repository suite before finishing a change:

```powershell
py -3.12 -m pip install pytest pytest-timeout
py -3.12 -m pytest -q
```

The current suite contains 249 tests and does not run GPU training or download
models. Use a focused test while iterating, for example:

```powershell
py -3.12 -m pytest -q tests/test_diagnostics.py
py -3.12 -m pytest -q tests/test_coordinator.py
py -3.12 -m pytest -q tests/test_policy_flow.py
```

Validate deployment and k6 syntax separately:

```powershell
docker compose config --quiet
docker run --rm -v "${PWD}/stress_tests/load:/scripts:ro" grafana/k6:latest inspect /scripts/sim_burst.js
docker run --rm -v "${PWD}/stress_tests/load:/scripts:ro" grafana/k6:latest inspect /scripts/loadtest.js
```

The k6 load/endurance tests are state-changing: they create sessions and real
annotations and may trigger training. Follow [stress_tests/README.md](stress_tests/README.md)
and the role-specific operator guides before running them. A real diagnostics
smoke test additionally requires the Linux/CUDA deployment host: start a short
capture in Admin, exercise the web workload, stop it, and inspect/download both
bundle classes.

The isolated RLHF-vs-plain-SFT convergence experiment has its own commands and
artifacts under [rlhf_loop_tests/README.md](rlhf_loop_tests/README.md); it is
not executed by pytest. The default suite tests its orchestration and artifact
contracts without running GPU training.

---

## Hardware Budget (RTX 3090, 24 GB)

| Phase | VRAM |
|-------|------|
| Hypothesis generation (Whisper float16) | ~3.5 GB |
| Annotation web UI | ~0 GB (CPU only) |
| Reward model scoring (Whisper enc + BERT + MLP) | ~2.5 GB |
| Reward model training (MLP only, encoder frozen) | ~4 GB |
| DPO training (Whisper policy + frozen exact FP16 ref) | hardware-dependent |

---

## Multi-Machine Annotation

Each machine has its own `_outputs/feedback.sqlite`. After annotation sessions:

```
# On each annotator machine — export
py -3.12 run_merge_feedback.py --export

# On the merge machine — ingest all exports
py -3.12 run_merge_feedback.py --merge machine1_export.json machine2_export.json
```

UUID4 primary keys guarantee no collisions. See [docs/MULTI_MACHINE.md](docs/MULTI_MACHINE.md).

---

## CLI / Local Setup (without Docker)

For running directly on Windows or Linux with a local Python environment.

### Prerequisites

```
Python 3.12+   (tested on 3.12 on Windows)
CUDA GPU       (NVIDIA RTX 3090 or equivalent — 24 GB VRAM recommended)
```

Install dependencies:
```powershell
py -3.12 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
py -3.12 -m pip install -r requirements.txt
```

### Configure paths (once per machine)

Edit `default_paths.json` and fill in the paths for your data. Paths can be
**relative to the project root** (e.g. `_data/manifests/admedvoice_train.csv`)
or absolute. Run `prepare_data.py` to generate manifests automatically from ZIP archives.

```json
{
  "admedvoice_train_csv":   "_data/manifests/admedvoice_train.csv",
  "admedvoice_test_csv":    "_data/manifests/admedvoice_test.csv"
}
```

CSV files must use `;` as delimiter.

### Typical workflow

```
Step 1   Generate hypothesis pairs     py run_generate_pairs.py
Step 2   Annotate (repeat freely)      open http://localhost:8080
Step 3   Run one coordinated round     docker compose up web coordinator
         RM -> optional SFT -> synthetic pool -> DPO candidate -> WER gate
Step 4   View recorded WER history     py run_evaluate.py --report
         Repeat from Step 1
```

After the candidate passes Model Evaluation on the newest frozen evaluation set against the deployed policy, it becomes `active_policy_checkpoint`. The gate resamples paired clips and recomputes corpus WER from total word errors rather than averaging per-clip WER; its 95% CI upper bound may allow at most +1.0 percentage point regression. A rejected or uncertain candidate remains registered for diagnosis while the deployed policy is unchanged. Subsequent pair generation and refill automatically attach the active LoRA adapter; `--policy_checkpoint` is only needed to override it for a reproducible run.

Deployment does not automatically overwrite the model consumed by external
applications. In **Models & Publishing**, use **Publish as latest** to copy the
currently deployed, healthy adapter into `latest_model_dir` (by default
`_outputs/latest`). Consumers read that stable directory and may verify files
against `publication.json`. Configure the shared location once in
`default_paths.json` when it should live outside this repository.

During pre-alpha experiments, the admin Settings section exposes guarded SFT
and DPO convergence controls plus a separate training-data mix: maximum
synthetic pairs per human pair, candidate-pool multiplier, minimum reward
margin, and minimum token edits. Gradient clipping and optional epoch-level
early stopping apply to both SFT and DPO. The coordinator snapshots these
values when a run starts, so edits apply only to the next run. The dashboard
reports realized human/synthetic counts, optimizer updates, effective warmup,
loss, gradient norm, reward margin, and DPO log-ratio for the latest attempt.
and the effective values are saved with its checkpoints. Reference precision,
LoRA scope, batch geometry, and deployment gates are intentionally not editable
from the browser.

### CLI entry points

| Script | Purpose | Key flags |
|--------|---------|-----------|
| `run_generate_pairs.py` | Generate pairs from the active Whisper policy | `--n 200` `--policy_checkpoint path/` `--catalogue_only` |
| `run_generate_synthetic.py` | Generate ephemeral current-round DPO JSONL | coordinator-managed |
| `run_score_rewards.py` | Compatibility/analysis scoring; not automated DPO input | `--checkpoint path/` |
| `run_reward_audit.py` | Coordinator worker for one queued Reward Audit job | coordinator-managed |
| `run_sft.py` | SFT fine-tune on human-corrected transcripts | `--epochs 3` `--resume path/` |
| `run_train_reward.py` | Train AudioTextRewardModel (MLP head) on human preferences | needs ≥50 actionable annotations |
| `run_rlhf.py` | DPO on human + current synthetic preferences | `--synthetic_data path.jsonl` `--resume path/` |
| `run_evaluate.py` | WER/CER evaluation on any checkpoint | `--lora path/` `--report` `--markdown` |
| `run_evaluate_canary.py` | Resume beam-5 health evaluation or gate a pending policy candidate | `--checkpoint-id UUID` `--promote-pending` |
| `run_merge_feedback.py` | Multi-machine DB merge | `--export` `--merge a.json b.json` |
| `simulate_rlhf.py` | WER-oracle RLHF simulation (no human annotators) | `--rounds 50` `--error_rate 0.05` `--skip_sft` |
