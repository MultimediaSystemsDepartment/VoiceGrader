# VoiceGrader

A framework for speech synthesis, speech recognition and fine-tuning of Whisper neural network
based on reinforcement learning from human feedback (RLHF).
Fine-tuning pipeline can: manage audio corpora,
compare Whisper transcriptions, train LoRA adapters, evaluate candidates and
deliver approved models to a file/live transcription service.
Optional TTS augmentation generates additional audio from corpus transcripts.

## Components

| Service | Purpose | Hardware |
| --- | --- | --- |
| `rlhf` | Annotation, corpus management and administration | CPU |
| `training` | Pair generation, reward training, SFT/DPO, evaluation and model publication | NVIDIA GPU |
| `asr` | File/live transcription and report-only audio quality assessment | NVIDIA GPU |
| `tts-api`, `tts-piper` | Local synthesis jobs and Piper speech generation | CPU |

The default recognition policy is Whisper Large-v3. Training is opt-in; the web
UI does not launch GPU work. Use the root [Compose configuration](compose.yaml),
not the imported component deployment files.

## Quick start

Requires Docker Compose v2 and Linux with NVIDIA Container Toolkit, or Docker
Desktop with a WSL2 Linux backend and NVIDIA GPU support. The annotation UI can
run without a GPU. Builds and initial model downloads need internet access and
substantial disk space.

Run these **PowerShell** commands from the repository root:

```powershell
# First installation only; do not overwrite an existing .env.
Copy-Item .env.example .env
docker compose --profile training config --quiet
docker compose build rlhf asr
docker compose up -d rlhf asr
```

- **Annotation:** <http://localhost:8080>
- **Administration:** <http://localhost:8080/admin>
- **File/live transcription:** <http://localhost:8000>

Sign in to administration with the initial password **`change-me`**, then use
**Change Password** before allowing access beyond localhost. The password
configuration persists across restarts. For interactive recovery:

```powershell
docker compose run --rm --no-deps rlhf python -m utils.admin_password
```

Ports are loopback-only by default and configurable in [.env.example](.env.example).
For remote access, configure HTTPS and appropriate access controls; the ASR
interface should not be treated as protected by the annotation admin password.
Browser microphone access requires a secure context (localhost or HTTPS).

## Everyday workflow

### 1. Add recordings

In **Audio Corpora**, choose **Training corpora** or **Evaluation sets**. Upload
audio files with a UTF-8, semicolon-separated CSV using plain filenames:

```csv
file_path;transcript
recording-001.wav;Example reference transcript.
recording-002.wav;Another reference transcript.
```

Each upload accepts **1-200 clips**, **0.1-30 seconds** each, up to **500 MiB**
total and a **1 MiB** manifest. Transcripts are optional for training and required
for evaluation. Browser imports accept files, not ZIP archives.

Review the preview, listen, and **Accept import**. Training corpora remain
inactive until explicitly enabled. Evaluation recordings cannot enter learning.
Validation checks decoding, duplicates and evaluation/learning overlap, but
does not verify transcript accuracy.

### 2. Prepare evaluation

Evaluation imports create an editable **WER Monitor** draft. Review and verify
its reference transcripts, freeze it, and select it as the promotion gate.
The coordinator evaluates candidates against this frozen set before activation.
Keep an independent benchmark for release-quality claims.

### 3. Generate pairs and annotate

Enable a training corpus and run the coordinator to generate transcription
pairs. Annotators enter a participant identifier, listen and compare the two
transcriptions. Configure refill and automatic-training thresholds in
administration.

On a single GPU, run training/generation separately from ASR:

```powershell
docker compose build training
docker compose stop asr
docker compose up -d rlhf training
docker compose logs -f training
```

The coordinator can refill the annotation queue and train automatically when
configured thresholds are met. A queued refill does nothing while `training`
is stopped, and an exhausted corpus needs new eligible recordings or a new
active policy.

Wait for active work to finish before switching back:

```powershell
docker compose stop training
docker compose up -d asr
```

GPU workloads are not automatically scheduled. Do not assume ASR, training and
GPU TTS fit concurrently on one GPU.

### 4. Use approved models

Passing the frozen WER gate activates a candidate. The training worker merges
and converts approved adapters, checks converted-model WER and publishes
immutable artifacts. ASR checks for updates and drains existing work before
reloading; new requests are temporarily rejected during the reload.

Without an approved publication, ASR uses stock Large-v3. Model delivery uses
a separate generated service credential, not the admin password.

## Optional: TTS augmentation

Local Piper uses the pinned Polish Gosia voice. Review the applicable model
terms before downloading. With Python 3.12 installed:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r tts\tools\requirements-model-assets.txt
$env:TTS_ASSET_ROOT = (Join-Path (Get-Location) '_data\tts')
.\.venv\Scripts\python.exe tts\tools\model_assets.py download --engines piper --accept-model-terms
.\.venv\Scripts\python.exe tts\tools\model_assets.py verify --engines piper
docker compose build tts-api tts-piper
docker compose up -d rlhf tts-piper tts-api
```

In **TTS Augmentation**, select an active corpus with transcripts, intended use
and **Local Piper / Gosia**. Estimate the work, then queue a campaign. The CPU
web dispatcher submits and imports audio even when training is stopped.

- Local texts are limited to **280 characters**; longer texts are counted and
  skipped, never truncated.
- **Assess quality** uses the running ASR service. Scores are report-only;
  listen before marking a recording reviewed. MOS does not verify its transcript.
- **Retry collection** fetches an existing job. **Submit new synthesis** creates
  new work and may duplicate an earlier submission.
- SFT and annotation-refill TTS shares both default to **0%**. Enable them
  explicitly; they are independent maximum shares, not targets.
- Changing `TTS_BASE_URL` affects new campaigns only. Existing campaigns keep
  their provider and profiles.

XTTS, Chatterbox and OmniVoice are optional GPU profiles, not ready-to-use
defaults. They require separately obtained model assets and an authorized
`_data\tts\reference_voices\Arek.wav`; OmniVoice also needs a matching UTF-8
`Arek.txt`. XTTS requires explicit model-terms acceptance. See the
[TTS integration guide](tts/README.md#integration-in-this-archive).

## Operations and data

```powershell
docker compose ps -a
docker compose logs --tail 100 rlhf
# Source is copied into images: rebuild after code/template changes.
docker compose build rlhf
docker compose up -d --no-deps rlhf
```

Back up runtime data separately from source:

| Location | Contents |
| --- | --- |
| `_data\rlhf\` | Input recordings, manifests and accepted browser uploads |
| `_data\tts\` | TTS models and private reference voices |
| `rlhf_outputs` volume | Feedback database, checkpoints, logs, password configuration and publications |
| `tts_state` volume | Synthesis jobs and result archives |
| `bridge_auth` volume | Independent service credential |
| `rlhf_model_cache`, `asr_models`, `asr_cache` volumes | Cached models |

**Do not use `docker compose down --volumes` unless you intend to delete runtime
data.** Keep recordings, credentials, databases and weights out of Git.
TTS archive downloads expire after 24 hours; stored jobs/archives are not
automatically pruned.

## Documentation and development

- [Architecture](rlhf/docs/ARCHITECTURE.md) and [invariants](rlhf/CONTRACT.md)
- [Training guide](rlhf/docs/TRAINING_GUIDE.md)
- [Corpus import CLI](rlhf/run_import_corpus.py) (`--help`)
- [ASR interface and API](asr/README.md)
- [Source provenance](docs/SOURCE_PROVENANCE.md)
- [Archived development README](docs/archive/README-2026-11-24.md)
  (historical notes, not installation instructions)

The root [requirements.txt](requirements.txt) is a provisional web dependency
entry point only; it does not install PyTorch/torchaudio or the full service
stack. Use Compose for deployment.

With the relevant Python development dependencies installed:

```powershell
py -3.12 -m pytest -q tests
Push-Location rlhf
py -3.12 -m pytest -q
Pop-Location
```