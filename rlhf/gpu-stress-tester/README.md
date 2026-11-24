# VoiceMed ASR RLHF Stress Testing Panel

A standalone Docker appliance that repeatedly runs real Whisper LoRA SFT and
DPO training on ADMEDVOICE. A transcript-based oracle supplies deterministic
preference annotations, so no human annotator or production RLHF service is
required.

The service starts idle. Training begins only after an operator opens the web
dashboard and selects **Start run**.

## What It Exercises

- ADMEDVOICE audio decoding and 16 kHz normalization
- Whisper Medium beam and temperature-sampled generation in Polish
- Ground-truth preference generation and correction labels
- LoRA supervised fine-tuning
- Exact FP16 round-start reference model and DPO training
- Repeated model loading, optimizer work, checkpoint writes, and GPU telemetry

This tester has no production database, model promotion, annotation queue, or
publishing integration. Every run is disposable and isolated under `outputs/`.

## Host Requirements

- Linux host with an NVIDIA GPU
- NVIDIA driver compatible with CUDA 12.4
- Docker Engine with the Compose plugin
- NVIDIA Container Toolkit configured for Docker
- Approximately 40 GB of free disk for the image, model cache, and checkpoints
- ADMEDVOICE archive in its original ZIP layout
- Internet access for the first image build and first model download

Confirm GPU container support before setup:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Setup

1. Extract this directory on the GPU host.
2. Create the runtime directories:

   ```bash
   mkdir -p data outputs model-cache
   ```

3. Place ADMEDVOICE at `data/archive.zip`. The archive must contain the
   per-session CSV files and their referenced audio files. CSV rows use the
   original `filename;id;phrase;...` format.
4. Optionally create local configuration:

   ```bash
   cp .env.example .env
   ```

5. Build and start the idle service:

   ```bash
   docker compose up --build
   ```

6. Open <http://localhost:8080> and confirm that Dataset shows **Ready** and
   the GPU name is visible.

For detached operation:

```bash
docker compose up -d --build
docker compose logs -f tester
```

## Running a Test

Choose a profile and select **Start run**. Optional numeric fields override the
profile defaults for that run.

| Profile | Default workload | Intended use |
|---|---:|---|
| Smoke | 8 clips, 1 round | Installation check |
| Standard | 64 clips, 1 round | Representative training pass |
| Endurance | 128 clips per round | Continuous operation until stopped |
| Max load | 256 clips per round, batch 4 | Sustained high GPU and VRAM load |

The dashboard shows the active stage, round, progress, GPU utilization, VRAM,
temperature, power, and the last 300 log lines. Only one run may be active.

Start with Smoke. Whisper Medium DPO keeps both the policy and an exact frozen
reference model in GPU memory. Reduce samples or batch size in source if the
host cannot accommodate the default workload; do not quantize only the
reference because that invalidates DPO's round-start comparison.

## Stopping

**Stop** creates a cooperative stop request. Generation and training check it
between clips or batches, save the run state, release GPU memory, and exit.

**Force stop** sends SIGTERM to the worker process group and uses SIGKILL only
if it has not exited after ten seconds. Partial artifacts remain for diagnosis.

To stop the entire service:

```bash
docker compose down
```

## Outputs

Each run receives a UUID directory:

```text
outputs/
├── current.json
└── runs/<run-id>/
    ├── config.json
    ├── state.json
    ├── run.log
    └── round-00001/
        ├── preferences.jsonl
        ├── summary.json
        ├── sft/
        └── dpo/
```

Delete old run directories only while no run is active. The Hugging Face cache
under `model-cache/` can be retained between deployments to avoid downloading
the base model again.

## Remote Access

The dashboard binds to localhost by default and has no authentication. For a
trusted private network, set `BIND_ADDRESS=0.0.0.0` in `.env` and restrict port
8080 with the host firewall. For wider access, keep the localhost binding and
place an authenticated TLS reverse proxy in front of it.

Never expose this control API directly to the public internet: it can start a
costly GPU workload.

## Offline Delivery

On an internet-connected build machine:

```bash
docker compose build
docker save rlhf-gpu-training-stress-tester:latest \
  -o rlhf-gpu-training-stress-tester.tar
```

Copy this source directory, the image archive, and a populated `model-cache/`
directory to the provider. On the offline host:

```bash
docker load -i rlhf-gpu-training-stress-tester.tar
docker compose up --no-build
```

The model cache must contain `openai/whisper-medium`; model weights are not
included in the source ZIP.

## Packaging the Source

On Linux (`zip` must be installed):

```bash
./package.sh
```

From PowerShell:

```powershell
.\package.ps1
```

Both scripts create `gpu-stress-tester.zip` beside this directory and exclude
runtime data, checkpoints, caches, local environment settings, and test caches.
ADMEDVOICE data and generated checkpoints should be transferred under the
provider's approved data-handling process, not bundled with source code.

## Development Validation

CPU-only tests do not load PyTorch or require a GPU:

```bash
python3.12 -m pip install pytest soundfile numpy jiwer fastapi jinja2
python3.12 -m pytest -q -c pyproject.toml tests
python3.12 -m compileall -q src
```

The definitive runtime validation is the Smoke profile on the target GPU.

## Troubleshooting

- **Dataset missing:** confirm `data/archive.zip` exists and is readable inside
  the container with `docker compose exec tester ls -l /data/archive.zip`.
- **GPU unavailable:** rerun the NVIDIA CUDA check above and verify NVIDIA
  Container Toolkit configuration.
- **CUDA out of memory:** stop the run, remove unrelated GPU workloads, and use
  Smoke before attempting Standard. Whisper DPO requires policy and reference
  models concurrently.
- **Model download fails:** verify outbound Hugging Face access or populate the
  mounted model cache before delivery.
- **Interrupted after restart:** this is intentional. Start a new run; existing
  logs and partial checkpoints remain under the old UUID directory.
- **Port already in use:** set another `PORT` value in `.env`, then restart
  Compose.