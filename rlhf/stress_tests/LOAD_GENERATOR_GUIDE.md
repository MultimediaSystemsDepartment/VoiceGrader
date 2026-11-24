# Remote Load Generator Guide

This guide is for the operator of the separate Linux machine that generates
load. The target deployment, database, audio, GPU, and coordinator remain on
the deployment owner's machine.

## Before You Start

Obtain the deployment's externally reachable base URL and a clear confirmation
from its owner that the test window is open. The test creates real sessions and
annotations; with the coordinator enabled, those annotations can trigger real
training and policy promotion.

Also obtain the diagnostics UUID4 shown in the deployment Admin panel. Server
telemetry is recorded on the RLHF machine; k6 logs remain on this machine. The
shared run ID and UTC timestamps correlate the two artifact sets.

The Linux machine needs Git, Python 3.10 or newer, Docker, `curl`, and `tmux`.
It does not need a GPU, application dependencies, data archives, Kubernetes
credentials, or a local application stack. For Debian or Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git python3 docker.io tmux
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
```

Log out and back in after adding the Docker group. Do not run the endurance
suite as root.

```bash
git --version
python3 --version
docker version
docker run --rm grafana/k6:latest version
tmux -V
```

Your user must be able to run Docker without an interactive password prompt.
Do not use `localhost` as the target: that would address the load-generator
machine, not the deployment.

## Fresh Clone

```bash
git clone <repository-url> rlhf-for-asr
cd rlhf-for-asr
git checkout <approved-stress-suite-commit>
git status --short
```

Create a private runner configuration from the checked-in deployment profile.
The target is already set to the CAISE Kubernetes ingress. A diagnostics UUID4
is optional; use one only when it was supplied by the deployment owner.

```bash
cp stress_tests/load/remote.env.example remote.env
chmod 600 remote.env
nano remote.env
. ./remote.env
mkdir -p "$STRESS_LOG_ROOT"
curl --fail --show-error "$TARGET_URL/setup" >/dev/null
curl --fail --show-error "$TARGET_URL/api/stats"
docker pull "$K6_IMAGE"
docker run --rm "$K6_IMAGE" version
```

These `curl` commands are optional observations. A failed probe, invalid
application response, shallow queue, or unavailable diagnostics capture does
not prevent a remote run: k6 records target-side failures in each burst log.
Only load-station failures such as a missing k6 runtime should block startup.

In Admin settings, set `max_concurrent_sessions` to at least `60` before the
rehearsal. The application default is `10`, which admits only ten virtual users
and returns a capacity page to the rest. Failed k6 bursts are recorded in their
individual logs and summaries, and the schedule continues with the next event.
Use `--fail-fast` only when an operator deliberately wants the first failed
burst to stop the run.

## Rehearsal

Preview the deterministic schedule without contacting the deployment:

```bash
python3 stress_tests/load/simulate_month.py \
  --remote --dry-run --days 1 --seed 42
```

Then run the compressed one-day smoke test. Remote mode uses Docker k6, fetches
real audio, and disables load-generator-side reseeding and container metrics.

```bash
python3 stress_tests/load/simulate_month.py \
  --remote --smoke --seed 42 \
  --base-url "$TARGET_URL" \
  --log-dir "$STRESS_LOG_ROOT/rehearsal"
```

Review every burst log under `$STRESS_LOG_ROOT/rehearsal/`. A k6 threshold
failure gives that burst a nonzero exit code, but the orchestrator records it
and continues. HTTP errors, audio failures, database-lock errors, and sustained
latency remain visible as test results without interrupting the schedule.

For the historical strict 50-user baseline, run separately:

```bash
docker run --rm \
  -v "$PWD/stress_tests/load:/scripts:ro" \
  "$K6_IMAGE" run \
  --env BASE_URL="$TARGET_URL" \
  --env DIAGNOSTICS_RUN_ID="$DIAGNOSTICS_RUN_ID" \
  /scripts/loadtest.js
```

This ramps to 50 users, holds for three minutes, fetches audio, and uses the
original strict thresholds. It is more abrupt than the endurance schedule.

## Start the 30-Day Run

Use a dedicated log directory and a persistent terminal session:

```bash
tmux new -s asr-endurance
cd ~/rlhf-for-asr
. ./remote.env
python3 stress_tests/load/simulate_month.py \
  --remote --days 30 --speed 1 \
  --base-url "$TARGET_URL" \
  --log-dir "$STRESS_LOG_ROOT/endurance"
```

Detach with `Ctrl-b`, then `d`. Reattach with:

```bash
tmux attach -t asr-endurance
```

The run creates `checkpoint.json`, one text log and one structured k6 summary
JSON per burst, daily summaries, and a final `monthly_report.txt` under the
selected log directory. The JSON files preserve latency percentiles, request
and failure rates, checks, submissions, and error counters for later graphs.
Check available disk space periodically; do not edit or remove the checkpoint
during the run.

To produce a denser schedule while retaining lower overnight and higher
work-hour activity, choose one of these options:

- `--bursts-per-day 48` creates exactly 9 night, 26 work, and 13 evening bursts
  each day.
- `--average-bursts-per-day 48` uses the same proportions with natural Poisson
  variation around 48 total bursts per day.

Omit both options for the default Poisson schedule of roughly 9-10 bursts per
day.

## Resume After Interruption

Use the same target and log directory. Repeat `--remote` because execution mode
is intentionally supplied by the operator, while schedule and progress come
from the checkpoint.

```bash
python3 stress_tests/load/simulate_month.py \
  --remote --resume \
  --base-url "$TARGET_URL" \
  --log-dir "$STRESS_LOG_ROOT/endurance"
```

Keep the completed logs and checkpoint when handing off results. Resume rejects
a target URL, remote/local mode, or k6 runtime that differs from
`checkpoint.json`.

## Operator Stops

Target-side HTTP, audio, capacity, latency, and database failures are test
results: k6 records them and the orchestrator continues. Stop the generator
only for a load-station problem such as insufficient local disk, a missing k6
runtime, corrupted checkpoint state, or an explicit request from the deployment
owner.

At handoff, preserve the complete selected `--log-dir`. The server-side Admin
bundle does not ingest or replace these k6 burst logs.

Create a compressed handoff artifact after the run has stopped:

```bash
tar -C "$STRESS_LOG_ROOT" -czf \
  "$HOME/rlhf-asr-endurance-$(date -u +%Y%m%dT%H%M%SZ).tar.gz" endurance
sha256sum "$HOME"/rlhf-asr-endurance-*.tar.gz
```
