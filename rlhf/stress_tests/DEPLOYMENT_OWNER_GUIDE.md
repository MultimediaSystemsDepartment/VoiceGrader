# Deployment Owner Guide

This guide is for the owner of the real web and coordinator deployment. The
load generator is a separate Linux machine and must not have database, Docker,
GPU, or filesystem access to this host.

## Safety Agreement

Treat the endurance test as a real annotation and training workload. It creates
sessions, consumes pending pairs, stores synthetic user choices, can cross
automatic training thresholds, and can produce policy candidates. Use only
approved or disposable data and artifacts. Do not point it at valuable
production annotation data without an explicit decision to retain its output.

Agree with the load-generator operator on the URL, start time, stop conditions,
communication channel, and who may approve resumption or rollback.

## Prepare the Deployment

On the deployment host, update and start the normal full system:

```bash
git pull --ff-only
docker compose up -d --build web coordinator
docker compose ps
```

The coordinator requires NVIDIA Container Toolkit, a working GPU, the expected
archives/manifests under `_data/`, and writable persistent `_outputs/` storage.
Keep the normal queue-refill path enabled. Do not use the synthetic stress-test
seeder for this full-system run: it creates virtual audio paths and cannot test
the audio or training workflow.

Before opening the test window, verify all of the following:

- The web and coordinator containers are healthy and stable.
- The deployment URL is reachable from outside the host with valid TLS.
- Pending pairs were generated from real, readable audio by the active policy.
- `max_concurrent_sessions` is at least `60` in the admin settings.
- Queue refill can keep up with consumption without repeatedly producing zero pairs.
- Training and evaluation inputs exclude the frozen WER Monitor audio.
- The WER Monitor is locked and policy promotion still requires the normal paired-WER and observable-change gate.
- `_outputs/` and the database have a current backup or snapshot and enough free space for a month of logs and checkpoints.
- GPU memory, host memory, and disk capacity have conservative headroom.

Check the public endpoints without reserving a pair:

```bash
export TARGET_URL="https://deployment.example.com"
curl --fail --show-error "$TARGET_URL/setup" >/dev/null
curl --fail --show-error "$TARGET_URL/api/stats"
```

Confirm readiness to the load-generator operator only after the real queue and
audio are ready. The remote runner intentionally cannot reseed this deployment.

## Start Server Diagnostics

Diagnostics starts automatically as an isolated thread in the coordinator. It
uses the coordinator's selected GPU, shared outputs volume, and runtime-visible
`/proc`; no additional Docker Compose service, Kubernetes workload,
container-runtime socket, host namespace mount, or Kubernetes API credential is
required. In Kubernetes, OS metrics describe the coordinator pod's runtime view
and are not a replacement for node or cluster monitoring.

1. Log in to `/admin/diagnostics`.
2. Confirm the collector status is `idle` and its heartbeat is current.
3. Enter a run label and select **Start capture**.
4. Copy the displayed UUID4 run ID and give it to the load-generator operator.

The collector records runtime-visible CPU, memory, load, disk capacity and I/O,
network rates, selected-GPU utilization, VRAM, temperature and power, web
reachability, queue and training state, coordinator heartbeat, and deduplicated
problem events. Container lifecycle, restart, OOM, and per-workload limit data
must come from the deployment platform's normal observability stack. A
provider-facing report should cite UTC windows and summarized findings and
attach the sanitized ZIP; do not send the sensitive ZIP without a separate
privacy review.

The service writes and flushes line-oriented metrics continuously and replaces
a sanitized partial ZIP atomically every 60 seconds. Admin may also request a
fresh sanitized or sensitive partial bundle while capture continues. Partial
bundles show their capture time and sample count and omit an incomplete trailing
record if a writer was interrupted mid-line.

The Compose restart policy restarts a killed diagnostics process. On restart,
the newest running capture resumes in append mode and records a collector
restart event; older orphaned running manifests are marked `interrupted` and
remain bundleable. Already-created partial ZIPs and raw run files remain under
`_outputs/diagnostics/runs`. This recovery depends on persistent `_outputs/`
storage and does not protect against host disk loss or manual deletion.

## Monitor During the Run

Keep server-side monitoring on this host. Useful live views are:

```bash
docker compose ps
docker compose logs --since 10m web coordinator
nvidia-smi
df -h _data _outputs
```

Use the site's admin and model-health views to watch queue depth, annotation
counts, training rounds, WER Monitor readiness, pending candidates, activation
history, and failures. Preserve web, coordinator, GPU, disk, and system logs
with timestamps that can be correlated with the load generator's burst logs.

Use `/admin/diagnostics` to watch collector heartbeat, samples, free space, and
active issues. Retention is manual, so completed captures remain until an admin
deletes them and free disk space remains the deployment owner's responsibility.

The source tree is bind-mounted. Normal Python or template corrections need a
container restart, but do not change code or configuration mid-run without
recording the time and reason; it changes the meaning of the endurance result.

## Pause and Recovery

Ask the load-generator operator to pause if queue refill stalls, repeated audio
or HTTP failures occur, SQLite lock errors appear, training loops fail, disk or
GPU resources approach exhaustion, model-health evaluation is unavailable, or
an unexpected policy transition occurs.

Let in-flight training reach a known state when possible, preserve logs and the
database before repair, and use only the application's audited activation or
rollback controls for policy changes. Never repair a failed promotion by
editing the active checkpoint pointer or activation records directly.

After recovery, verify web health, coordinator health, real pending pairs,
audio delivery, model-health state, and free capacity. Explicitly authorize the
load-generator operator to resume from their existing checkpoint.

## End of Test

Stop the capture from `/admin/diagnostics` and wait for `completed`. The service
automatically replaces the rolling sanitized ZIP with a final snapshot; rebuild
or download the **sanitized** bundle first. It contains metrics, safe request
events, issue history, manifests, and fingerprints. It excludes the database,
transcripts, participant IDs, raw logs, audio, checkpoints, password
configuration, tokens, and environment dumps.

Build the **sensitive** bundle only for authorized forensic analysis. It adds a
consistent SQLite snapshot and raw application logs and must be handled as
annotation data. Neither bundle contains audio, model checkpoints, password
configuration, or environment dumps. Verify the SHA-256 shown in Admin.

At completion, archive the database, activation history, coordinator and web
logs, model-health reports, training checkpoints, GPU/system metrics, and the
exact deployed commit and configuration. Obtain the load generator's log tree
and correlate failed bursts with server-side events before deciding which test
annotations or trained artifacts may be retained.
