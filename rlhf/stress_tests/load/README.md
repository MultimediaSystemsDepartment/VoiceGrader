# Load Test — k6

Simulates 50 concurrent annotators running the full annotation flow against the
running web container. Uses the official `grafana/k6` Docker image — no local
k6 install required.

---

## Prerequisites

1. **Web container running:**
   ```
   docker compose up -d web
   ```

2. **Seed test data** (run once before each test run — seed is consumed as pairs
   are annotated; re-seed if the queue empties mid-test):
   ```
   docker compose run --rm web python stress_tests/fixtures/seed_db.py --pairs 1000
   ```
   This also sets `max_concurrent_sessions = 60` in the DB (default is 10 —
   the test would silently fail to create sessions without this).

3. **Do NOT share CPU with the load generator.** Run k6 on a separate machine
   or at minimum in a separate container. If k6 and uvicorn compete for the
   same CPU cores, both latency numbers and error rates will be artificially
   inflated. The `--network` flag below connects k6 to the Compose network so
   it can reach the container by service name.

---

## Running

### Against localhost (k6 on the same host, web container exposed on 8080)

```bash
docker run --rm \
  -v "$(pwd)/stress_tests/load:/scripts" \
  grafana/k6 run \
  --env BASE_URL=http://host.docker.internal:8080 \
  /scripts/loadtest.js
```

On Linux, `host.docker.internal` may not resolve. Use your host's IP:
```bash
docker run --rm \
  -v "$(pwd)/stress_tests/load:/scripts" \
  grafana/k6 run \
  --env BASE_URL=http://172.17.0.1:8080 \
  /scripts/loadtest.js
```

### Against the Compose network (recommended — no port exposure needed)

```bash
docker run --rm \
  --network rlhf-asr_default \
  -v "$(pwd)/stress_tests/load:/scripts" \
  grafana/k6 run \
  --env BASE_URL=http://web:8080 \
  /scripts/loadtest.js
```

> The Compose project name is `rlhf-asr` (from `name: rlhf-asr` in
> `docker-compose.yml`), so the network is `rlhf-asr_default`.

## Thresholds

| Metric | Threshold | What it catches |
|--------|-----------|-----------------|
| `http_req_failed` | rate < 1 % | HTTP 4xx/5xx errors |
| `http_req_duration` | p(95) < 1 500 ms | overall latency |
| `submit_duration` | p(95) < 2 000 ms | write path (SQLite) latency |
| `db_locked_errors` | count < 1 | `database is locked` in any response body |

---

## Tuning think time

`THINK_TIME_MIN` and `THINK_TIME_MAX` in `loadtest.js` control the simulated
listening + decision time per pair (default: 5–12 s).

To stress the server harder (more requests/s, less realistic):
```js
const THINK_TIME_MIN = 0;
const THINK_TIME_MAX = 0.5;
```

To simulate slow annotators (lower throughput, more realistic):
```js
const THINK_TIME_MIN = 15;
const THINK_TIME_MAX = 60;
```

---

## Cleanup after test

```
docker compose run --rm web python stress_tests/fixtures/seed_db.py --cleanup
```

---

## Expected bottleneck

Fix A (`asyncio.to_thread`) + Fix B (4 uvicorn workers) are applied — see
`tests/db/README.md` for details and benchmark results (1.9× throughput gain).
Remaining bottleneck is SQLite single-writer WAL lock; p50 latency at 50 VU
is ~15 s. For heavier production loads consider PostgreSQL.

---

## Monthly load simulation

`simulate_month.py` + `sim_burst.js` orchestrate a realistic 30-day burst
schedule using three intensity zones (night / work / evening) with Poisson
arrival times and Gaussian intra-day peaks.

For a separate Linux load generator targeting a real deployment, use
[`../LOAD_GENERATOR_GUIDE.md`](../LOAD_GENERATOR_GUIDE.md). Its `--remote` mode
runs k6 in Docker, fetches audio, and disables local reseeding and monitoring.

### Quick start

```bash
# Prerequisites: web container up + seed data loaded
docker compose up -d web
docker compose exec web python stress_tests/fixtures/seed_db.py --pairs 500

# Smoke test (~3 min — validates orchestrator end-to-end)
python stress_tests/load/simulate_month.py --smoke

# Preview the schedule for 3 days (no k6 run)
python stress_tests/load/simulate_month.py --dry-run --days 3

# Full 30-day real-time simulation (run in screen / tmux)
python stress_tests/load/simulate_month.py

# Resume after interruption
python stress_tests/load/simulate_month.py --resume
```

### Speed flag

`--speed N` compresses simulated time: 1 real second = N simulated seconds.

| Command | Real duration | Simulated duration |
|---------|---------------|--------------------|
| `--smoke` (= `--speed 1440`) | ~3 min | 1 day |
| `--speed 60` | ~24 min | 1 day |
| `--speed 24` | ~1 hour | 1 day |
| (default) | 24 hours | 1 day |

In speed mode, k6 burst duration is compressed too (floor: 10 s real).

### Log structure

```
stress_tests/load/logs/
├── checkpoint.json          # resume state (auto-updated after every burst)
├── monthly_report.txt       # written at simulation end
├── 2026-07-21/
│   ├── orchestrator.log     # timestamped sleep/burst events
│   ├── 09-15-42_35vus_work.txt     # k6 stdout+stderr per burst
│   ├── 09-15-42_35vus_work_summary.json  # structured k6 metrics for graphs
│   ├── 14-03-11_48vus_work.txt
│   └── summary.txt          # daily aggregation (written at midnight)
└── 2026-07-22/
    └── …
```

### Intensity zones

| Zone | Hours | VUs | Mean bursts/day | Notes |
|------|-------|-----|-----------------|-------|
| night | 00–09 | 3–15 | ~1.5 | insomniacs / automation |
| work | 09–17 | 15–50 | ~5 | peaks at 10:30 and 14:30 |
| evening | 17–24 | 5–25 | ~2.5 | post-work usage |

Auto-reseed triggers when `pending_pairs < 200` (configurable via `RESEED_THRESHOLD`).
It applies only to local mode; remote mode leaves queue refill to the target's
coordinator.
