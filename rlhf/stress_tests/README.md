# Stress Tests

This directory preserves the load suite from commit `ea20ff6` on branch
`tests-under-stress-BERT`. The request flows, load profiles, thresholds, and
monthly Poisson schedule are unchanged. Only paths and current-use instructions
were updated after extraction.

## Warning

These are state-changing tests. They create sessions, reserve pending pairs,
submit annotations, and can advance automatic training counters. Run them only
against a prepared disposable deployment or database. Do not run them against
valuable production annotation data.

The target must have enough pending pairs and valid audio for the strict test,
and `max_concurrent_sessions` must be at least 60. For the standard 50-user run,
prepare at least 1,000 pairs.

## Remote Deployment

For the two-machine endurance workflow, follow both role-specific runbooks:

- [LOAD_GENERATOR_GUIDE.md](LOAD_GENERATOR_GUIDE.md) — fresh-clone Linux load generator
- [DEPLOYMENT_OWNER_GUIDE.md](DEPLOYMENT_OWNER_GUIDE.md) — real web, coordinator, GPU, queue, and monitoring

The k6 clients are independent of the application runtime and accept any
deployment URL through `BASE_URL`.

From Linux or an SSH session:

```bash
docker run --rm \
  -v "$PWD/stress_tests/load:/scripts:ro" \
  grafana/k6 run \
  --env BASE_URL=https://www.caise.com \
  /scripts/loadtest.js
```

From PowerShell:

```powershell
docker run --rm `
  -v "${PWD}/stress_tests/load:/scripts:ro" `
  grafana/k6 run `
  --env BASE_URL=https://www.caise.com `
  /scripts/loadtest.js
```

The strict test ramps to 50 virtual users, holds for three minutes, downloads
audio, submits annotations, and checks the original latency and SQLite-lock
thresholds. Those thresholds were established on the historical four-worker
deployment; use them as comparative data until a baseline is recorded for the
current one-worker deployment.

For a shorter database-focused burst without audio downloads:

```bash
docker run --rm \
  -v "$PWD/stress_tests/load:/scripts:ro" \
  grafana/k6 run \
  --env BASE_URL=https://www.caise.com \
  --env SIM_VUS=10 \
  --env SIM_DURATION=30s \
  /scripts/sim_burst.js
```

## Local Preparation

The seeder intentionally requires application and database access, so it runs
inside the local web container rather than over the public API:

```bash
docker compose up -d web
docker compose exec web python stress_tests/fixtures/seed_db.py --pairs 1000
```

Clean up the synthetic records afterward:

```bash
docker compose exec web python stress_tests/fixtures/seed_db.py --cleanup
```

The historical seeder writes one physical WAV and virtual paths for the other
pairs. It is suitable for `sim_burst.js`, which omits audio. Use real queued
audio for `loadtest.js`, because that script downloads every assigned clip.

## Monthly Simulation

Remote Linux mode runs k6 through Docker, fetches real audio, and deliberately
does not reseed or monitor containers on the load-generator host:

```bash
python3 stress_tests/load/simulate_month.py --remote --dry-run --days 3
python3 stress_tests/load/simulate_month.py --remote --smoke --base-url https://deployment.example.com
python3 stress_tests/load/simulate_month.py --remote --days 30 --base-url https://deployment.example.com
python3 stress_tests/load/simulate_month.py --remote --resume --base-url https://deployment.example.com
```

Without `--remote`, the runner preserves its historical native-k6, local
`RESEED_CMD`, and local `docker stats` behavior. The deployment owner, not the
remote generator, is responsible for real queue refill and server-side metrics.

See [load/README.md](load/README.md) for the original detailed suite guide.
