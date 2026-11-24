/**
 * stress_tests/load/loadtest.js — k6 load test for the RLHF-ASR annotation web UI.
 *
 * Simulates 50 concurrent annotators, each maintaining their own session and
 * performing the full annotation loop:
 *   POST /setup → GET /api/pair → GET /audio/<id> → POST /api/submit → GET /api/stats
 * with periodic POST /api/heartbeat.
 *
 * PREREQUISITE: Run the seed script inside the web container BEFORE this test:
 *   docker compose run --rm web python stress_tests/fixtures/seed_db.py --pairs 1000
 *
 * RUN (see stress_tests/load/README.md for the full Docker command):
 *   BASE_URL=http://localhost:8080 k6 run stress_tests/load/loadtest.js
 */

import http from "k6/http";
import { check, sleep } from "k6";
import { Counter, Trend } from "k6/metrics";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const BASE_URL = __ENV.BASE_URL || "http://localhost:8080";
const DIAGNOSTICS_RUN_ID = __ENV.DIAGNOSTICS_RUN_ID || "";

function requestParams(endpoint, extra = {}) {
  return {
    ...extra,
    headers: {
      ...(extra.headers || {}),
      ...(DIAGNOSTICS_RUN_ID ? { "X-Diagnostics-Run-ID": DIAGNOSTICS_RUN_ID } : {}),
    },
    tags: { ...(extra.tags || {}), endpoint },
  };
}

// Think time = simulated listening + decision time.
// Decrease THINK_TIME_MIN/MAX to stress the server harder (e.g. 0 / 0.5).
const THINK_TIME_MIN = 5;   // seconds
const THINK_TIME_MAX = 12;  // seconds

// Probability of including a correction text (drives SFT path in submit handler).
const CORRECTION_PROBABILITY = 0.3;

// Heartbeat every N pairs (~60 s at average think time of 8 s).
const HEARTBEAT_EVERY_N = 7;

// ---------------------------------------------------------------------------
// Custom metrics
// ---------------------------------------------------------------------------

const dbLockedErrors   = new Counter("db_locked_errors");
const submitDuration   = new Trend("submit_duration", true);   // true = report ms
const pairFetchEmpty   = new Counter("pair_queue_empty");

// ---------------------------------------------------------------------------
// Load profile: ramp to 50 VUs in 30 s, hold for 3 min, ramp down
// ---------------------------------------------------------------------------

export const options = {
  scenarios: {
    annotators: {
      executor: "ramping-vus",
      startVUs: 0,
      stages: [
        { duration: "30s", target: 50 },   // ramp up
        { duration: "3m",  target: 50 },   // sustained load
        { duration: "20s", target: 0  },   // ramp down
      ],
      gracefulRampDown: "15s",
    },
  },
  thresholds: {
    // Overall error rate < 1 %
    http_req_failed: ["rate<0.01"],

    // 95th percentile of all requests < 1500 ms
    http_req_duration: ["p(95)<1500"],

    // 95th percentile of submit specifically < 2000 ms
    submit_duration: ["p(95)<2000"],

    // Zero database-is-locked errors during the test
    db_locked_errors: ["count<1"],
  },
};

// ---------------------------------------------------------------------------
// Global setup: verify server is reachable (runs once, before any VU starts)
// ---------------------------------------------------------------------------

export function setup() {
  const res = http.get(`${BASE_URL}/setup`, requestParams("setup"));
  check(res, {
    "server is reachable (GET /setup = 200)": (r) => r.status === 200,
  });
  if (res.status !== 200) {
    console.error(`[setup] Server not reachable at ${BASE_URL}. Aborting.`);
  }
  return { baseUrl: BASE_URL };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function randomChoice(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function thinkTime() {
  return THINK_TIME_MIN + Math.random() * (THINK_TIME_MAX - THINK_TIME_MIN);
}

function checkForDbLock(response, tag) {
  const body = response.body || "";
  if (body.includes("database is locked") || body.includes("OperationalError")) {
    dbLockedErrors.add(1, { endpoint: tag });
    console.error(`[VU ${__VU}] DB locked at ${tag}: ${body.substring(0, 200)}`);
  }
}

// ---------------------------------------------------------------------------
// Per-VU state (module-level = isolated per VU in k6)
// ---------------------------------------------------------------------------

let sessionEstablished = false;
let pairCount          = 0;

// ---------------------------------------------------------------------------
// Main VU function
// ---------------------------------------------------------------------------

export default function ({ baseUrl }) {
  // ── 1. Establish session (once per VU lifetime) ──────────────────────────
  if (!sessionEstablished) {
    const setupRes = http.post(
      `${baseUrl}/setup`,
      // application/x-www-form-urlencoded -- FastAPI Form() accepts this.
      { participant_id: `load_vu${__VU}_${Date.now()}` },
      // Follow the 303 redirect so /annotate page is also loaded (realistic).
      requestParams("setup")
    );

    const setupOk = check(setupRes, {
      "setup: landed on /annotate (200)": (r) => r.status === 200,
      "setup: no capacity error": (r) =>
        !r.body.includes("Server is at capacity"),
    });

    if (!setupOk) {
      console.warn(
        `[VU ${__VU}] Setup failed (status ${setupRes.status}). ` +
          "Ensure seed_db.py was run and max_concurrent_sessions >= 60."
      );
      sleep(5);
      return; // retry on next iteration
    }

    sessionEstablished = true;
    sleep(0.3);
  }

  // ── 2. Heartbeat (every HEARTBEAT_EVERY_N pairs) ─────────────────────────
  if (pairCount > 0 && pairCount % HEARTBEAT_EVERY_N === 0) {
    const hbRes = http.post(
      `${baseUrl}/api/heartbeat`,
      null,
      requestParams("heartbeat")
    );
    check(hbRes, {
      "heartbeat: ok=true": (r) => {
        try { return r.json().ok === true; } catch { return false; }
      },
    });
    checkForDbLock(hbRes, "heartbeat");
  }

  // ── 3. Fetch next pair ────────────────────────────────────────────────────
  const pairRes = http.get(
    `${baseUrl}/api/pair`,
    requestParams("get_pair")
  );

  check(pairRes, {
    "get_pair: status 200": (r) => r.status === 200,
    "get_pair: valid JSON": (r) => {
      try { r.json(); return true; } catch { return false; }
    },
  });
  checkForDbLock(pairRes, "get_pair");

  if (pairRes.status === 401) {
    // Session lost (e.g. container restarted mid-test) -- re-establish.
    sessionEstablished = false;
    sleep(1);
    return;
  }

  let pairData;
  try {
    pairData = pairRes.json();
  } catch {
    sleep(2);
    return;
  }

  if (pairData.done) {
    pairFetchEmpty.add(1);
    console.warn(`[VU ${__VU}] Queue empty -- increase --pairs in seed_db.py`);
    sleep(5);
    return;
  }

  const pairId  = pairData.pair_id;
  const audioId = pairData.audio_id;

  check(pairData, {
    "get_pair: has pair_id":  (d) => typeof d.pair_id  === "string" && d.pair_id.length > 0,
    "get_pair: has hyp_a":    (d) => typeof d.hyp_a    === "string",
    "get_pair: has hyp_b":    (d) => typeof d.hyp_b    === "string",
    "get_pair: done is false":(d) => d.done === false,
  });

  // ── 4. Fetch audio (streaming -- just download, measure latency) ──────────
  const audioRes = http.get(
    `${baseUrl}/audio/${pairId}`,
    requestParams("audio", { responseType: "binary" })
  );

  check(audioRes, {
    "audio: status 200":        (r) => r.status === 200,
    "audio: content-type WAV":  (r) =>
      (r.headers["Content-Type"] || "").includes("audio/wav"),
    "audio: non-empty body":    (r) => r.body && r.body.byteLength > 44, // > WAV header
  });
  checkForDbLock(audioRes, "audio");

  // ── 5. Think time (annotator listens and decides) ────────────────────────
  // Decrease THINK_TIME_MIN / THINK_TIME_MAX to put more pressure on the server.
  sleep(thinkTime());

  // ── 6. Submit annotation ─────────────────────────────────────────────────
  const preference = randomChoice(["a", "b", "equal"]);
  const correction =
    Math.random() < CORRECTION_PROBABILITY
      ? "Poprawiona transkrypcja testowa."
      : "";

  const submitBody = {
    pair_id:         pairId,
    preference:      preference,      // 'a' | 'b' | 'equal'  (lowercase)
    correction_text: correction,
    strength:        "5",             // float 0-10, FastAPI converts to float
  };

  const submitStart = Date.now();
  const submitRes = http.post(
    `${baseUrl}/api/submit`,
    submitBody,
    requestParams("submit")
  );
  submitDuration.add(Date.now() - submitStart);

  check(submitRes, {
    "submit: status 200":  (r) => r.status === 200,
    "submit: ok=true":     (r) => {
      try { return r.json().ok === true; } catch { return false; }
    },
    "submit: has stats":   (r) => {
      try { return typeof r.json().stats === "object"; } catch { return false; }
    },
  });
  checkForDbLock(submitRes, "submit");

  pairCount++;

  // ── 7. Fetch stats (as frontend does after submit) ────────────────────────
  const statsRes = http.get(
    `${baseUrl}/api/stats`,
    requestParams("stats")
  );

  check(statsRes, {
    "stats: status 200":             (r) => r.status === 200,
    "stats: has pending_pairs key":  (r) => {
      try { return "pending_pairs" in r.json(); } catch { return false; }
    },
  });
  checkForDbLock(statsRes, "stats");

  // Small inter-request pause (browser UI refresh cadence).
  sleep(0.2);
}
