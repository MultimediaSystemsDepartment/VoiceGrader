/**
 * stress_tests/load/sim_burst.js — Single configurable k6 burst for the monthly simulator.
 *
 * Unlike loadtest.js (ramping-vus, audio streaming, full perf thresholds),
 * this script runs a fixed VU count for a fixed duration and focuses on
 * annotation DB operations: setup → get pair → submit → stats.
 * Audio fetching is configurable. Local synthetic seed pairs use virtual file
 * paths, while remote endurance runs can exercise the real audio endpoint.
 *
 * Parameters (via env vars — set by simulate_month.py):
 *   BASE_URL      target server          (default: http://localhost:8080)
 *   SIM_VUS       number of virtual users (default: 10)
 *   SIM_DURATION  k6 duration string      (default: "60s")
 *   SIM_FETCH_AUDIO fetch each pair's audio (default: "0")
 *
 * Standalone smoke run:
 *   k6 run --env SIM_VUS=5 --env SIM_DURATION=30s stress_tests/load/sim_burst.js
 */

import http from "k6/http";
import { check, sleep } from "k6";
import { Counter, Trend } from "k6/metrics";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const BASE_URL    = __ENV.BASE_URL    || "http://localhost:8080";
const SIM_VUS     = parseInt(__ENV.SIM_VUS    || "10");
const SIM_DURATION = __ENV.SIM_DURATION || "60s";
const SIM_FETCH_AUDIO = (__ENV.SIM_FETCH_AUDIO || "0") === "1";
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

export const options = {
  scenarios: {
    annotators: {
      executor: "constant-vus",
      vus: SIM_VUS,
      duration: SIM_DURATION,
      gracefulStop: "60s",
    },
  },
  noCookiesReset: true,
  thresholds: {
    checks:           ["rate>0.95"],
    http_req_failed:  ["rate<0.01"],
    audio_duration:   ["p(95)<8000"],
    submit_duration:  ["p(95)<8000"],  // 8 s at p95 (single-worker baseline)
    db_locked_errors: ["count<1"],     // zero tolerance for lock errors
    audio_errors:     ["count<1"],     // remote mode requires valid WAV responses
    capacity_errors:  ["count<1"],
    pair_queue_empty: ["count<1"],
    session_errors:   ["count<1"],
    submit_errors:    ["count<1"],
    successful_submissions: ["count>0"],
  },
};

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

const dbLockedErrors = new Counter("db_locked_errors");
const audioDuration  = new Trend("audio_duration", true);
const submitDuration = new Trend("submit_duration", true);
const pairQueueEmpty = new Counter("pair_queue_empty");
const audioErrors    = new Counter("audio_errors");
const capacityErrors = new Counter("capacity_errors");
const sessionErrors  = new Counter("session_errors");
const submitErrors   = new Counter("submit_errors");
const successfulSubmissions = new Counter("successful_submissions");

// ---------------------------------------------------------------------------
// Per-VU state (k6: module-level vars are isolated per VU)
// ---------------------------------------------------------------------------

let sessionReady = false;
let pairCount    = 0;

// ---------------------------------------------------------------------------
// Global setup — verify server is up before VUs start
// ---------------------------------------------------------------------------

export function setup() {
  const res = http.get(`${BASE_URL}/setup`, requestParams("setup"));
  if (!check(res, { "server reachable": (r) => r.status === 200 })) {
    console.error(`[setup] ${BASE_URL}/setup returned ${res.status} — is the container running?`);
  }
  return { baseUrl: BASE_URL };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function _checkDbLock(resp, tag) {
  const body = resp.body || "";
  if (
    typeof body === "string" &&
    (body.includes("database is locked") || body.includes("OperationalError"))
  ) {
    dbLockedErrors.add(1, { endpoint: tag });
    console.error(`[VU ${__VU}] DB lock at ${tag}`);
  }
}

// ---------------------------------------------------------------------------
// Main VU loop
// ---------------------------------------------------------------------------

export default function ({ baseUrl }) {

  // ── 1. Session setup (once per VU lifetime) ──────────────────────────────
  if (!sessionReady) {
    const participantId = `sim_vu${__VU}_${Date.now()}`;
    const res = http.post(
      `${baseUrl}/setup`,
      { participant_id: participantId },
      requestParams("setup", { redirects: 0 })
    );
    const capacityError = (res.body || "").includes("Server is at capacity");
    if (capacityError) capacityErrors.add(1);
    const sessionCookie = (res.cookies.session_id || [])[0];
    const participantCookie = (res.cookies.participant_id || [])[0];
    const jar = http.cookieJar();
    if (sessionCookie) {
      jar.set(baseUrl, "session_id", sessionCookie.value, { path: "/" });
    }
    if (participantCookie) {
      jar.set(baseUrl, "participant_id", participantCookie.value, { path: "/" });
    }
    const annotateRes = sessionCookie
      ? http.get(`${baseUrl}/annotate`, requestParams("annotate"))
      : null;
    const ok = check(res, {
      "setup redirect 303":  (r) => r.status === 303,
      "session cookie issued": () => Boolean(sessionCookie),
      "no capacity error":   () => !capacityError,
    }) && check(annotateRes, {
      "annotate 200": (r) => r !== null && r.status === 200,
    });
    if (!ok) {
      sessionErrors.add(1);
      sleep(3);
      return;   // retry on next iteration
    }
    sessionReady = true;
    sleep(0.2);
  }

  // ── 2. Periodic heartbeat (every 7 pairs) ────────────────────────────────
  if (pairCount > 0 && pairCount % 7 === 0) {
    const hb = http.post(`${baseUrl}/api/heartbeat`, null, requestParams("heartbeat"));
    check(hb, { "heartbeat ok": (r) => { try { return r.json().ok; } catch { return false; } } });
    _checkDbLock(hb, "heartbeat");
  }

  // ── 3. Get next pair ──────────────────────────────────────────────────────
  const pairRes = http.get(`${baseUrl}/api/pair`, requestParams("get_pair"));
  _checkDbLock(pairRes, "get_pair");

  if (pairRes.status === 401) {
    sessionErrors.add(1);
    sessionReady = false;   // session expired (e.g. container restarted)
    sleep(1);
    return;
  }

  let pairData;
  try { pairData = pairRes.json(); } catch { sleep(2); return; }

  if (pairData.done) {
    pairQueueEmpty.add(1);
    sleep(5);
    return;
  }

  check(pairData, {
    "pair has pair_id": (d) => typeof d.pair_id === "string" && d.pair_id.length > 0,
    "pair done=false":  (d) => d.done === false,
  });

  // ── 4. Fetch audio when running the full remote workflow ─────────────────
  if (SIM_FETCH_AUDIO) {
    const audioStarted = Date.now();
    const audioRes = http.get(
      `${baseUrl}/audio/${pairData.pair_id}`,
      requestParams("audio", { responseType: "binary" })
    );
    audioDuration.add(Date.now() - audioStarted);
    const audioOk = check(audioRes, {
      "audio 200": (r) => r.status === 200,
      "audio WAV": (r) => (r.headers["Content-Type"] || "").includes("audio/wav"),
      "audio non-empty": (r) => r.body && r.body.byteLength > 44,
    });
    if (!audioOk) {
      const errorCode = audioRes.error_code || "none";
      const errorText = audioRes.error || "none";
      audioErrors.add(1, {
        status: String(audioRes.status),
        error_code: String(errorCode),
      });
      console.error(
        `[VU ${__VU}] Audio failed pair=${pairData.pair_id} ` +
        `status=${audioRes.status} error_code=${errorCode} error=${errorText}`
      );
      return;
    }
    _checkDbLock(audioRes, "audio");
  }

  // ── 5. Think time (annotator decides — shorter than real-use to stress DB) ─
  sleep(1 + Math.random() * 2);   // 1–3 s

  // ── 6. Submit annotation ──────────────────────────────────────────────────
  const prefs = ["a", "b", "equal"];
  const pref  = prefs[Math.floor(Math.random() * prefs.length)];
  const corr  = Math.random() < 0.25 ? "Poprawiona transkrypcja testowa." : "";

  const t0 = Date.now();
  const submitRes = http.post(
    `${baseUrl}/api/submit`,
    { pair_id: pairData.pair_id, preference: pref, correction_text: corr, strength: "5" },
    requestParams("submit")
  );
  submitDuration.add(Date.now() - t0);

  const submitOk = check(submitRes, {
    "submit 200":   (r) => r.status === 200,
    "submit ok":    (r) => { try { return r.json().ok === true; } catch { return false; } },
  });
  if (submitOk) {
    successfulSubmissions.add(1);
  } else {
    submitErrors.add(1);
  }
  _checkDbLock(submitRes, "submit");
  pairCount++;

  // ── 7. Stats refresh (as the frontend does after each submit) ────────────
  const statsRes = http.get(`${baseUrl}/api/stats`, requestParams("stats"));
  check(statsRes, { "stats 200": (r) => r.status === 200 });
  _checkDbLock(statsRes, "stats");

  sleep(0.2);
}
