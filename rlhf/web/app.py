"""
web/app.py -- FastAPI web annotation interface.
Serves the annotation UI on port 8080.

Usage (direct):
    uvicorn web.app:app --host 0.0.0.0 --port 8080

Usage (Docker):
    docker compose up web

Inline coordinator (local dev without the coordinator Docker service):
    Set INLINE_COORDINATOR=1 before starting the web app.  The web process
    will then poll the DB and run the training pipeline itself when the admin
    presses "Trigger Training Now" or the auto-threshold fires.

    INLINE_COORDINATOR=1 uvicorn web.app:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import math
import os
import re
import secrets
import shutil
import sqlite3
import sys
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

# Ensure project root is on sys.path when running from the web/ directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from diagnostics.artifacts import bundle_path, list_runs
from paths import DEFAULT_PATHS, PROJECT_ROOT
from web.corpus_import import (
    accept_stage, cancel_stage, preview_stage, stage_upload,
)
from storage.feedback_store import FeedbackStore
from storage.schema import (
    Annotation,
    AnnotationSession,
    CanarySet,
    RewardAuditItem,
    RewardAuditJob,
)
from utils.audio import cache_normalized_wav, load_audio
from utils.log import get_logger, setup_logging
from utils.model_publisher import publish_latest_model, read_publication
from utils.bridge_auth import ensure_bridge_token
from web.publication_api import router as publication_router
from utils.structured_events import StructuredEventWriter
from utils.admin_password import (
    MIN_PASSWORD_LENGTH, PasswordConfig, load_password_config, password_config_path,
    password_fingerprint, save_password_config, verify_password,
)

setup_logging(logs_dir=DEFAULT_PATHS.logs_dir)
log = get_logger(__name__)
_event_writer: StructuredEventWriter | None = None
_CORPUS_UPLOAD_ROOT = PROJECT_ROOT / "_data" / "corpus_uploads"

# ---------------------------------------------------------------------------
# Background cleanup: release pair reservations that timed out
# ---------------------------------------------------------------------------

_RESERVATION_SECONDS = int(os.environ.get("PAIR_RESERVATION_SECONDS", "300"))  # 5 min
_CLEANUP_INTERVAL_SECONDS = 60
_SESSION_TIMEOUT_SECONDS = 120  # sessions not seen within 2 min are considered inactive
_QUEUE_REFILL_POLL_INTERVAL = int(os.environ.get("QUEUE_REFILL_POLL_INTERVAL", "60"))

# ---------------------------------------------------------------------------
# Password-based admin authentication
# ---------------------------------------------------------------------------

_ADMIN_CFG_PATH = str(password_config_path())


def _load_password_config() -> PasswordConfig | None:
    try:
        return load_password_config(Path(_ADMIN_CFG_PATH))
    except (OSError, ValueError) as exc:
        log.error("Admin password configuration could not be loaded: %s", exc)
        raise HTTPException(status_code=503, detail="Admin password configuration unavailable") from exc


def _verify_password(password: str) -> bool:
    return verify_password(password, _load_password_config())


def _is_admin(request: Request) -> bool:
    session_cookie = request.cookies.get("admin_session", "")
    if not session_cookie:
        return False
    stored = _get_store().get_setting("admin_session_id")
    if not stored or not secrets.compare_digest(session_cookie.encode("utf-8"), stored.encode("utf-8")):
        return False
    fingerprint = _get_store().get_setting("admin_session_password_fingerprint", "")
    return secrets.compare_digest(fingerprint, password_fingerprint(_load_password_config()))


async def _reservation_cleanup_loop() -> None:
    """Periodically release expired pair reservations and stale sessions."""
    while True:
        await asyncio.sleep(_CLEANUP_INTERVAL_SECONDS)
        try:
            store = _get_store()
            released = store.release_expired_reservations()
            if released:
                log.info(
                    "Released %d expired pair reservation(s) back to queue", released
                )
            stale = store.purge_stale_sessions(timeout_s=_SESSION_TIMEOUT_SECONDS)
            if stale:
                log.info("Purged %d stale session(s)", stale)
        except Exception as exc:
            log.warning("Cleanup loop error: %s", exc)


# ---------------------------------------------------------------------------
# Inline coordinator (activated by INLINE_COORDINATOR=1 env var)
# ---------------------------------------------------------------------------
# When the dedicated coordinator Docker service is NOT running (e.g. during
# local development), set INLINE_COORDINATOR=1 to have the web process handle
# training triggers itself.  Pipeline steps run as subprocesses so the event
# loop stays unblocked and annotation remains available during training.

_INLINE_COORDINATOR = os.environ.get("INLINE_COORDINATOR", "0") == "1"
_INLINE_POLL_INTERVAL = int(os.environ.get("COORDINATOR_POLL_INTERVAL", "30"))
_INLINE_STEP_TIMEOUT = 14400  # 4 hours per step (same as coordinator.py)

# Use the same Python that launched the web app so it works on any machine.
_PYTHON_EXE = sys.executable
# Project root is two levels above this file (web/app.py → web/ → project root).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Reuse the canonical pipeline definition from coordinator so both codepaths
# stay in sync automatically. Importing coordinator sets its module-level
# env-var globals (SKIP_GENERATE, GENERATE_N, GENERATE_DATASETS) from the
# same environment, so the pipeline is identical in both modes.
from scripts.coordinator import (  # noqa: E402
    _find_latest_rm_checkpoint,
    _pipeline as _inline_pipeline_steps,
    is_policy_gate_rejection,
    normalize_legacy_gate_rejection,
    training_trigger_progress,
)
from utils.policy_checkpoint import (  # noqa: E402
    resolve_active_policy_checkpoint,
    rollback_active_policy_checkpoint,
)
from utils.training_tuning import TrainingTuning  # noqa: E402


async def _inline_run_pipeline(store: FeedbackStore) -> bool:
    """Run the training pipeline in-process (as subprocesses). Returns True on success."""
    store.set_setting("training_status", "running")
    store.set_setting("training_requested", "false")
    store.set_setting("training_error", "")
    log.info("[inline-coordinator] Training pipeline starting.")
    try:
        policy_checkpoint = resolve_active_policy_checkpoint(
            store, Path(DEFAULT_PATHS.checkpoints_dir), Path(_PROJECT_ROOT),
            base_model_id=DEFAULT_PATHS.whisper_model_id,
        )
        include_sft = bool(store.get_sft_training_rows())
        tuning = TrainingTuning.from_store(store)
        round_id = datetime.now().strftime("v%Y%m%d_%H%M%S_%f")
        store.set_setting("training_sft_included", "true" if include_sft else "false")
        for stage in ("sft", "synthetic", "dpo"):
            store.set_setting(f"training_{stage}_diagnostics", "")
        steps = _inline_pipeline_steps(
            policy_checkpoint,
            include_sft=include_sft,
            round_id=round_id,
            tuning=tuning,
        )
        total = len(steps)
        store.set_setting("training_step_total", str(total))
        store.set_setting("training_step", "")
        store.set_setting("training_step_n", "0")
        for n, (cmd, description) in enumerate(steps, 1):
            store.set_setting("training_step", description)
            store.set_setting("training_step_n", str(n))
            store.set_setting("training_step_log", "")
            log.info(
                "[inline-coordinator] Step %d/%d: %s  |  cmd: %s",
                n,
                total,
                description,
                " ".join(cmd),
            )
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=_PROJECT_ROOT,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            tail_lines: list[str] = []

            async def _stream_with_tail(p=proc, s=store):
                async for raw in p.stdout:
                    line = raw.decode("utf-8", errors="replace").rstrip()
                    if line:
                        log.info("[inline-coordinator] | %s", line)
                        tail_lines.append(line)
                        if len(tail_lines) > 15:
                            tail_lines.pop(0)
                        try:
                            s.set_setting("training_step_log", line[:300])
                        except Exception:
                            pass

            stream_task = asyncio.create_task(_stream_with_tail())
            timed_out = False
            try:
                await asyncio.wait_for(proc.wait(), timeout=_INLINE_STEP_TIMEOUT)
            except asyncio.TimeoutError:
                timed_out = True
                proc.kill()
                await proc.wait()
            try:
                await asyncio.wait_for(stream_task, timeout=10)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                stream_task.cancel()
            if timed_out:
                log.error("[inline-coordinator] Step '%s' timed out.", description)
                store.set_setting("training_status", "failed")
                store.set_setting(
                    "training_error",
                    f"Step '{description}' timed out after {_INLINE_STEP_TIMEOUT} s.",
                )
                return False
            if is_policy_gate_rejection(description, proc.returncode):
                log.warning(
                    "[inline-coordinator] Policy candidate blocked by the WER gate; "
                    "active policy unchanged."
                )
                store.set_setting("training_status", "guarded")
                store.set_setting("training_error", "")
                shutil.rmtree(
                    Path(_PROJECT_ROOT) / "_outputs" / "tmp" / "dpo_rounds" / round_id,
                    ignore_errors=True,
                )
                return True
            if proc.returncode != 0:
                log.error(
                    "[inline-coordinator] Step '%s' failed (exit %d).",
                    description,
                    proc.returncode,
                )
                store.set_setting("training_status", "failed")
                store.set_setting(
                    "training_error",
                    f"Step '{description}' failed (exit code {proc.returncode}).\n\n"
                    + "\n".join(tail_lines),
                )
                # Leave training_step/n set so admin can see which step failed.
                return False
        store.set_setting("training_status", "idle")
        store.set_setting("training_step", "")
        store.set_setting("training_step_n", "0")
        store.set_setting("training_error", "")
        shutil.rmtree(
            Path(_PROJECT_ROOT) / "_outputs" / "tmp" / "dpo_rounds" / round_id,
            ignore_errors=True,
        )
        log.info("[inline-coordinator] Training pipeline completed successfully.")
        return True
    except Exception as exc:
        log.error("[inline-coordinator] Unexpected error: %s", exc, exc_info=True)
        store.set_setting("training_status", "failed")
        return False


async def _queue_refill_loop() -> None:
    """Periodically signal that the pending queue needs topping up.

    Sets queue_refill_requested=true in the DB when pending pairs drop below
    queue_min_size.  The actual pair generation is performed by whoever has
    access to the GPU:
      - External Docker mode  : the coordinator service polls the flag.
      - INLINE_COORDINATOR=1  : the inline coordinator loop in this process.
    The web process never spawns ML subprocesses so the web image stays slim.
    """
    while True:
        await asyncio.sleep(_QUEUE_REFILL_POLL_INTERVAL)
        try:
            store = _get_store()

            min_size_str = store.get_setting("queue_min_size", "20")
            min_size = int(min_size_str) if min_size_str.isdigit() else 20
            if min_size == 0:
                continue  # feature disabled by admin

            # Keep GPU generation and training mutually exclusive.
            if store.get_setting("training_status", "idle") == "running":
                continue

            # Already signalled or running
            if store.get_setting("queue_refill_requested", "false") == "true":
                continue
            if store.get_setting("queue_refill_status", "idle") in (
                "running",
                "queued",
            ):
                continue

            pending = store.count_pending_pairs()
            if pending >= min_size:
                continue

            batch_str = store.get_setting("queue_refill_batch", "50")
            batch = int(batch_str) if batch_str.isdigit() else 50
            log.info(
                "[queue-refill] Pending %d < threshold %d — requesting %d new pairs.",
                pending,
                min_size,
                batch,
            )
            store.set_setting("queue_refill_requested", "true")
            store.set_setting("queue_refill_status", "queued")
            store.set_setting(
                "queue_refill_log",
                f"Queued: pending={pending}, will generate {batch} pairs",
            )

        except Exception as exc:
            log.error("[queue-refill] Poll error: %s", exc, exc_info=True)


async def _run_refill_generate(store: FeedbackStore) -> None:
    """Actually spawn run_generate_pairs.py.  Only called from coordinator paths
    (inline coordinator loop or coordinator.py) that run in the GPU environment."""
    try:
        batch_str = store.get_setting("queue_refill_batch", "50")
        batch = int(batch_str) if batch_str.isdigit() else 50
        pending = store.count_pending_pairs()
        log.info(
            "[queue-refill] Running generate: --n %d (pending=%d).", batch, pending
        )
        store.set_setting("queue_refill_status", "running")
        store.set_setting("queue_refill_log", f"Running… pending={pending}")
        cmd = [_PYTHON_EXE, "run_generate_pairs.py", "--n", str(batch)]
        rm_checkpoint = _find_latest_rm_checkpoint()
        if rm_checkpoint is not None:
            cmd += ["--rm_checkpoint", str(rm_checkpoint)]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=_PROJECT_ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        last_line = ""
        async for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line:
                log.info("[queue-refill] | %s", line)
                last_line = line
                try:
                    store.set_setting("queue_refill_log", line[:300])
                except Exception:
                    pass
        await proc.wait()
        if proc.returncode == 0:
            new_pending = store.count_pending_pairs()
            added = max(0, new_pending - pending)
            store.set_setting("queue_refill_status", "idle")
            store.set_setting(
                "queue_refill_log", f"+{added} pairs added (queue: {new_pending})"
            )
            log.info(
                "[queue-refill] Done. +%d pairs, queue now at %d.", added, new_pending
            )
        else:
            store.set_setting("queue_refill_status", "failed")
            store.set_setting(
                "queue_refill_log", last_line[:300] if last_line else "(no output)"
            )
            log.error(
                "[queue-refill] run_generate_pairs.py exited %d.", proc.returncode
            )
    except Exception as exc:
        log.error("[queue-refill] Generate error: %s", exc, exc_info=True)
        try:
            store.set_setting("queue_refill_status", "failed")
        except Exception:
            pass


async def _inline_ensure_data_prepared() -> None:
    """Auto-run prepare_data.py on first boot if manifests are missing but a source ZIP is present.

    Mirrors scripts/coordinator.py::_ensure_data_prepared(). _data/ is gitignored
    and machine-local, so manifests never travel with a git pull; without this,
    generate/refill silently find 0 audio files every cycle.
    """
    from paths import DEFAULT_PATHS as _paths

    train_csv = _paths.admedvoice_train_csv
    if train_csv and os.path.exists(train_csv):
        return  # already prepared

    archive = os.path.join(_PROJECT_ROOT, "_data", "archive.zip")
    if not os.path.exists(archive):
        log.warning(
            "[inline-coordinator] No manifests found and _data/archive.zip is missing -- "
            "skipping auto-prepare. Place the source ZIP in _data/ and restart, or run "
            "'python prepare_data.py' manually."
        )
        return

    log.info(
        "[inline-coordinator] Manifests missing but _data/archive.zip found -- "
        "running prepare_data.py automatically."
    )
    proc = await asyncio.create_subprocess_exec(
        _PYTHON_EXE,
        "prepare_data.py",
        "--datasets",
        "admedvoice",
        cwd=_PROJECT_ROOT,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    async for raw in proc.stdout:
        line = raw.decode("utf-8", errors="replace").rstrip()
        if line:
            log.info("[inline-coordinator] | %s", line)
    await proc.wait()
    if proc.returncode == 0:
        log.info(
            "[inline-coordinator] Auto-prepare complete -- manifests written to _data/manifests/."
        )
    else:
        log.error(
            "[inline-coordinator] Auto-prepare failed (exit %d). Run 'python prepare_data.py' "
            "manually to see full output.",
            proc.returncode,
        )


async def _inline_coordinator_loop() -> None:
    """Asyncio background task that mirrors coordinator.py logic."""
    log.info(
        "[inline-coordinator] Active (poll interval: %d s). "
        "Use 'Trigger Training Now' in the admin panel or wait for the "
        "auto-threshold to fire.",
        _INLINE_POLL_INTERVAL,
    )
    store = _get_store()
    normalize_legacy_gate_rejection(store)
    store.fail_running_reward_audit_jobs(
        "Inline coordinator restarted before scoring completed"
    )
    await _inline_ensure_data_prepared()
    # Reset stale 'running' status from a previously crashed run.
    if store.get_setting("training_status") == "running":
        log.warning(
            "[inline-coordinator] Stale 'running' status found -- resetting to 'failed'."
        )
        store.set_setting("training_status", "failed")

    while True:
        await asyncio.sleep(_INLINE_POLL_INTERVAL)
        try:
            status = store.get_setting("training_status", "idle")
            if status == "running":
                continue

            # --- Full training pipeline ---------------------------------------
            triggered = False
            reason = ""

            if store.get_setting("training_requested", "false") == "true":
                triggered = True
                reason = "manual request from admin panel"
            else:
                new_ann, threshold = training_trigger_progress(store)
                if new_ann >= threshold:
                    triggered = True
                    reason = (
                        f"auto-threshold ({new_ann}/{threshold} actionable annotations)"
                    )

            if triggered:
                log.info("[inline-coordinator] Training triggered: %s", reason)
                stats = store.get_stats()
                success = await _inline_run_pipeline(store)
                if success:
                    store.set_setting(
                        "annotations_at_last_training",
                        str(stats.get("actionable_annotations", 0)),
                    )
                    store.set_setting("training_failed_at_annotations", "0")
                else:
                    store.set_setting(
                        "training_failed_at_annotations",
                        str(stats.get("actionable_annotations", 0)),
                    )
                continue

            # --- Model evaluation -------------------------------------------
            if store.get_setting("wer_monitor_evaluation_requested", "false") == "true":
                canary_set_id = store.get_setting(
                    "wer_monitor_evaluation_canary_set_id", ""
                )
                store.set_setting("wer_monitor_evaluation_requested", "false")
                store.set_setting("wer_monitor_evaluation_status", "running")
                store.set_setting("wer_monitor_evaluation_log", "Starting evaluation")
                proc = await asyncio.create_subprocess_exec(
                    _PYTHON_EXE,
                    "run_evaluate_canary.py",
                    "--if-ready",
                    *(["--canary-set-id", canary_set_id] if canary_set_id else []),
                    cwd=_PROJECT_ROOT,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                tail = ""
                async for raw in proc.stdout:
                    tail = raw.decode("utf-8", errors="replace").rstrip()
                    if tail:
                        store.set_setting("wer_monitor_evaluation_log", tail[:300])
                await proc.wait()
                store.set_setting(
                    "wer_monitor_evaluation_status",
                    "completed" if proc.returncode == 0 else "failed",
                )
                if proc.returncode == 0:
                    store.set_setting(
                        "wer_monitor_evaluation_log", "Evaluation completed"
                    )
                continue

            # --- Reward Audit scoring --------------------------------------
            if store.has_queued_reward_audit_job():
                proc = await asyncio.create_subprocess_exec(
                    _PYTHON_EXE,
                    "run_reward_audit.py",
                    cwd=_PROJECT_ROOT,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                async for raw in proc.stdout:
                    line = raw.decode("utf-8", errors="replace").rstrip()
                    if line:
                        log.info("[inline-coordinator] [reward-audit] %s", line)
                await proc.wait()
                continue

            # --- Queue refill (generate-only, no full pipeline) ---------------
            if (
                store.get_setting("queue_refill_requested", "false") == "true"
                and store.get_setting("queue_refill_status", "idle") != "running"
            ):
                store.set_setting("queue_refill_requested", "false")
                log.info("[inline-coordinator] Queue refill triggered.")
                asyncio.create_task(_run_refill_generate(store))
        except Exception as exc:
            log.error("[inline-coordinator] Poll error: %s", exc, exc_info=True)


@asynccontextmanager
async def _lifespan(app_: object):
    global _event_writer
    ensure_bridge_token()
    store = _get_store()
    store.fail_interrupted_audio_quality()
    quality_stop = asyncio.Event()
    quality_task = asyncio.create_task(_quality_request_loop(store, quality_stop))
    tts_task = asyncio.create_task(_tts_request_loop(store, quality_stop))
    cfg = _load_password_config()
    _event_writer = StructuredEventWriter(
        Path(DEFAULT_PATHS.logs_dir) / "web_events.jsonl"
    )
    asyncio.create_task(_reservation_cleanup_loop())
    asyncio.create_task(_queue_refill_loop())
    if _INLINE_COORDINATOR:
        asyncio.create_task(_inline_coordinator_loop())
    if cfg is None:
        log.warning(
            "The public bootstrap admin password is in use. Change it at "
            "/admin/password before exposing this deployment beyond localhost."
        )
    else:
        log.info("Admin password configured. Visit /admin to log in.")
    if _INLINE_COORDINATOR:
        log.info(
            "INLINE_COORDINATOR=1 -- training triggered from admin panel runs here."
        )
    log.info(
        "Pair reservation window: %d s  |  cleanup interval: %d s",
        _RESERVATION_SECONDS,
        _CLEANUP_INTERVAL_SECONDS,
    )
    try:
        yield
    finally:
        quality_stop.set()
        await asyncio.gather(quality_task, tts_task)
        _event_writer.close()
        _event_writer = None


async def _tts_request_loop(store: FeedbackStore, stop: asyncio.Event):
    """Dispatch persisted campaigns without GPU training or ML subprocesses."""
    from tts.worker import TTSWorker

    worker = TTSWorker(store, Path(DEFAULT_PATHS.output_dir) / "tts_audio")
    while not stop.is_set():
        worked = False
        try:
            worked = await asyncio.to_thread(worker.run_once)
        except (OSError, ValueError, RuntimeError, sqlite3.Error):
            log.exception("TTS campaign dispatcher failed; persisted work retained")
        try:
            await asyncio.wait_for(stop.wait(), timeout=1 if worked else 5)
        except asyncio.TimeoutError:
            pass


async def _quality_request_loop(store: FeedbackStore, stop: asyncio.Event):
    """CPU HTTP client only; quality inference stays in the separate ASR service."""
    from tts.quality import run_quality_once

    while not stop.is_set():
        try:
            await asyncio.to_thread(run_quality_once, store)
        except (OSError, ValueError, RuntimeError, sqlite3.Error):
            log.exception("Quality request dispatcher failed")
        try:
            await asyncio.wait_for(stop.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass


# ---------------------------------------------------------------------------
# Path translation
# ---------------------------------------------------------------------------

# Inside the container _data/ is always mounted at /app/_data/.
# DBs populated on a different host (e.g. Windows) store absolute paths
# for that host. Preserve local/project-relative archives; only legacy
# unavailable host paths are remapped by basename.
_CONTAINER_DATA_DIR = os.environ.get("RLHF_DATA_DIR", "/app/_data")


def _container_path(path: str) -> str:
    """Remap a host-absolute zip: path to the container-local _data/ mount.

    zip:C:/Users/foo/_data/archive.zip!inner/file.wav
    -> zip:/app/_data/archive.zip!inner/file.wav
    """
    if not path.startswith("zip:"):
        return path
    rest = path[4:]
    if "!" not in rest:
        return path
    zip_part, inner = rest.split("!", 1)
    if Path(zip_part).is_file():
        return path
    if not Path(zip_part).is_absolute() and not re.match(r"^[A-Za-z]:[/\\]", zip_part):
        archive = Path(_PROJECT_ROOT) / zip_part
        if not archive.is_file():
            relative = zip_part.replace("\\", "/")
            if relative.startswith("_data/"):
                archive = Path(_CONTAINER_DATA_DIR) / relative[len("_data/"):]
            elif "/" not in relative:
                archive = Path(_CONTAINER_DATA_DIR) / relative
        if not archive.is_file():
            raise FileNotFoundError(f"Audio archive not found: {archive}")
        return f"zip:{archive}!{inner}"
    zip_name = os.path.basename(zip_part.replace("\\", "/"))
    return f"zip:{_CONTAINER_DATA_DIR}/{zip_name}!{inner}"


def _active_policy_view(store: FeedbackStore) -> dict | None:
    checkpoint = resolve_active_policy_checkpoint(
        store, Path(DEFAULT_PATHS.checkpoints_dir), Path(_PROJECT_ROOT),
        base_model_id=DEFAULT_PATHS.whisper_model_id,
    )
    if checkpoint is None:
        return None
    return {
        "path": str(checkpoint.resolve()),
        "name": checkpoint.name,
    }


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="RLHF for ASR Annotation", lifespan=_lifespan)
app.include_router(publication_router)


@app.middleware("http")
async def record_request_event(request: Request, call_next):
    started = time.perf_counter()
    status = 500
    error_type = None
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    except Exception as exc:
        error_type = type(exc).__name__
        raise
    finally:
        route = request.scope.get("route")
        route_template = getattr(route, "path", "unmatched")
        raw_run_id = request.headers.get("X-Diagnostics-Run-ID", "")
        try:
            parsed_run_id = uuid.UUID(raw_run_id) if raw_run_id else None
            run_id = (
                str(parsed_run_id)
                if parsed_run_id is not None and parsed_run_id.version == 4
                else None
            )
        except ValueError:
            run_id = None
        if _event_writer is not None:
            _event_writer.emit(
                "http_request",
                severity="error" if status >= 500 else "info",
                route=route_template,
                method=request.method,
                status=status,
                duration_ms=round((time.perf_counter() - started) * 1000, 3),
                run_id=run_id,
                error_type=error_type,
            )

_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(_DIR, "templates"))

# Lazy singleton -- created on first request so startup doesn't fail if
# _outputs/ doesn't exist yet.
_store: FeedbackStore | None = None


def _get_store() -> FeedbackStore:
    global _store
    if _store is None:
        _store = FeedbackStore(DEFAULT_PATHS.db_path)
    return _store


def _request_diagnostics(action: str, **fields: object) -> str:
    request_id = str(uuid.uuid4())
    request = {
        "request_id": request_id,
        "action": action,
        "requested_at": datetime.utcnow().isoformat() + "Z",
        **fields,
    }
    _get_store().set_setting("diagnostics_request", json.dumps(request))
    return request_id


def _diagnostics_pending_action(store: FeedbackStore, status: dict) -> str | None:
    raw_request = store.get_setting("diagnostics_request", "")
    try:
        command = json.loads(raw_request) if raw_request else {}
    except json.JSONDecodeError:
        return None
    request_id = str(command.get("request_id", ""))
    if request_id and request_id != str(status.get("last_request_id", "")):
        return str(command.get("action", "request"))
    return None


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return RedirectResponse(url="/setup")


@app.get("/setup", response_class=HTMLResponse)
async def setup_page(request: Request):
    return templates.TemplateResponse(
        request, "setup.html", {"error": None}
    )


def _start_annotation_session(request: Request, participant_id: str):
    store = _get_store()
    try:
        max_sessions = int(store.get_setting("max_concurrent_sessions", "10"))
    except ValueError:
        max_sessions = 10
    active = store.count_active_sessions(timeout_s=_SESSION_TIMEOUT_SECONDS)
    if active >= max_sessions:
        return templates.TemplateResponse(
            request,
            "setup.html",
            {
                "error": (
                    f"Osiągnięto limit aktywnych adnotatorów ({active}/{max_sessions}). "
                    "Spróbuj ponownie za kilka minut. / "
                    f"Server is at capacity ({active}/{max_sessions} active annotators). "
                    "Please try again in a few minutes."
                ),
            },
        )
    session = AnnotationSession.new(
        participant_id=participant_id,
        machine_id="web",
    )
    store.open_session(session)
    store.upsert_session(session.id, participant_id)
    resp = RedirectResponse(url="/annotate", status_code=303)
    resp.set_cookie("session_id", session.id, httponly=True)
    resp.set_cookie("participant_id", participant_id)
    return resp


@app.post("/setup")
async def setup_submit(
    request: Request,
    participant_id: str = Form(...),
):
    if not participant_id.strip():
        return templates.TemplateResponse(
            request,
            "setup.html",
            {
                "error": "Identyfikator uczestnika jest wymagany. / Participant ID is required.",
            },
        )
    return _start_annotation_session(request, participant_id.strip())


@app.get("/annotate", response_class=HTMLResponse)
async def annotate_page(request: Request):
    session_id = request.cookies.get("session_id")
    store = _get_store()
    if not session_id or not store.session_exists(session_id):
        return RedirectResponse(url="/setup")
    stats = store.get_stats()
    return templates.TemplateResponse(
        request,
        "annotate.html",
        {
            "stats": stats,
            "participant_id": request.cookies.get("participant_id", "unknown"),
            "show_reference": store.get_setting("show_reference", "true") == "true",
            "show_wer": store.get_setting("show_wer", "true") == "true",
        },
    )


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@app.post("/api/heartbeat")
async def heartbeat(request: Request):
    """Keep-alive ping from the annotate page. Refreshes the active session."""
    session_id = request.cookies.get("session_id")
    if not session_id:
        return JSONResponse({"ok": False})
    participant_id = request.cookies.get("participant_id", "")
    _get_store().upsert_session(session_id, participant_id)
    return JSONResponse({"ok": True})


@app.get("/api/pair")
async def get_next_pair(request: Request):
    """Atomically return and reserve the next pending pair, or {done: true}."""
    session_id = request.cookies.get("session_id")
    if not session_id:
        raise HTTPException(
            status_code=401, detail="No active session. Visit /setup first."
        )
    store = _get_store()
    pair = store.get_and_reserve_pair(duration_s=_RESERVATION_SECONDS)
    if pair is None:
        return JSONResponse(
            {
                "done": True,
                "queue_refill_status": store.get_setting("queue_refill_status", "idle"),
            }
        )
    af = store.get_audio_file(pair.audio_id) if pair.audio_id else None
    return JSONResponse(
        {
            "done": False,
            "pair_id": pair.id,
            "audio_id": pair.audio_id,
            "hyp_a": pair.hyp_a,
            "hyp_b": pair.hyp_b,
            "model_version": pair.model_version,
            "reference": af.reference_transcript if af else "",
            "file_path": af.file_path if af else "",
        }
    )


def _cached_audio_wav(file_path: str) -> Path:
    resolved = _container_path(file_path)
    return cache_normalized_wav(
        resolved,
        Path(DEFAULT_PATHS.output_dir) / "audio_cache",
    )


@app.get("/audio/{pair_id}")
async def serve_audio(pair_id: str):
    """Stream the audio for a given pair as WAV."""
    store = _get_store()
    audio_id = store.get_audio_id_for_pair(pair_id)
    if not audio_id:
        raise HTTPException(status_code=404, detail="Pair not found")
    af = store.get_audio_file(audio_id)
    if not af:
        raise HTTPException(status_code=404, detail="Audio file not found")
    try:
        cached_path = await asyncio.to_thread(_cached_audio_wav, af.file_path)
        return FileResponse(cached_path, media_type="audio/wav")
    except Exception as exc:
        log.error("Audio load failed for %s: %s", af.file_path, exc)
        raise HTTPException(status_code=500, detail=f"Audio load error: {exc}")


@app.post("/api/submit")
async def submit_annotation(
    request: Request,
    pair_id: str = Form(...),
    preference: str = Form(default="equal"),
    correction_text: str = Form(default=""),
    strength: float = Form(default=5.0),
    derive_preference: bool = Form(default=True),
):
    session_id = request.cookies.get("session_id")
    if not session_id:
        raise HTTPException(status_code=401, detail="No active session.")
    store = _get_store()
    if not store.session_exists(session_id):
        raise HTTPException(
            status_code=401,
            detail="Session expired. Please visit /setup to start a new session.",
        )

    correction = correction_text.strip()

    # Auto-derive preference from WER when annotator typed a correction but
    # left preference as "equal".  The correction is ground truth; whichever
    # hypothesis is closer (lower WER) is the better one.
    if derive_preference and preference == "equal" and correction:
        pair = store.get_pair_by_id(pair_id)
        if pair and pair.hyp_a and pair.hyp_b:
            try:
                import jiwer

                wer_a = jiwer.wer(correction, pair.hyp_a)
                wer_b = jiwer.wer(correction, pair.hyp_b)
                gap = abs(wer_a - wer_b)
                if gap > 0.05:
                    preference = "a" if wer_a < wer_b else "b"
                    log.info(
                        "Auto-preference from correction: %s  "
                        "WER_a=%.2f  WER_b=%.2f  gap=%.2f  pair=%s",
                        preference,
                        wer_a,
                        wer_b,
                        gap,
                        pair_id,
                    )
            except Exception as exc:
                log.warning("WER auto-preference failed for pair %s: %s", pair_id, exc)

    annotation = Annotation.new(
        session_id=session_id,
        pair_id=pair_id,
        preference=preference,
        preference_strength=strength / 10.0,  # normalize 0-10 -> 0-1
        correction_text=correction or None,
    )
    store.insert_annotation(annotation)
    stats = store.get_stats()
    return JSONResponse({"ok": True, "stats": stats})


@app.post("/api/skip")
async def skip_pair(
    request: Request,
    pair_id: str = Form(...),
):
    session_id = request.cookies.get("session_id")
    if not session_id:
        raise HTTPException(status_code=401, detail="No active session.")
    store = _get_store()
    if not store.session_exists(session_id):
        raise HTTPException(
            status_code=401,
            detail="Session expired. Please visit /setup to start a new session.",
        )
    annotation = Annotation.new(
        session_id=session_id,
        pair_id=pair_id,
        preference="skip",
    )
    store.insert_annotation(annotation)
    return JSONResponse({"ok": True})


@app.post("/api/exclude")
async def exclude_audio(
    request: Request,
    audio_id: str = Form(...),
    reason: str = Form(default="noise"),
):
    session_id = request.cookies.get("session_id")
    store = _get_store()
    if not session_id or not store.session_exists(session_id):
        raise HTTPException(status_code=401, detail="No active session.")
    actor = request.cookies.get("participant_id", "").strip()
    if not actor:
        raise HTTPException(status_code=401, detail="Annotator identity is missing.")
    store.exclude_audio_file(audio_id, reason.strip() or "noise", actor=actor)
    return JSONResponse({"ok": True})


@app.get("/api/stats")
async def get_stats():
    return JSONResponse(_get_store().get_stats())


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------


@app.get("/admin/login", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    if _is_admin(request):
        return RedirectResponse("/admin")
    return templates.TemplateResponse(request, "admin_login.html", {"error": None})


@app.post("/admin/login")
async def admin_login_submit(request: Request):
    form = await request.form()
    password = str(form.get("password", ""))
    cfg = _load_password_config()
    if not verify_password(password, cfg):
        return templates.TemplateResponse(
            request,
            "admin_login.html",
            {"error": "Nieprawidłowe hasło. / Invalid password."},
        )
    session_id = secrets.token_urlsafe(32)
    _get_store().set_setting("admin_session_id", session_id)
    _get_store().set_setting(
        "admin_session_password_fingerprint", password_fingerprint(cfg)
    )
    resp = RedirectResponse("/admin", status_code=303)
    resp.set_cookie("admin_session", session_id, httponly=True, samesite="strict")
    return resp


@app.get("/admin/password", response_class=HTMLResponse)
async def admin_password_page(request: Request):
    if not _is_admin(request):
        return RedirectResponse("/admin/login")
    return templates.TemplateResponse(request, "admin_login.html", {
        "error": None, "changing_password": True, "min_password_length": MIN_PASSWORD_LENGTH,
    })


@app.post("/admin/password")
async def admin_password_submit(request: Request):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    form = await request.form()
    password = str(form.get("new_password", ""))
    error = None
    if not _verify_password(str(form.get("current_password", ""))):
        error = "Nieprawidłowe bieżące hasło. / Invalid current password."
    elif password != str(form.get("confirm_password", "")):
        error = "Hasła nie są zgodne. / Passwords do not match."
    elif len(password) < MIN_PASSWORD_LENGTH or not password.strip():
        error = f"Hasło musi mieć co najmniej {MIN_PASSWORD_LENGTH} znaków. / Password must have at least {MIN_PASSWORD_LENGTH} characters."
    if error:
        return templates.TemplateResponse(request, "admin_login.html", {
            "error": error, "changing_password": True,
            "min_password_length": MIN_PASSWORD_LENGTH,
        }, status_code=400)
    try:
        await asyncio.to_thread(save_password_config, Path(_ADMIN_CFG_PATH), password)
    except OSError as exc:
        log.error("Admin password could not be saved: %s", exc)
        raise HTTPException(status_code=500, detail="Admin password could not be saved") from exc
    return _clear_admin_session()


@app.get("/admin", response_class=HTMLResponse)
async def admin_panel(request: Request):
    if not _is_admin(request):
        return RedirectResponse("/admin/login")
    store = _get_store()
    training_step_n = store.get_setting("training_step_n", "0")
    training_step_total = store.get_setting("training_step_total", "4")
    training_step_n_int = int(training_step_n) if training_step_n.isdigit() else 0
    training_step_total_int = (
        int(training_step_total) if training_step_total.isdigit() else 4
    )
    sft_included = store.get_setting("training_sft_included", "true") == "true"
    training_tuning = TrainingTuning.from_store(store)
    training_diagnostics = {}
    for stage in ("sft", "synthetic", "dpo"):
        raw = store.get_setting(f"training_{stage}_diagnostics", "")
        try:
            training_diagnostics[stage] = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            training_diagnostics[stage] = None
    settings = {
        **training_tuning.as_settings(),
        "max_concurrent_sessions": store.get_setting("max_concurrent_sessions", "10"),
        "training_threshold": store.get_setting("training_threshold", "50"),
        "training_status": store.get_setting("training_status", "idle"),
        "training_requested": store.get_setting("training_requested", "false"),
        "training_step": store.get_setting("training_step", ""),
        "training_step_n": training_step_n_int,
        "training_step_total": training_step_total_int,
        "training_step_pct": (
            round(training_step_n_int / training_step_total_int * 100)
            if training_step_total_int
            else 0
        ),
        "training_step_log": store.get_setting("training_step_log", ""),
        "training_error": store.get_setting("training_error", ""),
        "queue_min_size": store.get_setting("queue_min_size", "20"),
        "queue_refill_batch": store.get_setting("queue_refill_batch", "50"),
        "queue_refill_status": store.get_setting("queue_refill_status", "idle"),
        "queue_refill_log": store.get_setting("queue_refill_log", ""),
        "show_reference": store.get_setting("show_reference", "true"),
        "show_wer": store.get_setting("show_wer", "true"),
        "asr_publication_status": store.get_setting("asr_publication_status", "not_ready"),
        "asr_publication_log": store.get_setting("asr_publication_log", ""),
        "asr_publication_id": store.get_setting("asr_publication_id", ""),
    }
    stats = store.get_stats()
    new_actionable, training_threshold = training_trigger_progress(store)
    training_remaining = max(0, training_threshold - new_actionable)
    return templates.TemplateResponse(
        request,
        "admin.html",
        {
            "stats": stats,
            "new_actionable": new_actionable,
            "training_remaining": training_remaining,
            "active_sessions": store.get_active_sessions(
                timeout_s=_SESSION_TIMEOUT_SECONDS
            ),
            "settings": settings,
            "active_policy": _active_policy_view(store),
            "training_diagnostics": training_diagnostics,
            "inline_coordinator": _INLINE_COORDINATOR,
            "pipeline_steps": [
                {"pl": "trening modelu nagrody", "en": "train reward model"},
                *(
                    [{"pl": "dostrajanie SFT", "en": "SFT fine-tuning"}]
                    if sft_included
                    else []
                ),
                {"pl": "generowanie danych DPO", "en": "generate synthetic DPO data"},
                {"pl": "dostrajanie DPO", "en": "DPO fine-tuning"},
                {"pl": "ocena modelu", "en": "evaluate model"},
            ],
        },
    )


@app.get("/admin/diagnostics", response_class=HTMLResponse)
async def admin_diagnostics(request: Request):
    if not _is_admin(request):
        return RedirectResponse("/admin/login")
    store = _get_store()
    raw_status = store.get_setting("diagnostics_status", "")
    try:
        status = json.loads(raw_status) if raw_status else {"state": "unavailable"}
    except json.JSONDecodeError:
        status = {"state": "error", "error": "Invalid collector status"}
    diagnostics_dir = Path(DEFAULT_PATHS.output_dir) / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(diagnostics_dir)
    return templates.TemplateResponse(
        request,
        "diagnostics.html",
        {
            "status": status,
            "pending_action": _diagnostics_pending_action(store, status),
            "requested_action": request.query_params.get("requested", ""),
            "runs": list_runs(DEFAULT_PATHS),
            "disk_free_gib": disk.free / (1024**3),
        },
    )


@app.post("/admin/diagnostics/start")
async def admin_diagnostics_start(request: Request, name: str = Form(default="")):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    raw_status = _get_store().get_setting("diagnostics_status", "")
    status = json.loads(raw_status) if raw_status else {}
    if _diagnostics_pending_action(_get_store(), status):
        raise HTTPException(status_code=409, detail="A diagnostics request is pending")
    if status.get("state") == "running":
        raise HTTPException(status_code=409, detail="Diagnostics capture is running")
    _request_diagnostics("start", run_id=str(uuid.uuid4()), name=name.strip()[:120])
    return RedirectResponse("/admin/diagnostics?requested=start", status_code=303)


@app.post("/admin/diagnostics/stop")
async def admin_diagnostics_stop(request: Request):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    raw_status = _get_store().get_setting("diagnostics_status", "")
    status = json.loads(raw_status) if raw_status else {}
    if _diagnostics_pending_action(_get_store(), status):
        raise HTTPException(status_code=409, detail="A diagnostics request is pending")
    _request_diagnostics("stop")
    return RedirectResponse("/admin/diagnostics?requested=stop", status_code=303)


@app.post("/admin/diagnostics/{run_id}/bundle/{kind}")
async def admin_diagnostics_bundle(
    request: Request,
    run_id: str,
    kind: str,
    confirm_sensitive: str = Form(default=""),
):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    if kind not in {"sanitized", "sensitive"}:
        raise HTTPException(status_code=404)
    if kind == "sensitive" and confirm_sensitive != "yes":
        raise HTTPException(status_code=400, detail="Sensitive bundle confirmation required")
    try:
        bundle_path(DEFAULT_PATHS, run_id, kind)
    except ValueError as exc:
        raise HTTPException(status_code=404) from exc
    raw_status = _get_store().get_setting("diagnostics_status", "")
    status = json.loads(raw_status) if raw_status else {}
    if _diagnostics_pending_action(_get_store(), status):
        raise HTTPException(status_code=409, detail="A diagnostics request is pending")
    _request_diagnostics("bundle", run_id=run_id, kind=kind)
    return RedirectResponse(
        f"/admin/diagnostics?requested={kind}-bundle", status_code=303
    )


@app.get("/admin/diagnostics/{run_id}/download/{kind}")
async def admin_diagnostics_download(
    request: Request,
    run_id: str,
    kind: str,
    confirm_sensitive: str = "",
):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    try:
        path = bundle_path(DEFAULT_PATHS, run_id, kind)
    except ValueError as exc:
        raise HTTPException(status_code=404) from exc
    if kind == "sensitive" and confirm_sensitive != "yes":
        raise HTTPException(status_code=400, detail="Sensitive download confirmation required")
    if not path.is_file() or path.is_symlink():
        raise HTTPException(status_code=404, detail="Diagnostics bundle is not ready")
    return FileResponse(
        path,
        filename=path.name,
        media_type="application/zip",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/admin/diagnostics/{run_id}/delete")
async def admin_diagnostics_delete(request: Request, run_id: str):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    try:
        bundle_path(DEFAULT_PATHS, run_id, "sanitized")
    except ValueError as exc:
        raise HTTPException(status_code=404) from exc
    raw_status = _get_store().get_setting("diagnostics_status", "")
    status = json.loads(raw_status) if raw_status else {}
    if _diagnostics_pending_action(_get_store(), status):
        raise HTTPException(status_code=409, detail="A diagnostics request is pending")
    _request_diagnostics("delete", run_id=run_id)
    return RedirectResponse("/admin/diagnostics?requested=delete", status_code=303)


@app.get("/admin/canary", response_class=HTMLResponse)
async def admin_canary(request: Request):
    if not _is_admin(request):
        return RedirectResponse("/admin/login")
    store = _get_store()
    all_canary_sets = store.get_canary_sets()
    show_archived = request.query_params.get("show_archived") == "1"
    canary_sets = [
        row for row in all_canary_sets if show_archived or row["status"] != "retired"
    ]
    requested_set_id = request.query_params.get("canary_set_id")
    active_set = next(
        (item for item in canary_sets if item["id"] == requested_set_id), None
    )
    if active_set is None:
        active_set = next(
            (item for item in canary_sets if item["status"] == "collecting"),
            canary_sets[0] if canary_sets else None,
        )
    items = store.get_canary_items(active_set["id"]) if active_set else []
    current_item = next((item for item in items if item["verified_at"] is None), None)
    provenance_counts = {
        provenance: sum(item["transcript_provenance"] == provenance for item in items)
        for provenance in ("manual", "manual_after_reveal", "copied_gt")
    }
    bulk_result = {
        key: request.query_params.get(key)
        for key in ("bulk_copied", "bulk_completed", "bulk_missing")
    }
    return templates.TemplateResponse(
        request,
        "canary.html",
        {
            "canary_sets": canary_sets,
            "monitor_count": len(all_canary_sets),
            "archived_count": sum(
                row["status"] == "retired" for row in all_canary_sets
            ),
            "show_archived": show_archived,
            "active_set": active_set,
            "items": items,
            "current_item": current_item,
            "provenance_counts": provenance_counts,
            "bulk_result": bulk_result,
            "evaluation_status": store.get_setting(
                "wer_monitor_evaluation_status", "idle"
            ),
            "evaluation_log": store.get_setting("wer_monitor_evaluation_log", ""),
            "promotion_gate_id": (store.get_promotion_gate_canary_set() or {}).get(
                "id"
            ),
        },
    )


def _reward_audit_checkpoints() -> list[Path]:
    base = Path(DEFAULT_PATHS.reward_model_dir)
    if not base.exists():
        return []
    return sorted(
        (
            path.resolve()
            for path in base.glob("reward_model_*")
            if path.is_dir() and (path / "mlp_head.pt").exists()
        ),
        key=lambda path: path.name,
        reverse=True,
    )


@app.get("/admin/reward-audit", response_class=HTMLResponse)
async def admin_reward_audit(request: Request):
    if not _is_admin(request):
        return RedirectResponse("/admin/login")
    store = _get_store()
    jobs = store.get_reward_audit_jobs()
    requested_job_id = request.query_params.get("job_id")
    selected_job = next(
        (job for job in jobs if job["id"] == requested_job_id),
        jobs[0] if jobs else None,
    )
    items = store.get_reward_audit_items(selected_job["id"]) if selected_job else []
    for item in items:
        item["margin"] = (
            abs(item["reward_a"] - item["reward_b"])
            if item["reward_a"] is not None and item["reward_b"] is not None
            else None
        )
        item["reward_preference"] = (
            (
                "a"
                if item["reward_a"] > item["reward_b"]
                else "b" if item["reward_b"] > item["reward_a"] else "equal"
            )
            if item["margin"] is not None
            else None
        )
        item["agreement"] = (
            item["reward_preference"] == item["preference"]
            if item["preference"] in {"a", "b"}
            and item["reward_preference"] is not None
            else None
        )
    return templates.TemplateResponse(
        request,
        "reward_audit.html",
        {
            "checkpoints": _reward_audit_checkpoints(),
            "jobs": jobs,
            "selected_job": selected_job,
            "items": items,
            "error": request.query_params.get("error"),
        },
    )


@app.post("/admin/reward-audit/create")
async def admin_create_reward_audit(request: Request):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    form = await request.form()
    checkpoint_path = str(form.get("checkpoint_path", ""))
    allowed = {str(path): path for path in _reward_audit_checkpoints()}
    checkpoint = allowed.get(checkpoint_path)
    if checkpoint is None:
        raise HTTPException(status_code=400, detail="Invalid reward checkpoint")
    selection_mode = str(form.get("selection_mode", "random"))
    query = str(form.get("query", "")).strip()
    sample_raw = str(form.get("sample_size", "10"))
    if not sample_raw.isdigit() or not 1 <= int(sample_raw) <= 100:
        raise HTTPException(status_code=400, detail="Sample size must be 1-100")
    store = _get_store()
    try:
        selected = store.select_reward_audit_pairs(
            selection_mode, int(sample_raw), query
        )
        job = RewardAuditJob.new(
            checkpoint_path=str(checkpoint),
            checkpoint_name=checkpoint.name,
            selection_mode=selection_mode,
            sample_size=int(sample_raw),
            query=query or None,
        )
        store.create_reward_audit_job(
            job,
            [RewardAuditItem.new(job.id, row["pair_id"]) for row in selected],
        )
    except ValueError as exc:
        return RedirectResponse(
            f"/admin/reward-audit?error={quote(str(exc))}", status_code=303
        )
    return RedirectResponse(f"/admin/reward-audit?job_id={job.id}", status_code=303)


@app.get("/admin/reward-audit/audio/{pair_id}")
async def admin_reward_audit_audio(request: Request, pair_id: str):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    store = _get_store()
    pair = store.get_pair_by_id(pair_id)
    audio = store.get_audio_file(pair.audio_id) if pair else None
    if audio is None:
        raise HTTPException(status_code=404, detail="Audit audio not found")
    try:
        waveform, sample_rate = load_audio(_container_path(audio.file_path))
        buffer = io.BytesIO()
        sf.write(buffer, waveform, sample_rate, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="audio/wav")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audio load error: {exc}") from exc


@app.get("/admin/tts", response_class=HTMLResponse)
async def admin_tts(request: Request):
    if not _is_admin(request):
        return RedirectResponse("/admin/login")
    from tts.provider import DEFAULT_PROVIDER_URL, profiles_for_url, is_local_provider, provider_for_url

    store = _get_store()
    campaigns = await asyncio.to_thread(store.get_tts_campaigns)
    selected_id = request.query_params.get("campaign_id")
    selected = next((c for c in campaigns if c["id"] == selected_id), None)
    if selected_id is not None and selected is None:
        raise HTTPException(status_code=404, detail="Campaign not found")
    selected = selected or (campaigns[0] if campaigns else None)
    try:
        page = int(request.query_params.get("page", "1"))
        if page < 1:
            raise ValueError
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid page") from exc
    items = await asyncio.to_thread(store.get_tts_campaign_items, selected["id"], 25, (page - 1) * 25) if selected else []
    corpora = [dict(c, eligible_texts=len(store.get_synthesis_source_utterances(c["id"])))
               for c in store.get_corpora() if c["active"] and c["language"] == "pl"
               and c["origin"] != "synthesized"]
    quality = await asyncio.to_thread(
        store.get_audio_quality, [item["audio_id"] for item in items if item["audio_id"]]
    )
    for score in quality.values():
        score["metrics"] = json.loads(score["metrics_json"]) if score["metrics_json"] else {}
    provider_url = os.environ.get("TTS_BASE_URL", os.environ.get("TTS_PROVIDER_URL", DEFAULT_PROVIDER_URL))
    profiles = profiles_for_url(provider_url)
    available_profiles = {profile["id"] for profile in profiles}
    provider_error = None
    if is_local_provider(provider_url):
        try:
            capabilities = await asyncio.to_thread(provider_for_url(provider_url).capabilities)
            available = {(engine["id"], voice["id"]) for engine in capabilities["engines"] for voice in engine["voices"]}
            available_profiles = {profile["id"] for profile in profiles if (profile["engine"], profile["voice"]) in available}
        except (OSError, ValueError, KeyError) as exc:
            log.error("Local TTS capabilities unavailable: %s", exc)
            available_profiles = set()
            provider_error = "Local TTS API unavailable; no synthesis profiles can be queued."
    return templates.TemplateResponse(request, "tts_campaigns.html", {
        "campaigns": campaigns, "selected": selected, "items": items, "page": page,
        "profiles": profiles, "corpora": corpora,
        "provider_url": provider_url, "available_profiles": available_profiles,
        "provider_error": provider_error, "local_tts": is_local_provider(provider_url),
        "sft_share": store.get_setting("tts_sft_share_percent", "0"),
        "refill_share": store.get_setting("tts_refill_share_percent", "0"),
        "composition": store.get_setting("queue_refill_composition", "{}"),
        "quality": quality,
        "profile_labels": {profile["id"]: profile["label"] for profile in profiles_for_url(selected["provider_url"] if selected else provider_url)},
    })


@app.post("/admin/tts/create")
async def admin_tts_create(request: Request):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    from tts.provider import validate_profiles, validate_provider_url, DEFAULT_PROVIDER_URL, is_local_provider, provider_for_url

    form = await request.form()
    try:
        provider_url = validate_provider_url(os.environ.get("TTS_BASE_URL", os.environ.get("TTS_PROVIDER_URL", DEFAULT_PROVIDER_URL)))
        profiles = validate_profiles(form.getlist("profiles"), provider_url=provider_url)
        if is_local_provider(provider_url):
            try:
                capabilities = await asyncio.to_thread(provider_for_url(provider_url).capabilities)
            except (OSError, ValueError) as exc:
                log.error("Local TTS API unavailable: %s", exc)
                raise HTTPException(503, "Local TTS API unavailable") from exc
            available = {(engine["id"], voice["id"]) for engine in capabilities["engines"] for voice in engine["voices"]}
            if any((profile["engine"], profile["voice"]) not in available for profile in profiles):
                raise ValueError("Selected local engine is stopped or its model/reference assets are missing")
        store = _get_store()
        args = (str(form.get("corpus_id", "")), profiles, str(form.get("mode", "all")),
                float(str(form.get("value", "100"))), provider_url)
        if form.get("action") == "preview":
            return await asyncio.to_thread(store.preview_tts_campaign, *args)
        campaign = await asyncio.to_thread(
            store.create_tts_campaign, args[0], profiles, str(form.get("intended_use", "both")),
            args[2], args[3], provider_url)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(f"/admin/tts?campaign_id={campaign['id']}", status_code=303)


@app.post("/admin/tts/cancel")
async def admin_tts_cancel(request: Request):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    form = await request.form()
    try:
        await asyncio.to_thread(_get_store().cancel_tts_campaign, str(form.get("campaign_id", "")))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse("/admin/tts", status_code=303)


@app.post("/admin/tts/review")
async def admin_tts_review(request: Request):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    form = await request.form()
    try:
        await asyncio.to_thread(_get_store().review_tts_item, str(form.get("item_id", "")),
            str(form.get("action", "")), actor="admin", job_id=str(form.get("job_id", "")),
            confirm_duplicate=form.get("confirm_duplicate") == "1")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    campaign_id = str(form.get("campaign_id", ""))
    return RedirectResponse(
        f"/admin/tts?campaign_id={quote(campaign_id)}" if campaign_id else "/admin/tts",
        status_code=303,
    )


@app.post("/admin/tts/shares")
async def admin_tts_shares(request: Request):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    form = await request.form()
    try:
        await asyncio.to_thread(_get_store().set_tts_shares,
            float(str(form.get("sft_share", "0"))), float(str(form.get("refill_share", "0"))))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse("/admin/tts", status_code=303)


@app.post("/admin/tts/quality")
async def admin_tts_quality(request: Request):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    form = await request.form()
    store = _get_store()
    try:
        action = str(form.get("action", ""))
        if action == "assess":
            await asyncio.to_thread(store.request_audio_quality, str(form.get("audio_id", "")))
        elif action == "review":
            await asyncio.to_thread(
                store.review_audio_quality, str(form.get("review_id", "")),
                str(form.get("notes", "")),
            )
        else:
            raise ValueError("Unknown quality action")
    except ValueError as exc:
        log.error("Quality review request rejected: %s", exc)
        raise HTTPException(400, str(exc)) from exc
    campaign_id = str(form.get("campaign_id", ""))
    return RedirectResponse(f"/admin/tts?campaign_id={quote(campaign_id)}", status_code=303)


@app.get("/admin/corpora", response_class=HTMLResponse)
async def admin_corpora(request: Request):
    if not _is_admin(request):
        return RedirectResponse("/admin/login")
    return await _corpora_response(request)


async def _corpora_response(request: Request, error: str = "", status_code: int = 200):
    store = _get_store()
    tab = request.query_params.get("tab", "training")
    if tab not in {"training", "evaluation"}:
        raise HTTPException(status_code=400, detail="Unknown corpus category")
    corpora = [
        corpus for corpus in await asyncio.to_thread(store.get_corpus_summaries)
        if corpus["purpose"] == tab
    ]
    corpus_id = request.query_params.get("corpus_id")
    selected = next((corpus for corpus in corpora if corpus["id"] == corpus_id), None)
    if corpus_id is not None and selected is None:
        raise HTTPException(status_code=404, detail="Corpus not found")
    if corpus_id is None:
        selected = corpora[0] if corpora else None
    try:
        page = int(request.query_params.get("page", "1"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Page must be a positive integer") from exc
    if page < 1:
        raise HTTPException(status_code=400, detail="Page must be a positive integer")
    page_size = 25
    pages = max(1, (selected["audio_count"] + page_size - 1) // page_size) if selected else 1
    if page > pages:
        raise HTTPException(status_code=404, detail="Corpus page not found")
    audio_files = await asyncio.to_thread(
        store.get_corpus_audio_files, selected["id"], page_size, (page - 1) * page_size
    ) if selected else []
    return templates.TemplateResponse(request, "corpora.html", {
        "corpora": corpora, "selected": selected, "audio_files": audio_files,
        "page": page, "pages": pages,
        "tab": tab, "error": error,
        "evaluation_sets": await asyncio.to_thread(store.get_canary_sets),
        "training_running": store.get_setting("training_status", "idle") == "running",
    }, status_code=status_code)


@app.post("/admin/corpora/import", response_class=HTMLResponse)
async def admin_corpus_import(request: Request):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    try:
        async with request.form(max_files=201, max_fields=10) as form:
            stage_id = await stage_upload(_CORPUS_UPLOAD_ROOT, form, _get_store())
    except (OSError, ValueError, RuntimeError, csv.Error, sqlite3.Error) as exc:
        log.error("Corpus upload rejected: %s", exc)
        return await _corpora_response(request, str(exc), 400)
    return RedirectResponse(f"/admin/corpora/import/{stage_id}", status_code=303)


@app.get("/admin/corpora/import/{stage_id}", response_class=HTMLResponse)
async def admin_corpus_import_preview(request: Request, stage_id: str):
    if not _is_admin(request):
        return RedirectResponse("/admin/login")
    try:
        metadata, files = await asyncio.to_thread(preview_stage, _CORPUS_UPLOAD_ROOT, stage_id)
    except (OSError, ValueError, RuntimeError, csv.Error) as exc:
        log.error("Corpus preview failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return templates.TemplateResponse(request, "corpus_import.html", {
        "stage_id": stage_id, "metadata": metadata, "audio_files": files,
    })


@app.get("/admin/corpora/import/{stage_id}/audio/{index}")
async def admin_corpus_import_audio(request: Request, stage_id: str, index: int):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    try:
        _, files = await asyncio.to_thread(preview_stage, _CORPUS_UPLOAD_ROOT, stage_id)
        if not 0 <= index < len(files):
            raise ValueError("Unknown preview recording")
        import soundfile as sf

        waveform, sample_rate = await asyncio.to_thread(load_audio, files[index].file_path)
        output = io.BytesIO()
        await asyncio.to_thread(sf.write, output, waveform, sample_rate,
                                format="WAV", subtype="PCM_16")
        return Response(output.getvalue(), media_type="audio/wav")
    except (OSError, ValueError, RuntimeError, csv.Error) as exc:
        log.error("Corpus preview audio failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/admin/corpora/import/{stage_id}/accept")
async def admin_corpus_import_accept(request: Request, stage_id: str):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    try:
        result = await asyncio.to_thread(accept_stage, _CORPUS_UPLOAD_ROOT, stage_id, _get_store())
    except (OSError, ValueError, RuntimeError, csv.Error, sqlite3.Error) as exc:
        log.error("Corpus import rejected: %s", exc)
        return await _corpora_response(request, str(exc), 400)
    if result["monitor_id"]:
        return RedirectResponse(
            f"/admin/canary?canary_set_id={result['monitor_id']}", status_code=303
        )
    return RedirectResponse(f"/admin/corpora?corpus_id={result['corpus_id']}", status_code=303)


@app.post("/admin/corpora/import/{stage_id}/cancel")
async def admin_corpus_import_cancel(request: Request, stage_id: str):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    try:
        await asyncio.to_thread(cancel_stage, _CORPUS_UPLOAD_ROOT, stage_id, _get_store())
    except (OSError, ValueError, RuntimeError, csv.Error, sqlite3.Error) as exc:
        log.error("Corpus import cancellation failed: %s", exc)
        return await _corpora_response(request, str(exc), 400)
    return RedirectResponse("/admin/corpora", status_code=303)


@app.post("/admin/corpora/activation")
async def admin_corpus_activation(request: Request):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    form = await request.form()
    active = str(form.get("active", ""))
    if active not in {"0", "1"}:
        raise HTTPException(status_code=400, detail="Active must be 0 or 1")
    store = _get_store()
    corpus = next(
        (corpus for corpus in store.get_corpora() if corpus["id"] == form.get("corpus_id")),
        None,
    )
    if corpus is None:
        raise HTTPException(status_code=404, detail="Corpus not found")
    try:
        await asyncio.to_thread(store.set_corpus_active, corpus["name"], active == "1")
    except ValueError as exc:
        log.error("Corpus activation rejected: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(f"/admin/corpora?corpus_id={corpus['id']}", status_code=303)


@app.get("/admin/corpora/audio/{audio_id}")
async def admin_corpus_audio(request: Request, audio_id: str):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    audio = _get_store().get_audio_file(audio_id)
    if audio is None:
        raise HTTPException(status_code=404, detail="Corpus audio not found")
    try:
        cached_path = await asyncio.to_thread(_cached_audio_wav, audio.file_path)
        return FileResponse(cached_path, media_type="audio/wav")
    except (OSError, ValueError, RuntimeError, KeyError, zipfile.BadZipFile) as exc:
        log.error("Corpus audio load failed for %s: %s", audio.file_path, exc)
        raise HTTPException(status_code=500, detail="Could not load corpus audio") from exc


@app.get("/admin/annotation-audit", response_class=HTMLResponse)
async def admin_annotation_audit(request: Request):
    if not _is_admin(request):
        return RedirectResponse("/admin/login")
    store = _get_store()
    view = request.query_params.get("view", "interactions")
    if view not in {"interactions", "excluded"}:
        view = "interactions"
    state = request.query_params.get("state", "all")
    participant_id = request.query_params.get("participant_id", "").strip() or None
    dataset = request.query_params.get("dataset", "").strip() or None
    query = request.query_params.get("query", "").strip()
    try:
        page = max(1, int(request.query_params.get("page", "1")))
    except ValueError:
        page = 1
    annotations = []
    annotators = []
    events = []
    excluded_audio = []
    excluded_datasets = []
    if view == "excluded":
        try:
            excluded_audio = store.get_excluded_audio_audit(
                limit=100,
                offset=(page - 1) * 100,
                dataset=dataset,
                query=query or None,
                state=state,
            )
        except ValueError:
            state = "all"
            excluded_audio = store.get_excluded_audio_audit(
                limit=100,
                offset=(page - 1) * 100,
                dataset=dataset,
                query=query or None,
            )
        excluded_datasets = store.get_excluded_audio_datasets()
    else:
        annotators = store.get_annotator_audit()
        events = store.get_annotation_moderation_events()
        try:
            annotations = store.get_annotation_audit(
                limit=100,
                offset=(page - 1) * 100,
                participant_id=participant_id,
                state=state,
            )
        except ValueError:
            state = "all"
            annotations = store.get_annotation_audit(
                limit=100,
                offset=(page - 1) * 100,
                participant_id=participant_id,
            )
    return templates.TemplateResponse(
        request,
        "annotation_audit.html",
        {
            "annotations": annotations,
            "annotators": annotators,
            "events": events,
            "excluded_audio": excluded_audio,
            "excluded_datasets": excluded_datasets,
            "view": view,
            "state": state,
            "participant_id": participant_id or "",
            "dataset": dataset or "",
            "query": query,
            "page": page,
            "training_running": store.get_setting("training_status", "idle")
            == "running",
        },
    )


@app.get("/admin/annotation-audit/excluded-audio/{audio_id}")
async def admin_excluded_audio(request: Request, audio_id: str):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    store = _get_store()
    audio = store.get_audio_file(audio_id)
    if audio is None or not store.is_audio_ever_excluded(audio_id):
        raise HTTPException(status_code=404, detail="Excluded audio not found")
    try:
        waveform, sample_rate = load_audio(_container_path(audio.file_path))
        buffer = io.BytesIO()
        sf.write(buffer, waveform, sample_rate, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="audio/wav")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audio load error: {exc}") from exc


@app.post("/admin/annotation-audit/audio-curation")
async def admin_audio_curation(request: Request):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    form = await request.form()
    audio_id = str(form.get("audio_id", ""))
    action = str(form.get("action", ""))
    reason = str(form.get("reason", "")).strip()
    if action not in {"exclude", "restore"}:
        raise HTTPException(status_code=400, detail="Invalid audio curation action")
    if not reason or len(reason) > 240:
        raise HTTPException(status_code=400, detail="A curation reason is required")
    try:
        _get_store().set_audio_excluded(
            audio_id,
            action == "exclude",
            actor="admin",
            reason=reason,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return RedirectResponse("/admin/annotation-audit?view=excluded", status_code=303)


@app.post("/admin/annotation-audit/annotation")
async def admin_mute_annotation(request: Request):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    form = await request.form()
    annotation_id = str(form.get("annotation_id", ""))
    action = str(form.get("action", ""))
    reason = str(form.get("reason", "")).strip()
    if action not in {"mute", "unmute"}:
        raise HTTPException(status_code=400, detail="Invalid moderation action")
    if not reason or len(reason) > 240:
        raise HTTPException(status_code=400, detail="A moderation reason is required")
    try:
        _get_store().set_annotation_muted(
            annotation_id,
            action == "mute",
            reason=reason,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return RedirectResponse("/admin/annotation-audit", status_code=303)


@app.post("/admin/annotation-audit/annotator")
async def admin_mute_annotator(request: Request):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    form = await request.form()
    participant_id = str(form.get("participant_id", ""))
    action = str(form.get("action", ""))
    reason = str(form.get("reason", "")).strip()
    if action not in {"mute", "unmute"}:
        raise HTTPException(status_code=400, detail="Invalid moderation action")
    if not reason or len(reason) > 240:
        raise HTTPException(status_code=400, detail="A moderation reason is required")
    try:
        _get_store().set_annotator_muted(
            participant_id,
            action == "mute",
            reason=reason,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return RedirectResponse("/admin/annotation-audit", status_code=303)


@app.get("/admin/annotation-audit/audio/{pair_id}")
async def admin_annotation_audit_audio(request: Request, pair_id: str):
    return await admin_reward_audit_audio(request, pair_id)


@app.get("/admin/model-health", response_class=HTMLResponse)
async def admin_model_health(request: Request):
    if not _is_admin(request):
        return RedirectResponse("/admin/login")
    store = _get_store()
    canary_sets = [row for row in store.get_canary_sets() if row["status"] == "ready"]
    requested_set_id = request.query_params.get("canary_set_id")
    canary_set = next(
        (row for row in canary_sets if row["id"] == requested_set_id), None
    )
    if canary_set is None:
        canary_set = store.get_promotion_gate_canary_set()
    evaluations = store.get_canary_evaluations(canary_set["id"]) if canary_set else []
    completed = []
    for evaluation in evaluations:
        if evaluation["status"] != "completed" or not evaluation["aggregate_json"]:
            continue
        row = dict(evaluation)
        row["metrics"] = json.loads(row["aggregate_json"])
        row["warnings"] = json.loads(row["warning_json"]) if row["warning_json"] else []
        completed.append(row)

    def checkpoint_time(row: dict) -> tuple[int, str]:
        if row["stage"] == "base":
            return (0, "")
        match = re.search(r"(20\d{6})[_-](\d{6})", row.get("checkpoint_path") or "")
        if match:
            return (2, "".join(match.groups()))
        return (1, row.get("created_at") or "")

    completed.sort(key=checkpoint_time)

    from evaluation.model_health import paired_corpus_wer_from_predictions
    from evaluation.prediction_diff import prediction_difference

    last_by_stage = {}
    predictions_by_evaluation = {}
    base_predictions = None
    for row in completed:
        current_predictions = store.get_canary_predictions(row["id"])
        predictions_by_evaluation[row["id"]] = current_predictions
        if row["stage"] == "base":
            base_predictions = current_predictions
        elif base_predictions:
            comparison = paired_corpus_wer_from_predictions(
                base_predictions, current_predictions
            )
            row["metrics"].setdefault("vs_base", {})["wer"] = comparison
            scale = 0.2
            low = max(-scale, min(scale, float(comparison["ci_low"])))
            high = max(-scale, min(scale, float(comparison["ci_high"])))
            delta = max(-scale, min(scale, float(comparison["delta"])))
            row["wer_ci"] = {
                "left": (low + scale) / (2 * scale) * 100,
                "width": max((high - low) / (2 * scale) * 100, 1),
                "marker": (delta + scale) / (2 * scale) * 100,
                "delta": float(comparison["delta"]),
                "low": float(comparison["ci_low"]),
                "high": float(comparison["ci_high"]),
            }
        previous = last_by_stage.get(row["stage"])
        if previous is not None:
            previous_predictions = predictions_by_evaluation[previous["id"]]
            difference = prediction_difference(
                previous_predictions, current_predictions
            )
            row["output_diff"] = {
                **difference,
                "previous_model_key": previous["model_key"],
                "previous_hypotheses": {
                    prediction["canary_item_id"]: prediction["hypothesis"]
                    for prediction in previous_predictions
                },
            }
        last_by_stage[row["stage"]] = row

    active_policy = _active_policy_view(store)
    active_evaluation = None
    if active_policy:
        for row in completed:
            checkpoint_path = row.get("checkpoint_path")
            row["is_active"] = bool(
                checkpoint_path
                and str(Path(checkpoint_path).resolve()) == active_policy["path"]
            )
            if row["is_active"]:
                active_evaluation = row

    latest_model_dir = Path(DEFAULT_PATHS.latest_model_dir)
    publication = read_publication(latest_model_dir)
    publication_matches_active = bool(
        publication
        and active_policy
        and publication.get("source_checkpoint") == active_policy["path"]
    )
    publication_monitor = store.get_promotion_gate_canary_set()
    publication_evaluation = None
    if publication_monitor and active_policy:
        publication_evaluations = (
            evaluations
            if canary_set and publication_monitor["id"] == canary_set["id"]
            else store.get_canary_evaluations(publication_monitor["id"])
        )
        publication_evaluation = next(
            (
                row
                for row in publication_evaluations
                if row["status"] == "completed"
                and row["health_status"] == "healthy"
                and row.get("checkpoint_path")
                and str(Path(row["checkpoint_path"]).resolve()) == active_policy["path"]
            ),
            None,
        )
    publication_allowed = bool(
        active_policy
        and publication_evaluation
        and store.get_setting("training_status", "idle") != "running"
    )

    selected_id = request.query_params.get("evaluation_id")
    if not selected_id and completed:
        selected_id = completed[-1]["id"]
    selected = next((row for row in completed if row["id"] == selected_id), None)
    predictions = predictions_by_evaluation.get(selected_id, []) if selected else []
    for prediction in predictions:
        prediction["metrics"] = json.loads(prediction["metrics_json"])
        if selected.get("output_diff"):
            prediction["previous_hypothesis"] = selected["output_diff"][
                "previous_hypotheses"
            ].get(prediction["canary_item_id"])
            prediction["changed_from_previous"] = (
                prediction["canary_item_id"] in selected["output_diff"]["changed_ids"]
            )

    chart_width = 760
    chart_left = 48
    chart_right = 20
    chart_bottom = 178
    chart_height = 138
    x_step = (chart_width - chart_left - chart_right) / max(len(completed) - 1, 1)

    max_error = max(
        [
            float(row["metrics"].get(metric, 0))
            for row in completed
            for metric in ("wer", "cer")
        ]
        or [1.0]
    )
    chart_ceiling = max(0.05, math.ceil(max_error * 20) / 20)

    def chart_y(value: float) -> float:
        return chart_bottom - min(value, chart_ceiling) / chart_ceiling * chart_height

    chart_rows = []
    for index, row in enumerate(completed):
        wer = float(row["metrics"].get("wer", 0))
        cer = float(row["metrics"].get("cer", 0))
        chart_rows.append(
            {
                "x": chart_left + index * x_step,
                "wer": wer,
                "wer_y": chart_y(wer),
                "cer": cer,
                "cer_y": chart_y(cer),
            }
        )
    chart_ticks = [
        {
            "y": chart_bottom - fraction * chart_height,
            "label": f"{chart_ceiling * fraction * 100:.0f}%",
        }
        for fraction in (0, 0.25, 0.5, 0.75, 1)
    ]
    return templates.TemplateResponse(
        request,
        "model_health.html",
        {
            "canary_set": canary_set,
            "canary_sets": canary_sets,
            "evaluations": evaluations,
            "completed": completed,
            "active_policy": active_policy,
            "active_evaluation": active_evaluation,
            "latest_model_dir": str(latest_model_dir),
            "publication": publication,
            "publication_matches_active": publication_matches_active,
            "publication_evaluation": publication_evaluation,
            "publication_monitor": publication_monitor,
            "publication_allowed": publication_allowed,
            "publication_notice": request.query_params.get("publication"),
            "queue_refill_status": store.get_setting("queue_refill_status", "idle"),
            "selected": selected,
            "predictions": predictions,
            "activations": store.get_policy_activations(),
            "wer_points": " ".join(
                f'{point["x"]:.1f},{point["wer_y"]:.1f}' for point in chart_rows
            ),
            "cer_points": " ".join(
                f'{point["x"]:.1f},{point["cer_y"]:.1f}' for point in chart_rows
            ),
            "chart_rows": chart_rows,
            "chart_ticks": chart_ticks,
        },
    )


@app.post("/admin/model-health/publish-latest")
async def admin_model_health_publish_latest(request: Request):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    store = _get_store()
    if store.get_setting("training_status", "idle") == "running":
        raise HTTPException(
            status_code=409,
            detail="Wait for the running training pipeline before publishing",
        )

    active_path = resolve_active_policy_checkpoint(
        store, Path(DEFAULT_PATHS.checkpoints_dir), Path(_PROJECT_ROOT),
        base_model_id=DEFAULT_PATHS.whisper_model_id,
    )
    if active_path is None:
        raise HTTPException(status_code=409, detail="No deployed policy to publish")
    active_path = active_path.resolve()
    checkpoint = store.get_model_checkpoint_by_path(str(active_path))
    canary_set = store.get_promotion_gate_canary_set()
    if checkpoint is None or canary_set is None:
        raise HTTPException(
            status_code=409,
            detail="The deployed policy needs a completed healthy model evaluation",
        )
    evaluation = next(
        (
            row
            for row in store.get_canary_evaluations(canary_set["id"])
            if row["checkpoint_id"] == checkpoint["id"]
            and row["status"] == "completed"
            and row["health_status"] == "healthy"
            and row["aggregate_json"]
        ),
        None,
    )
    if evaluation is None:
        raise HTTPException(
            status_code=409,
            detail="The deployed policy needs a completed healthy model evaluation",
        )

    metrics = json.loads(evaluation["aggregate_json"])
    try:
        publish_latest_model(
            active_path,
            Path(DEFAULT_PATHS.latest_model_dir),
            model_key=checkpoint["model_key"],
            checkpoint_id=checkpoint["id"],
            base_model_id=DEFAULT_PATHS.whisper_model_id,
            monitor_id=canary_set["id"],
            monitor_fingerprint=canary_set["fingerprint"],
            wer=float(metrics["wer"]),
        )
    except (OSError, ValueError) as exc:
        log.exception("Failed to publish latest model")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RedirectResponse(
        "/admin/model-health?publication=published", status_code=303
    )


@app.post("/admin/model-health/rollback/{checkpoint_id}")
async def admin_model_health_rollback(request: Request, checkpoint_id: str):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    store = _get_store()
    if store.get_setting("training_status", "idle") == "running":
        raise HTTPException(
            status_code=409,
            detail="Wait for the running training pipeline before rollback",
        )
    target = store.get_model_checkpoint(checkpoint_id)
    if target is None:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    target_path = Path(target["checkpoint_path"] or "")
    model_version = f"{DEFAULT_PATHS.whisper_model_id}+lora:{target_path.name}"
    try:
        rollback_active_policy_checkpoint(
            store,
            checkpoint_id,
            Path(_PROJECT_ROOT),
            actor="admin",
            model_version=model_version,
            base_model_id=DEFAULT_PATHS.whisper_model_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse("/admin/model-health", status_code=303)


@app.post("/admin/canary/create")
async def admin_create_canary(request: Request):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    form = await request.form()
    name = str(form.get("name", "Model Evaluation 1")).strip() or "Model Evaluation 1"
    seed_raw = str(form.get("seed", "42")).strip()
    if not seed_raw.lstrip("-").isdigit():
        raise HTTPException(status_code=400, detail="Seed must be an integer")
    canary_set = CanarySet.new(
        name=name,
        selection_seed=int(seed_raw),
        selection_method="random_unannotated",
        target_size=40,
    )
    try:
        _get_store().create_canary_set(canary_set)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse("/admin/canary", status_code=303)


@app.get("/admin/canary/audio/{item_id}")
async def admin_canary_audio(request: Request, item_id: str):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    item = _get_store().get_canary_item(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Evaluation set item not found")
    try:
        waveform, sample_rate = load_audio(_container_path(item["file_path"]))
        buffer = io.BytesIO()
        sf.write(buffer, waveform, sample_rate, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="audio/wav")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audio load error: {exc}") from exc


@app.post("/admin/canary/reveal/{item_id}")
async def admin_reveal_canary_gt(request: Request, item_id: str):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    try:
        transcript = _get_store().reveal_canary_ground_truth(item_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"transcript": transcript})


@app.post("/admin/canary/transcript")
async def admin_save_canary_transcript(request: Request):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    form = await request.form()
    item_id = str(form.get("item_id", ""))
    transcript = str(form.get("transcript", ""))
    copied_gt = str(form.get("copied_gt", "false")) == "true"
    try:
        _get_store().save_canary_transcript(item_id, transcript, copied_gt)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse("/admin/canary", status_code=303)


@app.post("/admin/canary/copy-all/{canary_set_id}")
async def admin_bulk_copy_canary_gt(request: Request, canary_set_id: str):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    try:
        result = _get_store().bulk_copy_canary_ground_truth(canary_set_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(
        "/admin/canary?"
        f"bulk_copied={result['copied']}&"
        f"bulk_completed={result['already_completed']}&"
        f"bulk_missing={result['missing_gt']}",
        status_code=303,
    )


@app.post("/admin/canary/freeze/{canary_set_id}")
async def admin_freeze_canary(request: Request, canary_set_id: str):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    try:
        _get_store().freeze_canary_set(canary_set_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse("/admin/canary", status_code=303)


@app.post("/admin/canary/unfreeze/{canary_set_id}")
async def admin_unfreeze_canary(request: Request, canary_set_id: str):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    try:
        revision_id = _get_store().unfreeze_canary_set(canary_set_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(
        f"/admin/canary?canary_set_id={revision_id}", status_code=303
    )


@app.post("/admin/canary/reroll/{item_id}")
async def admin_reroll_canary_item(request: Request, item_id: str):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    store = _get_store()
    item = store.get_canary_item(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Evaluation set item not found")
    try:
        store.reroll_canary_item(item_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(
        f'/admin/canary?canary_set_id={item["canary_set_id"]}', status_code=303
    )


@app.post("/admin/canary/remove/{canary_set_id}")
async def admin_remove_canary(request: Request, canary_set_id: str):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    try:
        _get_store().remove_canary_set(canary_set_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse("/admin/canary", status_code=303)


@app.post("/admin/canary/evaluate/{canary_set_id}")
async def admin_evaluate_canary(request: Request, canary_set_id: str):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    try:
        _get_store().request_wer_monitor_evaluation(canary_set_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse("/admin/canary", status_code=303)


@app.post("/admin/canary/promotion-gate/{canary_set_id}")
async def admin_set_promotion_gate(request: Request, canary_set_id: str):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    try:
        _get_store().set_promotion_gate_canary_set(canary_set_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return RedirectResponse(
        f"/admin/canary?canary_set_id={canary_set_id}", status_code=303
    )


@app.post("/admin/settings")
async def admin_settings(request: Request):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    form = await request.form()
    max_s = str(form.get("max_concurrent_sessions", "10")).strip()
    thr = str(form.get("training_threshold", "50")).strip()
    queue_min = str(form.get("queue_min_size", "20")).strip()
    queue_batch = str(form.get("queue_refill_batch", "50")).strip()
    try:
        training_tuning = TrainingTuning.from_mapping(form)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    show_ref = "true" if form.get("show_reference") else "false"
    show_wer = "true" if form.get("show_wer") else "false"
    if not max_s.isdigit() or int(max_s) < 1:
        raise HTTPException(
            status_code=400,
            detail="max_concurrent_sessions must be a positive integer.",
        )
    if not thr.isdigit() or int(thr) < 1:
        raise HTTPException(
            status_code=400, detail="training_threshold must be a positive integer."
        )
    if not queue_min.isdigit():
        raise HTTPException(
            status_code=400,
            detail="queue_min_size must be a non-negative integer (0 = disabled).",
        )
    if not queue_batch.isdigit() or int(queue_batch) < 1:
        raise HTTPException(
            status_code=400, detail="queue_refill_batch must be a positive integer."
        )
    store = _get_store()
    store.set_setting("max_concurrent_sessions", max_s)
    store.set_setting("training_threshold", thr)
    store.set_setting("queue_min_size", queue_min)
    store.set_setting("queue_refill_batch", queue_batch)
    for key, value in training_tuning.as_settings().items():
        store.set_setting(key, value)
    store.set_setting("show_reference", show_ref)
    store.set_setting("show_wer", show_wer)
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/trigger_refill")
async def admin_trigger_refill(request: Request):
    """Manually request a queue refill from the admin panel."""
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    store = _get_store()
    conflict = None
    if store.get_setting("queue_refill_status", "idle") in ("running", "queued"):
        conflict = "Uzupełnianie już oczekuje lub trwa. / Queue refill is already pending or running."
    elif store.get_setting("training_status", "idle") == "running":
        conflict = "Trening trwa; uzupełnianie nastąpi później. / Training is running; refill will resume afterward."
    if conflict:
        if "text/html" in request.headers.get("accept", ""):
            return RedirectResponse(f"/admin?refill_error={quote(conflict)}", status_code=303)
        raise HTTPException(status_code=409, detail=conflict)
    pending = store.count_pending_pairs()
    batch_str = store.get_setting("queue_refill_batch", "50")
    batch = int(batch_str) if batch_str.isdigit() else 50
    log.info(
        "[queue-refill] Manual trigger from admin — will generate %d pairs (pending=%d).",
        batch,
        pending,
    )
    store.set_setting("queue_refill_requested", "true")
    store.set_setting("queue_refill_status", "queued")
    store.set_setting(
        "queue_refill_log",
        f"Manual trigger: pending={pending}, will generate {batch} pairs",
    )
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/trigger_training")
async def admin_trigger_training(request: Request):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    store = _get_store()
    if store.get_setting("training_status") == "running":
        raise HTTPException(status_code=409, detail="Training is already running.")
    store.set_setting("training_requested", "true")
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/cancel_training")
async def admin_cancel_training(request: Request):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    _get_store().set_setting("training_requested", "false")
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/logout")
async def admin_logout(request: Request):
    if not _is_admin(request):
        raise HTTPException(status_code=403)
    return _clear_admin_session()


def _clear_admin_session():
    resp = RedirectResponse("/admin/login", status_code=303)
    _get_store().set_setting("admin_session_id", "")
    _get_store().set_setting("admin_session_password_fingerprint", "")
    resp.delete_cookie("admin_session")
    return resp
