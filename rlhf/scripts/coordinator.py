"""
scripts/coordinator.py -- Training orchestrator.

Runs as a long-lived Docker service. Polls the DB every POLL_INTERVAL seconds
for two triggers:

  1. Manual request  -- admin presses "Trigger Training Now" in the admin panel.
  2. Auto-threshold  -- new human annotations since the last training run reach
                        the configured training_threshold.

When triggered, it runs the training pipeline in sequence:
    train reward model -> optional SFT -> generate ephemeral DPO data -> DPO/evaluate

Pair generation is handled exclusively by the auto-refill loop (run_queue_refill),
which keeps the annotation queue filled continuously using the latest available
reward-model checkpoint.  Generating pairs inside the training pipeline was
redundant: both paths used the same checkpoint and the newly generated pairs
would be unannotated anyway, contributing nothing to the current training run.

The web annotation service remains fully available during training because:
  - SQLite runs in WAL mode (concurrent reads + single writer, no blocking).
  - The coordinator is a completely separate Docker container.
  - Training scripts hold only short write locks (status updates, checkpoint saves).

Environment variables
---------------------
COORDINATOR_POLL_INTERVAL : int (default 30)
    Seconds between DB polls.
EVAL_MAX_SAMPLES : int (default 40)
    Maximum test examples per split during DPO evaluation. Set to 0 for full.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from utils.training_tuning import TrainingTuning

# Resolve the project root relative to this file so the coordinator works
# both inside Docker (/app/scripts/coordinator.py) and on any host machine.
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from paths import DEFAULT_PATHS
from diagnostics.collector import DiagnosticsCollector
from storage.feedback_store import FeedbackStore
from utils.policy_checkpoint import (
    ensure_active_policy_model_version,
    resolve_active_policy_checkpoint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [coordinator] %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def _enable_file_logging() -> None:
    logs_dir = Path(DEFAULT_PATHS.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"coordinator-{datetime.now().strftime('%Y-%m-%d')}.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [coordinator] %(levelname)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(handler)

POLL_INTERVAL = int(os.environ.get("COORDINATOR_POLL_INTERVAL", "30"))
EVAL_MAX_SAMPLES = int(os.environ.get("EVAL_MAX_SAMPLES", "40"))

# Use the same Python interpreter that launched this script so it works both
# inside Docker (python3.12) and on any host machine (py, python, python3, …).
_PYTHON = sys.executable
_WORKDIR = str(_PROJECT_ROOT)
_STEP_TIMEOUT = 14400  # 4 hours max per step


def _find_latest_rm_checkpoint() -> Optional[Path]:
    """Return the latest reward_model_vN checkpoint dir, or None if none exists."""
    base = Path(DEFAULT_PATHS.reward_model_dir)
    if not base.exists():
        return None
    checkpoints = sorted(base.glob("reward_model_v*"), key=lambda p: p.name)
    return checkpoints[-1] if checkpoints else None


def _ensure_data_prepared() -> None:
    """Auto-run prepare_data.py on first boot if manifests are missing but a source ZIP is present.

    _data/ is gitignored and machine-local (see .gitignore): manifests never
    travel with a git pull. Without this check, generate/refill silently find
    0 audio files and loop forever every poll cycle, burning GPU time for nothing.
    """
    train_csv = DEFAULT_PATHS.admedvoice_train_csv
    if train_csv and Path(train_csv).exists():
        return  # already prepared

    archive = _PROJECT_ROOT / "_data" / "archive.zip"
    if not archive.exists():
        log.warning(
            "No manifests found and _data/archive.zip is missing -- skipping auto-prepare. "
            "Place the source ZIP in _data/ and restart the coordinator, or run "
            "'docker compose run --rm prepare' manually."
        )
        return

    log.info(
        "Manifests missing but _data/archive.zip found -- running prepare_data.py automatically."
    )
    try:
        rc = subprocess.run(
            [_PYTHON, "prepare_data.py", "--datasets", "admedvoice"],
            cwd=_WORKDIR,
            timeout=_STEP_TIMEOUT,
        ).returncode
    except subprocess.TimeoutExpired:
        log.error("Auto-prepare timed out after %d s.", _STEP_TIMEOUT)
        return
    if rc == 0:
        log.info("Auto-prepare complete -- manifests written to _data/manifests/.")
    else:
        log.error(
            "Auto-prepare failed (exit %d). Run 'docker compose run --rm prepare' "
            "manually to see full output.",
            rc,
        )


def _pipeline(
    policy_checkpoint: Optional[Path] = None,
    include_sft: bool = True,
    round_id: str = "vmanual",
    tuning: Optional[TrainingTuning] = None,
) -> list[tuple[list[str], str]]:
    """Return ordered list of (command, description) for the training pipeline."""
    tuning = tuning or TrainingTuning()
    round_dir = Path(DEFAULT_PATHS.output_dir) / "tmp" / "dpo_rounds" / round_id
    reward_checkpoint = (
        Path(DEFAULT_PATHS.reward_model_dir) / f"reward_model_{round_id}"
    )
    synthetic_artifact = round_dir / "synthetic_pairs.jsonl"
    steps = [
        ([_PYTHON, "run_train_reward.py", "--version", round_id], "train reward model"),
    ]
    if include_sft:
        dpo_start = Path(DEFAULT_PATHS.checkpoints_dir) / f"sft_{round_id}"
        sft_cmd = [
            _PYTHON,
            "run_sft.py",
            "--epochs",
            str(tuning.sft_epochs),
            "--learning_rate",
            str(tuning.sft_learning_rate),
            "--warmup_steps",
            str(tuning.sft_warmup_steps),
            "--max_grad_norm",
            str(tuning.max_grad_norm),
            "--early_stopping_patience",
            str(tuning.early_stopping_patience),
            "--output_dir",
            str(dpo_start),
        ]
        if policy_checkpoint is not None:
            sft_cmd += ["--resume", str(policy_checkpoint)]
        steps.append((sft_cmd, "SFT fine-tuning"))
    else:
        dpo_start = policy_checkpoint

    if tuning.synthetic_ratio > 0:
        synthetic_cmd = [
            _PYTHON,
            "run_generate_synthetic.py",
            "--round_id",
            round_id,
            "--output",
            str(synthetic_artifact),
            "--rm_checkpoint",
            str(reward_checkpoint),
            "--max_synthetic_ratio",
            str(tuning.synthetic_ratio),
            "--pool_multiplier",
            str(tuning.synthetic_pool_multiplier),
            "--minimum_reward_margin",
            str(tuning.synthetic_reward_margin),
            "--minimum_token_edits",
            str(tuning.synthetic_min_token_edits),
        ]
        if dpo_start is not None:
            synthetic_cmd += ["--policy_checkpoint", str(dpo_start)]
        steps.append((synthetic_cmd, "generate synthetic DPO data"))

    dpo_cmd = [
        _PYTHON,
        "run_rlhf.py",
        "--max_eval_samples",
        str(EVAL_MAX_SAMPLES),
        "--epochs",
        str(tuning.dpo_epochs),
        "--learning_rate",
        str(tuning.dpo_learning_rate),
        "--dpo_beta",
        str(tuning.dpo_beta),
        "--max_synthetic_ratio",
        str(tuning.synthetic_ratio),
        "--max_grad_norm",
        str(tuning.max_grad_norm),
        "--early_stopping_patience",
        str(tuning.early_stopping_patience),
    ]
    if tuning.synthetic_ratio > 0:
        dpo_cmd += ["--synthetic_data", str(synthetic_artifact)]
    if dpo_start is not None:
        dpo_cmd += [
            "--resume",
            str(dpo_start),
            "--reference",
            str(dpo_start),
        ]
    steps.append((dpo_cmd, "DPO fine-tuning"))
    steps.append(
        (
            [
                _PYTHON,
                "run_evaluate_canary.py",
                "--promote-pending",
            ],
            "model-evaluation policy promotion",
        )
    )
    return steps


def _run_step_streaming(
    cmd: list[str],
    cwd: str,
    timeout: int,
    store: FeedbackStore,
    log_key: str = "training_step_log",
) -> tuple[int, str]:
    """
    Run a pipeline step via Popen, stream stdout+stderr to the logger AND
    to the `training_step_log` DB setting so the admin panel can display
    live progress (last non-empty line) in real time.

    Returns (exit_code, last_lines_text) where last_lines_text is the tail
    of the output (up to 15 lines) for error reporting.
    """
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )

    _tail: list[str] = []

    def _reader():
        for raw in proc.stdout:
            line = raw.rstrip()
            if not line:
                continue
            log.info("  | %s", line)
            _tail.append(line)
            if len(_tail) > 15:
                _tail.pop(0)
            try:
                store.set_setting(log_key, line[:300])
            except Exception:
                pass

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        timed_out = True
    finally:
        reader.join(timeout=15)

    if timed_out:
        raise subprocess.TimeoutExpired(cmd, timeout)
    return proc.returncode, "\n".join(_tail)


def is_policy_gate_rejection(description: str, exit_code: int) -> bool:
    return description == "model-evaluation policy promotion" and exit_code == 3


def normalize_legacy_gate_rejection(store: FeedbackStore) -> bool:
    error = store.get_setting("training_error", "")
    if not (
        store.get_setting("training_status", "idle") == "failed"
        and store.get_setting("promotion_gate_status", "") == "rejected"
        and "Step 'model-evaluation policy promotion' failed (exit code 3)." in error
    ):
        return False
    store.set_setting("training_status", "guarded")
    store.set_setting("training_error", "")
    log.info("Normalized legacy policy gate rejection to guarded status.")
    return True


def run_training(store: FeedbackStore) -> bool:
    """Execute the training pipeline. Returns True on success, False on failure."""
    store.set_setting("training_status", "running")
    store.set_setting("training_requested", "false")
    store.set_setting("training_error", "")
    log.info("Training pipeline starting.")
    try:
        policy_checkpoint = resolve_active_policy_checkpoint(
            store,
            Path(DEFAULT_PATHS.checkpoints_dir),
            _PROJECT_ROOT,
            base_model_id=DEFAULT_PATHS.whisper_model_id,
        )
        ensure_active_policy_model_version(
            store, DEFAULT_PATHS.whisper_model_id, policy_checkpoint
        )
        include_sft = bool(store.get_sft_training_rows())
        tuning = TrainingTuning.from_store(store)
        round_id = datetime.now().strftime("v%Y%m%d_%H%M%S_%f")
        store.set_setting("training_sft_included", "true" if include_sft else "false")
        for stage in ("sft", "synthetic", "dpo"):
            store.set_setting(f"training_{stage}_diagnostics", "")
        log.info(
            "Training starts from policy=%s; SFT=%s",
            policy_checkpoint or DEFAULT_PATHS.whisper_model_id,
            "enabled" if include_sft else "skipped (no human corrections)",
        )
        log.info("Training tuning snapshot: %s", tuning)
        steps = _pipeline(
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
            log.info("-- Step %d/%d: %s", n, total, description)
            log.info("   Command: %s", " ".join(str(c) for c in cmd))
            try:
                rc, tail = _run_step_streaming(
                    [str(c) for c in cmd], _WORKDIR, _STEP_TIMEOUT, store
                )
            except subprocess.TimeoutExpired:
                log.error("Step '%s' timed out after %d s.", description, _STEP_TIMEOUT)
                store.set_setting("training_status", "failed")
                store.set_setting(
                    "training_error",
                    f"Step '{description}' timed out after {_STEP_TIMEOUT} s.",
                )
                return False
            if is_policy_gate_rejection(description, rc):
                log.warning(
                    "Policy candidate was blocked by the WER gate; active policy unchanged."
                )
                store.set_setting("training_status", "guarded")
                store.set_setting("training_error", "")
                shutil.rmtree(
                    Path(DEFAULT_PATHS.output_dir) / "tmp" / "dpo_rounds" / round_id,
                    ignore_errors=True,
                )
                return True
            if rc != 0:
                log.error("Step '%s' failed (exit code %d).", description, rc)
                store.set_setting("training_status", "failed")
                store.set_setting(
                    "training_error",
                    f"Step '{description}' failed (exit code {rc}).\n\n{tail}",
                )
                return False
        store.set_setting("training_status", "idle")
        store.set_setting("training_step", "")
        store.set_setting("training_step_n", "0")
        store.set_setting("training_step_log", "")
        store.set_setting("training_error", "")
        shutil.rmtree(
            Path(DEFAULT_PATHS.output_dir) / "tmp" / "dpo_rounds" / round_id,
            ignore_errors=True,
        )
        log.info("Training pipeline completed successfully.")
        run_asr_publication_if_needed(store)
        return True
    except subprocess.TimeoutExpired as exc:
        log.error("Step timed out after %d s: %s", _STEP_TIMEOUT, exc)
        store.set_setting("training_status", "failed")
        store.set_setting("training_error", f"Training timed out: {exc}")
        return False
    except Exception as exc:
        log.error("Unexpected pipeline error: %s", exc, exc_info=True)
        store.set_setting("training_status", "failed")
        store.set_setting(
            "training_error",
            f"Unexpected pipeline error: {type(exc).__name__}: {exc}",
        )
        return False


def run_queue_refill(store: FeedbackStore) -> bool:
    """Run only run_generate_pairs.py to top up the annotation queue.

    Called only after confirming training is NOT about to trigger (see main()).
    Returns True on success, False on failure.
    """
    batch_str = store.get_setting("queue_refill_batch", "50")
    batch = batch_str if batch_str.isdigit() else "50"
    pending = store.count_pending_pairs()
    rm_ckpt = _find_latest_rm_checkpoint()
    log.info(
        "[queue-refill] Generating %s pairs (pending=%d, rm_checkpoint=%s).",
        batch,
        pending,
        rm_ckpt or "none",
    )
    store.set_setting("queue_refill_status", "running")
    store.set_setting("queue_refill_log", f"Running… pending={pending}")
    cmd = [str(_PYTHON), "run_generate_pairs.py", "--n", batch]
    if rm_ckpt is not None:
        cmd += ["--rm_checkpoint", str(rm_ckpt)]
    try:
        rc, tail = _run_step_streaming(
            cmd, _WORKDIR, _STEP_TIMEOUT, store, log_key="queue_refill_log"
        )
    except subprocess.TimeoutExpired:
        log.error("[queue-refill] Timed out after %d s.", _STEP_TIMEOUT)
        store.set_setting("queue_refill_status", "failed")
        store.set_setting("queue_refill_log", f"Timed out after {_STEP_TIMEOUT} s")
        return False
    if rc == 0:
        new_pending = store.count_pending_pairs()
        added = max(0, new_pending - pending)
        store.set_setting("queue_refill_status", "idle")
        store.set_setting(
            "queue_refill_log", f"+{added} pairs added (queue: {new_pending})"
        )
        log.info("[queue-refill] Done. +%d pairs.", added)
        return True
    log.error("[queue-refill] run_generate_pairs.py failed (exit %d).", rc)
    store.set_setting("queue_refill_status", "failed")
    store.set_setting(
        "queue_refill_log", (tail.splitlines()[-1] if tail else "(no output)")[:300]
    )
    return False


def run_wer_monitor_evaluation(store: FeedbackStore) -> bool:
    """Evaluate all uncached checkpoints after a monitor is frozen."""
    canary_set_id = store.get_setting("wer_monitor_evaluation_canary_set_id", "")
    store.set_setting("wer_monitor_evaluation_requested", "false")
    store.set_setting("wer_monitor_evaluation_status", "running")
    store.set_setting("wer_monitor_evaluation_log", "Starting evaluation")
    cmd = [str(_PYTHON), "run_evaluate_canary.py", "--if-ready"]
    if canary_set_id:
        cmd += ["--canary-set-id", canary_set_id]
    try:
        rc, tail = _run_step_streaming(
            cmd,
            _WORKDIR,
            _STEP_TIMEOUT,
            store,
            log_key="wer_monitor_evaluation_log",
        )
    except subprocess.TimeoutExpired:
        store.set_setting("wer_monitor_evaluation_status", "failed")
        store.set_setting("wer_monitor_evaluation_log", "Evaluation timed out")
        return False
    if rc == 0:
        store.set_setting("wer_monitor_evaluation_status", "completed")
        store.set_setting("wer_monitor_evaluation_log", "Evaluation completed")
        run_asr_publication_if_needed(store)
        return True
    store.set_setting("wer_monitor_evaluation_status", "failed")
    store.set_setting(
        "wer_monitor_evaluation_log",
        (tail.splitlines()[-1] if tail else "Evaluation failed")[:300],
    )
    return False


def run_reward_audit(store: FeedbackStore) -> bool:
    """Run one queued diagnostic reward-model scoring job."""
    try:
        rc, _ = _run_step_streaming(
            [str(_PYTHON), "run_reward_audit.py"],
            _WORKDIR,
            _STEP_TIMEOUT,
            store,
        )
    except subprocess.TimeoutExpired:
        log.error("[reward-audit] Timed out after %d s.", _STEP_TIMEOUT)
        return False
    if rc != 0:
        log.error("[reward-audit] Scoring failed (exit %d).", rc)
        return False
    return True


def training_trigger_progress(store: FeedbackStore) -> tuple[int, int]:
    threshold_str = store.get_setting("training_threshold", "50")
    threshold = int(threshold_str) if threshold_str.isdigit() else 50
    last_str = store.get_setting("annotations_at_last_training", "0")
    last = int(last_str) if last_str.isdigit() else 0
    failed_str = store.get_setting("training_failed_at_annotations", "0")
    failed = int(failed_str) if failed_str.isdigit() else 0
    current = store.get_stats().get("actionable_annotations", 0)
    return max(0, current - max(last, failed)), threshold


def check_trigger(store: FeedbackStore) -> tuple[bool, str]:
    """Return (should_train, reason_string). Never starts training itself."""
    status = store.get_setting("training_status", "idle")
    if status == "running":
        return False, "already running"

    if store.get_setting("training_requested", "false") == "true":
        return True, "manual request from admin panel"

    new_annotations, threshold = training_trigger_progress(store)
    if new_annotations >= threshold:
        return (
            True,
            f"auto-threshold reached ({new_annotations}/{threshold} actionable annotations)",
        )

    return (
        False,
        f"waiting ({new_annotations}/{threshold} actionable annotations since last run)",
    )


def queue_existing_wer_monitor_evaluation(store: FeedbackStore) -> bool:
    """Ensure a frozen monitor present before this feature is evaluated once."""
    canary_set = store.get_promotion_gate_canary_set()
    if canary_set is None:
        return False
    if store.get_setting("wer_monitor_evaluation_status", "idle") in (
        "queued",
        "running",
        "completed",
    ):
        return False
    store.request_wer_monitor_evaluation(canary_set["id"])
    return True


def run_training_once(store: FeedbackStore) -> bool:
    """Run one canonical pipeline and persist its annotation watermark."""
    stats = store.get_stats()
    success = run_training(store)
    annotation_count = str(stats.get("actionable_annotations", 0))
    if success:
        store.set_setting("annotations_at_last_training", annotation_count)
        store.set_setting("training_failed_at_annotations", "0")
    else:
        store.set_setting("training_failed_at_annotations", annotation_count)
    return success


def update_coordinator_heartbeat(store: FeedbackStore) -> str:
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    store.set_setting("coordinator_heartbeat_at", timestamp)
    return timestamp


def run_embedded_diagnostics(
    stop_event: threading.Event,
    collector_factory=DiagnosticsCollector,
) -> None:
    """Run diagnostics independently without allowing failures to stop training."""
    interval_s = float(os.environ.get("DIAGNOSTICS_INTERVAL_SECONDS", "5"))
    proc_root = Path(os.environ.get("DIAGNOSTICS_PROC_ROOT", "/proc"))
    while not stop_event.is_set():
        try:
            collector = collector_factory(
                FeedbackStore(DEFAULT_PATHS.db_path),
                DEFAULT_PATHS,
                interval_s=interval_s,
                proc_root=proc_root,
            )
            log.info("Embedded diagnostics collector started.")
            while not stop_event.is_set():
                try:
                    collector.tick()
                except Exception as exc:
                    log.error("Diagnostics tick failed: %s", exc, exc_info=True)
                stop_event.wait(interval_s)
        except Exception as exc:
            log.error("Diagnostics collector setup failed: %s", exc, exc_info=True)
            stop_event.wait(interval_s)


def start_embedded_diagnostics() -> threading.Thread:
    stop_event = threading.Event()
    thread = threading.Thread(
        target=run_embedded_diagnostics,
        args=(stop_event,),
        name="diagnostics-collector",
        daemon=True,
    )
    thread.start()
    return thread


def run_asr_publication_if_needed(store):
    """Attempt each approved deployed identity once; failure never changes policy."""
    if store.get_setting("training_status", "idle") == "running":
        return False
    try:
        from utils.asr_publication import approved_policy, read_current_publication

        try:
            checkpoint, monitor, _ = approved_policy(store, DEFAULT_PATHS)
        except ValueError:
            return False
        current = read_current_publication(DEFAULT_PATHS)
        if (current and current["checkpoint_id"] == checkpoint["id"]
                and current["monitor_fingerprint"] == monitor["fingerprint"]):
            return False
        identity = checkpoint["id"] + ":" + monitor["fingerprint"]
        if store.get_setting("asr_publication_attempted_identity", "") == identity:
            return False
        store.set_setting("asr_publication_attempted_identity", identity)
        store.set_setting("asr_publication_status", "running")
        rc, tail = _run_step_streaming(
            [str(_PYTHON), "run_publish_model.py", "--if-ready"],
            _WORKDIR, _STEP_TIMEOUT, store, log_key="asr_publication_log",
        )
        if rc != 0:
            store.set_setting("asr_publication_status", "failed")
            store.set_setting("asr_publication_log", (tail or f"Publisher failed (exit {rc})")[-500:])
        return True
    except Exception as exc:
        log.exception("ASR publication failed; deployed policy retained")
        store.set_setting("asr_publication_status", "failed")
        store.set_setting("asr_publication_log", f"{type(exc).__name__}: {exc}"[:500])
        return False


def main() -> None:
    _enable_file_logging()
    parser = argparse.ArgumentParser(description="RLHF training coordinator")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one training pipeline immediately and exit.",
    )
    args = parser.parse_args()
    log.info("Coordinator starting. Poll interval: %d s", POLL_INTERVAL)

    store = FeedbackStore(DEFAULT_PATHS.db_path)
    policy_checkpoint = resolve_active_policy_checkpoint(
        store, Path(DEFAULT_PATHS.checkpoints_dir), _PROJECT_ROOT,
        base_model_id=DEFAULT_PATHS.whisper_model_id,
    )
    ensure_active_policy_model_version(
        store, DEFAULT_PATHS.whisper_model_id, policy_checkpoint
    )
    update_coordinator_heartbeat(store)

    normalize_legacy_gate_rejection(store)
    queue_existing_wer_monitor_evaluation(store)
    abandoned_audits = store.fail_running_reward_audit_jobs(
        "Coordinator restarted before scoring completed"
    )
    if abandoned_audits:
        log.warning("Marked %d abandoned Reward Audit job(s) failed.", abandoned_audits)

    if not args.once:
        start_embedded_diagnostics()

    _ensure_data_prepared()

    if args.once:
        raise SystemExit(0 if run_training_once(store) else 1)

    # Reset stale 'running' status left by a previous crashed run.
    if store.get_setting("training_status") == "running":
        log.warning("Stale 'running' status found on startup -- resetting to 'failed'.")
        store.set_setting("training_status", "failed")

    while True:
        try:
            update_coordinator_heartbeat(store)
            trigger, reason = check_trigger(store)
            if trigger:
                log.info("Training triggered: %s", reason)
                success = run_training_once(store)
                if not success:
                    log.warning(
                        "Training pipeline did not complete successfully. "
                        "Check logs above for the failing step."
                    )
            elif (
                store.get_setting("wer_monitor_evaluation_requested", "false") == "true"
            ):
                run_wer_monitor_evaluation(store)
            elif run_asr_publication_if_needed(store):
                pass
            elif store.has_queued_reward_audit_job():
                run_reward_audit(store)
            elif store.get_setting("queue_refill_requested", "false") == "true":
                # Double-check the threshold after observing the refill flag.
                would_train, train_reason = check_trigger(store)
                if would_train:
                    log.info(
                        "[queue-refill] Deferred until training completes (%s).",
                        train_reason,
                    )
                    store.set_setting(
                        "queue_refill_log", "Deferred until training completes"
                    )
                else:
                    store.set_setting("queue_refill_requested", "false")
                    run_queue_refill(store)
            else:
                log.debug("No training trigger: %s", reason)
        except Exception as exc:
            log.error("Coordinator poll error: %s", exc, exc_info=True)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
