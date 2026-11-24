"""
Tests for the coordinator training pipeline definition.

These are pure unit tests — no subprocesses, no DB, no GPU.
They verify the pipeline structure itself: step order, command correctness,
absence of dead CLI arguments.

Regression coverage for:
- run_train_reward.py missing from pipeline (caused 'score rewards' to fail on
  cold start because no reward model checkpoint existed)
- Dead --algorithm dpo argument passed to run_rlhf.py (would have crashed on
  the DPO step after the --algorithm flag was removed from run_rlhf.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the project root importable even when running from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.coordinator as coordinator
from scripts.coordinator import (
    _pipeline,
    check_trigger,
    run_embedded_diagnostics,
    run_training_once,
    update_coordinator_heartbeat,
)
from storage.feedback_store import FeedbackStore
from utils.training_tuning import TrainingTuning

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _step_names(steps) -> list[str]:
    return [description for _, description in steps]


def _all_args(steps) -> list[str]:
    """Flatten all command tokens from all pipeline steps."""
    return [token for cmd, _ in steps for token in cmd]


# ---------------------------------------------------------------------------
# Step presence
# ---------------------------------------------------------------------------


class TestPipelineStepPresence:
    def test_train_reward_model_is_present(self):
        """Regression: step was missing, causing cold-start failure."""
        names = _step_names(_pipeline())
        assert "train reward model" in names

    def test_synthetic_generation_is_present(self):
        names = _step_names(_pipeline())
        assert "generate synthetic DPO data" in names

    def test_sft_fine_tuning_is_present(self):
        names = _step_names(_pipeline())
        assert "SFT fine-tuning" in names

    def test_dpo_fine_tuning_is_present(self):
        names = _step_names(_pipeline())
        assert "DPO fine-tuning" in names


def test_asr_publication_hook_runs_once_per_active_identity_and_detects_rollback(store, monkeypatch):
    import utils.asr_publication as publication

    active = {"id": "approved-a"}
    monitor = {"fingerprint": "frozen"}
    monkeypatch.setattr(publication, "approved_policy", lambda store, paths: (active, monitor, {}))
    monkeypatch.setattr(publication, "read_current_publication", lambda paths: None)
    calls = []

    def run(cmd, cwd, timeout, active_store, log_key):
        calls.append(cmd)
        assert log_key == "asr_publication_log"
        return 1, "conversion failed"

    monkeypatch.setattr(coordinator, "_run_step_streaming", run)
    store.set_setting("active_policy_checkpoint", "unchanged")
    assert coordinator.run_asr_publication_if_needed(store)
    assert not coordinator.run_asr_publication_if_needed(store)
    assert store.get_setting("active_policy_checkpoint") == "unchanged"
    assert store.get_setting("asr_publication_status") == "failed"
    active["id"] = "rollback-target"
    assert coordinator.run_asr_publication_if_needed(store)
    assert len(calls) == 2
    assert calls[0][-2:] == ["run_publish_model.py", "--if-ready"]


# ---------------------------------------------------------------------------
# Step ordering
# ---------------------------------------------------------------------------


class TestPipelineStepOrder:
    def test_train_reward_before_synthetic_generation(self):
        names = _step_names(_pipeline())
        assert names.index("train reward model") < names.index(
            "generate synthetic DPO data"
        )

    def test_sft_before_synthetic_generation(self):
        names = _step_names(_pipeline())
        assert names.index("SFT fine-tuning") < names.index(
            "generate synthetic DPO data"
        )

    def test_sft_before_dpo(self):
        names = _step_names(_pipeline())
        assert names.index("SFT fine-tuning") < names.index("DPO fine-tuning")


# ---------------------------------------------------------------------------
# Command correctness — no dead CLI arguments
# ---------------------------------------------------------------------------


class TestPipelineCommandArgs:
    def test_no_dead_algorithm_arg(self):
        """Regression: --algorithm was removed from run_rlhf.py but coordinator
        still passed --algorithm dpo, which would crash the DPO step."""
        all_args = _all_args(_pipeline())
        assert "--algorithm" not in all_args

    def test_dpo_step_uses_run_rlhf(self):
        steps_by_name = {desc: cmd for cmd, desc in _pipeline()}
        dpo_cmd = steps_by_name["DPO fine-tuning"]
        assert any("run_rlhf.py" in str(arg) for arg in dpo_cmd)

    def test_dpo_step_caps_evaluation_samples(self):
        steps_by_name = {desc: cmd for cmd, desc in _pipeline()}
        dpo_cmd = steps_by_name["DPO fine-tuning"]
        limit_index = dpo_cmd.index("--max_eval_samples")
        assert dpo_cmd[limit_index + 1] == "40"

    def test_dpo_candidate_is_wer_gated_before_promotion(self):
        steps_by_name = {desc: cmd for cmd, desc in _pipeline()}
        gate_cmd = steps_by_name["model-evaluation policy promotion"]

        assert "--promote-pending" in gate_cmd
        assert _step_names(_pipeline()).index("DPO fine-tuning") < _step_names(
            _pipeline()
        ).index("model-evaluation policy promotion")

    def test_train_reward_step_uses_run_train_reward(self):
        steps_by_name = {desc: cmd for cmd, desc in _pipeline()}
        cmd = steps_by_name["train reward model"]
        assert any("run_train_reward.py" in str(arg) for arg in cmd)

    def test_synthetic_step_uses_ephemeral_generator(self):
        steps_by_name = {desc: cmd for cmd, desc in _pipeline()}
        cmd = steps_by_name["generate synthetic DPO data"]
        assert any("run_generate_synthetic.py" in str(arg) for arg in cmd)
        assert "--output" in cmd

    def test_persistent_reward_scoring_not_in_pipeline(self):
        assert not any("run_score_rewards.py" in arg for arg in _all_args(_pipeline()))


class TestPipelineSteps:
    def test_generate_pairs_not_in_pipeline(self):
        # pair generation is handled by auto-refill, not the training pipeline
        assert "generate pairs" not in _step_names(_pipeline())

    def test_pipeline_starts_with_reward_model(self):
        assert _step_names(_pipeline())[0] == "train reward model"

    def test_canary_gate_runs_after_dpo_and_blocks_publication(self):
        steps = _pipeline()
        names = _step_names(steps)
        assert names.index("model-evaluation policy promotion") > names.index(
            "DPO fine-tuning"
        )
        command = dict((description, cmd) for cmd, description in steps)[
            "model-evaluation policy promotion"
        ]
        assert "--promote-pending" in command
        assert "--non-blocking" not in command

    def test_report_only_evaluation_is_not_a_pipeline_step(self):
        assert "evaluate WER" not in _step_names(_pipeline())

    def test_active_policy_is_resumed_by_sft(self, tmp_path):
        policy = tmp_path / "whisper_dpo_previous"
        steps = {desc: cmd for cmd, desc in _pipeline(policy)}
        assert steps["SFT fine-tuning"][-2:] == ["--resume", str(policy)]
        dpo_cmd = steps["DPO fine-tuning"]
        resume_index = dpo_cmd.index("--resume")
        reference_index = dpo_cmd.index("--reference")
        assert dpo_cmd[resume_index + 1] == dpo_cmd[reference_index + 1]
        assert dpo_cmd[resume_index + 1].endswith("sft_vmanual")

    def test_dpo_resumes_active_policy_when_sft_is_skipped(self, tmp_path):
        policy = tmp_path / "whisper_dpo_previous"
        steps = {desc: cmd for cmd, desc in _pipeline(policy, include_sft=False)}
        assert "SFT fine-tuning" not in steps
        dpo_cmd = steps["DPO fine-tuning"]
        assert dpo_cmd[dpo_cmd.index("--resume") + 1] == str(policy)
        assert dpo_cmd[dpo_cmd.index("--reference") + 1] == str(policy)

    def test_pipeline_passes_per_run_training_tuning(self):
        tuning = TrainingTuning(
            sft_epochs=4,
            sft_learning_rate=2e-5,
            sft_warmup_steps=3,
            dpo_epochs=5,
            dpo_learning_rate=3e-5,
            dpo_beta=0.2,
            synthetic_ratio=0.5,
            synthetic_pool_multiplier=7.0,
            synthetic_reward_margin=0.3,
            synthetic_min_token_edits=4,
            max_grad_norm=2.0,
            early_stopping_patience=2,
        )
        steps = {desc: cmd for cmd, desc in _pipeline(tuning=tuning)}
        sft_cmd = steps["SFT fine-tuning"]
        synthetic_cmd = steps["generate synthetic DPO data"]
        dpo_cmd = steps["DPO fine-tuning"]

        assert sft_cmd[sft_cmd.index("--epochs") + 1] == "4"
        assert sft_cmd[sft_cmd.index("--learning_rate") + 1] == "2e-05"
        assert sft_cmd[sft_cmd.index("--warmup_steps") + 1] == "3"
        assert dpo_cmd[dpo_cmd.index("--epochs") + 1] == "5"
        assert dpo_cmd[dpo_cmd.index("--learning_rate") + 1] == "3e-05"
        assert dpo_cmd[dpo_cmd.index("--dpo_beta") + 1] == "0.2"
        assert synthetic_cmd[synthetic_cmd.index("--pool_multiplier") + 1] == "7.0"
        assert (
            synthetic_cmd[synthetic_cmd.index("--minimum_reward_margin") + 1] == "0.3"
        )
        assert synthetic_cmd[synthetic_cmd.index("--minimum_token_edits") + 1] == "4"
        assert dpo_cmd[dpo_cmd.index("--max_synthetic_ratio") + 1] == "0.5"
        assert sft_cmd[sft_cmd.index("--max_grad_norm") + 1] == "2.0"
        assert dpo_cmd[dpo_cmd.index("--max_grad_norm") + 1] == "2.0"
        assert sft_cmd[sft_cmd.index("--early_stopping_patience") + 1] == "2"
        assert dpo_cmd[dpo_cmd.index("--early_stopping_patience") + 1] == "2"

    def test_zero_synthetic_ratio_skips_generation(self):
        steps = _pipeline(tuning=TrainingTuning(synthetic_ratio=0.0))

        assert "generate synthetic DPO data" not in _step_names(steps)
        dpo_cmd = dict((description, cmd) for cmd, description in steps)[
            "DPO fine-tuning"
        ]
        assert "--synthetic_data" not in dpo_cmd


def test_failed_automatic_round_waits_for_new_feedback():
    class Store:
        settings = {
            "training_status": "failed",
            "training_requested": "false",
            "training_threshold": "50",
            "annotations_at_last_training": "0",
            "training_failed_at_annotations": "60",
        }

        def get_setting(self, key, default=None):
            return self.settings.get(key, default)

        def get_stats(self):
            return {"actionable_annotations": 60}

    should_train, reason = check_trigger(Store())

    assert should_train is False
    assert "0/50" in reason


@pytest.mark.parametrize(("success", "expected_failed"), [(True, "0"), (False, "7")])
def test_run_training_once_persists_annotation_watermark(
    tmp_path, monkeypatch, success, expected_failed
) -> None:
    store = FeedbackStore(tmp_path / "feedback.sqlite")
    store.get_stats = lambda: {"actionable_annotations": 7}
    monkeypatch.setattr(coordinator, "run_training", lambda active_store: success)

    assert run_training_once(store) is success
    assert store.get_setting("training_failed_at_annotations") == expected_failed
    if success:
        assert store.get_setting("annotations_at_last_training") == "7"


def test_coordinator_heartbeat_is_persisted(tmp_path) -> None:
    store = FeedbackStore(tmp_path / "feedback.sqlite")

    timestamp = update_coordinator_heartbeat(store)

    assert timestamp.endswith("Z")
    assert store.get_setting("coordinator_heartbeat_at") == timestamp


def test_embedded_diagnostics_failure_does_not_escape(monkeypatch) -> None:
    class StopEvent:
        stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, timeout):
            self.stopped = True
            return True

    class FailingCollector:
        def __init__(self, *args, **kwargs):
            pass

        def tick(self):
            raise RuntimeError("telemetry failed")

    stop_event = StopEvent()

    run_embedded_diagnostics(stop_event, collector_factory=FailingCollector)

    assert stop_event.stopped is True


def test_trigger_progress_uses_latest_attempt_watermark():
    class Store:
        settings = {
            "training_threshold": "50",
            "annotations_at_last_training": "791",
            "training_failed_at_annotations": "843",
        }

        def get_setting(self, key, default=None):
            return self.settings.get(key, default)

        def get_stats(self):
            return {"actionable_annotations": 871}

    assert coordinator.training_trigger_progress(Store()) == (28, 50)


def test_requested_wer_monitor_evaluation_records_completion(
    tmp_path, monkeypatch
) -> None:
    store = FeedbackStore(tmp_path / "feedback.sqlite")
    store.set_setting("wer_monitor_evaluation_requested", "true")
    store.set_setting("wer_monitor_evaluation_canary_set_id", "monitor-123")
    observed = {}

    def fake_run(command, cwd, timeout, target_store, log_key):
        observed["command"] = command
        observed["log_key"] = log_key
        return 0, "completed"

    monkeypatch.setattr(coordinator, "_run_step_streaming", fake_run)

    assert coordinator.run_wer_monitor_evaluation(store) is True
    assert "run_evaluate_canary.py" in observed["command"]
    assert "--if-ready" in observed["command"]
    assert observed["command"][-2:] == ["--canary-set-id", "monitor-123"]
    assert observed["log_key"] == "wer_monitor_evaluation_log"
    assert store.get_setting("wer_monitor_evaluation_requested") == "false"
    assert store.get_setting("wer_monitor_evaluation_status") == "completed"


def test_reward_audit_uses_dedicated_scoring_entrypoint(tmp_path, monkeypatch) -> None:
    store = FeedbackStore(tmp_path / "feedback.sqlite")
    observed = {}

    def fake_run(command, cwd, timeout, target_store):
        observed["command"] = command
        return 0, "completed"

    monkeypatch.setattr(coordinator, "_run_step_streaming", fake_run)

    assert coordinator.run_reward_audit(store) is True
    assert observed["command"][-1] == "run_reward_audit.py"


def test_setup_failure_is_persisted_for_admin_panel(tmp_path, monkeypatch) -> None:
    store = FeedbackStore(tmp_path / "feedback.sqlite")

    def fail_policy_resolution(*args, **kwargs):
        raise RuntimeError("broken policy metadata")

    monkeypatch.setattr(
        coordinator, "resolve_active_policy_checkpoint", fail_policy_resolution
    )

    assert coordinator.run_training(store) is False
    assert store.get_setting("training_status") == "failed"
    assert store.get_setting("training_error") == (
        "Unexpected pipeline error: RuntimeError: broken policy metadata"
    )


def test_training_repairs_base_policy_identity_before_feedback_filtering(
    tmp_path, monkeypatch
) -> None:
    from storage.schema import HypothesisPair
    from tests.conftest import annotate, make_audio, make_session

    store = FeedbackStore(tmp_path / "feedback.sqlite")
    audio = make_audio(store)
    session = make_session(store)
    pair = HypothesisPair.new(
        audio_id=audio.id,
        hyp_a="a",
        hyp_b="b",
        strategy_a="beam",
        strategy_b="sample",
        model_version=coordinator.DEFAULT_PATHS.whisper_model_id,
    )
    store.insert_hypothesis_pair(pair)
    annotate(store, session.id, pair.id, "a", correction_text="corrected")
    observed = {}

    monkeypatch.setattr(
        coordinator, "resolve_active_policy_checkpoint", lambda *args, **kwargs: None
    )

    def fake_pipeline(*args, **kwargs):
        observed["include_sft"] = kwargs["include_sft"]
        return []

    monkeypatch.setattr(coordinator, "_pipeline", fake_pipeline)

    assert coordinator.run_training(store) is True
    assert store.get_setting("active_policy_model_version") == (
        coordinator.DEFAULT_PATHS.whisper_model_id
    )
    assert observed["include_sft"] is True


def test_policy_gate_rejection_is_guarded_not_failed(tmp_path, monkeypatch) -> None:
    store = FeedbackStore(tmp_path / "feedback.sqlite")
    monkeypatch.setattr(
        coordinator, "resolve_active_policy_checkpoint", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        coordinator,
        "_pipeline",
        lambda *args, **kwargs: [
            (["python", "run_evaluate_canary.py"], "model-evaluation policy promotion")
        ],
    )
    monkeypatch.setattr(
        coordinator,
        "_run_step_streaming",
        lambda *args, **kwargs: (3, "[REJECT] Candidate WER regression blocked."),
    )

    assert coordinator.run_training(store) is True
    assert store.get_setting("training_status") == "guarded"
    assert store.get_setting("training_error") == ""
    assert store.get_setting("training_step") == "model-evaluation policy promotion"


def test_legacy_policy_gate_rejection_is_normalized(tmp_path) -> None:
    store = FeedbackStore(tmp_path / "feedback.sqlite")
    store.set_setting("training_status", "failed")
    store.set_setting("promotion_gate_status", "rejected")
    store.set_setting(
        "training_error",
        "Step 'model-evaluation policy promotion' failed (exit code 3).\n\n[REJECT]",
    )

    assert coordinator.normalize_legacy_gate_rejection(store) is True
    assert store.get_setting("training_status") == "guarded"
    assert store.get_setting("training_error") == ""


def test_coordinator_startup_queues_existing_frozen_monitor(tmp_path) -> None:
    from storage.schema import CanarySet
    from tests.conftest import make_audio

    store = FeedbackStore(tmp_path / "feedback.sqlite")
    make_audio(store, "audio/monitor.wav")
    monitor = CanarySet.new("WER Monitor", 3, "random_unannotated", target_size=1)
    item = store.create_canary_set(monitor)[0]
    store.save_canary_transcript(item.id, "tekst")
    store.freeze_canary_set(monitor.id)
    store.set_setting("wer_monitor_evaluation_requested", "false")
    store.set_setting("wer_monitor_evaluation_status", "idle")

    assert coordinator.queue_existing_wer_monitor_evaluation(store) is True
    assert store.get_setting("wer_monitor_evaluation_requested") == "true"
    assert store.get_setting("wer_monitor_evaluation_status") == "queued"
