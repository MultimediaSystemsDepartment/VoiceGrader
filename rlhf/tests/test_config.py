"""
Tests for configuration dataclasses: RLHFConfig, InferenceConfig.

No GPU, no Whisper, no I/O beyond tmp_path JSON roundtrip.

Regression coverage for:
- InferenceConfig default datasets was ["admedvoice", "commonvoice"] — changed to
  ["admedvoice"] only for the PoC scope. This test pins that intent.
- PPO fields (ppo_epochs, clip_range, vf_coef) were removed from RLHFConfig.
  Tests verify they are gone and don't silently reappear.
- RLHFConfig.from_json() must ignore unknown keys (forward-compatible) rather
  than raising TypeError.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest

from inference.config import InferenceConfig
from paths import PROJECT_ROOT, PathsConfig, load_paths
from rlhf.config import RLHFConfig
from utils.training_tuning import TrainingTuning

# ---------------------------------------------------------------------------
# InferenceConfig defaults
# ---------------------------------------------------------------------------


class TestInferenceConfigDefaults:
    def test_default_datasets_is_admedvoice_only(self):
        """Regression: was ['admedvoice', 'commonvoice'] — must stay admedvoice-only."""
        config = InferenceConfig()
        assert config.datasets == ["admedvoice"]
        assert "commonvoice" not in config.datasets

    def test_default_splits_is_train(self):
        config = InferenceConfig()
        assert config.splits == ["train"]


class TestInferenceConfigRoundtrip:
    def test_json_roundtrip_preserves_datasets(self, tmp_path):
        cfg = InferenceConfig(datasets=["admedvoice"], splits=["train"])
        path = tmp_path / "inf.json"
        cfg.to_json(path)

        loaded = InferenceConfig.from_json(path)
        assert loaded.datasets == ["admedvoice"]
        assert loaded.splits == ["train"]

    def test_from_json_ignores_unknown_keys(self, tmp_path):
        """Forward-compatible: extra keys in JSON must not raise TypeError."""
        data = {
            "model_id": "openai/whisper-medium",
            "datasets": ["admedvoice"],
            "future_unknown_key": "some value",
        }
        path = tmp_path / "inf.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        config = InferenceConfig.from_json(path)
        assert config.datasets == ["admedvoice"]


def test_latest_model_directory_is_configurable(tmp_path) -> None:
    config_path = tmp_path / "paths.json"
    config_path.write_text(
        json.dumps({"latest_model_dir": "shared/latest"}), encoding="utf-8"
    )

    config = load_paths(config_path)

    assert (
        Path(config.latest_model_dir)
        == Path(__file__).resolve().parents[1] / "shared/latest"
    )
    assert Path(PathsConfig().latest_model_dir).name == "latest"


def test_default_manifest_paths_are_portable_to_host_checkout() -> None:
    config = load_paths()

    assert Path(config.admedvoice_train_csv) == (
        PROJECT_ROOT / "_data/manifests/admedvoice_train.csv"
    )
    assert Path(config.admedvoice_test_csv) == (
        PROJECT_ROOT / "_data/manifests/admedvoice_test.csv"
    )


def test_paths_environment_override_isolates_mutable_outputs(tmp_path: Path) -> None:
    config_path = tmp_path / "arm" / "paths.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "db_path": "outputs/feedback.sqlite",
                "checkpoints_dir": "outputs/checkpoints",
                "reward_model_dir": "outputs/reward_model",
                "logs_dir": "outputs/logs",
            }
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["RLHF_PATHS_FILE"] = str(config_path)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from paths import DEFAULT_PATHS; "
            "print(json.dumps(DEFAULT_PATHS.__dict__))",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    paths = json.loads(result.stdout)

    assert paths["db_path"] == str(config_path.parent / "outputs/feedback.sqlite")
    assert paths["checkpoints_dir"] == str(
        config_path.parent / "outputs/checkpoints"
    )
    assert paths["reward_model_dir"] == str(
        config_path.parent / "outputs/reward_model"
    )
    assert paths["logs_dir"] == str(config_path.parent / "outputs/logs")


# ---------------------------------------------------------------------------
# RLHFConfig defaults
# ---------------------------------------------------------------------------


class TestRLHFConfigDefaults:
    def test_default_algorithm_is_dpo(self):
        config = RLHFConfig()
        assert config.algorithm == "dpo"

    def test_default_language_is_polish(self):
        config = RLHFConfig()
        assert config.language == "pl"

    def test_default_lora_targets_all_matching_projections(self):
        """LoRA targets all Whisper q_proj/v_proj modules through PEFT suffix matching."""
        config = RLHFConfig()
        assert set(config.lora_target_modules) == {"q_proj", "v_proj"}

    def test_sdpa_is_the_default_attention_backend(self):
        assert RLHFConfig().attn_implementation == "sdpa"
        assert InferenceConfig().attn_implementation == "sdpa"

    def test_dpo_defaults_preserve_reference_consistency(self):
        config = RLHFConfig()

        assert config.reference_in_4bit is False
        assert config.learning_rate == pytest.approx(1e-5)
        assert config.num_epochs == 2
        assert config.max_synthetic_to_human_ratio == pytest.approx(1.0)
        assert config.min_reward_scores == (
            config.batch_size * config.gradient_accumulation_steps
        )


class TestRLHFConfigNoPPOFields:
    """Regression: PPO fields were removed — ensure they don't silently reappear."""

    def test_ppo_epochs_field_does_not_exist(self):
        field_names = {f.name for f in fields(RLHFConfig)}
        assert "ppo_epochs" not in field_names

    def test_clip_range_field_does_not_exist(self):
        field_names = {f.name for f in fields(RLHFConfig)}
        assert "clip_range" not in field_names

    def test_vf_coef_field_does_not_exist(self):
        field_names = {f.name for f in fields(RLHFConfig)}
        assert "vf_coef" not in field_names


class TestRLHFConfigRoundtrip:
    def test_json_roundtrip_preserves_all_fields(self, tmp_path):
        cfg = RLHFConfig(lora_r=8, dpo_beta=0.05, language="pl")
        path = tmp_path / "rlhf.json"
        cfg.to_json(path)

        loaded = RLHFConfig.from_json(path)
        assert loaded.lora_r == 8
        assert loaded.dpo_beta == 0.05
        assert loaded.language == "pl"
        assert loaded.algorithm == "dpo"


def test_training_tuning_roundtrips_through_settings() -> None:
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

    assert TrainingTuning.from_mapping(tuning.as_settings()) == tuning


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("training_sft_epochs", "0"),
        ("training_sft_learning_rate", "0.1"),
        ("training_sft_warmup_steps", "101"),
        ("training_dpo_epochs", "11"),
        ("training_dpo_learning_rate", "bad"),
        ("training_dpo_beta", "0"),
        ("training_synthetic_ratio", "2.1"),
        ("training_synthetic_pool_multiplier", "1"),
        ("training_synthetic_reward_margin", "-1"),
        ("training_synthetic_min_token_edits", "0"),
        ("training_max_grad_norm", "5.1"),
        ("training_early_stopping_patience", "11"),
    ],
)
def test_training_tuning_rejects_unsafe_values(key, value) -> None:
    values = TrainingTuning().as_settings()
    values[key] = value

    with pytest.raises(ValueError):
        TrainingTuning.from_mapping(values)

    def test_from_json_ignores_unknown_keys(self, tmp_path):
        """Config files written with old PPO fields must not crash on load."""
        data = {
            "algorithm": "dpo",
            "lora_r": 16,
            "ppo_epochs": 4,  # removed field — must be silently ignored
            "clip_range": 0.2,  # removed field — must be silently ignored
        }
        path = tmp_path / "rlhf.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        config = RLHFConfig.from_json(path)
        assert config.algorithm == "dpo"
        assert config.lora_r == 16

    def test_to_json_str_is_valid_json(self):
        cfg = RLHFConfig()
        parsed = json.loads(cfg.to_json_str())
        assert parsed["algorithm"] == "dpo"
