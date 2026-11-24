from pathlib import Path

import pytest

from gpu_stress_tester.config import RunConfig, profile_summaries, resolve_config


def test_resolve_profile_with_overrides() -> None:
    config = resolve_config("smoke", rounds=3, samples=12)

    assert config.profile == "smoke"
    assert config.rounds == 3
    assert config.samples == 12


def test_config_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    expected = resolve_config("standard", seed=7)

    expected.to_json(path)

    assert RunConfig.from_json(path) == expected


def test_config_rejects_unsafe_learning_rate() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        resolve_config("standard", learning_rate=0.5)


def test_profile_summaries_match_runtime_defaults() -> None:
    profiles = profile_summaries()

    assert profiles["smoke"]["samples"] == 8
    assert profiles["standard"]["batch_size"] == 2
    assert profiles["endurance"]["rounds"] == 100_000
    assert profiles["max-load"]["batch_size"] == 4