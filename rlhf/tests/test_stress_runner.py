from pathlib import Path
import importlib.util
import sys

import pytest


_MODULE_PATH = (
    Path(__file__).parents[1] / "stress_tests" / "load" / "simulate_month.py"
)
_SPEC = importlib.util.spec_from_file_location("simulate_month", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
simulate_month = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = simulate_month
_SPEC.loader.exec_module(simulate_month)


def test_remote_url_requires_https_origin() -> None:
    target = "https://rlhf-asr.ingress.caise.k8s.apl.task.gda.pl"

    assert simulate_month._validate_remote_url(target + "/") == target
    with pytest.raises(ValueError, match="HTTPS origin"):
        simulate_month._validate_remote_url("http://example.test")
    with pytest.raises(ValueError, match="HTTPS origin"):
        simulate_month._validate_remote_url("https://example.test/path")


def test_resume_rejects_different_target_or_execution_mode() -> None:
    checkpoint = {
        "base_url": "https://target.example.test",
        "remote": True,
        "k6_runtime": "docker",
    }

    simulate_month._validate_resume_identity(
        checkpoint,
        "https://target.example.test",
        True,
        "docker",
    )
    with pytest.raises(ValueError, match="base-url differs"):
        simulate_month._validate_resume_identity(
            checkpoint,
            "https://other.example.test",
            True,
            "docker",
        )
    with pytest.raises(ValueError, match="remote mode differs"):
        simulate_month._validate_resume_identity(
            checkpoint,
            "https://target.example.test",
            False,
            "docker",
        )


def test_missing_resume_checkpoint_fails_before_external_access(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simulate_month.py",
            "--remote",
            "--resume",
            "--base-url",
            "https://target.example.test",
            "--log-dir",
            str(tmp_path),
        ],
    )
    monkeypatch.setattr(
        simulate_month,
        "_assert_docker",
        lambda: pytest.fail("Docker must not be accessed"),
    )
    monkeypatch.setattr(
        simulate_month,
        "_preflight",
        lambda *args: pytest.fail("target must not be accessed"),
    )

    with pytest.raises(SystemExit) as exc_info:
        simulate_month.main()

    assert exc_info.value.code == 2


def test_runner_records_failed_bursts_and_continues_by_default() -> None:
    source = _MODULE_PATH.read_text(encoding="utf-8")

    assert 'failed_burst = result["exit_code"] != 0' in source
    assert '"next_event_idx": idx + 1' in source
    assert "failed_burst and args.fail_fast" in source
    assert 'parser.add_argument("--fail-fast"' in source
    assert "if not args.remote:\n            _preflight" in source
    assert "new remote runs require --run-id" not in source


def test_runner_exports_structured_k6_summary_per_burst() -> None:
    source = _MODULE_PATH.read_text(encoding="utf-8")

    assert 'summary_json = day_dir / f"{artifact_stem}_summary.json"' in source
    assert '"--summary-export", summary_target' in source
    assert '"summary":         str(summary_json)' in source


def test_exact_daily_burst_count_preserves_zone_profile() -> None:
    schedule = simulate_month.generate_schedule(
        days=2,
        rng=simulate_month.random.Random(42),
        bursts_per_day=48,
    )

    assert len(schedule) == 96
    for day in range(2):
        day_events = [event for event in schedule if event.day == day]
        assert len(day_events) == 48
        assert sum(event.zone == "night" for event in day_events) == 9
        assert sum(event.zone == "work" for event in day_events) == 26
        assert sum(event.zone == "evening" for event in day_events) == 13


def test_average_daily_burst_count_preserves_poisson_variation() -> None:
    days = 100
    schedule = simulate_month.generate_schedule(
        days=days,
        rng=simulate_month.random.Random(42),
        average_bursts_per_day=48,
    )
    daily_counts = [sum(event.day == day for event in schedule) for day in range(days)]

    assert abs(len(schedule) / days - 48) < 1.5
    assert len(set(daily_counts)) > 1
    assert sum(event.zone == "work" for event in schedule) > sum(
        event.zone == "night" for event in schedule
    )