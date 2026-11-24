from __future__ import annotations

import json

import pytest

from evaluation.model_health import (
    aggregate_metrics,
    classify_regression,
    compute_clip_metrics,
    normalize_lexical,
    paired_bootstrap_delta,
    paired_bootstrap_corpus_wer,
    policy_promotion_decision,
    policy_promotion_from_predictions,
    profile_hash,
)
from evaluation.prediction_diff import prediction_difference


def test_lexical_normalization_ignores_case_and_punctuation() -> None:
    assert normalize_lexical("Zastawka, prawidłowa.") == "zastawka prawidłowa"


def test_fragmentation_and_hyphen_metrics_detect_degenerate_text() -> None:
    clean = compute_clip_metrics("pozostały zastawki", "pozostały zastawki")
    fragmented = compute_clip_metrics(
        "pozostały zastawki",
        "Po-o-sta-ły za-sta-w-ki",
    )

    assert fragmented["hyphens_per_100_chars"] > clean["hyphens_per_100_chars"]
    assert fragmented["fragmented_token_rate"] > clean["fragmented_token_rate"]


def test_aggregate_metrics_use_corpus_wer() -> None:
    result = aggregate_metrics(
        ["jeden dwa", "trzy"],
        ["jeden", "trzy"],
    )

    assert result["n_clips"] == 2
    assert result["wer"] == pytest.approx(1 / 3)
    assert result["metric_version"] == 1


def test_regression_classifier_reports_accuracy_and_formatting() -> None:
    baseline = {"wer": 0.1, "cer": 0.05, "fragmented_token_rate": 0.01}
    candidate = {"wer": 0.15, "cer": 0.08, "fragmented_token_rate": 0.04}

    assert classify_regression(baseline, candidate) == [
        "wer_increase",
        "cer_increase",
        "fragmentation_increase",
    ]


def test_profile_hash_is_stable_and_sensitive() -> None:
    assert profile_hash() == profile_hash()
    assert profile_hash({"num_beams": 5}) != profile_hash({"num_beams": 1})


def test_paired_bootstrap_delta_is_deterministic() -> None:
    result = paired_bootstrap_delta(
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.2, 0.5, 0.5],
        iterations=500,
    )

    assert result["delta"] == pytest.approx(0.1)
    assert result["ci_low"] <= result["delta"] <= result["ci_high"]
    assert result == paired_bootstrap_delta(
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.2, 0.5, 0.5],
        iterations=500,
    )


def test_corpus_wer_bootstrap_weights_clips_by_reference_words() -> None:
    references = ["a b c d", "x"]
    baseline = ["a b", "x"]
    candidate = ["a b c d", "y"]

    comparison = paired_bootstrap_corpus_wer(
        references, baseline, candidate, iterations=500
    )
    mean_clip_delta = sum(
        compute_clip_metrics(reference, hypothesis)["wer"]
        - compute_clip_metrics(reference, old_hypothesis)["wer"]
        for reference, old_hypothesis, hypothesis in zip(
            references, baseline, candidate
        )
    ) / len(references)

    assert mean_clip_delta > 0
    assert comparison["metric"] == "corpus_wer"
    assert comparison["delta"] == pytest.approx(-0.2)


def test_policy_promotion_uses_corpus_wer_when_transcripts_are_available() -> None:
    references = ["a b c d", "x"]
    active = [
        {
            "canary_item_id": f"clip-{index}",
            "transcript": reference,
            "hypothesis": hypothesis,
            "metrics_json": json.dumps(compute_clip_metrics(reference, hypothesis)),
        }
        for index, (reference, hypothesis) in enumerate(zip(references, ["a b", "x"]))
    ]
    candidate = [
        {
            "canary_item_id": f"clip-{index}",
            "transcript": reference,
            "hypothesis": hypothesis,
            "metrics_json": json.dumps(compute_clip_metrics(reference, hypothesis)),
        }
        for index, (reference, hypothesis) in enumerate(
            zip(references, ["a b c d", "y"])
        )
    ]

    decision = policy_promotion_from_predictions(active, candidate)

    assert decision["metric"] == "corpus_wer"
    assert decision["delta"] == pytest.approx(-0.2)


def test_policy_promotion_rejects_regression_against_active_policy() -> None:
    decision = policy_promotion_decision(
        [0.10, 0.10, 0.10, 0.10],
        [0.12, 0.12, 0.12, 0.12],
    )

    assert decision["promote"] is False
    assert decision["ci_high"] > decision["max_wer_regression"]


def test_policy_promotion_accepts_non_regressing_candidate() -> None:
    decision = policy_promotion_decision(
        [0.20, 0.20, 0.20, 0.20],
        [0.19, 0.19, 0.19, 0.19],
    )

    assert decision["promote"] is True


def test_policy_promotion_requires_matching_monitor_clips() -> None:
    active = [
        {
            "canary_item_id": "clip-a",
            "metrics_json": '{"wer": 0.1}',
        }
    ]
    candidate = [
        {
            "canary_item_id": "clip-b",
            "metrics_json": '{"wer": 0.1}',
        }
    ]

    with pytest.raises(ValueError, match="same monitor clips"):
        policy_promotion_from_predictions(active, candidate)


def test_policy_promotion_rejects_behaviorally_unchanged_candidate() -> None:
    active = [
        {
            "canary_item_id": "clip-a",
            "hypothesis": "bez zmian",
            "metrics_json": '{"wer": 0.1}',
        }
    ]
    candidate = [
        {
            "canary_item_id": "clip-a",
            "hypothesis": "bez zmian",
            "metrics_json": '{"wer": 0.1}',
        }
    ]

    decision = policy_promotion_from_predictions(active, candidate)

    assert decision["promote"] is False
    assert decision["changed_predictions"] == 0
    assert decision["compared_predictions"] == 1


def test_policy_promotion_accepts_changed_non_regressing_candidate() -> None:
    active = [
        {
            "canary_item_id": "clip-a",
            "hypothesis": "stara",
            "metrics_json": '{"wer": 0.1}',
        }
    ]
    candidate = [
        {
            "canary_item_id": "clip-a",
            "hypothesis": "nowa",
            "metrics_json": '{"wer": 0.1}',
        }
    ]

    decision = policy_promotion_from_predictions(active, candidate)

    assert decision["promote"] is True
    assert decision["changed_predictions"] == 1


def test_prediction_difference_counts_changed_shared_clips() -> None:
    previous = [
        {"canary_item_id": "a", "hypothesis": "bez zmian"},
        {"canary_item_id": "b", "hypothesis": "stara"},
    ]
    current = [
        {"canary_item_id": "b", "hypothesis": "nowa"},
        {"canary_item_id": "a", "hypothesis": "bez zmian"},
    ]

    assert prediction_difference(previous, current) == {
        "changed": 1,
        "total": 2,
        "changed_ids": {"b"},
    }
