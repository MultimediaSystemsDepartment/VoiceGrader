"""Versioned canary metrics for lexical accuracy and formatting degeneration."""

from __future__ import annotations

import hashlib
import json
import random
import re
import string
from statistics import mean
from typing import Sequence

import jiwer

METRIC_VERSION = 1
CANARY_DECODE_PROFILE = {
    "version": 1,
    "language": "pl",
    "task": "transcribe",
    "num_beams": 5,
    "do_sample": False,
}


def profile_hash(profile: dict | None = None) -> str:
    payload = profile or CANARY_DECODE_PROFILE
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def normalize_lexical(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _rate(count: int, denominator: int) -> float:
    return count * 100.0 / max(denominator, 1)


def compute_clip_metrics(reference: str, hypothesis: str) -> dict[str, float]:
    normalized_reference = normalize_lexical(reference)
    normalized_hypothesis = normalize_lexical(hypothesis)
    tokens = hypothesis.split()
    fragmented = sum(
        token.count("-") >= 2
        or bool(re.search(r"(?:^|-)\w(?:-\w){2,}(?:-|$)", token, re.UNICODE))
        for token in tokens
    )
    dotted_abbreviations = len(
        re.findall(r"(?:\b\w\.){2,}", hypothesis, flags=re.UNICODE)
    )
    punctuation = sum(
        character in string.punctuation and character != "-" for character in hypothesis
    )
    reference_length = len(normalized_reference)
    return {
        "wer": float(jiwer.wer(normalized_reference, normalized_hypothesis)),
        "cer": float(jiwer.cer(normalized_reference, normalized_hypothesis)),
        "raw_cer": float(jiwer.cer(reference, hypothesis)),
        "length_ratio": len(normalized_hypothesis) / max(reference_length, 1),
        "hyphens_per_100_chars": _rate(hypothesis.count("-"), len(hypothesis)),
        "punctuation_per_100_chars": _rate(punctuation, len(hypothesis)),
        "dotted_abbreviations_per_100_chars": _rate(
            dotted_abbreviations, len(hypothesis)
        ),
        "fragmented_token_rate": fragmented / max(len(tokens), 1),
    }


def aggregate_metrics(
    references: Sequence[str], hypotheses: Sequence[str]
) -> dict[str, float | int]:
    if len(references) != len(hypotheses):
        raise ValueError("Reference and hypothesis counts must match")
    if not references:
        raise ValueError("Cannot aggregate an empty canary evaluation")
    clips = [
        compute_clip_metrics(reference, hypothesis)
        for reference, hypothesis in zip(references, hypotheses)
    ]
    normalized_references = [normalize_lexical(value) for value in references]
    normalized_hypotheses = [normalize_lexical(value) for value in hypotheses]
    averaged = {
        key: mean(float(clip[key]) for clip in clips)
        for key in clips[0]
        if key not in {"wer", "cer"}
    }
    return {
        "metric_version": METRIC_VERSION,
        "n_clips": len(clips),
        "wer": float(jiwer.wer(normalized_references, normalized_hypotheses)),
        "cer": float(jiwer.cer(normalized_references, normalized_hypotheses)),
        "mean_clip_wer": mean(float(clip["wer"]) for clip in clips),
        "mean_clip_cer": mean(float(clip["cer"]) for clip in clips),
        **averaged,
    }


def classify_regression(
    baseline: dict,
    candidate: dict,
    wer_delta_warning: float = 0.03,
    cer_delta_warning: float = 0.02,
    fragmentation_multiplier: float = 1.5,
    fragmentation_floor: float = 0.02,
) -> list[str]:
    reasons = []
    if candidate["wer"] - baseline["wer"] >= wer_delta_warning:
        reasons.append("wer_increase")
    if candidate["cer"] - baseline["cer"] >= cer_delta_warning:
        reasons.append("cer_increase")
    baseline_fragmentation = max(
        float(baseline["fragmented_token_rate"]), fragmentation_floor
    )
    if candidate["fragmented_token_rate"] >= (
        baseline_fragmentation * fragmentation_multiplier
    ):
        reasons.append("fragmentation_increase")
    return reasons


def paired_bootstrap_delta(
    baseline: Sequence[float],
    candidate: Sequence[float],
    iterations: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("Paired metric vectors must be non-empty and equal length")
    deltas = [right - left for left, right in zip(baseline, candidate)]
    rng = random.Random(seed)
    samples = sorted(
        mean(deltas[rng.randrange(len(deltas))] for _ in deltas)
        for _ in range(iterations)
    )
    return {
        "delta": mean(deltas),
        "ci_low": samples[int(0.025 * (iterations - 1))],
        "ci_high": samples[int(0.975 * (iterations - 1))],
    }


def _word_error_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    result = jiwer.process_words(
        normalize_lexical(reference), normalize_lexical(hypothesis)
    )
    errors = result.substitutions + result.deletions + result.insertions
    reference_words = result.hits + result.substitutions + result.deletions
    return errors, reference_words


def paired_bootstrap_corpus_wer(
    references: Sequence[str],
    baseline_hypotheses: Sequence[str],
    candidate_hypotheses: Sequence[str],
    iterations: int = 2000,
    seed: int = 42,
) -> dict[str, float | str]:
    if not references or not (
        len(references) == len(baseline_hypotheses) == len(candidate_hypotheses)
    ):
        raise ValueError("Paired transcripts must be non-empty and equal length")
    clips = []
    for reference, baseline, candidate in zip(
        references, baseline_hypotheses, candidate_hypotheses
    ):
        baseline_errors, reference_words = _word_error_counts(reference, baseline)
        candidate_errors, candidate_reference_words = _word_error_counts(
            reference, candidate
        )
        if reference_words != candidate_reference_words:
            raise ValueError("Reference word counts must match within each pair")
        clips.append((baseline_errors, candidate_errors, reference_words))

    def delta(indices: Sequence[int]) -> float:
        reference_words = sum(clips[index][2] for index in indices)
        if reference_words == 0:
            raise ValueError("Corpus WER requires at least one reference word")
        baseline_errors = sum(clips[index][0] for index in indices)
        candidate_errors = sum(clips[index][1] for index in indices)
        return (candidate_errors - baseline_errors) / reference_words

    rng = random.Random(seed)
    samples = sorted(
        delta([rng.randrange(len(clips)) for _ in clips]) for _ in range(iterations)
    )
    return {
        "metric": "corpus_wer",
        "delta": delta(range(len(clips))),
        "ci_low": samples[int(0.025 * (iterations - 1))],
        "ci_high": samples[int(0.975 * (iterations - 1))],
    }


def paired_corpus_wer_from_predictions(
    baseline_predictions: Sequence[dict],
    candidate_predictions: Sequence[dict],
) -> dict[str, float | str]:
    baseline = {row["canary_item_id"]: row for row in baseline_predictions}
    candidate = {row["canary_item_id"]: row for row in candidate_predictions}
    if not baseline or baseline.keys() != candidate.keys():
        raise ValueError("Promotion evaluations must contain the same monitor clips")
    ordered_ids = sorted(baseline)
    return paired_bootstrap_corpus_wer(
        [baseline[item_id]["transcript"] for item_id in ordered_ids],
        [baseline[item_id]["hypothesis"] for item_id in ordered_ids],
        [candidate[item_id]["hypothesis"] for item_id in ordered_ids],
    )


def policy_promotion_decision(
    active_clip_wer: Sequence[float],
    candidate_clip_wer: Sequence[float],
    max_wer_regression: float = 0.01,
) -> dict[str, float | bool]:
    comparison = paired_bootstrap_delta(active_clip_wer, candidate_clip_wer)
    return {
        **comparison,
        "max_wer_regression": max_wer_regression,
        "promote": comparison["ci_high"] <= max_wer_regression,
    }


def policy_promotion_from_predictions(
    active_predictions: Sequence[dict],
    candidate_predictions: Sequence[dict],
    max_wer_regression: float = 0.01,
    min_changed_predictions: int = 1,
) -> dict[str, float | int | bool]:
    active = {
        row["canary_item_id"]: json.loads(row["metrics_json"])["wer"]
        for row in active_predictions
    }
    candidate = {
        row["canary_item_id"]: json.loads(row["metrics_json"])["wer"]
        for row in candidate_predictions
    }
    if not active or active.keys() != candidate.keys():
        raise ValueError("Promotion evaluations must contain the same monitor clips")
    ordered_ids = sorted(active)
    if all(
        "transcript" in row for row in (*active_predictions, *candidate_predictions)
    ):
        comparison = paired_corpus_wer_from_predictions(
            active_predictions, candidate_predictions
        )
        decision = {
            **comparison,
            "max_wer_regression": max_wer_regression,
            "promote": comparison["ci_high"] <= max_wer_regression,
        }
    else:
        decision = policy_promotion_decision(
            [active[item_id] for item_id in ordered_ids],
            [candidate[item_id] for item_id in ordered_ids],
            max_wer_regression=max_wer_regression,
        )
    active_hypotheses = {
        row["canary_item_id"]: row["hypothesis"] for row in active_predictions
    }
    candidate_hypotheses = {
        row["canary_item_id"]: row["hypothesis"] for row in candidate_predictions
    }
    changed_predictions = sum(
        active_hypotheses[item_id] != candidate_hypotheses[item_id]
        for item_id in ordered_ids
    )
    return {
        **decision,
        "promote": (
            decision["promote"] and changed_predictions >= min_changed_predictions
        ),
        "changed_predictions": changed_predictions,
        "compared_predictions": len(ordered_ids),
        "min_changed_predictions": min_changed_predictions,
    }
