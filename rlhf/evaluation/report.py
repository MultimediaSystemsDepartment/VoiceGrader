"""
Report generation -- per-round WER delta table from the training_runs DB table.
"""

from __future__ import annotations

from typing import List


def generate_text_report(training_runs: List[dict]) -> str:
    """
    Generate a human-readable WER delta report from training_runs rows.

    Args:
        training_runs: rows from FeedbackStore.get_training_runs()

    Returns:
        Multi-line string suitable for printing or writing to a .txt file.
    """
    if not training_runs:
        return "No training runs recorded yet."

    lines = [
        "RLHF Training History",
        "=" * 72,
        f"{'Run':<4}  {'Algorithm':<12}  {'ADMED WER delta':>12}  {'CV WER delta':>10}  "
        f"{'Annotations':>12}  {'Started':>20}",
        "-" * 72,
    ]

    prev_wer_admed: dict = {}
    prev_wer_cv: dict = {}

    for i, run in enumerate(training_runs):
        alg       = run.get("algorithm", "?")
        wer_b_a   = run.get("wer_before_admed")
        wer_a_a   = run.get("wer_after_admed")
        wer_b_c   = run.get("wer_before_cv")
        wer_a_c   = run.get("wer_after_cv")
        n_human   = run.get("n_human_annotations") or 0
        n_synth   = run.get("n_synthetic_annotations") or 0
        started   = (run.get("started_at") or "")[:19]

        def delta(before, after):
            if before is None or after is None:
                return "N/A"
            d = (after - before) * 100
            sign = "+" if d > 0 else ""
            return f"{sign}{d:.2f}pp"

        lines.append(
            f"{i + 1:<4}  {alg:<12}  {delta(wer_b_a, wer_a_a):>12}  "
            f"{delta(wer_b_c, wer_a_c):>10}  "
            f"{n_human:>6}h/{n_synth:<5}s  {started:>20}"
        )

    lines += [
        "-" * 72,
        "(h=human annotations  s=synthetic RLAIF annotations  pp=percentage points)",
    ]
    return "\n".join(lines)


def generate_markdown_report(training_runs: List[dict]) -> str:
    """Same data as generate_text_report but as a Markdown table."""
    if not training_runs:
        return "_No training runs recorded yet._"

    header = (
        "| # | Algorithm | ADMED WER before | ADMED WER after | delta | "
        "CV WER before | CV WER after | delta | Human ann. | Synth. ann. | Started |\n"
        "|---|-----------|-----------------|----------------|---|"
        "--------------|-------------|---|------------|-------------|---------|"
    )
    rows = [header]

    for i, run in enumerate(training_runs):
        def pct(v):
            return f"{v * 100:.2f}%" if v is not None else "--"
        def delta(b, a):
            if b is None or a is None:
                return "--"
            d = (a - b) * 100
            sign = "+" if d > 0 else ""
            return f"**{sign}{d:.2f}pp**"

        rows.append(
            f"| {i+1} | {run.get('algorithm','?')} "
            f"| {pct(run.get('wer_before_admed'))} | {pct(run.get('wer_after_admed'))} "
            f"| {delta(run.get('wer_before_admed'), run.get('wer_after_admed'))} "
            f"| {pct(run.get('wer_before_cv'))} | {pct(run.get('wer_after_cv'))} "
            f"| {delta(run.get('wer_before_cv'), run.get('wer_after_cv'))} "
            f"| {run.get('n_human_annotations') or 0} "
            f"| {run.get('n_synthetic_annotations') or 0} "
            f"| {(run.get('started_at') or '')[:10]} |"
        )

    return "\n".join(rows)
