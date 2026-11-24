#!/usr/bin/env python
"""
simulate_rlhf.py — end-to-end RLHF simulation with WER-based synthetic annotation.

Simulates human annotators by comparing hypothesis WER against the reference
transcript stored in the DB.  An optional error_rate flips the correct preference
with the given probability, modelling noisy human judgement.

Usage examples
--------------
# 10 rounds, 5 % annotator error, 50 pairs per round
py simulate_rlhf.py --rounds 10 --error_rate 0.05 --pairs_per_round 50

# Full 50-round ablation across all four noise levels
py simulate_rlhf.py --rounds 50 --error_rate 0.00 --tag clean
py simulate_rlhf.py --rounds 50 --error_rate 0.02 --tag noise2
py simulate_rlhf.py --rounds 50 --error_rate 0.05 --tag noise5
py simulate_rlhf.py --rounds 50 --error_rate 0.10 --tag noise10

Output
------
Console table + CSV at _outputs/simulation_<tag>.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _run(description: str, cmd: list[str], extra_env: dict | None = None) -> int:
    """Run a subprocess, stream its stdout/stderr, return exit code."""
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    print(f"\n{'='*60}")
    print(f"[{_ts()}]  {description}")
    print(f"  cmd: {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run(cmd, env=env)
    return result.returncode


def _find_latest_checkpoint(base: Path = Path("_outputs/checkpoints")) -> Path | None:
    """Return the most recent DPO checkpoint directory, or None."""
    candidates = sorted(base.glob("whisper_dpo_*"))
    return candidates[-1] if candidates else None


def _find_latest_rm_checkpoint(base: Path = Path("_outputs/reward_model")) -> Path | None:
    candidates = sorted(base.glob("reward_model_v*"))
    return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# WER-based synthetic annotation
# ---------------------------------------------------------------------------

def _wer_simple(reference: str, hypothesis: str) -> float:
    """Fast word-level WER without jiwer (no extra import in the hot path)."""
    import jiwer
    transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ])
    try:
        return jiwer.process_words(
            reference, hypothesis,
            reference_transform=transform,
            hypothesis_transform=transform,
        ).wer
    except Exception:
        # Fallback: token overlap
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        if not ref_tokens:
            return 0.0 if not hyp_tokens else 1.0
        errors = sum(a != b for a, b in zip(ref_tokens, hyp_tokens))
        errors += abs(len(ref_tokens) - len(hyp_tokens))
        return errors / len(ref_tokens)


def simulate_annotations(
    store,
    session_id: str,
    error_rate: float,
    rng: random.Random,
    limit: int,
) -> dict:
    """
    Fetch pending pairs, compare both hypotheses to the reference transcript
    using WER, then insert synthetic annotations.

    Returns a summary dict with counts.
    """
    from storage.schema import Annotation

    pairs = store.get_pending_pairs(limit=limit)
    counts = {"total": 0, "a": 0, "b": 0, "equal": 0, "flipped": 0, "skipped_no_ref": 0}

    for pair in pairs:
        audio = store.get_audio_file(pair.audio_id)
        ref = (audio.reference_transcript or "").strip() if audio else ""

        if not ref:
            # No ground truth — skip this pair
            counts["skipped_no_ref"] += 1
            continue

        wer_a = _wer_simple(ref, pair.hyp_a)
        wer_b = _wer_simple(ref, pair.hyp_b)

        # Correct oracle preference
        if abs(wer_a - wer_b) < 1e-6:
            correct = "equal"
        elif wer_a < wer_b:
            correct = "a"
        else:
            correct = "b"

        # Apply annotator error: flip a↔b with probability error_rate
        preference = correct
        flipped = False
        if correct in ("a", "b") and rng.random() < error_rate:
            preference = "b" if correct == "a" else "a"
            flipped = True

        wer_gap = abs(wer_a - wer_b)
        strength = min(10.0, round(wer_gap * 10, 1))   # rough confidence proxy

        annotation = Annotation.new(
            session_id=session_id,
            pair_id=pair.id,
            preference=preference,
            preference_strength=strength,
            notes=(
                f"sim: wer_a={wer_a:.3f} wer_b={wer_b:.3f}"
                + (" [FLIPPED]" if flipped else "")
            ),
        )
        store.insert_annotation(annotation)

        counts["total"] += 1
        counts[preference] += 1
        if flipped:
            counts["flipped"] += 1

    return counts


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RLHF simulation with WER-based synthetic feedback")
    p.add_argument("--rounds",          type=int,   default=10,
                   help="Number of RLHF cycles (default: 10)")
    p.add_argument("--error_rate",      type=float, default=0.05,
                   help="Probability of flipping the correct preference (default: 0.05)")
    p.add_argument("--pairs_per_round", type=int,   default=50,
                   help="New pairs to annotate per round (default: 50)")
    p.add_argument("--n_candidates",    type=int,   default=6,
                   help="Beam candidates for N-best generation (default: 6)")
    p.add_argument("--datasets",        default="admedvoice",
                   help="Datasets for pair generation (default: admedvoice)")
    p.add_argument("--eval_max_samples", type=int,  default=400,
                   help="Max eval utterances per split (default: 400)")
    p.add_argument("--skip_sft",        action="store_true",
                   help="Skip SFT fine-tuning (simulation has no human corrections)")
    p.add_argument("--seed",            type=int,   default=42,
                   help="RNG seed for annotation noise (default: 42)")
    p.add_argument("--tag",             default="",
                   help="Optional label appended to output CSV filename")
    p.add_argument("--db_path",         default="_outputs/feedback.sqlite",
                   help="SQLite DB path (default: _outputs/feedback.sqlite)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    # Resolve project root so imports work regardless of cwd
    project_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root))

    from paths import DEFAULT_PATHS
    from storage.feedback_store import FeedbackStore
    from storage.schema import AnnotationSession

    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = FeedbackStore(str(db_path))

    py = sys.executable
    rng = random.Random(args.seed)

    tag = f"_{args.tag}" if args.tag else f"_err{int(args.error_rate * 100):02d}"
    out_csv = project_root / "_outputs" / f"simulation{tag}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    eval_env = {"EVAL_MAX_SAMPLES": str(args.eval_max_samples)}

    print(f"""
{'=' * 60}
  RLHF Simulation
  rounds={args.rounds}  error_rate={args.error_rate * 100:.0f}%
  pairs_per_round={args.pairs_per_round}  n_candidates={args.n_candidates}
  datasets={args.datasets}  eval_max_samples={args.eval_max_samples}
  seed={args.seed}  skip_sft={args.skip_sft}
  output CSV: {out_csv}
{'=' * 60}
""")

    results: list[dict] = []
    latest_ckpt: Path | None = _find_latest_checkpoint()
    rm_ckpt: Path | None = _find_latest_rm_checkpoint()
    first_round = True

    for round_num in range(1, args.rounds + 1):
        print(f"\n{'#' * 60}")
        print(f"#  ROUND {round_num}/{args.rounds}   "
              f"error_rate={args.error_rate:.0%}   "
              f"checkpoint={latest_ckpt}")
        print(f"{'#' * 60}")

        round_result = {
            "round": round_num,
            "error_rate": args.error_rate,
            "annotations_added": 0,
            "annotations_total": 0,
            "flipped": 0,
            "wer_before": None,
            "wer_after": None,
            "wer_delta_pp": None,
        }

        # ------------------------------------------------------------------
        # Step 1: Generate pairs
        # ------------------------------------------------------------------
        gen_cmd = [
            py, "run_generate_pairs.py",
            "--n", str(args.pairs_per_round),
            "--n_candidates", str(args.n_candidates),
            "--datasets", *args.datasets.split(),
        ]
        if rm_ckpt:
            gen_cmd += ["--rm_checkpoint", str(rm_ckpt)]
        rc = _run(f"Round {round_num}: generate pairs", gen_cmd, eval_env)
        if rc != 0:
            print(f"[WARN] generate pairs exited {rc} — continuing")

        # ------------------------------------------------------------------
        # Step 2: Simulate annotations (WER-based oracle + noise)
        # ------------------------------------------------------------------
        print(f"\n[{_ts()}]  Round {round_num}: simulating annotations "
              f"(error_rate={args.error_rate:.0%})")

        # Create a fresh session per round for clean provenance
        sim_session = AnnotationSession.new(
            participant_id=f"simulation_r{round_num}",
            machine_id=socket.gethostname(),
            notes=f"WER-oracle simulation round {round_num}, error_rate={args.error_rate}",
        )
        store.open_session(sim_session)

        ann_counts = simulate_annotations(
            store=store,
            session_id=sim_session.id,
            error_rate=args.error_rate,
            rng=rng,
            limit=args.pairs_per_round * 2,   # consume slightly more than generated
        )
        store.close_session(sim_session.id)

        summary = store.annotation_summary()
        print(f"  Annotated this round: {ann_counts['total']}  "
              f"(a={ann_counts['a']} b={ann_counts['b']} "
              f"equal={ann_counts['equal']} "
              f"flipped={ann_counts['flipped']} "
              f"skipped_no_ref={ann_counts['skipped_no_ref']})")
        print(f"  DB totals: {summary}")

        round_result["annotations_added"]  = ann_counts["total"]
        round_result["annotations_total"]  = summary["total"]
        round_result["flipped"]            = ann_counts["flipped"]

        # ------------------------------------------------------------------
        # Step 3: Train reward model (once we have enough annotations)
        # ------------------------------------------------------------------
        if summary["actionable"] >= 50:
            rc = _run(f"Round {round_num}: train reward model",
                      [py, "run_train_reward.py"], eval_env)
            if rc == 0:
                rm_ckpt = _find_latest_rm_checkpoint()
        else:
            print(f"  [SKIP] reward model training — only {summary['actionable']} "
                  f"actionable annotations (need 50)")

        # ------------------------------------------------------------------
        # Step 4: Score rewards
        # ------------------------------------------------------------------
        score_cmd = [py, "run_score_rewards.py"]
        if rm_ckpt:
            score_cmd += ["--checkpoint", str(rm_ckpt)]
        _run(f"Round {round_num}: score rewards", score_cmd, eval_env)

        # ------------------------------------------------------------------
        # Step 5: SFT (first round only, unless --skip_sft)
        # ------------------------------------------------------------------
        if first_round and not args.skip_sft:
            rc = _run(f"Round {round_num}: SFT fine-tuning",
                      [py, "run_sft.py", "--epochs", "3"], eval_env)
            if rc == 0:
                # DPO resumes from SFT if no prior DPO checkpoint
                if latest_ckpt is None:
                    latest_ckpt = Path("_outputs/checkpoints/sft")

        # ------------------------------------------------------------------
        # Step 6: DPO fine-tuning  (--skip_eval so we control evaluation)
        # ------------------------------------------------------------------
        dpo_cmd = [py, "run_rlhf.py", "--skip_eval"]
        if latest_ckpt:
            dpo_cmd += ["--resume", str(latest_ckpt)]
        rc = _run(f"Round {round_num}: DPO fine-tuning", dpo_cmd, eval_env)
        if rc != 0:
            print(f"[WARN] DPO exited {rc} — skipping WER eval for this round")
            results.append(round_result)
            _write_csv(out_csv, results)
            first_round = False
            continue

        new_ckpt = _find_latest_checkpoint()
        if new_ckpt and new_ckpt != latest_ckpt:
            latest_ckpt = new_ckpt

        # ------------------------------------------------------------------
        # Step 7: Evaluate WER  (baseline from cache + fine-tuned model)
        # ------------------------------------------------------------------
        eval_cmd = [py, "run_evaluate.py", "--report"]
        _run(f"Round {round_num}: evaluate WER", eval_cmd, eval_env)

        # Read the last two training_runs rows from DB for WER
        wer_before, wer_after = _read_latest_wer(store)
        round_result["wer_before"] = wer_before
        round_result["wer_after"]  = wer_after
        if wer_before is not None and wer_after is not None:
            round_result["wer_delta_pp"] = round((wer_after - wer_before) * 100, 2)

        print(f"\n  *** Round {round_num} WER: before={_pct(wer_before)}  "
              f"after={_pct(wer_after)}  "
              f"delta={_delta(round_result['wer_delta_pp'])} pp ***")

        results.append(round_result)
        _write_csv(out_csv, results)
        first_round = False

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  SIMULATION COMPLETE — {args.rounds} rounds  "
          f"error_rate={args.error_rate:.0%}")
    print(f"{'=' * 70}")
    print(f"{'Round':>6}  {'WER before':>12}  {'WER after':>10}  {'Delta pp':>10}  "
          f"{'Annotated':>10}  {'Flipped':>8}")
    print("-" * 70)
    for r in results:
        print(f"  {r['round']:4d}   {_pct(r['wer_before']):>12}  "
              f"{_pct(r['wer_after']):>10}  "
              f"{_delta(r['wer_delta_pp']):>10}  "
              f"{r['annotations_added']:>10}  "
              f"{r['flipped']:>8}")
    print(f"\nCSV written to: {out_csv}")
    return 0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _read_latest_wer(store) -> tuple[float | None, float | None]:
    """
    Return (wer_before_admed, wer_after_admed) from the most recently
    completed training run in the DB.
    """
    try:
        with store._conn() as conn:
            row = conn.execute(
                """SELECT wer_before_admed, wer_after_admed
                   FROM training_runs
                   WHERE completed_at IS NOT NULL
                   ORDER BY completed_at DESC
                   LIMIT 1"""
            ).fetchone()
        if row:
            return row["wer_before_admed"], row["wer_after_admed"]
    except Exception:
        pass
    return None, None


def _pct(v: float | None) -> str:
    return f"{v * 100:.2f}%" if v is not None else "N/A"


def _delta(v: float | None) -> str:
    if v is None:
        return "N/A"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.2f}"


def _write_csv(path: Path, results: list[dict]) -> None:
    if not results:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    sys.exit(main())
