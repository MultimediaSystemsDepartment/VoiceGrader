from __future__ import annotations

import argparse
from pathlib import Path

from rlhf_loop_tests.experiment.evaluation import evaluate_experiment_round


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate matched experiment arms on the frozen final holdout"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--round", type=int, required=True, dest="round_number")
    args = parser.parse_args()
    report = evaluate_experiment_round(
        args.manifest.resolve(), args.replicate, args.round_number
    )
    comparison = report["rlhf_minus_sft"]
    print(
        "RLHF-SFT corpus WER delta: "
        f"{comparison['delta']:.4f} "
        f"(95% CI {comparison['ci_low']:.4f}, {comparison['ci_high']:.4f})"
    )


if __name__ == "__main__":
    main()
