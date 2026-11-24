from __future__ import annotations

import argparse
from pathlib import Path

from rlhf_loop_tests.experiment.reporting import build_experiment_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the completed multi-seed RLHF versus SFT report"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    report = build_experiment_report(args.manifest.resolve())
    result = report["across_seed_final_rlhf_minus_sft_wer"]
    print(
        f"Across-seed RLHF-SFT WER delta: {result['delta']:.4f} "
        f"(95% CI {result['ci_low']:.4f}, {result['ci_high']:.4f})"
    )


if __name__ == "__main__":
    main()
