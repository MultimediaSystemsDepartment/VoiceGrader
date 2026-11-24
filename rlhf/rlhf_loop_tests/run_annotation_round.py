from __future__ import annotations

import argparse
from pathlib import Path

from rlhf_loop_tests.experiment.rounds import run_annotation_round


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and annotate one matched three-arm experiment round"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--round", type=int, required=True, dest="round_number")
    args = parser.parse_args()
    result = run_annotation_round(
        args.manifest.resolve(), args.replicate, args.round_number
    )
    print(
        f"Completed replicate {result['replicate']} round {result['round']} "
        f"with {result['arms'][0]['annotation_count']} annotations per arm"
    )


if __name__ == "__main__":
    main()
