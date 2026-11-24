from __future__ import annotations

import argparse
from pathlib import Path

from rlhf_loop_tests.experiment.training import train_experiment_round


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train matched plain-SFT and canonical-RLHF experiment arms"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--round", type=int, required=True, dest="round_number")
    args = parser.parse_args()
    result = train_experiment_round(
        args.manifest.resolve(), args.replicate, args.round_number
    )
    outcomes = ", ".join(
        f"{arm['arm']}={arm['outcome']}" for arm in result["arms"]
    )
    print(f"Completed replicate {result['replicate']} round {result['round']}: {outcomes}")


if __name__ == "__main__":
    main()
