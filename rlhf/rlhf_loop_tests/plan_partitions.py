from __future__ import annotations

import argparse
from pathlib import Path

from rlhf_loop_tests.experiment.planning import plan_partitions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan deterministic RLHF experiment data partitions"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = plan_partitions(args.manifest.resolve())
    print(f"Planned {len(manifest['replicates'])} replicate schedules")


if __name__ == "__main__":
    main()
