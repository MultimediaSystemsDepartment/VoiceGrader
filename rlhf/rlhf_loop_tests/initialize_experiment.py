from __future__ import annotations

import argparse
from pathlib import Path

from rlhf_loop_tests.experiment.initialization import initialize_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Populate and materialize an isolated RLHF loop experiment"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = initialize_experiment(args.manifest.resolve())
    print(f"Initialized {len(manifest['materialized_arms'])} isolated arms")


if __name__ == "__main__":
    main()
