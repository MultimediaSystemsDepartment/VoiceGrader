from __future__ import annotations

import argparse
from pathlib import Path

from rlhf_loop_tests.experiment.materialization import materialize_arm_states


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clone the common initial state into isolated experiment arms"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = materialize_arm_states(args.manifest.resolve())
    print(f"Materialized {len(manifest['materialized_arms'])} isolated arms")


if __name__ == "__main__":
    main()
