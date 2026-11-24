from __future__ import annotations

import argparse
import json
from pathlib import Path

from paths import DEFAULT_PATHS, PROJECT_ROOT
from rlhf_loop_tests.experiment import (
    ExperimentConfig,
    ExperimentWorkspace,
    run_experiment_to_completion,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an isolated SFT versus clean/noisy RLHF experiment"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu", help="Override GPU selection, e.g. 0, 1, or cpu")
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config)
    if args.gpu is not None:
        values = config.as_dict()
        values["gpu"] = args.gpu
        config = ExperimentConfig(**values)
        config.validate()
    workspace = ExperimentWorkspace(config, DEFAULT_PATHS, PROJECT_ROOT)
    manifest_path = workspace.prepare(resume=args.resume)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"Prepared {len(manifest['arms'])} isolated arms at {workspace.root}")
    print(f"Manifest: {manifest_path}")
    if not args.prepare_only:
        report = run_experiment_to_completion(manifest_path)
        comparison = report["across_seed_final_rlhf_minus_sft_wer"]
        noisy = report["across_seed_final_rlhf_noisy_minus_sft_wer"]
        print(
            "Complete: clean RLHF-SFT WER delta="
            f"{comparison['delta']:.4f}; noisy RLHF-SFT WER delta="
            f"{noisy['delta']:.4f}"
        )


if __name__ == "__main__":
    main()