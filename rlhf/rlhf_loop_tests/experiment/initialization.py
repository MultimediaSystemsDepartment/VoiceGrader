from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from .materialization import materialize_arm_states
from .planning import plan_partitions
from .runtime import subprocess_environment


def initialize_experiment(
    manifest_path: Path,
    runner: Callable[..., object] = subprocess.run,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "prepared":
        raise ValueError("Experiment initialization requires a prepared manifest")

    project_root = Path(manifest["project_root"])
    paths_file = manifest["initial"]["paths_file"]
    runner(
        [sys.executable, str(project_root / "run_generate_pairs.py"), "--catalogue_only"],
        cwd=project_root,
        env=subprocess_environment(paths_file, "cpu"),
        check=True,
    )
    plan_partitions(manifest_path)
    return materialize_arm_states(manifest_path)
