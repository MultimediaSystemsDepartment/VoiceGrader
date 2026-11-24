from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from paths import PathsConfig

from .config import ExperimentConfig


_MUTABLE_PATHS = (
    "output_dir",
    "db_path",
    "checkpoints_dir",
    "latest_model_dir",
    "reward_model_dir",
    "logs_dir",
)


class ExperimentWorkspace:
    def __init__(
        self,
        config: ExperimentConfig,
        source_paths: PathsConfig,
        project_root: Path,
    ) -> None:
        self.config = config
        self.source_paths = source_paths
        self.project_root = project_root.resolve()
        output_root = Path(config.output_root)
        if not output_root.is_absolute():
            output_root = self.project_root / output_root
        self.root = (output_root / config.experiment_id).resolve()

    def prepare(self, resume: bool = False) -> Path:
        manifest_path = self.root / "manifest.json"
        if self.root.exists() and any(self.root.iterdir()):
            if not resume:
                raise FileExistsError(
                    f"Experiment directory is not empty: {self.root}. Use --resume."
                )
            self._validate_resume(manifest_path)
            return manifest_path

        self.root.mkdir(parents=True, exist_ok=True)
        production_paths = {
            Path(getattr(self.source_paths, name)).resolve()
            for name in _MUTABLE_PATHS
            if getattr(self.source_paths, name)
        }
        initial_root = self.root / "initial"
        initial_paths_file = self._write_isolated_paths(
            initial_root, production_paths
        )
        arms = []
        arm_specs = (
            ("sft", 0.0),
            ("rlhf", 0.0),
            ("rlhf_noisy", self.config.noisy_feedback_error_rate),
        )
        for replicate, seed in enumerate(self.config.seeds, 1):
            for arm_index, (arm, feedback_error_rate) in enumerate(arm_specs):
                arm_root = self.root / f"replicate-{replicate:02d}" / arm
                paths_file = self._write_isolated_paths(arm_root, production_paths)
                arms.append(
                    {
                        "replicate": replicate,
                        "seed": seed,
                        "arm": arm,
                        "feedback_error_rate": feedback_error_rate,
                        "port": self.config.base_port + (replicate - 1) * 3 + arm_index,
                        "root": str(arm_root),
                        "paths_file": str(paths_file),
                    }
                )
        manifest = {
            "schema_version": 2,
            "status": "prepared",
            "config": self.config.as_dict(),
            "project_root": str(self.project_root),
            "git_commit": self._git_commit(),
            "python": sys.version,
            "initial": {
                "root": str(initial_root),
                "paths_file": str(initial_paths_file),
            },
            "arms": arms,
        }
        self._write_json_atomic(manifest_path, manifest)
        return manifest_path

    def _write_isolated_paths(
        self, arm_root: Path, production_paths: set[Path]
    ) -> Path:
        output = arm_root / "outputs"
        values = asdict(self.source_paths)
        values.update(
            {
                "output_dir": str(output),
                "db_path": str(output / "feedback.sqlite"),
                "checkpoints_dir": str(output / "checkpoints"),
                "latest_model_dir": str(output / "latest"),
                "reward_model_dir": str(output / "reward_model"),
                "logs_dir": str(output / "logs"),
            }
        )
        for name in _MUTABLE_PATHS:
            candidate = Path(values[name]).resolve()
            if candidate in production_paths:
                raise ValueError(f"Experiment {name} aliases a production path: {candidate}")
            if self.root not in candidate.parents:
                raise ValueError(f"Experiment {name} escapes experiment root: {candidate}")
        arm_root.mkdir(parents=True, exist_ok=True)
        paths_file = arm_root / "paths.json"
        self._write_json_atomic(paths_file, values)
        return paths_file

    def _validate_resume(self, manifest_path: Path) -> None:
        if not manifest_path.exists():
            raise ValueError("Cannot resume: manifest.json is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config") != self.config.as_dict():
            raise ValueError("Cannot resume with a different experiment configuration")

    def _git_commit(self) -> str:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.project_root,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    @staticmethod
    def _write_json_atomic(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        temporary.replace(path)