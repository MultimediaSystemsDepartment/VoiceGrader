from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    rounds: int = 1
    annotations_per_round: int = 32
    seeds: list[int] = field(default_factory=lambda: [42])
    tie_wer_threshold: float = 0.05
    noisy_feedback_error_rate: float = 0.05
    submit_gt_correction: bool = True
    missing_gt: str = "error"
    interaction_samples: int = 64
    monitor_samples: int = 20
    holdout_samples: int = 40
    initial_policy: str = "base"
    gpu: str = "0"
    base_port: int = 8180
    output_root: str = "_outputs/rlhf_loop_tests"

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {key: value for key, value in data.items() if key in cls.__dataclass_fields__}
        config = cls(**known)
        config.validate()
        return config

    def validate(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.experiment_id):
            raise ValueError("experiment_id must be a filesystem-safe name")
        if self.rounds < 1 or self.annotations_per_round < 1:
            raise ValueError("rounds and annotations_per_round must be positive")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a non-empty list of unique integers")
        if not 0.0 <= self.tie_wer_threshold <= 1.0:
            raise ValueError("tie_wer_threshold must be between 0 and 1")
        if not 0.0 <= self.noisy_feedback_error_rate <= 1.0:
            raise ValueError("noisy_feedback_error_rate must be between 0 and 1")
        if self.missing_gt not in {"error", "skip"}:
            raise ValueError("missing_gt must be 'error' or 'skip'")
        if min(self.interaction_samples, self.monitor_samples, self.holdout_samples) < 1:
            raise ValueError("all data partitions must contain at least one sample")
        required_interactions = self.rounds * self.annotations_per_round
        if self.interaction_samples < required_interactions:
            raise ValueError(
                "interaction_samples must cover rounds * annotations_per_round"
            )
        if self.initial_policy != "base" and not self.initial_policy.strip():
            raise ValueError("initial_policy must be 'base' or a checkpoint path")
        if not 1024 <= self.base_port <= 65000 - 3 * len(self.seeds):
            raise ValueError("base_port cannot allocate three ports per replicate")
        if self.gpu != "cpu" and not re.fullmatch(r"\d+(,\d+)*", self.gpu):
            raise ValueError("gpu must be 'cpu', an index, or comma-separated indices")

    def as_dict(self) -> dict:
        return asdict(self)