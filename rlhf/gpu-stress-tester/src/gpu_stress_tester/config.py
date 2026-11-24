from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    profile: str = "standard"
    rounds: int = 1
    samples: int = 64
    sft_epochs: int = 1
    dpo_epochs: int = 1
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    dpo_beta: float = 0.1
    temperature: float = 0.8
    sampled_candidates: int = 3
    seed: int = 42
    model_id: str = "openai/whisper-medium"
    max_duration_seconds: float = 30.0

    def validate(self) -> RunConfig:
        if self.profile not in PROFILES:
            raise ValueError(f"Unknown profile: {self.profile}")
        for name in (
            "rounds",
            "samples",
            "sft_epochs",
            "dpo_epochs",
            "batch_size",
            "gradient_accumulation_steps",
            "sampled_candidates",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if not 0.0 < self.learning_rate <= 1e-3:
            raise ValueError("learning_rate must be between 0 and 0.001")
        if not 0.0 < self.dpo_beta <= 1.0:
            raise ValueError("dpo_beta must be between 0 and 1")
        if not 0.1 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.1 and 2.0")
        return self

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def from_json(cls, path: Path) -> RunConfig:
        return cls(**json.loads(path.read_text(encoding="utf-8"))).validate()


PROFILES: dict[str, RunConfig] = {
    "smoke": RunConfig(profile="smoke", samples=8, batch_size=1),
    "standard": RunConfig(profile="standard", samples=64),
    "endurance": RunConfig(profile="endurance", rounds=100_000, samples=128),
    "max-load": RunConfig(
        profile="max-load",
        rounds=100_000,
        samples=256,
        batch_size=4,
        gradient_accumulation_steps=8,
    ),
}

PROFILE_DESCRIPTIONS = {
    "smoke": "Quick installation check with the smallest GPU workload.",
    "standard": "One representative RLHF training round.",
    "endurance": "Repeated training rounds intended to run until stopped.",
    "max-load": "Repeated rounds with more clips and a larger GPU batch.",
}


def profile_summaries() -> dict[str, dict[str, object]]:
    fields = (
        "rounds",
        "samples",
        "sft_epochs",
        "dpo_epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "temperature",
        "sampled_candidates",
    )
    return {
        name: {
            "description": PROFILE_DESCRIPTIONS[name],
            **{field: getattr(config, field) for field in fields},
        }
        for name, config in PROFILES.items()
    }


def resolve_config(profile: str, **overrides: object) -> RunConfig:
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}")
    values = {key: value for key, value in overrides.items() if value is not None}
    return replace(PROFILES[profile], **values).validate()
