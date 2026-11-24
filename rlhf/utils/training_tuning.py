from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class TrainingTuning:
    sft_epochs: int = 3
    sft_learning_rate: float = 1e-5
    sft_warmup_steps: int = 10
    dpo_epochs: int = 2
    dpo_learning_rate: float = 1e-5
    dpo_beta: float = 0.1
    synthetic_ratio: float = 1.0
    synthetic_pool_multiplier: float = 5.0
    synthetic_reward_margin: float = 0.1
    synthetic_min_token_edits: int = 1
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 0

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "TrainingTuning":
        defaults = cls()
        try:
            tuning = cls(
                sft_epochs=int(values.get("training_sft_epochs", defaults.sft_epochs)),
                sft_learning_rate=float(
                    values.get("training_sft_learning_rate", defaults.sft_learning_rate)
                ),
                sft_warmup_steps=int(
                    values.get("training_sft_warmup_steps", defaults.sft_warmup_steps)
                ),
                dpo_epochs=int(values.get("training_dpo_epochs", defaults.dpo_epochs)),
                dpo_learning_rate=float(
                    values.get("training_dpo_learning_rate", defaults.dpo_learning_rate)
                ),
                dpo_beta=float(values.get("training_dpo_beta", defaults.dpo_beta)),
                synthetic_ratio=float(
                    values.get("training_synthetic_ratio", defaults.synthetic_ratio)
                ),
                synthetic_pool_multiplier=float(
                    values.get(
                        "training_synthetic_pool_multiplier",
                        defaults.synthetic_pool_multiplier,
                    )
                ),
                synthetic_reward_margin=float(
                    values.get(
                        "training_synthetic_reward_margin",
                        defaults.synthetic_reward_margin,
                    )
                ),
                synthetic_min_token_edits=int(
                    values.get(
                        "training_synthetic_min_token_edits",
                        defaults.synthetic_min_token_edits,
                    )
                ),
                max_grad_norm=float(
                    values.get("training_max_grad_norm", defaults.max_grad_norm)
                ),
                early_stopping_patience=int(
                    values.get(
                        "training_early_stopping_patience",
                        defaults.early_stopping_patience,
                    )
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Training tuning values must be numeric") from exc
        tuning.validate()
        return tuning

    @classmethod
    def from_store(cls, store) -> "TrainingTuning":
        defaults = cls().as_settings()
        return cls.from_mapping(
            {key: store.get_setting(key, value) for key, value in defaults.items()}
        )

    def validate(self) -> None:
        if not 1 <= self.sft_epochs <= 20:
            raise ValueError("SFT epochs must be between 1 and 20")
        if not 1e-7 <= self.sft_learning_rate <= 5e-4:
            raise ValueError("SFT learning rate must be between 1e-7 and 5e-4")
        if not 0 <= self.sft_warmup_steps <= 100:
            raise ValueError("SFT warmup cap must be between 0 and 100")
        if not 1 <= self.dpo_epochs <= 10:
            raise ValueError("DPO epochs must be between 1 and 10")
        if not 1e-7 <= self.dpo_learning_rate <= 5e-4:
            raise ValueError("DPO learning rate must be between 1e-7 and 5e-4")
        if not 0.01 <= self.dpo_beta <= 1.0:
            raise ValueError("DPO beta must be between 0.01 and 1.0")
        if not 0.0 <= self.synthetic_ratio <= 2.0:
            raise ValueError("Synthetic ratio must be between 0 and 2")
        if not 2.0 <= self.synthetic_pool_multiplier <= 10.0:
            raise ValueError("Synthetic pool multiplier must be between 2 and 10")
        if not 0.0 <= self.synthetic_reward_margin <= 5.0:
            raise ValueError("Synthetic reward margin must be between 0 and 5")
        if not 1 <= self.synthetic_min_token_edits <= 20:
            raise ValueError("Synthetic minimum token edits must be between 1 and 20")
        if not 0.1 <= self.max_grad_norm <= 5.0:
            raise ValueError("Maximum gradient norm must be between 0.1 and 5")
        if not 0 <= self.early_stopping_patience <= 10:
            raise ValueError("Early-stopping patience must be between 0 and 10")

    def as_settings(self) -> dict[str, str]:
        return {f"training_{key}": str(value) for key, value in asdict(self).items()}
