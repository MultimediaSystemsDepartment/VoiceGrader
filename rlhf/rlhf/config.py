"""
RLHFConfig -- configuration for Whisper DPO fine-tuning.

Algorithm: DPO (Direct Preference Optimisation, Rafailov et al. 2023).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Union
from paths import DEFAULT_WHISPER_MODEL_ID


@dataclass
class RLHFConfig:
    # -- Algorithm --------------------------------------------------------------
    algorithm: str = "dpo"

    # -- Whisper policy model ---------------------------------------------------
    whisper_model_id: str = DEFAULT_WHISPER_MODEL_ID
    # Path to a LoRA checkpoint to resume from (None = start from base)
    resume_from_checkpoint: str = ""
    # Frozen policy at the start of this DPO round (defaults to resume checkpoint)
    reference_checkpoint: str = ""

    # -- LoRA adapter ----------------------------------------------------------
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # -- Compute ----------------------------------------------------------------
    device: str = "cuda"
    torch_dtype: str = "float16"
    attn_implementation: str = "sdpa"
    # Reference precision must match the policy so round-start log-ratios are zero.
    reference_in_4bit: bool = False

    # -- Training loop ----------------------------------------------------------
    learning_rate: float = 1e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 32
    num_epochs: int = 2  # small policy-scoped rounds need multiple updates
    warmup_steps: int = 10
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 0
    save_steps: int = 200
    logging_steps: int = 10

    # -- DPO-specific -----------------------------------------------------------
    # beta controls KL regularisation strength (lower = more deviation allowed)
    dpo_beta: float = 0.1
    synthetic_pool_size: int = 500
    synthetic_n_candidates: int = 6
    synthetic_seed: int = 42
    max_synthetic_to_human_ratio: float = 1.0

    # -- Data guard -------------------------------------------------------------
    # One default effective batch: batch_size * gradient_accumulation_steps.
    min_reward_scores: int = 32

    # -- Language ---------------------------------------------------------------
    language: str = "pl"
    task: str = "transcribe"
    max_label_length: int = 440  # 448 - 8 special prefix tokens

    # -- Serialisation ----------------------------------------------------------

    @staticmethod
    def from_json(path: Union[str, Path]) -> RLHFConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {k: v for k, v in data.items() if k in RLHFConfig.__dataclass_fields__}
        return RLHFConfig(**known)

    def to_json(self, path: Union[str, Path]) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def to_json_str(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)
