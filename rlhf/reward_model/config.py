"""
RewardModelConfig -- configuration for the audio-grounded reward model.

Architecture: Whisper encoder (frozen) + Polish BERT encoder (frozen) + MLP head.
Trained with Bradley-Terry pairwise loss on human preference annotations.
See docs/MAINTENANCE_LOG.md 2026-07-15 for rationale.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Union


@dataclass
class RewardModelConfig:
    # -- Encoder model IDs (both frozen during RM training) --------------------
    whisper_model_id: str = "openai/whisper-medium"   # audio encoder
    bert_model_id: str = "dkleczek/bert-base-polish-uncased-v1"  # text encoder

    # -- MLP head architecture -------------------------------------------------
    # Input dim = whisper_hidden (1024) + bert_hidden (768) = 1792 by default
    mlp_hidden_dim: int = 512
    mlp_dropout: float = 0.1

    # -- Compute ---------------------------------------------------------------
    device: str = "cuda"
    torch_dtype: str = "float16"
    attn_implementation: str = "sdpa"

    # -- Training --------------------------------------------------------------
    learning_rate: float = 1e-4
    batch_size: int = 16
    num_epochs: int = 20
    warmup_steps: int = 10
    save_steps: int = 50
    max_text_length: int = 128          # max BERT token length

    # Minimum pair gap to include a pair in training (|r_w - r_l| must exceed this
    # after the model converges -- not used during training itself, only for
    # synthetic pair generation in run_generate_pairs.py)
    reward_gap_threshold: float = 0.1

    # Guard: refuse to train if human annotations < this count
    min_annotations_for_training: int = 50

    # -- Versioning ------------------------------------------------------------
    # Bumped before each training run; stored in reward_scores.source as
    # 'reward_model_v{N}'
    reward_model_version: str = "v1"

    # -- Serialisation ----------------------------------------------------------

    @staticmethod
    def from_json(path: Union[str, Path]) -> RewardModelConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {k: v for k, v in data.items() if k in RewardModelConfig.__dataclass_fields__}
        return RewardModelConfig(**known)

    def to_json(self, path: Union[str, Path]) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
