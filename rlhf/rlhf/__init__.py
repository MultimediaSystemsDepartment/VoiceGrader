"""
rlhf package -- Whisper RLHF fine-tuning via DPO.

Lightweight imports (no torch) are available at package level.
Heavy imports (torch, transformers, peft) are deferred so that scripts
that only need RLHFConfig or RewardBuffer don't load GPU frameworks.
"""

from .config import RLHFConfig
from .reward_buffer import RewardBuffer

# torch-dependent classes are imported on demand (not at package init)
# from .dataset import WhisperRLHFCollator, WhisperRLHFDataset
# from .reference_model import ReferenceWhisper
# from .trainer import WhisperRLHFTrainer

__all__ = [
    "RLHFConfig",
    "RewardBuffer",
    "WhisperRLHFDataset",
    "WhisperRLHFCollator",
    "ReferenceWhisper",
    "WhisperRLHFTrainer",
]
