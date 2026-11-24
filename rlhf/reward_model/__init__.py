"""
reward_model package -- audio-grounded reward model for RLHF-ASR.

Architecture: Whisper encoder (frozen) + Polish BERT (frozen) + MLP head.
Trained with Bradley-Terry pairwise loss on human preference annotations.

See docs/MAINTENANCE_LOG.md 2026-07-15 for the rationale behind replacing
the previous Gemma-based RLAIF approach.
"""

from .config import RewardModelConfig
from .audio_text_reward import AudioTextRewardModel
from .train_reward_model import RewardModelTrainer

__all__ = [
    "RewardModelConfig",
    "AudioTextRewardModel",
    "RewardModelTrainer",
]
