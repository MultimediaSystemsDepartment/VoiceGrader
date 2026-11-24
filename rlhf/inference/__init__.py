"""
inference package -- Whisper hypothesis generation for the annotation queue.

Public API:
    HypothesisGenerator  -- main class: catalogue ingestion + pair generation
    InferenceConfig      -- configuration dataclass
    DecodingStrategy     -- per-strategy decoding parameters
    BEAM_5               -- base config for the deterministic and sampling passes
"""

from .config import InferenceConfig
from .decoding_strategies import (
    BEAM_5,
    DecodingStrategy,
)
from .hypothesis_generator import HypothesisGenerator

__all__ = [
    "HypothesisGenerator",
    "InferenceConfig",
    "DecodingStrategy",
    "BEAM_5",
]
