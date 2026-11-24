"""
storage package -- SQLite persistence layer for RLHF-ASR training.

Public API:
    FeedbackStore   -- all DB read/write operations
    AudioFile       -- dataclass: audio_files table
    HypothesisPair  -- dataclass: hypothesis_pairs table
    AnnotationSession -- dataclass: annotation_sessions table
    Annotation      -- dataclass: annotations table
    RewardScore     -- dataclass: reward_scores table
    TrainingRun     -- dataclass: training_runs table
"""

from .feedback_store import FeedbackStore
from .schema import (
    Annotation,
    AnnotationSession,
    AudioFile,
    Corpus,
    Utterance,
    CanaryEvaluation,
    CanaryItem,
    CanaryPrediction,
    CanarySet,
    HypothesisPair,
    ModelCheckpoint,
    PolicyActivation,
    RewardAuditItem,
    RewardAuditJob,
    RewardScore,
    TrainingRound,
    TrainingRun,
)

__all__ = [
    "FeedbackStore",
    "AudioFile",
    "Corpus",
    "Utterance",
    "HypothesisPair",
    "AnnotationSession",
    "Annotation",
    "CanarySet",
    "CanaryItem",
    "CanaryPrediction",
    "CanaryEvaluation",
    "RewardScore",
    "ModelCheckpoint",
    "PolicyActivation",
    "RewardAuditItem",
    "RewardAuditJob",
    "TrainingRound",
    "TrainingRun",
]
