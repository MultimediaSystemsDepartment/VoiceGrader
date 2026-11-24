"""
Shared pytest fixtures for the RLHF-ASR test suite.
No GPU, no Whisper, no real audio — all tests run on CPU with temp SQLite DBs.
"""

from __future__ import annotations

import pytest

from storage.feedback_store import FeedbackStore
from storage.schema import (
    Annotation,
    AnnotationSession,
    AudioFile,
    HypothesisPair,
    RewardScore,
)


@pytest.fixture()
def store(tmp_path):
    """Fresh FeedbackStore backed by a temp file DB (deleted after each test)."""
    return FeedbackStore(tmp_path / "test.sqlite")


# ---------------------------------------------------------------------------
# Helpers shared across test modules
# ---------------------------------------------------------------------------

def make_audio(store: FeedbackStore, path: str = "audio/test.wav") -> AudioFile:
    af = AudioFile.new(path, dataset="admedvoice", reference_transcript=f"test ref {path}")
    store.insert_audio_file(af)
    return af


def make_pair(store: FeedbackStore, audio_id: str) -> HypothesisPair:
    pair = HypothesisPair.new(
        audio_id=audio_id,
        hyp_a="hypothesis A",
        hyp_b="hypothesis B",
        strategy_a="beam_5",
        strategy_b="temp_0.8",
        model_version="openai/whisper-medium",
    )
    store.insert_hypothesis_pair(pair)
    return pair


def make_session(store: FeedbackStore, participant: str = "tester") -> AnnotationSession:
    session = AnnotationSession.new(participant_id=participant, machine_id="test-machine")
    store.open_session(session)
    return session


def annotate(
    store: FeedbackStore,
    session_id: str,
    pair_id: str,
    preference: str,
    correction_text: str | None = None,
) -> Annotation:
    ann = Annotation.new(
        session_id=session_id,
        pair_id=pair_id,
        preference=preference,
        correction_text=correction_text,
    )
    store.insert_annotation(ann)
    return ann
