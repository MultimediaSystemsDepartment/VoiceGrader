"""
Dataclasses mirroring the SQLite schema in storage/migrations/001_initial.sql.

Each class maps 1-to-1 with a DB table. Use FeedbackStore for all DB access --
never construct raw SQL outside of feedback_store.py.

Factory methods (MyClass.new(...)) auto-generate UUID4 primary keys and should
be used for all new records. Direct construction is only for rows read from DB.
"""

from __future__ import annotations

import uuid
import unicodedata
from dataclasses import dataclass
from typing import Optional


def _new_id() -> str:
    return str(uuid.uuid4())


def normalize_utterance_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    normalized = " ".join(unicodedata.normalize("NFC", text).split())
    return normalized or None


@dataclass
class Corpus:
    id: str
    name: str
    origin: str = "recorded"
    language: str = "pl"
    provider: Optional[str] = None
    active: bool = True
    purpose: str = "training"

    @staticmethod
    def new(
        name: str, origin: str = "recorded", language: str = "pl",
        provider: Optional[str] = None,
        purpose: str = "training",
    ) -> Corpus:
        if not name.strip() or not language.strip():
            raise ValueError("Corpus name and language are required")
        if origin not in {"recorded", "synthesized", "mixed"}:
            raise ValueError(f"Unknown corpus origin: {origin}")
        if purpose not in {"training", "evaluation"}:
            raise ValueError(f"Unknown corpus purpose: {purpose}")
        return Corpus(_new_id(), name.strip(), origin, language.strip(), provider,
                      purpose == "training", purpose)


@dataclass
class AudioQualityReview:
    id: str
    audio_id: str

    @staticmethod
    def new(audio_id: str) -> AudioQualityReview:
        return AudioQualityReview(_new_id(), audio_id)


@dataclass
class Utterance:
    id: str
    text: str
    normalized_text: str
    language: str = "pl"

    @staticmethod
    def new(text: str, language: str = "pl") -> Utterance:
        normalized = normalize_utterance_text(text)
        if normalized is None or not language.strip():
            raise ValueError("Utterance text and language are required")
        return Utterance(_new_id(), text, normalized, language.strip())


@dataclass
class TTSCampaign:
    id: str
    source_corpus_id: str
    intended_use: str
    selection_mode: str
    selection_value: float
    profiles_json: str
    provider_url: str
    status: str = "active"
    created_at: Optional[str] = None

    @staticmethod
    def new(source_corpus_id, intended_use, selection_mode, selection_value,
            profiles_json, provider_url) -> TTSCampaign:
        return TTSCampaign(_new_id(), source_corpus_id, intended_use,
                           selection_mode, selection_value, profiles_json, provider_url)


@dataclass
class TTSItem:
    id: str
    utterance_id: str
    profile_id: str
    profile_json: str
    provider_url: str
    state: str = "pending"
    job_id: Optional[str] = None
    audio_id: Optional[str] = None
    attempts: int = 0
    get_failures: int = 0
    next_attempt_at: float = 0
    lease_until: float = 0
    error: Optional[str] = None
    review_log_json: str = "[]"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    @staticmethod
    def new(utterance_id, profile_id, profile_json, provider_url) -> TTSItem:
        return TTSItem(_new_id(), utterance_id, profile_id, profile_json, provider_url)


# -- audio_files ----------------------------------------------------------------


@dataclass
class AudioFile:
    id: str
    file_path: str
    dataset: str  # Compatibility name for the corpus.
    reference_transcript: Optional[str] = None
    duration_s: Optional[float] = None
    sample_rate: Optional[int] = None
    created_at: Optional[str] = None
    corpus_id: Optional[str] = None
    utterance_id: Optional[str] = None
    origin: str = "recorded"
    provenance_json: Optional[str] = None

    @staticmethod
    def new(
        file_path: str,
        dataset: str,
        reference_transcript: Optional[str] = None,
        duration_s: Optional[float] = None,
        sample_rate: Optional[int] = None,
        corpus_id: Optional[str] = None,
        utterance_id: Optional[str] = None,
        origin: str = "recorded",
        provenance_json: Optional[str] = None,
    ) -> AudioFile:
        return AudioFile(
            id=_new_id(),
            file_path=file_path,
            dataset=dataset,
            reference_transcript=reference_transcript,
            duration_s=duration_s,
            sample_rate=sample_rate,
            corpus_id=corpus_id,
            utterance_id=utterance_id,
            origin=origin,
            provenance_json=provenance_json,
        )


# -- hypothesis_pairs -----------------------------------------------------------


@dataclass
class HypothesisPair:
    id: str
    audio_id: str
    hyp_a: str
    hyp_b: str
    strategy_a: str
    strategy_b: str
    model_version: str
    generated_at: Optional[str] = None
    status: str = "pending"  # 'pending' | 'annotated' | 'skipped'

    @staticmethod
    def new(
        audio_id: str,
        hyp_a: str,
        hyp_b: str,
        strategy_a: str,
        strategy_b: str,
        model_version: str,
    ) -> HypothesisPair:
        return HypothesisPair(
            id=_new_id(),
            audio_id=audio_id,
            hyp_a=hyp_a,
            hyp_b=hyp_b,
            strategy_a=strategy_a,
            strategy_b=strategy_b,
            model_version=model_version,
        )


# -- annotation_sessions --------------------------------------------------------


@dataclass
class AnnotationSession:
    id: str
    participant_id: str
    machine_id: str
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    notes: Optional[str] = None

    @staticmethod
    def new(
        participant_id: str, machine_id: str, notes: Optional[str] = None
    ) -> AnnotationSession:
        return AnnotationSession(
            id=_new_id(),
            participant_id=participant_id,
            machine_id=machine_id,
            notes=notes,
        )


# -- annotations ----------------------------------------------------------------


@dataclass
class Annotation:
    id: str
    session_id: str
    pair_id: str
    preference: str  # 'a' | 'b' | 'equal' | 'skip'
    preference_strength: Optional[float] = None
    correction_text: Optional[str] = None
    notes: Optional[str] = None
    replay_count: int = 0
    is_muted: bool = False
    annotated_at: Optional[str] = None

    @staticmethod
    def new(
        session_id: str,
        pair_id: str,
        preference: str,
        preference_strength: Optional[float] = None,
        correction_text: Optional[str] = None,
        notes: Optional[str] = None,
        replay_count: int = 0,
    ) -> Annotation:
        return Annotation(
            id=_new_id(),
            session_id=session_id,
            pair_id=pair_id,
            preference=preference,
            preference_strength=preference_strength,
            correction_text=correction_text,
            notes=notes,
            replay_count=replay_count,
        )


@dataclass
class AnnotationModerationEvent:
    id: str
    target_type: str
    target_id: str
    action: str
    actor: str
    reason: Optional[str] = None
    created_at: Optional[str] = None

    @staticmethod
    def new(
        target_type: str,
        target_id: str,
        action: str,
        actor: str,
        reason: Optional[str] = None,
    ) -> AnnotationModerationEvent:
        return AnnotationModerationEvent(
            id=_new_id(),
            target_type=target_type,
            target_id=target_id,
            action=action,
            actor=actor,
            reason=reason,
        )


@dataclass
class AudioCurationEvent:
    id: str
    audio_id: str
    action: str
    actor: str
    reason: str
    pair_id: Optional[str] = None
    created_at: Optional[str] = None

    @staticmethod
    def new(
        audio_id: str,
        action: str,
        actor: str,
        reason: str,
        pair_id: Optional[str] = None,
    ) -> AudioCurationEvent:
        return AudioCurationEvent(
            id=_new_id(),
            audio_id=audio_id,
            action=action,
            actor=actor,
            reason=reason,
            pair_id=pair_id,
        )


# -- reward_scores --------------------------------------------------------------


@dataclass
class RewardScore:
    id: str
    pair_id: str
    reward_a: float
    reward_b: float
    source: str  # 'rlaif_zeroshot' | 'rlaif_fewshot' | 'reward_model_vN'
    gemma_version: str
    scored_at: Optional[str] = None

    @staticmethod
    def new(
        pair_id: str,
        reward_a: float,
        reward_b: float,
        source: str,
        gemma_version: str,
    ) -> RewardScore:
        return RewardScore(
            id=_new_id(),
            pair_id=pair_id,
            reward_a=reward_a,
            reward_b=reward_b,
            source=source,
            gemma_version=gemma_version,
        )


# -- training_runs --------------------------------------------------------------


@dataclass
class TrainingRun:
    id: str
    algorithm: str  # 'ppo' | 'reinforce' | 'dpo'
    whisper_version_before: str
    whisper_version_after: Optional[str] = None
    n_human_annotations: Optional[int] = None
    n_synthetic_annotations: Optional[int] = None
    wer_before_admed: Optional[float] = None
    wer_after_admed: Optional[float] = None
    wer_before_cv: Optional[float] = None
    wer_after_cv: Optional[float] = None
    config_json: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    @staticmethod
    def new(
        algorithm: str,
        whisper_version_before: str,
        n_human_annotations: Optional[int] = None,
        n_synthetic_annotations: Optional[int] = None,
        config_json: Optional[str] = None,
    ) -> TrainingRun:
        return TrainingRun(
            id=_new_id(),
            algorithm=algorithm,
            whisper_version_before=whisper_version_before,
            n_human_annotations=n_human_annotations,
            n_synthetic_annotations=n_synthetic_annotations,
            config_json=config_json,
        )


# -- reward audit ---------------------------------------------------------------


@dataclass
class RewardAuditJob:
    id: str
    checkpoint_path: str
    checkpoint_name: str
    selection_mode: str
    sample_size: int
    query: Optional[str] = None
    status: str = "queued"
    error_summary: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    @staticmethod
    def new(
        checkpoint_path: str,
        checkpoint_name: str,
        selection_mode: str,
        sample_size: int,
        query: Optional[str] = None,
    ) -> RewardAuditJob:
        return RewardAuditJob(
            id=_new_id(),
            checkpoint_path=checkpoint_path,
            checkpoint_name=checkpoint_name,
            selection_mode=selection_mode,
            sample_size=sample_size,
            query=query,
        )


@dataclass
class RewardAuditItem:
    id: str
    job_id: str
    pair_id: str
    reward_a: Optional[float] = None
    reward_b: Optional[float] = None
    created_at: Optional[str] = None

    @staticmethod
    def new(job_id: str, pair_id: str) -> RewardAuditItem:
        return RewardAuditItem(id=_new_id(), job_id=job_id, pair_id=pair_id)


# -- model health ---------------------------------------------------------------


@dataclass
class ModelCheckpoint:
    id: str
    model_key: str
    stage: str
    checkpoint_path: Optional[str] = None
    parent_checkpoint_id: Optional[str] = None
    round_id: Optional[str] = None
    provenance: str = "pipeline"
    health_status: str = "unknown"
    available: bool = True
    metadata_json: Optional[str] = None
    created_at: Optional[str] = None

    @staticmethod
    def new(
        model_key: str,
        stage: str,
        checkpoint_path: Optional[str] = None,
        parent_checkpoint_id: Optional[str] = None,
        round_id: Optional[str] = None,
        provenance: str = "pipeline",
        metadata_json: Optional[str] = None,
    ) -> ModelCheckpoint:
        return ModelCheckpoint(
            id=_new_id(),
            model_key=model_key,
            stage=stage,
            checkpoint_path=checkpoint_path,
            parent_checkpoint_id=parent_checkpoint_id,
            round_id=round_id,
            provenance=provenance,
            metadata_json=metadata_json,
        )


@dataclass
class PolicyActivation:
    id: str
    new_checkpoint_id: str
    reason: str
    actor: str
    previous_checkpoint_id: Optional[str] = None
    warning_present: bool = False
    activated_at: Optional[str] = None

    @staticmethod
    def new(
        new_checkpoint_id: str,
        reason: str,
        actor: str,
        previous_checkpoint_id: Optional[str] = None,
        warning_present: bool = False,
    ) -> PolicyActivation:
        return PolicyActivation(
            id=_new_id(),
            new_checkpoint_id=new_checkpoint_id,
            reason=reason,
            actor=actor,
            previous_checkpoint_id=previous_checkpoint_id,
            warning_present=warning_present,
        )


@dataclass
class TrainingRound:
    id: str
    round_key: str
    starting_checkpoint_id: Optional[str] = None
    config_json: Optional[str] = None
    status: str = "running"
    health_status: str = "unknown"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    @staticmethod
    def new(
        round_key: str,
        starting_checkpoint_id: Optional[str] = None,
        config_json: Optional[str] = None,
    ) -> TrainingRound:
        return TrainingRound(
            id=_new_id(),
            round_key=round_key,
            starting_checkpoint_id=starting_checkpoint_id,
            config_json=config_json,
        )


@dataclass
class CanarySet:
    id: str
    name: str
    selection_seed: int
    selection_method: str
    target_size: int
    status: str = "collecting"
    fingerprint: Optional[str] = None
    created_at: Optional[str] = None
    frozen_at: Optional[str] = None

    @staticmethod
    def new(
        name: str,
        selection_seed: int,
        selection_method: str,
        target_size: int = 40,
    ) -> CanarySet:
        return CanarySet(
            id=_new_id(),
            name=name,
            selection_seed=selection_seed,
            selection_method=selection_method,
            target_size=target_size,
        )


@dataclass
class CanaryItem:
    id: str
    canary_set_id: str
    audio_id: str
    transcript: Optional[str] = None
    transcript_provenance: Optional[str] = None
    gt_available: bool = False
    gt_revealed: bool = False
    verified_at: Optional[str] = None
    created_at: Optional[str] = None

    @staticmethod
    def new(
        canary_set_id: str,
        audio_id: str,
        gt_available: bool = False,
    ) -> CanaryItem:
        return CanaryItem(
            id=_new_id(),
            canary_set_id=canary_set_id,
            audio_id=audio_id,
            gt_available=gt_available,
        )


@dataclass
class CanaryEvaluation:
    id: str
    canary_set_id: str
    checkpoint_id: str
    profile_hash: str
    status: str = "queued"
    health_status: str = "unknown"
    aggregate_json: Optional[str] = None
    warning_json: Optional[str] = None
    error_summary: Optional[str] = None

    @staticmethod
    def new(
        canary_set_id: str,
        checkpoint_id: str,
        profile_hash: str,
    ) -> CanaryEvaluation:
        return CanaryEvaluation(
            id=_new_id(),
            canary_set_id=canary_set_id,
            checkpoint_id=checkpoint_id,
            profile_hash=profile_hash,
        )


@dataclass
class CanaryPrediction:
    id: str
    evaluation_id: str
    canary_item_id: str
    hypothesis: str
    metrics_json: str
    created_at: Optional[str] = None

    @staticmethod
    def new(
        evaluation_id: str,
        canary_item_id: str,
        hypothesis: str,
        metrics_json: str,
    ) -> CanaryPrediction:
        return CanaryPrediction(
            id=_new_id(),
            evaluation_id=evaluation_id,
            canary_item_id=canary_item_id,
            hypothesis=hypothesis,
            metrics_json=metrics_json,
        )
