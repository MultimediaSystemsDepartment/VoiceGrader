"""
AudioTextRewardModel -- audio-grounded reward model for ASR preference learning.

Architecture:
    audio -> Whisper encoder (frozen) -> mean-pool -> audio_vec (1024-dim)
                                                           |
    hypothesis -> Polish BERT (frozen) -> [CLS] -> text_vec (768-dim)
                                                           |
                          MLP( cat([audio_vec, text_vec]) ) -> scalar reward (0-1)

Both encoders are frozen -- only the MLP head is trained.
Trained with Bradley-Terry pairwise loss (see reward_model/train_reward_model.py).

torch / transformers are imported lazily inside load_model() so this module can
be imported without torch being installed (consistent with project convention).

See docs/MAINTENANCE_LOG.md 2026-07-15 and docs/DPO_PRIMER.md section 11 for rationale.

Usage (inference):
    with AudioTextRewardModel(config, checkpoint_path) as rm:
        scores = rm.score_pairs(pairs, audio_paths)
        # scores: List[Tuple[str, float, float]] -- (pair_id, reward_a, reward_b)

Usage (training -- called by RewardModelTrainer):
    rm = AudioTextRewardModel(config)
    rm.load_model()
    module = rm.module      # nn.Module -- only mlp params have requires_grad=True
    # ... train, then:
    rm.save(output_dir)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .config import RewardModelConfig
from utils.torch_runtime import configure_torch_runtime

log = logging.getLogger(__name__)

# Encoder output dimensions -- fixed for the chosen model family
_WHISPER_MEDIUM_HIDDEN = 1024   # whisper-medium encoder last hidden state
_BERT_BASE_HIDDEN = 768         # bert-base-* pooler_output


def _build_module(
    config: RewardModelConfig,
    whisper_encoder: Any,
    bert: Any,
    mlp_state_dict: Optional[dict] = None,
) -> Any:
    """
    Construct the AudioTextRewardModule (an nn.Module).

    Called inside load_model() after torch has been imported lazily.
    Defining the nn.Module subclasses here avoids a top-level torch import,
    consistent with project convention for GPU models.

    Args:
        config:          RewardModelConfig instance.
        whisper_encoder: WhisperEncoder already on the target device.
        bert:            AutoModel (BERT) already on the target device.
        mlp_state_dict:  Optional pre-trained MLP weights to load.

    Returns:
        AudioTextRewardModule instance in eval() mode.
    """
    import torch
    import torch.nn as nn

    class _RewardMLP(nn.Module):
        """MLP head: concatenated encoder vectors -> unbounded utility logit."""

        def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x: Any) -> Any:
            return self.net(x).squeeze(-1)  # (B,)

    class _AudioTextRewardModule(nn.Module):
        """
        Combines Whisper encoder + Polish BERT + MLP head.
        Encoders run under torch.no_grad() -- only MLP parameters are trained.
        """

        def __init__(
            self,
            whisper_encoder: nn.Module,
            bert: nn.Module,
            mlp: _RewardMLP,
        ) -> None:
            super().__init__()
            self.whisper_encoder = whisper_encoder
            self.bert = bert
            self.mlp = mlp
            for p in self.whisper_encoder.parameters():
                p.requires_grad_(False)
            for p in self.bert.parameters():
                p.requires_grad_(False)

        def forward(
            self,
            audio_features: Any,   # (B, 80, 3000) log-mel
            input_ids: Any,        # (B, L)
            attention_mask: Any,   # (B, L)
        ) -> Any:                  # (B,) utility logits
            audio_vec = self.encode_audio(audio_features)
            text_vec = self.encode_text(input_ids, attention_mask)
            combined = torch.cat([audio_vec, text_vec], dim=-1).to(
                dtype=next(self.mlp.parameters()).dtype
            )
            return self.mlp(combined)                              # (B,)

        def encode_audio(self, audio_features: Any) -> Any:
            with torch.no_grad():
                return (
                    self.whisper_encoder(audio_features)
                    .last_hidden_state.mean(dim=1)      # (B, 1024)
                )

        def encode_text(self, input_ids: Any, attention_mask: Any) -> Any:
            with torch.no_grad():
                return (
                    self.bert(input_ids=input_ids, attention_mask=attention_mask)
                    .pooler_output                      # (B, 768)
                )

        def score_candidates(
            self, audio_features: Any, input_ids: Any, attention_mask: Any
        ) -> Any:
            audio_vec = self.encode_audio(audio_features)
            text_vec = self.encode_text(input_ids, attention_mask)
            audio_vec = audio_vec.expand(text_vec.shape[0], -1)
            combined = torch.cat([audio_vec, text_vec], dim=-1).to(
                dtype=next(self.mlp.parameters()).dtype
            )
            return self.mlp(combined)                              # (B,)

        def score_candidate_batches(
            self,
            audio_features: Any,
            input_ids: Any,
            attention_mask: Any,
            audio_indices: Any,
        ) -> Any:
            audio_vec = self.encode_audio(audio_features)
            text_vec = self.encode_text(input_ids, attention_mask)
            audio_vec = audio_vec.index_select(0, audio_indices)
            combined = torch.cat([audio_vec, text_vec], dim=-1).to(
                dtype=next(self.mlp.parameters()).dtype
            )
            return self.mlp(combined)

    device = config.device
    mlp = _RewardMLP(
        input_dim=_WHISPER_MEDIUM_HIDDEN + _BERT_BASE_HIDDEN,
        hidden_dim=config.mlp_hidden_dim,
        dropout=config.mlp_dropout,
    ).to(device)

    if mlp_state_dict is not None:
        mlp.load_state_dict(mlp_state_dict)

    module = _AudioTextRewardModule(
        whisper_encoder=whisper_encoder,
        bert=bert,
        mlp=mlp,
    )
    module.eval()
    return module


# ---------------------------------------------------------------------------
# Public wrapper class
# ---------------------------------------------------------------------------

class AudioTextRewardModel:
    """
    Inference and lifecycle wrapper for the audio+text reward module.

    Manages GPU memory, feature extraction, BERT tokenisation, and scoring.
    Always use as a context manager:

        with AudioTextRewardModel(config, checkpoint_path) as rm:
            scores = rm.score_pairs(pairs, audio_paths)

    checkpoint_path is the directory saved by save().  Pass None to start
    with random MLP weights (only useful during initial training setup).
    """

    _CHECKPOINT_FILE = "mlp_head.pt"
    _CONFIG_FILE = "rm_config.json"

    def __init__(
        self,
        config: RewardModelConfig,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        if self.checkpoint_path is not None:
            checkpoint_config = self.checkpoint_path / self._CONFIG_FILE
            if checkpoint_config.exists():
                stored = RewardModelConfig.from_json(checkpoint_config)
                stored.device = config.device
                stored.torch_dtype = config.torch_dtype
                config = stored
        self.config = config
        self._module: Optional[Any] = None
        self._feature_extractor: Optional[Any] = None
        self._bert_tokenizer: Optional[Any] = None

    # -- GPU lifecycle ----------------------------------------------------------

    def load_model(self) -> None:
        """Load Whisper encoder, BERT, and MLP head into GPU memory."""
        if self._module is not None:
            return

        configure_torch_runtime()
        import torch
        from transformers import (
            AutoModel,
            AutoTokenizer,
            WhisperFeatureExtractor,
            WhisperModel,
        )

        device = self.config.device
        dtype = torch.float16 if self.config.torch_dtype == "float16" else torch.float32

        log.info("Loading Whisper encoder: %s", self.config.whisper_model_id)
        whisper = WhisperModel.from_pretrained(
            self.config.whisper_model_id,
            torch_dtype=dtype,
            attn_implementation=self.config.attn_implementation,
        )
        whisper_encoder = whisper.encoder.to(device)
        self._feature_extractor = WhisperFeatureExtractor.from_pretrained(
            self.config.whisper_model_id
        )
        del whisper  # decoder not needed

        log.info("Loading Polish BERT: %s", self.config.bert_model_id)
        bert = AutoModel.from_pretrained(
            self.config.bert_model_id,
            torch_dtype=dtype,
            attn_implementation=self.config.attn_implementation,
        ).to(device)
        self._bert_tokenizer = AutoTokenizer.from_pretrained(self.config.bert_model_id)

        mlp_state = None
        if self.checkpoint_path is not None:
            mlp_path = self.checkpoint_path / self._CHECKPOINT_FILE
            if not mlp_path.exists():
                raise FileNotFoundError(
                    f"MLP checkpoint not found: {mlp_path}\n"
                    "Run run_train_reward.py first."
                )
            mlp_state = torch.load(str(mlp_path), map_location=device, weights_only=True)
            log.info("Loaded MLP weights from %s", mlp_path)
        else:
            log.warning(
                "No checkpoint_path given -- MLP has random weights. "
                "Train the reward model before scoring."
            )

        self._module = _build_module(
            config=self.config,
            whisper_encoder=whisper_encoder,
            bert=bert,
            mlp_state_dict=mlp_state,
        )
        log.info(
            "AudioTextRewardModel ready  (device=%s  checkpoint=%s)",
            device, self.checkpoint_path,
        )

    def unload_model(self) -> None:
        if self._module is None:
            return
        del self._module, self._feature_extractor, self._bert_tokenizer
        self._module = None
        self._feature_extractor = None
        self._bert_tokenizer = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        log.info("AudioTextRewardModel unloaded")

    def __enter__(self) -> AudioTextRewardModel:
        self.load_model()
        return self

    def __exit__(self, *_) -> None:
        self.unload_model()

    # -- Inference --------------------------------------------------------------

    def score_pairs(
        self,
        pairs: list,                     # List[HypothesisPair]
        audio_paths: Dict[str, str],     # audio_id -> absolute file path
    ) -> List[Tuple[str, float, float]]:
        """
        Score a list of HypothesisPair objects.

        Args:
            pairs:       HypothesisPair instances (need .id, .audio_id, .hyp_a, .hyp_b).
            audio_paths: Mapping audio_id -> absolute path on disk.

        Returns:
            List of (pair_id, reward_a, reward_b).
            Pairs whose audio cannot be loaded are logged and skipped.
        """
        if self._module is None:
            raise RuntimeError("Model not loaded. Use as context manager.")

        from utils.audio import load_audio

        results: List[Tuple[str, float, float]] = []
        for pair in pairs:
            path = audio_paths.get(pair.audio_id)
            if path is None:
                log.warning(
                    "No audio path for audio_id=%s -- skipping pair %s",
                    pair.audio_id, pair.id,
                )
                continue
            try:
                waveform, _ = load_audio(path)
                r_a = self._score_one(waveform, pair.hyp_a)
                r_b = self._score_one(waveform, pair.hyp_b)
                results.append((pair.id, r_a, r_b))
            except Exception as exc:
                log.warning("Error scoring pair %s: %s", pair.id, exc)

        log.info("Scored %d / %d pairs", len(results), len(pairs))
        return results

    def score_single(
        self,
        audio_path: Union[str, Path],
        hypothesis: str,
    ) -> float:
        """Score one audio/text hypothesis pair with an unbounded utility logit."""
        if self._module is None:
            raise RuntimeError("Model not loaded. Use as context manager.")

        from utils.audio import load_audio

        return self._score_one(load_audio(str(audio_path))[0], hypothesis)

    def score_candidates(self, waveform: Any, hypotheses: List[str]) -> List[float]:
        """Score candidate texts while encoding their shared audio exactly once."""
        if self._module is None:
            raise RuntimeError("Model not loaded. Use as context manager.")
        if not hypotheses:
            return []

        import torch

        dtype = torch.float16 if self.config.torch_dtype == "float16" else torch.float32
        features = self._feature_extractor(
            waveform, sampling_rate=16_000, return_tensors="pt"
        ).input_features.to(device=self.config.device, dtype=dtype)
        encoded = self._bert_tokenizer(
            hypotheses,
            max_length=self.config.max_text_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.config.device)
        attention_mask = encoded["attention_mask"].to(self.config.device)
        with torch.no_grad():
            rewards = self._module.score_candidates(
                features, input_ids, attention_mask
            )
        return [float(value) for value in rewards.detach().float().cpu().view(-1)]

    def score_candidate_batches(
        self, waveforms: List[Any], hypotheses: List[List[str]]
    ) -> List[List[float]]:
        """Score candidates for multiple clips in one audio/text encoder batch."""
        if self._module is None:
            raise RuntimeError("Model not loaded. Use as context manager.")
        if len(waveforms) != len(hypotheses):
            raise ValueError("Waveform and hypothesis batch sizes must match")
        counts = [len(items) for items in hypotheses]
        flat_hypotheses = [text for items in hypotheses for text in items]
        if not flat_hypotheses:
            return [[] for _ in waveforms]

        import torch

        dtype = torch.float16 if self.config.torch_dtype == "float16" else torch.float32
        features = self._feature_extractor(
            waveforms, sampling_rate=16_000, return_tensors="pt", padding="max_length"
        ).input_features.to(device=self.config.device, dtype=dtype)
        encoded = self._bert_tokenizer(
            flat_hypotheses,
            max_length=self.config.max_text_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.config.device)
        attention_mask = encoded["attention_mask"].to(self.config.device)
        audio_indices = torch.repeat_interleave(
            torch.arange(len(waveforms), device=self.config.device),
            torch.tensor(counts, device=self.config.device),
        )
        with torch.inference_mode():
            rewards = self._module.score_candidate_batches(
                features, input_ids, attention_mask, audio_indices
            ).detach().float().cpu().tolist()

        grouped = []
        offset = 0
        for count in counts:
            grouped.append(rewards[offset : offset + count])
            offset += count
        return grouped

    def _score_one(self, waveform: Any, hypothesis: str) -> float:
        """Internal: score a pre-loaded 16 kHz mono float32 waveform against text."""
        import torch

        device = self.config.device
        dtype = torch.float16 if self.config.torch_dtype == "float16" else torch.float32

        feats = self._feature_extractor(
            waveform,
            sampling_rate=16_000,
            return_tensors="pt",
        ).input_features.to(device=device, dtype=dtype)       # (1, 80, 3000)

        enc = self._bert_tokenizer(
            hypothesis,
            max_length=self.config.max_text_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            reward = self._module(feats, input_ids, attention_mask)

        return float(reward.item())

    # -- Checkpoint I/O --------------------------------------------------------

    def save(self, output_dir: Union[str, Path]) -> Path:
        """
        Persist the trained MLP head and config to output_dir.
        Encoder weights are NOT saved -- they are re-loaded from HF Hub on demand.

        Returns:
            Path to the saved directory.
        """
        if self._module is None:
            raise RuntimeError("Model not loaded. Cannot save.")

        import torch

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self._module.mlp.state_dict(), str(out / self._CHECKPOINT_FILE))
        (out / self._CONFIG_FILE).write_text(
            json.dumps(asdict(self.config), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("Saved reward model checkpoint to %s", out)
        return out

    @classmethod
    def load_config(cls, checkpoint_dir: Union[str, Path]) -> RewardModelConfig:
        """Read back the RewardModelConfig from a saved checkpoint directory."""
        cfg_path = Path(checkpoint_dir) / cls._CONFIG_FILE
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        known = {k: v for k, v in data.items() if k in RewardModelConfig.__dataclass_fields__}
        return RewardModelConfig(**known)

    # -- Module accessor (used by RewardModelTrainer) --------------------------

    @property
    def module(self) -> Optional[Any]:
        """The underlying nn.Module -- available after load_model()."""
        return self._module
