"""
PyTorch Dataset and DataCollator for Whisper DPO training.

WhisperRLHFDataset   -- wraps reward buffer rows into a torch Dataset.
WhisperRLHFCollator  -- loads audio, computes log-mel features, tokenises
                       transcripts; returns batched tensors ready for the trainer.

Each item yields (file_path, chosen_hyp, rejected_hyp) for DPO loss.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.audio import load_audio

log = logging.getLogger(__name__)


# -- Dataset --------------------------------------------------------------------

class WhisperRLHFDataset(Dataset):

    def __init__(
        self,
        rows: List[dict],
        min_reward_delta: float = 0.0,
    ) -> None:
        """
        Args:
            rows:              Rows from RewardBuffer.rows (already joined with DB).
            min_reward_delta:  Skip pairs where |r_a - r_b| < this.
                               Discards weak-preference pairs. Default 0 keeps all.
        """
        self._items: List[dict] = []

        for row in rows:
            if row.get("chosen_hyp") and row.get("rejected_hyp"):
                self._items.append(row)
                continue
            delta = abs(row["reward_a"] - row["reward_b"])
            if delta <= min_reward_delta:
                continue
            self._items.append(row)

        log.info("WhisperRLHFDataset  items=%d  (from %d rows)",
                 len(self._items), len(rows))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        return self._items[idx]


# -- DataCollator ---------------------------------------------------------------

class WhisperRLHFCollator:
    """
    Collates a batch of reward-buffer rows into model-ready tensors.
    Returns input_features, chosen_labels, rejected_labels for DPO loss.
    """

    def __init__(self, processor, config) -> None:
        """
        Args:
            processor:  WhisperProcessor (feature extractor + tokenizer).
            config:     RLHFConfig (for language, task, max_label_length).
        """
        self.processor = processor
        self.config    = config

    def __call__(self, batch: List[dict]) -> Dict[str, torch.Tensor]:
        waveforms = []
        for item in batch:
            try:
                wav, _ = load_audio(item["file_path"])
                waveforms.append(wav)
            except Exception as exc:
                log.warning("Audio load failed (%s): %s", exc, item["file_path"])
                # Replace with silence so batch shape is preserved
                waveforms.append(np.zeros(16_000, dtype=np.float32))

        # Mel-bin count comes from the model-specific feature extractor.
        input_features = self.processor.feature_extractor(
            waveforms,
            sampling_rate=16_000,
            return_tensors="pt",
            padding="max_length",   # always pad to 3000 mel frames (30 s)
            truncation=True,
        ).input_features

        return self._collate_dpo(batch, input_features)

    # -- DPO collation ----------------------------------------------------------

    def _collate_dpo(
        self, batch: List[dict], input_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        chosen_texts   = []
        rejected_texts = []
        for item in batch:
            if item.get("chosen_hyp") and item.get("rejected_hyp"):
                chosen_texts.append(item["chosen_hyp"])
                rejected_texts.append(item["rejected_hyp"])
            elif item["reward_a"] > item["reward_b"]:
                chosen_texts.append(item["hyp_a"])
                rejected_texts.append(item["hyp_b"])
            else:
                chosen_texts.append(item["hyp_b"])
                rejected_texts.append(item["hyp_a"])

        return {
            "input_features":  input_features,
            "chosen_labels":   self._tokenise_labels(chosen_texts),
            "rejected_labels": self._tokenise_labels(rejected_texts),
        }

    # -- Tokenisation helper ----------------------------------------------------

    def _tokenise_labels(self, texts: List[str]) -> torch.Tensor:
        """Tokenise a list of transcript texts into padded label tensor."""
        enc = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_label_length,
        )
        labels = enc.input_ids.clone()
        # Mask padding tokens so they don't contribute to the loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return labels
