"""
RewardModelTrainer -- Bradley-Terry pairwise trainer for AudioTextRewardModel.

Trains the MLP head of AudioTextRewardModel on human preference annotations.
Both encoders (Whisper, BERT) are frozen; only the MLP parameters are updated.

Bradley-Terry loss for a single pair:
    L = -log sigmoid(r_w - r_l)
where r_w = reward(audio, winner_hypothesis)
      r_l = reward(audio, loser_hypothesis)

The training set comes from store.get_preference_dataset() which returns only
clear-preference annotations (preference IN ('a', 'b')).  'equal' and 'skip'
annotations are excluded.

Checkpoint saved to output_dir/reward_model_vN/ containing:
    mlp_head.pt      MLP state dict
    rm_config.json   RewardModelConfig snapshot

torch / transformers imported lazily (consistent with project convention).

Usage:
    trainer = RewardModelTrainer(config, store, output_dir)
    checkpoint_path = trainer.train()          # returns Path to saved checkpoint
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .audio_text_reward import AudioTextRewardModel
from .config import RewardModelConfig
from utils.torch_runtime import use_fused_optimizer

log = logging.getLogger(__name__)


class RewardModelTrainer:
    """
    Trains the MLP head of AudioTextRewardModel with Bradley-Terry pairwise loss.

    Args:
        config:     RewardModelConfig with training hyperparameters.
        store:      FeedbackStore instance (source of preference annotations).
        output_dir: Root directory for saving checkpoints.
    """

    def __init__(
        self,
        config: RewardModelConfig,
        store: Any,         # FeedbackStore -- avoid circular import
        output_dir: Union[str, Path],
    ) -> None:
        self.config = config
        self.store = store
        self.output_dir = Path(output_dir)

    # -- Public entry point ----------------------------------------------------

    def train(self) -> Path:
        """
        Run the full training loop and save the checkpoint.

        Returns:
            Path to the saved checkpoint directory.

        Raises:
            ValueError: if not enough clear-preference annotations are available.
        """
        summary = self.store.annotation_summary()
        log.info(
            "Annotation summary -- total: %d  actionable (a/b): %d  equal: %d  skip: %d",
            summary["total"], summary["actionable"], summary["equal"], summary["skip"],
        )

        n_usable = summary["actionable"]
        if n_usable < self.config.min_annotations_for_training:
            raise ValueError(
                f"Need at least {self.config.min_annotations_for_training} clear-preference "
                f"(A or B) annotations to train the reward model, found {n_usable}. "
                f"Total annotations: {summary['total']} "
                f"({summary['equal']} marked 'equal', {summary['skip']} skipped). "
                "Collect more pairwise preference annotations."
            )

        rows = self.store.get_preference_dataset()
        log.info("Usable preference pairs loaded: %d", len(rows))
        if not rows:
            raise ValueError(
                f"No clear-preference data found despite {summary['total']} total annotations "
                f"({summary['equal']} equal, {summary['skip']} skip). "
                "Make sure annotators select A or B, not just Equal."
            )

        pairs = self._build_pairs(rows)
        log.info("Training pairs after audio validation: %d", len(pairs))

        checkpoint_dir = (
            self.output_dir
            / f"reward_model_{self.config.reward_model_version}"
        )

        rm = AudioTextRewardModel(config=self.config, checkpoint_path=None)
        rm.load_model()
        try:
            self._run_training(rm, pairs)
            rm.save(checkpoint_dir)
        finally:
            rm.unload_model()

        log.info(
            "Reward model training complete. Checkpoint: %s", checkpoint_dir
        )
        return checkpoint_dir

    # -- Data preparation ------------------------------------------------------

    def _build_pairs(
        self, rows: List[Dict]
    ) -> List[Tuple[str, str, str]]:
        """
        Convert preference dataset rows to (audio_path, winner_text, loser_text).
        Skips rows where the audio file is missing or unreadable.
        """
        pairs = []
        for row in rows:
            file_path = row.get("file_path")
            if not file_path:
                log.debug("Skipping row with no file_path: %s", row.get("pair_id"))
                continue

            pref = row["preference"]
            hyp_a = row["hyp_a"]
            hyp_b = row["hyp_b"]

            if pref == "a":
                winner, loser = hyp_a, hyp_b
            elif pref == "b":
                winner, loser = hyp_b, hyp_a
            else:
                continue  # 'equal' / 'skip' -- no clear signal

            pairs.append((file_path, winner, loser))

        return pairs

    # -- Training loop ---------------------------------------------------------

    def _run_training(
        self,
        rm: AudioTextRewardModel,
        pairs: List[Tuple[str, str, str]],
    ) -> None:
        """
        Inner training loop.  Operates directly on rm.module (the nn.Module).
        Encoders are frozen; only MLP parameters are optimised.
        """
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
        from utils.audio import load_audio

        module = rm.module
        feature_extractor = rm._feature_extractor
        tokenizer = rm._bert_tokenizer

        device = self.config.device
        dtype = torch.float16 if self.config.torch_dtype == "float16" else torch.float32

        # Only optimise the MLP head (encoders are frozen)
        optimizer = torch.optim.Adam(
            module.mlp.parameters(),
            lr=self.config.learning_rate,
            fused=use_fused_optimizer(device),
        )

        batch_size = self.config.batch_size
        num_epochs = self.config.num_epochs

        log.info(
            "Training: epochs=%d  batch_size=%d  lr=%g  pairs=%d",
            num_epochs, batch_size, self.config.learning_rate, len(pairs),
        )

        module.whisper_encoder.eval()
        module.bert.eval()
        module.mlp.train()

        winner_inputs = []
        loser_inputs = []
        for batch_start in range(0, len(pairs), batch_size):
            batch = pairs[batch_start : batch_start + batch_size]
            try:
                audio_feats, winner_enc, loser_enc = self._encode_batch(
                    batch, feature_extractor, tokenizer,
                    device, dtype, load_audio,
                )
            except Exception as exc:
                log.warning("Skipping batch at index %d: %s", batch_start, exc)
                continue

            with torch.inference_mode():
                audio_vec = (
                    module.whisper_encoder(audio_feats)
                    .last_hidden_state.mean(dim=1)
                )
                combined_text = {
                    key: torch.cat([winner_enc[key], loser_enc[key]], dim=0)
                    for key in winner_enc
                }
                text_vec = module.bert(**combined_text).pooler_output
                winner_vec, loser_vec = text_vec.split(len(batch), dim=0)
                mlp_dtype = next(module.mlp.parameters()).dtype
                winner_inputs.append(
                    torch.cat([audio_vec, winner_vec], dim=-1)
                    .to(dtype=mlp_dtype).cpu()
                )
                loser_inputs.append(
                    torch.cat([audio_vec, loser_vec], dim=-1)
                    .to(dtype=mlp_dtype).cpu()
                )

        if not winner_inputs:
            raise ValueError("Reward training produced no usable encoded pairs")

        embedding_dataset = TensorDataset(
            torch.cat(winner_inputs), torch.cat(loser_inputs)
        )
        embedding_loader = DataLoader(
            embedding_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=str(device).startswith("cuda"),
        )
        log.info(
            "Cached frozen reward embeddings for %d pairs",
            len(embedding_dataset),
        )

        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0.0
            n_batches = 0

            for winner_input, loser_input in embedding_loader:
                optimizer.zero_grad()
                winner_input = winner_input.to(device, non_blocking=True)
                loser_input = loser_input.to(device, non_blocking=True)
                r_w = module.mlp(winner_input)
                r_l = module.mlp(loser_input)

                # Bradley-Terry loss: -log sigmoid(r_w - r_l)
                loss = -F.logsigmoid(r_w - r_l).mean()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            log.info("Epoch %d/%d  loss=%.4f", epoch, num_epochs, avg_loss)

        module.eval()

    def _encode_batch(
        self,
        batch: List[Tuple[str, str, str]],
        feature_extractor: Any,
        tokenizer: Any,
        device: str,
        dtype: Any,
        load_audio: Any,
    ) -> Tuple[Any, Dict, Dict]:
        """
        Encode a batch of (audio_path, winner_text, loser_text) triples.

        Returns:
            audio_feats:  (B, 80, 3000) tensor on device
            winner_enc:   dict of BERT input tensors for winner hypotheses
            loser_enc:    dict of BERT input tensors for loser hypotheses
        """
        import torch

        waveforms = [load_audio(path)[0] for path, _, _ in batch]
        winner_texts = [winner for _, winner, _ in batch]
        loser_texts = [loser for _, _, loser in batch]

        # Whisper feature extraction -- pads/truncates to 30 s, returns (B, 80, 3000)
        # padding="max_length" ensures every clip is padded to exactly 3000 mel frames
        # regardless of batch composition; padding=True only pads to the longest clip.
        audio_feats = feature_extractor(
            waveforms,
            sampling_rate=16_000,
            return_tensors="pt",
            padding="max_length",
        ).input_features.to(device=device, dtype=dtype)

        def _tokenize(texts: List[str]) -> Dict:
            enc = tokenizer(
                texts,
                max_length=self.config.max_text_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return {k: v.to(device) for k, v in enc.items()}

        return audio_feats, _tokenize(winner_texts), _tokenize(loser_texts)
