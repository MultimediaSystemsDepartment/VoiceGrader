"""
WhisperSFTTrainer -- supervised fine-tuning of Whisper on human-corrected transcripts.

For each annotation where the human typed a corrected transcript, we have a gold
(audio, text) pair. Standard seq2seq cross-entropy loss (teacher forcing) is the
most sample-efficient way to incorporate this signal -- much stronger per example
than pairwise preference.

Usage:
    trainer = WhisperSFTTrainer(config, store, paths)
    ckpt_dir = trainer.train("_outputs/checkpoints/sft")

Intended to run BEFORE DPO so the RLHF loop starts from an already
human-corrected base rather than vanilla Whisper-medium.
"""

from __future__ import annotations

import logging
import math
import json
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.audio import load_audio
from utils.training_control import EarlyStopping
from utils.torch_runtime import configure_torch_runtime, use_fused_optimizer
from utils.origin_sampling import composition, select_origin_capped
from .config import RLHFConfig

log = logging.getLogger(__name__)


def _effective_warmup_steps(
    batches_per_epoch: int,
    num_epochs: int,
    gradient_accumulation_steps: int,
    requested_warmup_steps: int,
) -> tuple[int, int]:
    updates_per_epoch = math.ceil(batches_per_epoch / gradient_accumulation_steps)
    planned_updates = updates_per_epoch * num_epochs
    warmup_steps = min(requested_warmup_steps, planned_updates // 10)
    return warmup_steps, planned_updates


# -- Dataset --------------------------------------------------------------------


class _SFTDataset(Dataset):
    def __init__(self, rows: List[Dict]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> Dict:
        return self._rows[idx]


class _SFTCollator:
    """Load audio + tokenise correction_text into model-ready tensors."""

    def __init__(self, processor, config: RLHFConfig) -> None:
        self.processor = processor
        self.config = config

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        waveforms = []
        texts = []
        for item in batch:
            try:
                wav, _ = load_audio(item["file_path"])
                waveforms.append(wav)
            except Exception as exc:
                log.warning("Audio load failed (%s): %s", exc, item["file_path"])
                waveforms.append(np.zeros(16_000, dtype=np.float32))
            texts.append(item["correction_text"])

        input_features = self.processor.feature_extractor(
            waveforms,
            sampling_rate=16_000,
            return_tensors="pt",
            padding="max_length",  # always pad to 3000 mel frames (30 s)
            truncation=True,
        ).input_features

        enc = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_label_length,
        )
        labels = enc.input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"input_features": input_features, "labels": labels}


# -- Trainer --------------------------------------------------------------------


class WhisperSFTTrainer:
    """
    Fine-tune Whisper on human-corrected (audio, transcript) pairs.

    Uses LoRA for parameter efficiency. The resulting checkpoint is the
    starting policy and frozen reference for the following DPO step.
    """

    MIN_SAMPLES = 1  # allow training even on a single correction

    def __init__(self, config: RLHFConfig, store, paths) -> None:
        self.config = config
        self.store = store
        self.paths = paths

    def train(self, output_dir: Union[str, Path]) -> Path:
        """
        Run SFT. Returns path to saved LoRA checkpoint directory.

        Raises:
            ValueError: if no correction_text annotations in DB.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        candidates = (
            self.store.get_sft_training_candidates()
            if hasattr(self.store, "get_sft_training_candidates") else
            self.store.get_current_policy_sft_corrections()
            if hasattr(self.store, "get_current_policy_sft_corrections")
            else self.store.get_sft_corrections()
        )
        share = self.store.get_setting("tts_sft_share_percent", "0") if hasattr(self.store, "get_setting") else 0
        rows = select_origin_capped(candidates, share, one_tts_per_utterance=True)
        if len(rows) < self.MIN_SAMPLES:
            raise ValueError(
                f"No human corrections found in DB. "
                f"Annotate some pairs and type corrected transcripts first."
            )
        log.info("SFT dataset: %d permitted examples (TTS cap %s%%)", len(rows), share)

        model, processor = self._load_policy()
        model.train()

        dataset = _SFTDataset(rows)
        collator = _SFTCollator(processor, self.config)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collator,
            pin_memory=str(self.config.device).startswith("cuda"),
        )

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config.learning_rate,
            fused=use_fused_optimizer(self.config.device),
        )
        warmup_steps, planned_updates = _effective_warmup_steps(
            len(loader),
            self.config.num_epochs,
            self.config.gradient_accumulation_steps,
            self.config.warmup_steps,
        )
        scheduler = (
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-3,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            if warmup_steps
            else None
        )
        log.info(
            "SFT optimizer plan: updates=%d warmup_updates=%d lr=%g",
            planned_updates,
            warmup_steps,
            self.config.learning_rate,
        )

        device = self.config.device
        total_steps = 0
        optimizer_steps = 0
        running_loss = 0.0
        loss_sum = 0.0
        batch_count = 0
        gradient_norm_sum = 0.0
        completed_epochs = 0
        early_stopped = False
        early_stopping = EarlyStopping(self.config.early_stopping_patience)
        epoch_composition = []

        for epoch in tqdm(range(self.config.num_epochs), desc="SFT epochs"):
            dataset._rows = select_origin_capped(candidates, share, seed=epoch, one_tts_per_utterance=True)
            epoch_composition.append(composition(dataset._rows))
            epoch_loss_sum = 0.0
            epoch_batch_count = 0
            for batch in tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False):
                input_features = batch["input_features"].to(
                    device=device, dtype=model.dtype, non_blocking=True
                )
                labels = batch["labels"].to(device, non_blocking=True)

                # Force decoder language/task tokens
                forced_ids = processor.get_decoder_prompt_ids(
                    language=self.config.language, task=self.config.task
                )
                model.config.forced_decoder_ids = forced_ids

                # Call the underlying WhisperForConditionalGeneration directly,
                # bypassing PeftModelForSeq2SeqLM.forward() which in newer PEFT
                # versions double-passes `input_ids` and causes a TypeError.
                # LoRA adapters are injected into the module weights in-place,
                # so gradients still flow through them correctly.
                outputs = model.base_model.model(
                    input_features=input_features,
                    labels=labels,
                )
                loss = outputs.loss / self.config.gradient_accumulation_steps
                loss.backward()

                total_steps += 1
                batch_loss = loss.item() * self.config.gradient_accumulation_steps
                running_loss += batch_loss
                loss_sum += batch_loss
                batch_count += 1
                epoch_loss_sum += batch_loss
                epoch_batch_count += 1

                is_accumulation_boundary = (
                    total_steps % self.config.gradient_accumulation_steps == 0
                )
                is_epoch_tail = total_steps % len(loader) == 0
                if is_accumulation_boundary or is_epoch_tail:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.max_grad_norm
                    )
                    gradient_norm_sum += float(gradient_norm.detach().cpu())
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                    optimizer_steps += 1

                    if (
                        total_steps // self.config.gradient_accumulation_steps
                    ) % self.config.logging_steps == 0:
                        avg_loss = running_loss / self.config.logging_steps
                        log.info("SFT step %d  loss=%.4f", total_steps, avg_loss)
                        running_loss = 0.0

                if (
                    total_steps
                    % (self.config.save_steps * self.config.gradient_accumulation_steps)
                    == 0
                ):
                    ckpt = output_dir / f"sft_step_{total_steps}"
                    model.save_pretrained(ckpt)
                    log.info("Checkpoint saved: %s", ckpt)

            completed_epochs = epoch + 1
            epoch_loss = epoch_loss_sum / epoch_batch_count
            if early_stopping.update(epoch_loss):
                early_stopped = True
                log.info(
                    "SFT early stopping after epoch %d (loss=%.4f patience=%d)",
                    completed_epochs,
                    epoch_loss,
                    self.config.early_stopping_patience,
                )
                break

        if optimizer_steps == 0:
            raise ValueError("SFT completed without an optimizer step")

        log.info(
            "SFT optimization complete: optimizer_updates=%d mean_loss=%.4f "
            "mean_gradient_norm=%.4f final_lr=%g",
            optimizer_steps,
            loss_sum / batch_count,
            gradient_norm_sum / optimizer_steps,
            optimizer.param_groups[0]["lr"],
        )
        self.store.set_setting(
            "training_sft_diagnostics",
            json.dumps(
                {
                    "planned_optimizer_updates": planned_updates,
                    "optimizer_updates": optimizer_steps,
                    "effective_warmup_steps": warmup_steps,
                    "completed_epochs": completed_epochs,
                    "mean_loss": loss_sum / batch_count,
                    "mean_gradient_norm": gradient_norm_sum / optimizer_steps,
                    "final_learning_rate": optimizer.param_groups[0]["lr"],
                    "early_stopped": early_stopped,
                    "tts_share_cap_percent": float(share),
                    "epoch_composition": epoch_composition,
                }
            ),
        )
        (output_dir / "sft_data_snapshot.json").write_text(
            json.dumps({"tts_share_cap_percent": float(share), "epochs": epoch_composition},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        # Final save
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        log.info("SFT complete. LoRA checkpoint saved to %s", output_dir)
        return output_dir

    # -- Model loading ----------------------------------------------------------

    def _load_policy(self):
        """Load Whisper + LoRA adapter. Returns (model, processor)."""
        from utils.policy_checkpoint import validate_policy_checkpoint

        if self.config.resume_from_checkpoint:
            validate_policy_checkpoint(
                Path(self.config.resume_from_checkpoint), self.config.whisper_model_id
            )
        configure_torch_runtime()
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        log.info("Loading Whisper for SFT: %s", self.config.whisper_model_id)
        dtype = (
            torch.float16 if self.config.torch_dtype == "float16" else torch.bfloat16
        )

        processor = WhisperProcessor.from_pretrained(self.config.whisper_model_id)
        processor.tokenizer.set_prefix_tokens(
            language=self.config.language, task=self.config.task
        )

        resume = self.config.resume_from_checkpoint
        if resume:
            from peft import PeftModel

            base = WhisperForConditionalGeneration.from_pretrained(
                self.config.whisper_model_id,
                torch_dtype=dtype,
                attn_implementation=self.config.attn_implementation,
            ).to(self.config.device)
            model = PeftModel.from_pretrained(base, resume, is_trainable=True)
            log.info("Resumed from checkpoint: %s", resume)
        else:
            base = WhisperForConditionalGeneration.from_pretrained(
                self.config.whisper_model_id,
                torch_dtype=dtype,
                attn_implementation=self.config.attn_implementation,
            ).to(self.config.device)
            lora_cfg = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                task_type=TaskType.SEQ_2_SEQ_LM,
                bias="none",
            )
            model = get_peft_model(base, lora_cfg)
            trainable, total = model.get_nb_trainable_parameters()
            log.info(
                "LoRA applied -- trainable params: %d / %d (%.2f%%)",
                trainable,
                total,
                100 * trainable / total,
            )

        return model, processor
