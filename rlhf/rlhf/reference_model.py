"""
ReferenceWhisper -- frozen copy of Whisper for KL-divergence computation.

DPO loads the reference with the same precision as its round-start policy.

Provides compute_logps() which returns the per-batch sum of log-probabilities
for given label sequences -- used by DPO.
"""

from __future__ import annotations

import logging
from typing import Optional, Union
from pathlib import Path

import torch
from utils.torch_runtime import configure_torch_runtime
from utils.policy_checkpoint import validate_policy_checkpoint
from paths import DEFAULT_WHISPER_MODEL_ID

log = logging.getLogger(__name__)


class ReferenceWhisper:
    """Frozen Whisper model for log-probability computation."""

    def __init__(
        self,
        model_id: str = DEFAULT_WHISPER_MODEL_ID,
        adapter_checkpoint: str = "",
        load_in_4bit: bool = False,
        device: str = "cuda",
        torch_dtype: str = "float16",
        attn_implementation: str = "sdpa",
    ) -> None:
        self.model_id    = model_id
        self.adapter_checkpoint = adapter_checkpoint
        self.load_in_4bit = load_in_4bit
        self.device      = device
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self._model: Optional[object] = None

    def load(self) -> None:
        if self._model is not None:
            return

        if self.adapter_checkpoint:
            validate_policy_checkpoint(Path(self.adapter_checkpoint), self.model_id)
        configure_torch_runtime()
        from transformers import WhisperForConditionalGeneration, BitsAndBytesConfig

        log.info("Loading reference Whisper: %s  (4-bit=%s)", self.model_id, self.load_in_4bit)

        dtype = torch.float16 if self.torch_dtype == "float16" else torch.bfloat16
        bnb_cfg = None
        if self.load_in_4bit:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        base = WhisperForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=bnb_cfg,
            torch_dtype=dtype if not self.load_in_4bit else None,
            device_map=self.device,
            attn_implementation=self.attn_implementation,
        )
        if self.adapter_checkpoint:
            adapter_path = Path(self.adapter_checkpoint)
            if not (adapter_path / "adapter_config.json").is_file():
                raise FileNotFoundError(
                    f"Invalid reference policy checkpoint: {adapter_path}"
                )
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(base, adapter_path)
            log.info("Reference LoRA checkpoint loaded: %s", adapter_path)
        else:
            self._model = base
        self._model.eval()
        # Freeze all parameters
        for p in self._model.parameters():
            p.requires_grad_(False)

        log.info("Reference Whisper loaded and frozen")

    def unload(self) -> None:
        if self._model is None:
            return
        del self._model
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("Reference Whisper unloaded")

    def __enter__(self) -> ReferenceWhisper:
        self.load()
        return self

    def __exit__(self, *_) -> None:
        self.unload()

    # -- Core computation -------------------------------------------------------

    @torch.inference_mode()
    def compute_logps(
        self,
        input_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute sum of log-probabilities for each item in the batch.

        Args:
            input_features: model-specific log-mel tensor on GPU.
            labels:         (B, T) label token IDs; -100 = padding (ignored).

        Returns:
            (B,) tensor of summed log-probs (negative = lower likelihood).
        """
        if self._model is None:
            raise RuntimeError("Reference model not loaded.")

        feats = input_features.to(self._model.device)
        lbls  = labels.to(self._model.device)

        if self.adapter_checkpoint:
            outputs = self._model.base_model.model(input_features=feats, labels=lbls)
        else:
            outputs = self._model(input_features=feats, labels=lbls)
        # outputs.loss = mean NLL over non-padding tokens (scalar per batch element
        # would require per-sample reduction, but Whisper returns batch mean).
        # We approximate per-sample log probs using batch mean NLL.
        n_tokens = (lbls != -100).sum(dim=1).float()   # (B,)  tokens per item

        # Recompute per-sample NLL via logits (precise)
        logits  = outputs.logits                        # (B, T, V)
        shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
        shift_labels = lbls[:, 1:].contiguous()         # (B, T-1)

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        # Gather log prob for each target token
        gathered  = log_probs.gather(2, shift_labels.clamp(min=0).unsqueeze(2)).squeeze(2)
        # Zero out padding positions
        mask      = (shift_labels != -100).float()
        logps_sum = (gathered * mask).sum(dim=1)        # (B,)

        return logps_sum.float()
