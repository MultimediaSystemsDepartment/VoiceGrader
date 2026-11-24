"""
Decoding strategies for Whisper hypothesis generation.

Each DecodingStrategy has a short `name` (stored in hypothesis_pairs.strategy_a/b
in the DB so every pair is traceable) and produces kwargs for model.generate().

Pre-defined strategies:
    BEAM_5    -- beam search, 5 beams; used as the base config for diverse beam
                 search inside _transcribe_files_nbest()

Diverse N-best pairing (the only generation mode) uses HuggingFace diverse
beam search with num_beam_groups == n_candidates and a diversity_penalty so
each beam group is penalised for tokens already chosen by earlier groups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class DecodingStrategy:
    name: str                   # stored in DB -- must be unique and stable
    language: str = "pl"
    task: str = "transcribe"
    # Beam search
    num_beams: int = 1
    # Sampling
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    # Output length cap (448 ~= Whisper's limit for ~30 s audio at ~15 tok/s)
    max_new_tokens: int = 440  # 448 - 8 special prefix tokens (lang + task + notimestamps + BOS)

    def to_generate_kwargs(self) -> Dict[str, Any]:
        """Build the kwargs dict passed directly to WhisperForConditionalGeneration.generate()."""
        kwargs: Dict[str, Any] = {
            "language": self.language,
            "task": self.task,
            "max_new_tokens": self.max_new_tokens,
            # Explicitly suppress max_length to avoid the "both max_new_tokens and
            # max_length are set" warning from Whisper's generation config.
            "max_length": None,
        }
        if self.do_sample:
            kwargs["do_sample"] = True
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
            kwargs["top_k"] = self.top_k
            kwargs["num_beams"] = 1     # sampling is incompatible with beam search
        else:
            kwargs["do_sample"] = False
            kwargs["num_beams"] = self.num_beams
        return kwargs


# -- Pre-defined strategies -----------------------------------------------------

BEAM_5 = DecodingStrategy(
    name="beam_5",
    num_beams=5,
    do_sample=False,
)
