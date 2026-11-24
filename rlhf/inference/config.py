"""
InferenceConfig -- configuration for two-pass Whisper hypothesis generation.

Pair generation combines deterministic beam search with temperature sampling.

Load from / save to JSON:
    config = InferenceConfig.from_json("config/inference_config.json")
    config.to_json("config/inference_config.json")
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Union
from paths import DEFAULT_WHISPER_MODEL_ID


@dataclass
class InferenceConfig:
    # -- Model -----------------------------------------------------------------
    # HuggingFace Hub ID or absolute path to a local Whisper (or LoRA) checkpoint
    model_id: str = DEFAULT_WHISPER_MODEL_ID

    # -- Compute ---------------------------------------------------------------
    device: str = "cuda"
    torch_dtype: str = "float16"        # "float16" | "bfloat16"
    attn_implementation: str = "sdpa"

    # -- Batching --------------------------------------------------------------
    # How many audio files to transcribe in one forward pass.
    batch_size: int = 1

    # -- Queue management ------------------------------------------------------
    # How many pairs to generate per run_generate_pairs.py invocation
    target_batch_size: int = 50

    # -- N-best generation -----------------------------------------------------
    # Temperature for the sampling pass (Pass 2) in _transcribe_files_nbest.
    # Pass 1 always uses deterministic beam search; Pass 2 draws n_candidates-1
    # stochastic alternatives at this temperature.  0.8 gives natural variation
    # on medical speech without excessive hallucination.
    nbest_temperature: float = 0.8
    generation_seed: Optional[int] = None

    # -- Dataset scope ---------------------------------------------------------
    datasets: List[str] = field(default_factory=lambda: ["admedvoice"])
    splits: List[str] = field(default_factory=lambda: ["train"])

    # -- Serialisation ----------------------------------------------------------

    @staticmethod
    def from_json(path: Union[str, Path]) -> InferenceConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {k: v for k, v in data.items() if k in InferenceConfig.__dataclass_fields__}
        return InferenceConfig(**known)

    def to_json(self, path: Union[str, Path]) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
