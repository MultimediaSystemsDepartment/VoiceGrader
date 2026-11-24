from __future__ import annotations

import gc
import json
import logging
import random
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

import numpy as np

from .admedvoice import AudioRecord, load_audio
from .config import RunConfig
from .state import check_stop

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreferenceRecord:
    member: str
    reference: str
    hypothesis_a: str
    hypothesis_b: str
    chosen: str
    rejected: str
    wer_a: float
    wer_b: float


def normalize_text(text: str) -> str:
    text = re.sub(r"[^\w\s]", "", text.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def word_error_rate(reference: str, hypothesis: str) -> float:
    from jiwer import wer

    normalized_reference = normalize_text(reference)
    normalized_hypothesis = normalize_text(hypothesis)
    if normalized_reference == normalized_hypothesis:
        return 0.0
    return float(wer(normalized_reference, normalized_hypothesis))


def oracle_preference(
    record: AudioRecord,
    hypothesis_a: str,
    hypothesis_b: str,
) -> PreferenceRecord | None:
    wer_a = word_error_rate(record.transcript, hypothesis_a)
    wer_b = word_error_rate(record.transcript, hypothesis_b)
    if wer_a < wer_b:
        chosen, rejected = hypothesis_a, hypothesis_b
    elif wer_b < wer_a:
        chosen, rejected = hypothesis_b, hypothesis_a
    elif normalize_text(record.transcript) not in {
        normalize_text(hypothesis_a),
        normalize_text(hypothesis_b),
    }:
        chosen, rejected = record.transcript, hypothesis_a
    else:
        return None
    if normalize_text(chosen) == normalize_text(rejected):
        return None
    return PreferenceRecord(
        member=record.member,
        reference=record.transcript,
        hypothesis_a=hypothesis_a,
        hypothesis_b=hypothesis_b,
        chosen=chosen,
        rejected=rejected,
        wer_a=wer_a,
        wer_b=wer_b,
    )


def _dtype(torch, name: str = "float16"):
    return torch.float16 if name == "float16" else torch.bfloat16


def _load_policy(config: RunConfig, checkpoint: Path | None, trainable: bool):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(config.model_id)
    processor.tokenizer.set_prefix_tokens(language="pl", task="transcribe")
    base = WhisperForConditionalGeneration.from_pretrained(
        config.model_id,
        torch_dtype=_dtype(torch),
        attn_implementation="sdpa",
    ).to("cuda")
    if checkpoint is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, checkpoint, is_trainable=trainable)
    elif trainable:
        from peft import LoraConfig, TaskType, get_peft_model

        model = get_peft_model(
            base,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
                task_type=TaskType.SEQ_2_SEQ_LM,
                bias="none",
            ),
        )
    else:
        model = base
    return model, processor


def _empty_cuda_cache() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def generate_preferences(
    archive: Path,
    records: list[AudioRecord],
    config: RunConfig,
    checkpoint: Path | None,
    stop_file: Path,
    output_path: Path,
    progress: Callable[[int, int], None],
) -> list[PreferenceRecord]:
    import torch

    model, processor = _load_policy(config, checkpoint, trainable=False)
    model.eval()
    preferences: list[PreferenceRecord] = []
    random.seed(config.seed)
    try:
        for index, record in enumerate(records, start=1):
            check_stop(stop_file)
            waveform = load_audio(
                archive,
                record.member,
                max_duration_seconds=config.max_duration_seconds,
            )
            features = processor.feature_extractor(
                waveform,
                sampling_rate=16_000,
                return_tensors="pt",
            ).input_features.to(device="cuda", dtype=_dtype(torch))
            with torch.inference_mode():
                beam_ids = model.generate(
                    input_features=features,
                    language="pl",
                    task="transcribe",
                    num_beams=5,
                    do_sample=False,
                    max_new_tokens=256,
                )
                torch.manual_seed(config.seed + index)
                sampled_ids = model.generate(
                    input_features=features,
                    language="pl",
                    task="transcribe",
                    num_beams=1,
                    do_sample=True,
                    temperature=config.temperature,
                    num_return_sequences=config.sampled_candidates,
                    max_new_tokens=256,
                )
            beam = processor.batch_decode(beam_ids, skip_special_tokens=True)[0].strip()
            samples = processor.batch_decode(sampled_ids, skip_special_tokens=True)
            alternative = max(
                (sample.strip() for sample in samples),
                key=lambda sample: 1.0 - SequenceMatcher(None, beam, sample).ratio(),
            )
            preference = oracle_preference(record, beam, alternative)
            if preference is not None:
                preferences.append(preference)
                with output_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(asdict(preference), ensure_ascii=False) + "\n")
            progress(index, len(records))
    finally:
        del model, processor
        _empty_cuda_cache()
    if not preferences:
        raise ValueError("Generation produced no usable preference pairs")
    return preferences


class _AudioCollator:
    def __init__(self, archive: Path, processor, config: RunConfig, mode: str) -> None:
        self.archive = archive
        self.processor = processor
        self.config = config
        self.mode = mode

    def __call__(self, batch: list[PreferenceRecord]):
        waveforms = [
            load_audio(
                self.archive,
                item.member,
                max_duration_seconds=self.config.max_duration_seconds,
            )
            for item in batch
        ]
        features = self.processor.feature_extractor(
            waveforms,
            sampling_rate=16_000,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).input_features
        if self.mode == "sft":
            return features, self._labels([item.reference for item in batch])
        return (
            features,
            self._labels([item.chosen for item in batch]),
            self._labels([item.rejected for item in batch]),
        )

    def _labels(self, texts: list[str]):
        tokens = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=440,
        ).input_ids
        tokens[tokens == self.processor.tokenizer.pad_token_id] = -100
        return tokens


def train_sft(
    archive: Path,
    rows: list[PreferenceRecord],
    config: RunConfig,
    starting_checkpoint: Path | None,
    output_dir: Path,
    stop_file: Path,
    progress: Callable[[int, int], None],
) -> Path:
    import torch
    from torch.utils.data import DataLoader

    model, processor = _load_policy(config, starting_checkpoint, trainable=True)
    processor.tokenizer.set_prefix_tokens(language="pl", task="transcribe")
    model.train()
    loader = DataLoader(
        rows,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=_AudioCollator(archive, processor, config, "sft"),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
    )
    total = len(loader) * config.sft_epochs
    step = 0
    optimizer.zero_grad()
    try:
        for epoch in range(config.sft_epochs):
            for batch_index, (features, labels) in enumerate(loader, start=1):
                check_stop(stop_file)
                features = features.to("cuda", dtype=_dtype(torch))
                labels = labels.to("cuda")
                loss = model.base_model.model(input_features=features, labels=labels).loss
                (loss / config.gradient_accumulation_steps).backward()
                if (
                    batch_index % config.gradient_accumulation_steps == 0
                    or batch_index == len(loader)
                ):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                step += 1
                log.info("SFT epoch=%d step=%d/%d loss=%.4f", epoch + 1, step, total, loss.item())
                progress(step, total)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        return output_dir
    finally:
        del model, processor
        _empty_cuda_cache()


def _sequence_logps(torch, logits, labels):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    gathered = log_probs.gather(2, shift_labels.clamp(min=0).unsqueeze(2)).squeeze(2)
    return (gathered * (shift_labels != -100)).sum(dim=1)


def _forward(model, features, labels, has_adapter: bool):
    if has_adapter:
        return model.base_model.model(input_features=features, labels=labels)
    return model(input_features=features, labels=labels)


def train_dpo(
    archive: Path,
    rows: list[PreferenceRecord],
    config: RunConfig,
    reference_checkpoint: Path,
    output_dir: Path,
    stop_file: Path,
    progress: Callable[[int, int], None],
) -> Path:
    import torch
    from torch.utils.data import DataLoader

    policy, processor = _load_policy(config, reference_checkpoint, trainable=True)
    reference, _ = _load_policy(config, reference_checkpoint, trainable=False)
    policy.eval()
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    loader = DataLoader(
        rows,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=_AudioCollator(archive, processor, config, "dpo"),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in policy.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
    )
    total = len(loader) * config.dpo_epochs
    step = 0
    optimizer.zero_grad()
    try:
        for epoch in range(config.dpo_epochs):
            for batch_index, (features, chosen, rejected) in enumerate(loader, start=1):
                check_stop(stop_file)
                features = features.to("cuda", dtype=_dtype(torch))
                chosen = chosen.to("cuda")
                rejected = rejected.to("cuda")
                policy_chosen = _sequence_logps(
                    torch,
                    _forward(policy, features, chosen, True).logits,
                    chosen,
                )
                policy_rejected = _sequence_logps(
                    torch,
                    _forward(policy, features, rejected, True).logits,
                    rejected,
                )
                with torch.inference_mode():
                    reference_chosen = _sequence_logps(
                        torch,
                        _forward(reference, features, chosen, True).logits,
                        chosen,
                    )
                    reference_rejected = _sequence_logps(
                        torch,
                        _forward(reference, features, rejected, True).logits,
                        rejected,
                    )
                offset = (policy_chosen - reference_chosen) - (
                    policy_rejected - reference_rejected
                )
                if step == 0 and float(offset.detach().abs().mean().cpu()) > 0.1:
                    raise RuntimeError("DPO policy/reference mismatch exceeds 0.1")
                loss = -torch.nn.functional.logsigmoid(config.dpo_beta * offset).mean()
                (loss / config.gradient_accumulation_steps).backward()
                if (
                    batch_index % config.gradient_accumulation_steps == 0
                    or batch_index == len(loader)
                ):
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                step += 1
                log.info("DPO epoch=%d step=%d/%d loss=%.4f", epoch + 1, step, total, loss.item())
                progress(step, total)
        output_dir.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        return output_dir
    finally:
        del policy, reference, processor
        _empty_cuda_cache()
