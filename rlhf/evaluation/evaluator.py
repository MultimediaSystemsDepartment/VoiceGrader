"""
WhisperEvaluator -- compute WER/CER on ADMEDVOICE and CommonVoice test splits.

Loads Whisper (policy or base), runs greedy inference on the test CSV,
computes WER and CER using jiwer, and writes results to the training_runs
table in the DB.

Usage:
    evaluator = WhisperEvaluator(model_id, paths, store)
    result = evaluator.evaluate()
    print(result)
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from tqdm import tqdm
from utils.torch_runtime import configure_torch_runtime

log = logging.getLogger(__name__)


@dataclass
class EvalResult:
    model_id: str
    wer_admed: Optional[float]      # Word Error Rate on ADMEDVOICE test
    cer_admed: Optional[float]
    wer_cv: Optional[float]         # Word Error Rate on CommonVoice test
    cer_cv: Optional[float]
    n_admed: int = 0                # number of utterances evaluated
    n_cv: int = 0
    primary_set: str = "ADMEDVOICE"

    def __str__(self) -> str:
        def _fmt(v):
            return f"{v * 100:.2f}%" if v is not None else "N/A"
        return (
            f"Model:          {self.model_id}\n"
            f"{self.primary_set} WER: {_fmt(self.wer_admed)}  CER: {_fmt(self.cer_admed)}  "
            f"(n={self.n_admed})\n"
            f"CommonVoice WER:{_fmt(self.wer_cv)}  CER: {_fmt(self.cer_cv)}  "
            f"(n={self.n_cv})"
        )


class WhisperEvaluator:

    def __init__(
        self,
        model_id: str,
        paths,                      # PathsConfig
        store=None,                 # FeedbackStore -- optional, for logging
        device: str = "cuda",
        torch_dtype: str = "float16",
        lora_checkpoint: Optional[str] = None,
        max_samples: Optional[int] = None,   # None = full test set
        attn_implementation: str = "sdpa",
        decode_num_beams: int = 1,
        use_monitor: bool = False,
    ) -> None:
        self.model_id        = model_id
        self.paths           = paths
        self.store           = store
        self.device          = device
        self.torch_dtype     = torch_dtype
        self.lora_checkpoint = lora_checkpoint
        self.max_samples     = max_samples
        self.attn_implementation = attn_implementation
        self.decode_num_beams = decode_num_beams
        self.monitor = store.get_training_monitor_rows() if use_monitor and store else None
        if self.monitor:
            self.decode_num_beams = 5
        self._model          = None
        self._processor      = None

    # -- Subset hash (for baseline WER caching) --------------------------------

    def compute_subset_hash(self) -> str:
        """Return a SHA-256 hex digest of the model_id + test subset.

        Deterministic: uses the same sampling logic as _eval_split so the hash
        changes if and only if the set of (file_path, reference) pairs changes
        or the model ID changes.  Skips loading the GPU model.
        Returns a stable fallback hash if the test CSV is not configured yet
        (cold-start / missing manifest).
        """
        if self.monitor:
            cv_path = self.paths.commonvoice_test_csv
            cv_hash = (
                hashlib.sha256(Path(cv_path).read_bytes()).hexdigest()
                if cv_path and Path(cv_path).is_file() else None
            )
            fingerprint = json.dumps(
                [self.model_id, self.monitor["id"], self.monitor["fingerprint"],
                 self.decode_num_beams, self.torch_dtype, self.monitor["rows"],
                 cv_hash, self.paths.commonvoice_audio_dir, self.max_samples],
                sort_keys=True, ensure_ascii=False,
            ).encode()
            return hashlib.sha256(fingerprint).hexdigest()
        csv_path = self.paths.admedvoice_test_csv
        if not csv_path or not Path(csv_path).exists():
            # No test CSV yet — return a deterministic placeholder so the
            # cache simply stores nothing useful and re-evaluates once data arrives.
            fallback = f"no-csv:{self.model_id}"
            return hashlib.sha256(fallback.encode()).hexdigest()
        rows = self._load_csv(
            csv_path,
            self.paths.admedvoice_audio_dir,
        )
        if self.max_samples and len(rows) > self.max_samples:
            rows = random.Random(42).sample(rows, self.max_samples)
        fingerprint = json.dumps(
            [self.model_id] + sorted((r["file_path"], r["reference"]) for r in rows),
            ensure_ascii=False,
        ).encode()
        return hashlib.sha256(fingerprint).hexdigest()

    # -- Main entry -------------------------------------------------------------

    def evaluate(self) -> EvalResult:
        """Run evaluation on both test sets. Returns EvalResult."""
        self._load_model()

        try:
            if self.monitor:
                log.info("Training evaluation uses frozen WER Monitor %s (%d clips)",
                         self.monitor["id"], len(self.monitor["rows"]))
                wer_a, cer_a, n_a = self._eval_rows(self.monitor["rows"], "WER Monitor")
            else:
                wer_a, cer_a, n_a = self._eval_split(
                    self.paths.admedvoice_test_csv,
                    self.paths.admedvoice_audio_dir,
                    "admedvoice",
                )
            wer_c, cer_c, n_c = self._eval_split(
                self.paths.commonvoice_test_csv,
                self.paths.commonvoice_audio_dir,
                "commonvoice",
            )
        finally:
            self._unload_model()

        result = EvalResult(
            model_id=self.lora_checkpoint or self.model_id,
            wer_admed=wer_a,
            cer_admed=cer_a,
            wer_cv=wer_c,
            cer_cv=cer_c,
            n_admed=n_a,
            n_cv=n_c,
            primary_set="WER Monitor" if self.monitor else "ADMEDVOICE",
        )
        log.info("Evaluation complete:\n%s", result)
        return result

    # -- Per-split evaluation ---------------------------------------------------

    def _eval_split(
        self,
        csv_path: str,
        audio_dir: str,
        dataset_name: str,
    ) -> Tuple[Optional[float], Optional[float], int]:
        """Returns (wer, cer, n_samples)."""
        if not csv_path or not Path(csv_path).exists():
            log.warning("Test CSV not found for %s -- skipping", dataset_name)
            return None, None, 0

        rows = self._load_csv(csv_path, audio_dir)
        if self.max_samples and len(rows) > self.max_samples:
            # Fixed seed so the same subset is chosen on every run
            # (comparability across training cycles).
            rng = random.Random(42)
            rows = rng.sample(rows, self.max_samples)
        if not rows:
            return None, None, 0

        return self._eval_rows(rows, dataset_name)

    def _eval_rows(self, rows: List[dict], dataset_name: str) -> Tuple[float, float, int]:

        log.info("Evaluating %s  (%d utterances)", dataset_name, len(rows))

        references  = []
        hypotheses  = []
        batch_size  = 1

        batches = range(0, len(rows), batch_size)
        for start in tqdm(batches, desc=f"Evaluating {dataset_name}", unit="batch"):
            batch = rows[start : start + batch_size]
            refs  = [r["reference"] for r in batch]
            paths = [r["file_path"] for r in batch]

            preds = self._transcribe_batch(paths)
            references.extend(refs)
            hypotheses.extend(preds)

        wer, cer = _compute_metrics(references, hypotheses)
        log.info("%s  WER=%.2f%%  CER=%.2f%%", dataset_name, wer * 100, cer * 100)
        return wer, cer, len(rows)

    def _load_csv(self, csv_path: str, audio_dir: str) -> List[dict]:
        """Load file_path and reference_transcript from a CSV manifest."""
        delimiter = self.paths.csv_delimiter
        rows = []

        # Reuse column-detection from inference module
        from inference.hypothesis_generator import _detect_columns
        from utils.audio import validate_audio_file
        path_col, transcript_col = _detect_columns(csv_path, delimiter)

        with open(csv_path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            for row in reader:
                raw = (row.get(path_col) or "").strip()
                ref = (row.get(transcript_col) or "").strip()
                if not raw or not ref:
                    continue

                # zip: URIs are self-contained -- don't apply audio_dir prefix
                # and don't test with Path.exists() (zip: is not a filesystem path).
                if raw.startswith("zip:"):
                    file_path = raw
                else:
                    audio_path = Path(raw)
                    if not audio_path.is_absolute() and audio_dir:
                        audio_path = Path(audio_dir) / audio_path
                    if not audio_path.exists():
                        continue
                    file_path = str(audio_path)

                rows.append({"file_path": file_path, "reference": ref})

        return rows

    # -- Transcription ----------------------------------------------------------

    def transcribe_files(
        self,
        file_paths: List[str],
        batch_size: int = 1,
        on_batch: Optional[Callable[[int, List[str]], None]] = None,
    ) -> List[str]:
        """Decode an explicit file list with one model load and preserve ordering."""
        self._load_model()
        try:
            hypotheses = []
            for start in range(0, len(file_paths), batch_size):
                batch_hypotheses = self._transcribe_batch(
                    file_paths[start : start + batch_size]
                )
                hypotheses.extend(batch_hypotheses)
                if on_batch:
                    on_batch(start, batch_hypotheses)
            return hypotheses
        finally:
            self._unload_model()

    def _transcribe_batch(self, file_paths: List[str]) -> List[str]:
        import torch
        from utils.audio import load_audio

        waveforms = []
        for fp in file_paths:
            try:
                wav, _ = load_audio(fp)
                waveforms.append(wav)
            except Exception as exc:
                log.warning("Audio load failed (%s): %s", exc, fp)
                waveforms.append(None)

        valid_wavs = [w for w in waveforms if w is not None]
        if not valid_wavs:
            return [""] * len(file_paths)

        inputs = self._processor(
            valid_wavs,
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True,
        )
        feats = inputs.input_features.to(self._model.device, dtype=self._model.dtype)

        with torch.inference_mode():
            pred_ids = self._model.generate(
                input_features=feats,
                language="pl",
                task="transcribe",
                num_beams=self.decode_num_beams,
                do_sample=False,
            )

        transcriptions = self._processor.batch_decode(pred_ids, skip_special_tokens=True)

        t_iter = iter(transcriptions)
        return [next(t_iter) if w is not None else "" for w in waveforms]

    # -- Model lifecycle --------------------------------------------------------

    def _load_model(self) -> None:
        from utils.policy_checkpoint import validate_policy_checkpoint

        if self.lora_checkpoint:
            validate_policy_checkpoint(Path(self.lora_checkpoint), self.model_id)
        configure_torch_runtime()
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        log.info("Loading Whisper for evaluation: %s", self.model_id)
        dtype = torch.float16 if self.torch_dtype == "float16" else torch.bfloat16

        self._processor = WhisperProcessor.from_pretrained(self.model_id)
        base = WhisperForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            attn_implementation=self.attn_implementation,
        ).to(self.device)

        if self.lora_checkpoint:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(base, self.lora_checkpoint)
            log.info("LoRA checkpoint loaded: %s", self.lora_checkpoint)
        else:
            self._model = base

        self._model.eval()

    def _unload_model(self) -> None:
        import torch
        del self._model, self._processor
        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# -- WER / CER computation ------------------------------------------------------

def _compute_metrics(
    references: List[str],
    hypotheses: List[str],
) -> Tuple[float, float]:
    """Compute WER and CER using jiwer. Both return values in [0, 1]."""
    try:
        import jiwer
    except ImportError as exc:
        raise ImportError("jiwer is required for evaluation: pip install jiwer") from exc

    transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ])
    char_transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfChars(),
    ])

    wer_out = jiwer.process_words(references, hypotheses, reference_transform=transform, hypothesis_transform=transform)
    cer_out = jiwer.process_characters(references, hypotheses, reference_transform=char_transform, hypothesis_transform=char_transform)
    return float(wer_out.wer), float(cer_out.cer)
