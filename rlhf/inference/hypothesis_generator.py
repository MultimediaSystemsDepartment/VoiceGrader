"""
HypothesisGenerator -- ingests audio CSVs and populates the hypothesis_pairs queue.

Two responsibilities:
  1. load_audio_catalogue() -- read CSV manifests -> insert AudioFile rows into DB
  2. generate_candidates(audio_files, n) -- run deterministic beam search plus
      temperature sampling and return n_candidates hypotheses per file

Designed as a context manager so the GPU is released promptly:

    with HypothesisGenerator(store, config, paths) as gen:
        gen.load_audio_catalogue()
        candidates = gen.generate_candidates(audio_files, n_candidates=6)
    # Whisper unloaded here

Can also be used without context manager by calling load_model() / unload_model()
explicitly -- useful when the caller wants to reuse the loaded model across calls.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from tqdm import tqdm
from paths import PathsConfig
from storage import Annotation, AudioFile, FeedbackStore, HypothesisPair
from utils.audio import cache_normalized_wav, load_audio, validate_audio_file
from utils.torch_runtime import configure_torch_runtime
from utils.policy_checkpoint import validate_policy_checkpoint

from .config import InferenceConfig

log = logging.getLogger(__name__)


def _clip_generation_seed(generation_seed: int, file_path: str) -> int:
    payload = f"{generation_seed}\0{file_path}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace for WER comparison."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _compute_wer(hypothesis: str, reference: str) -> float:
    """Word Error Rate between hypothesis and reference. 0.0 = exact normalized match."""
    hyp_n = _normalize_text(hypothesis)
    ref_n = _normalize_text(reference)
    if hyp_n == ref_n:
        return 0.0
    try:
        from jiwer import wer as _jiwer_wer
        return float(_jiwer_wer(ref_n, hyp_n))
    except Exception:
        return 1.0

# Map dataset names to PathsConfig attribute names for each split
_CSV_MAP: Dict[str, List[Tuple[str, str]]] = {
    "admedvoice": [
        ("train", "admedvoice_train_csv"),
        ("val",   "admedvoice_val_csv"),
        ("test",  "admedvoice_test_csv"),
    ],
    "commonvoice": [
        ("train", "commonvoice_train_csv"),
        ("val",   "commonvoice_val_csv"),
        ("test",  "commonvoice_test_csv"),
    ],
}

# Ordered candidates for auto-detecting CSV column names
_PATH_CANDIDATES       = ["file_path", "path", "audio", "filename", "file", "audio_path"]
_TRANSCRIPT_CANDIDATES = ["sentence", "transcript", "text", "transcription", "label", "reference"]


class HypothesisGenerator:
    """
    Loads Whisper lazily (only when generate_pairs is called).
    Thread-safe for catalogue loading; generate_pairs is GPU-bound and not
    intended for concurrent use.
    """

    def __init__(
        self,
        store: FeedbackStore,
        config: InferenceConfig,
        paths: PathsConfig,
        policy_checkpoint: Optional[Union[str, Path]] = None,
    ) -> None:
        self.store = store
        self.config = config
        self.paths = paths
        self.policy_checkpoint = Path(policy_checkpoint) if policy_checkpoint else None
        self._model = None
        self._processor = None

    @property
    def model_version(self) -> str:
        if self.policy_checkpoint is None:
            return self.config.model_id
        return f"{self.config.model_id}+lora:{self.policy_checkpoint.name}"

    # -- Model lifecycle --------------------------------------------------------

    def load_model(self) -> None:
        """Load Whisper to GPU. No-op if already loaded."""
        if self._model is not None:
            return

        if self.policy_checkpoint is not None:
            validate_policy_checkpoint(self.policy_checkpoint, self.config.model_id)
        configure_torch_runtime()
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        log.info(
            "Loading Whisper policy: %s  (dtype=%s, device=%s)",
            self.model_version, self.config.torch_dtype, self.config.device,
        )

        dtype = torch.float16 if self.config.torch_dtype == "float16" else torch.bfloat16

        self._processor = WhisperProcessor.from_pretrained(self.config.model_id)
        base = WhisperForConditionalGeneration.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
            attn_implementation=self.config.attn_implementation,
        ).to(self.config.device)
        if self.policy_checkpoint is not None:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(base, self.policy_checkpoint)
        else:
            self._model = base
        self._model.eval()

        log.info("Whisper ready on %s", self.config.device)

    def unload_model(self) -> None:
        """Release Whisper from GPU memory."""
        if self._model is None:
            return

        import torch

        del self._model, self._processor
        self._model = None
        self._processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log.info("Whisper unloaded")

    def __enter__(self) -> HypothesisGenerator:
        self.load_model()
        return self

    def __exit__(self, *_) -> None:
        self.unload_model()

    # -- Audio catalogue ingestion ----------------------------------------------

    def load_audio_catalogue(
        self,
        datasets: Optional[List[str]] = None,
        splits: Optional[List[str]] = None,
    ) -> int:
        """
        Read CSV manifests and register all audio files in the DB (idempotent).
        Already-registered paths are silently skipped.

        Args:
            datasets: subset of ['admedvoice', 'commonvoice']. None -> use config default.
            splits:   subset of ['train', 'val', 'test'].       None -> use config default.

        Returns:
            Number of newly inserted audio_file rows.
        """
        datasets = datasets or self.config.datasets
        splits   = splits   or self.config.splits
        total    = 0

        # Fetch all already-registered paths in one query so _ingest_csv can do
        # O(1) set lookups instead of one DB round-trip per row.
        known_paths: set = self.store.get_all_audio_paths()
        log.info("Catalogue: %d paths already registered", len(known_paths))

        for dataset in datasets:
            if dataset not in _CSV_MAP:
                log.warning("Unknown dataset %r -- skipping", dataset)
                continue
            for split, paths_attr in _CSV_MAP[dataset]:
                if split not in splits:
                    continue
                csv_path = getattr(self.paths, paths_attr, "")
                if not csv_path:
                    log.warning("CSV path not configured for %s/%s -- skipping", dataset, split)
                    continue
                if not Path(csv_path).exists():
                    log.warning("CSV not found: %s -- skipping", csv_path)
                    continue
                audio_dir = getattr(self.paths, f"{dataset}_audio_dir", "")
                n = self._ingest_csv(csv_path, dataset, audio_dir, known_paths)
                # known_paths is mutated inside _ingest_csv (.add() per inserted row)
                log.info("Catalogued %s/%s: +%d audio files", dataset, split, n)
                total += n

        return total

    def _ingest_csv(
        self,
        csv_path: str,
        dataset: str,
        audio_dir: str,
        known_paths: set,
    ) -> int:
        """Read one CSV manifest and register each audio file row. Returns count inserted.

        known_paths is a pre-fetched set of already-registered file_path values.
        Membership is checked via O(1) set lookup (no per-row DB queries).
        Newly inserted paths are added to known_paths so subsequent CSV files
        in the same run stay consistent.

        Validation is intentionally lightweight: zip paths use the cached
        central-directory check; plain paths just need to exist on disk.
        Full audio decoding / duration probing is deferred to inference time.
        """
        delimiter = self.paths.csv_delimiter
        path_col, transcript_col = _detect_columns(csv_path, delimiter)
        log.debug("CSV columns detected: path=%r  transcript=%r", path_col, transcript_col)

        inserted = 0
        to_insert: list = []
        with open(csv_path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            for row in tqdm(reader, desc=f"Cataloguing {dataset}", unit="rows", leave=False):
                raw_path = (row.get(path_col) or "").strip()
                if not raw_path:
                    continue

                # zip:-prefixed paths must NOT go through Path() -- it converts
                # forward slashes to backslashes on Windows, breaking zip lookups.
                if raw_path.startswith("zip:"):
                    file_path = raw_path
                else:
                    audio_path = Path(raw_path)
                    if not audio_path.is_absolute() and audio_dir:
                        audio_path = Path(audio_dir) / audio_path
                    file_path = str(audio_path)

                # Fast deduplication: O(1) set lookup instead of a DB query per row.
                if file_path in known_paths:
                    continue

                # Lightweight existence check only -- full audio validation
                # happens at inference time when the file is actually loaded.
                if raw_path.startswith("zip:"):
                    ok, msg = validate_audio_file(file_path)  # uses cached ZIP directory
                    if not ok:
                        log.debug("Skipping [%s]: %s", msg, file_path)
                        continue
                else:
                    if not Path(file_path).exists():
                        log.debug("Skipping (not found): %s", file_path)
                        continue

                reference = (row.get(transcript_col) or "").strip() or None

                af = AudioFile.new(
                    file_path=file_path,
                    dataset=dataset,
                    reference_transcript=reference,
                    duration_s=None,   # probing every file header is too slow; defer to load time
                    sample_rate=16_000,
                )
                to_insert.append(af)
                known_paths.add(file_path)  # keep set consistent for remaining CSVs

        # Single-transaction bulk insert -- far faster than one commit per row.
        inserted = self.store.bulk_insert_audio_files(to_insert)
        return inserted

    # -- Whisper transcription --------------------------------------------------

    def _transcribe_files_nbest(
        self,
        file_paths: List[str],
        n_candidates: int,
    ) -> List[List[Optional[str]]]:
        """
        Two-pass hybrid generation per file:

          Pass 1 — Beam search (num_beams=5): one deterministic best hypothesis.
          Pass 2 — Temperature sampling (temperature from config.nbest_temperature,
                   n=n_candidates-1): stochastic alternatives.

        Returns a list per file where index 0 is the beam result and indices
        1..n_candidates-1 are temperature samples.

        In cold-start pairing (_generate_nbest in run_generate_pairs.py) the
        caller picks the sample most dissimilar to the beam result (by character
        edit distance) as the "worse" hypothesis.

        Why not diverse beam search (num_beam_groups)?
        Diverse beam search collapses to identical outputs on high-confidence
        medical speech regardless of diversity_penalty value.  In transformers
        >=4.48 it also requires trust_remote_code=True (community model).
        Temperature sampling naturally produces distinct transcriptions for any
        confidence level and needs no external dependencies.
        """
        import torch
        from .decoding_strategies import BEAM_5

        n_samples = max(1, n_candidates - 1)
        temperature = getattr(self.config, "nbest_temperature", 0.8)

        # --- Load all waveforms once (shared between both passes) -------------
        waveforms: List[Optional[object]] = []
        for fp in tqdm(file_paths, desc="Loading audio", unit="file", leave=False):
            try:
                wav, _ = load_audio(fp)
                waveforms.append(wav)
            except Exception as exc:
                log.warning("Audio load failed (%s): %s", exc, fp)
                waveforms.append(None)
                continue
            try:
                cache_normalized_wav(
                    fp,
                    Path(self.paths.output_dir) / "audio_cache",
                    waveform=wav,
                )
            except Exception as exc:
                log.warning("Browser audio cache write failed (%s): %s", exc, fp)

        def _batch_generate(valid_wavs: list, gen_kwargs: dict) -> List[str]:
            inputs = self._processor(
                valid_wavs, sampling_rate=16_000, return_tensors="pt", padding=True
            )
            feats = inputs.input_features.to(self._model.device, dtype=self._model.dtype)
            with torch.inference_mode():
                ids = self._model.generate(input_features=feats, **gen_kwargs)
            return self._processor.batch_decode(ids, skip_special_tokens=True)

        # --- Pass 1: beam search → best hypothesis ----------------------------
        beam_kwargs = BEAM_5.to_generate_kwargs()
        beam_kwargs["num_beams"] = 5
        beam_kwargs["num_return_sequences"] = 1
        beam_kwargs["do_sample"] = False

        beam_batch = max(1, self.config.batch_size)
        beam_texts: List[Optional[str]] = []
        for start in tqdm(range(0, len(waveforms), beam_batch),
                          desc="Beam search pass", unit="batch", leave=False):
            batch_wavs = waveforms[start: start + beam_batch]
            valid_wavs = [w for w in batch_wavs if w is not None]
            if not valid_wavs:
                beam_texts.extend([None] * len(batch_wavs))
                continue
            texts = _batch_generate(valid_wavs, beam_kwargs)
            it = iter(texts)
            for w in batch_wavs:
                beam_texts.append(next(it) if w is not None else None)

        # --- Pass 2: temperature sampling → diverse alternatives --------------
        sample_kwargs = BEAM_5.to_generate_kwargs()
        sample_kwargs["do_sample"] = True
        sample_kwargs["temperature"] = temperature
        sample_kwargs["num_beams"] = 1
        sample_kwargs["num_return_sequences"] = n_samples

        # Reduce batch size: HF returns n_samples seqs per input item
        sample_batch = max(1, self.config.batch_size // n_samples)
        generation_seed = getattr(self.config, "generation_seed", None)
        if generation_seed is not None:
            sample_batch = 1
        sample_texts: List[List[Optional[str]]] = []
        for start in tqdm(range(0, len(waveforms), sample_batch),
                          desc=f"Sampling pass (t={temperature})", unit="batch", leave=False):
            batch_wavs = waveforms[start: start + sample_batch]
            valid_wavs = [w for w in batch_wavs if w is not None]
            if not valid_wavs:
                sample_texts.extend([[None] * n_samples] * len(batch_wavs))
                continue
            if generation_seed is not None:
                clip_seed = _clip_generation_seed(generation_seed, file_paths[start])
                torch.manual_seed(clip_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(clip_seed)
            texts = _batch_generate(valid_wavs, sample_kwargs)
            grouped = [
                texts[i * n_samples: (i + 1) * n_samples]
                for i in range(len(valid_wavs))
            ]
            it = iter(grouped)
            for w in batch_wavs:
                sample_texts.append(next(it) if w is not None else [None] * n_samples)

        log.debug(
            "  beam pass: %d files  sample pass: %d files  n_samples=%d  t=%.2f",
            len(beam_texts), len(sample_texts), n_samples, temperature,
        )

        # --- Combine: index 0 = beam result, 1..n = samples ------------------
        return [[beam] + samples for beam, samples in zip(beam_texts, sample_texts)]

    def generate_candidates(
        self,
        audio_files: List["AudioFile"],
        n_candidates: int,
        sampling_strategy: Optional[DecodingStrategy] = None,
    ) -> List[List[Optional[str]]]:
        """
        For each audio file generate n_candidates transcriptions using N-best
        beam search (single forward pass, hypotheses ranked by beam score).

        This is the Beam(x, N) approach from Prabhavalkar et al. 2018 / the RSO
        paper: one beam search with beam_size=N, returning all N hypotheses.
        Compared to N separate sampling passes this is faster, produces more
        diverse hypotheses (the beam explores genuinely different paths), and the
        best/worst gap is typically larger (cleaner training signal).

        Args:
            audio_files:       AudioFile records to transcribe.
            n_candidates:      Beam size -- how many hypotheses to return per clip.
            sampling_strategy: Ignored (kept for API compatibility).

        Returns:
            Outer list aligns with audio_files.
            Inner list has n_candidates entries, best hypothesis first.
        """
        if self._model is None:
            raise RuntimeError(
                "Whisper not loaded. Use as context manager or call load_model() first."
            )

        log.info(
            "N-best beam+sample  n_candidates=%d  files=%d",
            n_candidates, len(audio_files),
        )
        return self._transcribe_files_nbest(
            [af.file_path for af in audio_files], n_candidates
        )


# -- CSV column auto-detection --------------------------------------------------

def _detect_columns(csv_path: str, delimiter: str) -> Tuple[str, str]:
    """
    Read the CSV header and return (path_column, transcript_column) names.
    Falls back to first and second columns if no known name is found.
    """
    with open(csv_path, encoding="utf-8", newline="") as fh:
        header = next(csv.reader(fh, delimiter=delimiter), [])

    path_col = next(
        (h for h in header if h.lower().strip() in _PATH_CANDIDATES),
        header[0] if header else "path",
    )
    transcript_col = next(
        (h for h in header if h.lower().strip() in _TRANSCRIPT_CANDIDATES),
        header[1] if len(header) > 1 else "sentence",
    )
    return path_col, transcript_col
