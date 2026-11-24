"""
Audio loading and validation utilities.

All audio in this project must be 16 kHz mono float32 before it is passed
to Whisper or stored as a hypothesis pair. This module enforces that invariant.

Usage:
    from utils.audio import load_audio, get_duration
    waveform, sr = load_audio("path/to/file.wav")        # plain file
    waveform, sr = load_audio("zip:archive.zip!inner/file.wav")  # inside a ZIP
"""

from __future__ import annotations

import io
import hashlib
import logging
import os
import zipfile
from pathlib import Path
from typing import Tuple, Union

import numpy as np

log = logging.getLogger(__name__)

# Module-level cache: zip_path -> frozenset of inner names.
# Avoids reopening the ZIP central directory for every validate_audio_file() call.
_zip_name_cache: dict = {}

TARGET_SR = 16_000          # Whisper's required sample rate
TARGET_CHANNELS = 1         # mono
MAX_DURATION_S = 30.0       # Whisper's hard limit per segment
MIN_DURATION_S = 0.1        # reject near-silent / empty clips

# Prefix used for paths that live inside a ZIP archive.
# Format:  zip:<absolute_or_relative_zip_path>!<inner_path>
# Example: zip:_data/archive.zip!anoni/anoni/cat_02-001/.../file.wav
_ZIP_PREFIX = "zip:"


# Project root is two levels up from this file (utils/audio.py -> utils/ -> root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def cached_wav_path(source_path: Union[str, Path], cache_dir: Union[str, Path]) -> Path:
    source_key = str(source_path)
    if source_key.startswith(_ZIP_PREFIX) and "!" in source_key:
        archive, inner = source_key[len(_ZIP_PREFIX):].split("!", 1)
        resolved = Path(_resolve_zip_archive(archive))
        archive_key = (
            resolved.resolve().as_posix()
            if resolved.is_file()
            else archive.replace("\\", "/").rsplit("/", 1)[-1]
        )
        source_key = f"{_ZIP_PREFIX}{archive_key}!{inner}"
    key = hashlib.sha256(source_key.encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{key}.wav"


def cache_normalized_wav(
    source_path: Union[str, Path],
    cache_dir: Union[str, Path],
    waveform: np.ndarray | None = None,
    sample_rate: int = TARGET_SR,
) -> Path:
    destination = cached_wav_path(source_path, cache_dir)
    if destination.is_file():
        return destination

    if waveform is None:
        waveform, sample_rate = load_audio(source_path)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{id(waveform)}.tmp")
    try:
        import soundfile as sf

        sf.write(temporary, waveform, sample_rate, format="WAV", subtype="PCM_16")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _resolve_zip_archive(zip_path: str) -> str:
    """
    Resolve the archive portion of a zip: path.

    When the stored absolute path no longer exists (e.g. the archive was on a
    different machine or in OneDrive, and is now in _data/ inside Docker or on
    a new host), fall back to _data/<filename> relative to the project root.

    This makes zip: paths portable across machines without re-running prepare_data.py.
    """
    if Path(zip_path).exists():
        return zip_path
    fallback = _PROJECT_ROOT / "_data" / Path(zip_path).name
    if fallback.exists():
        return str(fallback)
    return zip_path  # return original so the FileNotFoundError is informative


def _open_audio_bytes(path: str) -> Tuple[io.BytesIO, str]:
    """
    Return a (BytesIO, filename) pair for a plain file or a zip:-prefixed path.
    The filename is used to hint the format to soundfile / torchaudio.
    """
    if path.startswith(_ZIP_PREFIX):
        rest = path[len(_ZIP_PREFIX):]
        if "!" not in rest:
            raise ValueError(
                f"zip: path must contain '!' to separate archive from inner path: {path}"
            )
        zip_path, inner = rest.split("!", 1)
        zip_path = _resolve_zip_archive(zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            try:
                data = zf.read(inner)
            except KeyError:
                if not inner.startswith("human/human/"):
                    raise
                data = zf.read(f"human_voices/{inner}")
        return io.BytesIO(data), inner
    else:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Audio file not found: {p}")
        return io.BytesIO(p.read_bytes()), str(p)


def _read_waveform(buf: io.BytesIO, filename: str) -> Tuple[np.ndarray, int]:
    """
    Decode audio bytes to (waveform, sample_rate).
    Tries soundfile first; falls back to torchaudio for formats like MP3.
    """
    ext = Path(filename).suffix.lower()

    # soundfile handles WAV, FLAC, OGG natively
    if ext not in (".mp3",):
        try:
            import soundfile as sf
            buf.seek(0)
            waveform, sr = sf.read(buf, dtype="float32", always_2d=True)
            return waveform, sr
        except Exception:
            pass  # fall through to torchaudio

    # torchaudio handles MP3 and is a good fallback
    try:
        import torchaudio
        buf.seek(0)
        tensor, sr = torchaudio.load(buf, format=ext.lstrip(".") if ext else None)
        waveform = tensor.numpy().T  # (samples, channels)
        return waveform.astype(np.float32), sr
    except Exception as exc:
        raise RuntimeError(
            f"Cannot decode audio '{filename}': {exc}. "
            "Install torchaudio for MP3 support."
        ) from exc


def load_audio(
    path: Union[str, Path],
    target_sr: int = TARGET_SR,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and return (waveform, sample_rate).

    Accepts:
      - Plain file path (str or Path)
      - zip:-prefixed path: "zip:<zipfile>!<inner_path>"

    - Resamples to target_sr if needed.
    - Converts to mono by averaging channels.
    - Normalises dtype to float32 in [-1.0, 1.0].

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file is empty or exceeds MAX_DURATION_S.
    """
    path = str(path)
    buf, filename = _open_audio_bytes(path)
    waveform, sr = _read_waveform(buf, filename)

    # waveform shape: (samples, channels) or (samples,) from torchaudio transpose
    # waveform shape: (samples, channels)

    # Ensure always_2d (samples, channels)
    if waveform.ndim == 1:
        waveform = waveform[:, np.newaxis]

    # Mono conversion
    if waveform.shape[1] > 1:
        waveform = waveform.mean(axis=1)
    else:
        waveform = waveform[:, 0]

    # Resample if needed
    if sr != target_sr:
        try:
            import resampy
            waveform = resampy.resample(waveform, sr, target_sr)
        except ImportError:
            # Fallback: scipy
            try:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(target_sr, sr)
                waveform = resample_poly(waveform, target_sr // g, sr // g).astype(np.float32)
            except ImportError as exc:
                raise ImportError(
                    "Resampling requires either resampy or scipy: "
                    "pip install resampy  OR  pip install scipy"
                ) from exc
        sr = target_sr

    duration = len(waveform) / sr
    if duration < MIN_DURATION_S:
        raise ValueError(
            f"Audio too short ({duration:.3f}s < {MIN_DURATION_S}s): {path}"
        )
    if duration > MAX_DURATION_S:
        log.warning(
            "Audio exceeds Whisper's 30 s limit (%.1f s) -- will be truncated: %s",
            duration, path,
        )
        waveform = waveform[: int(MAX_DURATION_S * sr)]

    return waveform.astype(np.float32), sr


def get_duration(path: Union[str, Path]) -> float:
    """
    Return audio duration in seconds.
    For plain files uses soundfile's fast header-only read.
    For zip:-prefixed paths falls back to a full decode (headers are inside the zip).
    """
    path = str(path)
    if not path.startswith(_ZIP_PREFIX):
        try:
            import soundfile as sf
            return sf.info(path).duration
        except Exception:
            pass
    # fallback: full load
    waveform, sr = load_audio(path)
    return len(waveform) / sr


def validate_audio_file(path: Union[str, Path]) -> Tuple[bool, str]:
    """
    Quick sanity check for an audio file (plain or zip:-prefixed).
    Returns (ok: bool, message: str). Intended for dataset ingestion scripts.

    For zip:-prefixed paths uses the ZIP central directory (no decode) --
    checks existence and non-zero compressed size only.
    """
    path = str(path)

    if path.startswith(_ZIP_PREFIX):
        # Fast path: check cached name set -- opens each ZIP only once per process.
        rest = path[len(_ZIP_PREFIX):]
        if "!" not in rest:
            return False, f"Invalid zip: path (missing '!'): {path}"
        zip_path, inner = rest.split("!", 1)
        zip_path = _resolve_zip_archive(zip_path)
        try:
            if zip_path not in _zip_name_cache:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    _zip_name_cache[zip_path] = frozenset(
                        info.filename for info in zf.infolist()
                    )
            if inner not in _zip_name_cache[zip_path]:
                return False, f"Not found in zip: {inner}"
        except Exception as exc:
            return False, f"Cannot open zip: {exc}"
        return True, "OK (zip entry)"

    # Plain file path -- use header-only read
    if not Path(path).exists():
        return False, f"File not found: {path}"
    try:
        duration = get_duration(path)
    except Exception as exc:
        return False, f"Cannot read audio header: {exc}"
    if duration < MIN_DURATION_S:
        return False, f"Too short: {duration:.3f}s"
    return True, f"OK ({duration:.2f}s)"
