from __future__ import annotations

import csv
import io
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np


@dataclass(frozen=True)
class AudioRecord:
    member: str
    transcript: str


def catalogue(archive: Path) -> list[AudioRecord]:
    if not archive.is_file():
        raise FileNotFoundError(f"ADMEDVOICE archive not found: {archive}")
    records: list[AudioRecord] = []
    with zipfile.ZipFile(archive) as bundle:
        members = set(bundle.namelist())
        for csv_name in sorted(name for name in members if name.lower().endswith(".csv")):
            content = bundle.read(csv_name).decode("utf-8", errors="replace")
            parent = PurePosixPath(csv_name).parent
            for row in csv.DictReader(content.splitlines(), delimiter=";", quotechar='"'):
                filename = (row.get("filename") or "").strip().strip('"')
                transcript = (row.get("phrase") or "").strip().strip('"')
                member = str(parent / filename)
                if filename and transcript and member in members:
                    records.append(AudioRecord(member=member, transcript=transcript))
    if not records:
        raise ValueError("ADMEDVOICE archive contains no usable audio/transcript rows")
    return records


def sample_records(records: list[AudioRecord], count: int, seed: int) -> list[AudioRecord]:
    if count > len(records):
        raise ValueError(f"Requested {count} samples but archive contains {len(records)}")
    return random.Random(seed).sample(records, count)


def load_audio(
    archive: Path,
    member: str,
    target_rate: int = 16_000,
    max_duration_seconds: float = 30.0,
) -> np.ndarray:
    import soundfile as sf

    with zipfile.ZipFile(archive) as bundle:
        payload = bundle.read(member)
    waveform, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if sample_rate != target_rate:
        import resampy

        waveform = resampy.resample(waveform, sample_rate, target_rate).astype(np.float32)
    maximum = int(target_rate * max_duration_seconds)
    waveform = waveform[:maximum]
    if len(waveform) < target_rate // 10:
        raise ValueError(f"Audio clip is shorter than 0.1 seconds: {member}")
    return waveform
