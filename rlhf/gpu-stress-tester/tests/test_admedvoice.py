import io
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from gpu_stress_tester.admedvoice import catalogue, load_audio, sample_records


def _archive(path: Path) -> None:
    audio = io.BytesIO()
    sf.write(audio, np.zeros(16_000, dtype=np.float32), 16_000, format="WAV")
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("session/metadata.csv", "filename;id;phrase\nclip.wav;1;Dzien dobry\n")
        bundle.writestr("session/clip.wav", audio.getvalue())


def test_catalogue_and_load_real_archive_member(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    _archive(archive)

    records = catalogue(archive)
    waveform = load_audio(archive, records[0].member)

    assert records[0].transcript == "Dzien dobry"
    assert waveform.shape == (16_000,)
    assert waveform.dtype == np.float32


def test_sampling_is_reproducible(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    _archive(archive)
    records = catalogue(archive)

    assert sample_records(records, 1, 7) == sample_records(records, 1, 7)
