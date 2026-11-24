from pathlib import Path
import wave
import zipfile

import numpy as np

from utils.audio import cache_normalized_wav, cached_wav_path, load_audio


def test_zip_cache_key_survives_container_path_translation(tmp_path: Path) -> None:
    host_path = "zip:C:/dataset/source.zip!clips/sample.wav"
    container_path = "zip:/app/_data/source.zip!clips/sample.wav"

    assert cached_wav_path(host_path, tmp_path) == cached_wav_path(
        container_path, tmp_path
    )


def test_normalized_wav_cache_reuses_existing_file(
    tmp_path: Path, monkeypatch
) -> None:
    waveform = np.zeros(1_600, dtype=np.float32)
    cached = cache_normalized_wav(
        "zip:/app/_data/source.zip!clips/sample.wav",
        tmp_path,
        waveform=waveform,
    )
    initial_bytes = cached.read_bytes()

    monkeypatch.setattr(
        "utils.audio.load_audio",
        lambda path: (_ for _ in ()).throw(AssertionError("cache miss")),
    )

    reused = cache_normalized_wav(
        "zip:C:/dataset/source.zip!clips/sample.wav",
        tmp_path,
    )

    assert reused == cached
    assert reused.read_bytes() == initial_bytes


def test_historical_human_zip_path_recovers_archive_prefix(tmp_path: Path) -> None:
    wav_path = tmp_path / "sample.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(np.zeros(1_600, dtype=np.int16).tobytes())

    archive_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(
            wav_path,
            "human_voices/human/human/session/sample.wav",
        )

    waveform, sample_rate = load_audio(
        f"zip:{archive_path}!human/human/session/sample.wav"
    )

    assert waveform.shape == (1_600,)
    assert sample_rate == 16_000


def test_existing_archives_with_same_basename_do_not_share_audio_cache(tmp_path: Path) -> None:
    archives = []
    for corpus in ("first", "second"):
        directory = tmp_path / corpus
        directory.mkdir()
        archive = directory / "clips.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("sample.wav", b"fixture")
        archives.append(f"zip:{archive}!sample.wav")
    assert cached_wav_path(archives[0], tmp_path) != cached_wav_path(archives[1], tmp_path)