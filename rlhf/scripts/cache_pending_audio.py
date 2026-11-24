"""Materialize browser-ready WAV files for pending annotation pairs."""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from paths import DEFAULT_PATHS
from storage.feedback_store import FeedbackStore
from utils.audio import cache_normalized_wav

log = logging.getLogger(__name__)


def cache_pending_audio(limit: int = 0, workers: int = 4) -> tuple[int, int]:
    store = FeedbackStore(DEFAULT_PATHS.db_path)
    pairs = store.get_pending_pairs(limit=limit or store.count_pending_pairs())
    audio_files = {}
    for pair in pairs:
        audio = store.get_audio_file(pair.audio_id)
        if audio is not None:
            audio_files[audio.id] = audio

    cache_dir = Path(DEFAULT_PATHS.output_dir) / "audio_cache"
    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(cache_normalized_wav, audio.file_path, cache_dir): audio
            for audio in audio_files.values()
        }
        for future in as_completed(futures):
            audio = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures += 1
                log.error("Failed to cache %s: %s", audio.file_path, exc)
    return len(audio_files) - failures, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="0 caches all pending pairs")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cached, failures = cache_pending_audio(limit=args.limit, workers=args.workers)
    print(f"Cached pending audio: {cached} succeeded, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())