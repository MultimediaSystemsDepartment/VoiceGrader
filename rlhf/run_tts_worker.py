"""Run one bounded persisted TTS work item, or explicitly drain ready work."""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    from paths import DEFAULT_PATHS
    from storage.feedback_store import FeedbackStore
    from tts.worker import TTSWorker

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_PATHS.db_path)
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_PATHS.output_dir) / "tts_audio")
    parser.add_argument("--max-items", type=int, default=1, help="Bound ready work processed (no waiting)")
    args = parser.parse_args()
    if not 1 <= args.max_items <= 100:
        parser.error("--max-items must be 1-100")
    worker = TTSWorker(FeedbackStore(args.db), args.output_dir)
    for _ in range(args.max_items):
        if not worker.run_once():
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
