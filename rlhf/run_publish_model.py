"""Publish the gate-approved active adapter as a validated CTranslate2 artifact."""
from __future__ import annotations

import argparse
import json
import logging

from paths import DEFAULT_PATHS
from storage.feedback_store import FeedbackStore
from utils.asr_publication import approved_policy, publish_approved_model
from utils.log import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--if-ready", action="store_true")
    args = parser.parse_args()
    setup_logging(logs_dir=DEFAULT_PATHS.logs_dir)
    store = FeedbackStore(DEFAULT_PATHS.db_path)
    if args.if_ready:
        try:
            approved_policy(store, DEFAULT_PATHS)
        except ValueError as exc:
            logging.getLogger(__name__).info("ASR publication not ready: %s", exc)
            store.set_setting("asr_publication_status", "not_ready")
            store.set_setting("asr_publication_log", str(exc))
            return 0
    store.set_setting("asr_publication_status", "running")
    try:
        manifest = publish_approved_model(store, DEFAULT_PATHS)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        store.set_setting("asr_publication_status", "failed")
        store.set_setting("asr_publication_log", f"{type(exc).__name__}: {exc}")
        logging.getLogger(__name__).exception("ASR publication failed")
        raise
    store.set_setting("asr_publication_status", "ready")
    store.set_setting("asr_publication_id", manifest["publication_id"])
    store.set_setting("asr_publication_log", "Verified CTranslate2 publication ready")
    print(json.dumps({"publication_id": manifest["publication_id"],
                      "checkpoint_id": manifest["checkpoint_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
