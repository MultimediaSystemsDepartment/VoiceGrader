"""
run_merge_feedback.py -- merge JSON exports from multiple annotation machines.

Usage:
    # Export from this machine first
    py run_merge_feedback.py --export --output exports/machine_A_2026-06-27.json

    # Merge several exports into the master DB
    py run_merge_feedback.py --merge exports/machine_A.json exports/machine_B.json

    # Export + merge in one step (export this machine, merge with others)
    py run_merge_feedback.py --export --output exports/this_machine.json ^
        --merge exports/machine_B.json exports/machine_C.json
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import DEFAULT_PATHS
from storage import FeedbackStore
from storage.merger import merge_exports, report_merge_summary
from utils.log import get_logger, setup_logging

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export / merge annotation feedback across machines")
    p.add_argument("--export", action="store_true",
                   help="Export this machine's DB to a JSON file")
    p.add_argument("--output", type=Path, default=None,
                   help="Path for the exported JSON (default: exports/<hostname>_<date>.json)")
    p.add_argument("--merge", nargs="+", type=Path, default=None, metavar="JSON",
                   help="JSON export files to merge into the master DB")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.export and not args.merge:
        print("Specify --export, --merge, or both.  Use --help for usage.")
        sys.exit(1)

    setup_logging(logs_dir=DEFAULT_PATHS.logs_dir)
    log.info("=== run_merge_feedback ===")

    store = FeedbackStore(DEFAULT_PATHS.db_path)

    if args.export:
        import socket
        out = args.output or (
            Path("exports") /
            f"{socket.gethostname()}_{datetime.now().strftime('%Y-%m-%d_%H%M')}.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        store.export_to_json(out)
        log.info("Exported to: %s", out)

    if args.merge:
        counts = merge_exports(args.merge, master_store=store)
        print(report_merge_summary(counts))


if __name__ == "__main__":
    main()
