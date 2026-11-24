"""
storage/merger.py -- merge JSON exports from multiple annotation machines.

Each machine runs its own SQLite DB and periodically exports via:
    store.export_to_json("session_MACHINE_DATE.json")

This module merges N such files into a single master FeedbackStore.

Usage:
    from storage.merger import merge_exports
    counts = merge_exports(
        json_paths=["machine_a.json", "machine_b.json"],
        master_store=FeedbackStore("master.sqlite"),
    )
    print(counts)  # {'audio_files': 0, 'annotations': 47, ...}

Conflict rules (enforced by INSERT OR IGNORE + UUID PKs):
  - All tables: INSERT OR IGNORE by primary key -- UUID4 guarantees no collisions
    between machines. True duplicates (same export run twice) are silently skipped.
  - Logical duplicates (same pair_id + same participant_id from parallel sessions)
    are kept as separate rows -- both contribute to inter-annotator agreement data.
    Downstream queries should deduplicate by (pair_id, participant_id, MAX(annotated_at))
    when a single-label-per-annotator-per-pair is needed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Union

from .feedback_store import FeedbackStore

log = logging.getLogger(__name__)


def merge_exports(
    json_paths: List[Union[str, Path]],
    master_store: FeedbackStore,
) -> Dict[str, int]:
    """
    Merge one or more JSON exports into master_store.

    Args:
        json_paths:    Paths to JSON files produced by FeedbackStore.export_to_json().
        master_store:  Target FeedbackStore (may be empty or already populated).

    Returns:
        Aggregated insertion counts per table across all files.
    """
    totals: Dict[str, int] = {}

    for raw_path in json_paths:
        path = Path(raw_path)
        if not path.exists():
            log.warning("Export file not found, skipping: %s", path)
            continue

        log.info("Merging: %s", path.name)
        counts = master_store.merge_from_export(path)

        for table, n in counts.items():
            totals[table] = totals.get(table, 0) + n
            if n:
                log.info("  %-25s  +%d rows", table, n)

    return totals


def report_merge_summary(counts: Dict[str, int]) -> str:
    """Format a human-readable summary of merge_exports() output."""
    lines = ["Merge summary:"]
    for table, n in counts.items():
        lines.append(f"  {table:<26} +{n}")
    total = sum(counts.values())
    lines.append(f"  {'TOTAL':<26} +{total}")
    return "\n".join(lines)
