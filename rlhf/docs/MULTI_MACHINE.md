# Multi-Machine Annotation Guide

## Overview

Each annotator machine runs its own SQLite database at `_outputs/feedback.sqlite`. After annotation sessions, exports are merged on a designated "master" machine. UUID4 primary keys guarantee no collisions.

---

## Setup Per Machine

1. Clone or copy the project directory.
2. Edit `default_paths.json` with paths valid for that machine.
3. Ensure audio files referenced in the CSVs are accessible on that machine.
4. Each machine generates its own hypothesis pairs independently (or they can be shared by copying the DB before annotation begins).

---

## Recommended Workflow

### Option A — Independent pair generation (simplest)

Each machine generates its own pairs from the shared dataset:

```
Machine 1:   py run_generate_pairs.py
Machine 2:   py run_generate_pairs.py
...
```

After annotation, merge all feedback. Duplicate pairs may exist across machines but are handled gracefully (different UUIDs → both kept).

### Option B — Shared pair pool (more efficient)

One machine generates all pairs, exports the DB, and shares it:

```
Master:     py run_generate_pairs.py
            py run_merge_feedback.py --export   → master_initial.json
            
Each node:  copy master_initial.json locally
            py run_merge_feedback.py --merge master_initial.json
            (now has the same pairs; annotators work from the shared pool)
```

---

## Exporting Annotations

At the end of an annotation session (or any time), export the local DB:

```
py run_merge_feedback.py --export
```

This creates `_outputs/feedback_export_YYYYMMDD_HHMMSS.json`.

---

## Merging on the Master Machine

Copy all export JSON files to one machine, then:

```
py run_merge_feedback.py --merge machine1.json machine2.json machine3.json
```

The tool prints a summary:

```
Merge summary:
  audio_files:         [A] inserted  [B] already present
  hypothesis_pairs:    [A] inserted  [B] already present
  annotation_sessions: [A] inserted  [B] already present
  annotations:         [A] inserted  [B] already present
  reward_scores:       [A] inserted  [B] already present
  training_runs:       [A] inserted  [B] already present
```

After merging, run reward scoring and training on the master machine.

---

## Conflict Resolution Rules

| Scenario | Rule |
|----------|------|
| Same `pair_id` + same `participant_id` | Keep most recent `annotated_at` (de-duplicate re-annotations of the same pair by the same person) |
| Same `pair_id` + different `participant_id` | Keep both rows (inter-annotator agreement data) |
| Same `audio_file` path but different UUID | Both kept — likely generated on different machines |
| Same `training_run` UUID | `INSERT OR IGNORE` — no overwrite (training runs are machine-local) |

---

## Why This Works Without a Coordinator

All primary keys are generated as `uuid.uuid4()` in Python before insertion. The probability of two UUIDs colliding is astronomically small (~1 in 5.3×10³⁶). No sequence, no central counter, no clock-based ID — pure random 128-bit values.

The merge is a pure **set union** per table:
- If a row's UUID is new → insert.
- If a row's UUID already exists → ignore (the data is already present).

---

## IAA Analysis

Annotations from multiple participants on the same pair are preserved. To compute inter-annotator agreement:

```python
from storage.feedback_store import FeedbackStore

store = FeedbackStore("_outputs/feedback.sqlite")

# Get all annotations for pairs that were annotated by >1 participant
import sqlite3, collections

conn = sqlite3.connect("_outputs/feedback.sqlite")
rows = conn.execute("""
    SELECT a.pair_id, s.participant_id, a.preference
    FROM annotations a
    JOIN annotation_sessions s ON a.session_id = s.id
    WHERE a.preference != 'skip'
    ORDER BY a.pair_id, s.participant_id
""").fetchall()

by_pair = collections.defaultdict(list)
for pair_id, participant, pref in rows:
    by_pair[pair_id].append((participant, pref))

multi = {k: v for k, v in by_pair.items() if len({p for p, _ in v}) > 1}
print(f"Pairs with multi-annotator data: {len(multi)}")
```

---

## Notes

- Reward scoring and RLHF training are always run on the master machine after merge.
- Each annotator machine runs its own `web` service and local SQLite database;
  pair generation is optional when queues are prepared centrally.
- The browser UI can remain local to the machine; export and merge use JSON files.
