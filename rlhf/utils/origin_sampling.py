"""Shared hard caps for persistent TTS audio, independent of DPO artifacts."""
from __future__ import annotations

import math
import random
from collections import Counter


def select_origin_capped(rows, share_percent=0, limit=None, seed=0, one_tts_per_utterance=False):
    share = float(share_percent)
    if not math.isfinite(share) or not 0 <= share <= 100:
        raise ValueError("TTS share must be between 0 and 100")
    rows = list(rows)
    recorded = [r for r in rows if r.get("origin", "recorded") == "recorded"]
    tts = [r for r in rows if r.get("origin") == "synthesized"]
    rng = random.Random(seed)
    rng.shuffle(recorded)
    rng.shuffle(tts)
    if one_tts_per_utterance:
        seen = set()
        unique = []
        for row in tts:
            utterance = row.get("utterance_id")
            if utterance is not None and utterance not in seen:
                unique.append(row)
                seen.add(utterance)
        tts = unique
    if limit is not None:
        if limit < 0:
            raise ValueError("Sample limit must be nonnegative")
        # Reserve the recorded quota even if recorded data is unavailable.
        max_tts = math.floor(limit * share / 100)
        recorded = recorded[:limit - max_tts]
        if share < 100:
            max_tts = min(max_tts, math.floor(len(recorded) * share / (100 - share)))
        tts = tts[:max_tts]
        # Recorded audio may fill a missing TTS quota, never the converse.
        chosen_ids = {id(r) for r in recorded}
        extra = [r for r in rows if r.get("origin", "recorded") == "recorded"
                 and id(r) not in chosen_ids]
        recorded += extra[:limit - len(recorded) - len(tts)]
    else:
        max_tts = len(tts) if share == 100 else math.floor(len(recorded) * share / (100 - share))
        tts = tts[:max_tts]
    selected = recorded + tts
    rng.shuffle(selected)
    return selected


def composition(rows):
    return {
        "total": len(rows),
        "origins": dict(Counter(r.get("origin", "recorded") for r in rows)),
        "corpora": dict(Counter(r.get("dataset", "") for r in rows)),
        "samples": [
            {"audio_id": r.get("audio_id", r.get("id")), **{
                k: r.get(k) for k in ("dataset", "origin", "utterance_id", "provenance_json")
            }} for r in rows
        ],
    }
