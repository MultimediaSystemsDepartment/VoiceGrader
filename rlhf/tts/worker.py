"""Bounded CPU campaign worker used by the web dispatcher or explicit CLI."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from utils.audio import load_audio
from .provider import provider_for_url, ProviderError, SubmissionRejected

log = logging.getLogger(__name__)


def validate_audio(path):
    info = sf.info(str(path))
    if not np.isfinite(info.duration) or not 0.1 <= info.duration <= 30:
        raise ProviderError("Native TTS duration must be 0.1-30 seconds")
    waveform, _ = load_audio(str(path))
    if not np.isfinite(waveform).all() or not np.any(np.abs(waveform) > 1e-6):
        raise ProviderError("TTS waveform is nonfinite or silent")
    return info.duration, info.samplerate


class TTSWorker:
    def __init__(self, store, output_dir, provider_factory=provider_for_url, clock=time.time):
        self.store = store
        self.output_dir = Path(output_dir)
        self.provider_factory = provider_factory
        self.clock = clock
        self.max_get_failures = 6
        self.max_polls = 120

    def run_once(self):
        item = self.store.claim_tts_item(now=self.clock())
        if item is None:
            return False
        provider = self.provider_factory(item["provider_url"])
        lease = item["lease_until"]
        if item["claimed_state"] == "submitting":
            try:
                job_id = provider.submit(item)
            except SubmissionRejected as exc:
                attempts = item["attempts"] + 1
                retry = exc.retryable and attempts < self.max_polls
                self.store.update_tts_item(
                    item["id"], "pending" if retry else "failed", error=str(exc),
                    attempts=attempts, next_attempt_at=self.clock() + min(300, 5 * 2 ** min(attempts, 6)),
                    lease_until=lease,
                )
            except Exception as exc:
                # Timeout, interruption, malformed acceptance and HTTP failure can
                # all occur after the provider accepted work. No automatic POST retry.
                self.store.update_tts_item(item["id"], "review_required",
                                           error=f"Submission outcome uncertain: {type(exc).__name__}",
                                           lease_until=lease)
            else:
                try:
                    self.store.update_tts_item(item["id"], "accepted", job_id=job_id, lease_until=lease)
                except ValueError as exc:
                    self.store.update_tts_item(item["id"], "review_required",
                        error=f"Accepted job {job_id} could not be associated safely: {exc}",
                        lease_until=lease)
            return True

        staging = None
        try:
            status = provider.poll(item)
            if status["status"] in {"failed", "error", "cancelled", "expired"}:
                self.store.update_tts_item(item["id"], "failed", error=status.get("error") or "Provider job " + status["status"],
                                           lease_until=lease)
                return True
            if status["status"] != "completed":
                attempts = item["attempts"] + 1
                state = "failed" if attempts >= self.max_polls else "accepted"
                self.store.update_tts_item(item["id"], state, attempts=attempts,
                                           error="Polling deadline exceeded" if state == "failed" else None,
                                           next_attempt_at=self.clock() + min(300, 5 * 2 ** min(attempts, 6)),
                                           lease_until=lease)
                return True
            result, entry = provider.result(item)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            # Identity is the durable database item, not an untrusted filename.
            staging = self.output_dir / (item["id"] + f".{int(lease * 1000)}.part")
            staging.unlink(missing_ok=True)
            downloaded_url = provider.download(entry["audio_url"], staging)
            duration, sample_rate = validate_audio(staging)
            digest = hashlib.sha256(staging.read_bytes()).hexdigest()
            if entry.get("sha256") and entry["sha256"] != digest:
                raise ProviderError("Provider checksum mismatch")
            final = self.output_dir / (item["id"] + "." + digest + ".wav")
            os.replace(staging, final)
            staging = None
            # fsync directory where supported, so publish precedes DB completion.
            if os.name != "nt":
                fd = os.open(self.output_dir, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            provenance = {"provider": item["provider_url"], "job_id": item["job_id"],
                          "tts_item_id": item["id"], "utterance_id": item["utterance_id"],
                          "campaigns_at_import": item.get("campaigns", []),
                          "profile": json.loads(item["profile_json"]), "sha256": digest,
                          "audio_url": downloaded_url, "provider_audio": entry,
                          "provider_metrics": result.get("metrics", {})}
            self.store.import_tts_audio(item["id"], str(final.resolve()), duration, sample_rate,
                                        provenance, lease_until=lease)
        except Exception as exc:
            attempts = item["attempts"] + 1
            failures = item["get_failures"] + 1
            # Explicit validation failures are permanent; transport GETs retry boundedly.
            state = "failed" if (isinstance(exc, ProviderError)
                                or failures >= self.max_get_failures
                                or attempts >= self.max_polls) else "accepted"
            self.store.update_tts_item(item["id"], state, error=f"Collection failed: {type(exc).__name__}: {exc}"[:500],
                                       attempts=attempts, get_failures=failures,
                                       next_attempt_at=self.clock() + min(300, 5 * 2 ** min(failures, 6)),
                                       lease_until=lease)
        finally:
            if staging is not None:
                staging.unlink(missing_ok=True)
        return True
