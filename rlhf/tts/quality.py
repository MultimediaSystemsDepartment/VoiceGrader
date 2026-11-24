"""On-demand local ASR quality assessment; scores never change learning eligibility."""

from __future__ import annotations

import io
import json
import logging
import math
import os
import urllib.request
import urllib.parse
from uuid import uuid4

import soundfile as sf

from utils.audio import load_audio
from utils.bridge_auth import read_bridge_token

log = logging.getLogger(__name__)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Quality service redirects are not permitted")


def assess_audio(file_path: str) -> dict:
    base = os.environ.get("VOICEGRADER_ASR_URL", "").rstrip("/")
    url = urllib.parse.urlsplit(base)
    if (url.scheme not in {"http", "https"} or not url.netloc or url.username
            or url.password or url.query or url.fragment):
        raise ValueError("Configure a valid local VOICEGRADER_ASR_URL")
    waveform, sample_rate = load_audio(file_path)
    output = io.BytesIO()
    sf.write(output, waveform, sample_rate, format="WAV", subtype="PCM_16")
    boundary = "quality-" + uuid4().hex
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + output.getvalue() + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        base + "/api/quality", data=body, method="POST",
        headers={"Authorization": "Bearer " + read_bridge_token(),
                 "Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=150) as response:
        raw = response.read(65537)
        if len(raw) > 65536 or response.headers.get_content_type() != "application/json":
            raise ValueError("Invalid quality response size/type")
        result = json.loads(raw)
    if (not isinstance(result, dict) or result.get("schema_version") != 1
            or result.get("report_only") is not True
            or result.get("profile") != "asr-whisqa-multi-silero-snr-v1"):
        raise ValueError("Unsupported quality response")
    metrics = result.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != {"snr", "qual"}:
        raise ValueError("Quality metrics are missing")
    for group in metrics.values():
        if not isinstance(group, dict) or not group:
            raise ValueError("Invalid quality metric group")
        for value in group.values():
            if value is not None and (type(value) not in {int, float} or not math.isfinite(value)):
                raise ValueError("Nonfinite or nonnumeric quality score")
    if metrics["qual"].get("mos") is None:
        raise ValueError("WhiSQA MOS is missing")
    return result


def run_quality_once(store, assess=assess_audio) -> bool:
    review = store.claim_audio_quality()
    if review is None:
        return False
    try:
        result = assess(review["file_path"])
    except (OSError, ValueError, RuntimeError) as exc:
        log.exception("Quality assessment failed; recording and learning settings unchanged")
        store.finish_audio_quality(review["id"], error=f"{type(exc).__name__}: {exc}"[:1000])
    else:
        store.finish_audio_quality(review["id"], profile=result["profile"], metrics=result["metrics"])
    return True
