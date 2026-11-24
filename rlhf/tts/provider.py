"""Temporary TTSCompare *current UI API* adapter, NOT the v0.7 batch/ZIP API.

CONTRACT-CHANGE POINTS when provider source arrives: PROFILES (legacy Arek.wav
mapping), submit payload/HTTP200 acceptance, status/result envelopes and individual
audio URLs. One local item means exactly one text and one engine/voice profile.
POST has no proven idempotency key: never retry it here. Metadata is provenance,
not a remote deduplication guarantee. No capability probing or voice discovery.
"""
from __future__ import annotations

import json
import os
import re
import time
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

DEFAULT_PROVIDER_URL = "https://ttscompare.rmp.caise.pl"
PROFILES = [
    {"id": "piper-gosia-v1", "engine": "piper", "voice": "default",
     "voice_used": "pl_PL-gosia-medium", "label": "Piper / Gosia", "verified": True},
    *[{"id": f"{engine}-arek-v1", "engine": engine, "voice": "Arek.wav",
       "voice_used": "Arek.wav", "label": f"{label} / Arek (unverified)",
       "verified": False}
      for engine, label in (("xtts", "XTTS"), ("chatterbox", "Chatterbox"), ("omnivoice", "OmniVoice"))],
]


class ProviderError(ValueError):
    pass


class SubmissionRejected(ProviderError):
    """The provider explicitly rejected work before accepting a job."""

    def __init__(self, message, retryable=False):
        super().__init__(message)
        self.retryable = retryable


DEFAULT_LOCAL_URL = "http://tts-api:8080"


def local_provider_url():
    return os.environ.get("TTS_LOCAL_URL", DEFAULT_LOCAL_URL).rstrip("/")


def is_local_provider(url):
    return url.rstrip("/") == local_provider_url()


def profiles_for_url(url):
    if is_local_provider(url):
        from .local_provider import LOCAL_PROFILES
        return LOCAL_PROFILES
    return PROFILES


def provider_for_url(url):
    if is_local_provider(url):
        from .local_provider import LocalProvider
        return LocalProvider(url)
    return CurrentAPIProvider(url)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validate_provider_url(url):
    parsed = urlsplit(url)
    if (parsed.scheme not in {"https", "http"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in {"", "/"}):
        raise ValueError("Provider URL must be an origin without credentials, path or query")
    if (parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            and url.rstrip("/") != DEFAULT_LOCAL_URL):
        raise ValueError("Provider must use HTTPS (localhost HTTP is for fixtures only)")
    return url.rstrip("/")


def validate_profiles(profiles, provider_url=None):
    known = {p["id"]: p for p in (profiles_for_url(provider_url) if provider_url else PROFILES)}
    selected = []
    for profile in profiles:
        profile_id = profile if isinstance(profile, str) else profile["id"]
        if profile_id not in known or (isinstance(profile, dict) and profile != known[profile_id]):
            raise ValueError("Unknown or changed TTS profile")
        selected.append(dict(known[profile_id]))
    if not selected or len({p["id"] for p in selected}) != len(selected):
        raise ValueError("Select distinct TTS profiles")
    return selected


class CurrentAPIProvider:
    def __init__(self, base_url=DEFAULT_PROVIDER_URL, opener=None, timeout=30, max_audio_bytes=20_000_000):
        self.base_url = validate_provider_url(base_url)
        self.opener = opener or build_opener(_NoRedirect())
        self.timeout = timeout
        self.max_audio_bytes = max_audio_bytes

    def _chunks(self, response, max_bytes):
        deadline = time.monotonic() + self.timeout
        size = 0
        read = getattr(response, "read1", response.read)
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError("Provider response exceeded its wall-time bound")
            chunk = read(min(65536, max_bytes - size + 1))
            if not chunk:
                return
            size += len(chunk)
            if size > max_bytes:
                raise ProviderError("Provider response exceeded size bound")
            yield chunk

    def _headers(self):
        return {"Accept": "application/json", "Content-Type": "application/json"}

    def _json(self, path, payload=None, expected_status=200):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(self.base_url + path, data=data,
                          headers=self._headers(),
                          method="GET" if payload is None else "POST")
        with self.opener.open(request, timeout=self.timeout) as response:
            if response.status != expected_status or "application/json" not in response.headers.get("Content-Type", ""):
                raise ProviderError(f"Expected HTTP {expected_status} JSON from provider API")
            raw = b"".join(self._chunks(response, 2_000_000))
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise ProviderError("Provider response is not an object")
            return result

    @staticmethod
    def _job_path(job_id):
        if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", job_id):
            raise ProviderError("Malformed provider job identity")
        return "/api/v1/tts/jobs/" + job_id

    def submit(self, item):
        profile = json.loads(item["profile_json"])
        result = self._json("/api/v1/tts/jobs", {
            "text": item["text"], "language": "pl", "voice": profile["voice"],
            "engines": [profile["engine"]],
            "metadata": {"voicegrader_item_id": item["id"], "utterance_id": item["utterance_id"],
                         "profile_id": item["profile_id"],
                         "campaign_ids": [c["id"] for c in item.get("campaigns", [])]},
        })
        self._job_path(result.get("job_id"))
        if not isinstance(result.get("status"), str) or not isinstance(result.get("stage"), str):
            raise ProviderError("Acceptance envelope changed")
        return result["job_id"]

    def poll(self, item):
        result = self._json(self._job_path(item["job_id"]))
        if result.get("job_id", item["job_id"]) != item["job_id"]:
            raise ProviderError("Status job identity mismatch")
        if not isinstance(result.get("status"), str):
            raise ProviderError("Missing job status")
        if result["status"] == "completed":
            profile = json.loads(item["profile_json"])
            if result.get("engines") != [profile["engine"]]:
                raise ProviderError("Status engines do not match the single requested profile")
        return result

    def result(self, item):
        result = self._json(self._job_path(item["job_id"]) + "/result")
        profile = json.loads(item["profile_json"])
        if result.get("job_id") != item["job_id"] or result.get("status") != "completed":
            raise ProviderError("Result job identity/status mismatch")
        if result.get("errors"):
            raise ProviderError("Provider result includes explicit errors")
        audio = result.get("audio", {})
        if not isinstance(audio, dict) or set(audio) != {profile["engine"]}:
            raise ProviderError("Result engines mismatch")
        entry = audio[profile["engine"]]
        if (not isinstance(entry, dict) or entry.get("engine") != profile["engine"]
                or entry.get("status") not in {"completed", "success"}
                or entry.get("voice_used") != profile["voice_used"]
                or entry.get("real_tts") is not True or entry.get("demo_mode") is not False):
            raise ProviderError("Engine/voice/real-TTS result contract mismatch")
        return result, entry

    def audio_url(self, url):
        if not isinstance(url, str) or not url:
            raise ProviderError("Missing audio URL")
        target = urljoin(self.base_url + "/", url)
        base, parsed = urlsplit(self.base_url), urlsplit(target)
        if (parsed.scheme, parsed.hostname, parsed.port) != (base.scheme, base.hostname, base.port):
            raise ProviderError("Audio URL is not on the approved provider origin")
        if parsed.username or parsed.password or parsed.fragment:
            raise ProviderError("Unsafe audio URL")
        return target

    def download(self, url, destination):
        target = self.audio_url(url)
        request = Request(target, headers={"Accept": "audio/*"})
        with self.opener.open(request, timeout=self.timeout) as response:
            if response.status != 200:
                raise ProviderError("Audio download did not return HTTP 200")
            length = response.headers.get("Content-Length")
            if length is not None and (int(length) < 1 or int(length) > self.max_audio_bytes):
                raise ProviderError("Audio size outside download bound")
            size = 0
            with open(destination, "xb") as stream:
                for chunk in self._chunks(response, self.max_audio_bytes):
                    size += len(chunk)
                    stream.write(chunk)
                stream.flush()
                import os
                os.fsync(stream.fileno())
            if not size or (length is not None and size != int(length)):
                raise ProviderError("Empty or truncated audio download")
        return target
