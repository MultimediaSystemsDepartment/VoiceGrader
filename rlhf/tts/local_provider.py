"""Authenticated v0.7 local jobs and bounded, verified ZIP collection."""
from __future__ import annotations

import hashlib
import io
import json
import re
import urllib.error
from pathlib import PurePosixPath
from urllib.request import Request
import zipfile

from utils.bridge_auth import read_bridge_token
from .provider import CurrentAPIProvider, ProviderError, SubmissionRejected

LOCAL_PROFILES = [
    {"id": "piper-gosia-local-v1", "engine": "piper", "voice": "pl_PL-gosia-medium",
     "voice_used": "pl_PL-gosia-medium", "label": "Local Piper / Gosia", "verified": True},
    *[{"id": f"{engine}-arek-local-v1", "engine": engine, "voice": "Arek",
       "voice_used": "Arek", "label": f"Local {label} / Arek", "verified": False}
      for engine, label in (("xtts", "XTTS"), ("chatterbox", "Chatterbox"), ("omnivoice", "OmniVoice"))],
]


def request_identity(item):
    retries = sum(event.get("action") == "retry" for event in json.loads(item["review_log_json"]))
    return f"{item['id']}-{retries}"


class LocalProvider(CurrentAPIProvider):
    def _headers(self):
        return dict(super()._headers(), Authorization="Bearer " + read_bridge_token())

    def capabilities(self):
        result = self._json("/v1/capabilities")
        if result.get("api_version") != "1" or result.get("languages") != ["pl"]:
            raise ProviderError("Unsupported local TTS capabilities")
        engines = result.get("engines")
        if not isinstance(engines, list) or any(
            not isinstance(engine, dict) or not isinstance(engine.get("id"), str)
            or not isinstance(engine.get("voices"), list)
            or any(not isinstance(voice, dict) or not isinstance(voice.get("id"), str)
                   for voice in engine["voices"])
            for engine in engines
        ):
            raise ProviderError("Malformed local TTS engine/voice capabilities")
        return result

    def submit(self, item):
        profile = json.loads(item["profile_json"])
        if not 1 <= len(item["text"]) <= 280:
            raise SubmissionRejected("Local TTS supports texts of 1-280 characters; text was not truncated")
        payload = {
            "request_id": request_identity(item), "language": "pl", "audio_format": "wav", "archive_format": "zip",
            "items": [{"item_id": item["id"], "text": item["text"], "engine": profile["engine"],
                       "voice_id": profile["voice"]}],
        }
        try:
            result = self._json("/v1/jobs", payload, expected_status=202)
        except urllib.error.HTTPError as exc:
            if exc.code in {409, 422, 429}:
                raise SubmissionRejected(
                    f"Local TTS rejected submission (HTTP {exc.code})", retryable=exc.code == 429,
                ) from exc
            raise
        self._job_path(result.get("job_id"))
        if result.get("status") not in {"queued", "running", "completed", "failed", "completed_with_failures"}:
            raise ProviderError("Invalid local TTS acceptance")
        return result["job_id"]

    def poll(self, item):
        self._job_path(item["job_id"])
        status = self._json("/v1/jobs/" + item["job_id"])
        if status.get("job_id") != item["job_id"] or status.get("request_id") != request_identity(item):
            raise ProviderError("Local TTS job/request identity mismatch")
        if status.get("status") not in {"queued", "running", "completed", "failed", "completed_with_failures"}:
            raise ProviderError("Invalid local TTS job status")
        if status["status"] in {"failed", "completed_with_failures"}:
            return dict(status, status="failed", error=status.get("failure") or "Local TTS engine failed; no audio imported")
        return status

    def result(self, item):
        status = self.poll(item)
        if status["status"] != "completed":
            raise ProviderError("Local TTS archive is not completed")
        url = self.audio_url(status.get("archive_url"))
        with self.opener.open(Request(url, headers=self._headers()), timeout=self.timeout) as response:
            if response.status != 200 or "application/zip" not in response.headers.get("Content-Type", ""):
                raise ProviderError("Local TTS archive must be HTTP 200 ZIP")
            raw = b"".join(self._chunks(response, 32 * 1024 * 1024))
        if hashlib.sha256(raw).hexdigest() != status.get("sha256") or len(raw) != status.get("byte_size"):
            raise ProviderError("Local TTS archive checksum/size mismatch")
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                entries = archive.infolist()
                names = [entry.filename for entry in entries]
                if len(entries) != 2 or len(set(names)) != 2 or "manifest.json" not in names:
                    raise ProviderError("Expected exactly one manifest and one audio file")
                for entry in entries:
                    path = PurePosixPath(entry.filename)
                    if path.is_absolute() or ".." in path.parts or "\\" in entry.filename or ":" in entry.filename or entry.flag_bits & 1:
                        raise ProviderError("Unsafe local TTS archive entry")
                    bound = 65536 if entry.filename == "manifest.json" else self.max_audio_bytes
                    if entry.file_size > bound:
                        raise ProviderError("Local TTS archive entry exceeds size bound")
                manifest = json.loads(archive.read("manifest.json"))
                if not isinstance(manifest, dict) or not isinstance(manifest.get("items"), list):
                    raise ProviderError("Malformed local TTS manifest")
                if (manifest.get("schema_version") != "1" or manifest.get("producer_mode") != "REAL_ONLY"
                        or manifest.get("job_id") != item["job_id"] or manifest.get("request_id") != request_identity(item)
                        or manifest.get("status") != "completed" or len(manifest.get("items", [])) != 1):
                    raise ProviderError("Local TTS manifest identity/schema mismatch")
                entry = manifest["items"][0]
                if not isinstance(entry, dict):
                    raise ProviderError("Malformed local TTS manifest item")
                profile = json.loads(item["profile_json"])
                if (entry.get("item_id") != item["id"] or entry.get("engine") != profile["engine"]
                        or entry.get("voice_id") != profile["voice"] or entry.get("voice_used") != profile["voice_used"]
                        or entry.get("status") != "success" or entry.get("real_tts") is not True
                        or entry.get("demo_mode") is not False or entry.get("error")
                        or entry.get("file") not in names or entry.get("file") == "manifest.json"):
                    raise ProviderError("Local TTS audio/actual voice provenance mismatch")
                metrics = entry.get("generation_metrics")
                if not isinstance(metrics, dict) or metrics.get("voice_used") != profile["voice_used"]:
                    raise ProviderError("Actual engine voice confirmation is missing")
                if profile["engine"] != "piper" and not re.fullmatch("[0-9a-f]{64}", metrics.get("speaker_wav_sha256", "")):
                    raise ProviderError("Reference voice checksum is missing")
                if profile["engine"] == "omnivoice" and metrics.get("ref_text_provided") is not True:
                    raise ProviderError("OmniVoice reference transcript was not confirmed")
                audio = archive.read(entry["file"])
                if len(audio) != entry.get("bytes") or hashlib.sha256(audio).hexdigest() != entry.get("sha256"):
                    raise ProviderError("Local TTS audio checksum/size mismatch")
        except (zipfile.BadZipFile, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ProviderError("Malformed local TTS result archive") from exc
        self._audio = audio
        self._archive_url = url
        return {"metrics": metrics, "manifest": manifest}, dict(
            entry, status="completed", audio_url=url, archive_file=entry["file"],
        )

    def download(self, url, destination):
        if not hasattr(self, "_audio") or url != self._archive_url:
            raise ProviderError("Local TTS audio was not collected and verified")
        with open(destination, "xb") as output:
            output.write(self._audio)
        return url
