import ast
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import time
import types
import wave
import zipfile
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.responses import Response
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1] / "tts"


@pytest.fixture
def service(monkeypatch, tmp_path):
    root = ROOT / "orchestration" / "rest"
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.setenv("RLHF_POC_DATA", str(tmp_path / "state"))
    monkeypatch.setenv("RLHF_PIPER_URL", "http://fixture.invalid")
    for name in ("XTTS", "CHATTERBOX", "OMNIVOICE"):
        monkeypatch.delenv("RLHF_" + name + "_URL", raising=False)
    token = tmp_path / "token"
    token.write_text("a" * 64)
    monkeypatch.setenv("VOICEGRADER_BRIDGE_TOKEN_FILE", str(token))
    spec = importlib.util.spec_from_file_location("tts_service_test", root / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.POOL.shutdown()
    module.POOL = types.SimpleNamespace(submit=lambda *args: None)
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setparams((1, 2, 22050, 0, "NONE", "not compressed"))
        wav.writeframes(b"\x01\x00" * 2205)
    module.generate_real_piper = lambda *args: (
        stream.getvalue(), {"voice_used": "pl_PL-gosia-medium"},
    )
    with TestClient(module.app) as client:
        client.headers["Authorization"] = "Bearer " + "a" * 64
        yield module, client


def payload():
    return {"request_id": "fixture-request", "language": "pl", "audio_format": "wav", "archive_format": "zip",
            "items": [{"item_id": "fixture-item", "text": "To jest test.", "engine": "piper", "voice_id": "pl_PL-gosia-medium"}]}


def test_local_service_auth_idempotency_limits_and_verified_archive(service):
    module, client = service
    assert client.get("/v1/jobs/unknown", headers={"Authorization": "wrong"}).status_code == 401
    response = client.post("/v1/jobs", json=payload())
    assert response.status_code == 202
    jid = response.json()["job_id"]
    assert client.post("/v1/jobs", json=payload()).json()["job_id"] == jid
    changed = payload()
    changed["items"][0]["text"] = "Different text."
    assert client.post("/v1/jobs", json=changed).status_code == 409
    busy = payload()
    busy["request_id"] = "different-request"
    assert client.post("/v1/jobs", json=busy).status_code == 429
    long = payload()
    long["items"][0]["text"] = "a" * 281
    assert client.post("/v1/jobs", json=long).status_code == 422
    module.process(jid)
    status = client.get("/v1/jobs/" + jid).json()
    assert status["status"] == "completed"
    raw = client.get(status["archive_url"]).content
    assert hashlib.sha256(raw).hexdigest() == status["sha256"]
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        record = json.loads(archive.read("manifest.json"))["items"][0]
        assert record["voice_used"] == "pl_PL-gosia-medium"
        assert record["real_tts"] and record["demo_mode"] is False
        assert hashlib.sha256(archive.read(record["file"])).hexdigest() == record["sha256"]
    # Connections must be closed, including on Windows.
    module.DB.rename(module.DB.with_suffix(".closed"))


def test_local_service_refuses_unconfirmed_actual_voice(service):
    module, client = service
    module.generate_real_piper = lambda *args: (b"audio", {"voice_used": "another-voice"})
    jid = client.post("/v1/jobs", json=payload()).json()["job_id"]
    module.process(jid)
    status = client.get("/v1/jobs/" + jid).json()
    assert status["status"] == "failed"
    assert status["archive_url"] is None
    assert "actual requested voice" in status["failure"]


def test_omnivoice_direct_path_uses_required_matching_transcript(tmp_path):
    import asyncio

    source = ast.parse((ROOT / "engines" / "omnivoice" / "app.py").read_text(encoding="utf-8"))
    function = next(node for node in source.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "generate_direct")
    function.decorator_list = []
    voice = tmp_path / "Arek.wav"
    voice.write_bytes(b"reference fixture")
    transcript = tmp_path / "Arek.txt"
    seen = []

    def synthesize(**args):
        seen.append(args["ref_text"])
        args["wav_path"].write_bytes(b"fixture audio")
        return 0.1, {"duration_sec": 0.2, "sample_rate": 24000}

    namespace = {
        "DirectGenerateRequest": object, "Response": Response, "HTTPException": HTTPException,
        "time": time, "tempfile": tempfile, "Path": Path,
        "OMNIVOICE_SPEAKER_WAV": str(voice), "OMNIVOICE_REF_TEXT": None,
        "OMNIVOICE_REF_TEXT_FILE": str(transcript), "OMNIVOICE_MODEL_ID": "fixture",
        "OMNIVOICE_SAMPLE_RATE": 24000, "synthesize_to_file": synthesize,
        "sha256_file": lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<original-direct-handler>", "exec"), namespace)
    request = types.SimpleNamespace(ref_audio=None, ref_text=None, text="Test.", language_id="pl", instruct=None, num_step=16, speed=1)
    with pytest.raises(HTTPException) as unavailable:
        asyncio.run(namespace["generate_direct"](request))
    assert unavailable.value.status_code == 503
    transcript.write_text("Matching reference words.", encoding="utf-8")
    response = asyncio.run(namespace["generate_direct"](request))
    assert seen == ["Matching reference words."]
    assert response.headers["X-Ref-Text-Provided"] == "true"
    assert response.headers["X-Speaker-Wav-SHA256"] == hashlib.sha256(voice.read_bytes()).hexdigest()


@pytest.mark.parametrize("accepted,marker", [("0", True), ("1", False)])
def test_xtts_requires_explicit_terms_before_loading_models(tmp_path, accepted, marker):
    source = ast.parse((ROOT / "engines" / "xtts" / "app.py").read_text(encoding="utf-8"))
    function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "ensure_model")
    if marker:
        (tmp_path / "tos_agreed.txt").write_text("isolated fixture")
    namespace = {"model": None, "ROOT": tmp_path, "os": types.SimpleNamespace(environ={"COQUI_TOS_AGREED": accepted})}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<original-xtts-loader>", "exec"), namespace)
    with pytest.raises(RuntimeError, match="Explicit XTTS model-terms acceptance"):
        namespace["ensure_model"]()
