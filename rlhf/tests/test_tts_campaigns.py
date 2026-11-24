from __future__ import annotations

import io
import json
import time
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import soundfile as sf

from storage.feedback_store import FeedbackStore
from storage.schema import Annotation, AnnotationSession, AudioFile, CanarySet, Corpus, HypothesisPair
from tts.provider import CurrentAPIProvider, DEFAULT_PROVIDER_URL, PROFILES, ProviderError
from tts.worker import TTSWorker, validate_audio
from utils.origin_sampling import select_origin_capped


def source(store, n=4):
    corpus = store.register_corpus(Corpus.new("source"))
    for i in range(n):
        store.insert_audio_file(AudioFile.new(f"recording{i}.wav", corpus["name"], f"Zdanie {i}."))
    return corpus


def campaign(store, corpus, **kwargs):
    return store.create_tts_campaign(corpus["id"], [PROFILES[0]], provider_url=DEFAULT_PROVIDER_URL, **kwargs)


def wav_bytes(duration=0.2):
    stream = io.BytesIO()
    sf.write(stream, np.sin(np.arange(int(duration * 16000)) * 0.05).astype(np.float32),
             16000, format="WAV")
    return stream.getvalue()


def test_quality_review_is_persisted_report_only_and_failure_requires_retry(store):
    from tts.quality import run_quality_once

    audio = AudioFile.new("synth.wav", "synth", "Test words.")
    audio.origin = "synthesized"
    store.insert_audio_file(audio)
    review_id = store.request_audio_quality(audio.id)
    with pytest.raises(ValueError, match="already queued"):
        store.request_audio_quality(audio.id)
    before = store.get_audio_file(audio.id)
    assert run_quality_once(store, assess=lambda path: {
        "profile": "asr-whisqa-multi-silero-snr-v1",
        "metrics": {"snr": {"snr": None}, "qual": {"mos": 1.2}},
    })
    result = store.get_audio_quality([audio.id])[audio.id]
    assert result["state"] == "completed"
    assert store.get_audio_file(audio.id) == before
    assert store.get_setting("tts_sft_share_percent") == "0"
    store.review_audio_quality(review_id, "Listened; low score, retained for review.")
    assert store.get_audio_quality([audio.id])[audio.id]["reviewer"] == "admin"
    retry_id = store.request_audio_quality(audio.id)

    def offline(path):
        raise OSError("ASR offline")

    assert run_quality_once(store, assess=offline)
    failed = store.get_audio_quality([audio.id])[audio.id]
    assert failed["id"] == retry_id and failed["state"] == "failed"
    assert "ASR offline" in failed["error"]
    assert run_quality_once(store, assess=offline) is False
    assert store.get_audio_file(audio.id) == before


def test_quality_request_rejects_recorded_audio_and_recovers_interrupted_work(store):
    recorded = AudioFile.new("recorded.wav", "source", "Words.")
    store.insert_audio_file(recorded)
    with pytest.raises(ValueError, match="synthesized"):
        store.request_audio_quality(recorded.id)
    audio = AudioFile.new("tts.wav", "synth", "Words.")
    audio.origin = "synthesized"
    store.insert_audio_file(audio)
    store.request_audio_quality(audio.id)
    assert store.claim_audio_quality()["audio_id"] == audio.id
    store.fail_interrupted_audio_quality()
    assert store.get_audio_quality([audio.id])[audio.id]["state"] == "failed"
    assert store.claim_audio_quality() is None


@pytest.mark.parametrize("bad_response", ["nan", "wrong_profile", "missing_mos"])
def test_quality_http_client_rejects_invalid_scores(monkeypatch, tmp_path, bad_response):
    import tts.quality as quality

    path = tmp_path / "clip.wav"
    path.write_bytes(wav_bytes(0.5))
    token = tmp_path / "credential"
    token.write_text("a" * 64)
    monkeypatch.setenv("VOICEGRADER_BRIDGE_TOKEN_FILE", str(token))
    monkeypatch.setenv("VOICEGRADER_ASR_URL", "http://asr:8000")
    result = {"schema_version": 1, "report_only": True,
              "profile": "asr-whisqa-multi-silero-snr-v1",
              "metrics": {"snr": {"snr": None}, "qual": {"mos": 4.1}}}
    if bad_response == "nan":
        result["metrics"]["qual"]["mos"] = float("nan")
    elif bad_response == "wrong_profile":
        result["profile"] = "unknown"
    else:
        result["metrics"]["qual"] = {"other": 1.0}

    class QualityResponse(io.BytesIO):
        headers = type("Headers", (), {"get_content_type": lambda self: "application/json"})()

    class Opener:
        def open(self, request, timeout):
            assert request.full_url == "http://asr:8000/api/quality"
            assert request.get_header("Authorization") == "Bearer " + "a" * 64
            assert b"audio/wav" in request.data
            return QualityResponse(json.dumps(result).encode())

    monkeypatch.setattr(quality.urllib.request, "build_opener", lambda *args: Opener())
    with pytest.raises(ValueError):
        quality.assess_audio(str(path))


class Response(io.BytesIO):
    def __init__(self, value, content_type="application/json"):
        raw = json.dumps(value).encode() if isinstance(value, dict) else value
        super().__init__(raw)
        self.status = 200
        self.headers = {"Content-Type": content_type, "Content-Length": str(len(raw))}


class FakeHTTP:
    def __init__(self):
        self.requests = []
        self.fail_submit = False
        self.fail_get = False
        self.audio = wav_bytes()
        self.voice = "pl_PL-gosia-medium"
        self.demo = False
        self.job = "job-1"

    def open(self, request, timeout):
        self.requests.append(request)
        if request.get_method() == "POST":
            if self.fail_submit:
                raise TimeoutError("accepted remotely, response lost")
            return Response({"job_id": self.job, "status": "queued", "stage": "pending"})
        if self.fail_get:
            raise TimeoutError("GET timed out")
        if request.full_url.endswith("/result"):
            return Response({"job_id": self.job, "status": "completed",
                "audio": {"piper": {"engine": "piper", "status": "completed",
                    "audio_url": "/audio.wav", "voice_used": self.voice,
                    "real_tts": True, "demo_mode": self.demo}}, "metrics": {}, "errors": []})
        if request.full_url.endswith("/audio.wav"):
            return Response(self.audio, "audio/wav")
        return Response({"job_id": self.job, "status": "completed", "engines": ["piper"],
                         "tts_results": {}, "errors": []})


def worker(store, tmp_path, http, now):
    return TTSWorker(store, tmp_path / "audio",
                     provider_factory=lambda url: CurrentAPIProvider(url, opener=http),
                     clock=lambda: now[0])


def test_frozen_campaign_coverage_only_fills_missing_combinations(store):
    corpus = source(store)
    first = campaign(store, corpus, mode="coverage", value=50)
    assert first["missing"] == 2
    initial = {r["id"] for r in store.get_tts_campaign_items(first["id"])}
    repeated = campaign(store, corpus, mode="coverage", value=50)
    assert repeated["missing"] == 0
    increased = campaign(store, corpus, mode="coverage", value=100)
    assert increased["missing"] == 2
    assert initial <= {r["id"] for r in store.get_tts_campaign_items(increased["id"])}
    count = campaign(store, corpus, mode="count", value=2)
    assert count["selected_utterances"] == 0
    store.insert_audio_file(AudioFile.new("later.wav", corpus["name"], "Dodane później."))
    assert len(store.get_tts_campaign_items(first["id"])) == 2
    assert campaign(store, corpus, mode="count", value=1)["missing"] == 1


def test_current_api_restart_import_once_and_human_feedback_independent_of_refill_cap(store, tmp_path):
    corpus = source(store, 1)
    created = campaign(store, corpus)
    http = FakeHTTP()
    now = [1000]
    assert worker(store, tmp_path, http, now).run_once()
    item = store.get_tts_campaign_items(created["id"])[0]
    assert item["state"] == "accepted" and item["job_id"] == "job-1"
    request = json.loads(http.requests[0].data)
    assert request["language"] == "pl" and request["voice"] == "default"
    assert request["engines"] == ["piper"] and request["metadata"]["voicegrader_item_id"] == item["id"]
    restarted = FeedbackStore(store.db_path)
    assert worker(restarted, tmp_path, http, now).run_once()
    imported = restarted.get_tts_campaign_items(created["id"])[0]
    assert imported["state"] == "completed"
    assert not worker(restarted, tmp_path, http, now).run_once()
    assert restarted.count_audio_files("TTSCompare") == 1
    assert len([r for r in http.requests if r.get_method() == "POST"]) == 1
    audio = restarted.get_audio_file(imported["audio_id"])
    assert audio.origin == "synthesized"
    assert json.loads(audio.provenance_json)["job_id"] == item["job_id"]
    assert {a.origin for a in restarted.get_unqueued_audio_files(10)} == {"recorded"}
    assert restarted.get_sft_training_rows() == []
    session = AnnotationSession.new("person", "machine")
    restarted.open_session(session)
    pair = HypothesisPair.new(audio.id, "pierwsza", "druga", "beam", "sample", "policy")
    restarted.insert_hypothesis_pair(pair)
    restarted.insert_annotation(Annotation.new(session.id, pair.id, "a"))
    assert restarted.get_preference_dataset("policy")[0]["origin"] == "synthesized"
    assert restarted.get_sft_corrections("policy") == []
    restarted.set_tts_shares(100, 100)
    assert restarted.get_sft_training_rows()[0]["correction_text"] == "Zdanie 0."


def test_web_lifespan_processes_campaign_without_gpu_coordinator(monkeypatch, store, tmp_path):
    import web.app as app_module
    import tts.worker as worker_module
    from fastapi.testclient import TestClient

    created = campaign(store, source(store, 1))
    http = FakeHTTP()
    consumer = worker(store, tmp_path, http, [1000])

    async def idle():
        pass

    monkeypatch.setattr(app_module, "_get_store", lambda: store)
    monkeypatch.setattr(app_module, "ensure_bridge_token", lambda: None)
    monkeypatch.setattr(app_module, "_load_password_config", lambda: {})
    monkeypatch.setattr(app_module, "_reservation_cleanup_loop", idle)
    monkeypatch.setattr(app_module, "_queue_refill_loop", idle)
    monkeypatch.setattr(app_module, "_INLINE_COORDINATOR", False)
    monkeypatch.setattr(app_module, "DEFAULT_PATHS", types.SimpleNamespace(
        logs_dir=tmp_path / "logs", output_dir=tmp_path,
    ))
    monkeypatch.setattr(worker_module, "TTSWorker", lambda *args: consumer)
    with TestClient(app_module.app) as client:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            item = store.get_tts_campaign_items(created["id"])[0]
            if item["state"] == "completed":
                break
            time.sleep(0.05)
        else:
            pytest.fail(f"Web-only campaign did not complete: {item['state']}")
        assert client.get("/api/stats").status_code == 200
    assert store.get_audio_file(item["audio_id"]).origin == "synthesized"
    assert len([r for r in http.requests if r.get_method() == "POST"]) == 1
    assert store.get_setting("tts_sft_share_percent") == "0"
    assert store.get_setting("tts_refill_share_percent") == "0"


def test_crash_after_durable_download_before_import_recovers_without_duplicate(store, tmp_path, monkeypatch):
    created = campaign(store, source(store, 1))
    http = FakeHTTP()
    now = [1000]
    runner = worker(store, tmp_path, http, now)
    runner.run_once()
    original_import = store.import_tts_audio

    def crash(*args, **kwargs):
        raise SystemExit("process interrupted before DB commit")

    monkeypatch.setattr(store, "import_tts_audio", crash)
    with pytest.raises(SystemExit):
        runner.run_once()
    assert store.count_audio_files("TTSCompare") == 0
    assert len(list((tmp_path / "audio").glob("*.wav"))) == 1
    monkeypatch.setattr(store, "import_tts_audio", original_import)
    now[0] += 400
    assert worker(FeedbackStore(store.db_path), tmp_path, http, now).run_once()
    assert store.get_tts_campaign_items(created["id"])[0]["state"] == "completed"
    assert store.count_audio_files("TTSCompare") == 1
    assert len(list((tmp_path / "audio").glob("*.wav"))) == 1


def test_uncertain_submission_never_automatically_retries_and_operator_can_attach_job(store, tmp_path):
    created = campaign(store, source(store, 1))
    http = FakeHTTP()
    http.fail_submit = True
    now = [1000]
    runner = worker(store, tmp_path, http, now)
    assert runner.run_once()
    item = store.get_tts_campaign_items(created["id"])[0]
    assert item["state"] == "review_required"
    assert not runner.run_once()
    with pytest.raises(ValueError, match="acknowledgement"):
        store.review_tts_item(item["id"], "retry")
    http.fail_submit = False
    store.review_tts_item(item["id"], "attach_job", job_id="job-1")
    assert worker(FeedbackStore(store.db_path), tmp_path, http, now).run_once()
    assert len([r for r in http.requests if r.get_method() == "POST"]) == 1
    assert len(json.loads(store.get_tts_campaign_items(created["id"])[0]["review_log_json"])) == 1


def test_interrupted_post_lease_requires_review_but_accepted_poll_lease_recovers(store):
    created = campaign(store, source(store, 2))
    claimed = store.claim_tts_item(now=10, lease_seconds=1)
    # An abrupt process exit after starting POST must not submit this item again.
    following = FeedbackStore(store.db_path).claim_tts_item(now=12, lease_seconds=1)
    assert following["id"] != claimed["id"]
    first = next(i for i in store.get_tts_campaign_items(created["id"]) if i["id"] == claimed["id"])
    assert first["state"] == "review_required"
    store.update_tts_item(following["id"], "accepted", job_id="known")
    poll = store.claim_tts_item(now=12, lease_seconds=1)
    resumed = store.claim_tts_item(now=14, lease_seconds=1)
    assert resumed["id"] == poll["id"] and resumed["job_id"] == "known"


def test_provider_job_identity_cannot_be_attached_to_two_different_items(store, tmp_path):
    created = campaign(store, source(store, 2))
    runner = worker(store, tmp_path, FakeHTTP(), [1000])
    runner.run_once()
    runner.run_once()
    # The fixture intentionally returns the same ID for the next text.
    runner.run_once()
    items = store.get_tts_campaign_items(created["id"])
    uncertain = next(i for i in items if i["state"] == "review_required")
    assert store.count_audio_files("TTSCompare") == 1
    with pytest.raises(ValueError, match="different"):
        store.review_tts_item(uncertain["id"], "attach_job", job_id="job-1")


def test_bounded_get_retries_explicit_failure_and_local_cancel(store, tmp_path):
    created = campaign(store, source(store, 1))
    http = FakeHTTP()
    now = [1000]
    runner = worker(store, tmp_path, http, now)
    runner.run_once()
    store.cancel_tts_campaign(created["id"])
    assert store.get_tts_campaign_items(created["id"])[0]["state"] == "accepted"
    http.fail_get = True
    for _ in range(runner.max_get_failures):
        assert runner.run_once()
        now[0] += 400
    assert store.get_tts_campaign_items(created["id"])[0]["state"] == "failed"
    assert not runner.run_once()
    assert len([r for r in http.requests if r.get_method() == "POST"]) == 1


@pytest.mark.parametrize("change", ["voice", "demo", "silence"])
def test_invalid_result_never_imports(store, tmp_path, change):
    created = campaign(store, source(store, 1))
    http = FakeHTTP()
    if change == "voice":
        http.voice = "wrong-voice"
    elif change == "demo":
        http.demo = True
    else:
        stream = io.BytesIO()
        sf.write(stream, np.zeros(3200), 16000, format="WAV")
        http.audio = stream.getvalue()
    runner = worker(store, tmp_path, http, [1000])
    runner.run_once()
    runner.run_once()
    assert store.get_tts_campaign_items(created["id"])[0]["state"] == "failed"
    assert store.count_audio_files("TTSCompare") == 0
    assert not list((tmp_path / "audio").glob("*.part")) if (tmp_path / "audio").exists() else True


def test_origin_bounds_and_native_duration(tmp_path):
    provider = CurrentAPIProvider(opener=FakeHTTP(), max_audio_bytes=10)
    with pytest.raises(ProviderError, match="origin"):
        provider.download("https://other.invalid/audio.wav", tmp_path / "wrong.wav")
    with pytest.raises(ProviderError, match="origin"):
        provider.audio_url("//other.invalid/audio.wav")
    with pytest.raises(ProviderError, match="bound"):
        provider.download("/audio.wav", tmp_path / "large.wav")
    path = tmp_path / "long.wav"
    path.write_bytes(wav_bytes(30.1))
    with pytest.raises(ProviderError, match="Native"):
        validate_audio(path)


def test_campaign_sources_and_all_renditions_exclude_monitor_text(store, tmp_path):
    corpus = source(store, 2)
    original = store.get_synthesis_source_utterances(corpus["id"])[0]
    held = AudioFile.new("held.wav", "holdout", original["text"])
    store.insert_audio_file(held)
    created = campaign(store, corpus)
    canary = CanarySet.new("monitor", target_size=1, selection_seed=1, selection_method="random")
    store.create_canary_set(canary, [held.id])
    assert len(store.get_synthesis_source_utterances(corpus["id"])) == 1
    http = FakeHTTP()
    runner = worker(store, tmp_path, http, [1000])
    runner.run_once()
    runner.run_once()
    states = {i["utterance_id"]: i["state"] for i in store.get_tts_campaign_items(created["id"])}
    assert states[original["id"]] == "cancelled"
    assert store.count_audio_files("TTSCompare") == 1
    assert campaign(store, corpus)["selected_utterances"] == 1


def test_caps_no_missing_recorded_backfill_and_one_rendition_per_epoch():
    recorded = [{"origin": "recorded", "audio_id": str(i)} for i in range(6)]
    tts = [{"origin": "synthesized", "utterance_id": str(i // 2), "audio_id": f"t{i}"} for i in range(20)]
    assert select_origin_capped(tts, 50, limit=10) == []
    assert {r["audio_id"] for r in select_origin_capped(recorded + tts, 0)} == {r["audio_id"] for r in recorded}
    selected = select_origin_capped(recorded + tts, 20, limit=10)
    assert sum(r["origin"] == "synthesized" for r in selected) <= len(selected) * 0.2
    selected = select_origin_capped(recorded + tts, 100, one_tts_per_utterance=True)
    synthetic = [r for r in selected if r["origin"] == "synthesized"]
    assert len({r["utterance_id"] for r in synthetic}) == len(synthetic)
    with pytest.raises(ValueError):
        select_origin_capped(tts, float("nan"))


def test_explicit_refill_schedule_cannot_bypass_zero_tts_cap(store, tmp_path):
    corpus = source(store, 1)
    created = campaign(store, corpus)
    runner = worker(store, tmp_path, FakeHTTP(), [1000])
    runner.run_once()
    runner.run_once()
    item = store.get_tts_campaign_items(created["id"])[0]
    with pytest.raises(ValueError, match="share cap"):
        store.get_eligible_audio_files_by_ids([item["audio_id"]], "policy")
    store.set_tts_shares(0, 100)
    assert store.get_eligible_audio_files_by_ids([item["audio_id"]], "policy")[0].origin == "synthesized"


def test_use_flags_and_concurrent_migration(store, tmp_path):
    corpus = source(store, 1)
    campaign(store, corpus, intended_use="sft")
    runner = worker(store, tmp_path, FakeHTTP(), [1000])
    runner.run_once()
    runner.run_once()
    store.set_tts_shares(100, 100)
    assert all(a.origin == "recorded" for a in store.get_unqueued_audio_files(10))
    assert len(store.get_sft_training_rows()) == 1
    campaign(store, corpus, intended_use="annotation")
    assert {a.origin for a in store.get_unqueued_audio_files(10)} == {"recorded", "synthesized"}
    store.set_corpus_active(corpus["name"], False)
    assert store.get_sft_training_rows() == []
    assert store.get_unqueued_audio_files(10) == []
    with ThreadPoolExecutor(max_workers=2) as executor:
        assert len(list(executor.map(FeedbackStore, [store.db_path, store.db_path]))) == 2


def test_campaign_ui_is_admin_only_and_only_persists_requests(monkeypatch, store):
    import web.app as app_module
    from fastapi.testclient import TestClient

    monkeypatch.setattr(app_module, "_get_store", lambda: store)
    monkeypatch.setattr(app_module, "_ADMIN_CFG_PATH", str(store.db_path.parent / "password.json"))
    client = TestClient(app_module.app)
    assert client.get("/admin/tts", follow_redirects=False).status_code == 307
    for route in ("create", "shares", "review", "cancel", "quality"):
        assert client.post("/admin/tts/" + route, data={}).status_code == 403
    client.post("/admin/login", data={"password": "change-me"})
    corpus = source(store, 1)
    form = {"corpus_id": corpus["id"], "profiles": PROFILES[0]["id"], "mode": "all",
            "value": "100", "intended_use": "both"}
    assert client.get("/admin/tts").status_code == 200
    response = client.post("/admin/tts/create", data=dict(form, action="preview"))
    assert response.json()["missing"] == 1 and store.get_tts_campaigns() == []
    assert client.post("/admin/tts/create", data=form, follow_redirects=False).status_code == 303
    assert store.get_tts_campaign_items(store.get_tts_campaigns()[0]["id"])[0]["state"] == "pending"
    assert client.post("/admin/tts/shares", data={"sft_share": "101", "refill_share": "10"}).status_code == 400
    assert store.get_setting("tts_sft_share_percent", "0") == "0"
    assert client.post("/admin/tts/create", data=dict(form, value="nan")).status_code == 400
    assert client.get("/admin/tts?campaign_id=missing").status_code == 404


def test_quality_ui_requests_assessment_and_review_without_changing_learning(monkeypatch, store, tmp_path):
    import web.app as app_module
    from fastapi.testclient import TestClient
    from tts.quality import run_quality_once

    monkeypatch.setattr(app_module, "_get_store", lambda: store)
    monkeypatch.setattr(app_module, "_ADMIN_CFG_PATH", str(tmp_path / "password.json"))
    corpus = source(store, 1)
    created = campaign(store, corpus)
    http, now = FakeHTTP(), [0]
    consumer = worker(store, tmp_path, http, now)
    assert consumer.run_once() and consumer.run_once()
    item = store.get_tts_campaign_items(created["id"])[0]
    assert item["audio_id"]
    client = TestClient(app_module.app)
    client.post("/admin/login", data={"password": "change-me"})
    assert "Assess quality" in client.get("/admin/tts").text
    form = {"action": "assess", "audio_id": item["audio_id"], "campaign_id": created["id"]}
    assert client.post("/admin/tts/quality", data=form).status_code == 200
    assert client.post("/admin/tts/quality", data=form).status_code == 400
    assert run_quality_once(store, assess=lambda path: {
        "profile": "asr-whisqa-multi-silero-snr-v1",
        "metrics": {"snr": {"snr": None, "perc_speech": 0.0}, "qual": {"mos": 4.1}},
    })
    page = client.get("/admin/tts")
    assert "4.10" in page.text and "Mark reviewed" in page.text
    assert "Unavailable" in page.text and "0.0 %" in page.text
    assert "tts-score-grid" in page.text and "<details open>" not in page.text
    review = store.get_audio_quality([item["audio_id"]])[item["audio_id"]]
    assert client.post("/admin/tts/quality", data={
        "action": "review", "review_id": review["id"], "campaign_id": created["id"], "notes": "Listened.",
    }).status_code == 200
    assert store.get_setting("tts_sft_share_percent") == "0"
    assert store.get_audio_quality([item["audio_id"]])[item["audio_id"]]["notes"] == "Listened."


def test_review_controls_retry_collection_without_resubmitting_and_preserve_campaign(monkeypatch, store, tmp_path):
    from html.parser import HTMLParser
    import web.app as app_module
    from fastapi.testclient import TestClient

    class ReviewForms(HTMLParser):
        def __init__(self):
            super().__init__()
            self.forms = []
            self.current = None

        def handle_starttag(self, tag, attributes):
            attrs = dict(attributes)
            if tag == "form" and attrs.get("action") == "/admin/tts/review":
                self.current = {}
            elif self.current is not None and tag in {"input", "button"} and attrs.get("name"):
                assert attrs.get("type") != "checkbox"
                self.current[attrs["name"]] = attrs.get("value", "")

        def handle_endtag(self, tag):
            if tag == "form" and self.current is not None:
                self.forms.append(self.current)
                self.current = None

    monkeypatch.setattr(app_module, "_get_store", lambda: store)
    monkeypatch.setattr(app_module, "_ADMIN_CFG_PATH", str(tmp_path / "password.json"))
    created = campaign(store, source(store, 1))
    http = FakeHTTP()
    consumer = worker(store, tmp_path, http, [1000])
    assert consumer.run_once()
    http.voice = "wrong-voice"
    assert consumer.run_once()
    item = store.get_tts_campaign_items(created["id"])[0]
    assert item["state"] == "failed"
    client = TestClient(app_module.app)
    client.post("/admin/login", data={"password": "change-me"})
    parser = ReviewForms()
    parser.feed(client.get(f"/admin/tts?campaign_id={created['id']}").text)
    collection = next(form for form in parser.forms
                      if form.get("action") == "attach_job" and form.get("job_id") == "job-1")
    resubmit = next(form for form in parser.forms if form.get("action") == "retry")
    assert resubmit["confirm_duplicate"] == "1"
    assert "confirm_duplicate" not in collection
    # Direct resubmission still requires acknowledgement even without a UI checkbox.
    assert client.post("/admin/tts/review", data={
        "item_id": item["id"], "action": "retry",
    }).status_code == 400
    response = client.post("/admin/tts/review", data=collection, follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == f"/admin/tts?campaign_id={created['id']}"
    http.voice = "pl_PL-gosia-medium"
    assert consumer.run_once()
    assert store.get_tts_campaign_items(created["id"])[0]["state"] == "completed"
    assert len([request for request in http.requests if request.get_method() == "POST"]) == 1


@pytest.mark.parametrize("invalid", [None, "voice", "hash", "path"])
def test_local_zip_campaign_verifies_provenance_and_does_not_change_legacy_profiles(monkeypatch, store, tmp_path, invalid):
    import hashlib
    import zipfile
    from tts.local_provider import LocalProvider, LOCAL_PROFILES, request_identity
    from tts.provider import profiles_for_url

    token = tmp_path / "token"
    token.write_text("a" * 64)
    monkeypatch.setenv("VOICEGRADER_BRIDGE_TOKEN_FILE", str(token))
    url = "http://tts-api:8080"
    profile = LOCAL_PROFILES[0]
    created = store.create_tts_campaign(source(store, 1)["id"], [profile], provider_url=url)
    item = store.get_tts_campaign_items(created["id"])[0]
    audio = wav_bytes()
    entry = {
        "item_id": item["id"], "engine": "piper", "voice_id": profile["voice"],
        "voice_used": "wrong" if invalid == "voice" else profile["voice"],
        "real_tts": True, "demo_mode": False, "status": "success", "error": None,
        "generation_metrics": {"voice_used": profile["voice"]},
        "file": "../escape.wav" if invalid == "path" else "audio/file-000001.wav",
        "bytes": len(audio), "sha256": "0" * 64 if invalid == "hash" else hashlib.sha256(audio).hexdigest(),
    }
    manifest = {"schema_version": "1", "producer_mode": "REAL_ONLY", "job_id": "job-local",
                "request_id": request_identity(item), "status": "completed", "items": [entry]}
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("manifest.json", json.dumps(manifest))
        output.writestr(entry["file"], audio)
    raw = archive.getvalue()
    requests = []

    class LocalHTTP:
        def open(self, request, timeout):
            requests.append(request)
            assert request.get_header("Authorization") == "Bearer " + "a" * 64
            if request.get_method() == "POST":
                assert json.loads(request.data)["request_id"] == request_identity(item)
                response = Response({"job_id": "job-local", "status": "queued"})
                response.status = 202
                return response
            if request.full_url.endswith("/archive"):
                return Response(raw, "application/zip")
            return Response({"job_id": "job-local", "request_id": request_identity(item), "status": "completed",
                             "archive_url": "/v1/results/job-local/archive", "byte_size": len(raw),
                             "sha256": hashlib.sha256(raw).hexdigest()})

    runner = TTSWorker(store, tmp_path / "audio", provider_factory=lambda origin: LocalProvider(origin, opener=LocalHTTP()))
    assert runner.run_once() and runner.run_once()
    result = store.get_tts_campaign_items(created["id"])[0]
    assert result["state"] == ("completed" if invalid is None else "failed")
    assert store.count_audio_files("TTSCompare") == (1 if invalid is None else 0)
    assert len([request for request in requests if request.get_method() == "POST"]) == 1
    assert profiles_for_url(DEFAULT_PROVIDER_URL) == PROFILES
    assert profiles_for_url(url) == LOCAL_PROFILES
    assert store.get_setting("tts_sft_share_percent") == "0"


def test_local_request_identity_changes_only_for_explicit_new_synthesis_and_busy_is_retryable(store, tmp_path):
    from tts.local_provider import LOCAL_PROFILES, request_identity
    from tts.provider import SubmissionRejected

    created = store.create_tts_campaign(source(store, 1)["id"], [LOCAL_PROFILES[0]], provider_url="http://tts-api:8080")
    item = store.get_tts_campaign_items(created["id"])[0]
    identity = request_identity(item)

    class Busy:
        def submit(self, item):
            raise SubmissionRejected("Local engine busy", retryable=True)

    runner = TTSWorker(store, tmp_path, provider_factory=lambda url: Busy(), clock=lambda: 100)
    assert runner.run_once()
    waiting = store.get_tts_campaign_items(created["id"])[0]
    assert waiting["state"] == "pending"
    assert request_identity(waiting) == identity
    assert waiting["next_attempt_at"] > 100
    assert runner.run_once() is False
    waiting["review_log_json"] = json.dumps([{"action": "attach_job"}])
    assert request_identity(waiting) == identity
    waiting["review_log_json"] = json.dumps([{"action": "attach_job"}, {"action": "retry"}])
    assert request_identity(waiting) != identity


def test_local_preview_reports_long_text_exclusions_without_truncating_external_sources(store):
    from tts.local_provider import LOCAL_PROFILES

    corpus = source(store, 1)
    long_text = "a" * 281
    store.insert_audio_file(AudioFile.new("long.wav", corpus["name"], long_text))
    local = store.preview_tts_campaign(corpus["id"], [LOCAL_PROFILES[0]], provider_url="http://tts-api:8080")
    assert local["eligible_utterances"] == 1 and local["excluded_texts_over_limit"] == 1
    external = store.preview_tts_campaign(corpus["id"], [PROFILES[0]], provider_url=DEFAULT_PROVIDER_URL)
    assert external["eligible_utterances"] == 2 and external["excluded_texts_over_limit"] == 0
    assert long_text in {utterance["text"] for utterance in store.get_synthesis_source_utterances(corpus["id"])}


def test_local_ui_disables_unavailable_engines_and_preserves_external_campaigns(monkeypatch, store, tmp_path):
    import re
    import web.app as app_module
    from fastapi.testclient import TestClient
    from tts.local_provider import LOCAL_PROFILES, LocalProvider

    monkeypatch.setattr(app_module, "_get_store", lambda: store)
    monkeypatch.setattr(app_module, "_ADMIN_CFG_PATH", str(tmp_path / "password.json"))
    monkeypatch.setenv("TTS_BASE_URL", "http://tts-api:8080")
    monkeypatch.setattr(LocalProvider, "capabilities", lambda self: {
        "engines": [{"id": "piper", "voices": [{"id": "pl_PL-gosia-medium"}]}],
    })
    corpus = source(store, 1)
    previous = campaign(store, corpus)
    client = TestClient(app_module.app)
    client.post("/admin/login", data={"password": "change-me"})
    page = client.get(f"/admin/tts?campaign_id={previous['id']}")
    assert page.status_code == 200
    inputs = re.findall(r'<input[^>]*name="profiles"[^>]*>', page.text)
    assert len(inputs) == 4
    assert "checked" in inputs[0] and "disabled" not in inputs[0]
    assert all("disabled" in entry for entry in inputs[1:])
    assert page.context["selected"]["provider_url"] == DEFAULT_PROVIDER_URL
    assert page.context["profile_labels"][PROFILES[0]["id"]] == PROFILES[0]["label"]
    form = {"action": "preview", "corpus_id": corpus["id"], "profiles": LOCAL_PROFILES[1]["id"]}
    assert client.post("/admin/tts/create", data=form).status_code == 400
    form["profiles"] = LOCAL_PROFILES[0]["id"]
    assert client.post("/admin/tts/create", data=form).json()["missing"] == 1

    def offline(self):
        raise OSError("fixture API offline")

    monkeypatch.setattr(LocalProvider, "capabilities", offline)
    assert "Local TTS API unavailable" in client.get("/admin/tts").text
    assert client.post("/admin/tts/create", data=form).status_code == 503
    assert len(store.get_tts_campaigns()) == 1
