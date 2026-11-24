from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
import queue
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "asr_runtime_control_test", ROOT / "asr" / "asr_processor" / "runtime_control.py",
)
control_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control_module)


def test_drain_blocks_new_jobs_but_waits_for_sessions_and_worker_completion():
    async def scenario():
        control = control_module.RuntimeControl()
        assert control.enter()
        assert control.enter(admitted=True)
        switched = []

        async def switch():
            switched.append(True)

        task = asyncio.create_task(control.reload(switch))
        await asyncio.sleep(0)
        assert not control.enter()
        control.leave()
        await asyncio.sleep(0.06)
        assert not switched
        control.leave()
        await task
        assert switched and control.enter()
        control.leave()
        assert control.snapshot()["state"] == "idle"

    asyncio.run(scenario())


def test_drain_timeout_does_not_interrupt_work_or_switch_models():
    async def scenario():
        control = control_module.RuntimeControl()
        control.enter()

        async def forbidden():
            pytest.fail("A busy model must not be reloaded")

        with pytest.raises(TimeoutError):
            await control.reload(forbidden, timeout=0)
        assert control.snapshot()["active_operations"] == 1
        assert control.enter()
        control.leave()
        control.leave()

    asyncio.run(scenario())


def test_admission_rejects_http_and_websocket_during_reload_but_allows_status():
    async def scenario():
        control = control_module.RuntimeControl()
        control.draining = True
        accepted = []

        async def app(scope, receive, send):
            accepted.append(scope["path"])

        middleware = control_module.AdmissionMiddleware(app, control)
        sent = []

        async def send(value):
            sent.append(value)

        await middleware({"type": "http", "path": "/process"}, None, send)
        assert sent[0]["status"] == 503
        await middleware({"type": "websocket", "path": "/ws/live_stream"}, None, send)
        assert sent[-1]["code"] == 1013
        await middleware({"type": "http", "path": "/api/model-publication"}, None, send)
        assert accepted == ["/api/model-publication"]

    asyncio.run(scenario())


@pytest.fixture
def asr_app(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "asr"))
    workflow = types.ModuleType("asr_processor.workflow_asr_multiproc")
    workflow.ASRWorkflow = type("ASRWorkflow", (), {})
    perf = types.ModuleType("asr_processor.perf_monitor")
    perf.PerfMonitor = lambda: types.SimpleNamespace()
    live = types.ModuleType("whisper_live.server")
    live.TranscriptionServer = type("TranscriptionServer", (), {})
    monkeypatch.setitem(sys.modules, workflow.__name__, workflow)
    monkeypatch.setitem(sys.modules, perf.__name__, perf)
    monkeypatch.setitem(sys.modules, live.__name__, live)
    # StaticFiles uses the ASR working directory in the actual image.
    monkeypatch.chdir(ROOT / "asr")
    name = "asr.app.runtime_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "asr" / "app" / "main.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    token = tmp_path / "token"
    token.write_text("a" * 64)
    monkeypatch.setenv("VOICEGRADER_BRIDGE_TOKEN_FILE", str(token))
    return module


def test_failed_new_runtime_restores_previous_model_and_reports_failure(asr_app, monkeypatch, tmp_path):
    module = asr_app
    monkeypatch.setenv("VOICEGRADER_ACTIVE_MODEL_DIR", "previous")
    monkeypatch.setenv("VOICEGRADER_ACTIVE_PUBLICATION_ID", "previous-id")
    starts = []

    async def stop():
        pass

    async def router():
        return

    def start():
        model = module.os.environ["VOICEGRADER_ACTIVE_MODEL_DIR"]
        starts.append(model)
        if model != "previous":
            raise RuntimeError("New model GPU load failed")

    monkeypatch.setattr(module, "stop_engines", stop)
    monkeypatch.setattr(module, "response_router", router)
    monkeypatch.setattr(module, "start_engines", start)
    monkeypatch.setattr(module, "refresh_model", lambda: None)
    pinned = []
    monkeypatch.setattr(module, "pin_installed_publication", pinned.append)
    monkeypatch.setattr(module, "installed_publication", lambda: (
        {"publication_id": "candidate-id"}, tmp_path / "candidate",
    ))

    async def scenario():
        with pytest.raises(RuntimeError, match="GPU load"):
            await module.runtime_control.reload(module.reload_engines)
        assert module.os.environ["VOICEGRADER_ACTIVE_PUBLICATION_ID"] == "previous-id"
        assert module.runtime_control.available
        assert module.runtime_control.state == "failed"
        assert starts[-1] == "previous"
        assert pinned == ["previous-id"]
        await module.router_task

    asyncio.run(scenario())


def test_live_server_shutdown_reaps_owned_process_without_psutil_wait(asr_app, monkeypatch):
    module = asr_app
    process = module.multiprocessing.get_context("spawn").Process(target=time.sleep, args=(60,))
    process.start()
    module.wl_process = process
    pid = process.pid
    wait_procs = module.psutil.wait_procs

    def wait_descendants(processes, **kwargs):
        assert all(child.pid != pid for child in processes)
        return wait_procs(processes, **kwargs)

    monkeypatch.setattr(module.psutil, "wait_procs", wait_descendants)
    try:
        module.stop_live_server()
        assert module.wl_process is None
        with pytest.raises(ValueError, match="closed"):
            process.is_alive()
        module.stop_live_server()
    finally:
        if module.wl_process is not None:
            process.kill()
            process.join(timeout=10)
            process.close()
            module.wl_process = None


def test_quality_endpoint_reuses_only_snr_and_quality_workers_and_requires_auth(asr_app, monkeypatch):
    import numpy as np

    module = asr_app
    submissions = []

    def processor(tag):
        def put(args):
            _, _, _, job_id = args
            submissions.append(tag)
            metrics = {"mos": 4.1} if tag == "qual" else {"snr": None, "perc_speech": 0.8}
            module.PROCESS_QUEUES[job_id].put((tag, metrics))
            module.finish_work(job_id, tag)
        return types.SimpleNamespace(queue=types.SimpleNamespace(put=put))

    monkeypatch.setattr(module, "workflow", types.SimpleNamespace(
        prepare_audio=lambda source: (np.ones(3200, dtype=np.float64) * 0.1, 16000),
        processors={tag: processor(tag) for tag in ("snr", "qual", "asr")},
    ))
    client = TestClient(module.app)
    assert client.post("/api/quality", files={"file": ("clip.wav", b"audio")}).status_code == 401
    response = client.post("/api/quality", files={"file": ("clip.wav", b"audio")},
                           headers={"Authorization": "Bearer " + "a" * 64})
    assert response.status_code == 200
    assert response.json()["metrics"]["qual"]["mos"] == 4.1
    assert response.json()["report_only"] is True
    assert submissions == ["snr", "qual"]
    assert module.runtime_control.active == 0
    assert not module.PENDING_WORK


def test_worker_reference_holds_drain_after_request_finishes(asr_app):
    module = asr_app
    closed = []
    shm = types.SimpleNamespace(close=lambda: closed.append("close"),
                                unlink=lambda: closed.append("unlink"))
    module.runtime_control.enter()
    module.register_work("job", shm, {"snr", "qual"})
    module.runtime_control.leave()
    module.finish_work("job", "snr")
    assert module.runtime_control.active == 1
    assert not closed
    module.finish_work("job", "qual")
    assert module.runtime_control.active == 0
    assert closed == ["close", "unlink"]


def test_result_router_releases_disconnected_job_memory_after_all_workers_finish(asr_app, monkeypatch):
    module = asr_app
    results = queue.Queue()
    monkeypatch.setattr(module, "workflow", types.SimpleNamespace(q_res=results))
    closed = []
    shm = types.SimpleNamespace(close=lambda: closed.append("close"),
                                unlink=lambda: closed.append("unlink"))
    module.register_work("disconnected", shm, {"snr", "qual"})
    # No HTTP/SSE consumer remains; results must still release its worker lease.
    results.put(("snr", {}, (None, None, "disconnected")))
    results.put(("qual", {}, (None, None, "disconnected")))
    results.put((None, None, None))
    asyncio.run(module.response_router())
    assert module.runtime_control.active == 0
    assert not module.PENDING_WORK
    assert closed == ["close", "unlink"]


def test_successful_reload_reports_new_identity_only_after_workers_are_ready(asr_app, monkeypatch, tmp_path):
    module = asr_app
    module.reported_publication_id = "previous-id"
    monkeypatch.setenv("VOICEGRADER_ACTIVE_MODEL_DIR", "previous")
    monkeypatch.setenv("VOICEGRADER_ACTIVE_PUBLICATION_ID", "previous-id")
    calls = []

    async def stop():
        calls.append("stop")

    async def router():
        return

    def start():
        assert module.reported_publication_id == "previous-id"
        assert not module.runtime_control.available
        assert module.os.environ["VOICEGRADER_ACTIVE_PUBLICATION_ID"] == "new-id"
        calls.append("ready")

    monkeypatch.setattr(module, "stop_engines", stop)
    monkeypatch.setattr(module, "start_engines", start)
    monkeypatch.setattr(module, "refresh_model", lambda: calls.append("refresh"))
    monkeypatch.setattr(module, "response_router", router)
    monkeypatch.setattr(module, "installed_publication",
                        lambda: ({"publication_id": "new-id"}, tmp_path / "new"))

    async def scenario():
        await module.runtime_control.reload(module.reload_engines)
        assert module.reported_publication_id == "new-id"
        assert module.runtime_control.available
        assert calls == ["stop", "refresh", "ready"]
        await module.router_task

    asyncio.run(scenario())


def test_reload_control_requires_service_credential(asr_app):
    client = TestClient(asr_app.app)
    assert client.post("/api/model-publication/reload").status_code == 401
