import sys
import os
import uuid
import asyncio
import time
import json
import logging
import multiprocessing
import queue
import threading
import socket
import math
from multiprocessing import shared_memory
from datetime import datetime
from io import BytesIO
from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    Header,
    Request,
    Query,
    WebSocket,
    WebSocketDisconnect,
    Response,
    HTTPException,
)
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import websockets
import numpy as np
import psutil
import anyio
from whisper_live.server import TranscriptionServer
from .strings import STRINGS

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from asr_processor.workflow_asr_multiproc import ASRWorkflow
from asr_processor.perf_monitor import PerfMonitor
from asr_processor.model_update import (
    startup_model_update, loaded_model_status, latest_publication, refresh_model,
    installed_publication, service_authorized,
    pin_installed_publication,
)
from asr_processor.runtime_control import RuntimeControl, AdmissionMiddleware

os.environ['PYTHONWARNINGS'] = 'ignore:resource_tracker'

# Silence info messages from whisper live (hardcoded root logger).
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s', force=True)

# Configure local logger.
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

workflow = None
wl_process = None
perf_monitor = PerfMonitor()
ACTIVE_SESSIONS = {}
STREAM_QUEUES = {}
PROCESS_QUEUES = {}
WHISPER_SERVER_URL = 'ws://127.0.0.1:9090'
runtime_control = RuntimeControl()
PENDING_WORK = {}
PENDING_LOCK = threading.Lock()
reload_task = None
router_task = None
attempted_publication = None
reported_publication_id = None


def register_work(job_id, shm, tags):
    runtime_control.enter(admitted=True)
    with PENDING_LOCK:
        PENDING_WORK[job_id] = (shm, set(tags))


def finish_work(job_id, tag):
    with PENDING_LOCK:
        pending = PENDING_WORK.get(job_id)
        if pending is None:
            return
        shm, tags = pending
        tags.discard(tag)
        if tags:
            return
        del PENDING_WORK[job_id]
    try:
        shm.close()
        shm.unlink()
    finally:
        runtime_control.leave()


def require_service(authorization):
    try:
        authorized = service_authorized(authorization)
    except (OSError, ValueError) as exc:
        log.error("Service authentication unavailable: %s", exc)
        raise HTTPException(503, "Service credential unavailable") from exc
    if not authorized:
        raise HTTPException(401, "Service authentication required")


def stop_live_server():
    global wl_process
    if wl_process is None:
        return
    if wl_process.is_alive():
        children = []
        try:
            children = psutil.Process(wl_process.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            pass
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        # multiprocessing must reap its own child; psutil.wait_procs would steal its exit status.
        wl_process.terminate()
        wl_process.join(timeout=10)
        if wl_process.is_alive():
            wl_process.kill()
            wl_process.join(timeout=5)
        _, alive = psutil.wait_procs(children, timeout=5)
        for remaining in alive:
            try:
                remaining.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(alive, timeout=5)
        if wl_process.is_alive():
            raise RuntimeError("Live transcription server failed to stop")
    wl_process.close()
    wl_process = None


def run_live_server():
    TranscriptionServer().run(
        host='127.0.0.1', backend='faster_whisper', batch_enabled=True,
    )


def start_engines():
    global workflow, wl_process
    workflow = ASRWorkflow()
    workflow.start()
    wl_process = multiprocessing.get_context('spawn').Process(target=run_live_server)
    wl_process.start()
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and wl_process.is_alive():
        try:
            with socket.create_connection(('127.0.0.1', 9090), timeout=1):
                return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError("Live transcription server failed readiness")


async def stop_engines():
    global router_task
    if router_task:
        workflow.q_res.put((None, None, None))
        await router_task
        router_task = None
    if workflow:
        await asyncio.to_thread(workflow.stop)
    await asyncio.to_thread(stop_live_server)


async def reload_engines():
    global router_task, reported_publication_id
    old_dir = os.environ.get('VOICEGRADER_ACTIVE_MODEL_DIR')
    old_id = os.environ.get('VOICEGRADER_ACTIVE_PUBLICATION_ID', '')
    runtime_control.available = False
    try:
        await stop_engines()
        await asyncio.to_thread(refresh_model)
        installed = installed_publication()
        if installed:
            manifest, directory = installed
            os.environ['VOICEGRADER_ACTIVE_MODEL_DIR'] = str(directory.resolve())
            os.environ['VOICEGRADER_ACTIVE_PUBLICATION_ID'] = manifest['publication_id']
        await asyncio.to_thread(start_engines)
        router_task = asyncio.create_task(response_router())
        reported_publication_id = os.environ.get('VOICEGRADER_ACTIVE_PUBLICATION_ID') or None
        runtime_control.available = True
        os.environ.pop('VOICEGRADER_MODEL_UPDATE_ERROR', None)
    except Exception as exc:
        log.exception("Model reload failed; restoring previous runtime")
        os.environ['VOICEGRADER_MODEL_UPDATE_ERROR'] = str(exc)
        await stop_engines()
        await asyncio.to_thread(pin_installed_publication, old_id or None)
        if old_dir is None:
            os.environ.pop('VOICEGRADER_ACTIVE_MODEL_DIR', None)
        else:
            os.environ['VOICEGRADER_ACTIVE_MODEL_DIR'] = old_dir
        os.environ['VOICEGRADER_ACTIVE_PUBLICATION_ID'] = old_id
        try:
            await asyncio.to_thread(start_engines)
            router_task = asyncio.create_task(response_router())
            runtime_control.available = True
        except Exception:
            runtime_control.available = False
            log.exception("Previous runtime could not be restored; ASR unavailable")
            await stop_engines()
            raise
        raise


async def perform_reload():
    try:
        await runtime_control.reload(reload_engines,
                                     timeout=float(os.environ.get('VOICEGRADER_DRAIN_TIMEOUT', '300')))
    except Exception:
        log.exception("Hot reload did not complete; see model-publication status")


async def watch_publications():
    global reload_task, attempted_publication
    while True:
        await asyncio.sleep(60)
        if reload_task and not reload_task.done():
            continue
        try:
            manifest = await asyncio.to_thread(latest_publication)
            if manifest is None:
                continue
            identity = manifest['publication_id']
            if identity != os.environ.get('VOICEGRADER_ACTIVE_PUBLICATION_ID') and identity != attempted_publication:
                attempted_publication = identity
                reload_task = asyncio.create_task(perform_reload())
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            log.exception("Publication polling failed; current runtime unchanged")
            os.environ['VOICEGRADER_MODEL_UPDATE_ERROR'] = str(exc)


async def response_router():
    while True:
        try:
            tag, result, extra_args = await asyncio.to_thread(workflow.q_res.get)
            if tag is None:
                break

            # If a unique tracking ID exists at the end of extra_args
            if extra_args and len(extra_args) > 2:
                job_id = extra_args[-1]
                # Route ONLY to the streaming endpoint if it's a stream
                stream_queue = STREAM_QUEUES.get(job_id)
                process_queue = PROCESS_QUEUES.get(job_id)
                if stream_queue is not None:
                    await stream_queue.put((tag, result))
                # Route ONLY to the /process endpoint if it's a sync file task
                elif process_queue is not None:
                    process_queue.put_nowait((tag, result))
                finish_work(job_id, tag)
        except Exception as e:
            log.error(f'Router error: {e}')
            await asyncio.sleep(0.1)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global workflow, wl_process, router_task, reported_publication_id
    await asyncio.to_thread(startup_model_update)

    # Explicitly increase FastAPI's internal thread limiter pool size.
    # By default, FastAPI caps worker threads at 40. We raise this limit to
    # prevent simultaneous client files from getting bottlenecked in the thread pool.
    limiter = anyio.to_thread.current_default_thread_limiter()
    limiter.total_tokens = 200

    perf_monitor.start(
        period=0.2,
        initialization_time=10.0,
        tag='asr',
        job_name=os.environ.get('POD_NAME', 'asr'),
        constant_update_period=60,
        mimir_url=os.environ.get('MONITORING_URL'),
        mimir_user=os.environ.get('MONITORING_USERNAME'),
        mimir_passwd=os.environ.get('MONITORING_PASSWORD'),
    )
    startup_error = None
    watcher = None
    try:
        await asyncio.to_thread(start_engines)
        router_task = asyncio.create_task(response_router())
        reported_publication_id = os.environ.get('VOICEGRADER_ACTIVE_PUBLICATION_ID') or None
        if os.environ.get('VOICEGRADER_HOT_RELOAD', '0') == '1':
            watcher = asyncio.create_task(watch_publications())
        yield
        # Application is running.
    except Exception as e:
        startup_error = e
    finally:
        # Stop all processes.
        if watcher:
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass
        if reload_task and not reload_task.done():
            await reload_task
        await stop_engines()
        perf_monitor.stop()
        if startup_error:
            raise startup_error


app = FastAPI(lifespan=lifespan)
app.add_middleware(AdmissionMiddleware, control=runtime_control)
templates = Jinja2Templates(directory='app/templates')
app.mount('/static', StaticFiles(directory='app/static'), name='static')


@app.get('/api/model-publication')
def model_publication_status():
    status = loaded_model_status()
    status['loaded_publication_id'] = reported_publication_id
    status['restart_required'] = status['downloaded_publication_id'] != reported_publication_id
    status['state'] = 'published' if reported_publication_id else 'stock'
    return dict(status, reload=runtime_control.snapshot())


@app.post('/api/model-publication/reload', status_code=202)
async def model_publication_reload(authorization: str | None = Header(default=None)):
    global reload_task
    require_service(authorization)
    if reload_task and not reload_task.done():
        raise HTTPException(409, "A model reload is already running")
    reload_task = asyncio.create_task(perform_reload())
    return {"status": "queued", "behavior": "drain"}


job_data = {}


def extract_metric(data, attr_name, round_to=None):
    """Extract metric value as float."""
    try:
        if isinstance(data, dict):
            val = data.get(attr_name, None)
        else:
            val = getattr(data, attr_name, None)
        if val is None:
            return 'N/A'
        val = float(val)
        if round_to is not None:
            val = round(val, round_to)
        return val
    except (ValueError, TypeError):
        return 'N/A'


@app.get('/', response_class=HTMLResponse)
def index(
    request: Request,
    lang: str = Query(default=None),
    accept_language: str | None = Header(default=None),
):
    """Frontend for file processing, with recording from microphone."""
    if not lang or lang not in STRINGS:
        lang = 'en'  # default
        if accept_language:
            # try to read language from browser header
            for part in accept_language.split(','):
                lang_tag = part[:2].lower()
                if lang_tag in STRINGS:
                    lang = lang_tag
                    break

    return templates.TemplateResponse(
        request=request,
        name='index.html',
        context={
            'request': request,
            'language': lang,
            't': STRINGS[lang],
        },
    )


@app.post('/process')
def process(
    file: Annotated[UploadFile, File()], ref_text: str = Form(None), language: str = Form(None)
):
    """File processing endpoint."""
    res = {}
    shm = None
    job_id = f'proc-{uuid.uuid4().hex}'

    try:
        perf_monitor.add(job_id)
        content = file.file.read()
        audio_file = BytesIO(content)

        # Prepare audio data
        audio_data, _ = workflow.prepare_audio(audio_file)
        audio_time = len(audio_data) / 16000

        my_queue = queue.Queue()
        PROCESS_QUEUES[job_id] = my_queue

        shm = shared_memory.SharedMemory(create=True, size=audio_data.nbytes)
        shared_arr = np.ndarray(audio_data.shape, dtype=audio_data.dtype, buffer=shm.buf)
        shared_arr[:] = audio_data[:]

        register_work(job_id, shm, workflow.processors)
        shm = None
        for processor in workflow.processors.values():
            processor.queue.put((PENDING_WORK[job_id][0].name, ref_text, language, job_id))

        received = {}
        t1 = time.perf_counter()
        while len(received) < len(workflow.processors):
            tag, result = my_queue.get(timeout=30.0)
            received[tag] = result

        res = received
        res['processing_time'] = time.perf_counter() - t1
        res['audio_time'] = audio_time
        res['usage'] = perf_monitor.get(job_id)._asdict()

    except Exception as e:
        log.error(f'Error in process endpoint: {e}')
        res = {'error': str(e)}
    finally:
        if shm:
            shm.close()
            try:
                shm.unlink()
            except Exception:
                pass
        PROCESS_QUEUES.pop(job_id, None)
        perf_monitor.remove(job_id)
    return res


@app.get('/status')
def get_status():
    """Return the current status of the workflow."""
    try:
        return {
            'ts': str(datetime.now()),
            'workflow': workflow.get_status(),
            'monitor': perf_monitor.status(),
            'pod': os.environ.get('POD_NAME'),
        }
    except Exception as e:
        return {
            'ts': str(datetime.now()),
            'error': str(e),
            'pod': os.environ.get('POD_NAME'),
        }


@app.post('/api/quality')
def assess_quality(
    file: Annotated[UploadFile, File()],
    authorization: str | None = Header(default=None),
):
    """Report existing SNR/WhiSQA metrics, without transcription or rejection."""
    require_service(authorization)
    if not workflow:
        raise HTTPException(503, "Quality workers unavailable")
    content = file.file.read(16 * 1024 * 1024 + 1)
    if len(content) > 16 * 1024 * 1024:
        raise HTTPException(413, "Quality audio exceeds 16 MiB")
    job_id = f'quality-{uuid.uuid4().hex}'
    shm = None
    try:
        audio_data, _ = workflow.prepare_audio(BytesIO(content))
        if not 0.1 <= len(audio_data) / 16000 <= 30 or not np.isfinite(audio_data).all():
            raise ValueError("Quality review requires finite audio of 0.1-30 seconds")
        my_queue = queue.Queue()
        PROCESS_QUEUES[job_id] = my_queue
        shm = shared_memory.SharedMemory(create=True, size=audio_data.nbytes)
        shared_arr = np.ndarray(audio_data.shape, dtype=audio_data.dtype, buffer=shm.buf)
        shared_arr[:] = audio_data[:]
        name = shm.name
        register_work(job_id, shm, {'snr', 'qual'})
        shm = None
        for tag in ('snr', 'qual'):
            workflow.processors[tag].queue.put((name, None, None, job_id))
        results = {}
        while len(results) < 2:
            tag, result = my_queue.get(timeout=120)
            if not isinstance(result, dict):
                raise RuntimeError(f"Quality worker {tag} failed")
            results[tag] = {
                key: float(value) if value is not None and math.isfinite(float(value)) else None
                for key, value in result.items()
            }
        if results['qual'].get('mos') is None:
            raise RuntimeError("WhiSQA MOS is unavailable")
        return {"schema_version": 1, "profile": "asr-whisqa-multi-silero-snr-v1",
                "report_only": True, "metrics": results}
    except (ValueError, OSError) as exc:
        log.exception("Quality audio validation failed")
        raise HTTPException(400, str(exc)) from exc
    except (queue.Empty, RuntimeError) as exc:
        log.exception("Quality assessment failed")
        raise HTTPException(503, "Quality assessment failed or timed out") from exc
    finally:
        PROCESS_QUEUES.pop(job_id, None)
        if shm:
            shm.close()
            shm.unlink()


@app.get('/perf')
def get_perf():
    """Return the latest performance statistics."""
    stats = perf_monitor.get()
    if stats:
        return {
            'ts': datetime.fromtimestamp(stats.ts),
            'gpu_mean': round(stats.gpu_mean, 2),
            'gpu_max': round(stats.gpu_max, 2),
            'mem_mean': round(stats.mem_mean, 3),
            'mem_max': round(stats.mem_max, 3),
            'pod': os.environ.get('POD_NAME'),
        }
    else:
        return {}


@app.get('/favicon.ico', include_in_schema=False)
def favicon():
    return FileResponse('app/static/images/favicon.ico')


# Internal stream processing for files.
# Do not run these endpoints direclty.


@app.post('/start_stream', include_in_schema=False)
async def start_stream_processing(
    file: Annotated[UploadFile, File()],
    ref_text: str = Form(None),
    language: str = Form(None),
):
    """Endpoint for the file processing: start the processing stream."""
    job_id = str(uuid.uuid4())
    content = await file.read()
    job_data[job_id] = {'file_content': content, 'language': language, 'ref_text': ref_text}
    headers = {'HX-Trigger': json.dumps({'triggerStartStream': {'jobId': job_id}})}
    return Response(content='', headers=headers)


@app.get('/stream/{job_id}', include_in_schema=False)
async def stream_results(job_id: str):
    """Process the stream initiated by the frontend."""

    async def event_generator():
        data = job_data.get(job_id)
        if not data or not workflow:
            yield 'event: close\ndata: done\n\n'
            return

        yield f'event: start\ndata: {job_id}\n\n'

        my_queue = asyncio.Queue()
        STREAM_QUEUES[job_id] = my_queue
        shm = None
        try:
            file_io = BytesIO(data['file_content'])
            perf_id = f'stream-{job_id}'
            perf_monitor.add(perf_id)
            t_start = time.perf_counter()
            audio_data, _ = workflow.prepare_audio(file_io)
            shm = shared_memory.SharedMemory(create=True, size=audio_data.nbytes)
            shared_arr = np.ndarray(audio_data.shape, dtype=audio_data.dtype, buffer=shm.buf)
            shared_arr[:] = audio_data[:]

            name = shm.name
            register_work(job_id, shm, workflow.processors)
            shm = None
            for processor in workflow.processors.values():
                processor.queue.put((name, data['ref_text'], data['language'], job_id))

            received_count = 0
            while received_count < len(workflow.processors):
                try:
                    tag, result = await asyncio.wait_for(my_queue.get(), timeout=30.0)
                    received_count += 1

                    if tag == 'asr':
                        text = result.get('recognized', '') if isinstance(result, dict) else ''
                        wer = extract_metric(result, 'wer', 2)
                        cer = extract_metric(result, 'cer', 2)
                        logprob = extract_metric(result, 'avg_logprob', 2)
                        if logprob != 'N/A':
                            logprob = round(max(0.0, logprob + 1.0), 2)
                        val = f'{text}|{wer}|{cer}|{logprob}'
                    elif tag == 'snr':
                        val = extract_metric(result, 'snr', 2)
                    elif tag == 'qual':
                        val = extract_metric(result, 'mos', 2)
                    else:
                        val = 'N/A'

                    yield f'event: {tag}\ndata: {val}\n\n'
                except asyncio.TimeoutError:
                    yield ': heartbeat\n\n'
                except Exception as _:
                    break

            t_elapsed = time.perf_counter() - t_start
            values = [format(t_elapsed, '.3f')]
            perf_stats = perf_monitor.get(perf_id)
            if perf_stats is not None:
                values.append(format(perf_stats.gpu_mean, '.2f'))
                values.append(format(perf_stats.gpu_max, '.2f'))
                values.append(str(round(perf_stats.mem_mean)))
                values.append(str(round(perf_stats.mem_max)))
                values.append(str(round(perf_stats.mem_total)))
            else:
                values += ['N/A'] * 5
            val = '|'.join(values)
            yield f'event: stats\ndata: {val}\n\n'

            yield 'event: close\ndata: done\n\n'
        except Exception as e:
            log.error('Processing error: %s', e)
        finally:
            if shm:
                shm.close()
                try:
                    shm.unlink()
                except Exception:
                    pass
            STREAM_QUEUES.pop(job_id, None)
            job_data.pop(job_id, None)
            perf_monitor.remove(perf_id)

    return StreamingResponse(event_generator(), media_type='text/event-stream')


@app.websocket('/ws/live_stream')
async def websocket_live_stream(
    websocket: WebSocket, language: str | None = None, model: str | None = None
):
    """Websocket communication for WhisperLive mode.
    websocket: socket to communicate with the frontend.
    language: recognition language code; default: None (auto..
    model: Whisper model name; default: 'large-v3'.
    """
    await websocket.accept()
    session_id = f'client_{uuid.uuid4().hex}'
    log.debug('Unified WebSocket session initialized: %s', session_id)
    perf_monitor.add(session_id)

    try:
        # Connect to the WhisperLive transcription server's websocket.
        async with websockets.connect(WHISPER_SERVER_URL) as whisper_ws:
            config = {
                'uid': session_id,
                'language': language or None,
                'task': 'transcribe',
                'model': model or os.environ.get('VOICEGRADER_ACTIVE_MODEL_DIR', 'large-v3'),
                'use_vad': False,
            }
            await whisper_ws.send(json.dumps(config))
            log.debug('Sent Whisper Config: %s', config)

            # Performance metrics task
            async def perf_monitor_loop():
                try:
                    while True:
                        await asyncio.sleep(5)
                        perf_data = perf_monitor.get(session_id)
                        await websocket.send_json(
                            {'uid': session_id, 'status': 'PERF', 'message': perf_data._asdict()}
                        )
                except asyncio.CancelledError:
                    perf_data = perf_monitor.get(session_id)
                    await websocket.send_json(
                        {'uid': session_id, 'status': 'PERF', 'message': perf_data._asdict()}
                    )
                    raise

            perf_monitor_task = asyncio.create_task(perf_monitor_loop())

            # Receive data from WhisperLive and pass it to the frontend.
            async def whisper_reader():
                transcript = ''
                last_completed = 0
                try:
                    async for message in whisper_ws:
                        if isinstance(message, str):
                            data = json.loads(message)
                            output_text = ''
                            if data.get('segments'):
                                new_completed = []
                                not_completed = []
                                for seg in data['segments']:
                                    if float(seg['start']) < last_completed:
                                        continue
                                    if seg.get('completed', False):
                                        new_completed.append(seg['text'].strip())
                                        last_completed = float(seg['end'])
                                        not_completed = []
                                    else:
                                        not_completed.append(seg['text'].strip())
                                if new_completed:
                                    transcript = (' '.join([transcript] + new_completed)).strip()
                                if not_completed:
                                    output_text = (' '.join([transcript] + not_completed)).strip()
                                else:
                                    output_text = transcript
                                await websocket.send_json(
                                    {
                                        'uid': session_id,
                                        'transcript': output_text,
                                        'language': data.get('language', ''),
                                    }
                                )

                            elif 'status' in data:
                                # Status message - send to the frontend.
                                log.info(
                                    'Status: %s, %s', data['status'], data.get('message', 'None')
                                )
                                await websocket.send_json(data)
                            elif 'message' in data:
                                # WhisperLive server messages (SERVER_READY, DISCONNECT).
                                log.info('Message: %s', data['message'])
                                await websocket.send_json(data)

                    perf_monitor_task.cancel()
                    try:
                        await perf_monitor_task
                    except asyncio.CancelledError:
                        pass

                    # Notify the frontend that processing has finished.
                    log.debug(
                        'Whisper server closed connection normally for session %s', session_id
                    )

                    await websocket.send_json(
                        {
                            'uid': session_id,
                            'status': 'FINISHED',
                            'message': 'Audio processing has finished',
                        }
                    )

                except Exception as e:
                    log.error('Whisper response socket loop closed: %s', e)
                finally:
                    try:
                        await websocket.close()
                    except Exception:
                        pass

            # Run the reader loop task worker.
            reader_task = asyncio.create_task(whisper_reader())

            # Receive data from the frontend and pass it to WhisperLive server.
            END_OF_AUDIO = b'END_OF_AUDIO'
            # last_perf_ts = time.perf_counter()
            # perf_period = 5.0

            try:
                while True:
                    # Read data sent from the frontend (audio stream).
                    data = await websocket.receive_bytes()
                    if data:
                        assert len(data) % 4 == 0 or data == END_OF_AUDIO
                        if len(data) > 0:
                            # Pass the data to WhisperLive server.
                            await whisper_ws.send(data)
                        if data == END_OF_AUDIO:
                            log.debug(
                                'Received END_OF_AUDIO signal from client %s. Awaiting final segments...',
                                session_id,
                            )
                            break

            except WebSocketDisconnect:
                log.debug('Browser terminated connection for session %s', session_id)
                reader_task.cancel()
            finally:
                perf_monitor_task.cancel()
                try:
                    await perf_monitor_task
                except asyncio.CancelledError:
                    pass
                try:
                    await reader_task
                except asyncio.CancelledError:
                    pass

    except Exception as e:
        log.error('Proxy tunnel connection pipeline breakdown: %s', e)
    finally:
        log.debug('Disconnected and torn down session: %s', session_id)
        perf_monitor.remove(session_id)
