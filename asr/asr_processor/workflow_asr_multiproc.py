"""workflow_asr_multiproc.py
Workflow for performing automatic speech recognition
and audio quality assessment.
Uses daemon processes that handle the processing stages.
Author: Grzegorz Szwoch, KSM PG, 2026.
"""

import multiprocessing as mp
from multiprocessing import shared_memory
import math
import queue
import time
from datetime import timedelta
import logging
import numpy as np
from .audiofile import AudioFile
from .snr import SNR
from .whisqa_predictor import WhiSQAPredictor
from .whisperx_transcriber import WhisperXTranscriber
from .wer import compute_metrics
# from .nisqa_predictor import NISQAPredictor


log = logging.getLogger('ASRWorkflow')


class Processor:
    """Contains a processor and an input queue."""

    def __init__(self, worker, out_que, *args):
        context = mp.get_context('spawn')
        self.queue = context.SimpleQueue()
        self.process = context.Process(target=worker, args=(self.queue, out_que, *args), daemon=True)

    @property
    def status(self):
        """Return the process status."""
        if self.process.is_alive():
            return 'Running'
        else:
            return f'Exited with code {self.process.exitcode}'


class BaseWorkflow:
    """Base class for the processing worflow."""

    def __init__(self, start=False):
        # output queue
        self.q_res = mp.get_context('spawn').Queue()
        # processors
        self.processors = {}
        self.t_start = None
        if start:
            self.start()

    def start(self, timeout=180):
        """Start the workers."""
        for processor in self.processors.values():
            processor.process.start()
        # block until all workers signal they are READY
        log.debug('Waiting for the workers to start.')
        ready_count = 0
        deadline = time.monotonic() + timeout
        while ready_count < len(self.processors):
            remaining = deadline - time.monotonic()
            try:
                msg = self.q_res.get(timeout=max(0.01, min(1, remaining)))
            except queue.Empty:
                if remaining <= 0 or any(
                    proc.process.exitcode is not None for proc in self.processors.values()
                ):
                    self.stop()
                    raise RuntimeError("Workers failed to become ready before deadline")
                continue
            if msg == 'READY':
                ready_count += 1
            elif msg is None:
                self.stop()
                raise RuntimeError('Not all workers could be started - exiting.')
        log.info('%d worker(s) have been started.', ready_count)
        self.t_start = time.perf_counter()
        return ready_count

    def stop(self):
        """Stop the workers"""
        for processor in self.processors.values():
            if processor.process.pid and processor.process.is_alive():
                processor.queue.put(None)
        log.debug('Waiting for the workers to finish.')
        for processor in self.processors.values():
            if processor.process.pid:
                processor.process.join(timeout=10)
                if processor.process.is_alive():
                    log.error("Worker %s failed to stop; terminating", processor.process.pid)
                    processor.process.terminate()
                    processor.process.join(timeout=5)
                if processor.process.is_alive():
                    processor.process.kill()
                    processor.process.join(timeout=5)
        log.info('All workers are stopped.')
        self.t_start = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop()

    @staticmethod
    def prepare_audio(filename, lufs=-23):
        """Read audio file and prepare it for processing.
        filename: path to audio file, or any object accepted by soundfile.read.
        lufs: loudness normalization level in dB LUFS,
            if None: normalize to peak level.
        Returns: single-channel audio data (16 kHz sampling rate), in NumPy array.
        """
        audio_file = AudioFile(filename)
        audio_file.resample(16000)
        if lufs is not None:
            audio_file.normalize_loudness(lufs)
            audio_file.prevent_clipping()
        else:
            audio_file.normalize(-0.1)
        return audio_file.audio_data, audio_file.srate

    @staticmethod
    def _base_worker(
        input_que, output_que, tag, process_class, process_fn, *init_args, **init_kwargs
    ):
        """Base worker code."""
        try:
            processor = process_class(*init_args, **init_kwargs)
            output_que.put('READY')
            # log.info('Worker %s ready.', tag)
        except Exception as _:
            log.exception('Exception when setting up the worker')
            output_que.put(None)
            return

        while True:
            input_data = input_que.get()
            if input_data is None:
                # log.debug('Worker %s finishing.', tag)
                return  # termination signal

            if isinstance(input_data, tuple):
                shm_name, extra_args = input_data[0], input_data[1:]
            else:
                shm_name, extra_args = input_data, ()

            shm = shared_memory.SharedMemory(name=shm_name)
            try:
                audio_data = np.frombuffer(shm.buf, dtype='float64')
                # Call the specific processing function.
                res = process_fn(processor, audio_data, *extra_args)
            except Exception as _:
                log.exception('Exception in the worker')
                res = None
            finally:
                output_que.put((tag, res, extra_args))
                if 'audio_data' in locals():
                    del audio_data
                shm.close()

    def process(self, wav, extra_args=None, verbose=False):
        """Process audio file with the workflow.
        wav: path to audio file.
        extra_args: additional args passed to workers.
        verbose: if True, print messages and results to the output.
        """
        t1 = time.perf_counter()
        if extra_args is None:
            extra_args = ()
        elif isinstance(extra_args, list):
            extra_args = tuple(extra_args)
        elif not isinstance(extra_args, tuple):
            extra_args = (extra_args,)
        if verbose:
            print('--- Prepare audio ---')
        audio_data, _ = self.prepare_audio(wav)
        audio_time = len(audio_data) / 16000
        received = {}
        if verbose:
            print('--- Start processing ---')
        try:
            # prepare shared mamory buffer and copy data
            shm = shared_memory.SharedMemory(create=True, size=audio_data.nbytes)
            shared_arr = np.ndarray(audio_data.shape, dtype=audio_data.dtype, buffer=shm.buf)
            shared_arr[:] = audio_data[:]
            # start the workflow
            for tag, processor in self.processors.items():
                processor.queue.put((shm.name, *extra_args))
            # collect the results
            while len(received) < len(self.processors):
                tag, result, *_ = self.q_res.get()
                assert tag not in received
                received[tag] = result
                if verbose:
                    print(f'--- {tag} ---\n{result}')

        finally:
            del audio_data
            shm.close()
            shm.unlink()

        received['processing_time'] = time.perf_counter() - t1
        received['audio_time'] = audio_time
        return received

    def get_status(self):
        """Return the workflow status as dict."""
        status = {}
        if self.t_start is not None:
            sec = time.perf_counter() - self.t_start
            delta = timedelta(seconds=sec)
            status['uptime'] = str(delta)
        else:
            status['uptime'] = 'Not running'
        for tag, proc in self.processors.items():
            status[tag] = proc.status
        return status


class QualityWorkflow(BaseWorkflow):
    """Workflow for performing audio quality assessment."""

    def __init__(self, start=False):
        super().__init__()
        # processors
        self.processors['snr'] = Processor(self.worker_snr, self.q_res)
        self.processors['qual'] = Processor(self.worker_qual, self.q_res, True)
        if start:
            self.start()

    @staticmethod
    def _run_snr(processor, data, *_):
        return processor.process(data)._asdict()

    @staticmethod
    def worker_snr(input_que, output_que):
        """Worker function for SNR (signal-to-noise ratio) analysis.
        input_que: Queue that receives data to process.
        output_que: Queue to which the results are written.
        Input queue item: audio_data: NumPy array with audio data.
        Output queue item: SNR metrics, as SNR_metrics named tuple.
        """
        ASRWorkflow._base_worker(
            input_que,
            output_que,
            tag='snr',
            process_class=SNR,
            process_fn=ASRWorkflow._run_snr,
        )

    @staticmethod
    def _run_qual(processor, data, *_):
        res = processor.predict(data, lufs=None)
        return {k: v for k, v in res._asdict().items() if not math.isnan(v)}

    @staticmethod
    def worker_qual(input_que, output_que, multi=False):
        """Worker function for audio quality assessment.
        input_que: Queue that receives data to process.
        output_que: Queue to which the results are written.
        multi: if True, use multidimensional model.
        Input queue item: audio_data: NumPy array with audio data.
        Output queue item: NISQA metrics, as NISQA_metrics named tuple.
        """
        ASRWorkflow._base_worker(
            input_que,
            output_que,
            tag='qual',
            process_class=WhiSQAPredictor,
            process_fn=ASRWorkflow._run_qual,
            multi=multi,
        )


class ASRWorkflow(QualityWorkflow):
    """Workflow for performing automatic speech recognition
    and audio quality assessment.
    """

    def __init__(self, start=False):
        super().__init__()
        # processors
        self.processors['asr'] = Processor(self.worker_asr, self.q_res)
        if start:
            self.start()

    @staticmethod
    def _run_asr(processor, data, ref_text=None, lang=None, *_):
        transcript = processor.transcribe(data, language=lang)
        res = {'recognized': processor.extract_text(transcript)}
        res.update(transcript)
        if 'segments' in transcript:
            res['avg_logprob'] = processor.get_avg_logprob(transcript)
        if ref_text:
            metrics = compute_metrics(res['recognized'], ref_text)
            res.update({'wer': metrics['wer'], 'cer': metrics['cer']})
        return res

    @staticmethod
    def worker_asr(input_que, output_que):
        """Worker function for WhisperX automatic speech recognition.
        input_que: Queue that receives data to process.
        output_que: Queue to which the results are written.
        Input queue item: (audio_data, ref_text).
        - audio_data: NumPy array with audio data.
        - ref_text: optional reference text for WER/CER calculation.
        - lang: language code for speech recognition.
        Output queue item: recognition results, as a dict.
        """
        ASRWorkflow._base_worker(
            input_que,
            output_que,
            tag='asr',
            process_class=WhisperXTranscriber,
            process_fn=ASRWorkflow._run_asr,
            language=None,
        )
