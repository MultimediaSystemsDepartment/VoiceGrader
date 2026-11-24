#!/home/ksmdev/env/venv313/bin/python3

"""audiofile.py
Workflow for performing automatic speech recognition
and audio quality assessment.
Single process workflow.
Author: Grzegorz Szwoch, KSM PG, 2026.
"""

import sys
import os
import concurrent.futures
import time
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from asr_processor.audiofile import AudioFile
from asr_processor.snr import SNR
from asr_processor.nisqa_predictor import NISQAPredictor
from asr_processor.whisqa_predictor import WhiSQAPredictor
from asr_processor.whisperx_transcriber import WhisperXTranscriber
from asr_processor.wer import compute_metrics


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


def snr_worker(audio_data):
    """Worker function for SNR (signal-to-noise ratio) analysis.
    audio_data: NumPy array with audio data.
    Returns: SNR metrics, as SNR_metrics named tuple.
    """
    model = SNR()
    return model.process(audio_data)


def nisqa_worker(audio_data, srate):
    """Worker function for NISQA - audio quality assessment.
    audio_data: NumPy array with audio data.
    Returns: NISQA metrics, as NISQA_metrics named tuple.
    """
    model = NISQAPredictor()
    return model.predict_segments(audio_data, srate=srate, lufs=None)


def whisqa_worker(audio_data):
    """Worker function for WhiSQA audio quality assessment.
    audio_data: NumPy array with audio data.
    Returns: WhiSQA metrics, as WhiSQA_metrics named tuple.
    """
    model = WhiSQAPredictor(multi=True)
    return model.predict(audio_data, lufs=None)


def whisper_worker(audio_data, ref_text=None):
    """Worker function for WhisperX automatic speec recognition.
    audio_data: NumPy array with audio data.
    ref_text: optional reference text for WER/CER calculation.
    Returns: recognition results, as a dict.
    """
    model = WhisperXTranscriber()
    recognized = model.transcribe(audio_data, language='pl')
    text = model.extract_text(recognized)
    result = {'recognized': text}
    result.update(recognized)
    result['avg_logprob'] = model.get_avg_logprob(recognized)
    if ref_text is not None:
        metrics = compute_metrics(text, ref_text)
        result['wer'] = metrics['wer']
        result['cer'] = metrics['cer']
    return result


def asr_workflow_sequential(audio_file, ref_text=None, verbose=False):
    """ASR workflow using sequential processing."""
    result = {'audio_file': audio_file, 'snr': None, 'nisqa': None, 'whisqa': None, 'asr': None}

    if verbose:
        print('--- Prepare audio ---')
    audio_data, srate = prepare_audio(audio_file)

    snr_metrics = snr_worker(audio_data)
    result['snr'] = snr_metrics
    if verbose:
        print('\n--- SNR metrics ---\n', snr_metrics)

    nisqa_metrics = nisqa_worker(audio_data, srate)
    result['nisqa'] = nisqa_metrics
    if verbose:
        print('\n--- NISQA metrics ---\n', nisqa_metrics)

    asr = whisper_worker(audio_data, ref_text)
    result['asr'] = asr
    if verbose:
        print('\n--- Whisper ---\n', asr)

    return result


def asr_workflow(audio_file, ref_text=None, verbose=False):
    """ASR workflow using concurrent processing."""
    audio_data, srate = prepare_audio(audio_file)
    result = {'audio_file': audio_file, 'snr': None, 'nisqa': None, 'asr': None}
    tasks = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # create concurrent tasks
        tasks[executor.submit(whisper_worker, audio_data, ref_text)] = 'asr'
        # tasks[executor.submit(nisqa_worker, audio_data, srate)] = 'nisqa'
        tasks[executor.submit(whisqa_worker, audio_data)] = 'whisqa'
        tasks[executor.submit(snr_worker, audio_data)] = 'snr'
        # collect the finished tasks
        for future in concurrent.futures.as_completed(tasks):
            tag = tasks[future]
            if tag == 'snr':
                result['snr'] = future.result()
                if verbose:
                    print('\n--- SNR metrics ---\n', result['snr'])
            elif tag == 'nisqa':
                result['nisqa'] = future.result()
                if verbose:
                    print('\n--- NISQA metrics ---\n', result['nisqa'])
            elif tag == 'whisqa':
                result['whisqa'] = future.result()
                if verbose:
                    print('\n--- WhiSQA metrics ---\n', result['whisqa'])
            elif tag == 'asr':
                result['asr'] = future.result()
                if verbose:
                    print('\n--- Whisper ---\n', result['asr'])
            else:
                print('Unknown tag')
    return result


def time_perf(wav, ref_text=None):
    # DON"T run with ProcessPoolExecutor - it freezes because of proess forking.
    # Don't run both - sequential and parallel
    t1 = time.perf_counter()
    res1 = asr_workflow_sequential(wav, ref_text, verbose=False)
    t2 = time.perf_counter()
    res2 = asr_workflow(wav, ref_text, verbose=False)
    t3 = time.perf_counter()
    dur_seq = t2 - t1
    dur_par = t3 - t2
    print(f'Sequential: {dur_seq:.3f} s.\nParallel:   {dur_par:.3f} s.')
    return res1, res2


if __name__ == '__main__':
    import os
    # import multiprocessing

    # multiprocessing.set_start_method('spawn')  # needed by CUDA, fork should not be used

    os.chdir('..')

    wav = 'data/testwav/Gabinet4_2025-07-25_133120_mono.wav'

    if 0:
        ref_text = None
    else:
        ref_text = Path('data/testwav/wh_ref.txt').read_text()

    verbose = True

    t1 = time.perf_counter()

    # res = asr_workflow_sequential(wav, ref_text, verbose=verbose)
    res = asr_workflow(wav, ref_text, verbose=verbose)

    t2 = time.perf_counter()
    elapsed = t2 - t1
    print(f'\nExecution time: {elapsed:.3f} seconds.')

    # print()
    # print(res)
