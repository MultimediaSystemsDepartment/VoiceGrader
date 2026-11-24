"""
nisqa_predictor.py
Performs audio quality assessment using nisqalib.
Non-Intrusive Speech Quality Assessment (NISQA v2.0).
Version based on the original NISQA implementation and nisqalib library.
Author: Grzegorz Szwoch, KSM PG, 2026.
"""

# NISQA (original):
# https://github.com/gabrielmittag/NISQA/tree/master
# nisqalib:
# https://github.com/kale4eat/nisqalib/tree/main

from collections import namedtuple
import warnings
import numpy as np
import torch
import soundfile as sf
import pyloudnorm
from . import nisqalib

warnings.filterwarnings('ignore', category=UserWarning, module='librosa')

NISQA_metrics = namedtuple('NISQA_metrics', 'mos noisiness discontinuity coloration loudness')


class NISQAPredictor:
    def __init__(self, device='auto', model='nisqa'):
        """Initializes the NISQA model.
        device: 'cuda', 'gpu', or 'auto' (default).
        model: 'nisqa' (default), 'nisqa_tts'.
        """
        self.predictor = nisqalib.NisqaModel(model, device)

    @staticmethod
    def _prepare_data(audio_data, srate=16000, lufs=None):
        if isinstance(audio_data, str):
            audio_data, srate = sf.read(audio_data)
        if isinstance(audio_data, np.ndarray):
            if audio_data.ndim == 2:
                audio_data = np.mean(audio_data, axis=np.argmin(audio_data.shape))
            elif audio_data.ndim != 1:
                raise ValueError('audio_data must be an 1D or 2D array.')
            if audio_data.dtype == 'int16':
                audio_data = (audio_data / 32768.0).astype('float32')
        elif not isinstance(audio_data, torch.Tensor):
            raise ValueError('Input must be a file path (str), NumPy array, or Torch tensor.')

        if lufs is not None:
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.numpy().squeeze()
            audio_data = NISQAPredictor.normalize_loudness(audio_data, srate, lufs)

        return audio_data, srate

    @staticmethod
    def normalize_loudness(audio_data, srate, lufs=-23):
        """Normalize loudness to a specific LUFS value in dB."""
        meter = pyloudnorm.Meter(srate)
        loudness = meter.integrated_loudness(audio_data)
        if loudness == float('-inf'):
            return audio_data
        normalized_audio = pyloudnorm.normalize.loudness(audio_data, loudness, lufs)
        return normalized_audio

    def predict(self, audio_data, srate=16000, lufs=-23):
        """Predicts audio quality from audio data.
        audio_data: file path, NumPy array, or Torch Tensor.
        srate: the sample rate of audio data.
        lufs: loudness normalization level, in dB LUFS.
        Returns: calculated metrics as NISQA_metrics named tuple.
        """
        audio_data, srate = self._prepare_data(audio_data, srate, lufs)
        if isinstance(audio_data, np.ndarray):
            audio_tensor = torch.from_numpy(audio_data.astype('float32'))
        else:
            audio_tensor = audio_data

        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        elif audio_tensor.ndim > 2:
            audio_tensor = audio_tensor.view(-1, audio_tensor.shape[-1])

        max_length = int(1300 * 0.04 * srate)  # max. 1300 windows with 40 ms hop
        if audio_tensor.shape[-1] > max_length:
            # limit of nisqalib exceeded
            audio_tensor = audio_tensor[:, :max_length]
            # print('Cut', audio_tensor.shape)

        result = self.predictor.predict(audio_tensor, srate)
        return NISQA_metrics(*map(float, result.values()))

    def predict_segments(self, audio_data, window_sec=10, hop_sec=5, srate=16000, lufs=-23):
        """Predicts audio quality averaged in short segments.
        audio_data: file name, NumPy array, or Torch tensor.
        window_sec: length of a single segment, in seconds (default: 10).
        hop_sec: length of a window hop, in seconds (default: 5).
        srate: the sample rate of audio data.
        lufs: loudness normalization level, in dB LUFS.
        Returns: calculated metrics (averaged for all segments), as NISQA_metrics named tuple.
        """
        audio_data, srate = self._prepare_data(audio_data, srate, lufs)
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.numpy().squeeze()
            if audio_data.ndim == 2:
                audio_data = np.mean(audio_data, axis=np.argmin(audio_data.shape))

        fade_samples = int(0.02 * srate)  # 20 ms fade-in / fade-out
        fade_in = np.linspace(0.0, 1.0, fade_samples)
        fade_out = np.linspace(1.0, 0.0, fade_samples)

        win_size = int(window_sec * srate)
        hop_size = int(hop_sec * srate)

        segment_results = []
        pos = 0

        while pos + win_size < len(audio_data) or len(segment_results) == 0:
            seg = np.copy(audio_data[pos : pos + win_size])
            seg[:fade_samples] *= fade_in
            seg[-fade_samples:] *= fade_out
            audio_tensor = torch.from_numpy(seg.astype('float32'))
            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            segment_results.append(self.predictor.predict(audio_tensor, srate))
            pos += hop_size

        assert len(segment_results) > 0
        means = {k: 0 for k in segment_results[0].keys()}
        for seg in segment_results:
            for k, v in seg.items():
                means[k] += v
        for k, v in means.items():
            means[k] = means[k] / len(segment_results)

        return NISQA_metrics(*map(float, means.values()))

    @property
    def device(self):
        """Return the currently used device."""
        return self.predictor.device


if __name__ == '__main__':
    import soundfile as sf

    predictor = NISQAPredictor()

    # wav = '/home/ksmdev/data/audio_gumed/Gabinet4_2025-07-25_133120_mono.wav'
    # wav = '/home/ksmdev/data/chatterbox/04_2026/test.wav'
    wav = '/home/ksmdev/data/audio_gumed/testowy.wav'

    # res = predictor.predict(wav)
    # print(res)

    audio_data, _ = sf.read(wav, dtype='float32')
    res = predictor.predict(audio_data)
    print(res)

    # res2 = predictor.predict_segments(audio_data)
    # print(res2)
