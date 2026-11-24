"""
whisqa_predictor.py
Performs audio quality assessment using WhiSQA
Whisper based Speech Quality Assessment.
Author: Grzegorz Szwoch, KSM PG, 2026.
"""

# Uses whisqa module from:
# https://github.com/leto19/WhiSQA@py_module

from collections import namedtuple
import numpy as np
import torch
import soundfile as sf
import pyloudnorm
from whisqa.models.whisper_ni_predictors import (
    whisperMetricPredictorEncoderLayersTransformerSmall,
    whisperMetricPredictorEncoderLayersTransformerSmalldim,
)
from whisqa.resources import package_file


WHISQA_metrics = namedtuple('WHISQA_metrics', 'mos noisiness discontinuity coloration loudness')


class WhiSQAPredictor:
    def __init__(self, device='auto', multi=False):
        """Initializes the WhiSQA model.
        device: 'cuda', 'gpu', or 'auto' (default).
        multi: if True, load multidimensional model (less accurate MOS),
               otherwise, load MOS-only model (more accurate MOS).
        """
        self.device = device
        if self.device not in ('cuda', 'cpu'):
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.multi = multi
        if multi:
            self.predictor = whisperMetricPredictorEncoderLayersTransformerSmalldim()
            checkpoint_name = 'multi_head_model.pt'
        else:
            self.predictor = whisperMetricPredictorEncoderLayersTransformerSmall()
            checkpoint_name = 'single_head_model.pt'
        with package_file('checkpoints', checkpoint_name) as checkpoint_path:
            self.predictor.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.predictor.eval()
        self.predictor.to(self.device)

    def predict(self, audio_data, lufs=-23):
        """Predicts audio quality from audio data.
        audio_data: file path, NumPy array, or Torch Tensor; 16 kHz mono.
        lufs: loudness normalization level, in dB LUFS.
        Returns: calculated metrics as WhiSQA_metrics named tuple.
        """
        # Prepare audio data.
        if isinstance(audio_data, str):
            audio_data, srate = sf.read(audio_data)
            if srate != 16000:
                raise ValueError('Audio data must have 16 kHz sampling rate.')
        if isinstance(audio_data, np.ndarray):
            if audio_data.ndim == 2:
                audio_data = np.mean(audio_data, axis=np.argmin(audio_data.shape))
            elif audio_data.ndim != 1:
                raise ValueError('audio_data must be an 1D or 2D array.')
            if audio_data.dtype == 'int16':
                audio_data = audio_data / 32768.0
        elif not isinstance(audio_data, torch.Tensor):
            raise ValueError('Input must be a file path (str), NumPy array, or Torch tensor.')

        if lufs is not None:
            # Normalize loudness.
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.numpy().squeeze()
            meter = pyloudnorm.Meter(16000)
            loudness = meter.integrated_loudness(audio_data)
            if loudness == float('-inf'):
                return audio_data
            audio_data = pyloudnorm.normalize.loudness(audio_data, loudness, lufs)

        if isinstance(audio_data, np.ndarray):
            audio_tensor = torch.from_numpy(audio_data.astype('float32'))
        else:
            audio_tensor = audio_data

        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        elif audio_tensor.ndim > 2:
            audio_tensor = audio_tensor.view(-1, audio_tensor.shape[-1])

        # Evaluation.
        audio_tensor = audio_tensor.to(self.device)  # not needed?
        with torch.inference_mode():
            result = self.predictor(audio_tensor)
        if self.multi:
            return WHISQA_metrics(*(x.item() * 5 for x in result.squeeze(0)))
        else:
            return WHISQA_metrics(
                result.item() * 5, float('nan'), float('nan'), float('nan'), float('nan')
            )


if __name__ == '__main__':
    wav = '/home/ksmdev/data/audio_gumed/testowy.wav'

    predictor = WhiSQAPredictor(multi=0)

    res = predictor.predict(wav)
    print(res)
