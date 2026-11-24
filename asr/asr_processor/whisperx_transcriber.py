"""
whisperx.py
Audio transcription with WhisperX (Systran).
"""

import warnings
import logging
import os
import numpy as np
import soundfile
import whisperx
from lightning.pytorch.utilities import disable_possible_user_warnings


DEF_MODEL_NAME = 'large-v3'
DEF_MODEL_PATH = './data/models/'

warnings.filterwarnings(
    'ignore', category=UserWarning, module='pyannote.audio.utils.reproducibility'
)
logging.getLogger('whisperx.vads.pyannote').setLevel(logging.WARNING)
logging.getLogger('lightning.pytorch').setLevel(logging.ERROR)
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)


class WhisperXTranscriber:
    """Performs audio data transcription to text with WhisperX (Systran)."""

    def __init__(
        self, model_name=None, model_path=DEF_MODEL_PATH, use_gpu=True, language=None
    ):
        """Create the transcription module.
        model_name: name of the model recognized by whisperx, e.g. 'large-v3'.
        model_path: path to directory with models.
        use_gpu: run transcription on GPU or CPU.
        language: language code for text recognition (default: None).
        """
        self.device = 'cuda' if use_gpu else 'cpu'
        compute_type = 'float16' if use_gpu else 'float32'
        self.lang = language
        self.result = None

        disable_possible_user_warnings()
        if model_name is None:
            model_name = os.environ.get('VOICEGRADER_ACTIVE_MODEL_DIR') or DEF_MODEL_NAME

        # open the model - it will be downloaded and saved to model_path if it does not exist
        self.model = whisperx.load_model(
            model_name,
            device=self.device,
            compute_type=compute_type,
            language=language,
            download_root=model_path,
        )

    def transcribe(self, audio_data, language=None, batch_size=32):
        """Transcribe audio to text.
        audio_data: NumPy array with audio data (16 kHz sampling rate),
                    path to audio file, or any object recognized by soundfile.read.
        language: language code; if None: use the one provided to the constructor.
        batch_size: number of audio chunks processed at the same time.
        Returns the transcription result as a dict: 'segments', 'language'.
        """
        if not isinstance(audio_data, np.ndarray):
            audio_data, srate = soundfile.read(audio_data, dtype='float32')
            if srate != 16000:
                raise ValueError('Audio file must have 16000 Hz sampling rate.')
        if audio_data.ndim != 1:
            audio_data = np.mean(audio_data, axis=np.argmin(audio_data.shape))
        if audio_data.dtype != 'float32':
            audio_data = audio_data.astype('float32')

        lang = language or self.lang

        return self.model.transcribe(audio_data, batch_size=batch_size, language=lang)

    @staticmethod
    def extract_text(result):
        """Extract text from the transcription result."""
        try:
            return ' '.join([seg['text'].strip() for seg in result['segments']])
        except KeyError:
            return ''

    @staticmethod
    def get_avg_logprob(result):
        """Return average logprob from segments in results."""
        try:
            num = 0.0
            den = 0.0
            for seg in result['segments']:
                dur = seg['end'] - seg['start']
                num += dur * seg['avg_logprob']
                den += dur
            return num / den
        except (KeyError, ZeroDivisionError):
            return None


if __name__ == '__main__':
    import soundfile as sf

    def test_whisper(audio_file):
        transcriber = WhisperXTranscriber(use_gpu=True)
        audio_data, _ = sf.read(audio_file)
        res = transcriber.transcribe(audio_data)
        print(res)

    test_whisper('/home/ksmdev/data/audio_gumed/Gabinet4_2025-07-25_133120_mono.wav')
