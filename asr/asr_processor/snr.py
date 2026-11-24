"""Calculate SNR (Signal-to-Noise Ratio),
using Silero VAD for segmentation of the recording.
Author: Grzegorz Szwoch, KSM PG, 2026.
"""

from collections import namedtuple
import numpy as np
import soundfile
import silero_vad as vad


SNR_metrics = namedtuple(
    'SNR_metrics', 'snr peak_snr level_speech level_noise perc_speech perc_noise'
)


class SNR:
    """Calculates signal-to-noise-ratio in a WAV file."""

    def __init__(self):
        self.vad_model = vad.load_silero_vad(onnx=True)

    @staticmethod
    def read_audio(filename):
        """Open sound file.
        filename: path to audio file, or any object recognized by soundfile.read.
        """
        wav, sr = soundfile.read(filename)
        if sr != 16000:
            raise ValueError('Input WAV file must have 16000 sampling rate.')
        if wav.ndim != 1:
            raise ValueError('Input WAV file must be single-channel.')
        return wav

    def vad(self, wav, min_speech_duration_ms=250, min_silence_duration_ms=350):
        """Perform voice activity detection.
        min_speech_duration_ms: minimum duration of a speech segment, in ms.
        min_silence_duration_ms: minimum duration of a silence segment, in ms.
        Returns a list of {start, end} dicts representing sample offsets of speech segments.
        """
        if hasattr(self.vad_model, 'reset_states'):
            self.vad_model.reset_states()
        elif hasattr(self.vad_model, 'model') and hasattr(self.vad_model.model, 'reset_states'):
            self.vad_model.model.reset_states()
        return vad.get_speech_timestamps(
            wav,
            self.vad_model,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            return_seconds=False,
        )

    @staticmethod
    def get_snr(wav, segments):
        """Compute SNR metrics.
        wav: audio data as NumPy array.
        segments: list of indices of speech segments as dict ('start', stop').
        Returns: SNR metrics as SNR_metrics named tuple.
        """
        num_samples = len(wav)
        if num_samples == 0:
            return SNR_metrics(
                snr=None,
                peak_snr=None,
                level_speech=None,
                level_noise=None,
                perc_speech=0.0,
                perc_noise=0.0,
            )

        # Create a boolean mask where True = speech, False = noise
        speech_mask = np.zeros(num_samples, dtype=bool)
        for seg in segments:
            speech_mask[seg['start'] : seg['end']] = True

        speech = wav[speech_mask]
        noise = wav[~speech_mask]

        count_speech = len(speech)
        count_noise = len(noise)

        perc_speech = count_speech / num_samples
        perc_noise = count_noise / num_samples

        snr = None
        peak_snr = None
        level_speech = None
        level_noise = None

        if count_speech > 0 and count_noise > 0:
            power_speech = np.sum(speech**2) / count_speech
            power_noise = np.sum(noise**2) / count_noise
            power_noise = max(power_noise, 1e-12)
            peak = np.max(np.abs(speech))

            level_noise = 10 * np.log10(power_noise)
            peak_snr = 10 * np.log10(peak**2 / power_noise)
            real_power_speech = power_speech - power_noise
            if real_power_speech > 0:
                level_speech = 10 * np.log10(real_power_speech)
                snr = 10 * np.log10(real_power_speech / power_noise)

        return SNR_metrics(
            snr=float(snr) if snr is not None else None,
            peak_snr=float(peak_snr) if peak_snr is not None else None,
            level_speech=float(level_speech) if level_speech is not None else None,
            level_noise=float(level_noise) if level_noise is not None else None,
            perc_speech=float(perc_speech),
            perc_noise=float(perc_noise),
        )

    def process(self, wav, *args, **kwargs):
        """Process the WAV file and calculate the metrics.
        wav: audio data as NumPy array.
        Any remaining arguments are passed to vad method.
        Returns: SNR metrics as SNR_metrics named tuple.
        """
        if isinstance(wav, str):
            wav = self.read_audio(wav)
        segments = self.vad(wav, *args, **kwargs)
        return self.get_snr(wav, segments)


if __name__ == '__main__':
    snr = SNR()

    for fname in ('wav/snr_05.wav', 'wav/snr_10.wav', 'wav/snr_15.wav', 'wav/snr_20.wav'):
        wav = snr.read_audio(fname)
        segments = snr.vad(wav)
        res = snr.get_snr(wav, segments)
        print(fname, res)
