"""audiofile.py
Class for operations on a single monophonic audio file.
Author: Grzegorz Szwoch, KSM PG, 2026.
"""

import numpy as np
import soundfile as sf
import soxr
import pyloudnorm
# import logging

# log = logging.getLogger(__name__)


class AudioFile:
    """Represents a single monophonic audio file."""

    def __init__(self, filename=None, audio_data=None, srate=None, copy=False):
        if audio_data is not None and copy:
            self.audio_data = np.copy(np.asarray(audio_data))
        else:
            self.audio_data = audio_data
        self.srate = srate
        if filename:
            self.open(filename)

    def open(self, filename):
        """Read the audio file."""
        try:
            audio_data, self.srate = sf.read(filename)
        except (sf.SoundFileError, FileNotFoundError):
            raise ValueError('Error in opening audio file')
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=-1)
        self.audio_data = audio_data

    def set_data(self, audio_data, srate=None, copy=True):
        """Set the audio data.
        audio_data: NumPy array with audio data.
        srate: sampling rate (if None: use the current one).
        copy: if True (default), copy the data into this object,
              if False: assign the reference to the array.
        """
        if self.srate is None and srate is None and audio_data is not None:
            raise ValueError('Sample rate must be provided.')
        if copy:
            self.audio_data = np.copy(np.asarray(audio_data))
        else:
            self.audio_data = audio_data
        if srate:
            self.srate = srate

    def duplicate(self):
        """Create a duplicate of the object."""
        audio_data = np.copy(self.audio_data) if self.audio_data is not None else None
        return AudioFile(audio_data=audio_data, srate=self.srate)

    @staticmethod
    def notempty(func):
        """Decorator to check if audio_data is available."""

        def internal(self, *args, **kwargs):
            if self.audio_data is None:
                raise ValueError('No audio data available')
            return func(self, *args, **kwargs)

        return internal

    @notempty
    def resample(self, new_srate, copy=False):
        """Resample the stream to a new sample rate.
        If copy=True: create a duplicate, resample and return it.
        """
        if copy:
            obj = self.duplicate()
        else:
            obj = self
        if new_srate != self.srate:
            obj.audio_data = soxr.resample(obj.audio_data, obj.srate, new_srate)
            obj.srate = new_srate
        return obj

    @notempty
    def amplify(self, gain, linear=False):
        """Amplify the audio data.
        gain: amplification in dB (if linear is False),
              or as a linear gain (if linear is True).
        """
        if not linear:
            lin_gain = 10 ** (0.05 * gain)
            self.audio_data *= lin_gain
        else:
            self.audio_data *= gain

    @notempty
    def normalize(self, peak_level=-0.1):
        """Normalize audio data to a given peak level in dB."""
        target_peak = 10 ** (0.05 * peak_level)
        cur_peak = np.max(np.abs(self.audio_data))
        gain = target_peak / cur_peak
        self.audio_data *= gain

    @notempty
    def prevent_clipping(self):
        """Scale the audio data if necesssary to prevent clipping."""
        cur_peak = np.max(np.abs(self.audio_data))
        max_peak = 0.9999
        if cur_peak > max_peak:
            gain = max_peak / cur_peak
            self.audio_data *= gain

    @notempty
    def set_rms(self, target_rms):
        """Modify audio data to obtain the desired RMS level."""
        cur_rms = self.rms
        gain_db = target_rms - cur_rms
        gain_lin = 10 ** (0.05 * gain_db)
        self.audio_data *= gain_lin

    @notempty
    def normalize_loudness(self, lufs, copy=False):
        """Normalize the loudness, according to ITU-T BS.1770-4.
        lufs: tarhet loudness level, in dB LUFS
              (should be <= 0, recommended: -23 or -26).
        copy: if True, return a new object with the modified data.
        """
        meter = pyloudnorm.Meter(self.srate)
        loudness = meter.integrated_loudness(self.audio_data)
        if loudness != float('-inf'):
            normalized_audio = pyloudnorm.normalize.loudness(self.audio_data, loudness, lufs)
        else:
            normalized_audio = self.audio_data.copy()  # do nothing
        if copy:
            return AudioFile(audio_data=normalized_audio, srate=self.srate)
        else:
            self.audio_data = normalized_audio

    @notempty
    def mix(self, other_audio, snr=None, copy=False):
        """Mix the audio data with another audio data.
        other_audio: NumPy array or AudioFile containing audio data to mix.
        snr: if True, mix audio data so that the desired Signal-to-Noise in dB
             is achieved (this audio data is the signal, the other is noise).
        copy: if True, return a copy of the audio data, otherwise process in place.
        """
        if copy:
            this_audio = self.duplicate()
        else:
            this_audio = self
        if isinstance(other_audio, AudioFile):
            if other_audio.srate != self.srate:
                raise ValueError('Sampling rates must be identical.')
            other_audio = other_audio.audio_data
        elif not isinstance(other_audio, np.ndarray):
            raise TypeError('other_audio must be a NumPy array or AudioFile object.')
        if snr is not None:
            cur_snr = this_audio.rms - 10 * np.log10(np.mean(other_audio**2))
            gain_db = cur_snr - snr
            gain = 10 ** (0.05 * gain_db)
        else:
            gain = 1
        pos = 0
        block_size = min(len(this_audio.audio_data), len(other_audio))
        while pos < len(this_audio.audio_data):
            block_size = min(block_size, len(this_audio.audio_data) - pos)
            this_audio.audio_data[pos : pos + block_size] += gain * other_audio[:block_size]
            pos += block_size
        return this_audio

    @notempty
    def write(self, output_file, output_format='PCM_16'):
        sf.write(output_file, self.audio_data, self.srate, output_format)

    @property
    def peak(self):
        """Return linear peak value."""
        if self.audio_data is not None:
            return np.max(np.abs(self.audio_data))
        else:
            return None

    @property
    def peak_level(self):
        """Return peak level in dB."""
        if self.audio_data is not None:
            peak = np.max(np.abs(self.audio_data))
            return 20 * np.log10(peak)
        else:
            return None

    @property
    def rms(self):
        """Return RMS value in dB."""
        if self.audio_data is not None:
            return 10 * np.log10(np.mean(self.audio_data**2))
        else:
            return None

    def __str__(self):
        if self.audio_data is not None:
            dur = len(self.audio_data) / self.srate
            return f'AudioFile: {len(self.audio_data)} samples, {dur:.3} sec, srate={self.srate}'
        else:
            return 'AudioFile (empty)'

    def __deepcopy__(self, _):
        """Object duplication (deep copy)"""
        if self.audio_data:
            return AudioFile(audio_data=np.copy(self.audio_file), srate=self.srate)
        else:
            return AudioFile()

    @notempty
    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, item):
        if self.audio_data is not None:
            return self.audio_data.__getitem__(item)
        else:
            raise IndexError('Audio data is empty.')

    def __call__(self):
        return self.audio_data


if __name__ == '__main__':

    def test_mix():
        signal = AudioFile('wav/test_chatter.wav')
        noise = AudioFile('wav/test_pink_noise.wav')
        if noise.srate != signal.srate:
            noise.resample(signal.srate)
        for snr in [5, 10, 15, 20]:
            res = signal.mix(noise, snr, copy=True)
            res.normalize()
            res.write(f'wav/snr_{snr:02d}.wav')

    test_mix()

    # wav = 'wav/test_chatter.wav'
    # audio = AudioFile(wav)
    # print(audio)
    # print('RMS =', audio.rms)
    # print('Peak =', audio.peak_level)
    # audio.set_rms(-20.0)
    # print('New RMS =', audio.rms)
