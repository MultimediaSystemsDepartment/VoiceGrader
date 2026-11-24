"""Testowanie zaszumionych sygnałów syntetycznych CAISE-ASR."""

from io import BytesIO
import requests
import soundfile as sf
import numpy as np


ENDPOINT = 'https://asr-caise.ingress.caise.k8s.apl.task.gda.pl/process'


def add_noise(signal, noise, target_snr_db):
    if noise is None:
        noise = np.random.normal(0, 1, signal.shape)
    rms_signal = np.sqrt(np.mean(signal**2))
    rms_noise = np.sqrt(np.mean(noise**2))
    cur_snr = rms_signal / rms_noise
    target_snr = 10 ** (0.05 * target_snr_db)
    gain = target_snr / cur_snr

    pos = 0
    mixed = np.zeros_like(signal)
    block_size = min(len(signal), len(noise))
    siglen = len(signal)
    while pos < siglen:
        block_size = min(block_size, siglen - pos)
        mixed[pos : pos + block_size] = gain * signal[pos : pos + block_size] + noise[:block_size]
        pos += block_size

    mixed /= np.max(np.abs(mixed))
    return mixed


num_runs = 10
dataset_file = 'admed_sample_1000_GT.txt'

dataset = []
with open(dataset_file) as fin:
    for line in fin:
        line = line.strip()
        if '|' in line:
            filename, ref_text = line.split('|', 1)
            filename = filename.replace('/recorded/', '/synth/')
            dataset.append((filename, ref_text))

for target_snr in (None, 20, 15, 10, 5, 0):
    print(f'--- SNR = {target_snr} ---')
    with open(f'synth_SNR_{target_snr}.csv', 'w') as fout:
        fout.write('FILE,WER,CER,MOS,SNR,LOGPROB\n')
        for fname, ref_text in dataset:
            if target_snr is not None:
                wav, sr = sf.read(fname)
                noisy = add_noise(wav, None, target_snr)
                fb = BytesIO()
                sf.write(fb, noisy, sr, 'PCM_16', format='WAV')
                fb.seek(0)
            else:
                fb = open(fname, 'rb')

            with fb:
                try:
                    res = requests.post(
                        ENDPOINT, files={'file': fb}, data={'language': 'pl', 'ref_text': ref_text}
                    )
                    res.raise_for_status()
                    data = res.json()
                    wer = data['asr']['wer']
                    cer = data['asr']['cer']
                    mos = data['qual']['mos']
                    snr = data['snr']['snr']
                    if snr is None:
                        snr = 0.0
                    logprob = data['asr']['avg_logprob']
                    if logprob is None:
                        logprob = -9.0
                    fout.write(f'{fname},{wer:.3f},{cer:.3f},{mos:.3f},{snr:.3f},{logprob:.3f}\n')
                except requests.HTTPError as http_err:
                    # 503 if server is busy
                    print(f'HTTP error occurred: {http_err.response.status_code} {http_err}')
                    raise SystemExit
                except requests.RequestException as err:
                    print(f'A network error occurred: {err}')
                    raise SystemExit
                except KeyError:
                    print('Invalid response from server:', data)
                    raise SystemExit
                # except Exception as e:
                #     print(e)
                #     raise SystemExit
