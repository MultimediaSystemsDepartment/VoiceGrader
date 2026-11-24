"""Testowanie stabilności metryk CAISE-ASR."""

import random
import requests

ENDPOINT = 'https://asr-caise.ingress.caise.k8s.apl.task.gda.pl/process'

num_runs = 10
dataset_file = 'admed_sample_1000_GT.txt'

dataset = []
wav_names = []
with open(dataset_file) as fin:
    for line in fin:
        line = line.strip()
        if '|' in line:
            filename, ref_text = line.split('|', 1)
            wav_names.append(filename)
            dataset.append((filename, ref_text))

for run in range(num_runs):
    print(f'--- RUN #{run + 1} ---')
    results = {wav: None for wav in wav_names}
    random.shuffle(dataset)

    for fname, ref_text in dataset:
        with open(fname, 'rb') as fb:
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
                num = 0
                den = 0
                for seg in data['asr']['segments']:
                    dur = seg['end'] - seg['start']
                    num += dur * seg['avg_logprob']
                    den += dur
                logprob = num / max(den, 1e-8)
                results[fname] = (wer, cer, mos, snr, logprob)
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
            except Exception as e:
                print(e)
                raise SystemExit

    with open(f'stability-run{run + 1:02}.csv', 'w') as fout:
        fout.write('FILE,WER,CER,MOS,SNR,LOGPROB\n')
        for fname, vals in results.items():
            fout.write(
                f'{fname},{vals[0]:.3f},{vals[1]:.3f},{vals[2]:.3f},{vals[3]:.3f},{vals[4]:.3f}\n'
            )
