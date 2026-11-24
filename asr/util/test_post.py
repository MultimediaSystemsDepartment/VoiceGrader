"""Test the asr_workflow endpoint."""

import json
import requests

# Endpoint on a virtual machine
# ENDPOINT = 'http://153.19.237.74:8000/process'

# Endpoint on a Kubernetes cluster
ENDPOINT = 'https://asr-caise.ingress.caise.k8s.apl.task.gda.pl/process'


def process(wav_file, ref_text=None, endpoint=ENDPOINT):
    """Process WAV file with ASR Workflow."""
    with open(wav_file, 'rb') as fin:
        files = {'file': fin}
        params = {'ref_text': ref_text}
        res = requests.post(endpoint, data=params, files=files)
    data = {'status_code': res.status_code, 'elapsed': res.elapsed}
    if res.status_code == 200:
        data['response'] = res.json()
    else:
        data['response'] = res.content
    return data


if __name__ == '__main__':
    wav = 'Gabinet4_2025-07-25_133120_mono.wav'
    ref_txt = 'wh_ref.txt'
    # ref_txt = None

    if ref_txt:
        with open(ref_txt) as fin:
            ref = fin.read()
    else:
        ref = None

    res = process(wav, ref)

    print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
