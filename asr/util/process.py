"""process.py
Process audio file with CAISE-ASR service.
Automatic speech recognition with sound quality assessment.

Author: Grzegorz Szwoch
Gdańsk University of Technology (PG)
Department of Multimedia Systems (KSM)
Project CAISE, 2026.
"""

import json
import requests

# Default URL of the process endpoint
DEF_URL = 'https://asr-caise.ingress.caise.k8s.apl.task.gda.pl/process'


def process(
    wav_file: str, lang: str = 'pl', ref_text: str = None, timeout: float = 10.0, url: str = DEF_URL
) -> dict:
    """Process audio file with the CAISE-ASR service.
    wav_file: audio file with speech recording (wav, mp3, flac).
    lang: recognition language; if empty: autodetection.
    ref_text: optional reference text to compare.
    timeout: request timeout in seconds.
    url: optional URL of the processing endpoint.
    Returns: JSON document as dict, with the processing results.
    """
    try:
        with open(wav_file, 'rb') as fin:
            files = {'file': fin}
            params = {'ref_text': ref_text, 'language': 'pl'}
            params = {}
            if lang:
                params['language'] = lang
            if ref_text:
                params['ref_text'] = ref_text
            res = requests.post(url, data=params, files=files, timeout=timeout)
        if res.status_code == 200:
            return res.json()
        else:
            print('Server returned an error:', res.content)
            return None
    except Exception as e:
        print(f'Processing error: {e}')
        return None


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        prog='process',
        description='Processes audio files with the CAISE-ASR service.',
    )

    parser.add_argument('input_file', help='audio file to process (WAV, MP3, FLAC)')
    parser.add_argument('-r', '--ref-text', help='text file containing reference text (optional)')
    parser.add_argument(
        '-o',
        '--output-file',
        help='output file for the processing results (if not given: prints to STDOUT)',
    )
    parser.add_argument('-l', '--lang', help='language code for recognition; default: auto detect')
    parser.add_argument(
        '-t', '--timeout', type=float, default=10.0, help='request timeout in seconds'
    )
    parser.add_argument(
        '-u',
        '--url',
        help='URL of the process endpoint (if other than the default)',
        default=DEF_URL,
    )

    args = parser.parse_args()

    try:
        if args.ref_text:
            with open(args.ref_text) as fin:
                ref_text = fin.read().strip()
        else:
            ref_text = None

        result = process(args.input_file, args.lang, args.ref_text, args.timeout, args.url)

        if result is None:
            raise SystemExit(1)

        dump = json.dumps(result, indent=2, ensure_ascii=False)
        if args.output_file:
            with open(args.output_file, 'w') as fout:
                fout.write(dump)
        else:
            print(dump)

    except Exception as e:
        raise SystemExit(e)
