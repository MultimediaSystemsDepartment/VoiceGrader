"""Test ARS-CAISE endpoint usning multiple clients."""

import time
import random
import requests
import multiprocessing as mp
import concurrent.futures as cf

ENDPOINT = 'https://asr-caise.ingress.caise.k8s.apl.task.gda.pl/process'


def client(file_list, _id=0):
    params = {'language': 'pl'}
    stats = {
        'id': _id,
        'processed_count': 0,
        'total_audio_time': 0,
        'total_request_time': 0,
        'total_processing_time': 0,
        'runtime': 0,
        'perf': [],
    }
    last_ts = {}
    perf_url = ENDPOINT.replace('/process', '/perf')

    with open(file_list) as fin:
        files = [x for line in fin if '.wav' in (x := line.strip())]

    random.shuffle(files)

    runtime_start = time.perf_counter()

    for fname in files:
        with open(fname, 'rb') as fb:
            try:
                t_start = time.perf_counter()
                res = requests.post(ENDPOINT, files={'file': fb}, data=params)
                res.raise_for_status()
                elapsed = time.perf_counter() - t_start
                data = res.json()
                stats['processed_count'] += 1
                stats['total_audio_time'] += data['audio_time']
                stats['total_request_time'] += elapsed
                stats['total_processing_time'] += data['processing_time']
            except requests.HTTPError as http_err:
                # 503 if server is busy
                print(f'HTTP error occurred: {http_err.response.status_code} {http_err}')
                break
            except requests.RequestException as err:
                print(f'A network error occurred: {err}')
                break
            except KeyError:
                print('Invalid response from server:', data)
                break
            except Exception as e:
                print(e)
                break

        try:
            resp = requests.get(perf_url)
            if resp.status_code == 200:
                data = resp.json()
                if data.pod not in last_ts:
                    last_ts[data.pod] = None
                if data and 'ts' in data and data['ts'] != last_ts[data.pod]:
                    stats['perf'].append(data.copy())
                    last_ts[data.pod] = data['ts']
        except Exception as _:
            pass

    stats['runtime'] = time.perf_counter() - runtime_start
    return stats


def process_executor(file_list, num_clients):
    results = []
    futures = []
    with cf.ProcessPoolExecutor(max_workers=32, mp_context=mp.get_context('spawn')) as ex:
        for i in range(num_clients):
            futures.append(ex.submit(client, file_list, i))
        for future in cf.as_completed(futures):
            results.append(future.result())
    return results


if __name__ == '__main__':
    import sys
    import json
    from datetime import timedelta

    try:
        num_procs = int(sys.argv[1])
    except Exception as _:
        num_procs = 1
    print('Number of clients:', num_procs)

    res = process_executor('admed_sample_1000.txt', num_procs)

    with open(f'result-2pods-{num_procs}.json', 'w') as fout:
        json.dump(res, fout, indent=2)

    for data in res:
        for k in data:
            if k != 'perf':
                print(f'{k}: {data[k]}')

    first = res[0]
    print('\nRunning time:', timedelta(seconds=first['runtime']))

    # RTF: Real-Time Factor
    if first['total_audio_time'] > 0:
        rtf_request = first['total_request_time'] / first['total_audio_time']
        rtf_process = first['total_processing_time'] / first['total_audio_time']
        print(f'RTF request = {100 * rtf_request:.2f}%, RTF process = {100 * rtf_process:.2f}%')
    print()
