"""Script for stress testing CAISE ASR app.
Requests are sent to the app's endpoint using multiple parallel clients.
Test is performed for a defined time.
Author: Grzegorz Szwoch, KSM PG, grzszwoc@pg.edu.pl
"""

import threading
import time
import random
import pathlib
import datetime
import multiprocessing as mp
import concurrent.futures as cf
import requests

# -- CONFIGURATION ---

# URL of the app (root endpoint)
URL = 'http://153.19.237.74:8000'
# URL = 'https://asr-caise.ingress.caise.k8s.apl.task.gda.pl'

# Number of parallel clients
NUM_CLIENTS = 4

# Duration of the test in seconds
TEST_DURATION = 600

# Directory containing WAV files
WAVDIR = './test-dataset'

# --- END CONFIGURATION ---


def client(file_list, _id=0):
    params = {'language': 'pl'}
    stats = {
        'id': _id,
        'processed_count': 0,
        'total_audio_time': 0,
        'total_request_time': 0,
        'total_processing_time': 0,
        'runtime': 0,
    }

    proc_url = f'{URL}/process'
    runtime_start = time.perf_counter()
    running = True

    while running:
        files = random.sample(file_list, len(file_list))
        for fname in files:
            with open(fname, 'rb') as fb:
                try:
                    t_start = time.perf_counter()
                    res = requests.post(proc_url, files={'file': fb}, data=params)
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
                    running = False
                    break
                except requests.RequestException as err:
                    print(f'A network error occurred: {err}')
                    running = False
                    break
                except KeyError:
                    print('Invalid response from server:', data)
                    running = False
                    break
                except KeyboardInterrupt:
                    running = False
                    break
                except Exception as e:
                    running = False
                    print(e)
                    break
                if time.perf_counter() - runtime_start > TEST_DURATION:
                    running = False
                    break

    stats['runtime'] = time.perf_counter() - runtime_start
    return stats


def progress_bar(duration, stop_event):
    t_start = time.perf_counter()
    last_progress = None
    while True:
        t_now = time.perf_counter()
        progress = min(round(100 * (t_now - t_start) / duration), 100)
        if progress != last_progress:
            last_progress = progress
            print(f'\rProgress: {progress:3}%', end='', flush=True)
        if stop_event.wait(1.0):
            print()
            break


def process_executor(file_list, num_clients):
    results = []
    futures = []
    stop_event = threading.Event()
    pbar = threading.Thread(target=progress_bar, args=(TEST_DURATION, stop_event))
    pbar.start()
    with cf.ProcessPoolExecutor(max_workers=32, mp_context=mp.get_context('spawn')) as ex:
        for i in range(num_clients):
            futures.append(ex.submit(client, file_list, i))
        for future in cf.as_completed(futures):
            results.append(future.result())
    stop_event.set()
    pbar.join()
    return results


def run_test():
    # Check if the endpoint is available
    print(f'Checking {URL}... ', end='', flush=True)
    res = requests.get(f'{URL}/status')
    if res.status_code == 200:
        print('OK')
    else:
        print(f'ERROR {res.status_code}')
        print(res.content)
        raise SystemExit(1)

    wav_files = [str(f) for f in pathlib.Path(WAVDIR).glob('*') if f.suffix in ('.wav', '.mp3')]
    if not wav_files:
        print(f'No audio files found in {WAVDIR}')
        raise SystemExit

    print(f'Running {NUM_CLIENTS} client{"s" if NUM_CLIENTS > 1 else ""}')
    print(
        'Estimated finish time:',
        datetime.datetime.now() + datetime.timedelta(seconds=TEST_DURATION),
    )
    res = process_executor(wav_files, NUM_CLIENTS)

    print('Test finished.')
    for r in res:
        print(
            f'#{r["id"]}: {r["processed_count"]} files processed; '
            f'proc.t={r["total_processing_time"]:.3f}; '
            f'req.t={r["total_request_time"]:.3f}'
        )


if __name__ == '__main__':
    run_test()
