#!/home/ksmdev/env/venv313/bin/python3

# Entry point for workflow_asr_multiproc

import time
import logging
import multiprocessing as mp
from asr_processor import workflow_asr_multiproc as asrworkflow


def test_workflow(wav, ref_text=None, iterations=1, verbose=False):
    with asrworkflow.ASRWorkflow(start=True) as workflow:
        print('Start workflow')
        t1 = time.perf_counter()
        for _ in range(iterations):
            workflow.process(wav, ref_text, verbose=verbose)
        t2 = time.perf_counter()
        print('Workflow ended')
        elapsed = t2 - t1
        if iterations > 1:
            print(
                f'{iterations} iterations: total time = {elapsed:.3f} s, average time = {(elapsed / iterations):.3f} s.'
            )
        else:
            print(f'1 iteration: total time = {elapsed:.3f} s.')
    return elapsed


def test_iterations(max_it, wav, ref_text=None):
    timings = {}
    for nit in range(1, max_it + 1):
        t = test_workflow(wav, ref_text, iterations=nit, verbose=False)
        timings[nit] = t
    print()
    for nit, elapsed in timings.items():
        print(
            f'iterations={nit:2}: total time = {elapsed:.3f} s, average time = {(elapsed / nit):.3f} s.'
        )


if __name__ == '__main__':
    from pathlib import Path

    logging.basicConfig(format='%(asctime)s: %(name)s [%(levelname)s]: %(message)s')
    asrworkflow.log.setLevel(logging.DEBUG)

    mp.set_start_method('spawn')  # required by CUDA

    wav = 'data/testwav/Gabinet4_2025-07-25_133120_mono.wav'

    if 0:
        ref_text = None
    else:
        ref_text = Path('./data/testwav/wh_ref.txt').read_text()

    res = test_workflow(wav, ref_text, verbose=True)

    # res = test_workflow(wav, ref_text, iterations=10, verbose=False)

    # test_iterations(10, wav, ref_text)

    # with asrworkflow.ASRWorkflow(start=True) as workflow:
    #     res = workflow.process(wav, ref_text, verbose=True)
    #     print(workflow.get_status())
    #     print(res['elapsed'])
