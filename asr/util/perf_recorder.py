#!/home/ksmdev/env/venv313/bin/python3

"""
perf_recorder.py
Record GPU and VRAM utilization.
Author: Grzegorz Szwoch, KSM PG, 2026.
"""

import time
from collections import namedtuple
import pynvml


PerfMetrics = namedtuple('PerfMetrics', 'ts, gpu, mem, mem_mb')


class PerfMonitor:
    """Monitors GPU/VRAM utilization."""

    def __init__(self, device=0):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device)

    def stop(self):
        pynvml.nvmlShutdown()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop()

    def __call__(self):
        """Returns metrics as PerfMetrics named tuple.
        - ts: POSIX timestamp.
        - gpu: GPU utilization in percents.
        - mem: VRAM utilization in percents.
        - mem_mb: VRAM usage in MB.
        """
        t_now = time.time()
        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        vram_mib = mem.used // (1024**2)
        vram_perc = 100 * mem.used / mem.total
        return PerfMetrics(t_now, util.gpu, vram_perc, vram_mib)


def record2(output_file, period=1.0, max_duration=None):
    if period <= 0:
        raise ValueError('Period must be larger than 0.')
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    with open(output_file, 'w') as fout:
        fout.write('Time,GPU%,VRAM%,VRAM_MB\n')
        t_start = time.time()
        while True:
            t_now = time.time()
            t_elapsed = t_now - t_start
            if max_duration is not None and t_elapsed >= max_duration:
                break
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_mib = mem.used // (1024**2)
            vram_perc = 100 * mem.used / mem.total
            fout.write(f'{t_elapsed:.3f},{util.gpu},{vram_perc:.2f},{vram_mib}\n')
            time.sleep(period)
    pynvml.nvmlShutdown()


def record(output_file=None, period=1.0, max_duration=None):
    if period <= 0:
        raise ValueError('Period must be larger than 0.')
    try:
        fout = None
        if output_file:
            fout = open(output_file, 'w')
            fout.write('Time,GPU%,VRAM%,VRAM_MB\n')
        t_start = time.time()
        with PerfMonitor() as perf:
            while True:
                met = perf()
                t_elapsed = met.ts - t_start
                if fout:
                    fout.write(f'{t_elapsed:.3f},{met.gpu},{met.mem:.2f},{met.mem_mb}\n')
                else:
                    print(
                        f'{t_elapsed:8.3f}: GPU={met.gpu:3d}%, VRAM={round(met.mem):3d}% ({met.mem_mb} MB)'
                    )
                if max_duration is not None and t_elapsed >= max_duration:
                    break
                time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        if fout:
            fout.close()


def test():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(util)
    print(mem)
    pynvml.nvmlShutdown()
    # c_nvmlUtilization_t(gpu: 0 %, memory: 0 %)
    # c_nvmlMemory_t(total: 100485038080 B, free: 99960487936 B, used: 524550144 B)


if __name__ == '__main__':
    # record('ttt.csv', 0.5, 20.0)
    record('perf.csv', 0.1, 35.0)
    # record(None, 0.5, 20.0)
