"""
perfmon.py
Record GPU and VRAM utilization across VM and MIG-enabled Kubernetes clusters.
Author: Grzegorz Szwoch, KSM PG, 2026.
"""

import multiprocessing as mp
import statistics as stat
import time
from collections import namedtuple
import pynvml

PerfMetrics = namedtuple('PerfMetrics', 'ts, gpu, mem, mem_mb')
PerfStats = namedtuple('PerfStats', 'mean std median min max')


class PerfReader:
    """Reads GPU & VRAM metrics using NVML."""

    def __init__(self, device=0):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        self.gpm_sample1 = None
        self.gpm_sample2 = None
        self.handle = handle
        self.is_mig = False

        try:
            # Check if standard telemetry works (Non-MIG profile)
            pynvml.nvmlDeviceGetUtilizationRates(handle)
            pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.read_strategy = self._read_standard
        except pynvml.NVMLError_NotSupported:
            # MIG is active. Find our assigned container MIG handle.
            try:
                # Look for the exact MIG device indices allocated to this physical GPU
                mig_count = pynvml.nvmlDeviceGetMaxMigDeviceCount(handle)
                for i in range(mig_count):
                    try:
                        # Try to grab the explicit MIG handle
                        mig_handle = pynvml.nvmlDeviceGetMigDeviceHandleByIndex(handle, i)
                        # Test if this container has visibility over this instance
                        pynvml.nvmlDeviceGetMemoryInfo(mig_handle)
                        self.handle = mig_handle
                        self.is_mig = True
                        break
                    except pynvml.NVMLError:
                        continue
            except Exception:
                pass

            self.read_strategy = self._read_mig_gpm
            self.reset()

    def reset(self):
        """Reset the state and start monitoring."""
        if self.is_mig:
            if self.gpm_sample1:
                pynvml.nvmlGpmSampleFree(self.gpm_sample1)
            if self.gpm_sample2:
                pynvml.nvmlGpmSampleFree(self.gpm_sample2)
            self.gpm_sample1 = pynvml.nvmlGpmSampleAlloc()
            self.gpm_sample2 = pynvml.nvmlGpmSampleAlloc()
            pynvml.nvmlGpmSampleGet(self.handle, self.gpm_sample2)

    def close(self):
        """Close all resources"""
        if self.gpm_sample1:
            pynvml.nvmlGpmSampleFree(self.gpm_sample1)
        if self.gpm_sample2:
            pynvml.nvmlGpmSampleFree(self.gpm_sample2)
        pynvml.nvmlShutdown()

    def __call__(self, ts=0):
        return self.read_strategy(ts)

    def _read_standard(self, ts=0):
        """Read data using device access. Not supported on MIG."""
        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return PerfMetrics(ts, util.gpu, 100.0 * mem.used / mem.total, mem.used // (1024**2))

    def _read_mig_gpm(self, ts=0):
        """Read data using the explicit MIG device handle and GPM counters."""
        # GPU
        gpu_perc = 0.0
        try:
            pynvml.nvmlGpmSampleGet(self.handle, self.gpm_sample1)
            metrics_get = pynvml.c_nvmlGpmMetricsGet_t()
            metrics_get.version = pynvml.NVML_GPM_METRICS_GET_VERSION
            metrics_get.numMetrics = 1
            metrics_get.sample1 = self.gpm_sample2
            metrics_get.sample2 = self.gpm_sample1
            metrics_get.metrics[0].metricId = pynvml.NVML_GPM_METRIC_SM_UTIL
            pynvml.nvmlGpmMetricsGet(metrics_get)
            if metrics_get.metrics[0].nvmlReturn == 0:
                gpu_perc = float(metrics_get.metrics[0].value)
            self.gpm_sample1, self.gpm_sample2 = self.gpm_sample2, self.gpm_sample1
        except Exception:
            gpu_perc = 0.0

        # VRAM
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            vram_mib = mem.used // (1024**2)
            vram_perc = 100 * mem.used / mem.total
        except Exception:
            vram_mib = 0
            vram_perc = 0.0

        return PerfMetrics(ts, gpu_perc, vram_perc, vram_mib)


class PerfMonitor:
    """Monitors GPU/VRAM utilization."""

    def __init__(self):
        self.output_queue = mp.SimpleQueue()
        self.stop_event = mp.Event()
        self.worker = None

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def start(self, period=1.0):
        """Start performance recording."""
        self.worker = mp.Process(
            target=PerfMonitor._worker,
            args=(self, period),
        )
        self.worker.start()

    def stop(self):
        """Stop performance recording and return data."""
        if not self.worker:
            return []
        self.stop_event.set()
        result = self.output_queue.get()
        self.worker.join()
        self.worker = None
        return result

    def _worker(self, period=1.0, device=0):
        if period <= 0:
            raise ValueError('Period must be larger than 0.')
        records = []
        reader = PerfReader(device)

        try:
            t_start = time.perf_counter()
            t_next = t_start + period
            while True:
                ts = time.perf_counter() - t_start
                met = reader(ts)
                records.append(met)
                if self.stop_event and self.stop_event.is_set():
                    break
                delta = t_next - time.perf_counter()
                t_next += period
                if delta > 0:
                    time.sleep(delta)
        except KeyboardInterrupt:
            pass
        finally:
            reader.close()

        self.output_queue.put(records)

    @staticmethod
    def get_stats(result, field):
        """Returns basic statistics."""
        try:
            data = [getattr(x, field) for x in result]
            stats = PerfStats(
                stat.mean(data), stat.stdev(data), stat.median(data), min(data), max(data)
            )
            return stats
        except (KeyError, stat.StatisticsError):
            return None


if __name__ == '__main__':
    monitor = PerfMonitor()
    monitor.start(0.2)
    time.sleep(5)
    res = monitor.stop()
    # for x in res[:10]:
    #     print(x)
    for k in ('gpu', 'mem'):
        print(k, monitor.get_stats(res, k))
