"""
perf_monitor.py
Record GPU and VRAM utilization across VM and MIG-enabled Kubernetes clusters.
Author: Grzegorz Szwoch, KSM PG, 2026.
"""

import multiprocessing as mp
import threading
import queue
import time
from collections import namedtuple
import pynvml

try:
    from prometheus_remote_writer import RemoteWriter
except ImportError:
    RemoteWriter = None

PerfMetrics = namedtuple('PerfMetrics', 'ts, gpu, mem_mb, mem_total', defaults=(0, 0, 0, 0))
PerfMetrics.mem_perc = property(lambda x: 100 * x.mem_mb / x.mem_total if x.mem_total > 0 else 0)

PerfStats = namedtuple(
    'PerfStats', 'ts, gpu_mean, gpu_max, mem_mean, mem_max, mem_total', defaults=(0, 0, 0, 0, 0, 0)
)
PerfStats.mem_mean_perc = property(
    lambda x: 100 * x.mem_mean / x.mem_total if x.mem_total > 0 else 0
)
PerfStats.mem_max_perc = property(lambda x: 100 * x.mem_max / x.mem_total if x.mem_total > 0 else 0)

# Name of the constant background collector
GLOBAL = 'global'


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
            pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.mem_total = mem.total // (1024**2)
            self.read_strategy = self._read_standard
        except pynvml.NVMLError_NotSupported:
            try:
                mig_count = pynvml.nvmlDeviceGetMaxMigDeviceCount(handle)
                for i in range(mig_count):
                    try:
                        mig_handle = pynvml.nvmlDeviceGetMigDeviceHandleByIndex(handle, i)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(mig_handle)
                        self.handle = mig_handle
                        self.is_mig = True
                        self.mem_total = mem.total // (1024**2)
                        break
                    except pynvml.NVMLError:
                        continue
            except Exception:
                raise RuntimeError('No valid measurement method was found.')

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
        return self.read_strategy()

    def _read_standard(self):
        """Read data using device access. Not supported on MIG."""
        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        ts = time.time()
        return PerfMetrics(ts, util.gpu, mem.used // (1024**2), mem.total // (1024**2))

    def _read_mig_gpm(self):
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
            mem_total = mem.total // (1024**2)
        except Exception:
            vram_mib = mem_total = 0

        ts = time.time()
        return PerfMetrics(ts, gpu_perc, vram_mib, mem_total)


class PerfCollector:
    """Collects performance statistics."""

    def __init__(self, alpha, init_count=0):
        self.alpha = alpha
        self.init_count = init_count
        self.reset()

    def reset(self):
        """Clear collected data."""
        self.gpu_max = None
        self.gpu_mean = None
        self.mem_max = None
        self.mem_mean = None
        self.mem_total = 0
        self.count = 0
        self.init_data = []

    def close(self):
        """Close all resources."""
        pass

    def update(self, data: PerfMetrics):
        """Update the statistics with new data."""
        self.mem_total = data.mem_total
        if self.count > self.init_count:
            self.gpu_max = max(self.gpu_max, data.gpu)
            self.mem_max = max(self.mem_max, data.mem_mb)
            self.gpu_mean = self.alpha * self.gpu_mean + (1 - self.alpha) * data.gpu
            self.mem_mean = self.alpha * self.mem_mean + (1 - self.alpha) * data.mem_mb
        else:
            self.init_data.append(data)
        if self.count == self.init_count:
            stats = self._from_init()
            self.gpu_mean = stats.gpu_mean
            self.gpu_max = stats.gpu_max
            self.mem_mean = stats.mem_mean
            self.mem_max = stats.mem_max
            self.init_data = []
        self.count += 1

    def _from_init(self):
        if not len(self.init_data):
            return PerfStats()
        mean_gpu = sum(x.gpu for x in self.init_data) / len(self.init_data)
        max_gpu = max(x.gpu for x in self.init_data)
        mean_mem = sum(x.mem_mb for x in self.init_data) / len(self.init_data)
        max_mem = max(x.mem_mb for x in self.init_data)
        ts = time.time()
        return PerfStats(ts, mean_gpu, max_gpu, mean_mem, max_mem, self.mem_total)

    def get(self):
        """Return the collected data."""
        if self.count > self.init_count:
            ts = time.time()
            return PerfStats(
                ts, self.gpu_mean, self.gpu_max, self.mem_mean, self.mem_max, self.mem_total
            )
        else:
            return self._from_init()


class ConstantPerfCollector(PerfCollector):
    """Collects performance statistics and aggregate them within the defined period."""

    def __init__(
        self,
        alpha,
        init_count=0,
        readout_period=60,
        tag='app',
        job_name='job',
        mimir_url=None,
        mimir_user=None,
        mimir_passwd=None,
    ):
        """Create a collector that constantly captures the statistics
        and optionally sends them to a Mimir endpoint.
        alpha: exponential aeraging coefficient.
        init_count: number of readouts used in the initialization stage.
        readout_period: period of statistics collection in seconds.
        tag: unique tag used in Mimir metrics.
        job_name: unique name identifying the source.
        mimir_url: URL of a Mimir endpoint (typically: "https://some.server.com/api/v1/push").
        miir_user: username for Mimir authentication.
        mimir_pass: passwprd for Mimir authentication.
        """
        super().__init__(alpha, init_count)
        self.readout_period = readout_period
        self.tag = tag
        self.job_name = job_name
        self.last_send_ts = None
        self.last_stats = None
        self.mimir_url = mimir_url
        self.mimir_user = mimir_user
        self.mimir_passwd = mimir_passwd
        self.last_status = None
        self.status_lock = threading.Lock()
        if mimir_url:
            self.queue = queue.SimpleQueue()
            self.thread = threading.Thread(target=self._worker)
            self.thread.start()
        else:
            self.thread = self.queue = None

    def reset(self):
        super().reset()
        self.last_stats = None
        self.last_send_ts = None

    def close(self):
        if self.thread and self.thread.is_alive():
            self.queue.put(None)
            self.thread.join()

    def update(self, data):
        super().update(data)
        if self.last_send_ts is not None:
            if data.ts >= self.last_send_ts + self.readout_period:
                self.last_stats = super().get()
                super().reset()
                if self.thread and self.thread.is_alive():
                    self.queue.put(self.last_stats)
                self.last_send_ts += self.readout_period
        else:
            self.last_send_ts = data.ts

    def get(self):
        return self.last_stats

    def status(self):
        with self.status_lock:
            last_status = self.last_status
        return {
            'mimir_thread_running': self.thread is not None and self.thread.is_alive(),
            'mimir_last_status': last_status,
        }

    def _worker(self):
        """Worker that sends the collected statistics to MONITORING_URL."""
        if not self.mimir_url or RemoteWriter is None:
            return
        headers = {'X-Scope-OrgID': self.mimir_user} if self.mimir_user else None
        auth = (
            {'username': self.mimir_user, 'password': self.mimir_passwd}
            if self.mimir_passwd
            else None
        )
        writer = RemoteWriter(url=self.mimir_url, headers=headers, auth=auth)

        # Test connection
        dummy_payload = [
            {
                'metric': {'__name__': 'conn_test_ping', 'job': 'ping'},
                'values': [],
                'timestamps': [],
            }
        ]
        try:
            res = writer.send(dummy_payload)
            status = res.last_response.status_code
        except Exception as _:
            status = None
        if status != 200:
            print('Connection to Mimir failed with code', status)
            # return  # no connection

        while True:
            stats = self.queue.get()
            if stats is None:
                break
            timestamp_ms = int(1000 * stats.ts)
            payload = [
                {
                    'metric': {'__name__': f'{self.tag}_gpu_max', 'job': self.job_name},
                    'values': [stats.gpu_max],
                    'timestamps': [timestamp_ms],
                },
                {
                    'metric': {'__name__': f'{self.tag}_gpu_mean', 'job': self.job_name},
                    'values': [stats.gpu_mean],
                    'timestamps': [timestamp_ms],
                },
                {
                    'metric': {'__name__': f'{self.tag}_vram_max', 'job': self.job_name},
                    'values': [stats.mem_max],
                    'timestamps': [timestamp_ms],
                },
                {
                    'metric': {'__name__': f'{self.tag}_vram_mean', 'job': self.job_name},
                    'values': [stats.mem_mean],
                    'timestamps': [timestamp_ms],
                },
            ]
            try:
                resp = writer.send(payload)
            except RuntimeError as e:
                print('PerfMonitor: error sending data to Prometheus:', e)
            except ValueError as e:
                print('PerfMonitor: error parsing data:', e)
            finally:
                with self.status_lock:
                    if resp:
                        self.last_status = resp.last_response.status_code
                    else:
                        self.last_status = None


class PerfMonitor:
    """Monitors GPU/VRAM utilization with thread-safe on-demand requests."""

    def __init__(self):
        self.worker = None
        self.alpha = 0.90  # tau ≈ 2 s for T = 0.2 s
        self.cmd_queue = mp.Queue()
        self.parent_conn, self.child_conn = mp.Pipe()
        self.lock = threading.Lock()

    def close(self):
        self.stop()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def add(self, _id):
        self.cmd_queue.put(('ADD', _id))

    def remove(self, _id):
        self.cmd_queue.put(('REMOVE', _id))

    def start(
        self,
        period=1.0,
        initialization_time=0,
        tag='app',
        job_name='job',
        constant_update_period=1,
        mimir_url=None,
        mimir_user=None,
        mimir_passwd=None,
    ):
        self.worker = mp.Process(
            target=self._worker,
            args=(
                period,
                initialization_time,
                0,
                self.cmd_queue,
                self.child_conn,
                self.alpha,
                constant_update_period,
                tag,
                job_name,
                mimir_url,
                mimir_user,
                mimir_passwd,
            ),
        )
        self.worker.start()

    def stop(self):
        if not self.worker:
            return
        self.cmd_queue.put(('STOP', None))
        self.worker.join()
        self.worker = None

    def get(self, _id=GLOBAL):
        """Get statistics from the collector '_id'."""
        if not self.worker:
            return None
        with self.lock:
            self.cmd_queue.put(('GET', _id))
            return self.parent_conn.recv()

    def status(self, _id=GLOBAL):
        """Get status of the monitor."""
        if not self.worker or not self.worker.is_alive():
            return {'active': False}
        with self.lock:
            self.cmd_queue.put(('STATUS', _id))
            st = self.parent_conn.recv()
            res = {'active': True}
            res.update(st)
            return res

    @staticmethod
    def _worker(
        period,
        initialization_time,
        device,
        cmd_queue,
        pipe_conn,
        alpha,
        constant_update_period,
        tag,
        job_name,
        mimir_url=None,
        mimir_user=None,
        mimir_passwd=None,
    ):
        if period <= 0:
            raise ValueError('Period must be larger than 0.')

        reader = PerfReader(device)
        collectors = {}
        init_count = round(initialization_time / period)

        if constant_update_period > 0:
            collectors[GLOBAL] = ConstantPerfCollector(
                alpha,
                init_count,
                constant_update_period,
                tag,
                job_name,
                mimir_url,
                mimir_user,
                mimir_passwd,
            )

        try:
            t_start = time.perf_counter()
            t_next = t_start + period
            while True:
                while not cmd_queue.empty():
                    cmd, val = cmd_queue.get_nowait()
                    if cmd == 'ADD':
                        if not collectors:
                            reader.reset()
                        collectors[val] = PerfCollector(alpha, init_count)
                    elif cmd == 'REMOVE':
                        if val in collectors:
                            collectors[val].close()
                            collectors.pop(val, None)
                    elif cmd == 'GET':
                        col = collectors.get(val)
                        pipe_conn.send(col.get() if col else None)
                    elif cmd == 'STATUS':
                        if (
                            val == GLOBAL
                            and GLOBAL in collectors
                            and isinstance(collectors[GLOBAL], ConstantPerfCollector)
                        ):
                            pipe_conn.send(collectors[GLOBAL].status())
                        else:
                            pipe_conn.send({})
                    elif cmd == 'STOP':
                        for col in collectors.values():
                            col.close()
                        return

                if collectors:
                    met = reader()
                    for col in collectors.values():
                        col.update(met)

                delta = t_next - time.perf_counter()
                t_next += period
                if delta > 0:
                    time.sleep(delta)
        except KeyboardInterrupt:
            pass
        finally:
            reader.close()


if __name__ == '__main__':
    monitor = PerfMonitor()
    monitor.start(0.2, initialization_time=10.0, constant_update_period=10)

    if 1:
        time.sleep(2)
        try:
            while True:
                time.sleep(10)
                print(monitor.get(GLOBAL))
        except KeyboardInterrupt:
            monitor.stop()
            raise SystemExit

    monitor.add(0)
    time.sleep(2.3)
    # res = monitor.get(0)
    # print(res)
    time.sleep(2.7)
    res = monitor.get(0)
    monitor.stop()
    print(res)
