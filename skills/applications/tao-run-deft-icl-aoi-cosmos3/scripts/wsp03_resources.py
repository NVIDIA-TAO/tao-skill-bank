# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exclusive host lease and honest resource observations; never kills foreign work."""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import resource
import subprocess
import threading
import time
import uuid
import xml.etree.ElementTree as ET

from wsp03_io import json_write


class LeaseError(RuntimeError):
    pass


def gpu_processes(gpu='0'):
    result = subprocess.run(['nvidia-smi', '-i', str(gpu), '-q', '-x'], text=True, capture_output=True, timeout=10, check=True)
    root = ET.fromstring(result.stdout)
    found = []
    for proc in root.findall('.//process_info'):
        memory = proc.findtext('used_memory') or ''
        found.append({'pid': int(proc.findtext('pid')), 'type': proc.findtext('type'),
                      'name': proc.findtext('process_name'),
                      'memory_bytes': int(memory.split()[0]) * 1024**2 if memory.endswith(' MiB') else None})
    return found


def append_event(path, record):
    with Path(path).open('a') as f:
        f.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def descendant(pid, roots):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid in roots:
            return True
        seen.add(pid)
        try:
            # comm may include spaces/parentheses. Fields after final ')' begin state,ppid.
            pid = int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            return False
    return pid in roots


class GPULease:
    def __init__(self, path, owner, *, gpu='0', query=None, allowed_pids=(), poll_seconds=.1):
        self.path, self.owner = Path(path), owner
        self.query = query or (lambda: gpu_processes(gpu))
        self.gpu, self.allowed = str(gpu), {}
        for pid in allowed_pids: self.allow_pid(pid)
        self.poll_seconds = poll_seconds
        self.stop = threading.Event()
        self.violation = None
        self.observations = []
        self.thread = None
        self.fd = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = self.path.open('a+')
        try:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise LeaseError('GPU lease already held') from error
            processes = self.query()
            if processes:
                raise LeaseError('GPU has existing owner/process; no launch')
            self.started_monotonic, self.started_at = time.monotonic(), time.time()
            self.receipt = self.path.with_name(self.path.name + '.owner.json')
            self.archive = self.path.with_name(self.path.name + '.' + uuid.uuid4().hex)
            self.trace = Path(str(self.archive) + '.observations.jsonl')
            self.previous_sample = self.started_monotonic
            self.max_gap = 0
            self.sample_count = 0
            self.memory_unknown_samples = 0
            self.memory_peak = None
            json_write(self.receipt, {'owner': self.owner, 'pid': os.getpid(), 'gpu': self.gpu,
                                     'started_at': self.started_at, 'exclusive_requested': True, 'state': 'held'})
            self.thread = threading.Thread(target=self._monitor, daemon=True)
            self.thread.start()
            return self
        except BaseException:
            self.fd.close()
            self.fd = None
            raise

    def allow_pid(self, pid):
        pid = int(pid)
        self.allowed[pid] = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]

    def _monitor(self):
        while not self.stop.wait(self.poll_seconds):
            try:
                rows = self.query()
                observation = {'time': time.time(), 'monotonic': time.monotonic(), 'processes': rows}
                append_event(self.trace, observation)
                self.sample_count += 1
                self.max_gap = max(self.max_gap, observation['monotonic'] - self.previous_sample)
                self.previous_sample = observation['monotonic']
                if any(p['memory_bytes'] is None for p in rows): self.memory_unknown_samples += 1
                else: self.memory_peak = max(self.memory_peak or 0, sum(p['memory_bytes'] for p in rows))
                live = set()
                for pid, born in list(self.allowed.items()):
                    try:
                        if Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19] == born: live.add(pid)
                    except OSError: pass
                foreign = [p for p in rows if not descendant(p['pid'], live)]
                if foreign:
                    self.violation = {'reason': 'foreign_gpu_process', 'observation': observation}
                    return
            except BaseException as error:
                self.violation = {'reason': 'gpu_monitor_unavailable', 'error': type(error).__name__, 'time': time.time()}
                return

    def check(self):
        if self.violation:
            raise LeaseError(json.dumps(self.violation))

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=12)
        elapsed = time.monotonic() - self.started_monotonic
        if self.thread and self.thread.is_alive():
            self.violation = {'reason': 'monitor_did_not_exit'}
        self.max_gap = max(self.max_gap, time.monotonic()-self.previous_sample)
        result = {'owner': self.owner, 'pid': os.getpid(), 'gpu': self.gpu, 'state': 'released',
                  'started_at': self.started_at, 'ended_at': time.time(), 'allocated_gpu_seconds': elapsed,
                  'allocation_method': 'one GPU exclusive lease interval, including idle',
                  'exclusive_observation_valid': self.violation is None, 'violation': self.violation,
                  'sampled_process_peak_bytes': self.memory_peak if self.memory_unknown_samples == 0 else None,
                  'known_sample_process_peak_bytes': self.memory_peak, 'unknown_memory_samples': self.memory_unknown_samples,
                  'sample_count': self.sample_count, 'max_poll_gap_seconds': self.max_gap,
                  'kernel_active_seconds': None, 'kernel_active_reason': 'not instrumented; NVML utilization is not kernel activity'}
        try:
            json_write(self.receipt, result)
            json_write(Path(str(self.archive) + '.receipt.json'), result)
        finally:
            if self.fd:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
                self.fd.close()
        if self.violation and exc is None:
            raise LeaseError('GPU exclusivity/monitoring was violated')


def cpu_memory():
    result = {'process_rss_high_water_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              'cgroup_memory_peak_bytes': None}
    try:
        cgroup = Path('/proc/self/cgroup').read_text().split('0::', 1)[1].strip()
        result['cgroup_memory_peak_bytes'] = int((Path('/sys/fs/cgroup') / cgroup.lstrip('/') / 'memory.peak').read_text())
    except (OSError, ValueError, IndexError):
        try:
            result['cgroup_memory_peak_bytes'] = int(Path('/sys/fs/cgroup/memory.peak').read_text())
        except (OSError, ValueError):
            pass
    return result
