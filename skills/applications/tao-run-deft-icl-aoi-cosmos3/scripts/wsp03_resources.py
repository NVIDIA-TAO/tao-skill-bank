# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exclusive host lease and honest resource observations; never kills foreign work."""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import resource
import re
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


def process_identity(pid):
    """Read identity without command lines, environment, or process contents."""
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        group=Path(f'/proc/{pid}/cgroup').read_text().strip()
        second=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        if fields[19]!=second[19]:return None
        return {'pid':int(pid),'ppid':int(second[1]),'birth':fields[19],'cgroup':group}
    except (OSError,ValueError,IndexError):return None


def process_presence(pid):
    try:Path(f'/proc/{pid}').stat();return 'present'
    except FileNotFoundError:return 'gone'
    except OSError:return 'unknown'


def container_snapshot(groups):
    result=[]
    if not groups:return result
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        identity=process_identity(int(path.name))
        if identity and identity['cgroup'] in groups:result.append(identity)
    return result


class GPULease:
    def __init__(self, path, owner, *, gpu='0', query=None, allowed_pids=(), poll_seconds=.1, identity=None, snapshot=None, presence=None):
        self.path, self.owner = Path(path), owner
        self.query = query or (lambda: gpu_processes(gpu))
        self.gpu, self.allowed = str(gpu), {}
        self.identity=identity or process_identity
        self.snapshot=snapshot or container_snapshot
        self.presence=presence or process_presence
        self.allowed_groups={}
        self.registration_lock=threading.RLock()
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

    def allow_pid(self, pid, *, container_id=None):
        record=self.identity(int(pid))
        if record is None:raise LeaseError('native container PID identity unavailable before gate')
        group=record['cgroup']
        matches=sorted(set(re.findall(r'(?<![a-f0-9])[a-f0-9]{64}(?![a-f0-9])',group)))
        if container_id is not None:
            if not re.fullmatch(r'[a-f0-9]{64}',container_id) or container_id not in matches:
                raise LeaseError('native container ID does not match exact process cgroup')
        elif len(matches)==1:container_id=matches[0]
        if not container_id or group in ('','0::/'):
            raise LeaseError('exact non-root container cgroup required; no host/daemon ancestry grants')
        with self.registration_lock:
            self.allowed[int(pid)]=record
            self.allowed_groups[group]={'container_id':container_id,'root':record,'registered_at':time.time()}
            if hasattr(self,'archive'):
                append_event(Path(str(self.archive)+'.lineage.jsonl'),{'event':'register_container','identity':record,'container_id':container_id,'time':time.time()})
        return record

    def revoke_pid(self, pid):
        with self.registration_lock:
            removed=self.allowed.pop(int(pid),None)
            if removed is None:return
            group=removed['cgroup']
            if not any(row['cgroup']==group for row in self.allowed.values()):self.allowed_groups.pop(group,None)
            if hasattr(self,'archive'):append_event(Path(str(self.archive)+'.lineage.jsonl'),{'event':'revoke_container_root','identity':removed,'time':time.time()})

    def sample(self):
        # Capture exact cgroup identities before NVML can report a process that
        # exits while its command is in flight. Evidence is scoped to this poll;
        # no PID-only historical cache can authorize a later incarnation.
        before=time.monotonic();before_wall=time.time()
        with self.registration_lock:groups=dict(self.allowed_groups)
        snapshot={p['pid']:p for p in self.snapshot(set(groups)) if p['cgroup'] in groups}
        rows=self.query();after=time.monotonic();proofs=[];unresolved=[]
        for row in rows:
            live=self.identity(row['pid']);previous=snapshot.get(row['pid'])
            if live is not None and live['cgroup'] in groups:
                # A live PID changing incarnation since the snapshot is still
                # accepted only by its currently bound exact container cgroup.
                proof={'pid':row['pid'],'owned':True,'method':'live_exact_container_cgroup','identity':live,
                       'container_id':groups[live['cgroup']]['container_id']}
            elif live is None and previous is not None and self.presence(row['pid'])=='gone':
                proof={'pid':row['pid'],'owned':True,'method':'same_poll_owned_prequery_snapshot','identity':previous,
                       'container_id':groups[previous['cgroup']]['container_id'],'evidence_window_start':before_wall,'evidence_window_seconds':after-before}
            else:
                proof={'pid':row['pid'],'owned':False,'method':'unresolved_vanished_pid' if live is None else 'different_live_cgroup','identity':live}
                unresolved.append(proof)
            proofs.append(proof)
        observation={'time':time.time(),'monotonic':after,'query_started_monotonic':before,'processes':rows,
            'owned_prequery_snapshot':list(snapshot.values()),'ownership_proofs':proofs}
        append_event(self.trace,observation)
        self.sample_count+=1;self.max_gap=max(self.max_gap,after-self.previous_sample);self.previous_sample=after
        if any(p['memory_bytes'] is None for p in rows):self.memory_unknown_samples+=1
        else:self.memory_peak=max(self.memory_peak or 0,sum(p['memory_bytes'] for p in rows))
        if unresolved:self.violation={'reason':'unproven_gpu_ownership','unresolved':unresolved,'observation':observation}

    def _monitor(self):
        while not self.stop.wait(self.poll_seconds):
            try:
                self.sample()
                if self.violation:return
            except BaseException as error:
                self.violation={'reason':'gpu_monitor_unavailable','error':type(error).__name__,'time':time.time()}
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
