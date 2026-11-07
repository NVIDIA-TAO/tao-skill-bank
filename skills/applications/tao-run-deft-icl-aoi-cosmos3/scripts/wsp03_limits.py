# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate cgroup-v2 RAM cap and whole-account retained-storage watchdog.

The hard memory cap covers the trusted host supervisor and every owned native
container. Storage is checked at admission and periodically; its sampled nature
and possible between-sample overshoot are recorded, never called a disk quota.
"""
from __future__ import annotations
import os,subprocess,threading,time,uuid
from contextlib import contextmanager
from pathlib import Path
from wsp03_io import json_write

class AggregateLimits:
    def __init__(self,output,*,roots=(),accounts=None,memory_bytes=128*1024**3,storage_bytes=250*1024**3,poll_seconds=1):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.accounts={name:[Path(p).resolve() for p in paths] for name,paths in (accounts or {"default":roots}).items()}
        self.reservations={};self.reservation_lock=threading.RLock();self.account_peaks={k:0 for k in self.accounts};self.scan_peak_seconds=0;self.cancel_seconds=None
        self.memory_bytes=memory_bytes;self.storage_bytes=storage_bytes;self.poll_seconds=poll_seconds
        self.slice='wsp03'+uuid.uuid4().hex+'.slice';self.group=Path('/sys/fs/cgroup')/self.slice
        self.error=None;self.live_jobs={};self.jobs_lock=threading.Lock();self.first_cancel_request_seconds=None;self.callback_dispatch_seconds=None;self.violation_detected_at=None;self.stop=threading.Event();self.peak_storage=0;self.joined={};self.thread=None
    def enter(self):
        if subprocess.check_output(['docker','info','--format','{{.CgroupDriver}}'],text=True).strip()!='systemd':raise RuntimeError('systemd cgroup-v2 driver required')
        subprocess.run(['sudo','-n','systemctl','start',self.slice],check=True)
        subprocess.run(['sudo','-n','systemctl','set-property','--runtime',self.slice,'MemoryMax='+str(self.memory_bytes),'MemorySwapMax=0','TasksMax=1024'],check=True)
        if int((self.group/'memory.max').read_text())!=self.memory_bytes or (self.group/'memory.swap.max').read_text().strip()!='0':raise RuntimeError('aggregate hard memory cap did not bind')
        subprocess.run(['sudo','-n','mkdir',str(self.group/'host')],check=True)
        self.join(os.getpid());self.check()
        self.thread=threading.Thread(target=self.watch,daemon=True);self.thread.start();return self
    def join(self,pid):
        if pid!=os.getpid():raise ValueError('only the trusted current supervisor may join directly')
        old=Path('/proc/'+str(pid)+'/cgroup').read_text().strip().split('::',1)[1]
        self.joined[pid]=old
        subprocess.run(['sudo','-n','tee',str(self.group/'host/cgroup.procs')],input=str(pid),text=True,stdout=subprocess.DEVNULL,check=True)
        if Path('/proc/'+str(pid)+'/cgroup').read_text().strip().split('::',1)[1]!='/'+self.slice+'/host':raise RuntimeError('host supervisor did not join aggregate cap')
    @staticmethod
    def size_roots(roots):
        # Descriptor-relative lstat throughout: never traverse a child symlink,
        # including a directory replaced between enumeration and open.
        import stat
        seen=set();total=0
        flags=os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW
        def entry(parent,name):
            nonlocal total
            try:info=os.stat(name,dir_fd=parent,follow_symlinks=False)
            except FileNotFoundError:return
            identity=(info.st_dev,info.st_ino)
            if stat.S_ISDIR(info.st_mode):
                try:child=os.open(name,flags,dir_fd=parent)
                except FileNotFoundError:return
                try:
                    opened=os.fstat(child)
                    if (opened.st_dev,opened.st_ino)!=identity:raise RuntimeError('storage directory changed during scan')
                    with os.scandir(child) as entries:names=[item.name for item in entries]
                    for item in names:entry(child,item)
                finally:os.close(child)
            elif identity not in seen:
                seen.add(identity);total+=info.st_size
        for root in roots:
            root=Path(root).absolute()
            if root==Path('/'):raise ValueError('whole host storage scan forbidden')
            fd=os.open('/',flags)
            try:
                for part in root.parts[1:-1]:
                    newer=os.open(part,flags,dir_fd=fd);os.close(fd);fd=newer
                entry(fd,root.name)
            except FileNotFoundError:pass
            finally:os.close(fd)
        return total
    def add_roots(self,account,paths):
        if account not in self.accounts:raise ValueError('unknown retained-storage account')
        with self.reservation_lock:
            self.accounts[account]=list(dict.fromkeys([*self.accounts[account],*(Path(p).absolute() for p in paths)]))
        self.check()
    def retained_bytes(self,account='default'):
        return self.size_roots(self.accounts[account])
    def available(self,account='default'):
        self.check()
        return self.storage_bytes-self.retained_bytes(account)-sum(v[1] for v in self.reservations.values() if v[0]==account)
    @contextmanager
    def reserve(self,amount,label,account='default'):
        if type(amount)is not int or amount<0:raise ValueError('nonnegative pre-write byte reservation required')
        with self.reservation_lock:
            if amount>self.available(account):raise RuntimeError('known large write does not fit retained storage budget')
            token=uuid.uuid4().hex;self.reservations[token]=(account,amount,label)
        try:yield
        finally:
            with self.reservation_lock:self.reservations.pop(token)
            self.check()
    def check(self):
        if self.error:raise RuntimeError(self.error)
        started=time.monotonic();values={name:self.retained_bytes(name) for name in self.accounts};self.scan_peak_seconds=max(self.scan_peak_seconds,time.monotonic()-started)
        for name,used in values.items():
            self.account_peaks[name]=max(self.account_peaks[name],used)
            if used>self.storage_bytes:
                self.error='retained storage limit exceeded for account '+name
                raise RuntimeError(self.error)
        used=self.size_roots({p for roots in self.accounts.values() for p in roots});self.peak_storage=max(self.peak_storage,used)
        if not (self.group/'memory.max').exists() or int((self.group/'memory.max').read_text())!=self.memory_bytes:raise RuntimeError('aggregate memory enforcement lost')
        return used
    def register_job(self,identity,cancel):
        with self.jobs_lock:
            if identity in self.live_jobs:raise ValueError('duplicate owned job')
            self.live_jobs[identity]=cancel
    def unregister_job(self,identity):
        with self.jobs_lock:self.live_jobs.pop(identity,None)
    def cancel_owned(self):
        started=time.monotonic()
        with self.jobs_lock:callbacks=list(self.live_jobs.values())
        def invoke(callback):
            with self.jobs_lock:
                elapsed=time.monotonic()-(self.violation_detected_at or started)
                self.callback_dispatch_seconds=elapsed if self.callback_dispatch_seconds is None else min(self.callback_dispatch_seconds,elapsed)
            try:callback()
            except BaseException:pass
        threads=[threading.Thread(target=invoke,args=(callback,),daemon=True) for callback in callbacks]
        for thread in threads:thread.start()
        deadline=time.monotonic()+15
        for thread in threads:thread.join(timeout=max(0,deadline-time.monotonic()))
        self.cancel_seconds=max(self.cancel_seconds or 0,time.monotonic()-started)
    def note_cancel_request(self):
        if self.violation_detected_at is None:return
        elapsed=time.monotonic()-self.violation_detected_at
        with self.jobs_lock:self.first_cancel_request_seconds=elapsed if self.first_cancel_request_seconds is None else min(self.first_cancel_request_seconds,elapsed)
    def watch(self):
        while not self.stop.wait(self.poll_seconds):
            try:self.check()
            except BaseException as error:
                self.error=str(error);self.violation_detected_at=time.monotonic()
                self.cancel_owned();self.stop.set()
    def close(self):
        self.stop.set()
        if self.thread:self.thread.join(timeout=5)
        final_scan={'after_owned_jobs_stopped':not bool(self.live_jobs),'values':None}
        try:
            if self.live_jobs:raise RuntimeError('owned jobs still live; post-stop storage coverage unknown')
            values={name:self.retained_bytes(name) for name in self.accounts}
            for name,used in values.items():self.account_peaks[name]=max(self.account_peaks[name],used)
            self.peak_storage=max(self.peak_storage,self.size_roots({p for roots in self.accounts.values() for p in roots}))
            final_scan['values']=values
        except BaseException as error:final_scan['unknown_reason']=str(error)
        receipt={'final_nofollow_storage_scan':final_scan,'slice':self.slice,'memory_max_bytes':self.memory_bytes,'memory_peak_bytes':int((self.group/'memory.peak').read_text()) if (self.group/'memory.peak').exists() else None,'memory_events':(self.group/'memory.events').read_text() if (self.group/'memory.events').exists() else None,'retained_peak_sampled_bytes':self.peak_storage,'retained_limit_bytes_per_account':self.storage_bytes,'account_roots':{k:list(map(str,v)) for k,v in self.accounts.items()},'account_peak_bytes':self.account_peaks,'scan_peak_seconds':self.scan_peak_seconds,'cancel_seconds':self.cancel_seconds,'detection_to_first_cancel_request_seconds':self.first_cancel_request_seconds,'detection_to_callback_dispatch_seconds':self.callback_dispatch_seconds,'live_jobs_at_close':list(self.live_jobs),'account_overshoot_bytes':{k:max(0,v-self.storage_bytes) for k,v in self.account_peaks.items()},'retained_measurement':'logical file st_size plus symlink text; nofollow admission/sampled watchdog, not allocated blocks or filesystem quota','poll_seconds':self.poll_seconds,'error':self.error}
        errors=[]
        for pid,old in self.joined.items():
            try:subprocess.run(['sudo','-n','tee',str(Path('/sys/fs/cgroup')/old.lstrip('/')/'cgroup.procs')],input=str(pid),text=True,stdout=subprocess.DEVNULL,check=True)
            except BaseException as error:errors.append({'stage':'restore_supervisor','error_type':type(error).__name__})
        # Never stop a slice while its trusted supervisor could still reside in it.
        if not errors:
            for command in (['sudo','-n','rmdir',str(self.group/'host')],['sudo','-n','systemctl','stop',self.slice]):
                try:subprocess.run(command,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
                except BaseException as error:errors.append({'stage':'release_slice','error_type':type(error).__name__})
        receipt['cleanup_errors']=errors
        json_write(self.output/'aggregate-resources.json',receipt)
        if errors:raise RuntimeError('aggregate resource cleanup incomplete; preserved receipt')
        return receipt
