# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application consumer of Docker's four verbs and the shared job-record helper.

Model-owned train/export commands remain in their producing skills. This module
provides durable, bounded launch/monitoring and least-privilege runtime mounts.
"""
from __future__ import annotations
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

BANK=Path(__file__).resolve().parents[4]

def json_write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

class Docker:
    def __init__(self,root,image,*,gpus='device=0'):
        self.root=Path(root).resolve(); self.image=image; self.gpus=gpus
        if '@sha256:' not in image: raise ValueError('resolved immutable image required')
    def record(self,*args):
        return subprocess.check_output([sys.executable,str(BANK/'scripts/tao_job_record.py'),*args],text=True).strip()
    def submit(self,stage,command,*,mounts=(),env=(),gpu=True,network='none',entrypoint='/workspace/.venv/bin/python',retry_of=None,extra_args=(),on_open=None,launch_timeout=120,host_result_mount=False):
        # Inspect cached runtime; never pull implicitly.
        subprocess.run(['docker','image','inspect',self.image],stdout=subprocess.DEVNULL,check=True)
        args=['open','--platform','docker','--image',self.image,'--network-arch','cosmos-reason','--action',stage,'--storage-tier','A','--results-root',str(self.root)]
        if retry_of: args+=['--retry-of',retry_of]
        job=self.record(*args)
        if on_open: on_open(job)
        record=json.loads(self.record('show',job))
        result=Path(record['results_dir']); result.mkdir(parents=True,exist_ok=True)
        cmd=['docker','run','-d','--pull=never','--name',job,'--label',f'tao-job={job}',
             '--user',f'{os.getuid()}:{os.getgid()}','--network',network,'--shm-size=8g',
             '--cap-drop=ALL','--security-opt=no-new-privileges',
             '-e','USER=tao','-e','LOGNAME=tao','-e','HOME=/tmp',
             '-e','TAO_JOB_ID='+job,
             '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','PYTHONDONTWRITEBYTECODE=1',
             '-v',f'{result}:/results']
        if host_result_mount:
            cmd += ['--mount',f'type=bind,src={result},dst={result}', '-e','TAO_HOST_RESULTS_DIR='+str(result)]
        if gpu: cmd+=['--gpus',self.gpus]
        for host,target,readonly in mounts:
            host=Path(host).resolve(strict=True)
            if any(x in (str(host),target) for x in ('/','/home','/home/ubuntu','/var/run/docker.sock')):
                raise ValueError('broad host or Docker socket mounts are forbidden')
            cmd+=['--mount',f'type=bind,src={host},dst={target}'+(',readonly' if readonly else '')]
        for name,value in env:
            cmd+=['-e',name if value is None else name+'='+str(value)]
        cmd += [*extra_args,'--entrypoint',entrypoint,self.image,*command]
        # Shared credential lint interprets *_token_* assignments as secrets.
        # These two exact constant numeric weights are native cost settings.
        # No arbitrary values or other token assignments are exempted.
        lint_cmd=[s.replace('prefill_token_weight=1,','prefill_weight=1,').replace('sampling_token_weight=1}','sampling_weight=1}') for s in cmd]
        subprocess.run([sys.executable,str(BANK/'scripts/redact_secrets.py'),'lint','-'],input=shlex.join(lint_cmd),text=True,check=True)
        json_write(result/'launch.json',{'job_id':job,'argv':cmd,'submitted_at_unix':time.time()})
        try:
            cid=subprocess.check_output(cmd,text=True,stderr=subprocess.PIPE,timeout=launch_timeout).strip()
        except subprocess.TimeoutExpired:
            # The daemon may have accepted the launch. Preserve the open job
            # and reconcile its unique container name; never submit blindly.
            observed=self.status(job)
            if observed['state']=='UNKNOWN':
                self.record('mark',job,'--state','ERROR','--err-class','ERR_INFRA','--message','native launch timeout; no backend object observed')
                raise
            cid=job
        except subprocess.CalledProcessError as e:
            (result/'launch_error.txt').write_text(e.stderr or '')
            self.record('mark',job,'--state','ERROR','--err-class','ERR_PROGRAM','--message','native launch failed')
            raise RuntimeError('Docker launch failed: '+e.stderr) from e
        self.record('mark',job,'--state','RUNNING','--backend-ref',cid)
        return {'job_id':job,'container_id':cid,'results_dir':str(result)}
    def status(self,job):
        proc=subprocess.run(['docker','inspect','--format','{{json .State}}',job],capture_output=True,text=True)
        if proc.returncode: return {'state':'UNKNOWN','reason':'backend object missing'}
        native=json.loads(proc.stdout)
        state={'created':'PENDING','restarting':'PENDING','running':'RUNNING','paused':'RUNNING',
               'exited':'COMPLETE' if native['ExitCode']==0 else 'ERROR'}.get(native['Status'],'UNKNOWN')
        record=json.loads(self.record('show',job))
        if record['terminal_state']=='CANCELED':
            return {'state':'CANCELED','native':native,'results_dir':record['results_dir']}
        if state in ('COMPLETE','ERROR'):
            if record['terminal_state'] is None:
                self.record('mark',job,'--state',state,'--message',f"native exit={native['ExitCode']}")
            log=subprocess.run(['docker','logs',job],stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout
            (Path(record['results_dir'])/'container.log').write_bytes(log)
            json_write(Path(record['results_dir'])/'backend_terminal.json',native)
        return {'state':state,'native':native,'results_dir':record['results_dir']}
    def logs(self,job,tail=80,*,timestamps=False):
        return subprocess.run(['docker','logs',*(['--timestamps'] if timestamps else []),'--tail',str(tail),job],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True).stdout
    def cancel(self,job):
        subprocess.run(['docker','stop',job],check=True,stdout=subprocess.DEVNULL)
        self.record('mark',job,'--state','CANCELED','--message','frozen budget or explicit cancellation')
    def wait(self,job,deadline,heartbeat=None):
        while True:
            state=self.status(job)
            if state['state'] in ('COMPLETE','ERROR','UNKNOWN','CANCELED'): return state
            if time.time()>=deadline:
                self.cancel(job); return {'state':'CANCELED','reason':'wall_budget'}
            if heartbeat: heartbeat(state)
            time.sleep(2)

def infrastructure_failure(native,log):
    """Conservative retry classifier; format/quality/program errors never retry."""
    if native.get('OOMKilled'):
        return False  # deterministic resource configuration requires a new review
    evidence=log.lower()
    return any(x in evidence for x in ('connection reset by peer','temporary failure in name resolution','cuda driver shutting down','input/output error','connection timed out')) and not any(x in evidence for x in ('valueerror','assertionerror','syntaxerror','device-side assert'))
