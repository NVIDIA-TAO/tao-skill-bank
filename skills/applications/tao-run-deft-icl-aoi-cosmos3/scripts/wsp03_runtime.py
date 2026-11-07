# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted process boundary for WSP03 probe and numeric kernel.

All containers use the existing platform four-verb/job-record consumer. GPU
workers wait behind a startup gate until the host registers their native PID.
The scientific editor is deliberately not launched by this engineering entry.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import uuid
from suite_runtime import Docker
from wsp03_io import json_write
from wsp03_resources import GPULease, gpu_processes
from wsp03_worker import digest, validate_requests

APP=Path(__file__).resolve().parents[1]
PYTHON='/workspace/.venv/bin/python'


def gated_job(runtime, lease, stage, command, mounts, *, deadline, env=(), on_admit=None):
    wrapper='while [ ! -f /results/lease-authorized ]; do sleep 0.05; done; exec '+shlex.join([PYTHON,*command])
    admitted={}
    def opened(job_id):
        record=json.loads(runtime.record('show',job_id))
        admitted.update(job_id=job_id,results_dir=record['results_dir'])
        if on_admit:on_admit(dict(admitted))
    try:
        job=runtime.submit(stage,['-c',wrapper],gpu=True,mounts=mounts,env=env,entrypoint='/bin/sh',on_open=opened,
            extra_args=['--read-only','--tmpfs','/tmp:rw,nosuid,nodev,size=2g','--memory','128g','--pids-limit','512'])
        native=runtime.status(job['job_id'])
        pid=native.get('native',{}).get('Pid',0)
        if not pid:raise RuntimeError('native GPU worker PID unavailable before gate')
        lease.allow_pid(pid);lease.check()
        json_write(Path(job['results_dir'])/'lease-authorized',{'host_pid':pid,'time':time.time(),'lease_owner':lease.owner})
        status=runtime.wait(job['job_id'],deadline=deadline,heartbeat=lambda _:lease.check())
        lease.check()
        if status['state']!='COMPLETE':raise RuntimeError('worker failed: '+json.dumps(status))
        return job
    except BaseException:
        if admitted and runtime.status(admitted['job_id'])['state'] in ('RUNNING','PENDING'):runtime.cancel(admitted['job_id'])
        raise


def validate_probe_inputs(requests, view, expected_ids_sha256):
    view=Path(view).resolve(strict=True)
    for rows in requests:
        validate_requests(rows,'generation','probe')
        if len(rows)!=64:raise ValueError('probe requires exactly64 preregistered queries per mode')
        if digest([r['id'] for r in rows])!=expected_ids_sha256:raise ValueError('frozen ordered probe membership hash mismatch')
    needed=set()
    for rows in requests:
        for row in rows:
            for message in row['messages']:
                for part in message['content']:
                    if part['type']!='image':continue
                    relative=Path(part['image'])
                    if not relative.is_relative_to('/input') or '..' in relative.parts:raise ValueError('invalid request image path')
                    path=view/relative.relative_to('/input')
                    if path.resolve(strict=True)!=path or any(x.is_symlink() for x in [path,*path.parents] if x.is_relative_to(view)):
                        raise ValueError('image symlink/escape forbidden')
                    if path.suffix.lower()!='.png' or not path.is_file():raise ValueError('PNG file required')
                    needed.add(path)
    actual={p for p in view.rglob('*') if p.is_file()}
    if actual!=needed or any(p.is_symlink() for p in view.rglob('*')):raise ValueError('image view must equal exact referenced allowlist')
    return needed


def compute_mode(gpu):
    p=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-gpu=compute_mode','--format=csv,noheader'],check=True,text=True,capture_output=True)
    return p.stdout.strip()


def set_compute_mode(gpu, mode):
    names={'Default':'0','Exclusive_Thread':'1','Prohibited':'2','Exclusive_Process':'3'}
    subprocess.run(['sudo','-n','nvidia-smi','-i',str(gpu),'-c',names.get(mode,mode)],check=True,stdout=subprocess.DEVNULL)


def probe(*, model, zs_requests, icl_requests, input_view, output, image, lease_path, probe_ids_sha256, gpu='0', query_seconds=180):
    """Exactly two admitted batch-one generation passes; failure receipts are durable."""
    output=Path(output).resolve(strict=False)
    if output.exists():raise ValueError('probe output must be new; reconcile previous job before retry')
    model=Path(model).resolve(strict=True);view=Path(input_view).resolve(strict=True)
    request_paths=[Path(zs_requests).resolve(strict=True),Path(icl_requests).resolve(strict=True)]
    requests=[json.loads(p.read_text()) for p in request_paths]
    validate_probe_inputs(requests,view,probe_ids_sha256)
    output.mkdir(parents=True,mode=0o700)
    previous=compute_mode(gpu);started=time.time();jobs=[];runtime=Docker(output/'jobs',image,gpus='device='+str(gpu))
    receipt={'schema':'wsp03-probe/2','status':'running','generation_passes':0,'batch_size':1,'requests_sha256':[digest(r) for r in requests],
        'probe_ids_sha256':probe_ids_sha256,'started_at':started,'scientific_editor_started':False,'prior_compute_mode':previous}
    json_write(output/'probe.json',receipt)
    try:
        with GPULease(lease_path,'wsp03-probe-'+uuid.uuid4().hex,gpu=gpu) as lease:
            changed=False
            try:
                if previous!='Exclusive_Process':set_compute_mode(gpu,'Exclusive_Process');changed=True
                for label,path in zip(('zs','icl'),request_paths):
                    def admitted(job):
                        jobs.append(dict(job,mode=label));receipt.update(jobs=jobs,generation_passes=len(jobs))
                        json_write(output/'probe.json',receipt)
                    gated_job(runtime,lease,'wsp03-probe-'+label,
                        ['/app/scripts/wsp03_worker.py','--model','/model','--requests','/request.json','--output','/results/worker','--mode','generation','--split','probe','--query-seconds',str(query_seconds)],
                        [(APP,'/app',True),(model,'/model',True),(view,'/input',True),(path,'/request.json',True)],
                        deadline=time.time()+64*query_seconds+900,on_admit=admitted)
                receipt['status']='complete'
            finally:
                for job in jobs:
                    state=runtime.status(job['job_id'])
                    if state['state'] in ('RUNNING','PENDING'):runtime.cancel(job['job_id'])
                until=time.monotonic()+10
                while gpu_processes(gpu) and time.monotonic()<until:time.sleep(.1)
                if gpu_processes(gpu):lease.violation={'reason':'GPU not empty at lease close'}
                if changed:set_compute_mode(gpu,previous)
                receipt['restored_compute_mode']=compute_mode(gpu)
                receipt['lease_archive']=str(lease.archive)+'.receipt.json'
    except BaseException as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        passes=[]
        for job in jobs:
            directory=Path(job['results_dir'])/'worker';answers=directory/'raw_answers.jsonl'
            rows=[json.loads(line) for line in answers.read_text().splitlines()] if answers.exists() else []
            times=sorted(r['wall_seconds'] for r in rows)
            passes.append({'mode':job['mode'],'job_id':job['job_id'],'n':len(rows),'forward_wall_seconds':sum(times),
                'mean_query_seconds':sum(times)/len(times) if times else None,
                'p95_query_seconds':times[min(len(times)-1,int(.95*len(times)))] if times else None,
                'input_tokens':sum(r['inference_input_tokens'] for r in rows),'output_tokens':sum(r['inference_output_tokens'] for r in rows),
                'peak_allocated_bytes':max((r['peak_allocated_bytes'] for r in rows),default=None),
                'peak_reserved_bytes':max((r['peak_reserved_bytes'] for r in rows),default=None),
                'runtime_profile':str(directory/'runtime_profile.json'),'raw_answers':str(answers)})
        receipt.update(wall_seconds=time.time()-started,passes=passes,jobs=jobs,generation_passes=len(jobs),ended_at=time.time())
        if receipt['status']=='complete':
            mean=max(p['mean_query_seconds'] for p in passes)
            receipt.update(full_dev_estimate_seconds=mean*2917,explore_estimate_seconds=mean*512)
        json_write(output/'probe.json',receipt)
    return receipt


class IsolatedKernel:
    def __init__(self, *, runtime, base, output, exchange):
        self.runtime=runtime;self.output=Path(output).resolve();self.exchange=Path(exchange).resolve()
        self.output.mkdir(parents=True,exist_ok=True);self.exchange.mkdir(parents=True,exist_ok=True)
        canary=runtime.root/('host-denial-canary-'+uuid.uuid4().hex)
        canary.write_text('synthetic host namespace canary; not a credential')
        self.job=runtime.submit('wsp03-cpu-kernel',['/app/scripts/wsp03_kernel_server.py','--base','/model','--output','/edits','--exchange','/exchange'],
            gpu=False,mounts=[(APP,'/app',True),(base,'/model',True),(self.output,'/edits',False),(self.exchange,'/exchange',False)],
            env=[('NVIDIA_VISIBLE_DEVICES','void'),('WSP_HOST_CANARY',str(canary))],extra_args=['--runtime','runc','--read-only','--tmpfs','/tmp:rw,nosuid,nodev,size=1g','--memory','64g','--cpus','8','--pids-limit','128'])
    def call(self, request, timeout=660):
        identity=uuid.uuid4().hex;input_file=self.exchange/(identity+'.request.json');response=self.exchange/(identity+'.response.json')
        json_write(input_file,request);deadline=time.monotonic()+timeout
        while not response.exists():
            if time.monotonic()>deadline:raise TimeoutError('kernel request uncertain; preserve admitted identity and reconcile')
            status=self.runtime.status(self.job['job_id'])
            if status['state'] not in ('RUNNING','PENDING'):raise RuntimeError('kernel stopped; namespace lost, ledger remains durable')
            time.sleep(.2)
        return json.loads(response.read_text())
    def close(self):
        if self.runtime.status(self.job['job_id'])['state'] in ('RUNNING','PENDING'):self.runtime.cancel(self.job['job_id'])


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('probe')
    for n in ('model','zs-requests','icl-requests','input-view','output','image','lease-path','probe-ids-sha256'):q.add_argument('--'+n,required=True)
    q.add_argument('--gpu',default='0');q.add_argument('--query-seconds',type=int,default=180)
    a=vars(p.parse_args());a.pop('command');print(json.dumps(probe(**a),indent=2))

if __name__=='__main__':main()
