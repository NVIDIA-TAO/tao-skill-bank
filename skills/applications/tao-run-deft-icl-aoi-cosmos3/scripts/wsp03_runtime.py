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
import math
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


def gated_job(runtime, lease, stage, command, mounts, *, deadline, env=(), on_admit=None, python=PYTHON):
    wrapper='while [ ! -f /results/lease-authorized ]; do sleep 0.05; done; exec '+shlex.join([python,*command])
    admitted={};pid=None
    def opened(job_id):
        record=json.loads(runtime.record('show',job_id))
        admitted.update(job_id=job_id,results_dir=record['results_dir'])
        if on_admit:on_admit(dict(admitted))
    try:
        job=runtime.submit(stage,['-c',wrapper],gpu=True,mounts=mounts,env=env,entrypoint='/bin/sh',on_open=opened,
            extra_args=['--read-only','--tmpfs','/tmp:rw,nosuid,nodev,size=2g','--memory','64g','--pids-limit','512'])
        native=runtime.status(job['job_id'])
        pid=native.get('native',{}).get('Pid',0)
        if not pid:raise RuntimeError('native GPU worker PID unavailable before gate')
        lease.allow_pid(pid,container_id=job['container_id']);lease.check()
        json_write(Path(job['results_dir'])/'lease-authorized',{'host_pid':pid,'time':time.time(),'lease_owner':lease.owner})
        status=runtime.wait(job['job_id'],deadline=deadline,heartbeat=lambda _:lease.check())
        lease.check()
        if status['state']!='COMPLETE':raise RuntimeError('worker failed: '+json.dumps(status))
        return job
    except BaseException:
        if admitted and runtime.status(admitted['job_id'])['state'] in ('RUNNING','PENDING'):runtime.cancel(admitted['job_id'])
        raise
    finally:
        if pid is not None and admitted and runtime.status(admitted['job_id'])['state'] in ('COMPLETE','ERROR','CANCELED'):
            lease.revoke_pid(pid)


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


def validate_worker_output(directory, requests, *, mode='generation', split='probe', candidate_manifest_sha256=None, artifact_owner=None, max_new_tokens=4096, query_seconds=180):
    directory=Path(directory)
    if artifact_owner is None:
        rows=[json.loads(line) for line in (directory/'raw_answers.jsonl').read_text().splitlines()]
        completion=json.loads((directory/'completion.json').read_text());profile=json.loads((directory/'runtime_profile.json').read_text())
    else:
        from wsp03_artifacts import child_bytes,child_json
        rows=[json.loads(line) for line in child_bytes(directory.parent,directory.name+'/raw_answers.jsonl',owner=artifact_owner,max_bytes=512*1024**2).decode().splitlines()]
        completion=child_json(directory.parent,directory.name+'/completion.json',owner=artifact_owner)
        profile=child_json(directory.parent,directory.name+'/runtime_profile.json',owner=artifact_owner)
    if [r['id'] for r in rows]!=[r['id'] for r in requests]:raise ValueError('worker output lacks exact ordered full membership')
    for row,request in zip(rows,requests):
        if row.get('completed') is not True or row.get('request_sha256')!=digest(request) or row.get('mode')!=mode:raise ValueError('worker answer identity/completion mismatch')
    if completion.get('status')!='complete' or completion.get('n')!=len(requests) or completion.get('requests_sha256')!=digest(requests) or completion.get('mode')!=mode or completion.get('split')!=split:raise ValueError('worker completion receipt mismatch')
    expected={'mode':mode,'split':split,'batch_size':1,'requests_sha256':digest(requests),'generation_hooks':False,
        'candidate_manifest_sha256':candidate_manifest_sha256,'seed':42,'dtype':'bfloat16','attention':'sdpa','do_sample':False,'max_new_tokens':max_new_tokens,'query_seconds':query_seconds}
    if any(profile.get(k)!=v for k,v in expected.items()):raise ValueError('worker runtime profile mismatch')
    return rows


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
                for label,path,batch in zip(('zs','icl'),request_paths,requests):
                    def admitted(job):
                        jobs.append(dict(job,mode=label));receipt.update(jobs=jobs,generation_passes=len(jobs))
                        json_write(output/'probe.json',receipt)
                    completed_job=gated_job(runtime,lease,'wsp03-probe-'+label,
                        ['/app/scripts/wsp03_worker.py','--model','/model','--requests','/request.json','--output','/results/worker','--mode','generation','--split','probe','--query-seconds',str(query_seconds)],
                        [(APP,'/app',True),(model,'/model',True),(view,'/input',True),(path,'/request.json',True)],
                        deadline=time.time()+64*query_seconds+900,on_admit=admitted)
                    validate_worker_output(Path(completed_job['results_dir'])/'worker',batch)
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
            rows=[];decode_errors=0
            for line in answers.read_text().splitlines() if answers.exists() else []:
                try:rows.append(json.loads(line))
                except (ValueError,TypeError):decode_errors+=1
            def numbers(key):
                return [r[key] for r in rows if type(r.get(key)) in (int,float) and math.isfinite(r[key]) and r[key]>=0]
            times=sorted(numbers('wall_seconds'))
            complete=receipt['status']=='complete' and len(rows)==64 and not decode_errors
            attempts_path=directory/'attempts.jsonl';attempts=[]
            for line in attempts_path.read_text().splitlines() if attempts_path.exists() else []:
                try:attempts.append(json.loads(line))
                except (ValueError,TypeError):decode_errors+=1
            answered={r.get('id') for r in rows}
            unresolved=sorted({r.get('id') for r in attempts if isinstance(r.get('id'),str)}-answered)
            passes.append({'mode':job['mode'],'job_id':job['job_id'],'n_completed_answers':len(rows),'n':len(rows),
                'completed_forward_wall_seconds':sum(times),'forward_wall_seconds':sum(times) if complete else None,
                'mean_query_seconds':sum(times)/len(times) if complete and times else None,
                'p95_query_seconds':times[min(len(times)-1,int(.95*len(times)))] if complete and times else None,
                'completed_input_tokens':sum(numbers('inference_input_tokens')),'completed_output_tokens':sum(numbers('inference_output_tokens')),
                'input_tokens':sum(numbers('inference_input_tokens')) if complete else None,
                'output_tokens':sum(numbers('inference_output_tokens')) if complete else None,
                'token_total_coverage':'complete' if complete else 'unknown; completed-answer subtotals only',
                'unresolved_query_ids':unresolved,'decode_errors':decode_errors,
                'peak_allocated_bytes':max(numbers('peak_allocated_bytes'),default=None),
                'peak_reserved_bytes':max(numbers('peak_reserved_bytes'),default=None),
                'runtime_profile':str(directory/'runtime_profile.json'),'raw_answers':str(answers),'attempts':str(attempts_path)})
        receipt.update(wall_seconds=time.time()-started,passes=passes,jobs=jobs,generation_passes=len(jobs),ended_at=time.time())
        if receipt['status']=='complete':
            mean=max(p['mean_query_seconds'] for p in passes)
            receipt.update(full_dev_estimate_seconds=mean*2917,explore_estimate_seconds=mean*512)
        json_write(output/'probe.json',receipt)
    return receipt


def grant_private_readonly(path, *, group, receipt):
    """Grant one dedicated group access to an explicitly staged private tree.

    Refuses symlinks/hardlinked files. Hashes before/after bind unchanged bytes;
    callers must choose their staged copy, never a source dataset/model tree.
    """
    import os
    from wsp03_bindings import file_hash
    path=Path(path).resolve(strict=True);items=[path,*sorted(path.rglob('*'))]
    if path==Path('/') or any(p.is_symlink() or p.is_file() and p.stat().st_nlink!=1 for p in items):raise ValueError('private independent regular tree required')
    before={str(p.relative_to(path)):{'uid':p.stat().st_uid,'gid':p.stat().st_gid,'mode':p.stat().st_mode&0o7777,'sha256':file_hash(p) if p.is_file() else None} for p in items}
    for p in items:
        subprocess.run(['sudo','-n','chgrp',str(group),str(p)],check=True)
        subprocess.run(['sudo','-n','chmod','0750' if p.is_dir() else '0440',str(p)],check=True)
    after={str(p.relative_to(path)):{'uid':p.stat().st_uid,'gid':p.stat().st_gid,'mode':p.stat().st_mode&0o7777,'sha256':file_hash(p) if p.is_file() else None} for p in items}
    if any(before[n]['sha256']!=after[n]['sha256'] or before[n]['uid']!=after[n]['uid'] for n in before):raise ValueError('read grant changed bytes/owner')
    record={'path':str(path),'group':group,'before':before,'after':after,'unchanged_bytes':True}
    json_write(receipt,record);return record


class DedicatedDocker(Docker):
    """Native jobs run under a real host tool UID, distinct from the supervisor.

    Only freshly allocated result directories and explicit writable tool mounts
    receive ownership. The supervisor CODEX_HOME/ledger must never be mounted.
    """
    def __init__(self, root, image, *, uid, gid, gpus='device=0', resources=None,resource_account='default'):
        import os
        if type(uid) is not int or uid in (0,os.getuid()) or type(gid) is not int or gid<1:
            raise ValueError('dedicated nonroot host tool UID must differ from supervisor')
        import threading
        self.uid=uid;self.gid=gid;self.resources=resources;self.resource_account=resource_account
        self.job_locks={};self.job_locks_guard=threading.Lock()
        super().__init__(root,image,gpus=gpus)
    def prepare_writable(self,path):
        import os
        path=Path(path)
        if path.is_symlink() or not path.is_dir():raise ValueError('tool writable mount must be real directory')
        subprocess.run(['sudo','-n','chown',str(self.uid)+':'+str(os.getgid()),str(path)],check=True)
        subprocess.run(['sudo','-n','chmod','2770',str(path)],check=True)
    def submit(self,stage,command,**kwargs):
        if self.resources:self.resources.check()
        for host,target,readonly in kwargs.get('mounts',()):
            if not readonly:self.prepare_writable(host)
        prior=kwargs.pop('on_open',None)
        def opened(job_id):
            record=json.loads(self.record('show',job_id));directory=Path(record['results_dir']);directory.mkdir(parents=True,exist_ok=True)
            self.prepare_writable(directory)
            if self.resources:self.resources.register_job(job_id,lambda:self.cancel(job_id))
            if prior:prior(job_id)
        extra=list(kwargs.pop('extra_args',()))
        if self.resources:extra+=['--cgroup-parent',self.resources.slice]
        if any(x=='--user' or str(x).startswith('--user=') for x in extra):raise ValueError('tool UID override forbidden')
        return super().submit(stage,command,on_open=opened,extra_args=[*extra,'--user',str(self.uid)+':'+str(self.gid)],**kwargs)


    def job_lock(self,job):
        import threading
        with self.job_locks_guard:return self.job_locks.setdefault(job,threading.RLock())
    def status(self,job):
        with self.job_lock(job):
            value=super().status(job)
            if self.resources and value['state'] in ('COMPLETE','ERROR','CANCELED'):self.resources.unregister_job(job)
            return value
    def cancel(self,job):
        with self.job_lock(job):
            record=json.loads(self.record('show',job))
            if record['terminal_state'] not in ('COMPLETE','ERROR','CANCELED'):
                if self.resources:self.resources.note_cancel_request()
                super().cancel(job)
            # A failed stop remains registered for cleanup; never lose ownership.
            if self.resources:self.resources.unregister_job(job)
    def wait(self,job,deadline,heartbeat=None):
        def guarded(state):
            if self.resources and self.resources.error:raise RuntimeError(self.resources.error)
            if heartbeat:heartbeat(state)
        return super().wait(job,deadline,heartbeat=guarded)


class IsolatedKernel:
    def __init__(self, *, runtime, base, output, exchange, access_policy=None, editing_policy=None, captures=None):
        self.runtime=runtime;self.output=Path(output).resolve();self.exchange=Path(exchange).resolve()
        self.output.mkdir(parents=True,exist_ok=True);self.exchange.mkdir(parents=True,exist_ok=True)
        runtime.root.mkdir(parents=True,exist_ok=True)
        canary=runtime.root/('host-denial-canary-'+uuid.uuid4().hex)
        canary.write_text('synthetic host namespace canary; not a credential')
        policy_dir=runtime.root/('kernel-policy-'+uuid.uuid4().hex);policy_dir.mkdir(mode=0o700)
        def stage_policy(path,name):
            source=Path(path)
            if source.is_symlink():raise ValueError('policy symlink forbidden')
            target=policy_dir/name;target.write_bytes(source.read_bytes());target.chmod(0o444);return target
        if access_policy:access_policy=stage_policy(access_policy,'access-policy.json')
        if editing_policy:editing_policy=stage_policy(editing_policy,'editing-policy.json')
        self.captures=Path(captures) if captures else runtime.root/'captures'
        self.captures.mkdir(parents=True,exist_ok=True,mode=0o750)
        if isinstance(runtime,DedicatedDocker):subprocess.run(['sudo','-n','chgrp',str(runtime.gid),str(self.captures)],check=True)
        extra_mounts=[(self.captures,'/captures',True)]
        if access_policy:extra_mounts.append((access_policy,'/access-policy.json',True))
        extra_command=['--captures','/captures']
        if access_policy:extra_command+=['--access-policy','/access-policy.json']
        if editing_policy:
            extra_mounts.append((editing_policy,'/editing-policy.json',True));extra_command+=['--editing-policy','/editing-policy.json']
        self.job=runtime.submit('wsp03-cpu-kernel',['/app/scripts/wsp03_kernel_server.py','--base','/model','--output','/edits','--exchange','/exchange',*extra_command],
            gpu=False,mounts=[(APP,'/app',True),(base,'/model',True),(self.output,'/edits',False),(self.exchange,'/exchange',False),*extra_mounts],
            env=[('NVIDIA_VISIBLE_DEVICES','void'),('WSP_HOST_CANARY',str(canary))],extra_args=['--runtime','runc','--read-only','--tmpfs','/tmp:rw,nosuid,nodev,size=1g','--memory','48g','--cpus','8','--pids-limit','128'])
    def call(self, request, timeout=660):
        identity=uuid.uuid4().hex;input_file=self.exchange/(identity+'.request.json');response=self.exchange/(identity+'.response.json')
        # Stage privately until the dedicated child group can read it; publish last.
        staged=self.exchange/(identity+'.pending.json')
        json_write(staged,request,mode=0o640)
        if isinstance(self.runtime,DedicatedDocker):
            subprocess.run(['sudo','-n','chgrp',str(self.runtime.gid),str(staged)],check=True)
        os.replace(staged,input_file);deadline=time.monotonic()+timeout
        while not response.exists():
            if time.monotonic()>deadline:raise TimeoutError('kernel request uncertain; preserve admitted identity and reconcile')
            status=self.runtime.status(self.job['job_id'])
            if status['state'] not in ('RUNNING','PENDING'):raise RuntimeError('kernel stopped; namespace lost, ledger remains durable')
            time.sleep(.2)
        from wsp03_artifacts import child_json
        return child_json(self.exchange,response.name,owner=getattr(self.runtime,'uid',os.getuid()))
    def close(self):
        if self.runtime.status(self.job['job_id'])['state'] in ('RUNNING','PENDING'):self.runtime.cancel(self.job['job_id'])


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('probe')
    for n in ('model','zs-requests','icl-requests','input-view','output','image','lease-path','probe-ids-sha256'):q.add_argument('--'+n,required=True)
    q.add_argument('--gpu',default='0');q.add_argument('--query-seconds',type=int,default=180)
    a=vars(p.parse_args());a.pop('command');print(json.dumps(probe(**a),indent=2))

if __name__=='__main__':main()
