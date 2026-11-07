# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight native Codex experiment launch, self-reported usage and GPU samples.

prepare starts one container, prints its path canary and waits. start releases
that same container after the experiment owner records preregistration/start.
monitor uses native session token_count logs and nvidia-smi every10seconds;
these are best-effort observations, not tamper-proof admission or accounting.
"""
from __future__ import annotations
import argparse,json,os,shlex,subprocess,time
from pathlib import Path
from suite_runtime import Docker,BANK


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def prepare(config_file):
    cfg=json.loads(Path(config_file).read_text());output=Path(cfg['output']).resolve();output.mkdir(parents=True,exist_ok=False)
    sources={key:Path(cfg[key]).resolve(strict=True) for key in ('development','model','context','work','codex_home','binary')}
    if any(path in (Path('/'),Path('/home'),Path('/workspace'),Path('/data')) for path in sources.values()):raise ValueError('only explicit narrow experiment mounts are allowed')
    task=Path(cfg['task'])
    if task.is_absolute() or '..' in task.parts or not (sources['context']/task).is_file():raise ValueError('task must name a file within sanitized context')
    bank=Path(cfg.get('bank',BANK)).resolve(strict=True)
    mounts=[(sources['development'],'/development',True),(sources['model'],'/model',True),(sources['context'],'/context',True),(bank,'/bank',True),(sources['work'],'/work',False),(sources['codex_home'],'/codex',False),(sources['binary'],'/usr/local/bin/codex',True)]
    for key,destination in [('development_embeddings','/development-embeddings.jsonl'),('frozen_r3','/frozen-r3')]:
        if cfg.get(key):mounts.append((Path(cfg[key]).resolve(strict=True),destination,True))
    forbidden=cfg.get('forbidden_paths',['/holdout','/benchmark','/gold','/var/run/docker.sock','/home/ubuntu/workspace/data','/home/ubuntu/workspace/runs'])
    if any(not isinstance(value,str) or not value.startswith('/') for value in forbidden):raise ValueError('absolute forbidden container paths required')
    native=['/usr/local/bin/codex','exec','--dangerously-bypass-approvals-and-sandbox','--json','--ignore-rules','--skip-git-repo-check','--color','never','-m',cfg.get('model_id','gpt-6-astra'),'-c','model_reasoning_effort='+json.dumps(cfg.get('effort','xhigh')),'-c','features.daemon_auto_start=false','-C','/work','-']
    # One namespace/path canary. Its output contains names only, never file contents.
    script='set -eu\nls -ld -- /development /model /context /bank /work /codex {forbidden} > /results/path-canary.txt 2>&1 || true\nfor path in {forbidden}; do\n  if [ -e "$path" ]; then printf \'BLOCKED forbidden path visible: %s\\n\' "$path" >> /results/path-canary.txt; exit 42; fi\ndone\nprintf \'PASS: explicit development mounts; withheld host paths absent\\n\' >> /results/path-canary.txt\ntouch /results/canary-ready\nwhile [ ! -f /work/LIGHT_START ]; do sleep 0.2; done\ndate -u +%Y-%m-%dT%H:%M:%SZ > /results/editor-started.txt\nset +e\n{native} < {task} > /results/codex.jsonl 2> /results/codex.stderr\nstatus=$?\nprintf \'%s\\n\' "$status" > /results/editor-exit-code.txt\nexit "$status"\n'.format(forbidden=shlex.join(forbidden),native=shlex.join(native),task=shlex.quote('/context/'+str(task)))
    runtime=Docker(output/'jobs',cfg['image'],gpus='device='+str(cfg.get('gpu','0')))
    env=[('CODEX_HOME','/codex'),('HOME','/codex'),('PATH','/workspace/.venv/bin:/usr/local/bin:/usr/bin:/bin'),('PYTHONPATH','/bank/skills/applications/tao-run-deft-icl-aoi-cosmos3/scripts')]
    job=runtime.submit('wsp03-light-native-editor',['-c',script],entrypoint='/bin/bash',gpu=True,network='bridge',mounts=mounts,env=env)
    record={'schema':'wsp03-light-run/1','config':cfg,'job':job,'mounts':[{'source':str(source),'destination':destination,'readonly':readonly} for source,destination,readonly in mounts],'builtins':'native default tools enabled','usage':'Codex native self-report; best-effort','source_commit':cfg.get('source_commit') or subprocess.check_output(['git','-C',str(bank),'rev-parse','HEAD'],text=True).strip(),'prepared_at':time.time()}
    write(output/'run.json',record)
    deadline=time.time()+60
    while time.time()<deadline:
        if (Path(job['results_dir'])/'canary-ready').exists():
            record['canary']='passed';write(output/'run.json',record);return record
        state=runtime.status(job['job_id'])
        if state['state'] not in ('RUNNING','PENDING'):raise RuntimeError('container canary failed; inspect native logs and path-canary.txt')
        time.sleep(.2)
    raise TimeoutError('canary pending; preserve this container and inspect; do not launch another')


def start(run_file):
    record=json.loads(Path(run_file).read_text());work=Path(record['config']['work'])
    if not (Path(record['job']['results_dir'])/'canary-ready').exists():raise RuntimeError('one path canary has not completed')
    marker=work/'LIGHT_START'
    with marker.open('x') as file:file.write(str(time.time())+'\n')
    return {'started':True,'job_id':record['job']['job_id'],'time':time.time()}


def usage(home):
    sessions={};malformed=0
    for path in Path(home).glob('sessions/**/*.jsonl'):
        identity=str(path);latest=None
        for line in path.read_text(errors='replace').splitlines():
            try:value=json.loads(line)
            except ValueError:malformed+=1;continue
            payload=value.get('payload',{})
            if value.get('type')=='session_meta':identity=payload.get('id',identity)
            if payload.get('type')!='token_count':continue
            totals=(payload.get('info') or {}).get('total_token_usage') or {}
            if all(type(totals.get(key)) is int for key in ('input_tokens','output_tokens')):
                candidate={'input_tokens':totals['input_tokens'],'output_tokens':totals['output_tokens'],'cached_input_tokens':totals.get('cached_input_tokens')}
                candidate['total_tokens']=candidate['input_tokens']+candidate['output_tokens']
                if latest is None or candidate['total_tokens']>latest['total_tokens']:latest=candidate
        if latest and (identity not in sessions or latest['total_tokens']>sessions[identity]['total_tokens']):sessions[identity]=latest
    return {'measurement':'native session token_count self-report; mutable; best-effort','total_tokens':sum(value['total_tokens'] for value in sessions.values()) if sessions else None,'sessions':sessions,'malformed_lines':malformed,'inflight_usage':'unknown until native usage event'}


def monitor(run_file):
    path=Path(run_file);record=json.loads(path.read_text());cfg=record['config'];job=record['job'];runtime=Docker(path.parent/'jobs',cfg['image'],gpus='device='+str(cfg.get('gpu','0')))
    started=Path(cfg['work'])/'LIGHT_START';last=None;reason=None
    try:
        while True:
            state=runtime.status(job['job_id']);last=usage(cfg['codex_home']);now=time.time()
            try:
                gpu=subprocess.run(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=5)
                gpu_value,gpu_status=(gpu.stdout.strip() if gpu.returncode==0 else None),gpu.returncode
            except (OSError,subprocess.TimeoutExpired) as error:gpu_value,gpu_status=None,type(error).__name__
            sample={'time':now,'gpu_csv':gpu_value,'gpu_status':gpu_status,'usage':last,'native_state':state['state']}
            with (path.parent/'samples.jsonl').open('a') as file:file.write(json.dumps(sample)+'\n')
            print(json.dumps({'time':now,'state':state['state'],'self_reported_tokens':last['total_tokens']}),flush=True)
            if state['state'] not in ('RUNNING','PENDING'):reason='native_terminal';break
            elapsed=now-float(started.read_text()) if started.exists() else 0
            if last['total_tokens'] is not None and last['total_tokens']>=cfg.get('token_target',5000000):reason='approximate_token_target'
            if elapsed>=cfg.get('wall_hours',24)*3600:reason='best_effort_wall_limit'
            if reason:runtime.cancel(job['job_id']);break
            time.sleep(10)
    except KeyboardInterrupt:
        reason='monitor_interrupted';runtime.cancel(job['job_id']);raise
    finally:
        report={'reason':reason,'usage':last,'ended_at':time.time(),'wall_seconds':time.time()-float(started.read_text()) if started.exists() else None,'cost_quality':'best-effort; native self-report; unknown in-flight usage retained','job':job}
        write(path.parent/'cost.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','start','monitor']);parser.add_argument('file');args=parser.parse_args()
    print(json.dumps({'prepare':prepare,'start':start,'monitor':monitor}[args.action](args.file),indent=2))
if __name__=='__main__':main()
