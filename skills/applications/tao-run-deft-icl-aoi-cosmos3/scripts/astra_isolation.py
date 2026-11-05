# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fresh per-query Codex container; generic tools, no task repository or labels."""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from suite_runtime import Docker,json_write
from suite_costs import codex_usage
from tht_contract import CLASSES,sha256

DENIED = ['/home/ubuntu/workspace/data/source','/home/ubuntu/workspace/data/private',
          '/home/ubuntu/workspace/data/packages','/home/ubuntu/workspace/tao-skill-bank',
          '/source','/private','/mining','/workspace','/var/run/docker.sock',
          '/home/ubuntu/.codex/config.toml','/home/ubuntu/.codex/skills']

PROMPT_TEMPLATE=('Classify the attached image of a through-hole solder joint into exactly one class: '+
                 ', '.join(CLASSES)+'.\nReturn exactly one JSON object with only the key "labels", whose value is an array of exactly one original class name: {{"labels":["class name"]}}. '+
                 'Do not include explanations, Markdown fences, or other text outside the JSON.\n'+
                 'Budget: {wall_seconds:g} seconds and {token_budget} total input/output tokens. Generic tools are available.')

def generic_mounts(codex_binary,python_binary,python_stdlib,system_libraries,site_packages=()):
    # Deliberately never mount host HOME, source checkout, full venv, data root,
    # Docker socket, user configuration, plugins or session history.
    mounts=[(codex_binary,'/usr/local/bin/codex',True),(python_binary,'/usr/bin/python3',True),
            (python_stdlib,'/usr/lib/python3.12',True),(system_libraries,'/usr/lib/x86_64-linux-gnu',True),
            ('/etc/ssl/certs','/etc/ssl/certs',True)]
    for helper in ('codex-code-mode-host','codex-exec-server'):
        mounts.append((Path(codex_binary).resolve().parent/helper,'/usr/local/bin/'+helper,True))
    mounts.append((Path(__file__).with_name('codex_capture.py'),'/runtime/codex_capture.py',True))
    for path in site_packages:
        p=Path(path)
        if p.name not in ('PIL','pillow.libs','numpy','numpy.libs'):
            raise ValueError('only generic image/numeric packages may enter A')
        mounts.append((p,'/opt/generic/'+p.name,True))
    return mounts

def probe_command():
    return ['-c', 'import os,json; paths='+repr(DENIED)+'; r={p:os.path.exists(p) or os.access(p,os.R_OK) for p in paths}; print(json.dumps(r)); assert not any(r.values()); assert not os.path.exists("/AGENTS.md"); assert not os.path.exists("/task/AGENTS.md"); assert not os.path.exists("/codex/config.toml")']

def environment():
    return [('HOME','/tmp/home'),('CODEX_HOME','/codex'),('PYTHONPATH','/opt/generic'),('PYTHONNOUSERSITE','1')]

def hardening():
    return ['--read-only','--tmpfs','/tmp:rw,exec,size=1g,mode=1777',
            '--tmpfs','/codex:rw,exec,size=256m,mode=1777',
            '--tmpfs','/task:rw,exec,size=1g,mode=1777','--pids-limit','128',
            '--memory','4g','--cpus','2','--workdir','/task']

def token_cap_config(limit):
    # Codex 0.166.0 native rollout accounting, with equal input/output weights.
    # The runtime checks completed responses before permitting continuation;
    # one in-flight response may overshoot. Session events retain actual usage.
    if int(limit)<2: raise ValueError('native rollout token limit must be at least two')
    return 'features.rollout_budget={enabled=true,enforce_limit=true,limit_tokens='+str(int(limit))+',reminder_at_remaining_tokens=['+str(max(1,int(limit)//2))+'],prefill_token_weight=1,sampling_token_weight=1}'

def isolation_probe(runtime: Docker,mounts,*,timeout=60,on_open=None):
    job=runtime.submit('astra-access-probe',probe_command(),mounts=mounts,gpu=False,
                       network='none',entrypoint='/usr/bin/python3',env=environment(),extra_args=hardening(),on_open=on_open)
    result=runtime.wait(job['job_id'],time.time()+timeout)
    if result['state']!='COMPLETE': raise RuntimeError('classifier runtime access-negative probe failed: '+runtime.logs(job['job_id']))
    return job

def classify(runtime: Docker, query, *, auth_file, mounts, model, effort, wall_seconds, token_budget, job=None, on_open=None):
    """No task skill, references, query gold, previous context or policy hints."""
    if not model or wall_seconds<=0 or token_budget<=0: raise ValueError('frozen A model and budgets required')
    prompt=PROMPT_TEMPLATE.format(wall_seconds=wall_seconds,token_budget=token_budget)
    # Binary is static; only its approved private auth file enters the container.
    # Auth is never copied into results, serialized into launch records or read.
    command=['exec','--ignore-user-config','--ignore-rules','--skip-git-repo-check',
             '--dangerously-bypass-approvals-and-sandbox','--json','--color','never',
             '-m',model,'-c',f'model_reasoning_effort="{effort}"',
             '-c','developer_instructions=""','-c','features.plugins=false',
             '-c','features.multi_agent=false','-C','/task','-i','/query.png',
             '-c',token_cap_config(token_budget),
             '-o','/results/final.txt',prompt]
    bootstrap=('import os,json,sys; from pathlib import Path; paths='+repr(DENIED)+'; '+
               'r={p:os.path.exists(p) or os.access(p,os.R_OK) for p in paths}; '+
               'Path("/results/access_probe.json").write_text(json.dumps(r)); assert not any(r.values()); '+
               'assert not Path("/codex/config.toml").exists(); assert not Path("/task/AGENTS.md").exists(); '+
               'os.execv("/usr/bin/python3",["python3","/runtime/codex_capture.py","/usr/local/bin/codex",*sys.argv[1:]])')
    if job is None:
        job=runtime.submit('astra-query',['-c',bootstrap,*command],mounts=[*mounts,(query,'/query.png',True),(auth_file,'/codex/auth.json',True)],
                           env=environment(),gpu=False,network='bridge',entrypoint='/usr/bin/python3',extra_args=hardening(),on_open=on_open)
    out=Path(job['results_dir'])
    if (out/'invocation.json').exists(): return json.loads((out/'invocation.json').read_text())
    launch=json.loads((out/'launch.json').read_text())
    prompt=launch['argv'][-1]
    started=launch['submitted_at_unix']; budget_hit=False; usage_events=[]
    def observe(_):
        nonlocal budget_hit
        raw=runtime.logs(job['job_id'],10000)
        # Sessions live only in this fresh container's tmpfs, never in host HOME.
        # Keep only usage events; do not read auth/config or copy session files.
        probe='import json; from pathlib import Path;\nfor p in Path("/codex/sessions").rglob("*.jsonl"):\n for line in p.read_text().splitlines():\n  try: e=json.loads(line)\n  except ValueError: continue\n  if e.get("payload",{}).get("type")=="token_count": print(json.dumps(e))'
        observed=subprocess.run(['docker','exec',job['job_id'],'/usr/bin/python3','-c',probe],capture_output=True,text=True)
        current=[]
        for line in observed.stdout.splitlines():
            try: current.append(json.loads(line))
            except ValueError: pass
        if current: usage_events[:]=current
        events=[]
        for line in raw.splitlines():
            try: events.append(json.loads(line))
            except ValueError: pass
        usage=codex_usage(events+usage_events)
        if usage['input_tokens'] is not None and usage['output_tokens'] is not None and usage['input_tokens']+usage['output_tokens']>=token_budget:
            # Stop only if still running. A completed answer is never discarded.
            if runtime.status(job['job_id'])['state']=='RUNNING':
                runtime.cancel(job['job_id']); budget_hit=True
    state=runtime.wait(job['job_id'],started+wall_seconds,heartbeat=observe)
    out=Path(job['results_dir']); raw=runtime.logs(job['job_id'],100000)
    # These are genuine top-level Codex JSON events, not tool stdout strings.
    from tool_evidence import capture,observations
    events=capture(runtime,job)
    (out/'codex_events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    captured=out/'codex_usage_events.jsonl'
    if captured.exists(): usage_events=[json.loads(s) for s in captured.read_text().splitlines()]
    else: captured.write_text(''.join(json.dumps(e)+'\n' for e in usage_events))
    usage=codex_usage(events+usage_events)
    usage['tool_observations']=observations(events)
    usage['event_source']=str(out/'codex_events.timestamped.log')
    budget_hit=budget_hit or 'rollout token budget exhausted' in raw
    final=out/'final.txt'
    messages=[e['item'] for e in events if e.get('type')=='item.completed' and e.get('item',{}).get('type')=='agent_message']
    # A proper final file is owned by Codex; tool outputs are never parsed as answers.
    answer=final.read_text() if final.is_file() and (final.stat().st_size or any(e.get('type')=='turn.completed' for e in events)) else None
    result={'job':job,'backend_state':state,'completed':answer is not None,'raw_answer':answer,
            'usage':usage,'started_at_unix':started,'ended_at_unix':time.time(),'budget_exhausted':budget_hit,
            'token_budget_observed_exceeded':budget_hit or (usage['input_tokens'] is not None and usage['output_tokens'] is not None and usage['input_tokens']+usage['output_tokens']>=token_budget),'prompt':prompt,
            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
            'image_sha256':sha256(query),'model':model,'effort':effort,
            'agent_messages':messages,'finish_reason':'budget' if budget_hit or state['state']=='CANCELED' else 'stop'}
    json_write(out/'invocation.json',result)
    return result
