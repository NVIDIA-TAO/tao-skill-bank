# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One adaptive ICL proposal at a time in a fresh, role-limited Codex context."""
import json
import time
from pathlib import Path
from astra_isolation import environment,hardening,token_cap_config
from suite_costs import codex_usage
from suite_runtime import BANK,json_write

SCHEMA={
    'type':'object','additionalProperties':False,'required':['action','reason','policy'],
    'properties':{
        'action':{'type':'string','enum':['trial','stop']},'reason':{'type':'string'},
        'policy':{'anyOf':[{'type':'null'},{'type':'object','additionalProperties':False,
          'required':['toolset','version','rationale','max_calls','parameters'],
          'properties':{'toolset':{'type':'string','enum':['siglip-retrieval-v1']},
                        'version':{'type':'integer','enum':[1]},'rationale':{'type':'string'},
                        'max_calls':{'type':'integer','enum':[1]},
                        'parameters':{'type':'object','additionalProperties':False,
                          'required':['reference_count','ordering','instruction'],
                          'properties':{'reference_count':{'type':'integer','minimum':1},
                            'ordering':{'type':'string','enum':['nearest_first','farthest_first','class_grouped']},
                            'instruction':{'type':'string'}}}}}]}}
}

def propose(runtime,mounts,auth_file,model,effort,context,output,wall_seconds,*,job=None,on_open=None):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    json_write(output/'context.json',context);json_write(output/'schema.json',SCHEMA)
    skill=BANK/'skills/data/tao-in-context-learning'
    prompt=('Use the ICL data skill at /skill.md and its toolset contract at /toolsets.md. '
            'Read /proposal/context.json. Propose exactly one next candidate from the supplied Proxy evidence, '
            'or stop if completed candidates suffice within the remaining budget. Do not preselect a search grid. '
            'The first proposal must be a trial. Use only generic reference instructions; do not invent class visual '
            'descriptions, derive visual hints from labels, or use an atlas. Preserve the exact query prompt/profile. '
            'The application executes the candidate; do not invoke a model or modify data yourself. '
            'Return the schema-conforming decision and policy. No Benchmark feedback is provided or permitted.')
    command=['exec','--ignore-user-config','--ignore-rules','--skip-git-repo-check',
             '--dangerously-bypass-approvals-and-sandbox','--json','--color','never','-C','/task',
             '-m',model,'-c',f'model_reasoning_effort="{effort}"','-c','features.plugins=false',
             '-c','features.multi_agent=false','--output-schema','/proposal/schema.json',
             '-c',token_cap_config(context['remaining_budget']['tokens']),
             '-o','/results/decision.json',prompt]
    started=time.time()
    if job is None:
      job=runtime.submit('icl-policy',['/runtime/codex_capture.py','/usr/local/bin/codex',*command],gpu=False,network='bridge',entrypoint='/usr/bin/python3',
                       mounts=[*mounts,(auth_file,'/codex/auth.json',True),(output,'/proposal',True),
                               (skill/'SKILL.md','/skill.md',True),(skill/'references/toolsets.md','/toolsets.md',True)],
                       env=environment(),extra_args=hardening(),launch_timeout=min(120,wall_seconds),on_open=on_open)
    result=Path(job['results_dir'])
    if (result/'proposal_result.json').exists():
        saved=json.loads((result/'proposal_result.json').read_text());return saved['decision'],saved['details']
    started=json.loads((result/'launch.json').read_text())['submitted_at_unix']
    state=runtime.wait(job['job_id'],started+wall_seconds)
    result=Path(job['results_dir'])
    from tool_evidence import capture,observations
    events=capture(runtime,job)
    (result/'codex_events.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in events))
    captured=result/'codex_usage_events.jsonl'
    usage=codex_usage(events+([json.loads(s) for s in captured.read_text().splitlines()] if captured.exists() else []))
    usage['tool_observations']=observations(events)
    usage['event_source']=str(result/'codex_events.timestamped.log')
    details={'job':job,'usage':usage,'backend':state,'started_at_unix':started,'ended_at_unix':time.time(),
             'budget_exhausted':'rollout token budget exhausted' in runtime.logs(job['job_id'],100000)}
    json_write(result/'proposal_usage.json',details)
    decision=None
    if state['state']=='COMPLETE' and (result/'decision.json').is_file():
        try: decision=json.loads((result/'decision.json').read_text())
        except ValueError: details['error']='invalid policy JSON'
    json_write(result/'proposal_result.json',{'decision':decision,'details':details})
    if decision is None: return None,details
    if decision['action']=='stop' and not context['trials']:
        raise ValueError('policy agent cannot stop before a completed trial')
    return decision,details
