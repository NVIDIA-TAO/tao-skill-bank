# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observable Codex tool events; Docker timestamps, never inferred durations."""
import datetime
import json
from collections import Counter
from pathlib import Path

KINDS={'command_execution','mcp_tool_call','web_search','image_view','file_change'}

def capture(runtime,job):
    raw=runtime.logs(job['job_id'],100000,timestamps=True)
    path=Path(job['results_dir'])/'codex_events.timestamped.log';path.write_text(raw)
    events=[]
    for ordinal,line in enumerate(raw.splitlines()):
        stamp,separator,body=line.partition(' ')
        try:
            event=json.loads(body)
            event['_observed_at']=datetime.datetime.fromisoformat(stamp.replace('Z','+00:00')).timestamp()
            event['_source_file']=str(path);event['_source_line']=ordinal
            events.append(event)
        except (ValueError,TypeError): continue
    return events

def observations(events):
    calls={}
    for event in events:
        item=event.get('item',{})
        if item.get('type') not in KINDS: continue
        identity=item.get('id')
        if identity is None: continue
        row=calls.setdefault(identity,{'id':identity,'type':item['type'],'started_at':None,'ended_at':None,
                                       'success':None,'elapsed_seconds':None,'raw':item})
        if event.get('type')=='item.started': row['started_at']=event.get('_observed_at')
        elif event.get('type')=='item.completed':
            row['ended_at']=event.get('_observed_at');row['raw']=item
            if item.get('exit_code') is not None: row['success']=item['exit_code']==0
            elif item.get('status') in ('completed','failed'): row['success']=item['status']=='completed'
            elif item.get('error') is not None: row['success']=False
            row['completed']=True
        if row['started_at'] is not None and row['ended_at'] is not None:
            row['elapsed_seconds']=row['ended_at']-row['started_at'] if row['ended_at']>=row['started_at'] else None
    return list(calls.values())

def summarize(invocations,planned_queries=271):
    calls=[];query_calls={};traces=[]
    for value in invocations:
        stage=value['stage'];obs=value.get('tool_observations',[])
        query_calls.setdefault(stage,0);query_calls[stage]+=len(obs)
        for call in obs: calls.append({**call,'stage':stage,'invocation_id':value['invocation_id']})
        if obs and len(traces)<3:
            traces.append({'stage':stage,'invocation_id':value['invocation_id'],'calls':obs,
                           'source':value.get('event_source'),'invocation_path':value.get('invocation_path'),
                           'raw_final_answer':value.get('raw_answer'),'completed_final':value.get('completed_final'),
                           'backend_state':value.get('backend_state'),
                           'selection':'first observed tool-using attempts; includes failures'})
    duration=[c['elapsed_seconds'] for c in calls if c['elapsed_seconds'] is not None]
    counts=Counter(c['type'] for c in calls)
    facts=[]
    if invocations:
        facts.append({'observation':f'{len(calls)} observable tool calls across {len(query_calls)} attempted query contexts.',
                      'evidence':'native item.started/item.completed events; not internal reasoning'})
    if duration:
        facts.append({'observation':f'{len(duration)} calls have paired Docker timestamps.',
                      'evidence':'elapsed wall time between observable item events, not model or CPU compute time'})
    return {'planned_queries':planned_queries,'attempted_query_contexts':len(query_calls),'attempts':len(invocations),
            'call_count':len(calls),'by_type':dict(counts),'calls_per_attempted_query':query_calls,
            'success':sum(c['success'] is True for c in calls),'failure':sum(c['success'] is False for c in calls),
            'unknown_outcome':sum(c['success'] is None for c in calls),
            'elapsed_tool_seconds':sum(duration) if duration else None,
            'timing_coverage':{'measured':len(duration),'total':len(calls),'unknown':len(calls)-len(duration)},
            'calls':calls,'representative_traces':traces,'findings':facts,
            'purpose_scope':'raw arguments/results expose operations; intentions or reasoning are not inferred',
            'limits':'failed and interrupted attempts included; parallel tool durations are not additive query latency; zero observed calls is not evidence of no internal reasoning'}
