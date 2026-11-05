# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize real Codex events without counting cumulative totals repeatedly."""
from __future__ import annotations

FIELDS = ('input_tokens', 'output_tokens', 'cached_input_tokens')

def codex_usage(events):
    """Support exec turn.completed usage and session token_count cumulative usage.

    Prefer authoritative per-turn events when both schemas are present. Callers
    pass one invocation/session, never concatenate unrelated sessions.
    """
    turns = {}
    cumulative = {}
    tools = []
    for index, event in enumerate(events):
        kind = event.get('type')
        item = event.get('item', {})
        if kind == 'item.completed' and item.get('type') in ('command_execution', 'mcp_tool_call', 'web_search', 'image_view', 'file_change'):
            tools.append(item)
        if kind == 'turn.completed' and event.get('usage') is not None:
            key = event.get('request_id') or event.get('turn_id') or event.get('event_id') or (str(event.get('session_id',''))+':'+str(event['ordinal']) if 'ordinal' in event else None) or (str(event['_source_file'])+':'+str(event['_source_line']) if '_source_file' in event and '_source_line' in event else str(index))
            usage = event['usage']
            # Without an ID, equal-sized distinct turns must remain distinct.
            # Repeated cumulative snapshots are handled separately below.
            turns[key] = usage
        payload = event.get('payload', event)
        if payload.get('type') == 'token_count':
            usage = (payload.get('info') or {}).get('total_token_usage')
            if usage:
                for key in FIELDS:
                    native = 'cached_input_tokens' if key == 'cached_input_tokens' else key
                    if native in usage:
                        cumulative[key] = max(cumulative.get(key, 0), usage[native])
    if turns:
        result = {k: sum(u.get(k, 0) for u in turns.values()) if all(k in u for u in turns.values()) else None for k in FIELDS}
        source = 'codex_turn_completed'
    elif cumulative:
        result = {k:cumulative.get(k) for k in FIELDS}
        source = 'codex_session_cumulative_max'
    else:
        result = {k:None for k in FIELDS}
        source = 'unavailable'
    # Cached input is a subset of input, never an extra additive cost.
    result.update(source=source, tool_calls=tools, remote_gpu_seconds=None, price=None)
    return result

def interval_usage(events,start,end,session_id,session_started_at=None):
    """Bounded cumulative deltas for one explicitly identified continuing session.

    Include the last pre-interval snapshot as baseline. Replayed snapshots do
    not add cost. A decreasing counter is a reset and requires a new session
    identity/export rather than inventing usage. Reasoning/cache are subsets.
    """
    from datetime import datetime
    def stamp(s): return datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
    lower,upper=stamp(start),stamp(end)
    snapshots=[]
    for e in events:
        if not isinstance(e,dict): raise ValueError('token export event must be an object')
        p=e.get('payload',e)
        if not isinstance(p,dict): raise ValueError('token export payload must be an object')
        if p.get('type')!='token_count' or not e.get('timestamp'): continue
        info=p.get('info') or {}
        if not isinstance(info,dict): raise ValueError('token export info must be an object')
        usage=info.get('total_token_usage')
        if usage:
            if not isinstance(usage,dict) or any(type(usage.get(k)) is not int or usage[k]<0 for k in FIELDS):
                raise ValueError('malformed cumulative token usage')
            if usage['cached_input_tokens']>usage['input_tokens']:
                raise ValueError('cached input exceeds total input')
            snapshots.append((stamp(e['timestamp']),e.get('ordinal',0),usage))
    snapshots.sort(key=lambda x:(x[0],x[1]))
    if not any(timestamp<=upper for timestamp,_,_ in snapshots):
        return {**{k:None for k in FIELDS},'session_id':session_id,'reason':'no usage snapshots at or before interval end'}
    born_inside=session_started_at is not None and lower<=stamp(session_started_at)<=upper
    baseline={k:0 for k in FIELDS} if born_inside else None
    last=baseline;total={k:0 for k in FIELDS}; count=0
    for timestamp,_,u in snapshots:
        if timestamp>upper: break
        if last is not None and any(u.get(k,0)<last.get(k,0) for k in FIELDS):
            raise ValueError('cumulative reset requires an explicit session boundary')
        if timestamp<lower: baseline=u;last=u;continue
        if last is None:
            return {**{k:None for k in FIELDS},'session_id':session_id,'reason':'pre-interval usage baseline missing'}
        for k in FIELDS: total[k]+=u.get(k,0)-last.get(k,0)
        last=u;count+=1
    return {**total,'session_id':session_id,'interval_start':start,'interval_end':end,
            'source':'bounded_cumulative_deltas','snapshots':count,'baseline_present':baseline is not None,
            'zero_baseline_from_session_birth':born_inside,'session_started_at':session_started_at,'price':None}

def aggregate(rows):
    """Roll up unique leaf invocations; shared/A agent-classifier calls count once."""
    leaves = {}
    for row in rows:
        if row.get('children'):
            continue
        key = row['invocation_id']
        if key in leaves and leaves[key] != row:
            raise ValueError('conflicting cost records for the same invocation')
        leaves[key] = row
    result = {'invocations':len(leaves), 'coverage':{}}
    for field in (*FIELDS, 'inference_input_tokens', 'inference_output_tokens', 'gpu_seconds', 'queue_seconds'):
        if field in FIELDS: eligible=[r for r in leaves.values() if any(k in r for k in FIELDS)]
        elif field.startswith('inference_'): eligible=[r for r in leaves.values() if any(k.startswith('inference_') for k in r)]
        elif field=='gpu_seconds': eligible=[r for r in leaves.values() if r.get('gpu_requested') or r.get(field) is not None]
        else: eligible=[r for r in leaves.values() if 'job_id' in r or field in r]
        values = [r.get(field) for r in eligible]
        present = [v for v in values if v is not None]
        result[field] = sum(present) if present else None
        result['coverage'][field] = {'measured':len(present), 'total':len(values),'population':'applicable invocations only'}
    intervals = sorted((r['started_at_unix'], r['ended_at_unix']) for r in leaves.values() if r.get('started_at_unix') is not None and r.get('ended_at_unix') is not None)
    merged = []
    for start,end in intervals:
        if end < start:
            raise ValueError('negative cost interval')
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start,end])
    result['active_wall_seconds'] = sum(b-a for a,b in merged) if merged else None
    result['elapsed_wall_seconds'] = max(b for a,b in intervals)-min(a for a,b in intervals) if intervals else None
    result['price'] = None
    devices={}
    for row in leaves.values():
        if row.get('gpu_resource') and row.get('started_at_unix') is not None and row.get('ended_at_unix') is not None:
            devices.setdefault(row['gpu_resource'],[]).append((row['started_at_unix'],row['ended_at_unix']))
    occupied={}
    for device,spans in devices.items():
        merged=[]
        for start,end in sorted(spans):
            if merged and start<=merged[-1][1]: merged[-1][1]=max(end,merged[-1][1])
            else: merged.append([start,end])
        occupied[device]=sum(end-start for start,end in merged)
    result['physical_device_occupied_seconds']=sum(occupied.values()) if occupied else None
    result['physical_devices']=occupied
    result['gpu_method']='gpu_seconds sums per-job allocated wall time; physical_device_occupied_seconds unions intervals per device, including held allocations during waits; neither measures kernel-active time'
    return result
