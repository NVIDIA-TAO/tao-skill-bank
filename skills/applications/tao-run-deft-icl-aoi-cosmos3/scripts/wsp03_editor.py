# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-owned text supervisor; only fenced tools enter a different-UID sandbox."""
from __future__ import annotations
import json
import os
from pathlib import Path
import time
from wsp03_io import json_write
from wsp03_supervisor import canonical,parse_blocks,dispatch_blocks

PROTOCOL='Reply once per round. There are no built-in tools. Allowed fenced blocks: python (numeric code), read (JSON activation.read parameters), forward (JSON eval.dev parameters), save_edit (JSON edit.commit parameters), tool (JSON {action,parameters}), handoff (text), freeze (JSON {}). All tool execution is performed by the trusted harness through its allowlist. freeze must be the last block and ends editing; trusted shortlist/full-dev/freeze follows. At each fortieth round include exactly one handoff block that preserves scientific state; usage, deadlines and limits remain outside your control. Authorized development labels remain available. Never request Benchmark/holdout or their gold paths, credentials, supervisor files or network access.'

def run_editor(*,service,supervisor,immutable_task,context):
    service.require_launch()
    for runtime in (service.kernel.runtime,service.worker.runtime):
        if getattr(runtime,'uid',os.getuid()) in (0,os.getuid()):raise ValueError('all tool runtimes require dedicated actual host UID')
    state_path=Path(supervisor.output)/'dialogue-state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'summary':'','history':[],'last_completed_round':supervisor.book.status()['rounds']}
    if state['last_completed_round']!=supervisor.book.status()['rounds']:raise RuntimeError('dialogue/provider state uncertain; reconcile before resuming')
    service.result_assets=dict(getattr(service,'result_assets',{}),**state.get('result_assets',{}))
    reason=None
    try:
        while True:
            counters=service.ledger.status();round_number=supervisor.book.status()['rounds']+1
            if counters['remaining_development_seconds']<=0:reason='development_deadline';break
            if counters['soft_stopped'] or counters.get('editor_closed') or service.ledger.editing_seconds()<=0:reason='editor_or_joint_budget_closed';break
            payload={'immutable_task':immutable_task,'context':context,'protocol':PROTOCOL,'prior_handoff':state['summary'],'recent_dialogue':state['history'],'trusted_counters':counters,'global_round':round_number,'handoff_required':round_number%40==0}
            prompt=canonical(payload)
            if round_number==1 and len(prompt.encode())>48000:raise RuntimeError("initial serialized editor payload exceeds48000 bytes")
            try:response=supervisor.round(prompt,timeout=min(1800,service.ledger.editing_seconds() if hasattr(service.ledger,'editing_seconds') else counters['remaining_development_seconds']),ledger=service.ledger)
            except RuntimeError as error:
                # A rejected budget/context admission launches no provider work.
                # Other exceptions remain evidence-preserving failures.
                from wsp03_state import AdmissionError
                if isinstance(error,AdmissionError):reason='pre-provider admission closed: '+str(error);break
                if 'upper bound exceeds' in str(error) or 'context byte bound exceeds' in str(error):reason=str(error);break
                raise
            blocks=parse_blocks(response['text'],require_handoff=round_number%40==0)
            feedback,handoff,freeze=dispatch_blocks(service,blocks)
            from wsp03_resources import append_event
            append_event(Path(supervisor.output)/'dispatch-receipts.jsonl',{'global_round':round_number,'blocks':blocks,'tool_feedback':feedback,'handoff':handoff,'freeze':freeze,'completed_at':time.time()})
            for index,item in enumerate(feedback):
                raw=canonical(item).encode()
                if len(raw)>8192:
                    import hashlib
                    name='tool-receipt-'+str(round_number)+'-'+str(index)+'.json';path=Path(supervisor.output)/name
                    json_write(path,item)
                    service.result_assets[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                    feedback[index]={'action':item.get('action'),'receipt':name,'bytes':path.stat().st_size,'truncated':False,'inline_omitted':True,'read_with':'context.read name='+name}
            state['result_assets']=service.result_assets
            state['history'].extend([{'assistant':response['text']},{'tool_feedback':feedback}]);state['last_completed_round']=round_number
            if handoff is not None:
                state['summary']=handoff;state['history']=[{'tool_feedback':feedback}] if feedback else []
                with service.ledger.tx():service.ledger.event('editor_handoff',{'global_round':round_number,'usage_preserved':True,'clock_preserved':True})
            json_write(state_path,state)
            if freeze or not blocks:reason='editor_freeze_requested' if freeze else 'editor_complete_without_tools';break
    finally:
        supervisor.book.close(reason or 'editor_exception')
        with service.ledger.tx():service.ledger.set('editor_closed',reason or 'editor_exception');service.ledger.event('editor_stopped',{'reason':reason,'at':time.time()})
    result={'editor_stopped':True,'reason':reason,'usage':supervisor.book.status(),'trusted_finalization_available':True}
    json_write(Path(supervisor.output)/'editor-result.json',result);return result
