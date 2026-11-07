# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted finalist and final-arm routing, deliberately absent from editor tools."""
from __future__ import annotations
import json
from pathlib import Path
import uuid
from wsp03_bindings import PROFILES
from wsp03_worker import digest
from wsp03_io import json_write


def after_both_freeze(r3,r4,callback,*,required_seconds=0):
    if any(ledger.status()['phase']!='frozen' or not ledger.get('selected') for ledger in (r3,r4)):
        raise RuntimeError('final source access requires both successful R3/R4 freezes')
    if required_seconds<0 or any(ledger.remaining_joint_seconds()<=required_seconds for ledger in (r3,r4)):
        raise RuntimeError("joint deadline/reserve does not permit final source access")
    return callback()


def register_reference_profiles(ledger,*,checkpoint_identity):
    if ledger.get('contract')['track']!='R4' or not checkpoint_identity:raise ValueError('R4 immutable checkpoint required')
    with ledger.tx():
        old=ledger.get('r4_checkpoint_identity')
        if old is not None and old!=checkpoint_identity:raise ValueError('R4 cannot change frozen R3 checkpoint')
        ledger.set('r4_checkpoint_identity',checkpoint_identity)
        for profile,(count,selector,ordering,cues) in PROFILES.items():
            value={'id':profile,'kind':'reference_strategy','valid':True,'changed_elements':0,'mean_reference_count':count,'checkpoint_identity':checkpoint_identity,'strategy':{'count':count,'selector':selector,'ordering':ordering,'cues':cues}}
            existing=ledger.db.execute('SELECT payload FROM candidates WHERE id=?',(profile,)).fetchone()
            if existing and json.loads(existing[0])!=value:raise ValueError('reference registry changed')
            if not existing:ledger.db.execute('INSERT INTO candidates VALUES(?,?)',(profile,json.dumps(value,sort_keys=True)))
        ledger.event('reference_registry',{'profiles':list(PROFILES),'checkpoint_identity':checkpoint_identity,'weight_edits':False})


def finalize_development(service):
    ledger=service.ledger;service.require_launch()
    shortlist=ledger.nominate() if ledger.get('phase')=='explore' else ledger.get('shortlist')
    for candidate in shortlist:
        if ledger.db.execute("SELECT 1 FROM results WHERE id=? AND phase='full'",(candidate,)).fetchone():continue
        prior=ledger.db.execute("SELECT 1 FROM requests WHERE kind='full' AND json_extract(payload,'$.candidate')=?",(candidate,)).fetchone()
        if prior:continue
        params={'candidate':candidate,'profile':candidate if service.contract.get('weights_frozen') else 'P00','split':'development','ids':service.contract['splits']['development']}
        from wsp03_state import AdmissionError
        try:service.forward('full',params,trusted=True)
        except AdmissionError as error:
            # No scientific retry; preserve a budget-excluded finalist explicitly.
            with ledger.tx():
                skipped=ledger.get('skipped_full') or {};skipped[candidate]=str(error)
                ledger.set('skipped_full',skipped);ledger.event('finalist_not_admitted',{'candidate':candidate,'reason':str(error)})
    return ledger.freeze()


def source_result(service,*,kind,candidate,profile,split):
    rows=service.ledger.db.execute("SELECT id,state,payload FROM requests WHERE kind IN (?, 'reuse') AND json_extract(payload,'$.candidate')=? AND json_extract(payload,'$.profile')=? AND json_extract(payload,'$.split')=?",(kind,candidate,profile,split)).fetchall()
    eligible=[(rid,json.loads(payload)) for rid,state,payload in rows if state=='complete' and (json.loads(payload).get('reused_kind',kind)==kind)]
    if len(eligible)!=1:raise ValueError('exact single completed source pass required for reuse')
    return eligible[0]


def reuse_generation(target,source,*,kind,candidate,profile,split,source_candidate,source_profile):
    """Reuse only identical model, ordered request bytes, image bytes and scores."""
    target.require_launch();source.require_launch()
    if kind not in ('explore','full','trusted'):raise ValueError('unsupported reuse phase')
    if kind=='trusted' and (target.ledger.status()['phase']!='frozen' or source.ledger.status()['phase']!='frozen'):raise RuntimeError('trusted reuse requires freezes')
    if kind=='full' and candidate not in (target.ledger.get('shortlist') or []):raise ValueError('reuse full outside shortlist')
    params={'candidate':candidate,'profile':profile,'split':split,'ids':target.contract['splits'][split]}
    rows=target.requests('generation',params,trusted=kind=='trusted')
    source_id,result=source_result(source,kind=kind,candidate=source_candidate,profile=source_profile,split=split)
    actual=target.contract.get('fixed_worker_candidate',candidate);source_actual=source.contract.get('fixed_worker_candidate',source_candidate)
    if target.worker.checkpoint_identity(actual)!=source.worker.checkpoint_identity(source_actual):raise ValueError('reuse checkpoint mismatch')
    if result.get('requests_sha256')!=digest(rows):raise ValueError('reuse requests differ')
    if target.worker.image_identity(rows)!=source.worker.image_identity(rows):raise ValueError('reuse image bytes differ')
    if result.get('ids')!=[r['id'] for r in rows] or not result.get('complete'):raise ValueError('reuse source incomplete')
    if target.ledger.db.execute("SELECT 1 FROM requests WHERE json_extract(payload,'$.candidate')=? AND json_extract(payload,'$.profile')=? AND json_extract(payload,'$.split')=? AND (kind=? OR json_extract(payload,'$.reused_kind')=?)",(candidate,profile,split,kind,kind)).fetchone():raise ValueError('target pass already attempted or reused')
    rid='reuse-'+uuid.uuid4().hex
    reused=dict(result,candidate=candidate,profile=profile,split=split,reused_kind=kind,source_request_id=source_id,source_track=source.ledger.get('contract')['track'],request_id=rid)
    with target.ledger.tx():
        target.ledger.db.execute('INSERT INTO requests VALUES(?,?,?,?)',(rid,'reuse','complete',json.dumps(reused,sort_keys=True)))
        target.ledger.event('reuse_generation',{'id':rid,'source_request_id':source_id,'kind':kind,'candidate':candidate,'split':split,'requests_sha256':digest(rows),'generation_debit':0})
    metrics=target.grader(reused,split);metrics['mean_reference_count']=sum(len(r.get('references',[])) for r in rows)/len(rows)
    if kind in ('explore','full'):target.ledger.result(candidate,kind,metrics)
    reused['metrics']=metrics;return reused


def remaining_final_reserve(r3,r4,completed):
    """Observed-rate conservative reserve, including both accounts' final work."""
    chosen=r4.ledger.get('selected');queries=0;cpu_queries=0;passes=0
    for split in ('holdout','benchmark'):
        count=len(r3.contract['splits'].get(split,[]))
        if count==0:continue
        for arm in ('R3','R4','R4-C'):
            if arm+':'+split in completed or arm=='R4' and chosen=='P00' or arm=='R4-C' and chosen=='P01':continue
            multiplier=1
            if arm!='R3' and split=='benchmark':multiplier=len(r4.contract['benchmark_reference_ids'])/len(r4.contract['splits']['development'])
            queries+=count;cpu_queries+=count*multiplier if arm!='R3' else 0;passes+=1
    return max(r3.ledger.estimate_seconds(queries,passes,fixed_seconds=600,cpu_queries=cpu_queries),r4.ledger.estimate_seconds(queries,passes,fixed_seconds=600,cpu_queries=cpu_queries))

def evaluate_final_arms(r3,r4,attach_final,output,*,required_seconds=0):
    """Final input/scorer closures are first invoked behind both freeze checks."""
    required_seconds=max(required_seconds,(r4.ledger.get('joint_schedule') or {}).get('downstream_seconds',0))
    after_both_freeze(r3.ledger,r4.ledger,lambda:attach_final(r3,r4),required_seconds=required_seconds)
    chosen3,chosen4=r3.ledger.get('selected'),r4.ledger.get('selected');results={}
    for split in ('holdout','benchmark'):
        def evaluate(service,candidate,profile):
            from wsp03_state import AdmissionError
            reserve=remaining_final_reserve(r3,r4,results)
            if min(r3.ledger.remaining_joint_seconds(),r4.ledger.remaining_joint_seconds())<reserve:raise AdmissionError('remaining final work no longer fits adaptive joint reserve')
            result=service.forward('trusted',{'candidate':candidate,'profile':profile,'split':split,'ids':service.contract['splits'][split]},trusted=True)
            result['metrics']=service.grader(result,split);return result
        r3_result=evaluate(r3,chosen3,'P00');results['R3:'+split]=r3_result
        if chosen4=='P00':
            r4_result=reuse_generation(r4,r3,kind='trusted',candidate='P00',profile='P00',split=split,source_candidate=chosen3,source_profile='P00')
        else:r4_result=evaluate(r4,chosen4,chosen4)
        results['R4:'+split]=r4_result
        if chosen4=='P01':
            results['R4-C:'+split]=dict(r4_result,alias_of='R4:'+split,generation_debit=0)
            with r4.ledger.tx():r4.ledger.event('final_alias',{'arm':'R4-C','source_arm':'R4','split':split,'source_request_id':r4_result['request_id']})
        else:results['R4-C:'+split]=evaluate(r4,'P01','P01')
        json_write(Path(output)/'final-arm-results.json',results)
    return results
