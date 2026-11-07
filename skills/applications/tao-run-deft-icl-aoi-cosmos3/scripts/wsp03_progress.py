# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Derive required scientific phase completion from charged actual dev receipts."""
import json

def choose_layers(layers,baseline,restored,ids):
    order={layer:index for index,layer in enumerate(layers)}
    return sorted(layers,key=lambda layer:(-sum(restored[layer][qid]-baseline[qid] for qid in ids)/len(ids),order[layer]))[:2]

def progress(service):
    recipe,ledger=service.request_factory,service.ledger;contract=service.contract.get('diagnostic_contract')
    if not contract:return {'enforced':False}
    cache_key=[ledger.db.execute("SELECT count(*) FROM requests WHERE kind='read' AND state='complete'").fetchone()[0],len(ledger.get('attribution_pairs') or {})]
    old=ledger.get('phase_progress_cache')
    if old and old['key']==cache_key:return old['value']
    expected={(qid,ref) for qid in service.contract['splits']['explore'] for ref in recipe.attribution_candidates(qid)}
    observed={(r['query_id'],r['reference_id']) for r in (ledger.get('attribution_pairs') or {}).values()}
    phase1=expected<=observed
    ids=set(contract['query_ids']);layers=contract['layers'];baseline={};restored={};heads={}
    for payload, in ledger.db.execute("SELECT payload FROM requests WHERE kind='read' AND state='complete'"):
        payload=json.loads(payload)
        if payload.get('candidate')!='base' or payload.get('profile','P00') in ('B04','P10','P11'):continue
        for condition,row in zip(payload.get('conditions',[]),payload.get('raw',[])):
            qid=condition['query_id'];patches=condition.get('capture',{}).get('patches',[])
            if qid not in ids or condition['references'] or not row.get('correct_class'):continue
            if not patches:baseline[qid]=row['correct_class']['mean_logprob'];continue
            if len(patches)!=1:continue
            patch=patches[0];source=patch.get('source_capture')
            if not source or source.get('scale',1)!=1 or patch.get('mode')!='replace' or patch.get('positions')!=['pre_label']:continue
            src=ledger.db.execute("SELECT payload FROM requests WHERE id=? AND kind='read' AND state='complete'",(source['request_id'],)).fetchone()
            if not src:continue
            src=json.loads(src[0]);matching=[r for r in src.get('raw',[]) if r['id']==source['record_id']]
            if src.get('candidate')!='base' or src.get('profile','P00') in ('B04','P10','P11') or len(matching)!=1 or matching[0].get('references')!=recipe.references(qid,'P01'):continue
            module=patch['module'];value=row['correct_class']['mean_logprob']
            if module in layers and patch.get('kind','output')=='output' and source['key']==module+':output':restored.setdefault(module,{})[qid]=value
            else:
                for layer,head_module in contract['head_modules'].items():
                    head=patch.get('head',{})
                    if module==head_module and patch.get('kind')=='input' and source['key']==module+':input' and head.get('count')==contract['head_count'] and head.get('dimension')==contract['head_dimension'] and type(head.get('index'))is int and 0<=head['index']<contract['head_count']:
                        heads.setdefault((layer,head['index']),{})[qid]=value
    layers_complete=ids<=set(baseline) and all(ids<=set(restored.get(layer,{})) for layer in layers)
    chosen=choose_layers(layers,baseline,restored,ids) if layers_complete else []
    heads_complete=bool(chosen) and all(ids<=set(heads.get((layer,head),{})) for layer in chosen for head in range(contract['head_count']))
    result={'enforced':True,'phase1_complete':phase1,'attribution_expected':len(expected),'attribution_observed':len(observed&expected),'phase2_layer_sweep_complete':layers_complete,'top_layers':chosen,'phase2_head_sweep_complete':heads_complete,'editing_ready':phase1 and layers_complete and heads_complete}
    with ledger.tx():ledger.set('phase_progress_cache',{'key':cache_key,'value':result})
    return result
