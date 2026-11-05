# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trusted ICL boundaries and adapters for existing TAO embedding/TMM outputs."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

TOOLSET='siglip-retrieval-v1'

def policy_hash(policy):
    return hashlib.sha256(json.dumps(policy,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def validate_policy(policy,capacity):
    if policy.get('toolset')!=TOOLSET or policy.get('version')!=1:
        raise ValueError('unsupported toolset/version; register a new implementation explicitly')
    if not isinstance(policy.get('rationale'),str) or not policy['rationale'].strip():
        raise ValueError('an evidence-based proposal rationale is required')
    if policy.get('max_calls')!=1: raise ValueError('retrieval v1 performs exactly one model call')
    params=policy['parameters']
    if set(params)!={'reference_count','ordering','instruction'}:
        raise ValueError('unknown or missing retrieval parameter')
    if type(params['reference_count']) is not int or not 1<=params['reference_count']<=capacity:
        raise ValueError('reference count exceeds measured capacity')
    if params['ordering'] not in ('nearest_first','farthest_first','class_grouped') or not isinstance(params['instruction'],str):
        raise ValueError('invalid presentation policy')
    return policy

def joint(row):
    if not row.get('object_name') or not row.get('scan_id'):
        raise ValueError('trusted normalized joint identity is required')
    return row['object_name'],row['scan_id'].removeprefix('SCN_')

def allowed_references(query, references):
    identity=query.get('source_id',query['id'])
    qjoint=joint(query)
    if not query.get('image_sha256'): raise ValueError('query source image hash is required')
    result=[]; seen=set(); seen_bytes=set()
    for ref in references:
        if ref.get('role','mining')!='mining': raise ValueError('reference is not Mining')
        rid=ref.get('source_id',ref['id'])
        rjoint=joint(ref)
        if not ref.get('image_sha256'): raise ValueError('reference source image hash is required')
        if rid==identity or ref['id']==query['id'] or ref['image_sha256']==query['image_sha256'] or rjoint==qjoint:
            continue
        if rid in seen or ref['image_sha256'] in seen_bytes: continue
        if not isinstance(ref.get('labels'),list) or len(ref['labels'])!=1: raise ValueError('exact single-label references required')
        seen.add(rid); seen_bytes.add(ref['image_sha256']); result.append(ref)
    return result

def cosine(a,b):
    if len(a)!=len(b) or not a: raise ValueError('embedding dimensions disagree')
    denominator=math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))
    if not denominator or not math.isfinite(denominator): raise ValueError('invalid embedding norm')
    return sum(x*y for x,y in zip(a,b))/denominator

def order_mined(policy, query_embedding, mined_rows, candidate_rows):
    """TMM owns selection; this adapter restores stable presentation order."""
    candidates={r['filepath']:r for r in candidate_rows}
    selected=[]
    for row in mined_rows:
        if row['filepath'] not in candidates: raise ValueError('miner emitted forbidden reference')
        value=dict(candidates[row['filepath']])
        value['similarity']=cosine(query_embedding,value['embedding'])
        selected.append(value)
    if len({r['id'] for r in selected})!=len(selected): raise ValueError('duplicate mined references')
    count=policy['parameters']['reference_count']
    if len(selected)<count: raise ValueError('insufficient allowed references')
    selected=sorted(selected,key=lambda r:(-r['similarity'],r['id']))[:count]
    mode=policy['parameters']['ordering']
    if mode=='farthest_first': selected=sorted(selected,key=lambda r:(r['similarity'],r['id']))
    elif mode=='class_grouped': selected=sorted(selected,key=lambda r:(r['labels'][0],-r['similarity'],r['id']))
    return selected

def messages(policy, query_path, references, prompt):
    content=[]
    instruction=policy['parameters']['instruction']
    if instruction: content.append({'type':'text','text':instruction})
    for number,ref in enumerate(references,1):
        content.extend([{'type':'text','text':f'Reference {number}: '+json.dumps({'labels':ref['labels']},separators=(',',':'))},
                        {'type':'image','image':ref['view_path']}])
    content.extend([{'type':'text','text':'Query:'},{'type':'image','image':query_path},{'type':'text','text':prompt}])
    return [{'role':'user','content':content}]

def embedding_spec(input_parquet,output_parquet,model_path,batch_size=32):
    return {'input_parquet':str(input_parquet),'output_parquet':str(output_parquet),'model':'SigLIP',
            'model_path':str(model_path),'model_config_path':'','batch_size':batch_size}

def mining_spec(source,target,output,count):
    return {'source_parquet':str(source),'target_parquet':str(target),'output_parquet':str(output),
            'topn':count,'knn_metric':'cosine','source_embed_column_name':'embedding',
            'target_embed_column_name':'embedding','filter_by_label':'false','distance_threshold':-1.0}
