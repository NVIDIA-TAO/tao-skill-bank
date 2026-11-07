# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Post-freeze query preparation/native embedding. This module never loads gold."""
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
from wsp03_io import json_write as private_json_write


def json_write(path,value):
    return private_json_write(path,value,mode=0o640)

from wsp03_bindings import file_hash
from wsp03_worker import digest


def prepare(plan,output):
    if set(plan.get('both_freeze_receipts',{}))!={'R3','R4'}:raise ValueError('both freeze receipts required before source access')
    from tht_contract import preprocess,within
    from suite_data import validate_identity_grammar
    import pyarrow as pa
    import pyarrow.parquet as pq
    public=Path(plan['public_package']).resolve(strict=True);output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('new immutable final input directory required')
    source=public/'mining/samples.jsonl';queries_path=public/'benchmark/queries.jsonl'
    if file_hash(source)!=plan['mining_sha256'] or file_hash(queries_path)!=plan['benchmark_queries_sha256']:raise ValueError('frozen final source changed')
    mining=[json.loads(line) for line in source.read_text().splitlines()];queries=[json.loads(line) for line in queries_path.read_text().splitlines()]
    if len(mining)!=plan['mining_count'] or len(queries)!=plan['benchmark_count']:raise ValueError('source membership count changed')
    if any(set(row)!={'id','image','image_sha256'} for row in queries):raise ValueError('Benchmark query must be label-free')
    identities=validate_identity_grammar(mining,queries);cache=Path(plan['dev_embeddings'])
    if file_hash(cache)!=plan['dev_embeddings_sha256']:raise ValueError('dev embeddings changed')
    cached=[json.loads(line) for line in cache.read_text().splitlines()];dev={r['id'] for r in cached};pool={r['id'] for r in mining}
    if len(dev)!=plan['dev_count'] or not dev<=pool or len(pool)!=len(mining) or pool&{r['id'] for r in queries}:raise ValueError('invalid final identities')
    rows={r['id']:dict(r,source_id=r['id']) for r in mining}
    rows.update({r['id']:dict(r,**{k:v for k,v in identities[r['id']].items() if k not in r}) for r in queries})
    output.mkdir(parents=True,exist_ok=True);(output/'images').mkdir();images={};provenance={};pending=[]
    for qid,row in sorted(rows.items()):
        image=within(public,row['image'])
        if file_hash(image)!=row['image_sha256']:raise ValueError('source bytes changed')
        name=digest(qid)+'.png';info=preprocess(image,output/'images'/name)
        images[qid]='images/'+name;provenance[qid]=info;row['joint']=[row['object_name'],row['scan_id'].removeprefix('SCN_')]
        if qid not in dev:pending.append({'id':qid,'filepath':'/prepared/images/'+name,'image_sha256':row['image_sha256'],'embedding_view_sha256':info['view_sha256']})
    parents={digest(r['joint']):digest(r['joint']) for r in rows.values()};seen={}
    def find(k):
        while parents[k]!=k:k=parents[k]
        return k
    for row in rows.values():
        key=digest(row['joint']);a,b=find(key),find(seen.setdefault(row['image_sha256'],key))
        if a!=b:parents[max(a,b)]=min(a,b)
    for row in rows.values():row['component']=find(digest(row['joint']))
    if plan.get('holdout_identity_file'):
        identity_path=Path(plan['holdout_identity_file'])
        if file_hash(identity_path)!=plan['holdout_identity_sha256']:raise ValueError('holdout identity source changed')
        identity_rows=[json.loads(line) for line in identity_path.read_text().splitlines()]
        if any('labels' in r or 'truth' in r for r in identity_rows):raise ValueError('holdout identity input must be label-free')
        holdout=sorted(r['id'] for r in identity_rows)
        excluded=pool-dev-set(holdout)
        if len(excluded)!=plan.get('excluded_mining_count',2) or len(set(holdout))!=len(holdout) or set(holdout)&dev or not set(holdout)<=pool:raise ValueError('frozen partition difference mismatch')
    else:holdout=sorted(pool-dev-set(plan['excluded_mining_ids']))
    if len(holdout)!=plan['holdout_count'] or plan.get('holdout_ids_sha256') and digest(holdout)!=plan['holdout_ids_sha256']:raise ValueError('frozen holdout membership changed')
    pq.write_table(pa.Table.from_pylist(pending),output/'embedding-input.parquet')
    result={'schema':'wsp03-final-inputs/1','rows':rows,'images':images,'provenance':provenance,'mining_ids':[r['id'] for r in mining],'development_ids':[r['id'] for r in cached],'holdout_ids':holdout,'benchmark_ids':[r['id'] for r in queries],'missing_embedding_ids':[r['id'] for r in pending],'both_freeze_receipts':plan['both_freeze_receipts'],'gold_read':False}
    json_write(output/'prepared.json',result);return {'prepared_sha256':file_hash(output/'prepared.json'),'missing_embeddings':len(pending),'gold_read':False}


def embed(spec,output):
    import pyarrow.parquet as pq
    subprocess.run(['embedding','image_embeddings','-e',str(spec)],check=True)
    rows=[{'id':r['id'],'embedding':r['embedding']} for r in pq.read_table(Path(output)/'embeddings.parquet').to_pylist()]
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('duplicate native embedding IDs')
    with (Path(output)/'embeddings.jsonl').open('x') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    return {'rows':len(rows),'embeddings_sha256':file_hash(Path(output)/'embeddings.jsonl')}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','embed']);parser.add_argument('--plan');parser.add_argument('--spec');parser.add_argument('--output',required=True);args=parser.parse_args()
    result=prepare(json.loads(Path(args.plan).read_text()),args.output) if args.action=='prepare' else embed(args.spec,args.output)
    print(json.dumps(result))
