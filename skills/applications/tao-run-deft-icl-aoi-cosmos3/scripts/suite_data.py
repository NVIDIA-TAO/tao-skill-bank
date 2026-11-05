# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trusted role-view preparation and existing-skill embedding/mining composition."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

from tht_contract import CLASSES,load_jsonl,preprocess,sha256,within,validate_sharegpt
from suite_runtime import BANK,json_write

ICL_SCRIPTS=BANK/'skills/data/tao-in-context-learning/scripts'
sys.path.insert(0,str(ICL_SCRIPTS))
from retrieval import allowed_references,embedding_spec,mining_spec,order_mined,messages,policy_hash,validate_policy

def training_view(public,selected_ids,prompt,output):
    public=Path(public); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    pool={r['id']:r for r in load_jsonl(public/'mining/samples.jsonl')}
    if len(set(selected_ids))!=len(selected_ids) or not selected_ids or set(selected_ids)-pool.keys():
        raise ValueError('training must select unique existing Mining identities')
    annotations=[]; provenance=[]
    for index,identity in enumerate(sorted(selected_ids)):
        row=pool[identity]
        image=within(public,row['image'])
        if sha256(image)!=row['image_sha256']: raise ValueError('Mining image changed')
        target=output/f'image-{index:05d}.png'
        info=preprocess(image,target)
        annotations.append({'id':identity,'images':['/input/'+target.name],
                            'conversations':[{'from':'human','value':prompt},
                                             {'from':'gpt','value':json.dumps({'labels':row['labels']},separators=(',',':'))}]})
        provenance.append({'id':identity,**info})
    validate_sharegpt(annotations,'tht_single')
    json_write(output/'train.json',annotations)
    json_write(output/'provenance.json',provenance)
    return output/'train.json'

def identity_from_query(row):
    """Label-free source filename grammar, checked against every Mining row."""
    import re
    match=re.fullmatch(r'(?:[a-z_]+__)?SCN_(.+?)_(MBO_.+)_A([12])(?:_[0-9a-f]+)?',Path(row['id']).stem)
    if not match: raise ValueError('query filename does not match validated source identity grammar')
    scan,obj,_=match.groups()
    return {'id':row['id'],'source_id':row['id'],'image_sha256':row['image_sha256'],
            'scan_id':scan,'object_name':obj}

def validate_identity_grammar(mining,queries):
    for row in mining:
        identity=identity_from_query(row)
        if identity['object_name']!=row['object_name'] or identity['scan_id']!=row['scan_id'].removeprefix('SCN_'):
            raise ValueError('source filename identity disagrees with Mining metadata')
    return {row['id']:identity_from_query(row) for row in queries}


def query_view(public,queries,prompt,output,*,policy=None,references=None):
    public=Path(public); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    requests=[]; provenance=[]
    for index,query in enumerate(queries):
        source=within(public,query['image'])
        if sha256(source)!=query['image_sha256']: raise ValueError('query image changed')
        name=f'q-{index:05d}.png'
        info=preprocess(source,output/name)
        view='/input/'+name
        selected=[]
        if policy is not None:
            for j,ref in enumerate(references[query['id']]):
                reference=dict(ref)
                rname=f'q-{index:05d}-r-{j:03d}.png'
                rsource=within(public,ref['image'])
                if sha256(rsource)!=ref['image_sha256']: raise ValueError('reference image changed')
                reference['preprocess']=preprocess(rsource,output/rname)
                reference['view_path']='/input/'+rname
                selected.append(reference)
            content=messages(policy,view,selected,prompt)
        else:
            content=[{'role':'user','content':[{'type':'image','image':view},{'type':'text','text':prompt}]}]
        request={'id':query['id'],'messages':content,'references':[r['id'] for r in selected]}
        if policy: request['policy_sha256']=policy_hash(policy)
        requests.append(request)
        provenance.append({'id':query['id'],**info,'references':[{k:v for k,v in r.items() if k!='embedding'} for r in selected]})
    json_write(output/'requests.json',requests)
    # The classifier mount contains only its actual allowed requests/images.
    json_write(output.parent/(output.name+'-provenance.json'),provenance)
    return output/'requests.json'

class RetrievalToolset:
    def __init__(self,task,run_job):
        self.task=task; self.run_job=run_job
        self.root=Path(task['results_dir'])/'retrieval'; self.root.mkdir(parents=True,exist_ok=True)
        self.public=Path(task['public_package'])
        self.mining=load_jsonl(self.public/'mining/samples.jsonl')
    def _run_spec(self,stage,command,spec,inputs,output_name):
        import yaml
        directory=self.root/stage; directory.mkdir(parents=True,exist_ok=True)
        config=directory/'spec.yaml'; config.write_text(yaml.safe_dump(spec,sort_keys=False))
        return self.run_job('retrieval/'+stage,command+[str(config)],image=self.task['data_services_image'],
                            entrypoint=command[0],strip_entrypoint=True,
                            mounts=[(directory,str(directory),False),*inputs],
                            required_output=directory/output_name)
    def embed(self,role,rows):
        import pyarrow as pa
        import pyarrow.parquet as pq
        name='embed-'+role; directory=self.root/name; directory.mkdir(parents=True,exist_ok=True)
        records=[]
        for row in rows:
            item={k:v for k,v in row.items() if k!='labels'}
            source=within(self.public,row['image'])
            if sha256(source)!=row['image_sha256']: raise ValueError('embedding source image changed')
            view=directory/'images'/(row['image_sha256']+'.png')
            if not view.exists(): preprocess(source,view)
            item['filepath']=str(view)
            item['embedding_view_sha256']=sha256(view)
            records.append(item)
        inp=directory/'input.parquet'; out=directory/'embeddings.parquet'
        if not inp.exists(): pq.write_table(pa.Table.from_pylist(records),inp)
        spec=embedding_spec(inp,out,self.task['siglip_path'])
        # Validate with the original producer skill before its native launch.
        import yaml,subprocess
        path=directory/'spec.yaml'; path.write_text(yaml.safe_dump(spec))
        subprocess.run([sys.executable,str(BANK/'skills/data/tao-generate-image-embeddings/scripts/verify_image_embeddings_spec.py'),'--spec',str(path)],check=True)
        self._run_spec(name,['embedding','image_embeddings','-e'],spec,
                       [(self.task['siglip_path'],self.task['siglip_path'],True)],'embeddings.parquet')
        table=pq.read_table(out).to_pylist()
        if len(table)!=len(rows) or any('embedding' not in r for r in table): raise ValueError('embedding output contract failed')
        return table
    def references(self,role,queries,policy,capacity):
        import pyarrow as pa
        import pyarrow.parquet as pq
        import subprocess,yaml
        validate_policy(policy,capacity)
        source=self.embed('mining',self.mining)
        target=self.embed(role,queries)
        source_by={r['id']:r for r in source}; target_by={r['id']:r for r in target}
        identities=validate_identity_grammar(self.mining,queries)
        result={}
        from tht_contract import json_hash
        cache_inputs={'source':sha256(self.root/'embed-mining/embeddings.parquet'),
                      'target':sha256(self.root/('embed-'+role)/'embeddings.parquet'),
                      'topn':capacity,'metric':'cosine','exclusion':'identity-bytes-normalized-joint-v1',
                      'siglip_revision':self.task['siglip_revision'],'image':self.task['data_services_image']}
        for index,query in enumerate(queries):
            permitted=allowed_references(identities[query['id']],self.mining)
            candidates=[{**r,**source_by[r['id']]} for r in permitted]
            key=json_hash({**cache_inputs,'query_id':query['id'],'allowed_ids':[r['id'] for r in permitted]})
            name=f'{role}-{key[:20]}'
            directory=self.root/name; directory.mkdir(parents=True,exist_ok=True)
            src=directory/'source.parquet'; tgt=directory/'target.parquet'; out=directory/'mined.parquet'
            if not src.exists(): pq.write_table(pa.Table.from_pylist(candidates),src)
            if not tgt.exists(): pq.write_table(pa.Table.from_pylist([target_by[query['id']]]),tgt)
            spec=mining_spec(src,tgt,out,capacity)
            json_write(directory/'cache-key.json',{**cache_inputs,'query_id':query['id'],'sha256':key})
            spec_path=directory/'spec.yaml'; spec_path.write_text(yaml.safe_dump(spec))
            subprocess.run([sys.executable,str(BANK/'skills/data/tao-mine-nearest-neighbors/scripts/verify_nearest_neighbors_spec.py'),'--spec',str(spec_path)],check=True)
            self._run_spec(name,['tmm','nearest_neighbors','-e'],spec,[],'mined.parquet')
            result[query['id']]=order_mined(policy,target_by[query['id']]['embedding'],pq.read_table(out).to_pylist(),candidates)
        return result
