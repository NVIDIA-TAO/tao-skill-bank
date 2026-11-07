# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Concrete trusted data, retrieval, rendering, scoring and native usage adapters.

This module never reads a Benchmark path at construction. Trusted final views
must be attached explicitly after both tracks freeze; development inputs are
exactly the recipe's hash-bound prepared view and embedding cache.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from wsp03_io import json_write
from wsp03_worker import digest


def image_name(identity):
    return hashlib.sha256(identity.encode()).hexdigest()+'.png'


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def pinned(path,expected=None,sha256=None):
    expected=expected or sha256
    path=Path(path).resolve(strict=True)
    if file_hash(path)!=expected:raise ValueError('frozen asset changed: '+path.name)
    return path


PROFILES={
 'P00':(0,'none','desc',False),'P01':(1,'nearest','desc',False),'P02':(2,'nearest','desc',False),
 'P03':(4,'nearest','desc',False),'P04':(4,'nearest','asc',False),'P05':(1,'helpful','desc',False),
 'P06':(4,'helpful','desc',False),'P07':(4,'balanced','desc',False),'P08':(4,'atlas','desc',False),
 'P09':(4,'hybrid','desc',False),'P10':(1,'nearest','desc',True),'P11':(1,'helpful','desc',True)}


class ReferenceSelector:
    def __init__(self,rows,embeddings,*,helpfulness,atlas,pool=None):
        self.rows,self.embeddings,self.helpfulness,self.atlas=rows,embeddings,helpfulness,set(atlas)
        self.pool=set(pool if pool is not None else rows);self.cache={}
        if not self.pool<=set(rows) or not self.pool<=set(embeddings):raise ValueError('incomplete frozen pool/embeddings')
        byclass={}
        for rid,h in helpfulness.items():
            if rid not in rows or not math.isfinite(h):raise ValueError('helpfulness must be finite and dev-derived')
            byclass.setdefault(rows[rid]['labels'][0],[]).append(h)
        self.classmean={key:sum(values)/len(values) for key,values in byclass.items()}
        self.globalmean=sum(helpfulness.values())/len(helpfulness) if helpfulness else 0.
    def cosine(self,a,b):
        x,y=self.embeddings[a],self.embeddings[b]
        if len(x)!=len(y):raise ValueError('embedding dimensions mismatch')
        norm=math.sqrt(sum(v*v for v in x)*sum(v*v for v in y))
        if not math.isfinite(norm) or norm<=0:raise ValueError('nonfinite/zero embedding')
        value=sum(v*w for v,w in zip(x,y))/norm
        if not math.isfinite(value):raise ValueError('nonfinite cosine')
        return max(-1.,min(1.,value))
    def allowed(self,qid,rid):
        q,r=self.rows[qid],self.rows[rid]
        return rid!=qid and all(q.get(k,qid)!=r.get(k,rid) for k in ('source_id','image_sha256','component','joint'))
    def select(self,qid,profile):
        if profile not in PROFILES:raise ValueError('unregistered retrieval strategy')
        k,selector,ordering,_=PROFILES[profile]
        if k==0:return []
        if qid not in self.cache:
            eligible=[];seen=set()
            for rid in sorted(self.pool):
                key=self.rows[rid]['image_sha256']
                if self.allowed(qid,rid) and key not in seen:eligible.append(rid);seen.add(key)
            self.cache[qid]={rid:self.cosine(qid,rid) for rid in eligible}
        similarities=self.cache[qid]
        nearest=sorted(similarities,key=lambda rid:(-similarities[rid],rid))
        def score(rid):
            h=self.helpfulness.get(rid,self.classmean.get(self.rows[rid]['labels'][0],self.globalmean))
            return ((similarities[rid]+1)/2)*math.exp(max(-2,min(2,h)))
        helpful=sorted(similarities,key=lambda rid:(-score(rid),rid))
        if selector=='nearest':selected=nearest[:k]
        elif selector=='helpful':selected=helpful[:k]
        elif selector=='balanced':
            selected=[];labels=set()
            for rid in nearest:
                label=self.rows[rid]['labels'][0]
                if label not in labels:selected.append(rid);labels.add(label)
                if len(selected)==k:break
        elif selector=='atlas':selected=[rid for rid in nearest if rid in self.atlas][:k]
        elif selector=='hybrid':selected=nearest[:2]+[rid for rid in helpful if rid not in nearest[:2]][:2]
        else:raise ValueError('unknown selector')
        if not selected or len(selected)<k and selector not in ('balanced','atlas','hybrid'):raise ValueError('insufficient valid references under frozen strategy')
        if ordering=='asc':selected=sorted(selected,key=lambda rid:(similarities[rid],rid))
        return selected


class Recipe:
    def __init__(self,path,expected_sha256):
        self.path=pinned(path,expected_sha256);self.config=json.loads(self.path.read_text())
        if self.config.get('schema')!='wsp03-recipe/1':raise ValueError('unsupported recipe')
        self.rows={};self.images={};self.gold={};self.members={}
        self.load_view('development',self.config['development_view'])
        explore=json.loads(pinned(self.config['explore_ids']['path'],self.config['explore_ids']['sha256']).read_text())
        if not set(explore)<=set(self.rows) or len(set(explore))!=len(explore):raise ValueError('explore membership outside dev')
        self.members['explore']=explore
        asset=self.config['embeddings'];embedding_rows=[json.loads(line) for line in pinned(asset['path'],asset['sha256']).read_text().splitlines()]
        embeddings={r['id']:r['embedding'] for r in embedding_rows}
        if len(embeddings)!=len(embedding_rows) or set(embeddings)!=set(self.rows):raise ValueError('development embedding membership mismatch')
        self.selector=ReferenceSelector(self.rows,embeddings,helpfulness={},atlas=[],pool=self.members['development'])
        self.prompt=pinned(**self.config['prompt']).read_text();self.cues=pinned(**self.config['class_cues']).read_text()
        adapter=pinned(**self.config['retrieval_adapter'])
        spec=importlib.util.spec_from_file_location('_wsp03_frozen_retrieval',adapter);self.retrieval=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.retrieval)
        self.helpfulness_frozen=False;self.cached_c={};self.cached_c_images={};self.lazy_gold={}
    def load_view(self,role,asset):
        directory=Path(asset['directory']).resolve(strict=True)
        files=asset['files']
        for name,expected in files.items():pinned(directory/name,expected)
        query={r['id']:r for r in self.jsonl(directory/'queries.jsonl')};sources=self.jsonl(directory/'source_rows.jsonl')
        provenance={r['id']:r for r in self.jsonl(directory/'provenance.jsonl')};gold=self.jsonl(directory/'gold.jsonl')
        if set(query)!={r['id'] for r in sources} or set(query)!={r['id'] for r in gold}:raise ValueError('view identities disagree')
        self.members[role]=[r['id'] for r in sources]
        # Component map comes from the frozen builder, not guessed from labels.
        for row in sources:
            qid=row['id'];value=dict(row,source_id=qid,joint=provenance[qid]['joint'],component=provenance[qid].get('component',provenance[qid]['joint']))
            self.rows[qid]=value;image=(directory/query[qid]['image']).resolve(strict=True)
            if not image.is_relative_to(directory) or file_hash(image)!=query[qid]['image_sha256']:raise ValueError('view image changed/escaped')
            self.images[qid]=image
        self.gold[role]=gold
        parents={digest(row['joint']):digest(row['joint']) for row in self.rows.values()};seen={}
        def find(key):
            while parents[key]!=key:key=parents[key]
            return key
        for row in self.rows.values():
            key=digest(row['joint']);other=seen.setdefault(row['image_sha256'],key);a,b=find(key),find(other)
            if a!=b:parents[max(a,b)]=min(a,b)
        for row in self.rows.values():row['component']=find(digest(row['joint']))
    @staticmethod
    def jsonl(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]
    def freeze_helpfulness_from_ledger(self,ledger,output):
        if self.helpfulness_frozen:raise ValueError('helpfulness already frozen')
        observations=ledger.get('attribution_pairs') or {};groups={}
        for row in observations.values():
            if row['query_id'] not in self.members['explore'] or row['reference_id'] not in self.members['development']:raise ValueError('attribution outside approved dev')
            groups.setdefault(row['reference_id'],[]).append(row['delta'])
        h={rid:sum(values)/len(values) for rid,values in groups.items()};atlas=[]
        classes=sorted({row['labels'][0] for row in self.rows.values()})
        for label in classes:
            seen=set();selected=[]
            for rid in sorted((r for r in h if self.rows[r]['labels']==[label]),key=lambda r:(-h[r],r)):
                component=self.rows[rid]['component']
                if component in seen:continue
                selected.append(rid);seen.add(component)
                if len(selected)==4:break
            atlas.extend(selected)
        self.selector=ReferenceSelector(self.rows,self.selector.embeddings,helpfulness=h,atlas=atlas,pool=self.members['development']);self.helpfulness_frozen=True
        artifact={'schema':'wsp03-helpfulness/1','helpfulness':h,'atlas':atlas,'pair_observations':observations,
            'coverage':{rid:len(values) for rid,values in groups.items()},'development_only':True,'imputation':'equal-reference class mean, then global, then zero if no observations'}
        json_write(output,artifact);return artifact

    def attribution_candidates(self,qid):
        self.selector.select(qid,'P01');similarities=self.selector.cache[qid]
        nearest=sorted(similarities,key=lambda rid:(-similarities[rid],rid))[:8]
        selected=list(nearest);label=self.rows[qid]['labels'][0]
        for same in (True,False):
            eligible=[rid for rid in similarities if rid not in selected and ((self.rows[rid]['labels'][0]==label)==same)]
            eligible.sort(key=lambda rid:(digest({'seed':42,'purpose':'wsp03-attribution','query_id':qid,'reference_id':rid}),rid))
            selected.extend(eligible[:2])
        return selected
    def references(self,qid,profile):
        if profile=='B04':return []
        if getattr(self,'final_pools',None):
            pool=self.final_pools['benchmark' if qid in self.members.get('benchmark',[]) else 'holdout']
            return pool.select(qid,profile)
        return self.selector.select(qid,profile)
    def render(self,qid,refs,mode,profile='P00'):
        if qid not in self.rows:raise ValueError('query outside attached trusted view')
        if profile=='P01' and qid in self.cached_c:
            row=self.cached_c[qid]
            if row.get('references')!=refs:raise ValueError('historical C exact-reference parity failed')
            return dict(row)
        prompt=self.prompt
        if profile=='B04' or PROFILES[profile][3]:prompt=self.cues+'\n\n'+prompt
        query_path='/input/'+image_name(qid)
        if refs:
            references=[dict(self.rows[rid],view_path='/input/'+image_name(rid)) for rid in refs]
            messages=self.retrieval.messages({'parameters':{'instruction':self.config['icl_instruction']}},query_path,references,prompt)
        else:messages=[{'role':'user','content':[{'type':'image','image':query_path},{'type':'text','text':prompt}]}]
        row={'id':qid,'messages':messages,'references':refs,'policy_sha256':digest({'profile':profile,'instruction':self.config['icl_instruction'],'prompt':prompt})}
        if mode=='read':row['forced_text']=json.dumps({'labels':self.rows[qid]['labels']},separators=(',',':'))
        return row
    def __call__(self,qid,refs,mode):return self.render(qid,refs,mode)
    def request_images(self,rows):
        ordinary={'/input/'+image_name(qid):path for qid,path in self.images.items()};ordinary.update(self.cached_c_images)
        result={}
        for row in rows:
            for message in row['messages']:
                for part in message['content']:
                    if part.get('type')!='image':continue
                    key=part['image'];relative=Path(key)
                    if not relative.is_relative_to('/input') or '..' in relative.parts or key not in ordinary:raise ValueError('request image outside explicit allowlist')
                    source=Path(ordinary[key])
                    if source.is_symlink():raise ValueError('request image symlink forbidden')
                    result[str(relative.relative_to('/input'))]=source
        return result

    def contract(self):
        return {'schema':'wsp03-service-contract/1','recipe_sha256':file_hash(self.path),'splits':dict(self.members,holdout=[],benchmark=[]),
            'max_references':4,'benchmark_reference_ids':[],
            **{key:{qid:row[column] for qid,row in self.rows.items()} for key,column in [('joints','joint'),('source_ids','source_id'),('image_hashes','image_sha256'),('components','component')]}}


class TrustedGrader:
    def __init__(self,recipe):
        from tht_contract import external_evaluator
        self.recipe=recipe;self.evaluator=external_evaluator(recipe.config['evaluator_directory'])
    def __call__(self,result,split):
        from tht_contract import grade
        ids=set(result['ids'])
        if split in self.recipe.lazy_gold and split not in self.recipe.gold:
            asset=self.recipe.lazy_gold[split];path=Path(asset['path'])
            if asset.get('sha256'):pinned(path,asset['sha256'])
            self.recipe.gold[split]=self.recipe.jsonl(path)
            json_write(Path(result['output_directory'])/'trusted-gold-binding.json',{'split':split,'sha256':file_hash(path),'ids_sha256':digest([r['id'] for r in self.recipe.gold[split]])})
        gold=[r for r in self.recipe.gold['development' if split=='explore' else split] if r['id'] in ids]
        scored=grade(result['raw'],gold,self.evaluator);metrics=scored['metrics']
        json_write(Path(result['output_directory'])/'trusted-score.json',scored)
        # Preserve the frozen evaluator's actual metric keys, while supplying
        # the ranking aliases expected by the generic ledger.
        invalid=metrics.get('invalid',metrics.get('invalid_count'))
        if invalid is None:invalid=sum(r['prediction'] not in self.evaluator.CLASSES for r in scored['predictions'])
        return dict(metrics,complete=True,n=len(gold),mcc=metrics['mcc'],macro_f1=metrics['macro_f1'],invalid=invalid)


class NativeWorker:
    def __init__(self,recipe,runtime,lease,ledger,edit_root,output,*,artifact_owner=None, read_root=None, frozen_manifest=None):
        self.recipe,self.runtime,self.lease,self.ledger=recipe,runtime,lease,ledger
        self.edit_root,self.output=Path(edit_root),Path(output)
        self.frozen_manifest=frozen_manifest
        self.artifact_owner=runtime.uid if artifact_owner is None else artifact_owner
        self.read_root=Path(read_root) if read_root else self.output/"read-artifacts"
    def registered_manifest(self,candidate):
        if self.frozen_manifest is not None:return self.frozen_manifest
        row=self.ledger.db.execute("SELECT payload FROM candidates WHERE id=?",(candidate,)).fetchone()
        if not row:raise ValueError("unregistered candidate")
        return json.loads(row[0])
    def checkpoint_identity(self,candidate):
        base=Path(self.recipe.config['model_directory']).resolve(strict=True)
        from wsp03_artifacts import child_json
        manifest=None
        if candidate!='base':
            value=child_json(self.edit_root,str(Path(candidate)/'manifest.json'),owner=self.artifact_owner)
            if value!=self.registered_manifest(candidate):raise ValueError('candidate changed after registration')
            manifest=digest(value)
        return digest({'base_directory':str(base),'candidate_manifest':manifest})
    def image_identity(self,rows):
        selected={row.get('query_id',row['id']) for row in rows}|{ref for row in rows for ref in row.get('references',[])}
        return digest({qid:file_hash(self.recipe.images[qid]) for qid in sorted(selected)})
    def source_capture_bytes(self,rows):
        from wsp03_artifacts import child_file
        sizes={}
        for row in rows:
            for patch in row.get('capture',{}).get('patches',[]):
                source=patch.get('source_capture')
                if not source:continue
                identity=(source['request_id'],source['file'])
                if identity in sizes:continue
                rid,name=identity
                if len(rid)!=32 or any(c not in '0123456789abcdef' for c in rid) or Path(name).name!=name:raise ValueError('invalid bound capture path')
                with child_file(self.read_root,rid+'/'+name,owner=os.getuid(),max_bytes=256*1024**2) as stream:sizes[identity]=os.fstat(stream.fileno()).st_size
        return sum(sizes.values())
    def __call__(self,kind,rows,candidate,split,rid):
        from contextlib import nullcontext
        resources=getattr(self.runtime,'resources',None)
        images=self.recipe.request_images(rows)
        image_bytes=sum(path.stat().st_size for path in images.values())
        patch_bytes=sum(item['elements']*2 for item in self.registered_manifest(candidate)['tensors']) if candidate!='base' else 0
        # Worker raw output, duplicated retained capture, and copied inputs/patch.
        amount=int(image_bytes+patch_bytes+self.source_capture_bytes(rows)+2*1024**3)
        with resources.reserve(amount,'native worker inputs/output/capture',self.runtime.resource_account) if resources else nullcontext():
            return self.execute(kind,rows,candidate,split,rid)
    def execute(self,kind,rows,candidate,split,rid):
        from wsp03_runtime import APP,gated_job,validate_worker_output
        directory=self.output/rid;directory.mkdir(parents=True);inputs=directory/'input';inputs.mkdir()
        selected=set()
        for row in rows:
            selected.add(row.get('query_id',row['id']));selected.update(row.get('references',[]))
        for relative,source in self.recipe.request_images(rows).items():
            destination=inputs/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            if not destination.resolve().is_relative_to(inputs.resolve()):raise ValueError('request input path escape')
            shutil.copyfile(source,destination)
        request=directory/'requests.json';json_write(request,rows,mode=0o640)
        subprocess.run(['sudo','-n','chgrp',str(self.runtime.gid),str(request)],check=True)
        mode='read' if kind=='read' else 'generation';command=['/app/scripts/wsp03_worker.py','--model','/model','--requests','/requests.json','--output','/results/worker','--mode',mode,'--split',split]
        mounts=[(APP,'/app',True),(self.recipe.config['model_directory'],'/model',True),(inputs,'/input',True),(request,'/requests.json',True)]
        candidate_hash=None
        captured=directory/'captures';capture_mount=False
        for row in rows:
            for patch in row.get('capture',{}).get('patches',[]):
                source=patch.get('source_capture')
                if not source:continue
                from wsp03_artifacts import child_file,bounded_copy
                rid_source=source['request_id'];name=source['file']
                if len(rid_source)!=32 or any(c not in '0123456789abcdef' for c in rid_source) or Path(name).name!=name:raise ValueError('invalid bound capture path')
                target=captured/rid_source/name;target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():
                    with child_file(self.read_root,rid_source+'/'+name,owner=os.getuid(),max_bytes=256*1024**2) as src,target.open('xb') as dst:bounded_copy(src,dst,max_bytes=256*1024**2)
                capture_mount=True
        if capture_mount:mounts.append((captured,'/captures',True))
        if candidate!='base':
            from wsp03_artifacts import stage_candidate
            staged=directory/'candidate'
            candidate_hash=digest(stage_candidate(self.edit_root,candidate,staged,owner=self.artifact_owner,group=self.runtime.gid,expected_manifest=self.registered_manifest(candidate)))
            command+=['--candidate','/candidate'];mounts.append((staged,'/candidate',True))
        remaining=self.ledger.forward_seconds(kind)
        started=time.monotonic()
        job=gated_job(self.runtime,self.lease,'wsp03-'+kind,command,mounts,deadline=time.time()+remaining)
        output=Path(job['results_dir'])/'worker'
        from wsp03_artifacts import child_bytes
        raw=[json.loads(line) for line in child_bytes(job['results_dir'],'worker/raw_answers.jsonl',owner=self.runtime.uid,max_bytes=512*1024**2).decode().splitlines()]
        validate_worker_output(output,rows,mode=mode,split=split,candidate_manifest_sha256=candidate_hash,artifact_owner=self.runtime.uid)
        complete=True
        if mode=='read' and complete:
            retained=self.read_root/rid;retained.mkdir(parents=True);records={}
            subprocess.run(['sudo','-n','chgrp',str(self.runtime.gid),str(retained)],check=True);retained.chmod(0o750)
            for row in raw:
                if not row.get('capture_file'):continue
                from wsp03_artifacts import child_file,bounded_copy
                name=row['capture_file']
                if Path(name).name!=name:raise ValueError('capture artifact escape')
                with child_file(job['results_dir'],'worker/'+name,owner=self.runtime.uid,max_bytes=256*1024**2) as source,(retained/name).open('xb') as target:bounded_copy(source,target,max_bytes=256*1024**2)
                (retained/name).chmod(0o440)
                row['capture_sha256']=file_hash(retained/name)
                subprocess.run(['sudo','-n','chgrp',str(self.runtime.gid),str(retained/name)],check=True)
                records[row['id']]={'file':name,'sha256':file_hash(retained/name),'capture':row['capture']}
            json_write(retained/'manifest.json',{'request_id':rid,'candidate':candidate,'records':records},mode=0o640)
            subprocess.run(['sudo','-n','chgrp',str(self.runtime.gid),str(retained/'manifest.json')],check=True)
        result={'complete':complete,'native_job_wall_seconds':time.monotonic()-started,'ids':[r['id'] for r in raw],'requests_sha256':digest(rows),'raw':raw,'output_directory':str(output),'job':job}
        json_write(directory/'result.json',result);return result
