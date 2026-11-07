# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model/dataset-independent numeric atomic operations with JSON receipts.

GPU activation/evaluation operations are routed by the trusted Service. This
module runs in the isolated CPU kernel; it never loads data labels or a model.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import time
import uuid
from wsp03_io import json_write
from wsp03_resources import append_event
from wsp03_worker import digest


class Operations:
    def __init__(self,store):
        self.store=store;self.root=store.output;self.families_path=self.root/'families.json'
        self.families=json.loads(self.families_path.read_text()) if self.families_path.exists() else {}
        self.active=None

    def call(self,operation,params):
        start=time.monotonic();identity=uuid.uuid4().hex
        receipt={'id':identity,'operation':operation,'parameters_sha256':digest(params),'started_at':time.time()}
        append_event(self.root/'atomic-started.jsonl',dict(receipt,parameters=params))
        try:
            value=self.execute(operation,params)
            receipt.update(ok=True,result=value)
        except BaseException as error:
            receipt.update(ok=False,error_type=type(error).__name__,message=str(error));raise
        finally:
            receipt.update(wall_seconds=time.monotonic()-start,ended_at=time.time())
            append_event(self.root/'atomic-receipts.jsonl',receipt)
        return receipt

    def family(self,p):
        required={'id','mechanism','tensors','lineage','configurations'}
        if set(p)!=required or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}',p['id']):raise ValueError('invalid semantic family declaration')
        configs=p['configurations']
        if not configs or len(configs)>5 or len({digest(c) for c in configs})!=len(configs):raise ValueError('at most5 distinct preregistered configurations')
        if not p['lineage'] or not p['tensors'] or any(n not in self.store.index for n in p['tensors']):raise ValueError('known tensors and mechanism lineage required')
        semantic=digest({k:sorted(p[k]) if k in ('tensors','lineage') else p[k] for k in ('mechanism','tensors','lineage')})
        if p['id'] in self.families or any(x['semantic_sha256']==semantic for x in self.families.values()):raise ValueError('family immutable; renaming cannot reset configurations')
        for config in configs:
            if not config or any(type(v) not in (int,float) or not math.isfinite(v) for v in config.values()):raise ValueError('finite preregistered numeric tuple required')
        self.families[p['id']]=dict(p,semantic_sha256=semantic)
        json_write(self.families_path,self.families);return self.families[p['id']]

    def config(self,p):
        family=self.families[p['family']];i=p['configuration']
        if type(i) is not int or not 0<=i<len(family['configurations']) or p['name'] not in family['tensors']:raise ValueError('configuration outside registered family')
        active={'family':p['family'],'semantic_sha256':family['semantic_sha256'],'configuration':i,'values':family['configurations'][i]}
        if self.active is not None and self.active!=active:raise ValueError('reset/commit before changing configuration; no Cartesian composition')
        self.active=active;return active['values']

    def export(self,candidate):
        from safetensors import safe_open
        from safetensors.torch import save_file
        self.store.verify(candidate)
        destination=self.root/'exports'/candidate
        if destination.exists():raise ValueError('immutable export already exists')
        inventory=[p for p in self.store.base.iterdir() if p.is_file()]
        needed=sum(p.stat().st_size for p in inventory)
        used=sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
        if used+needed>self.store.storage_limit:raise ValueError('retained storage budget')
        if any(p.is_symlink() for p in self.store.base.iterdir()):raise ValueError('base export refuses symlinks')
        destination.mkdir(parents=True,mode=0o700)
        manifest=self.store.manifest(candidate);byshard={}
        for item in manifest['tensors']:byshard.setdefault(self.store.index[item['name']],[]).append(item['name'])
        for path in inventory:
            out=destination/path.name
            if path.name in byshard:
                with safe_open(path,framework='pt',device='cpu') as file:
                    tensors={name:self.store.load_tensor(name,candidate) if name in byshard[path.name] else file.get_tensor(name) for name in file.keys()}
                    metadata=file.metadata()
                save_file(tensors,out,metadata=metadata)
            else:shutil.copyfile(path,out)
            out.chmod(0o400)
        shutil.copytree(self.root/candidate,destination/'replay-patch')
        inventory=[{'file':str(p.relative_to(destination)),'size':p.stat().st_size,'sha256':self.file_hash(p)} for p in sorted(destination.rglob('*')) if p.is_file()]
        receipt={'candidate':candidate,'directory':str(destination),'files':inventory,'inventory_sha256':digest(inventory),'format':'HuggingFace checkpoint plus replay-patch'}
        json_write(destination/'export.json',receipt);return receipt

    @staticmethod
    def file_hash(path):
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
        return h.hexdigest()

    def execute(self,operation,p):
        import torch
        from safetensors.torch import save_file
        if operation=='edit.family':return self.family(p)
        if operation=='tensor.read':
            value=self.store.load_tensor(p['name'],p.get('candidate','base'))
            if 'rows' in p:value=value[p['rows']]
            if 'columns' in p:value=value[...,p['columns']]
            if value.numel()*value.element_size()>256*1024**2:raise ValueError('select tensor slice within256MiB')
            name='numeric-'+uuid.uuid4().hex+'.safetensors';save_file({'value':value.contiguous()},self.root/name)
            return {'file':name,'shape':list(value.shape),'dtype':str(value.dtype),'preview':value.flatten()[:32].float().tolist(),'truncated_preview':value.numel()>32}
        if operation in ('edit.direction','edit.rank1','edit.memit','edit.head_scale'):
            config=self.config(p);weight=self.store.load_tensor(p['name']).float()
            alpha=config.get('alpha',1.)
            if operation=='edit.direction':
                direction=torch.as_tensor(p['direction'],dtype=torch.float32)
                if direction.shape!=weight.shape:raise ValueError('direction shape mismatch')
                edited=weight+alpha*direction
            elif operation=='edit.rank1':
                left=torch.as_tensor(p['left'],dtype=torch.float32);right=torch.as_tensor(p['right'],dtype=torch.float32)
                if weight.ndim!=2 or left.shape!=(weight.shape[0],) or right.shape!=(weight.shape[1],):raise ValueError('rank-one vectors mismatch')
                edited=weight+alpha*torch.outer(left,right)
            elif operation=='edit.memit':
                keys=torch.as_tensor(p['keys'],dtype=torch.float64);delta_values=torch.as_tensor(p['delta_values'],dtype=torch.float64)
                covariance=torch.as_tensor(p['covariance'],dtype=torch.float64);rank=config.get('rank')
                if weight.ndim!=2 or keys.ndim!=2 or delta_values.shape!=(weight.shape[0],keys.shape[1]) or keys.shape[0]!=weight.shape[1] or type(rank)is not int or not 1<=rank<=min(keys.shape):raise ValueError('closed-form key/delta-value/rank mismatch')
                if not p.get('covariance_source'):raise ValueError('dev preservation covariance provenance required')
                if rank<keys.shape[1]:
                    _,_,right=torch.linalg.svd(keys,full_matrices=False);project=right[:rank].T
                    keys=keys@project;delta_values=delta_values@project
                if covariance.ndim==1:
                    if covariance.shape!=(weight.shape[1],) or bool((covariance<0).any()):raise ValueError('nonnegative covariance diagonal required')
                    ridge=1e-4*max(covariance.mean().item(),1e-12)
                    inverse_diagonal=1/(covariance+ridge)
                    kt_d=keys.T*inverse_diagonal
                    solution=torch.linalg.solve(torch.eye(rank,dtype=torch.float64)+kt_d@keys,kt_d)
                elif covariance.shape==(weight.shape[1],weight.shape[1]):
                    if not torch.allclose(covariance,covariance.T) or bool((torch.linalg.eigvalsh(covariance)<-1e-10).any()):raise ValueError('positive semidefinite preservation covariance required')
                    ridge=1e-4*max(covariance.diagonal().mean().item(),1e-12)
                    system=covariance+keys@keys.T+ridge*torch.eye(weight.shape[1],dtype=torch.float64)
                    solution=torch.linalg.solve(system,keys).T
                else:raise ValueError('covariance dimensions mismatch')
                if not all(bool(torch.isfinite(v).all()) for v in (keys,delta_values,covariance,solution)):raise ValueError('nonfinite closed-form input/solution')
                edited=weight.double()+alpha*delta_values@solution

            else:
                axis=p['axis'];indices=p['indices'];scale=config['scale']
                if axis not in (0,1) or weight.ndim!=2 or not indices or len(set(indices))!=len(indices) or any(type(i)is not int or i<0 or i>=weight.shape[axis] for i in indices):raise ValueError('head slice outside matrix')
                edited=weight.clone()
                if axis==0:edited[indices,:]*=scale
                else:edited[:,indices]*=scale
            return dict(self.store.save_edit(p['name'],edited),configuration=self.active)
        if operation=='edit.commit':
            if self.active is None:raise ValueError('registered configuration required')
            manifest=self.store.commit_candidate(p['id'],p['hypothesis'],p.get('method','strict_wsp'))
            manifest['edit_configuration']=self.active;json_write(self.root/p['id']/'manifest.json',manifest)
            (self.root/p['id']/'manifest.json').chmod(0o400);self.active=None;return manifest
        if operation=='edit.replay':return self.store.verify(p['candidate'])
        if operation=='edit.revert':self.store.reset(p.get('parent','base'));self.active=None;return {'reverted_to':p.get('parent','base')}
        if operation=='export.checkpoint':return self.export(p['candidate'])
        if operation=='reference.attribute':
            groups={};records=[]
            for row in p['records']:
                if len(row.get('read_ids',[]))!=2:raise ValueError('paired source read identities required')
                delta=row['reference_logprob']-row['zero_shot_logprob']
                if not math.isfinite(delta):raise ValueError('finite measured paired logprobs required')
                groups.setdefault(row['reference'],[]).append(delta);records.append(dict(row,delta=delta))
            return {'records':records,'references':{k:{'n':len(v),'mean_delta':sum(v)/len(v)} for k,v in groups.items()},'semantics':'paired observed dev logprob delta; provenance must be bound by trusted caller'}
        raise ValueError('unknown CPU atomic operation')
