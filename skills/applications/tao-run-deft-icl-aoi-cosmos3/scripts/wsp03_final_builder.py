# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted lazy final preparation, pinned native embedding and exact C input reuse.

Constructing this adapter performs no source access. Its only call is behind
both successful freezes, a live joint deadline, and stopped editor kernels.
"""
from __future__ import annotations
import json,os,shutil,subprocess,time
from pathlib import Path
from wsp03_io import json_write
from wsp03_bindings import ReferenceSelector,file_hash,pinned
from wsp03_worker import digest
from wsp03_artifacts import child_json,child_bytes
from wsp03_runtime import APP,gated_job

class FinalInputBuilder:
    def __init__(self,config,*,runtime,data_runtime,lease,output):
        self.config=config;self.runtime=runtime;self.data_runtime=data_runtime;self.lease=lease;self.output=Path(output)
    def readable_json(self,path,value):
        json_write(path,value,mode=0o640);subprocess.run(['sudo','-n','chgrp',str(self.runtime.gid),str(path)],check=True)
    def stage_sources(self,cfg):
        from wsp03_artifacts import child_bytes,child_file,bounded_copy
        import hashlib
        root=Path(cfg['public_package']);target=self.output/'source-allowlist';target.mkdir(mode=0o750)
        owner=cfg.get('source_owner',os.getuid());rows=[]
        for relative,key in [('mining/samples.jsonl','mining_sha256'),('benchmark/queries.jsonl','benchmark_queries_sha256')]:
            raw=child_bytes(root,relative,owner=owner,max_bytes=32*1024**2)
            if hashlib.sha256(raw).hexdigest()!=cfg['prepare'][key]:raise ValueError('final source metadata hash changed')
            parsed=[json.loads(line) for line in raw.decode().splitlines()]
            if relative.startswith('benchmark/') and any(set(r)!={'id','image','image_sha256'} for r in parsed):raise ValueError('Benchmark query manifest must be label-free')
            destination=target/relative;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(raw);destination.chmod(0o444);rows.extend(parsed)
        copied={}
        for row in rows:
            relative=Path(row['image'])
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('source image escaped package')
            if str(relative) in copied:
                if copied[str(relative)]!=row['image_sha256']:raise ValueError('conflicting source image bytes')
                continue
            destination=target/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            with child_file(root,relative,owner=owner,max_bytes=512*1024**2) as source,destination.open('xb') as out:bounded_copy(source,out,max_bytes=512*1024**2)
            if file_hash(destination)!=row['image_sha256']:raise ValueError('final source image hash changed')
            destination.chmod(0o444);copied[str(relative)]=row['image_sha256']
        target.chmod(0o755)
        json_write(self.output/'source-allowlist.json',{'metadata':['mining/samples.jsonl','benchmark/queries.jsonl'],'images':copied,'gold_mounted':False})
        return target
    def __call__(self,r3,r4):
        from wsp03_final import after_both_freeze
        return after_both_freeze(r3.ledger,r4.ledger,lambda:self.attach(r3,r4),required_seconds=self.config['minimum_remaining_seconds'])
    def attach(self,r3,r4):
        for service in (r3,r4):
            if service.kernel.runtime.status(service.kernel.job['job_id'])['state'] in ('RUNNING','PENDING'):raise RuntimeError('untrusted kernel still running before final source access')
        from contextlib import nullcontext
        from wsp03_artifacts import child_bytes,child_file
        # Check identity metadata and file sizes only after the outer freeze gate.
        source=Path(self.config["public_package"]);owner=self.config.get("source_owner",os.getuid());sizes={}
        for relative,key in [("mining/samples.jsonl","mining_sha256"),("benchmark/queries.jsonl","benchmark_queries_sha256")]:
            import hashlib
            raw=child_bytes(source,relative,owner=owner,max_bytes=32*1024**2)
            if hashlib.sha256(raw).hexdigest()!=self.config["prepare"][key]:raise ValueError("source metadata changed before reservation")
            for row in (json.loads(line) for line in raw.decode().splitlines()):
                with child_file(source,row["image"],owner=owner,max_bytes=512*1024**2) as image:sizes[row["image"]]=os.fstat(image.fileno()).st_size
        resources=getattr(self.runtime,"resources",None)
        amount=sum(sizes.values())+2*1024**3
        with resources.reserve(amount,"final raw/preprocessed/embedding/C views",self.runtime.resource_account) if resources else nullcontext():
            return self.attach_reserved(r3,r4)
    def attach_reserved(self,r3,r4):
        for service in (r3,r4):
            if service.kernel.runtime.status(service.kernel.job['job_id'])['state'] in ('RUNNING','PENDING'):raise RuntimeError('all untrusted kernels must stop before final sources open')
        self.output.mkdir(parents=True,exist_ok=False)
        cfg=self.config;recipe=r3.request_factory;deadline=time.time()+min(r3.ledger.remaining_joint_seconds(),r4.ledger.remaining_joint_seconds())
        freeze={name:{'selected':s.ledger.get('selected'),'contract_sha256':s.ledger.get('service_contract_sha256')} for name,s in [('R3',r3),('R4',r4)]}
        source_view=self.stage_sources(cfg)
        plan=dict(cfg['prepare'],both_freeze_receipts=freeze,public_package='/public',dev_embeddings='/dev-embeddings.jsonl')
        cache=pinned(**recipe.config['embeddings']);staged_cache=self.output/'dev-embeddings.jsonl';shutil.copyfile(cache,staged_cache);staged_cache.chmod(0o444)
        identity_source=pinned(**cfg['holdout_identities']);identity_copy=self.output/'holdout-identities.jsonl'
        shutil.copyfile(identity_source,identity_copy);identity_copy.chmod(0o444)
        plan.update(holdout_identity_file='/holdout-identities.jsonl',holdout_identity_sha256=cfg['holdout_identities']['sha256'],excluded_mining_count=2)
        plan_path=self.output/'prepare-plan.json';self.readable_json(plan_path,plan)
        bank=APP.parents[2];script='/bank/'+str(APP.relative_to(bank))+'/scripts/wsp03_final_inputs.py'
        job=self.runtime.submit('wsp03-final-prepare',[script,'prepare','--plan','/plan.json','--output','/results/prepared'],gpu=False,
            mounts=[(bank,'/bank',True),(source_view,'/public',True),(plan_path,'/plan.json',True),(staged_cache,'/dev-embeddings.jsonl',True),(identity_copy,'/holdout-identities.jsonl',True)],
            extra_args=['--runtime','runc','--read-only','--tmpfs','/tmp:rw,nosuid,nodev,size=1g','--memory','8g','--cpus','8'])
        if self.runtime.wait(job['job_id'],deadline)['state']!='COMPLETE':raise RuntimeError('final preparation failed; preserve no-retry receipt')
        prepared=Path(job['results_dir'])/'prepared';bundle=child_json(job['results_dir'],'prepared/prepared.json',owner=self.runtime.uid,max_bytes=32*1024**2)
        import yaml
        embedding_spec=recipe.retrieval.embedding_spec('/prepared/embedding-input.parquet','/results/embedding/embeddings.parquet','/siglip')
        spec=self.output/'embedding.yaml';spec.write_text(yaml.safe_dump(embedding_spec));spec.chmod(0o444)
        embed_command=[script,'embed','--spec','/spec.yaml','--output','/results/embedding']
        embedded=gated_job(self.data_runtime,self.lease,'wsp03-final-embedding',embed_command,
            [(bank,'/bank',True),(prepared,'/prepared',True),(cfg['siglip_path'],'/siglip',True),(spec,'/spec.yaml',True)],deadline=deadline,python=cfg.get('data_python','python3'))
        values=[json.loads(line) for line in child_bytes(embedded['results_dir'],'embedding/embeddings.jsonl',owner=self.data_runtime.uid,max_bytes=128*1024**2).decode().splitlines()]
        if [r['id'] for r in values]!=bundle['missing_embedding_ids']:raise ValueError('final embedding ordered identity mismatch')
        embeddings=dict(recipe.selector.embeddings);embeddings.update({r['id']:r['embedding'] for r in values})
        if set(embeddings)!=set(bundle['rows']):raise ValueError('final embedding membership incomplete')
        images={qid:prepared/relative for qid,relative in bundle['images'].items()}
        for qid,path in images.items():
            if path.is_symlink() or not path.resolve().is_relative_to(prepared.resolve()) or file_hash(path)!=bundle['provenance'][qid]['view_sha256']:raise ValueError('prepared final image changed')
        for service in (r3,r4):
            target=service.request_factory;target.rows=bundle['rows'];target.images=images
            target.members.update(holdout=bundle['holdout_ids'],benchmark=bundle['benchmark_ids'])
            target.final_pools={split:ReferenceSelector(target.rows,embeddings,helpfulness=target.selector.helpfulness,atlas=target.selector.atlas,pool=bundle['development_ids'] if split=='holdout' else bundle['mining_ids']) for split in ('holdout','benchmark')}
            target.lazy_gold={'holdout':dict(cfg['holdout_gold']),'benchmark':dict(cfg['benchmark_gold'])}
            service.contract['splits'].update(holdout=bundle['holdout_ids'],benchmark=bundle['benchmark_ids']);service.contract['benchmark_reference_ids']=bundle['mining_ids']
            for key,column in [('joints','joint'),('source_ids','source_id'),('image_hashes','image_sha256'),('components','component')]:service.contract[key]={qid:row[column] for qid,row in target.rows.items()}
            service.members={k:set(v) for k,v in service.contract['splits'].items()}
        self.attach_c(r4.request_factory,cfg['historical_c'],self.output/'cached-c')
        json_write(self.output/'final-input-binding.json',{'freeze':freeze,'prepare_job':job,'embedding_job':embedded,'prepared_sha256':file_hash(prepared/'prepared.json'),'embedding_sha256':file_hash(Path(embedded['results_dir'])/'embedding/embeddings.jsonl'),'gold_read':False,'gold_access':'lazy scorer only'})
    @staticmethod
    def attach_c(recipe,asset,output):
        """Pin full original request bytes, then verify semantics and each pixel view."""
        path=pinned(**asset);rows=json.loads(path.read_text())
        if [r['id'] for r in rows]!=recipe.members['benchmark']:raise ValueError('historical C membership/order differs')
        output=Path(output);output.mkdir(parents=True,exist_ok=False);mapping={}
        for row in rows:
            refs=recipe.references(row['id'],'P01')
            if row.get('references')!=refs:raise ValueError('historical C reference selection differs from exact frozen pool selector')
            expected=recipe.render(row['id'],refs,'generation','P01');current=json.loads(json.dumps(row))
            actual_parts=[part for message in current['messages'] for part in message['content'] if part.get('type')=='image']
            expected_parts=[part for message in expected['messages'] for part in message['content'] if part.get('type')=='image']
            if len(actual_parts)!=len(expected_parts):raise ValueError('historical C image count differs')
            expected_images=recipe.request_images([expected])
            for actual,wanted in zip(actual_parts,expected_parts):
                old=Path(actual['image'])
                if old.parent!=Path('/input'):raise ValueError('historical C image path outside exact view')
                source=path.parent/old.name;prepared=expected_images[str(Path(wanted['image']).relative_to('/input'))]
                if source.is_symlink() or file_hash(source)!=file_hash(prepared):raise ValueError('historical C image bytes differ')
                destination=output/old.name
                if not destination.exists():shutil.copyfile(source,destination)
                mapping[actual['image']]=destination;actual['image']=wanted['image']
            if current['messages']!=expected['messages']:raise ValueError('historical C rendered semantics differ')
        # The original pinned request rows are retained exactly for target input.
        recipe.cached_c={r['id']:r for r in rows};recipe.cached_c_images=mapping
        json_write(output/'binding.json',{'request_sha256':asset['sha256'],'count':len(rows),'images':{key:file_hash(value) for key,value in mapping.items()},'exact_request_reuse':True})
