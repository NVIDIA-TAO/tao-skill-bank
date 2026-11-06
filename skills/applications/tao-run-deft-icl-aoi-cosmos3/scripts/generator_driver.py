# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent same-D + generator loop, reusing native workers and durable jobs."""
import json
import time
from pathlib import Path
from collections import Counter

from generator_contract import (immutable_json, verify_file, validate_admission,
                                combined_training_view, file_binding, _SHARED)
from generator_policy import allocate, select_clean
from generator_store import GeneratorStore
from generator_preflight import verify_resume
from suite_driver import Suite, APP
from suite_data import RetrievalToolset
from suite_runtime import BANK, Docker
from suite_selection import select_round
from tht_contract import external_evaluator, load_jsonl, json_hash, sha256


class FrozenRetrieval(RetrievalToolset):
    def embed(self,role,rows):
        import pyarrow.parquet as pq
        if role not in ('mining','proxy'): raise ValueError('generator routing accepts Mining/Proxy only')
        binding=self.task['retrieval_cache'][role]['vectors']
        table=pq.read_table(verify_file(binding)).to_pylist()
        if len(table)!=len(rows) or {r['id'] for r in table}!={r['id'] for r in rows}: raise ValueError('frozen vector role coverage mismatch')
        # Existing retrieval.references hashes these native cache paths. Preserve
        # bytes in a local read-only-by-contract copy; do not rerun embeddings.
        import shutil
        target=self.root/('embed-'+role)/'embeddings.parquet';target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists(): shutil.copyfile(binding['path'],target)
        if sha256(target)!=binding['sha256']: raise ValueError('local frozen vector cache changed')
        return table


class GeneratorSuite(Suite):
    cost_arms='G'

    def __init__(self,task,approval,*,resume_origin='operator'):
        verify_resume(task)
        self.task=task;self.root=Path(task['results_dir']).resolve()
        if self.root.is_relative_to(BANK): raise ValueError('private run outputs cannot enter code repository')
        self.store=GeneratorStore(self.root)
        resuming=self.store.get('started_at') is not None
        self.store.initialize(task,approval)
        for artifact in self.store.db.execute('SELECT path,sha256 FROM artifacts'):
            if sha256(artifact['path'])!=artifact['sha256']:
                raise ValueError('completed run artifact changed before resume')
        self.store.seal('started_at',self.store.get('started_at',time.time()))
        if resuming:
            self.store.event('resume',{'origin':resume_origin,'policy_changed':False})
        self.deadline=self.store.get('started_at')+task['budgets']['suite_wall_seconds']
        self.public=Path(task['public_package']);self.private=Path(task['private_package'])
        self.prompt=json.loads(Path(task['direct_profile']).read_text())['prompt']
        self.evaluator=external_evaluator(self.public/'evaluator')
        self.mining=load_jsonl(self.public/'mining/samples.jsonl');self.proxy=load_jsonl(self.public/'proxy/queries.jsonl')
        self.runtime=Docker(self.root/'jobs',task['framework_image'])
        self.retrieval=FrozenRetrieval(task,self.run_job);self.budget_scope=None
        self.mapping={k:task['class_mapping'][k] for k in task['class_order']}

    def admissions(self, through):
        values=[]
        for number in range(1,through+1):
            binding=self.store.get(f'G/admission/{number}')
            if binding is None: raise ValueError('missing cumulative generation admission')
            value=json.loads(verify_file(binding).read_text())
            if value['round']!=number: raise ValueError('generation history reordered')
            validate_admission(value,self.task['generator']['prepared_package']);values.append(value)
        return values

    def training_population(self,selected_ids,view,round_index):
        composition=combined_training_view(self.public,selected_ids,self.prompt,self.admissions(round_index),
                                          self.task['generator']['prepared_package'],view)
        self.store.seal(f'G/composition/{round_index}',composition)
        for name in ('composition.json','provenance.json'):
            self.store.artifact(f'G/train/{round_index}/{name}',view/name)
        return composition['train_records']

    def infer(self,arm,phase,candidate,model,**kwargs):
        if arm!='G' or phase not in ('proxy','benchmark'): raise ValueError('isolated generator arm required')
        if phase=='benchmark':
            if candidate!='final': raise ValueError('only the frozen final candidate may evaluate Benchmark')
            self.store.verify_freeze();self.admissions(5)
        # Read only the currently admissible role gold, on the trusted host.
        relative=f'evaluation/{phase}-gold.jsonl'
        checks={line.split(maxsplit=1)[1].lstrip('*'):line.split()[0] for line in (self.private/'SHA256SUMS').read_text().splitlines() if line.strip()}
        if sha256(self.private/relative)!=checks.get(relative): raise ValueError('trusted role gold differs from sealed package')
        return super().infer(arm,phase,candidate,model,**kwargs)

    def generation_request(self,number,gaps,history):
        allocation=allocate(self.mapping,gaps['predictions'],self.task['mining_class_counts'],history,number)
        source=self.retrieval.embed('mining',self.mining);vectors={r['id']:r for r in source}
        target=self.retrieval.embed('proxy',self.proxy)
        originals=load_jsonl(Path(self.task['generator']['prepared_package'])/'audit/admitted_sources.jsonl')
        clean=[]
        for row in originals:
            if row['kind']!='clean': continue
            identity=row['matched_identity']['source_id'];asset=next(a for a in row['assets'] if a['asset_role']=='clean_image')
            cad=next(a for a in row['assets'] if a['asset_role']=='cad_mask')
            clean.append({'id':identity,'stem':row['source_stem'],'prepared_path':asset['prepared_path'],
                          'image_sha256':asset['prepared_sha256'],'cad_mask':{'prepared_path':cad['prepared_path'],'sha256':cad['prepared_sha256']},'embedding':vectors[identity]['embedding']})
        work=select_clean(self.mapping,allocation,gaps['predictions'],target,clean)
        coverage=[]
        for native,label in self.mapping.items():
            errors=sorted(r['id'] for r in gaps['predictions'] if r['truth']==label and r['prediction']!=label)
            anchors=errors or sorted(r['id'] for r in gaps['predictions'] if r['truth']==label)
            assigned=[r for r in work if r['anomaly_type']==native]
            anchored=[r for r in assigned if r['anchor_id'] is not None]
            distinct={r['anchor_id'] for r in anchored}
            coverage.append({'native_type':native,'label':label,'anchor_list':'error' if errors else 'all-class' if anchors else 'native',
                             'allocated_cases':len(assigned),'anchored_cases':len(anchored),'distinct_used_anchors':len(distinct),
                             'eligible_anchors':len(anchors),'anchored_fraction':len(anchored)/len(assigned) if assigned else None,
                             'distinct_anchor_fraction':len(distinct)/len(anchors) if anchors else None})
        allocation['anchor_coverage']=coverage
        request={'round':number,'allocation':allocation,'work_items':work,'generator':self.task['generator'],
                 'incoming_proxy_sha256':json_hash(gaps),'history_sha256':json_hash(history),
                 'preflight_sha256':json_hash(self.task),'prepared_inventory':self.task['prepared_inventory'],
                 'embedding_bindings':{role:self.task['retrieval_cache'][role]['vectors'] for role in ('mining','proxy')},
                 'cosine_implementation':self.task['cosine_implementation']}
        path=self.root/'generation-requests'/f'round-{number:03d}.json';immutable_json(path,request)
        self.store.artifact(f'G/generate/{number}/request',path)
        self.store.seal(f'G/allocation/{number}',allocation)
        return path,request

    def generate(self,number,request_path):
        g=self.task['generator'];shared=next(p for p in _SHARED if p.is_dir())
        # No role packages or classifier gold are mounted. The external prepared
        # package contains only approved generation sources and audit evidence.
        sources=[Path(g[k]).resolve() for k in ('prepared_package','checkpoint','recipe','base_checkpoint','checkpoint_root','base_source')]
        mounts=[(APP,str(APP),True),(shared,str(shared),True),(request_path,str(request_path),True)]
        mounts += [(p,str(p),True) for p in sources if not any(p!=q and p.is_relative_to(q) for q in sources)]
        mounts += [(g['checkpoint_root'],'/workspace/cdf-anomalygen/checkpoints',True),
                   (g['base_source'],'/base-source',True)]
        mounts += [(m['source'],m['target'],True) for m in g.get('runtime_mounts',[])]
        env=list(g.get('environment',{}).items())+[('PYTHONPATH',str(APP)+':/workspace/cdf-anomalygen')]
        result=self.run_job(f'G/generate/{number}',[str(APP/'generator_worker.py'),'--request',str(request_path)],
                            image=g['image'],entrypoint='/opt/venv/bin/python',mounts=mounts,env=env,
                            required_output='admission.json',host_result_mount=True,
                            stage_wall_seconds=self.task['budgets']['generator_wall_seconds'])
        path=Path(result['job']['results_dir'])/'admission.json'
        admission=json.loads(path.read_text());validate_admission(admission,g['prepared_package'])
        if admission['round']!=number: raise ValueError('native attempt round differs')
        self.store.seal(f'G/admission/{number}',file_binding(path))
        return admission

    def run(self):
        if self.store.get('completed'):
            completed=self.store.get('completed');verify_file(completed)
            return json.loads(Path(completed['path']).read_text())
        if self.store.get('freeze'): return self.benchmark()
        gaps=self.infer('G','proxy','base',self.task['model_path'])
        selected=[];rounds=[];history=[];previous=None
        from tht_rcca import route
        for number in range(1,6):
            saved=self.store.get(f'G/round/{number}')
            if saved:
                self.admissions(number)
                history.append(saved['allocation_history']);rounds.append(saved);previous=saved
                selected=saved['selected_ids'];gaps=json.loads(Path(saved['metrics_path']).read_text());continue
            routing=route(gaps,self.proxy,phase='proxy',neighbors=4)
            routing['sdg']={'status':'enabled','policy':self.task['generator']['policy']['version']}
            path=self.root/'routing'/f'round-{number:03d}.json';immutable_json(path,routing)
            self.store.artifact(f'G/routing/{number}',path)
            policy={'toolset':'siglip-retrieval-v1','version':1,'rationale':'same-D four-neighbor Proxy error routing',
                    'max_calls':1,'parameters':{'reference_count':4,'ordering':'nearest_first','instruction':''}}
            refs=self.retrieval.references('proxy',self.proxy,policy,4)
            newly={r['id'] for q in routing['mining_targets'] for r in refs[q]}
            selected=sorted(set(selected)|newly)
            request_path,request=self.generation_request(number,gaps,history)
            admission=self.generate(number,request_path)
            trained=self.train('G',selected,7,round_index=number,previous=previous)
            next_gaps=self.infer('G','proxy',f'round-{number:03d}',trained['checkpoint'])
            counts=Counter(r['label'] for r in admission['records'])
            allocation_history={**request['allocation'],'generated_counts':{k:counts[c] for k,c in self.mapping.items()}}
            composition=self.store.get(f'G/composition/{number}')
            value=dict(trained,phase='proxy',status='complete',n=next_gaps['metrics']['n'],expected_n=len(self.proxy),
                       mcc=next_gaps['metrics']['mcc'],round=number,selected_ids=selected,
                       new_real_ids=len(newly-set(previous['selected_ids'])) if previous else len(newly),
                       metrics_path=str(self.root/'metrics/G/proxy'/f'round-{number:03d}.json'),
                       routing_artifact=str(path),allocation_history=allocation_history,
                       composition=composition,generation_admission=self.store.get(f'G/admission/{number}'),
                       real_exposures=7*composition['real_count'],synthetic_exposures=7*composition['synthetic_count'])
            self.store.seal(f'G/round/{number}',value)
            previous=value;rounds.append(value);history.append(allocation_history);gaps=next_gaps
        selected_round=select_round(rounds);self.save_selection('G',selected_round)
        folder=Path(selected_round['checkpoint'])
        files=[{'path':str(p),'sha256':sha256(p),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
               for p in sorted(folder.iterdir()) if p.is_file()]
        manifest=self.root/'frozen-model.json';immutable_json(manifest,files)
        binding=self.store.artifact('G/frozen-model',manifest)
        settings={'G':{'base_fingerprint':self.task['base_fingerprint'],'checkpoint':str(folder),'model_manifest':binding,
                       'prompt_sha256':self.task['prompt_sha256'],'preprocessing':'RGB/ImageOps.pad/256x256/NEAREST/PNG',
                       'generation':{'max_new_tokens':4096,'do_sample':False},'seed':42,
                       'policy_sha256':json_hash(self.task['generator']['policy']),
                       'preflight_sha256':json_hash(self.task),'statistics':self.task['statistics']}}
        self.store.freeze(settings)
        return self.benchmark()

    def benchmark(self):
        settings=self.store.verify_freeze()['settings']['G']
        self.infer('G','benchmark','final',settings['checkpoint'])
        expected={q['id'] for q in load_jsonl(self.public/'benchmark/queries.jsonl')}
        if not self.store.completed_benchmark(expected): raise ValueError('final Benchmark coverage incomplete')
        from generator_analysis import export
        report=export(self.store,self.task)
        self.store.seal('completed',file_binding(report['artifact']))
        return report
