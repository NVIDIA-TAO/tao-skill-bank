# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent final evaluation after light-mode editors freeze and stop.

No editor, strategy search, admission service, or automatic retry is involved.
The owner supplies a normalized, pinned selection config. Run once on the host;
only explicit query/image/model/patch inputs reach each native worker.
"""
from __future__ import annotations
import argparse,json,shutil,subprocess,threading,time
from pathlib import Path
from suite_runtime import Docker,BANK
from wsp03_bindings import Recipe,ReferenceSelector,PROFILES,file_hash,pinned
from wsp03_final_builder import FinalInputBuilder
from wsp03_io import json_write
from wsp03_runtime import validate_worker_output
from wsp03_statistics import run as statistics
from wsp03_worker import digest
from tht_contract import external_evaluator,grade

APP=Path(__file__).resolve().parent
CONTAINER_SCRIPTS='/bank/'+APP.relative_to(BANK).as_posix()

def code_mounts():
    relative='skills/data/tao-in-context-learning/scripts/retrieval.py'
    return [(APP,CONTAINER_SCRIPTS,True),(BANK/relative,'/bank/'+relative,True)]


def read(asset):
    return json.loads(pinned(**asset).read_text())


def claim(path,value):
    """Persist an attempt before dispatch. Existing attempts are never retried."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n')


def freeze_gate(cfg):
    if set(cfg['freezes'])!={'R3','R4'} or set(cfg['editor_containers'])!={'R3','R4'}:
        raise ValueError('both frozen tracks and their editor container lists required')
    freezes={name:read(asset) for name,asset in cfg['freezes'].items()}
    if any(value.get('status')!='FROZEN' for value in freezes.values()):
        raise ValueError('both tracks must have a successful FROZEN artifact; no base substitution')
    selection=cfg['r4_selection']
    if selection['freeze_sha256']!=cfg['freezes']['R4']['sha256'] or selection['profile'] not in PROFILES:
        raise ValueError('normalized R4 selection must bind its exact freeze and registered profile')
    states={}
    for track,containers in cfg['editor_containers'].items():
        if not containers:raise ValueError('editor container identity required for '+track)
        for container in containers:
            state=json.loads(subprocess.check_output(['docker','inspect','--format','{{json .State}}',container],text=True))
            if state['Running'] or state['Status'] not in ('exited','dead'):
                raise RuntimeError('editor container has not stopped: '+container)
            states[container]=state
    return freezes,states


def snapshot_patch(cfg,freeze,output):
    source=Path(cfg['r3_export']);patch=source/'patch';files={}
    for path in patch.rglob('*'):
        if not path.is_file():continue
        relative=path.relative_to(source).as_posix()
        matches=[value for name,value in freeze['reusable_exports'].items() if name.endswith('/frozen-r3/'+relative)]
        if len(matches)!=1 or file_hash(path)!=matches[0]:raise ValueError('R3 frozen patch bytes differ: '+relative)
        files[path.relative_to(patch).as_posix()]=matches[0]
    if 'manifest.json' not in files or files['manifest.json']!=cfg['r4_selection']['patch_manifest_sha256']:
        raise ValueError('R4 and R3 must bind the same frozen patch manifest')
    manifest=json.loads((patch/'manifest.json').read_text())
    if not manifest.get('valid') or manifest.get('changed_elements',0)<=0:raise ValueError('no valid nonzero frozen patch')
    shutil.copytree(patch,output)
    for name,expected in files.items():
        if file_hash(output/name)!=expected:raise ValueError('patch copy changed')
    json_write(output.parent/'patch-binding.json',{'files':files,'manifest_digest':digest(manifest),'replay':'absolute replacements once onto original model before CUDA','all_arms_same_weights':True})
    return manifest


def native(runtime,output,name,command,mounts,*,gpu,python=None,timeout=86400):
    attempt=output/'attempts'/(name+'.json')
    claim(attempt,{'stage':name,'status':'attempted','started_at':time.time(),'gpu':gpu,'retry':False})
    opened_id=[]
    def opened(jid):
        opened_id.append(jid);json_write(attempt,{'stage':name,'status':'native_opened','job_id':jid,'gpu':gpu,'retry':False})
    kwargs={'entrypoint':python} if python else {}
    job=None;started=time.time()
    try:
        job=runtime.submit('wsp03-light-final-'+name,command,mounts=mounts,gpu=gpu,on_open=opened,**kwargs)
        json_write(attempt,{'stage':name,'status':'running','job':job,'started_at':started,'gpu':gpu,'retry':False})
        status=runtime.wait(job['job_id'],time.time()+timeout)
        json_write(attempt,{'stage':name,'status':status['state'],'job':job,'started_at':started,'wall_seconds':time.time()-started,'gpu':gpu,'retry':False})
        if status['state']!='COMPLETE':raise RuntimeError('final native stage failed; preserve attempt, no retry: '+name)
        return job
    except BaseException as error:
        owned=job['job_id'] if job else (opened_id[0] if opened_id else None)
        if owned and runtime.status(owned)['state'] in ('RUNNING','PENDING'):runtime.cancel(owned)
        with (output/'failures.jsonl').open('a') as stream:stream.write(json.dumps({'stage':name,'error':repr(error),'job':job,'time':time.time()})+'\n')
        raise


def prepare_inputs(cfg,recipe,freezes,runtime,data_runtime,output):
    final=cfg['final_inputs'];directory=output/'preparation';directory.mkdir()
    # Reuse only the source allowlist builder; no Service/Ledger/UID wrapper.
    builder=FinalInputBuilder(final,runtime=runtime,data_runtime=data_runtime,lease=None,output=directory)
    source=builder.stage_sources(final)
    holdout=pinned(**final['holdout_identities']);cache=pinned(**recipe.config['embeddings'])
    plan=dict(final['prepare'],both_freeze_receipts=cfg['freezes'],public_package='/public',
              dev_embeddings='/dev-embeddings.jsonl',holdout_identity_file='/holdout-identities.jsonl',
              holdout_identity_sha256=final['holdout_identities']['sha256'],excluded_mining_count=2)
    plan_file=directory/'plan.json';json_write(plan_file,plan)
    job=native(runtime,output,'prepare',[CONTAINER_SCRIPTS+'/wsp03_final_inputs.py','prepare','--plan','/plan.json','--output','/results/prepared'],
        [*code_mounts(),(source,'/public',True),(plan_file,'/plan.json',True),(holdout,'/holdout-identities.jsonl',True),(cache,'/dev-embeddings.jsonl',True)],gpu=False)
    prepared=Path(job['results_dir'])/'prepared';bundle=json.loads((prepared/'prepared.json').read_text())
    import yaml
    spec=recipe.retrieval.embedding_spec('/prepared/embedding-input.parquet','/results/embedding/embeddings.parquet','/siglip')
    spec_file=directory/'embedding.yaml';spec_file.write_text(yaml.safe_dump(spec))
    embedded=native(data_runtime,output,'embedding',[CONTAINER_SCRIPTS+'/wsp03_final_inputs.py','embed','--spec','/spec.yaml','--output','/results/embedding'],
        [*code_mounts(),(prepared,'/prepared',True),(final['siglip_path'],'/siglip',True),(spec_file,'/spec.yaml',True)],gpu=True,python=cfg.get('data_python','python3'))
    values=Recipe.jsonl(Path(embedded['results_dir'])/'embedding/embeddings.jsonl')
    if [r['id'] for r in values]!=bundle['missing_embedding_ids']:raise ValueError('final embedding ordered membership differs')
    embeddings=dict(recipe.selector.embeddings);embeddings.update({r['id']:r['embedding'] for r in values})
    if set(embeddings)!=set(bundle['rows']):raise ValueError('incomplete final embedding coverage')
    reference=read(cfg['reference_state'])
    if reference['r3_freeze_sha256']!=cfg['freezes']['R3']['sha256']:raise ValueError('reference state must bind frozen R3 dev artifacts')
    if not (set(reference['helpfulness'])|set(reference['atlas']))<=set(bundle['development_ids']):raise ValueError('helpfulness and atlas must be dev-only')
    recipe.rows=bundle['rows'];recipe.images={qid:prepared/path for qid,path in bundle['images'].items()}
    for qid,path in recipe.images.items():
        if not path.resolve().is_relative_to(prepared.resolve()) or file_hash(path)!=bundle['provenance'][qid]['view_sha256']:raise ValueError('prepared image binding differs')
    recipe.members.update(holdout=bundle['holdout_ids'],benchmark=bundle['benchmark_ids'])
    recipe.final_pools={split:ReferenceSelector(recipe.rows,embeddings,helpfulness=reference['helpfulness'],atlas=reference['atlas'],
        pool=bundle['development_ids'] if split=='holdout' else bundle['mining_ids']) for split in ('holdout','benchmark')}
    # Exact pinned historical C requests and PNG bytes, first opened after freeze.
    FinalInputBuilder.attach_c(recipe,final['historical_c'],directory/'cached-c')
    json_write(directory/'binding.json',{'prepare_job':job,'embedding_job':embedded,'prepared_sha256':file_hash(prepared/'prepared.json'),
        'reference_state':cfg['reference_state'],'holdout_pool_count':len(bundle['development_ids']),'benchmark_pool_count':len(bundle['mining_ids']),'gold_read':False})
    return bundle


def evaluate(cfg,recipe,runtime,patch,manifest,output,arm,profile,split):
    directory=output/'arms'/split/arm;directory.mkdir(parents=True,exist_ok=False)
    rows=[recipe.render(qid,recipe.references(qid,profile),'generation',profile) for qid in recipe.members[split]]
    inputs=directory/'input';inputs.mkdir()
    images=recipe.request_images(rows)
    for relative,source in images.items():
        target=inputs/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    request=directory/'requests.json';json_write(request,rows)
    binding={'arm':arm,'profile':profile,'split':split,'requests_sha256':digest(rows),'patch_manifest_sha256':digest(manifest),
             'image_sha256':{key:file_hash(value) for key,value in images.items()},'attempted_at':time.time()}
    claim(directory/'attempt.json',binding)
    job=native(runtime,output,arm+'-'+split,[CONTAINER_SCRIPTS+'/wsp03_worker.py','--model','/model','--candidate','/candidate','--requests','/requests.json','--output','/results/worker','--mode','generation','--split',split],
        [*code_mounts(),(cfg['model'],'/model',True),(patch,'/candidate',True),(inputs,'/input',True),(request,'/requests.json',True)],gpu=True,timeout=cfg.get('pass_timeout_seconds',7200))
    worker=Path(job['results_dir'])/'worker';raw=validate_worker_output(worker,rows,mode='generation',split=split,candidate_manifest_sha256=digest(manifest))
    result=dict(binding,complete=True,job=job,raw_directory=str(worker),generation_debit=1,
        cost={'completed_input_tokens':sum(r['inference_input_tokens'] for r in raw),'completed_output_tokens':sum(r['inference_output_tokens'] for r in raw),
              'query_wall_seconds':sum(r['wall_seconds'] for r in raw),'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in raw),
              'peak_reserved_bytes':max(r['peak_reserved_bytes'] for r in raw),'kernel_active_seconds':None})
    json_write(directory/'result.json',result);return result


def score_all(cfg,recipe,results,output):
    evaluator=external_evaluator(cfg['evaluator_directory'])
    payload={'all_final_passes_complete':True,'both_freezes':cfg['freezes'],'splits':{}}
    for split in ('holdout','benchmark'):
        asset=cfg['final_inputs'][split+'_gold'];gold=Recipe.jsonl(pinned(**asset));arms={};aliases={}
        for arm in ('R3','R4','R4-C'):
            result=results[arm+':'+split]
            raw=Recipe.jsonl(Path(result['raw_directory'])/'raw_answers.jsonl')
            arms[arm]=grade(raw,gold,evaluator)
            json_write(output/'arms'/split/arm/'score.json',arms[arm])
            if result.get('alias_of'):aliases[arm]=result['alias_of']
        if split=='benchmark':
            for arm in ('B','C'):arms[arm]=read(cfg['historical_predictions'][arm])
        identities={qid:{key:recipe.rows[qid][key] for key in ('object_name','scan_id')} for qid in recipe.members[split]}
        payload['splits'][split]={'arms':arms,'identities':identities,'aliases':aliases}
    path=output/'statistics-input.json';json_write(path,payload)
    return statistics(path,output/'statistics.json',cfg['evaluator_directory'])


def sample_gpu(output,stop):
    while not stop.is_set():
        try:
            proc=subprocess.run(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
            value={'time':time.time(),'csv':proc.stdout.strip(),'exit_code':proc.returncode}
        except (OSError,subprocess.TimeoutExpired) as error:value={'time':time.time(),'error':type(error).__name__}
        with (output/'gpu-samples.jsonl').open('a') as stream:stream.write(json.dumps(value)+'\n')
        stop.wait(10)


def run(config_file):
    cfg=json.loads(Path(config_file).read_text());freezes,states=freeze_gate(cfg)
    output=Path(cfg['output']);output.mkdir(parents=True,exist_ok=False)
    claim(output/'attempt.json',{'config_sha256':file_hash(config_file),'freezes':cfg['freezes'],'editor_states':states,'started_at':time.time(),'automatic_retry':False})
    results={};started=time.time();stop=threading.Event();sampler=threading.Thread(target=sample_gpu,args=(output,stop),daemon=True);sampler.start();terminal='failed'
    try:
        patch=output/'frozen-patch';manifest=snapshot_patch(cfg,freezes['R3'],patch)
        if not cfg.get('model_files'):raise ValueError('original frozen model file inventory required')
        for relative,expected in cfg['model_files'].items():
            name=Path(relative)
            if name.is_absolute() or '..' in name.parts or file_hash(Path(cfg['model'])/name)!=expected:raise ValueError('original model inventory differs')
        json_write(output/'model-binding.json',{'files':cfg['model_files'],'all_arms':'same original model plus frozen absolute replacement patch'})
        recipe=Recipe(cfg['recipe']['path'],cfg['recipe']['sha256'])
        runtime=Docker(output/'jobs',cfg['image'],gpus='device='+str(cfg.get('gpu',0)))
        data_runtime=Docker(output/'jobs',cfg['data_image'],gpus='device='+str(cfg.get('gpu',0)))
        prepare_inputs(cfg,recipe,freezes,runtime,data_runtime,output)
        selected=cfg['r4_selection']['profile']
        for split in ('holdout','benchmark'):
            for arm,profile in [('R3','P00'),('R4-C','P01')]+([('R4',selected)] if selected not in ('P00','P01') else []):
                results[arm+':'+split]=evaluate(cfg,recipe,runtime,patch,manifest,output,arm,profile,split)
                json_write(output/'results.json',results)
            if selected in ('P00','P01'):
                source='R3' if selected=='P00' else 'R4-C';result=dict(results[source+':'+split],arm='R4',alias_of=source+':'+split,generation_debit=0)
                destination=output/'arms'/split/'R4';destination.mkdir();json_write(destination/'result.json',result);results['R4:'+split]=result
                json_write(output/'results.json',results)
        score_all(cfg,recipe,results,output);terminal='complete'
    finally:
        stop.set();sampler.join(timeout=6)
        json_write(output/'terminal.json',{'status':terminal,'wall_seconds':time.time()-started,'ended_at':time.time(),'completed_results':list(results),
            'completed_generation_passes':sum(r['generation_debit'] for r in results.values()),'attempted_generation_passes':len(list((output/'arms').glob('*/*/attempt.json'))),'cost_quality':'best-effort native worker tokens/query wall/peak allocation and 10-second nvidia-smi samples; no editor tokens; failed in-flight usage may be unknown'})
    return {'status':terminal,'output':str(output)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('config');args=parser.parse_args();print(json.dumps(run(args.config),indent=2))
