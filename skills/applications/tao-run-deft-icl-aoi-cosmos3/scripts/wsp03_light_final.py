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
from wsp03_statistics import run as statistics,paired
from wsp03_worker import digest
from tht_contract import external_evaluator,grade

EXTENSIONS = {'R3-ext': 'r3_ext', 'R3-ext2': 'r3_ext2', 'R4-ext': 'r4_ext'}

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
    required = {'R3', 'R4', 'R3-ext', 'R3-ext2', 'R4-ext'}
    if set(cfg.get('freezes', {})) != {'R3', 'R4'} or set(cfg.get('editor_containers', {})) != required:
        raise ValueError('all six frozen arms and all five stopped editor tracks required')
    if not cfg.get('r4_c_freeze') or any(not cfg.get(key) for key in ('r3_ext', 'r3_ext2', 'r4_ext')):
        raise ValueError('six-arm freeze gate: missing extension or R4-C binding')
    freezes = {name: read(asset) for name, asset in cfg['freezes'].items()}
    freezes['R4-C'] = read(cfg['r4_c_freeze'])
    for arm, key in EXTENSIONS.items():
        extension = cfg[key]
        freezes[arm] = read(extension['freeze'])
        preregistration = pinned(**extension['preregistration'])
        if freezes[arm].get('preregistration_sha256') != file_hash(preregistration):
            raise ValueError(arm + ' freeze must bind its own preregistration')
    if any(freezes[arm].get('status') != 'FROZEN' for arm in ('R3', 'R4', 'R4-C')):
        raise ValueError('original three arms must be FROZEN')
    for arm in EXTENSIONS:
        value = freezes[arm]
        if value.get('status') not in ('FROZEN', 'NO_VALID_PATCH'):
            raise ValueError('all six tracks must be frozen or terminal before protected reads')
        if value['status'] == 'NO_VALID_PATCH' and (
                not isinstance(value.get('reason'), str) or not value['reason'].strip()
                or not isinstance(value.get('counts'), dict) or not value['counts']):
            raise ValueError('NO_VALID_PATCH requires immutable reason and candidate/stage counts')
    selection = cfg['r4_selection']
    if selection['freeze_sha256'] != cfg['freezes']['R4']['sha256'] or selection['profile'] != 'P09':
        raise ValueError('original R4 selection must bind its exact freeze and P09')
    if freezes['R4-C'].get('profile') != 'P01' or freezes['R4-C'].get('patch_manifest_sha256') != selection['patch_manifest_sha256']:
        raise ValueError('R4-C must freeze P01 with original frozen R3 weights')
    if freezes['R3-ext2'].get('parent') != 'base':
        raise ValueError('R3-ext2 parent must be pristine base')
    r4ext = freezes['R4-ext']
    if (r4ext.get('parent') != 'R4' or r4ext.get('parent_freeze_sha256') != cfg['freezes']['R4']['sha256']
            or r4ext.get('profile') != 'P09' or r4ext.get('reference_state_sha256') != cfg['reference_state']['sha256']):
        raise ValueError('R4-ext must retain original R4 parent, P09 and reference state')
    plan = read(cfg['extension_statistics'])
    if set(plan) != set(EXTENSIONS):
        raise ValueError('pinned statistics plan required for all extension arms')
    available = {'B', 'C', 'R3', 'R4', 'R4-C'}
    for arm in EXTENSIONS:
        available.add(arm)
        pairs = plan[arm]
        if not pairs or any(len(pair) != 2 or pair[0] != arm or pair[1] not in available-{arm} for pair in pairs):
            raise ValueError('post-hoc statistics pairs must bind available earlier arms')
    if ['R4-ext', 'R4'] not in plan['R4-ext']:
        raise ValueError('R4-ext minus original R4 is the required primary weight comparison')
    states={}
    for track,containers in cfg['editor_containers'].items():
        if not containers:raise ValueError('editor container identity required for '+track)
        for container in containers:
            state=json.loads(subprocess.check_output(['docker','inspect','--format','{{json .State}}',container],text=True))
            if state['Running'] or state['Status'] not in ('exited','dead'):
                raise RuntimeError('editor container has not stopped: '+container)
            states[container]=state
    return freezes,states


def snapshot_patch(cfg,freeze,output,*,arm='R3'):
    extension=cfg.get(EXTENSIONS.get(arm,''),{})
    source=Path(extension['export'] if extension else cfg['r3_export']);patch=source/'patch';files={}
    suffix='/'+extension.get('export_directory_name',source.name)+'/'
    for path in patch.rglob('*'):
        if not path.is_file():continue
        relative=path.relative_to(source).as_posix()
        matches=[value for name,value in freeze['reusable_exports'].items() if name.endswith(suffix+relative)]
        if len(matches)!=1 or file_hash(path)!=matches[0]:raise ValueError(arm+' frozen patch bytes differ: '+relative)
        files[path.relative_to(patch).as_posix()]=matches[0]
    expected=extension['patch_manifest_sha256'] if extension else cfg['r4_selection']['patch_manifest_sha256']
    if 'manifest.json' not in files or files['manifest.json']!=expected:
        raise ValueError(arm+' must bind its exact frozen patch manifest')
    manifest=json.loads((patch/'manifest.json').read_text())
    if not manifest.get('valid') or manifest.get('changed_elements',0)<=0:raise ValueError('no valid nonzero frozen patch')
    shutil.copytree(patch,output)
    for name,expected in files.items():
        if file_hash(output/name)!=expected:raise ValueError('patch copy changed')
    filename='patch-binding-'+arm.lower()+'.json' if extension else 'patch-binding.json'
    json_write(output.parent/filename,{'files':files,'manifest_digest':digest(manifest),'replay':'absolute replacements once onto original model before CUDA',
        'arms':[arm] if extension else ['R3','R4','R4-C'],'post_hoc':bool(extension)})
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
    if arm in EXTENSIONS:binding['post_hoc']=True
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
    original=statistics(path,output/'statistics.json',cfg['evaluator_directory'])
    plan = read(cfg['extension_statistics'])
    scores = {split: dict(payload['splits'][split]['arms']) for split in ('holdout', 'benchmark')}
    for arm, key in EXTENSIONS.items():
        extension = {'arm': arm, 'analysis': 'post-hoc extension; original main analysis unchanged',
            'freeze': cfg[key]['freeze'], 'preregistration': cfg[key]['preregistration'],
            'statistics_plan': cfg['extension_statistics'], 'editor_token_limit': None, 'splits': {}}
        for split in ('holdout', 'benchmark'):
            result = results[arm+':'+split]
            pairs = tuple(tuple(pair) for pair in plan[arm])
            if result.get('status') == 'NO_VALID_PATCH':
                unavailable = {'status': 'UNAVAILABLE', 'reason': result['reason'],
                    'pairs': {}, 'skipped_pairs': [
                        {'left': left, 'right': right, 'reason': arm+' NO_VALID_PATCH: '+result['reason']}
                        for left, right in pairs]}
                json_write(output/'arms'/split/arm/'score.json', unavailable)
                extension['splits'][split] = unavailable
                continue
            raw = Recipe.jsonl(Path(result['raw_directory'])/'raw_answers.jsonl')
            gold = Recipe.jsonl(pinned(**cfg['final_inputs'][split+'_gold']))
            score = grade(raw, gold, evaluator)
            json_write(output/'arms'/split/arm/'score.json', score)
            scores[split][arm] = score
            source = payload['splits'][split]
            needed = {name for pair in pairs for name in pair}
            arms = {name: value for name, value in scores[split].items() if name in needed}
            valid_pairs = tuple(pair for pair in pairs if all(name in arms for name in pair))
            extension['splits'][split] = paired(arms, source['identities'], evaluator,
                expected_groups=255 if split=='benchmark' else 600, pairs=valid_pairs)
            extension['splits'][split]['skipped_pairs'] = [
                {'left': left, 'right': right,
                 'reason': results.get(right+':'+split, {}).get('reason', right+' unavailable for '+split)}
                for left, right in pairs if (left, right) not in valid_pairs]
        json_write(output/('statistics-'+arm.lower()+'.json'), extension)
    return original


def write_results(output,results):
    # Keep the original six-key result contract unchanged for existing reports.
    json_write(output/'results.json', {key: value for key, value in results.items()
        if key.split(':', 1)[0] in ('R3', 'R4', 'R4-C')})
    for arm in EXTENSIONS:
        extension = {key: value for key, value in results.items() if key.startswith(arm+':')}
        if extension:
            json_write(output/('results-'+arm.lower()+'.json'), extension)


def evaluate_arms(cfg,recipe,runtime,patch,manifest,output,ext_patch=None,ext_manifest=None,results=None,extension_patches=None,terminal_extensions=None):
    selected=cfg['r4_selection']['profile'];results={} if results is None else results
    patches = dict(extension_patches or {})
    if ext_patch is not None: patches['R3-ext'] = (ext_patch, ext_manifest)
    for split in ('holdout','benchmark'):
        for arm,profile in [('R3','P00'),('R4-C','P01')]+([('R4',selected)] if selected not in ('P00','P01') else []):
            results[arm+':'+split]=evaluate(cfg,recipe,runtime,patch,manifest,output,arm,profile,split)
            write_results(output,results)
        if selected in ('P00','P01'):
            source='R3' if selected=='P00' else 'R4-C';result=dict(results[source+':'+split],arm='R4',alias_of=source+':'+split,generation_debit=0)
            destination=output/'arms'/split/'R4';destination.mkdir();json_write(destination/'result.json',result);results['R4:'+split]=result
            write_results(output,results)
        for arm, key in EXTENSIONS.items():
            if not cfg.get(key):
                continue
            terminal = (terminal_extensions or {}).get(arm, {})
            if terminal.get('status') == 'NO_VALID_PATCH':
                unavailable = {'arm': arm, 'split': split, 'status': 'NO_VALID_PATCH',
                    'available': False, 'post_hoc': True, 'reason': terminal['reason'],
                    'counts': terminal['counts'], 'generation_debit': 0,
                    'inference_skipped': True, 'no_base_substitution': True}
                destination = output/'arms'/split/arm
                destination.mkdir(parents=True, exist_ok=False)
                json_write(destination/'result.json', unavailable)
                results[arm+':'+split] = unavailable
                write_results(output, results)
                continue
            extra_patch, extra_manifest = patches[arm]
            profile = 'P09' if arm == 'R4-ext' else 'P00'
            results[arm+':'+split] = evaluate(cfg, recipe, runtime, extra_patch, extra_manifest,
                                            output, arm, profile, split)
            results[arm+':'+split]['post_hoc'] = True
            write_results(output, results)
    return results


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
    claim(output/'attempt.json',{'config_sha256':file_hash(config_file),'freezes':dict(cfg['freezes'], **{'R4-C': cfg['r4_c_freeze']}, **{arm: cfg[key]['freeze'] for arm,key in EXTENSIONS.items()}),'editor_states':states,'started_at':time.time(),'automatic_retry':False})
    results={};started=time.time();stop=threading.Event();sampler=threading.Thread(target=sample_gpu,args=(output,stop),daemon=True);sampler.start();terminal='failed'
    try:
        patch=output/'frozen-patch';manifest=snapshot_patch(cfg,freezes['R3'],patch)
        extension_patches = {}
        for arm, key in EXTENSIONS.items():
            if freezes[arm]['status'] == 'NO_VALID_PATCH':
                continue
            ext_patch = output/('frozen-patch-'+arm.lower())
            ext_manifest = snapshot_patch(cfg, freezes[arm], ext_patch, arm=arm)
            extension_patches[arm] = (ext_patch, ext_manifest)
        if not cfg.get('model_files'):raise ValueError('original frozen model file inventory required')
        for relative,expected in cfg['model_files'].items():
            name=Path(relative)
            if name.is_absolute() or '..' in name.parts or file_hash(Path(cfg['model'])/name)!=expected:raise ValueError('original model inventory differs')
        json_write(output/'model-binding.json',{'files':cfg['model_files'],'original_arms':'same original model plus original frozen R3 patch','R3-ext':'same original model plus independent extension patch' if cfg.get('r3_ext') else None})
        recipe=Recipe(cfg['recipe']['path'],cfg['recipe']['sha256'])
        runtime=Docker(output/'jobs',cfg['image'],gpus='device='+str(cfg.get('gpu',0)))
        data_runtime=Docker(output/'jobs',cfg['data_image'],gpus='device='+str(cfg.get('gpu',0)))
        prepare_inputs(cfg,recipe,freezes,runtime,data_runtime,output)
        results=evaluate_arms(cfg,recipe,runtime,patch,manifest,output,results=results,extension_patches=extension_patches,terminal_extensions=freezes)
        score_all(cfg,recipe,results,output);terminal='complete'
    finally:
        stop.set();sampler.join(timeout=6)
        json_write(output/'terminal.json',{'status':terminal,'wall_seconds':time.time()-started,'ended_at':time.time(),'completed_results':list(results),
            'completed_generation_passes':sum(r['generation_debit'] for r in results.values()),'attempted_generation_passes':len(list((output/'arms').glob('*/*/attempt.json'))),'cost_quality':'best-effort native worker tokens/query wall/peak allocation and 10-second nvidia-smi samples; no editor tokens; failed in-flight usage may be unknown'})
    return {'status':terminal,'output':str(output)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('config');args=parser.parse_args();print(json.dumps(run(args.config),indent=2))
