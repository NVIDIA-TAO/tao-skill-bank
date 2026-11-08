# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owner-only aggregate oracle. Never mount this controller or its state in editor."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,re,shutil,subprocess,time,traceback
from suite_runtime import Docker
from tht_contract import external_evaluator,grade
from wsp03_runtime import validate_worker_output
from wsp03_worker import digest

ROOT=Path(os.environ['WSP05_RUN_ROOT']).expanduser().resolve()
BANK=Path(os.environ.get('WSP05_BANK',str(ROOT.parent/'bank'))).expanduser().resolve()
SCRIPT='/bank/skills/applications/tao-run-deft-icl-aoi-cosmos3/scripts'
MAX_QUERIES=20
FIELDS={'id','parent','profile','patch_relative','hypothesis','expected','plan_relative','plan_sha256','source_relative','gpu_released'}
METRIC_FIELDS={'n','correct','accuracy','macro_f1','balanced_accuracy','mcc','invalid','per_class','confusion_labels','confusion_matrix'}
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(8388608),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as stream:
        json.dump(v,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
def publish(p,v):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def safe_work(relative,work):
    if not isinstance(relative,str) or Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('work-relative path required')
    raw=work/relative
    if any(p.is_symlink() for p in [raw,*raw.parents] if p!=work.parent):
        raise ValueError('symlinks not allowed')
    result=raw.resolve(strict=True)
    if not result.is_relative_to(work.resolve()):raise ValueError('work path escape')
    return result
def validate_request(req,work):
    if not isinstance(req,dict) or set(req)!=FIELDS:raise ValueError('query schema')
    if not re.fullmatch(r'Q(?:0[1-9]|1[0-9]|20)',req['id']):raise ValueError('query identifier')
    if req['parent'] not in ['B','D','G','F'] or req['profile'] not in ['single','icl']:
        raise ValueError('parent/profile')
    if req['gpu_released'] is not True:raise ValueError('GPU release declaration')
    for field in ['hypothesis','expected']:
        if not isinstance(req[field],str) or not 10<=len(req[field])<=10000:raise ValueError('prewritten hypothesis required')
    plan=safe_work(req['plan_relative'],work)
    if not plan.is_file() or sha(plan)!=req['plan_sha256']:raise ValueError('plan digest')
    value=read(plan)
    if not isinstance(value,dict) or not value:raise ValueError('finite plan required')
    source=safe_work(req['source_relative'],work)
    if not source.is_dir():raise ValueError('source directory')
    patch=None
    if req['patch_relative'] is not None:
        patch=safe_work(req['patch_relative'],work)
        m=read(patch/'manifest.json')
        if not m.get('valid') or m.get('changed_elements',0)<=0 or not m.get('tensors'):
            raise ValueError('valid nonempty patch required')
        names=set()
        for item in m['tensors']:
            if item['name'] in names:raise ValueError('duplicate tensor')
            names.add(item['name']);name=item['file']
            if Path(name).name!=name or (patch/name).is_symlink() or not (patch/name).is_file():
                raise ValueError('unsafe patch file')
    return plan,source,patch
def snapshot(req,work,target,models):
    plan,source,patch=validate_request(req,work)
    target.mkdir(exist_ok=False);shutil.copyfile(plan,target/'plan.json')
    if sha(target/'plan.json')!=req['plan_sha256']:raise ValueError('plan changed while copying')
    files={}
    (target/'source').mkdir()
    for p in sorted(source.rglob('*')):
        if p.is_symlink():raise ValueError('source symlink')
        if not p.is_file() or '__pycache__' in p.parts:continue
        if p.suffix not in ['.py','.md','.txt','.json','.yaml','.yml','.sh']:continue
        rel=p.relative_to(source);dest=target/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dest);files[str(rel)]=sha(dest)
    patch_files=None
    if patch:
        (target/'patch').mkdir();m=read(patch/'manifest.json');patch_files={}
        for name in ['manifest.json',*[x['file'] for x in m['tensors']]]:
            shutil.copyfile(patch/name,target/'patch'/name);patch_files[name]=sha(target/'patch'/name)
        if read(target/'patch/manifest.json')!=m:raise ValueError('patch changed while copying')
    config={'parent':req['parent'],'profile':req['profile'],'model_files':models[req['parent']]['files'],
            'patch_files':patch_files,'max_new_tokens':128,'backend':'HF1 BF16 SDPA greedy seed42',
            'worker_sha256':sha(BANK/SCRIPT.removeprefix('/bank/')/'wsp03_worker.py'),
            'replay_sha256':sha(ROOT/'setup/wsp05_replay.py')}
    write(target/'configuration.json',config)
    write(target/'snapshot.json',{'request':req,'configuration_sha256':sha(target/'configuration.json'),
          'source_files':files,'created_at':time.time()})
    return config
def aggregate(score):
    metrics=score['metrics']
    if set(metrics)!=METRIC_FIELDS:raise ValueError('unapproved aggregate schema')
    return {k:metrics[k] for k in METRIC_FIELDS}
def evaluate(config,snapshot_dir,split,output):
    # Caller must enforce development/freeze gate before selecting split.
    if split not in ['A','B']:raise ValueError('fixed population required')
    output.mkdir(exist_ok=False)
    if config['worker_sha256']!=sha(BANK/SCRIPT.removeprefix('/bank/')/'wsp03_worker.py') or config['replay_sha256']!=sha(ROOT/'setup/wsp05_replay.py'):
        raise ValueError('frozen worker or replay source changed')
    private=ROOT/'private';upstream=read(private/'upstream-bindings.json')
    source=private/split/config['profile']
    split_receipt=read(ROOT/'setup/split-public.json')
    if sha(private/'partition.json')!=split_receipt['partition_sha256']:
        raise ValueError('partition binding changed')
    if sha(source/'requests.json')!=split_receipt['splits'][split][config['profile']+'_requests_sha256']:
        raise ValueError('fixed population request binding changed')
    requests=read(source/'requests.json')
    models=read(ROOT/'setup/model-bindings.json');base=models[config['parent']]
    if base['files']!=config['model_files']:raise ValueError('model inventory differs')
    for name,h in base['files'].items():
        if sha(Path(base['path'])/name)!=h:raise ValueError('parent model changed')
    runtime=Docker(output/'jobs',read(ROOT/'config.json')['image'],gpus='device=0')
    mounts=[(BANK,'/bank',True),(base['path'],'/model',True),
            (source/'input','/input',True),(source/'requests.json','/requests.json',True),
            (ROOT/'setup/wsp05_replay.py','/wsp05_replay.py',True)]
    command=['/wsp05_replay.py','--model','/model','--requests','/requests.json','--output','/results/worker',
             '--mode','generation','--split','benchmark','--query-seconds','600','--max-new-tokens','128']
    manifest=None
    if config['patch_files']:
        patch=snapshot_dir/'patch'
        for name,h in config['patch_files'].items():
            if sha(patch/name)!=h:raise ValueError('snapshot patch changed')
        manifest=read(patch/'manifest.json');mounts.append((patch,'/candidate',True));command+=['--candidate','/candidate']
    tick=time.time();write(output/'ATTEMPT.json',{'started_at':tick,'split':split,'configuration_sha256':sha(snapshot_dir/'configuration.json'),'no_retry':True})
    job=runtime.submit('wsp05-aggregate-'+split.lower(),command,mounts=mounts,gpu=True,
                       env=[('PYTHONPATH',SCRIPT)])
    write(output/'job.json',job)
    state=runtime.wait(job['job_id'],time.time()+21600)
    if state['state']!='COMPLETE':raise RuntimeError('private inference failed; no automatic retry')
    worker=Path(job['results_dir'])/'worker'
    raw=validate_worker_output(worker,requests,mode='generation',split='benchmark',
        candidate_manifest_sha256=digest(manifest) if manifest else None,max_new_tokens=128,query_seconds=600)
    membership={r['id'] for r in requests};goldasset=upstream['gold']
    if sha(goldasset['path'])!=goldasset['sha256']:raise ValueError('gold digest changed')
    gold=[r for line in Path(goldasset['path']).read_text().splitlines() if line.strip() for r in [json.loads(line)] if r['id'] in membership]
    score=grade(raw,gold,external_evaluator(upstream['evaluator_directory']))
    write(output/'private-score.json',score)
    cost={'wall_seconds':time.time()-tick,'query_wall_seconds':sum(r['wall_seconds'] for r in raw),
          'input_tokens':sum(r['inference_input_tokens'] for r in raw),'output_tokens':sum(r['inference_output_tokens'] for r in raw),
          'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in raw),'peak_reserved_bytes':max(r['peak_reserved_bytes'] for r in raw),
          'generation_calls':len(raw),'kernel_active_seconds':None,'quality':'best-effort; GPU occupied wall and query wall, not profiled kernel time'}
    result={'status':'COMPLETE','metrics':aggregate(score),'cost':cost}
    write(output/'aggregate.json',result);return result
def ledger_entries(private):
    p=private/'admissions.jsonl'
    return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
def admit(private,entry):
    entries=ledger_entries(private)
    if len(entries)>=MAX_QUERIES:raise ValueError('aggregate query budget exhausted')
    if any(x['id']==entry['id'] for x in entries):raise ValueError('duplicate admission')
    entry=dict(entry,number=len(entries)+1,admitted_at=time.time())
    with (private/'admissions.jsonl').open('a') as f:
        f.write(json.dumps(entry)+'\n');f.flush();os.fsync(f.fileno())
    with (private/'decisions.tsv').open('a') as f:
        fields=[str(entry['admitted_at']),entry['id'],str(entry['number']),entry['hypothesis'],entry['expected'],entry['configuration_sha256']]
        f.write('\t'.join(x.replace('\t',' ').replace('\n',' ') for x in fields)+'\n');f.flush();os.fsync(f.fileno())
    return entry
def serve():
    private=ROOT/'private';work=ROOT/'work';(private/'queries').mkdir(exist_ok=True)
    lock=(private/'oracle.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not (ROOT/'setup/oracle-fixtures.json').exists() or read(ROOT/'setup/oracle-fixtures.json')['status']!='PASS':
        raise ValueError('synthetic interface checks required')
    # Editor must have been canary-verified; actual mount receipt is owner-created.
    if read(ROOT/'setup/editor-isolation.json')['status']!='PASS':raise ValueError('isolation evidence required')
    ready={'schema':'wsp05-oracle-ready/1','split':read(ROOT/'setup/split-public.json'),
           'max_queries':MAX_QUERIES,'contract_sha256':sha(ROOT/'context/ORACLE_CONTRACT.md'),
           'code_sha256':sha(__file__),'created_at':time.time()}
    publish(work/'OWNER_ORACLE_READY.json',ready)
    models=read(ROOT/'setup/model-bindings.json')
    while True:
        if (work/'FREEZE.json').exists() or (work/'BLOCKED.json').exists():break
        for file in sorted((work/'outbox').glob('Q*.json')):
            q=file.stem;dest=work/'inbox'/(q+'.json');record=private/'queries'/q
            if record.exists() and (record/'response.json').exists():
                response=read(record/'response.json')
                if sha(file)!=read(record/'received.json')['sha256']:
                    publish(dest,{'status':'REJECTED','reason':'query ID reused with different bytes'})
                elif not dest.exists():publish(dest,response)
                continue
            if record.exists():
                # Interrupted admitted attempt is never silently re-run.
                publish(dest,{'status':'BLOCKED','reason':'incomplete private attempt; preserve state, owner reconciliation required'})
                continue
            if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():continue
            record.mkdir();write(record/'received.json',{'sha256':sha(file),'at':time.time()})
            try:
                req=read(file)
                if req['id']!=q:raise ValueError('request filename mismatch')
                config=snapshot(req,work,record/'snapshot',models)
                ch=sha(record/'snapshot/configuration.json')
                duplicate=None
                for entry in ledger_entries(private):
                    if entry['configuration_sha256']==ch:duplicate=entry;break
                if duplicate:
                    old=read(private/'queries'/duplicate['id']/'response.json')
                    response={**old,'query_id':q,'reused_query_id':duplicate['id']}
                else:
                    entry=admit(private,{'id':q,'hypothesis':req['hypothesis'],'expected':req['expected'],
                          'request_sha256':sha(file),'plan_sha256':req['plan_sha256'],'configuration_sha256':ch})
                    result=evaluate(config,record/'snapshot','A',record/'evaluation')
                    response={**result,'query_id':q,'query_number':entry['number'],'configuration_sha256':ch,
                              'queries_used':len(ledger_entries(private))}
            except BaseException as e:
                write(record/'private-error.json',{'type':type(e).__name__,'message':str(e),'at':time.time()})
                write(record/'error-locations.json',{'type':type(e).__name__,
                      'frames':[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(e.__traceback__)]})
                used=ledger_entries(private);consumed=any(x['id']==q for x in used)
                response={'status':'FAILED' if consumed else 'REJECTED','error_type':type(e).__name__,
                          'query_id':q,'admission_consumed':consumed,'queries_used':len(used),
                          'reason':'private operation failed; no row diagnostics or automatic retry'}
            write(record/'response.json',response);publish(dest,response)
        time.sleep(2)
    write(private/'oracle-terminal.json',{'at':time.time(),'queries_used':len(ledger_entries(private))})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['serve']);p.parse_args();serve()
