# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Once-only final inference and paired statistics after every WSP04 editor stops.

This host operation is never mounted in an editor. The config is assembled only
after all three freezes. It binds original main-suite request/image views and
historical B/C predictions; it never runs new baseline inference.
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, threading, time
from pathlib import Path
from suite_runtime import Docker
from wsp03_bindings import file_hash, pinned
from wsp03_worker import digest
from wsp03_runtime import validate_worker_output
from wsp03_light_final import code_mounts, native, CONTAINER_SCRIPTS, sample_gpu
from wsp03_io import json_write
from tht_contract import external_evaluator, grade
from wsp03_statistics import paired

ARMS=('W1','W2','W3')
PAIRS=(('W1','B'),('W1','C'),('W2','W1'),('W2','C'),('W3','C'),('W3','W1'))

def read(asset):return json.loads(pinned(**asset).read_text())

def freeze_gate(cfg, state_reader=None):
    if set(cfg['arms'])!=set(ARMS):raise ValueError('all three WSP04 records required')
    reg=read(cfg['registration'])
    if reg.get('schema')!='wsp04-registration/1':raise ValueError('wrong experiment registration')
    states={};freezes={}
    for arm in ARMS:
        row=cfg['arms'][arm];f=read(row['freeze']);freezes[arm]=f
        if f.get('track')!=arm or f.get('preregistration_sha256')!=cfg['registration']['sha256']:
            raise ValueError('arm/preregistration binding differs')
        if f.get('status') not in ('FROZEN','NO_VALID_PATCH'):raise ValueError('unfinished track')
        if f['status']=='NO_VALID_PATCH' and (not f.get('reason') or not row.get('terminal_evidence')):
            raise ValueError('terminal reason and pinned evidence required')
        if f['status']=='NO_VALID_PATCH':
            for asset in row['terminal_evidence']:pinned(**asset)
        if not row.get('editors'):raise ValueError('native editor identity missing')
        states[arm]=[]
        for cid in row['editors']:
            state=state_reader(cid) if state_reader else json.loads(subprocess.check_output(
                ['docker','inspect','--format','{{json .State}}',cid],text=True))
            if state.get('Running') or state.get('Status') not in ('exited','dead'):
                raise ValueError('every editor must be stopped before any test read')
            states[arm].append({'container':cid,'state':state})
        if f['status']=='FROZEN':
            manifest=read(row['patch_manifest'])
            if row['patch_manifest']['sha256'] not in json.dumps(f):
                raise ValueError('freeze does not bind this exact patch manifest')
            if not manifest.get('valid') or manifest.get('changed_elements',0)<=0:
                raise ValueError('frozen edited patch must be finite, nonempty and valid')
            for item in manifest['tensors']:
                p=Path(row['patch_manifest']['path']).parent/item['file']
                if not p.is_file() or p.is_symlink():raise ValueError('replacement tensor missing/unsafe')
    return freezes,states

def requests_and_images(asset, destination, expected_refs):
    rows=read(asset['requests'])
    if len(rows)!=271 or len({r['id'] for r in rows})!=271:
        raise ValueError('exact271 independent final requests required')
    for row in rows:
        if len(row['references'])!=expected_refs:raise ValueError('single/nearest1 request mismatch')
    source=Path(asset['images']).resolve(strict=True);destination.mkdir()
    image_hashes={}
    for row in rows:
        for message in row['messages']:
            for part in message['content']:
                if part.get('type')!='image':continue
                path=Path(part['image'])
                if path.parent!=Path('/input'):raise ValueError('image outside exact final view')
                src=source/path.name
                if src.is_symlink() or not src.resolve(strict=True).is_relative_to(source):
                    raise ValueError('unsafe historical image')
                expected=asset['image_sha256'].get(path.name)
                if not expected or file_hash(src)!=expected:raise ValueError('unpinned/changed image')
                if not (destination/path.name).exists():shutil.copyfile(src,destination/path.name)
                image_hashes[path.name]=expected
    return rows,image_hashes

def run(config):
    cfg=json.loads(Path(config).read_text());freezes,states=freeze_gate(cfg)
    output=Path(cfg['output']);output.mkdir(parents=True,exist_ok=False)
    json_write(output/'attempt.json',{'schema':'wsp04-final/1','config_sha256':file_hash(config),
        'started_at':time.time(),'registration':cfg['registration'],'editor_states':states,
        'freeze_bindings':{a:cfg['arms'][a]['freeze'] for a in ARMS},'automatic_retry':False})
    runtime=Docker(output/'jobs',cfg['image'],gpus='device=0')
    stop=threading.Event();sampler=threading.Thread(target=sample_gpu,args=(output,stop),daemon=True)
    sampler.start();results={}
    try:
        for name,expected in cfg['model_files'].items():
            if Path(name).name!=name or file_hash(Path(cfg['model'])/name)!=expected:
                raise ValueError('original model inventory changed')
        for arm in ARMS:
            d=output/arm;d.mkdir()
            if freezes[arm]['status']=='NO_VALID_PATCH':
                results[arm]={'status':'NO_VALID_PATCH','reason':freezes[arm]['reason'],'inference_passes':0}
                json_write(d/'result.json',results[arm]);continue
            row=cfg['arms'][arm];src=Path(row['patch_manifest']['path']).parent
            patch=d/'patch';shutil.copytree(src,patch)
            if file_hash(patch/'manifest.json')!=row['patch_manifest']['sha256']:raise ValueError('patch copy changed')
            manifest=json.loads((patch/'manifest.json').read_text())
            reqs,hashes=requests_and_images(cfg['icl_requests' if arm=='W3' else 'single_requests'],d/'input',1 if arm=='W3' else 0)
            requests=d/'requests.json';json_write(requests,reqs)
            json_write(d/'binding.json',{'requests_sha256':digest(reqs),'patch_sha256':row['patch_manifest']['sha256'],
                'image_sha256':hashes,'automatic_retry':False,'profile':'C-nearest1' if arm=='W3' else 'B-single'})
            job=native(runtime,output,arm+'-benchmark',[CONTAINER_SCRIPTS+'/wsp03_worker.py',
                '--model','/model','--candidate','/candidate','--requests','/requests.json',
                '--output','/results/worker','--mode','generation','--split','benchmark'],
                [*code_mounts(),(cfg['model'],'/model',True),(patch,'/candidate',True),
                 (d/'input','/input',True),(requests,'/requests.json',True)],gpu=True,timeout=7200)
            worker=Path(job['results_dir'])/'worker'
            raw=validate_worker_output(worker,reqs,mode='generation',split='benchmark',candidate_manifest_sha256=digest(manifest))
            if len(raw)!=271:raise ValueError('incomplete independent final pass')
            results[arm]={'status':'COMPLETE','inference_passes':1,'n':271,'job':job,'worker':str(worker),
                'cost':{'completed_input_tokens':sum(x['inference_input_tokens'] for x in raw),
                        'completed_output_tokens':sum(x['inference_output_tokens'] for x in raw),
                        'query_wall_seconds':sum(x['wall_seconds'] for x in raw),
                        'peak_allocated_bytes':max(x['peak_allocated_bytes'] for x in raw),
                        'peak_reserved_bytes':max(x['peak_reserved_bytes'] for x in raw),'kernel_active_seconds':None}}
            json_write(d/'result.json',results[arm])
        # No gold or historical answers are read until ALL new inference stops.
        json_write(output/'all-inference-complete.json',{'completed_at':time.time(),'results':results})
        gold=[json.loads(s) for s in pinned(**cfg['gold']).read_text().splitlines() if s.strip()]
        evaluator=external_evaluator(cfg['evaluator_directory']);arms={}
        gold_by_id={r['id']:r['labels'][0] for r in gold}
        if len(gold)!=271 or len(gold_by_id)!=271:raise ValueError('exact271 gold identities required')
        for arm in ('B','C'):
            arms[arm]=read(cfg['historical_predictions'][arm])
            historical={r['id']:r['truth'] for r in arms[arm]['predictions']}
            if historical!=gold_by_id or len(arms[arm]['predictions'])!=271:
                raise ValueError('historical baseline and current gold differ')
        for arm in ARMS:
            if results[arm]['status']!='COMPLETE':continue
            raw=[json.loads(s) for s in (Path(results[arm]['worker'])/'raw_answers.jsonl').read_text().splitlines()]
            arms[arm]=grade(raw,gold,evaluator);json_write(output/arm/'score.json',arms[arm])
        identities=read(cfg['identities'])
        stats=paired(arms,identities,evaluator,expected_groups=255,pairs=PAIRS)
        stats['unavailable']={a:results[a] for a in ARMS if results[a]['status']!='COMPLETE'}
        stats['skipped_pairs']=[{'left':a,'right':b,'reason':'one or both arms unavailable'} for a,b in PAIRS if a not in arms or b not in arms]
        p=stats['pairs'];lower=lambda key:p[key]['mcc']['lower'] if key in p else None
        gates={'W3-C_superiority':lower('W3__minus__C') is not None and lower('W3__minus__C')>0,
               'W1-C_noninferiority':lower('W1__minus__C') is not None and lower('W1__minus__C')>=-.03,
               'W2-C_noninferiority':lower('W2__minus__C') is not None and lower('W2__minus__C')>=-.03}
        stats['skill_gate']={'passed':any(gates.values()),'criteria':gates,'metric':'mcc','confidence':.95}
        json_write(output/'statistics.json',stats)
        json_write(output/'COMPLETE.json',{'completed_at':time.time(),'statistics_sha256':file_hash(output/'statistics.json'),
            'results':results,'skill_gate':stats['skill_gate'],'no_editor_reopened':True})
    finally:
        stop.set();sampler.join(timeout=15)
    return {'status':'COMPLETE','output':str(output),'skill_gate':stats['skill_gate']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');args=p.parse_args();print(json.dumps(run(args.config),indent=2))
