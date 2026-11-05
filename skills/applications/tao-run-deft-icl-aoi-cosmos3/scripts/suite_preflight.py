# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only task preflight: effective settings, provenance and M3 evidence gates."""
from __future__ import annotations
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from tht_contract import external_evaluator,load_jsonl,sha256,json_hash,validate_queries
from suite_runtime import BANK

EXPECTED_COUNTS={'mining':3642,'proxy':74,'benchmark':271}
PROMPT_SHA='c2b3b9aeb4c31b6b699dad32ea6f4c00081c7aadf2a8d65384f0bba1423326f6'

def validate(task):
    task=json.loads(json.dumps(task))
    required=['public_package','private_package','model_path','framework_image','data_services_image',
              'direct_profile','siglip_path','siglip_revision','results_dir','budgets','astra','m3_evidence','environment_manifest']
    missing=[x for x in required if x not in task]
    if missing: raise ValueError('missing task inputs: '+', '.join(missing))
    if task.get('mode')!='tht_single': raise ValueError('suite requires explicit tht_single mode')
    if task.get('skip_sdg') is not True: raise ValueError('this six-arm THT protocol requires explicit SDG skip')
    if task.get('platform')!='docker': raise ValueError('this suite implementation supports Docker on the selected Brev host')
    public=Path(task['public_package']).resolve(strict=True)
    private=Path(task['private_package']).resolve(strict=True)
    if public==private or private.is_relative_to(public) or public.is_relative_to(private):
        raise ValueError('private/public roles must be separate trees')
    bundle=json.loads((public/'bundle.json').read_text())
    if bundle['role_counts']!=EXPECTED_COUNTS or bundle['excluded_unique']!=310 or bundle['metadata_repaired']!=0 or bundle['known_source_joint_overlaps']!=2:
        raise ValueError('frozen package membership/policy mismatch')
    if bundle['split_policy']!='preserve_original_Maxwell_splits': raise ValueError('resplit not permitted')
    external_evaluator(public/'evaluator')
    for role in ('proxy','benchmark'):
        validate_queries(load_jsonl(public/role/'queries.jsonl'),count=EXPECTED_COUNTS[role])
    mining=load_jsonl(public/'mining/samples.jsonl')
    if len(mining)!=3642 or len({r['id'] for r in mining})!=3642:
        raise ValueError('Mining membership mismatch')
    profile=json.loads(Path(task['direct_profile']).read_text())
    if hashlib.sha256(profile['prompt'].encode()).hexdigest()!=PROMPT_SHA or profile['prompt_sha256']!=PROMPT_SHA:
        raise ValueError('exact Maxwell source prompt mismatch')
    model=Path(task['model_path']).resolve(strict=True)
    config=json.loads((model/'config.json').read_text())
    if config['model_type']!='qwen3_vl' or not (model/'tao_conversion_provenance.json').is_file():
        raise ValueError('model must be prepared by the shared model skill')
    prov=json.loads((model/'tao_conversion_provenance.json').read_text())
    for file in prov['prepared']['files']:
        if (model/file['name']).stat().st_size!=file['size'] or sha256(model/file['name'])!=file['sha256']:
            raise ValueError('prepared checkpoint fingerprint mismatch')
    # Deep fingerprints are verified at model preparation; immutable manifest is
    # sealed here and full artifact verification is required again at freeze.
    for key in ('framework_image','data_services_image'):
        if '@sha256:' not in task[key]: raise ValueError('image must be immutable: '+key)
        subprocess.run(['docker','image','inspect',task[key]],check=True,stdout=subprocess.DEVNULL)
    if not shutil.which('docker'): raise ValueError('Docker unavailable')
    evidence=json.loads(Path(task['m3_evidence']).read_text())
    gates=['external_scorer_parity','native_training','checkpoint_resume','native_export','image_1_2_k',
           'loss_mask','runtime_isolation','astra_invocation','embedding_and_mining','paired_compatibility',
           'live_token_cap','generic_tools','tokenizer_parity','new_round_resume']
    failures=[g for g in gates if evidence.get(g,{}).get('status')!='passed']
    if failures: raise ValueError('M3 gates incomplete: '+', '.join(failures))
    b=task['budgets']
    for key in ('suite_wall_seconds','icl_wall_seconds','icl_agent_tokens','query_wall_seconds','query_agent_tokens','stage_wall_seconds','trial_admission_seconds'):
        if not isinstance(b.get(key),(int,float)) or b[key]<=0: raise ValueError('positive measured budget required: '+key)
    for key in ('icl_agent_tokens','query_agent_tokens'):
        if type(b[key]) is not int or b[key]<2: raise ValueError('native token cap must be an integer >=2: '+key)
    for key,value in {'f_min_epochs':5,'f_max_epochs':20,'f_patience':3,'max_new_tokens':4096,'do_sample':False}.items():
        if key in task and task[key]!=value: raise ValueError('task conflicts with frozen THT protocol: '+key)
    if task.get('max_icl_trials',12)!=12 or task.get('max_deft_rounds',5)!=5:
        raise ValueError('THT protocol trial/round caps changed')
    task.update(max_icl_trials=12,max_deft_rounds=5,f_min_epochs=5,f_max_epochs=20,f_patience=3,
                benchmark_count=271,max_new_tokens=4096,do_sample=False,
                training_attention='cosmos',inference_attention='sdpa',label_mode='names',
                prompt_sha256=PROMPT_SHA,base_fingerprint=prov['prepared']['fingerprint'],
                anomalygen={'status':'skipped','reason':'user-requested','synthetic_images':0},
                overlap_disclosure='Two original Mining/Benchmark physical joints are retained by explicit source policy; training is not joint-disjoint. ICL excludes same-joint references.',
                original_counts={'train':3952,'val':74,'test':271},effective_counts=EXPECTED_COUNTS,
                original_exclusions={'multi_label':81,'missing_original_object_name':234,'overlap':5,'unique':310,'recoverable_not_restored':1})
    capacity=task.get('icl_reference_capacity',4)
    if not 1<=capacity<=evidence['image_1_2_k']['max_images']-1:
        raise ValueError('reference capacity exceeds real model smoke evidence')
    task['icl_reference_capacity']=capacity
    task['deft_neighbors']=4
    task['statistics']={'method':'paired_joint_percentile','grouping':'object_name+normalized_scan_id',
                        'seed':42,'replicates':2000,'confidence':0.95}
    task['budget_policy']={'native_rollout':'response-boundary enforced; equal input/output weights; one in-flight response and already-issued tools may complete beyond cap',
                           'no_final_at_cap':'incomplete, non-retryable; no fabricated answer',
                           'completed_invalid':'immutable completed response, retained in denominator',
                           'retries':'infrastructure only, at most two attempts; original wall/token envelopes persist'}
    task['routing_policy']='tht-error-neighbors-v1: descriptive Proxy confusion analysis, all wrong/INVALID targets, four native cosine neighbors, accumulated unique Mining; zero errors routes all Proxy'
    task['retention']={'F':'every completed epoch DCP and HF export retained (maximum20)',
                       'D':'one terminal DCP/export per round (maximum5); native continuation, dataloader reset between rounds',
                       'failed_attempts':'retained; checkpoints resume within round; no source/selected artifact deletion'}
    required_bytes=evidence['storage']['worst_case_additional_bytes']
    existing=Path(task['results_dir'])
    if (existing/'suite.sqlite').exists():
        required_bytes=max(0,required_bytes-sum(p.stat().st_size for p in existing.rglob('*') if p.is_file()))
    free=shutil.disk_usage(Path(task['results_dir']).parent).free
    if free<required_bytes: raise ValueError('insufficient disk for frozen checkpoint/export/recovery retention')
    task['resource_estimates']={**evidence['storage'],'available_bytes':free,'measurements':evidence.get('measurements')}
    from suite_data import validate_identity_grammar
    for role in ('proxy','benchmark'): validate_identity_grammar(mining,load_jsonl(public/role/'queries.jsonl'))
    task['identity_projection']='validated source filename grammar; matches every Mining object_name/scan_id; no private source-row reads'
    task['code']={'commit':subprocess.check_output(['git','-C',str(BANK),'rev-parse','HEAD'],text=True).strip(),
                  'branch':subprocess.check_output(['git','-C',str(BANK),'branch','--show-current'],text=True).strip()}
    if subprocess.check_output(['git','-C',str(BANK),'status','--porcelain'],text=True).strip():
        raise ValueError('formal preflight requires a committed clean checkout')
    if task.get('pass_fail_mapping'):
        mapping=task['pass_fail_mapping'];source=mapping['provenance']
        if sha256(source['path'])!=source['sha256']: raise ValueError('historical Fail/Pass source hash mismatch')
    task['environment']=json.loads(Path(task['environment_manifest']).read_text())
    from astra_isolation import PROMPT_TEMPLATE,generic_mounts
    task['astra']['prompt_template']=PROMPT_TEMPLATE
    task['astra']['prompt_template_sha256']=hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()
    astra=task['astra']
    if '@sha256:' not in astra['image']: raise ValueError('A requires an immutable runtime image')
    subprocess.run(['docker','image','inspect',astra['image']],check=True,stdout=subprocess.DEVNULL)
    version=subprocess.check_output([astra['codex_binary'],'--version'],text=True).strip()
    if version!='codex-cli 0.166.0': raise ValueError('native token cap must be revalidated on this Codex version')
    for source,_,_ in generic_mounts(astra['codex_binary'],astra['python_binary'],astra['python_stdlib'],astra['system_libraries'],astra.get('site_packages',[])):
        if not Path(source).exists(): raise ValueError('missing isolated runtime dependency: '+str(source))
    if not Path(astra['auth_file']).is_file(): raise ValueError('approved private Codex auth mount unavailable')
    task['astra']['codex_version']=version
    task['astra']['codex_binary_sha256']=sha256(astra['codex_binary'])
    # Defaults shared by D/F. Origins are visible and frozen before work starts.
    defaults={'rank':16,'alpha':32,'dropout':0.05,'lr':1e-4,'weight_decay':0.01,
              'targets':'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
              'seed':42,'effective_batch':1,'precision':'bfloat16'}
    supplied=task.get('training',{})
    task['training']={**defaults,**supplied}
    if task['training']!=evidence['training_settings']:
        raise ValueError('training settings require matching M3 native smoke evidence')
    if task['training']['effective_batch']!=1:
        raise ValueError('initial validated image adapter requires effective batch 1')
    task['origins']={'training.'+k:('task' if k in supplied else 'packaged_default') for k in defaults}
    task['origins'].update({'budgets':'M3 measurements and task preflight','baseline_prompt':'pinned Maxwell DirectMode',
                            'stop_selection_rules':'canonical THT protocol','platform':'authorized Brev-local Docker',
                            'sdg':'explicit user request'})
    task['provenance']={'bundle_sha256':sha256(public/'bundle.json'),
                        'public_checksums_sha256':sha256(public/'SHA256SUMS'),
                        'model_conversion_sha256':sha256(model/'tao_conversion_provenance.json'),
                        'direct_profile_sha256':sha256(task['direct_profile']),
                        'm3_evidence_sha256':sha256(task['m3_evidence'])}
    task['provenance']['siglip_files']={p.name:sha256(p) for p in sorted(Path(task['siglip_path']).iterdir()) if p.is_file()}
    return {'ready':True,'effective':task,'preflight_sha256':json_hash(task)}
