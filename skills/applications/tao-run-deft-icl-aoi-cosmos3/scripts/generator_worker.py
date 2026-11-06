#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native AnomalyGen composition in one durable, host-identically mounted attempt."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from generator_contract import (verify_prepared, file_binding, verify_file, bind_amp,
                                admit_generation, immutable_json, validate_cases, _SHARED)
from tht_contract import load_jsonl, sha256
from anomalygen_contract import filter_testcases


def execute(command, root, repo):
    started=time.time()
    result=subprocess.run([str(v) for v in command],cwd=repo,check=False)
    with (root/'commands.jsonl').open('a') as stream:
        stream.write(json.dumps({'argv':[str(v) for v in command],'started_at_unix':started,
                                'ended_at_unix':time.time(),'exit_code':result.returncode})+'\n')
    if result.returncode: raise subprocess.CalledProcessError(result.returncode,command)


def prepare(package, work, counts, output, seed, *, execute_native, repo):
    """Native per-class pairing fixes clean coverage; one placement call preserves native CAD."""
    package=Path(package); output=Path(output)
    prepared=verify_prepared(package)
    if set(counts)!=set(prepared['mapping']) or sum(counts.values())!=len(work): raise ValueError('work-item count vector mismatch')
    if output.exists(): raise FileExistsError('preserve every previous native preparation')
    output.mkdir(parents=True)
    view=output/'dataset-view';shutil.copytree(package/'dataset',view)
    # Every selected clean is reserved before fallback pairing so fallback cannot
    # consume a later class's predeclared nearest clean.
    reserved={r['clean']['id'] for r in work if r['clean'] is not None}
    if len(reserved)!=sum(r['clean'] is not None for r in work): raise ValueError('clean sources repeat within the round')
    execute_native([sys.executable,'-m','anomalygen.scripts.auto_mask_placement.roi_allocate',
                    '--num_sdg',str(sum(counts.values())),'--mode','inference',
                    '--defect_desc',view/'defect_spec.jsonl','--dataset_dir',view,
                    '--output_allocation',output/'allocation.json','--per_defect_counts',json.dumps(counts)],output,repo)
    if json.loads((output/'allocation.json').read_text())!=counts:
        raise ValueError('native allocated per-class vector differs from frozen request')
    used=set(); merged={}; native_artifacts=[file_binding(output/'allocation.json')]; resolved=[]
    source_rows=load_jsonl(package/'audit/admitted_sources.jsonl')
    cleans={r['matched_identity']['source_id']:r for r in source_rows if r['kind']=='clean'}
    for item in work:
        if item['clean'] is None: continue
        chosen=item['clean'];source=cleans.get(chosen['id'])
        if source is None: raise ValueError('chosen clean is not an approved source')
        asset=next(a for a in source['assets'] if a['asset_role']=='clean_image')
        cad=next(a for a in source['assets'] if a['asset_role']=='cad_mask')
        expected={'id':chosen['id'],'stem':source['source_stem'],'prepared_path':asset['prepared_path'],
                  'image_sha256':asset['prepared_sha256'],'cad_mask':{'prepared_path':cad['prepared_path'],'sha256':cad['prepared_sha256']}}
        if chosen!=expected: raise ValueError('chosen clean/CAD binding differs from prepared source')
    for ordinal,kind in enumerate(prepared['mapping']):
        count=counts[kind]
        if not count: continue
        requested=[r for r in work if r['anomaly_type']==kind]
        if len(requested)!=count: raise ValueError('class work-item count differs')
        texture=kind.split('+')[0]
        class_root=output/'pairing'/f'class-{ordinal:02d}';class_view=class_root/'dataset'
        class_view.mkdir(parents=True)
        selected={r['clean']['id'] for r in requested if r['clean'] is not None}
        fallback=any(r['clean'] is None for r in requested)
        if fallback and selected: raise ValueError('mixed anchor/fallback mode within one class')
        candidates=[r for i,r in cleans.items() if i in selected or (fallback and i not in reserved|used)]
        candidates=[r for r in candidates if any(a['prepared_path'].startswith('dataset/'+texture+'/') for a in r['assets'])]
        if len(candidates)<count: raise ValueError('insufficient unique native fallback clean sources')
        for row in candidates:
            for asset in row['assets']:
                relative=Path(asset['prepared_path']).relative_to('dataset');dest=class_view/relative
                dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(package/asset['prepared_path'],dest)
        for donor in (r for r in source_rows if r['kind']=='donor_reference' and r['matched_identity']['labels']==[prepared['mapping'][kind]]):
            for asset in donor['assets']:
                relative=Path(asset['prepared_path']).relative_to('dataset');dest=class_view/relative
                dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(package/asset['prepared_path'],dest)
        for name in ('defect_spec.jsonl','semantic_segmentation_labels.json'):
            shutil.copyfile(view/name,class_view/name)
        allocation={k:(count if k==kind else 0) for k in counts}
        immutable_json(class_root/'allocation.json',allocation)
        native_artifacts.append(file_binding(class_root/'allocation.json'))
        pairs=class_root/'amp_samples.json'
        execute_native([sys.executable,'-m','anomalygen.scripts.auto_mask_placement.roi_pair',
                        '--dataset_dir',class_view,'--defect_desc',class_view/'defect_spec.jsonl',
                        '--allocation',class_root/'allocation.json','--output_pair_path',pairs,
                        '--seed',str(seed),'--per_clean_pairing','on'],output,repo)
        original=json.loads(pairs.read_text());native_artifacts.append(file_binding(pairs))
        by_clean={}
        for pair in original:
            canonical=dict(pair)
            for key in ('clean_image','submask','cad_mask','cad_mask_label'):
                source=Path(pair[key]).resolve(strict=True)
                relative=source.relative_to(class_view.resolve());target=view/relative
                if sha256(source)!=sha256(target): raise ValueError('native pair copied source differs')
                native_artifacts.append(file_binding(source));canonical[key]=str(target)
            clean_relative='dataset/'+str(Path(canonical['clean_image']).relative_to(view))
            item=prepared['assets'][clean_relative]['source'];identity=item['matched_identity']['source_id']
            if identity in by_clean or identity in used or pair['n_seeds']!=1 or pair['defect_type']!=kind:
                raise ValueError('native pairing violated exact unique work-item coverage')
            by_clean[identity]=canonical
        if len(by_clean)!=count or (not fallback and set(by_clean)!=selected): raise ValueError('native pair set differs from selected clean set')
        fallback_ids=iter(sorted(by_clean,key=lambda i:(cleans[i]['source_stem'],i)))
        for item in requested:
            item=json.loads(json.dumps(item));identity=next(fallback_ids) if fallback else item['clean']['id']
            if fallback:
                row=cleans[identity];asset=next(a for a in row['assets'] if a['asset_role']=='clean_image')
                cad=next(a for a in row['assets'] if a['asset_role']=='cad_mask')
                item['clean']={'id':identity,'stem':row['source_stem'],'prepared_path':asset['prepared_path'],
                               'image_sha256':asset['prepared_sha256'],'cad_mask':{'prepared_path':cad['prepared_path'],'sha256':cad['prepared_sha256']}}
            if identity in used: raise ValueError('clean source reused across classes')
            used.add(identity);merged[item['slot']]=by_clean[identity];resolved.append(item)
    if sorted(merged)!=list(range(len(work))): raise ValueError('incomplete native work items')
    inference=output/'inference';inference.mkdir()
    immutable_json(inference/'allocation.json',counts)
    immutable_json(inference/'amp_samples.json',[merged[i] for i in sorted(merged)])
    (inference/'amp_samples.json.n_seeds').write_text('1')
    immutable_json(output/'pairing.json',{'requested':work,'resolved':sorted(resolved,key=lambda r:r['slot']),
                                        'native_artifacts':native_artifacts})
    execute_native([sys.executable,'-m','anomalygen.scripts.auto_mask_placement.roi_place',
                    '--input_pair_path',inference/'amp_samples.json','--defect_desc',view/'defect_spec.jsonl',
                    '--output_dir',inference,'--seed',str(seed),'--n_seeds','1'],output,repo)
    filtered=output/'native-filtered-testcase.jsonl'
    filter_testcases(inference/'testcase.jsonl',filtered,package/'audit/exclude_stems.txt',base_dir=repo)
    native_rows=load_jsonl(filtered);validate_cases(native_rows,counts)
    slots={(str(Path(merged[i]['clean_image']).resolve()),merged[i]['defect_type']):i for i in merged}
    finalized={};permutation=[]
    for position,row in enumerate(native_rows):
        key=(str(Path(row['image_filename']).resolve()),row['anomaly_type'])
        if key not in slots: raise ValueError('native testcase does not match a selected clean/class')
        slot=slots[key]
        if slot in finalized: raise ValueError('multiple native testcases for one work item')
        finalized[slot]={**row,'index':slot};permutation.append({'native_position':position,'slot':slot})
    if sorted(finalized)!=list(range(len(work))): raise ValueError('native AMP omitted a work item')
    final=output/'inference-testcase.jsonl'
    with final.open('x') as stream:
        stream.write(''.join(json.dumps(finalized[i],sort_keys=True)+'\n' for i in sorted(finalized)))
    immutable_json(output/'testcase-order.json',{'original':file_binding(inference/'testcase.jsonl'),
                                              'filtered':file_binding(filtered),'final':file_binding(final),
                                              'permutation':permutation})
    return output/'inference-testcase.jsonl'


def run(request_file, output, *, prepare_only=False):
    request_file=Path(request_file);request=json.loads(request_file.read_text());output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'admission.json').exists() or (output/'preparation').exists(): raise FileExistsError('native attempt cannot overwrite a previous run')
    g=request['generator'];package=Path(g['prepared_package']);repo=Path('/workspace/cdf-anomalygen')
    native_tmp=output/'native-tmp';native_tmp.mkdir()
    runtime_env={'TMPDIR':str(native_tmp),'CKPT_DIR':g['checkpoint_root'],
                 'HF_HUB_CACHE':str(Path(g['checkpoint_root'])/'hf'),'HOME':str(output/'home'),
                 'UV_CACHE_DIR':str(output/'cache/uv')}
    for name in ('HOME','UV_CACHE_DIR'): Path(runtime_env[name]).mkdir(parents=True,exist_ok=True)
    os.environ.update(runtime_env)
    immutable_json(output/'runtime-environment.json',runtime_env)
    seed=42+request['round']-1
    testcase=prepare(package,request['work_items'],request['allocation']['counts'],output/'preparation',seed,execute_native=execute,repo=repo)
    binding=bind_amp(package,output/'preparation',request['allocation']['counts'],native_tmp)
    immutable_json(output/'ancestry.json',binding)
    immutable_json(output/'request-binding.json',file_binding(request_file))
    if prepare_only:
        immutable_json(output/'prepare-completion.json',{'status':'COMPLETE','testcases':len(binding['cases'])})
        return
    shared=next(p for p in _SHARED if p.is_dir())
    command=[sys.executable,shared/'run_anomalygen.py','inference_only','--name',f'generator-round-{request["round"]}',
             '--output-dir',output/'native','--dataset-dir',output/'preparation/dataset-view',
             '--input-data-path',testcase,'--exclude-stems',package/'audit/exclude_stems.txt',
             '--checkpoint',g['checkpoint'],'--recipe',g['recipe'],'--base-checkpoint',g['base_checkpoint'],
             '--base-seed',str(seed),'--amp-seed',str(seed),'--num-search-run','3','--score','nn']
    execute(command,output,repo)
    admission=admit_generation(output/'native/manifest.json',output/'ancestry.json',package,request['round'])
    immutable_json(output/'admission.json',admission)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',required=True)
    p.add_argument('--output',default=os.environ.get('TAO_HOST_RESULTS_DIR'))
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    if not args.output: p.error('host-identical output path is required')
    run(args.request,args.output,prepare_only=args.prepare_only)

if __name__=='__main__': main()
