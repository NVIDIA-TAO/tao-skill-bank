# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU integration fixtures exercise native boundaries, actual population and recovery."""
import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from test_generator_lineage import Fixture, image, lines
from generator_contract import bind_amp, verify_ancestry, file_binding, sha256, pixel_identity
from generator_worker import prepare
from generator_preflight import tree_binding, verify_tree, verify_resume, TRAINING
from generator_driver import GeneratorSuite
from generator_store import GeneratorStore
from suite_driver import Suite
from suite_runtime import json_write


def add_clean(f):
    row=copy.deepcopy(f.sources[0]); row['source_stem']='extra';row['matched_identity']['source_id']='extra.png'
    row['matched_identity']['joint']=['extra','scan']
    for asset in row['assets']:
        path=f.package/Path(asset['prepared_path']).with_name('extra.png')
        image(path,(7,8,9) if asset['asset_role']=='clean_image' else 255,'RGB' if asset['asset_role']=='clean_image' else 'L')
        asset.update(prepared_path=str(path.relative_to(f.package)),prepared_sha256=sha256(path),original_sha256=sha256(path),bytes=path.stat().st_size,archive_member='original/extra.png')
        if asset['asset_role']=='clean_image':
            row['pixel_proof']['rgb_sha256']=pixel_identity(path)['rgb_sha256']
            row['matched_identity']['image_sha256']=sha256(path)
    f.sources.append(row)
    lines(f.package/'dataset/defect_spec.jsonl',[{'defect_type':k,'spatial_dependency':'cad_tht'} for k in f.mapping]);f.reseal()


def native_fixture(tmp,calls):
    def execute(command,root,repo):
        command=list(map(str,command));module=command[command.index('-m')+1];calls.append(command)
        def arg(key): return Path(command[command.index(key)+1])
        if module.endswith('roi_allocate'):
            json_write(arg('--output_allocation'),json.loads(command[command.index('--per_defect_counts')+1]))
        elif module.endswith('roi_pair'):
            view=arg('--dataset_dir');counts=json.loads(arg('--allocation').read_text());kind=next(k for k,n in counts.items() if n)
            texture,defect=kind.split('+');rows=[]
            # Native order differs from requested anchor order; adapter must join
            # by authoritative clean identity, never by CSV/pair position.
            cleans=list(reversed(sorted((view/texture/'clean_image').glob('*.png'))))[:counts[kind]]
            donor=next((view/texture/'mask'/defect).glob('*.png'))
            for clean in cleans:
                rows.append({'clean_image':str(clean),'defect_type':kind,'submask':str(donor),
                             'cad_mask':str(view/texture/'cad_mask'/clean.name),
                             'cad_mask_label':str(view/'semantic_segmentation_labels.json'),'n_seeds':1})
            json_write(arg('--output_pair_path'),rows)
        else:
            pairs=json.loads(arg('--input_pair_path').read_text());output=arg('--output_dir');cases=[];summary=[]
            for i,pair in enumerate(pairs):
                donor=Path(pair['submask']);processed=image(tmp/'native-tmp'/str(i)/(donor.stem+'_largest.png'),255,'L')
                placed=image(output/Path(pair['clean_image']).stem/pair['defect_type']/(processed.stem+'__seed0.png'),255,'L')
                cases.append({'image_filename':pair['clean_image'],'mask_filename':str(placed),'anomaly_type':pair['defect_type'],
                              'guidance':3.,'num_steps':35,'crop_ratio':2.,'num_generated_images':1,'poisson_blend':False,'iteration_generation_max_instance':5})
                summary.append({'name':Path(pair['clean_image']).stem,'defect_type':pair['defect_type'],'status':'OK','seeds':{'0':{'status':'OK'}}})
            json_write(output/'summary.json',summary);lines(output/'testcase.jsonl',list(reversed(cases)))
    return execute


def test_anchored_per_class_native_pairing_and_merged_lineage(tmp_path):
    f=Fixture(tmp_path);add_clean(f);types=list(f.mapping);counts={k:int(i<2) for i,k in enumerate(types)}
    cleans=[r for r in f.sources if r['kind']=='clean'];work=[]
    for slot,row in enumerate(cleans):
        asset=next(a for a in row['assets'] if a['asset_role']=='clean_image')
        work.append({'slot':slot,'anomaly_type':types[slot],'label':f.mapping[types[slot]],'anchor_id':f'q{slot}',
                     'anchor_mode':'errors','cosine':.9,'clean':{'id':row['matched_identity']['source_id'],'stem':row['source_stem'],
                                                              'prepared_path':asset['prepared_path'],'image_sha256':asset['prepared_sha256'],'cad_mask':{'prepared_path':row['assets'][1]['prepared_path'],'sha256':row['assets'][1]['prepared_sha256']}}})
    calls=[];output=tmp_path/'anchored'
    prepare(f.package,work,counts,output,42,execute_native=native_fixture(tmp_path,calls),repo=tmp_path)
    assert len(calls)==4 and calls[0][2].endswith('roi_allocate') and calls[-1][2].endswith('roi_place')
    bound=bind_amp(f.package,output,counts,tmp_path/'native-tmp');verify_ancestry(bound,f.package)
    assert [r['work_item']['anchor_id'] for r in bound['cases']]==['q0','q1']
    assert [r['testcase']['index'] for r in bound['cases']]==[0,1]
    assert len({r['clean_source']['matched_identity']['source_id'] for r in bound['cases']})==2
    pairfile=output/'pairing.json';pairing=json.loads(pairfile.read_text());pairing['resolved'][0]['clean']['id']='substituted'
    json_write(pairfile,pairing)
    with pytest.raises(ValueError): verify_ancestry(bound,f.package)


def test_native_fallback_excludes_reserved_other_class_source(tmp_path):
    f=Fixture(tmp_path);add_clean(f);types=list(f.mapping);counts={k:int(i<2) for i,k in enumerate(types)}
    row=f.sources[0];asset=row['assets'][0]
    work=[{'slot':0,'anomaly_type':types[0],'label':f.mapping[types[0]],'anchor_id':None,'anchor_mode':'native_fallback','cosine':None,'clean':None},
          {'slot':1,'anomaly_type':types[1],'label':f.mapping[types[1]],'anchor_id':'q','anchor_mode':'errors','cosine':.9,
           'clean':{'id':row['matched_identity']['source_id'],'stem':row['source_stem'],'prepared_path':asset['prepared_path'],'image_sha256':asset['prepared_sha256'],'cad_mask':{'prepared_path':row['assets'][1]['prepared_path'],'sha256':row['assets'][1]['prepared_sha256']}}}]
    calls=[];output=tmp_path/'fallback'
    prepare(f.package,work,counts,output,42,execute_native=native_fixture(tmp_path,calls),repo=tmp_path)
    resolved=json.loads((output/'pairing.json').read_text())['resolved']
    assert resolved[0]['clean']['id']=='extra.png'
    assert resolved[1]['clean']['id']==row['matched_identity']['source_id']
    with pytest.raises(FileExistsError): prepare(f.package,work,counts,output,42,execute_native=native_fixture(tmp_path,calls),repo=tmp_path)


def test_native_alias_symlink_and_bytes_are_frozen(tmp_path):
    root=tmp_path/'cache';root.mkdir();target=tmp_path/'weights';target.mkdir();(target/'model').write_bytes(b'weights')
    (root/'config').write_bytes(b'config');(root/'alias').symlink_to('/container-weights/model')
    bound=tree_binding(root,{'/container-weights':str(target)});verify_tree(bound)
    (root/'alias').unlink();(root/'alias').symlink_to('/container-weights/other')
    with pytest.raises(ValueError): verify_tree(bound)
    (root/'alias').unlink();(root/'alias').symlink_to('/container-weights/model');(target/'model').write_bytes(b'changed')
    with pytest.raises(ValueError): verify_tree(bound)


def test_resume_detects_changed_file_before_any_launch(tmp_path):
    source=tmp_path/'source';source.write_text('frozen')
    task={'bindings':{'files':[file_binding(source)],'trees':[]}}
    source.write_text('changed')
    with patch('generator_preflight.subprocess.check_output') as git:
        with pytest.raises(ValueError): verify_resume(task)
        git.assert_not_called()


def test_native_training_uses_admitted_population_and_own_continuation(tmp_path):
    suite=Suite.__new__(Suite);suite.root=tmp_path;suite.store=GeneratorStore(tmp_path)
    suite.task={'training':TRAINING,'model_path':str(tmp_path/'base'),'budgets':{'stage_wall_seconds':172800}}
    suite.public=tmp_path/'public';suite.prompt='prompt';launched=[]
    def population(ids,view,number):
        view.mkdir(parents=True,exist_ok=True);json_write(view/'train.json',[{'fixture':i} for i in range(3)])
        return 3
    suite.training_population=population
    def launch(stage,command,**kwargs):
        import tomllib
        spec=tomllib.loads(next(Path(host).read_text() for host,target,_ in kwargs['mounts'] if target=='/train.toml'))
        launched.append((stage,command,kwargs,spec));number=len(launched)
        job=tmp_path/'jobs'/str(number);cp=job/'run/checkpoints'/f'iter_{number*21:09d}'
        for name in ('model/.metadata','optim/.metadata','scheduler/.metadata','trainer/.metadata','dataloader/state'):
            p=cp/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('native-fixture')
        (cp.parent/'latest_checkpoint.txt').write_text(cp.name)
        (cp.parent.parent/'config.yaml').write_text('native: true')
        lines(cp.parent.parent/'training_telemetry.jsonl',[{'event':'batch_complete','examples':1}])
        lines(job/'status.json',[{'status':'SUCCESS','kpi':{'train/avg_loss':.1}}])
        return {'job':{'results_dir':str(job),'job_id':str(number)}}
    suite.run_job=launch;suite.export=lambda stage,job,ready: str(tmp_path/(stage.replace('/','-')))
    first=suite.train('G',['one-real-id'],7,round_index=1)
    second=suite.train('G',['one-real-id'],7,round_index=2,previous=first)
    assert first['iteration']==21 and second['iteration']==42
    assert first['unique_images']==3 and first['exposures']==21
    assert launched[0][3]['trainer']['max_iter']==21
    assert launched[1][3]['trainer']['max_iter']==42
    assert launched[1][3]['scheduler']['cycle_lengths']==[42]
    assert launched[0][3]['checkpoint']['load_path']=='???'
    assert launched[1][3]['checkpoint']['load_path']==first['dcp_host']
    assert 'checkpoint.keys_not_to_resume=[dataloader]' not in launched[0][1]
    assert 'checkpoint.keys_not_to_resume=[dataloader]' in launched[1][1]
    assert second['checkpoint'].endswith('G-export-2')


def test_generator_budget_uses_original_two_hour_envelope(tmp_path):
    suite=Suite.__new__(Suite);suite.root=tmp_path;suite.store=GeneratorStore(tmp_path)
    suite.task={'framework_image':'test@sha256:immutable','budgets':{'stage_wall_seconds':172800}}
    suite.deadline=10**12;suite.budget_scope=None
    a=suite.store.begin('G/generate/1');suite.store.finish(a,'infrastructure_error',{})
    suite.store.db.execute('UPDATE attempts SET started=1')
    with patch('suite_driver.Docker.submit') as submit:
        with pytest.raises(TimeoutError): suite.run_job('G/generate/1',[],stage_wall_seconds=7200)
        submit.assert_not_called()


def test_quality_selection_keeps_earliest_finite_nn_tie_and_rejects_replaced_image(tmp_path):
    import csv
    from generator_contract import verify_quality_selection
    dirs=[tmp_path/'sdg',tmp_path/'rounds/round_1']
    for number,directory in enumerate(dirs):
        for folder in ('reconstructed_image','original_image','original_mask'):
            image(directory/folder/'sample.png',(number,20,30))
        with (directory/'texture_ft_generation_result.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=['output_filename','index','anomaly_type']);writer.writeheader()
            writer.writerow({'output_filename':'sample.png','index':0,'anomaly_type':'Widget+defect'})
        json_write(directory/'kpi.json',{'Widget+defect':{'per_sample':[{'path':'sample.png','nn_score':.8}]}})
    import shutil
    selected=tmp_path/'searched';shutil.copytree(dirs[0],selected)
    result=verify_quality_selection({'selected_output':str(selected)},tmp_path,1)
    assert result[0]['tie_priority']==0
    image(selected/'reconstructed_image/sample.png',(1,20,30))
    with pytest.raises(ValueError,match='highest finite NN'):
        verify_quality_selection({'selected_output':str(selected)},tmp_path,1)


def test_dedup_retains_earliest_synthetic_and_conflicting_label_stops():
    from generator_contract import deduplicate_records
    early={'id':'early','label':'Bridge','pixel_key':'same','source':'synthetic','round':1,'index':9}
    late={**early,'id':'late','round':2,'index':0}
    kept,duplicates=deduplicate_records([late,early])
    assert [r['id'] for r in kept]==['early'] and len(duplicates)==1
    with pytest.raises(ValueError): deduplicate_records([early,{**late,'label':'Good'}])


def test_container_interpreter_chain_requires_exact_captured_target(tmp_path):
    root=tmp_path/'env';(root/'bin').mkdir(parents=True)
    (root/'bin/python').symlink_to('/image-only/python')
    (root/'bin/python3').symlink_to('python')
    captured=tmp_path/'captured';captured.write_bytes(b'original-image-interpreter')
    with pytest.raises(ValueError,match='unbound dangling'):
        tree_binding(root)
    targets={'/image-only/python':{'image':'test@sha256:pinned','artifact':file_binding(captured)}}
    bound=tree_binding(root,container_targets=targets);verify_tree(bound)
    assert len(bound['symlinks'])==2
    (root/'bin/python').unlink();(root/'bin/python').symlink_to('/another/image-python')
    with pytest.raises(ValueError): verify_tree(bound)


def test_cache_provenance_requires_original_completed_stage_output(tmp_path):
    import sqlite3
    from generator_preflight import validate_cache_provenance
    from tht_contract import json_hash
    public=tmp_path/'public';public.mkdir();(public/'SHA256SUMS').write_text('frozen');json_write(public/'bundle.json',{'fixture':True})
    base=tmp_path/'base';base.mkdir();json_write(base/'tao_conversion_provenance.json',{'prepared':{'fingerprint':'base'}})
    model=tmp_path/'siglip';model.mkdir();(model/'weights').write_bytes(b'original')
    task={'public_package':str(public),'private_package':str(tmp_path/'private'),'model_path':str(base),'siglip_path':str(model),'siglip_revision':'revision',
          'framework_image':'classifier@sha256:pinned','data_services_image':'embedding@sha256:pinned'}
    approved={**task,'base_fingerprint':'base','provenance':{'public_checksums_sha256':sha256(public/'SHA256SUMS'),
              'bundle_sha256':sha256(public/'bundle.json'),'siglip_files':{'weights':sha256(model/'weights')}}}
    preflight=tmp_path/'preflight.json';json_write(preflight,{'effective':approved,'preflight_sha256':json_hash(approved)})
    database=tmp_path/'original.sqlite';db=sqlite3.connect(database);db.execute('CREATE TABLE stages(name,status,result)')
    receipt={'original_preflight_semantic_sha256':json_hash(approved),'inventory':[],
             'readonly_original_stage_query':{'db':str(database)},'completed_original_producer_stages':[]}
    cache={'approved_preflight':file_binding(preflight)}
    for role in ('mining','proxy'):
        root=tmp_path/role;root.mkdir();vector=root/'vectors';vector.write_bytes(b'vectors-'+role.encode())
        json_write(root/'launch.json',{'fixture':role});terminal={'ExitCode':0,'Status':'exited','extra_native_field':1}
        json_write(root/'backend_terminal.json',terminal)
        cache[role]={'vectors':file_binding(vector),'launch':file_binding(root/'launch.json'),'terminal':file_binding(root/'backend_terminal.json')}
        result={'artifacts':[{'path':str(vector),'sha256':sha256(vector)}],'job':{'results_dir':str(root),'job_id':role}}
        stage={'name':'retrieval/embed-'+role,'status':'complete','result':result}
        db.execute('INSERT INTO stages VALUES(?,?,?)',(stage['name'],stage['status'],json.dumps(result)))
        receipt['completed_original_producer_stages'].append({**stage,'terminal':{'ExitCode':0,'Status':'exited'}})
        receipt['inventory'].extend(cache[role].values())
    # An unrelated final-result row is intentionally not part of the allowed query.
    db.execute('INSERT INTO stages VALUES(?,?,?)',('G/benchmark/final','complete','not-json-sensitive-fixture'))
    db.commit()
    proof=tmp_path/'receipt.json';json_write(proof,receipt);cache['provenance']=file_binding(proof);task['retrieval_cache']=cache
    assert validate_cache_provenance(task)
    db.execute("UPDATE stages SET status='incomplete' WHERE name='retrieval/embed-proxy'");db.commit()
    with pytest.raises(ValueError,match='producer completion'):
        validate_cache_provenance(task)
    db.close()
