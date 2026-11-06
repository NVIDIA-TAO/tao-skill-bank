# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regressions for independently reproduced provenance and lifecycle defects."""
import json
import sys
from pathlib import Path
from unittest.mock import patch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from test_generator_lineage import Fixture
from generator_contract import verify_prepared, file_binding, bind_amp, verify_ancestry
from generator_preflight import tree_binding, verify_tree, verify_resume
from generator_driver import GeneratorSuite


def test_real_mining_schema_has_only_id(tmp_path):
    f=Fixture(tmp_path)
    assert all('source_id' not in row for row in f.mining)
    verify_prepared(f.package,f.public)


def test_resealed_inventory_cannot_replace_approved_input(tmp_path):
    f=Fixture(tmp_path);bound=file_binding(f.package/'inventory.json')
    task={'bindings':{'files':[],'trees':[]},'prepared_inventory':bound,
          'generator':{'prepared_package':str(f.package)},'public_package':str(f.public)}
    (f.package/'audit/exclude_stems.txt').write_text('protected\nnew-exclusion\n')
    rows=json.loads((f.package/'inventory.json').read_text())
    for row in rows:
        if row['path']=='audit/exclude_stems.txt':
            current=file_binding(f.package/row['path']);row.update(sha256=current['sha256'],bytes=current['bytes'])
    (f.package/'inventory.json').write_text(json.dumps(rows))
    with patch('generator_preflight.subprocess.check_output') as git:
        with pytest.raises(ValueError): verify_resume(task)
        git.assert_not_called()


@pytest.mark.parametrize('change',['bytes','extra'])
def test_whole_preparation_reference_view_is_bound(tmp_path,change):
    f=Fixture(tmp_path);bound=bind_amp(f.package,f.prep,{k:1 for k in f.mapping},f.tmp)
    target=f.prep/'dataset-view'/Path(f.sources[1]['assets'][0]['prepared_path']).relative_to('dataset')
    if change=='bytes': target.write_bytes(b'substitution')
    else: (target.parent/'extra.png').write_bytes(b'extra')
    with pytest.raises(ValueError): verify_ancestry(bound,f.package)


@pytest.mark.parametrize('change',['bytes','extra','remove','chain'])
def test_external_directory_targets_and_chains_are_frozen(tmp_path,change):
    root=tmp_path/'root';root.mkdir();(root/'config').write_text('config')
    old=tmp_path/'old';old.mkdir();(old/'weights').write_bytes(b'old')
    new=tmp_path/'new';new.mkdir();(new/'weights').write_bytes(b'new')
    intermediate=tmp_path/'intermediate';intermediate.symlink_to(old,target_is_directory=True)
    (root/'weights').symlink_to(intermediate,target_is_directory=True)
    bound=tree_binding(root);verify_tree(bound)
    if change=='bytes': (old/'weights').write_bytes(b'changed')
    elif change=='extra': (old/'extra').write_text('extra')
    elif change=='remove': (old/'weights').unlink()
    else: intermediate.unlink();intermediate.symlink_to(new,target_is_directory=True)
    with pytest.raises(ValueError): verify_tree(bound)


def test_symlink_cycles_are_rejected(tmp_path):
    (tmp_path/'config').write_text('config');(tmp_path/'loop').symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError): tree_binding(tmp_path)


def test_fresh_root_rejects_unowned_requests_before_store(tmp_path):
    root=tmp_path/'run';root.mkdir();(root/'requests.json').write_text('stale')
    with patch('generator_driver.verify_resume'),patch('generator_driver.GeneratorStore') as store:
        with pytest.raises(ValueError): GeneratorSuite({'results_dir':str(root)}, {})
        store.assert_not_called()


@pytest.mark.parametrize('budget',['suite','stage'])
@pytest.mark.parametrize('backend',['RUNNING','COMPLETE','unlaunched'])
def test_expired_resume_reconciles_handle_before_stopping(tmp_path,budget,backend):
    from suite_driver import Suite
    from generator_store import GeneratorStore
    suite=Suite.__new__(Suite);suite.root=tmp_path;suite.store=GeneratorStore(tmp_path)
    suite.task={'framework_image':'fixture@sha256:pinned','budgets':{'stage_wall_seconds':172800}}
    suite.deadline=1 if budget=='suite' else 10**12;suite.budget_scope=None
    attempt=suite.store.begin('G/generate/1');suite.store.db.execute('UPDATE attempts SET started=1')
    if backend!='unlaunched': suite.store.attach_job(attempt,'native-fixture')
    calls=[]
    class Runtime:
        state=backend
        def record(self,*args): calls.append('record');return json.dumps({'results_dir':str(tmp_path/'job'),'backend_ref':'fixture'})
        def status(self,*args): calls.append('status');return {'state':self.state,'native':{'Status':self.state}}
        def cancel(self,*args): calls.append('cancel');self.state='CANCELED'
        def submit(self,*args,**kwargs): pytest.fail('expired run must never submit')
    suite._native_cost=lambda *args:calls.append('cost')
    with patch('suite_driver.Docker',return_value=Runtime()):
        with pytest.raises(TimeoutError): suite.run_job('G/generate/1',[],stage_wall_seconds=7200)
    row=suite.store.db.execute('SELECT * FROM attempts').fetchone()
    assert row['status']=='budget_exhausted' and row['ended'] is not None
    detail=json.loads(row['detail'])
    if backend=='unlaunched': assert detail['launch_interrupted_before_handle'] and not calls
    else:
        assert 'cost' in calls and detail['backend']['state']==('CANCELED' if backend=='RUNNING' else 'COMPLETE')
        assert ('cancel' in calls)==(backend=='RUNNING')


@pytest.mark.parametrize('orphan',['stale-prompt','changed-image','extra-image','partial-valid'])
def test_resume_views_derive_expected_prompt_and_media(tmp_path,orphan):
    from generator_store import GeneratorStore
    from suite_data import query_view
    f=Fixture(tmp_path/'source');suite=GeneratorSuite.__new__(GeneratorSuite)
    suite.root=tmp_path/'run';suite.store=GeneratorStore(suite.root);suite.public=f.public;suite.prompt='FROZEN PROMPT'
    suite.task={};query={k:f.mining[0][k] for k in ('id','image','image_sha256')}
    view=suite.root/'views/G/proxy/base'
    query_view(f.public,[query], 'STALE' if orphan=='stale-prompt' else suite.prompt,view)
    if orphan=='changed-image': (view/'q-00000.png').write_bytes(b'changed')
    if orphan=='extra-image': (view/'extra.png').write_bytes(b'extra')
    if orphan=='partial-valid': (view/'requests.json').unlink()
    if orphan=='partial-valid':
        suite.prepare_inference_view(view,[query],None,'proxy')
        assert 'FROZEN PROMPT' in (view/'requests.json').read_text()
        suite.prepare_inference_view(view,[query],None,'proxy')
    else:
        with pytest.raises(ValueError): suite.prepare_inference_view(view,[query],None,'proxy')


def test_resume_training_spec_cannot_adopt_orphan(tmp_path):
    suite=GeneratorSuite.__new__(GeneratorSuite);spec=tmp_path/'train.toml';spec.write_text('wrong epochs')
    with pytest.raises(ValueError): suite.write_training_spec(spec,'frozen epochs')
    spec.unlink();suite.write_training_spec(spec,'frozen epochs');suite.write_training_spec(spec,'frozen epochs')


@pytest.mark.parametrize('boundary',['worker','admission'])
def test_old_inventory_rejected_at_runtime_boundaries(tmp_path,boundary):
    from generator_worker import run
    from generator_contract import validate_admission
    f=Fixture(tmp_path);bound=file_binding(f.package/'inventory.json');f.reseal()
    # Keep a valid current inventory but change the externally approved identity.
    bound['sha256']='0'*64
    if boundary=='worker':
        request=tmp_path/'request.json';request.write_text(json.dumps({'generator':{'prepared_package':str(f.package)},'prepared_inventory':bound}))
        with pytest.raises(ValueError): run(request,tmp_path/'attempt',prepare_only=True)
        assert not (tmp_path/'attempt').exists()
    else:
        with pytest.raises(ValueError): validate_admission({},f.package,expected_inventory=bound)


@pytest.mark.parametrize('change',['bytes','extra'])
def test_cumulative_admission_rechecks_entire_quality_copy(tmp_path,change):
    from generator_contract import admit_generation, validate_admission
    f=Fixture(tmp_path);bound=bind_amp(f.package,f.prep,{k:1 for k in f.mapping},f.tmp)
    ancestry=tmp_path/'ancestry.json';ancestry.write_text(json.dumps(bound));manifest=f.generation()
    admission=admit_generation(manifest,ancestry,f.package,1,expected_search_rounds=0)
    target=manifest.parent/'dataset-view'/Path(f.sources[1]['assets'][0]['prepared_path']).relative_to('dataset')
    if change=='bytes': target.write_bytes(b'changed-unused-donor-image')
    else: (target.parent/'extra.png').write_bytes(b'extra')
    with pytest.raises(ValueError): validate_admission(admission,f.package)


def test_link_parent_traversal_uses_effective_intermediate_directory(tmp_path):
    root=tmp_path/'root';root.mkdir();external=tmp_path/'external';external.mkdir()
    (external/'nested/deep').mkdir(parents=True)
    (external/'jump').symlink_to(external/'nested/deep',target_is_directory=True)
    (external/'weights').write_bytes(b'decoy')
    actual=external/'nested/weights';actual.write_bytes(b'actual')
    (root/'weights').symlink_to(str(external/'jump')+'/../weights')
    bound=tree_binding(root);verify_tree(bound)
    assert (root/'weights').read_bytes()==b'actual'
    actual.write_bytes(b'changed')
    with pytest.raises(ValueError): verify_tree(bound)
