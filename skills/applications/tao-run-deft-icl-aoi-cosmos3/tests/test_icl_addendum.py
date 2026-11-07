# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""H protocol boundary tests use only synthetic fixtures; no model launches."""
import copy
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from tht_contract import json_hash, sha256
from suite_store import SuiteStore
from suite_runtime import json_write


def api(name):
    assert (SCRIPTS / (name + '.py')).is_file(), 'ICL addendum is not implemented'
    return importlib.import_module(name)


def binding(path):
    return {'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size}


def policy():
    return {'toolset':'siglip-retrieval-v1','version':1,'rationale':'fixture', 'max_calls':1,
            'parameters':{'reference_count':1,'ordering':'nearest_first','instruction':''}}


def config(tmp_path):
    return {'protocol':'icl-addendum-v1','fixed_checkpoint':str(tmp_path/'model'),
            'base_fingerprint':'base','prompt_sha256':'prompt','training':{'seed':42},
            'statistics':{'method':'paired_joint_percentile','grouping':'object_name+normalized_scan_id',
                          'seed':42,'replicates':2000,'confidence':0.95},
            'icl_reference_capacity':4,'benchmark_count':271,
            'baseline_metrics':{'n':74,'mcc':0.01},
            'budgets':{'suite_wall_seconds':345600,'stage_wall_seconds':172800,
                       'icl_wall_seconds':21600,'icl_agent_tokens':1000000,
                       'trial_admission_seconds':1400,'query_wall_seconds':180,'query_agent_tokens':100000},
            'astra':{'image':'fixture','codex_binary':'fixture','python_binary':'fixture',
                     'python_stdlib':'fixture','system_libraries':'fixture','auth_file':'fixture',
                     'model':'fixture','effort':'high'}}


def make_store(tmp_path):
    m=api('icl_store'); task=config(tmp_path)
    store=m.ICLStore(tmp_path/'run')
    store.initialize(task,{'preflight_sha256':json_hash(task),'approved_by':'test','approved_at':'test'})
    return store,task


def freeze(store,task):
    selected={'checkpoint':task['fixed_checkpoint'],'policy':policy(),'policy_sha256':json_hash(policy())}
    a=store.begin('H/select');store.finish(a,'complete',selected)
    settings={'H':{**selected,'base_fingerprint':task['base_fingerprint'],
                   'preflight_sha256':json_hash(task),'prompt_sha256':task['prompt_sha256'],
                   'seed':42,'statistics':task['statistics'],
                   'preprocessing':'RGB/ImageOps.pad/256x256/NEAREST/PNG',
                   'generation':{'max_new_tokens':4096,'do_sample':False}}}
    store.freeze(settings)
    return settings


def test_isolated_store_rejects_prior_runs_before_writes(tmp_path):
    old=SuiteStore(tmp_path/'old');old.initialize({}, {'preflight_sha256':json_hash({})});old.db.close()
    before=sha256(tmp_path/'old/suite.sqlite')
    with pytest.raises(ValueError): api('icl_store').ICLStore(tmp_path/'old')
    assert sha256(tmp_path/'old/suite.sqlite')==before


def test_only_fixed_selection_can_freeze_and_final_gate_is_unique(tmp_path):
    store,task=make_store(tmp_path)
    answer={'id':'q','raw_answer':'INVALID','completed':True}
    with pytest.raises(ValueError):store.record_answer('H','benchmark','final',answer)
    with pytest.raises(ValueError):store.begin('H/benchmark/final')
    with pytest.raises(ValueError):store.begin('H/train/1')
    with pytest.raises(ValueError):store.begin('G/generate/1')
    settings=freeze(store,task)
    store.record_answer('H','benchmark','final',answer)
    store.record_answer('H','benchmark','final',answer)
    assert not store.completed_benchmark(['q'])  # scientific contract requires 271
    for candidate in ('again','trial-001'):
        with pytest.raises(ValueError):store.record_answer('H','benchmark',candidate,answer)
    with pytest.raises(ValueError):store.record_answer('G','proxy','trial-001',answer)
    with pytest.raises(ValueError):store.record_answer('H','benchmark','final',{**answer,'raw_answer':'Good'})
    with pytest.raises(ValueError):store.begin('H/proposal/trial-002')
    settings['H']['checkpoint']='another'
    with pytest.raises(ValueError):store.freeze(settings)


def test_durable_failed_usage_and_live_jobs_and_two_attempt_limit(tmp_path):
    store,task=make_store(tmp_path)
    store.start_budget('H',21600,1000000)
    store.db.execute('UPDATE budgets SET started=1')
    a=store.begin('H/proposal/trial-001');store.attach_job(a,'existing-job')
    with pytest.raises(ValueError):store.begin('H/proposal/trial-001')
    store.record_usage('existing-job',{'input_tokens':100,'output_tokens':50},budget='H')
    store.finish(a,'infrastructure_error',{})
    store.db.close();store=api('icl_store').ICLStore(tmp_path/'run')
    store.start_budget('H',21600,1000000)
    assert store.remaining('H')=={'wall_seconds':0,'tokens':999850}
    with pytest.raises(ValueError):store.start_budget('H',21601,1000000)
    a=store.begin('H/proposal/trial-001');store.finish(a,'infrastructure_error',{})
    with pytest.raises(ValueError):store.begin('H/proposal/trial-001')
    a=store.begin('H/proposal/trial-002');store.finish(a,'program_error',{})
    with pytest.raises(ValueError):store.begin('H/proposal/trial-002')
    with pytest.raises(ValueError):store.record_usage('unknown',{'input_tokens':None,'output_tokens':1},budget='H')


def test_frozen_baseline_extraction_and_source_drift(tmp_path):
    m=api('icl_preflight');p=tmp_path/'context.json'
    json_write(p,{'arm':'E','trials':[],'baseline_metrics':{'n':74,'mcc':0.01},
                  'max_trials':12,'reference_capacity':4})
    b=binding(p)
    assert m.baseline_metrics(b)=={'n':74,'mcc':0.01}
    json_write(p,{'arm':'E','trials':[{'policy':'old'}],'baseline_metrics':{'n':74}})
    with pytest.raises(ValueError):m.baseline_metrics(b)
    with pytest.raises(ValueError):m.baseline_metrics(binding(p))


def test_checkpoint_exact_manifest_membership_bytes_and_lineage(tmp_path):
    m=api('icl_preflight');model=tmp_path/'model';model.mkdir();base=tmp_path/'base';base.mkdir()
    (base/'config.json').write_text('{}')
    (model/'weights').write_bytes(b'weights')
    json_write(model/'export_manifest.json',{'base_model_fingerprint':{'files':{'config.json':sha256(base/'config.json')}}})
    manifest=tmp_path/'frozen.json';selection=tmp_path/'selection.json'
    json_write(manifest,[{'path':str(p),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':sha256(p)} for p in sorted(model.iterdir())])
    json_write(selection,{'phase':'proxy','status':'complete','n':74,'checkpoint':str(model),'round':2})
    source={'selection':binding(selection),'model_manifest':binding(manifest),'checkpoint':str(model)}
    assert m.verify_checkpoint(source,base)==str(model)
    with pytest.raises(ValueError):m.verify_checkpoint({**source,'checkpoint':str(base)},base)
    (model/'extra').write_text('unsealed')
    with pytest.raises(ValueError):m.verify_checkpoint(source,base)
    (model/'extra').unlink();(model/'weights').write_bytes(b'changed')
    with pytest.raises(ValueError):m.verify_checkpoint(source,base)


def test_e_settings_cannot_be_relaxed(tmp_path):
    m=api('icl_preflight');task=config(tmp_path)
    m.validate_controls(task)
    for section,key,value in [('budgets','icl_wall_seconds',21601),('budgets','icl_agent_tokens',999999),
                              ('budgets','query_wall_seconds',179),('budgets','trial_admission_seconds',1399),
                              ('training','seed',43)]:
        changed=copy.deepcopy(task);changed[section][key]=value
        with pytest.raises(ValueError):m.validate_controls(changed)
    for key,value in [('icl_reference_capacity',5),('max_icl_trials',13),('max_new_tokens',2048),('do_sample',True)]:
        with pytest.raises(ValueError):m.validate_controls({**task,key:value})


def test_search_uses_original_method_and_independent_adaptive_context(tmp_path):
    m=api('icl_driver');from suite_driver import Suite
    assert m.ICLSuite.search is Suite.search
    assert m.ICLSuite.run_agent is Suite.run_agent
    assert m.ICLSuite.prepare_inference_view is Suite.prepare_inference_view
    store,task=make_store(tmp_path)
    s=object.__new__(m.ICLSuite);s.task=task;s.store=store;s.root=store.root
    s.proxy=[{'id':'proxy'}];s.deadline=10**20;s.budget_scope=None
    contexts=[]
    def agent(stage,runtime,invoke,**kw):
        context=store.get(stage+'/context');contexts.append(context)
        return {'decision':{'action':'trial','policy':policy()}} if len(contexts)==1 else {'decision':{'action':'stop'}}
    def infer(arm,phase,candidate,model,**kw):
        json_write(s.root/'views'/arm/phase/candidate/'requests.json',[{'messages':[{'content':[{'type':'text','text':'exact prompt'}]}]}])
        return {'metrics':{'n':1,'mcc':0.5}}
    s.run_agent=agent;s.infer=infer
    with patch('suite_driver.generic_mounts',return_value=[]),patch('suite_driver.Docker'):
        result=s.search('H',task['fixed_checkpoint'])
    assert result['checkpoint']==task['fixed_checkpoint']
    assert result['search_stop_reason']=='agent_stop'
    assert contexts[0]['trials']==[] and contexts[0]['baseline_metrics']==task['baseline_metrics']
    assert len(contexts[1]['trials'])==1
    assert contexts[0]['max_trials']==12 and contexts[0]['reference_capacity']==4
    assert not store.stage('B/proxy-summary')['result']


def test_no_training_generation_or_unfrozen_inference(tmp_path):
    m=api('icl_driver');s=object.__new__(m.ICLSuite)
    s.store,s.task=make_store(tmp_path);s.proxy=[{'id':'proxy'}]
    for method in ('train','export'):
        with pytest.raises(ValueError):getattr(s,method)('H',[],1)
    with pytest.raises(ValueError):s.run_job('H/train/1',[])
    with pytest.raises(ValueError):s.run_job('G/generate/1',[])
    with pytest.raises(ValueError):s.infer('H','benchmark','final',s.task['fixed_checkpoint'],policy=policy())
    with pytest.raises(ValueError):s.infer('H','proxy','trial-001','another',policy=policy())
    with pytest.raises(ValueError):s.infer('G','proxy','trial-001',s.task['fixed_checkpoint'],policy=policy())


def test_comparison_export_forbidden_before_final_without_reading_old_results(tmp_path):
    m=api('icl_analysis');store,task=make_store(tmp_path)
    task['comparison_runs']={'G':'DO-NOT-READ','E':'DO-NOT-READ','F':'DO-NOT-READ'}
    with patch.object(m,'read_comparator',side_effect=AssertionError('historical results opened')):
        with pytest.raises(ValueError):m.export(store,task)


def test_results_root_cannot_alias_inputs_or_historical_runs(tmp_path):
    m=api('icl_preflight');old=tmp_path/'old';old.mkdir();(old/'artifact').write_text('original')
    for result in (old,old/'child',tmp_path):
        with pytest.raises(ValueError):m.validate_output_root(result,[old])
    fresh=tmp_path/'new';m.validate_output_root(fresh,[old])
    fresh.symlink_to(old,target_is_directory=True)
    with pytest.raises(ValueError):m.validate_output_root(fresh,[old])


def test_proxy_answers_cannot_enter_after_freeze(tmp_path):
    store,task=make_store(tmp_path);freeze(store,task)
    with pytest.raises(ValueError):
        store.record_answer('H','proxy','trial-002',{'id':'q','raw_answer':'Good','completed':True})


def test_entrypoint_status_rejects_old_journal_without_mutation(tmp_path):
    m=api('icl_addendum')
    store=SuiteStore(tmp_path/'old');store.initialize({}, {'preflight_sha256':json_hash({})});store.db.close()
    before=sha256(tmp_path/'old/suite.sqlite')
    with patch('sys.argv',['icl_addendum','status','--results-dir',str(tmp_path/'old')]):
        with pytest.raises(ValueError):m.main()
    assert sha256(tmp_path/'old/suite.sqlite')==before


def test_entrypoint_invalid_approval_cannot_create_output(tmp_path):
    m=api('icl_addendum');task=config(tmp_path);task['results_dir']=str(tmp_path/'never')
    pre=tmp_path/'preflight.json';approval=tmp_path/'approval.json'
    json_write(pre,{'ready':True,'effective':task,'preflight_sha256':json_hash(task)})
    json_write(approval,{'preflight_sha256':'wrong','approved_by':'test','approved_at':'test'})
    with patch('sys.argv',['icl_addendum','run','--preflight',str(pre),'--approval',str(approval)]):
        with pytest.raises(ValueError):m.main()
    assert not (tmp_path/'never').exists()


def test_complete_export_uses_frozen_comparators_and_original_bootstrap(tmp_path):
    m=api('icl_analysis');store,task=make_store(tmp_path)
    public=tmp_path/'public';task['public_package']=str(public);task['overlap_disclosure']='fixture two retained joints'
    ids=[f'SCN_scan{i}_MBO_part_A1.jpg' for i in range(255)]+[f'SCN_scan{i}_MBO_part_A2.jpg' for i in range(16)]
    queries=[{'id':q,'image':'fixture.png','image_sha256':'fixture'} for q in ids]
    path=public/'benchmark/queries.jsonl';path.parent.mkdir(parents=True);path.write_text(''.join(json.dumps(q)+'\n' for q in queries))
    # Freeze is bound to the store's original minimal fixture config.
    freeze(store,store.get('config'))
    predictions=[{'id':q,'truth':'Good','prediction':'Good'} for q in ids]
    scored={'metrics':{'n':271},'predictions':predictions}
    for q in ids:store.record_answer('H','benchmark','final',{'id':q,'raw_answer':'Good','completed':True})
    hp=store.root/'metrics/H/benchmark/final.json';json_write(hp,scored)
    store.seal('completed',{'metrics':str(hp),'sha256':sha256(hp)})
    task['comparison_runs']={};snapshots={}
    for arm in ('G','E','F'):
        root=tmp_path/arm;old=SuiteStore(root)
        old.seal('protocol','generator-addendum-v1' if arm=='G' else 'tht-six-arm-v1')
        old.seal('config',task);settings={arm:{'checkpoint':task['fixed_checkpoint']}}
        old.seal('freeze',{'settings':settings,'sha256':json_hash(settings)});old.seal('completed',{'done':True})
        mp=root/'metrics'/arm/'benchmark/final.json';json_write(mp,scored)
        old.artifact(arm+'/benchmark/final/metrics',mp)
        a=old.begin(arm+'/benchmark/final');old.finish(a,'complete',{})
        for q in ids:
            old.db.execute('INSERT INTO answers VALUES(?,?,?,?,?,?)',(arm,'benchmark','final',q,'{}','fixture'))
        old.db.close();task['comparison_runs'][arm]=str(root)
        snapshots[arm]=sha256(root/'suite.sqlite')
    class FixtureEvaluator:
        def score(self,rows):
            return {k:sum(r['prediction']==r['truth'] for r in rows)/len(rows) for k in ('mcc','accuracy','macro_f1','balanced_accuracy')}
    with patch.object(m,'external_evaluator',return_value=FixtureEvaluator()):
        report=m.export(store,task)
    assert report['comparison']['joint_groups']==255
    assert set(report['comparison']['pairs'])=={'H-G','H-E','H-F'}
    assert report['comparison']['config']['replicates']==2000
    assert report['costs']['price'] is None
    assert Path(report['artifact']).is_file()
    for arm in snapshots:assert sha256(Path(task['comparison_runs'][arm])/'suite.sqlite')==snapshots[arm]


def test_trusted_gold_binding_is_checked_at_scoring_boundary(tmp_path):
    m=api('icl_preflight');private=tmp_path/'private';gold=private/'evaluation/benchmark-gold.jsonl'
    gold.parent.mkdir(parents=True);gold.write_text('fixture trusted gold')
    (private/'SHA256SUMS').write_text(sha256(gold)+'  evaluation/benchmark-gold.jsonl\n')
    m.verify_gold({'private_package':str(private)},'benchmark')
    gold.write_text('changed gold')
    with pytest.raises(ValueError):m.verify_gold({'private_package':str(private)},'benchmark')


def test_native_command_cannot_hide_training_under_inference_stage(tmp_path):
    m=api('icl_driver');s=object.__new__(m.ICLSuite);s.store,s.task=make_store(tmp_path)
    with patch('suite_driver.Suite.run_job',side_effect=AssertionError('native submission reached')):
        with pytest.raises(ValueError):s.run_job('H/proxy/trial-001',['-m','cosmos_framework.scripts.train'])
        with pytest.raises(ValueError):s.run_job('retrieval/embed-mining',['generator'])


def test_classifier_view_bytes_survive_resume_without_repair(tmp_path):
    m=api('icl_driver');s=object.__new__(m.ICLSuite);s.store,s.task=make_store(tmp_path)
    view=tmp_path/'view';view.mkdir();(view/'requests.json').write_text('[]');(view/'q.png').write_bytes(b'image')
    s.seal_view('H/proxy/trial-001',view)
    s.seal_view('H/proxy/trial-001',view)
    (view/'q.png').write_bytes(b'changed')
    with pytest.raises(ValueError):s.seal_view('H/proxy/trial-001',view)


def test_completed_run_retries_only_export_not_model_calls(tmp_path):
    m=api('icl_driver');s=object.__new__(m.ICLSuite);s.store,s.task=make_store(tmp_path)
    freeze(s.store,s.task);s.store.seal('completed',{'status':'complete'})
    with patch('icl_analysis.export',return_value={'status':'complete','artifact':'fixture'}) as export:
        assert s.run()['artifact']=='fixture'
        assert export.call_count==1


def test_readonly_preflight_and_resume_bind_every_input_without_benchmark_access(tmp_path,monkeypatch):
    m=api('icl_preflight');old=config(tmp_path)
    public=tmp_path/'public';private=tmp_path/'private';base=tmp_path/'base';model=tmp_path/'g/hf';encoder=tmp_path/'encoder'
    for p in (public,private,base,model,encoder):p.mkdir(parents=True)
    (public/'image').write_bytes(b'image');(base/'config.json').write_text('{}');(encoder/'config.json').write_text('{}')
    (model/'weights').write_bytes(b'weights')
    json_write(model/'export_manifest.json',{'base_model_fingerprint':{'files':{'config.json':sha256(base/'config.json')}}})
    conversion=base/'tao_conversion_provenance.json'
    json_write(conversion,{'prepared':{'fingerprint':'base','files':[{'name':'config.json','size':2,'sha256':sha256(base/'config.json')}]}})
    manifest=model.parent/'frozen-model.json';selection=model.parent/'selections/G.json'
    json_write(manifest,[{'path':str(p),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':sha256(p)} for p in model.iterdir()])
    json_write(selection,{'checkpoint':str(model),'phase':'proxy','status':'complete','n':74})
    def row(i,view=1):return {'id':f'SCN_scan{i}_MBO_part_A{view}.jpg','image':'image','image_sha256':sha256(public/'image')}
    mining=[{**row(i),'object_name':'MBO_part','scan_id':f'scan{i}','labels':['Good']} for i in range(3642)]
    for role,rows in [('mining',mining),('proxy',[row(i) for i in range(74)]),
                      ('benchmark',[row(i) for i in range(255)]+[row(i,2) for i in range(16)])]:
        p=public/role/('samples.jsonl' if role=='mining' else 'queries.jsonl');p.parent.mkdir()
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    json_write(public/'bundle.json',{'role_counts':m.EXPECTED_COUNTS,'known_source_joint_overlaps':2})
    (public/'SHA256SUMS').write_text(''.join(sha256(p)+'  '+str(p.relative_to(public))+'\n' for p in sorted(public.rglob('*')) if p.is_file()))
    (private/'evaluation').mkdir();(private/'evaluation/proxy-gold.jsonl').write_text('fixture proxy gold')
    forbidden=private/'evaluation/benchmark-gold.jsonl';forbidden.write_text('NEVER OPEN DURING PREFLIGHT')
    (private/'SHA256SUMS').write_text('fixture private inventory')
    profile=tmp_path/'profile.json';json_write(profile,{'prompt':'fixture prompt'})
    evidence=tmp_path/'evidence.json';json_write(evidence,{'fixture':True})
    env=tmp_path/'env.json';json_write(env,{'host':'fixture-host'})
    codex=tmp_path/'codex';codex.write_text('fixture binary');auth=tmp_path/'auth';auth.write_text('NEVER READ AUTH')
    old.update(mode='tht_single',platform='docker',skip_sdg=True,public_package=str(public),private_package=str(private),
               model_path=str(base),siglip_path=str(encoder),siglip_revision='revision',framework_image='framework@sha256:fixture',
               data_services_image='embedding@sha256:fixture',direct_profile=str(profile),m3_evidence=str(evidence),
               results_dir=str(tmp_path/'old'),max_icl_trials=12,max_new_tokens=4096,do_sample=False,budget_policy={},
               overlap_disclosure='fixture',prompt_sha256='fixture')
    old['astra'].update(codex_binary=str(codex),auth_file=str(auth),codex_binary_sha256=sha256(codex),image='runtime@sha256:fixture')
    old['provenance']={'bundle_sha256':sha256(public/'bundle.json'),'public_checksums_sha256':sha256(public/'SHA256SUMS'),
                      'direct_profile_sha256':sha256(profile),'m3_evidence_sha256':sha256(evidence),
                      'model_conversion_sha256':sha256(conversion),'siglip_files':{'config.json':sha256(encoder/'config.json')}}
    source_store=SuiteStore(model.parent)
    source_store.seal('protocol','generator-addendum-v1');source_store.seal('config',old)
    source_settings={'G':{'checkpoint':str(model),'model_manifest':{'path':str(manifest),'sha256':sha256(manifest)}}}
    source_store.seal('freeze',{'settings':source_settings,'sha256':json_hash(source_settings)})
    a=source_store.begin('G/select');source_store.finish(a,'complete',{})
    source_store.artifact('G/select',selection);source_store.artifact('G/frozen-model',manifest);source_store.db.close()
    original=tmp_path/'original.json';json_write(original,old)
    pre=tmp_path/'original-preflight.json';json_write(pre,{'ready':True,'effective':old,'preflight_sha256':json_hash(old)})
    context=tmp_path/'context.json';json_write(context,{'arm':'E','trials':[],'max_trials':12,'reference_capacity':4,'baseline_metrics':old['baseline_metrics']})
    request={'protocol':'icl-addendum-v1','reference_task':binding(original),'reference_preflight':binding(pre),
             'baseline_context':binding(context),'source_model':{'checkpoint':str(model),'selection':binding(selection),'model_manifest':binding(manifest)},
             'results_dir':str(tmp_path/'h-run'),'environment_manifest':binding(env),'storage_required_bytes':1,
             'comparison_runs':{'G':str(model.parent),'E':str(tmp_path/'old'),'F':str(tmp_path/'old')}}
    original_read=Path.read_text
    def guarded_read(path,*args,**kwargs):
        assert path not in (forbidden,auth), 'forbidden data read'
        return original_read(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',guarded_read)
    monkeypatch.setattr(m,'PROMPT_SHA',__import__('hashlib').sha256(b'fixture prompt').hexdigest())
    monkeypatch.setattr(m,'external_evaluator',lambda p:None)
    monkeypatch.setattr(m,'generic_mounts',lambda *args:[])
    monkeypatch.setattr(m,'code_identity',lambda:{'commit':'fixture','branch':'fixture'})
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:'codex-cli 0.166.0')
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:None)
    result=m.validate(request)
    assert result['ready'] and result['effective']['baseline_metrics']==old['baseline_metrics']
    assert not Path(request['results_dir']).exists()
    m.verify_resume(result['effective'])
    (public/'image').write_bytes(b'changed')
    with pytest.raises(ValueError):m.verify_resume(result['effective'])


@pytest.mark.parametrize('complete',[True,False])
def test_pending_proposal_is_reconciled_before_expired_admission_without_new_launch(tmp_path,complete):
    m=api('icl_driver');s=object.__new__(m.ICLSuite);s.store,s.task=make_store(tmp_path);s.root=s.store.root;s.deadline=100
    s.store.start_budget('H',21600,1000000);s.store.db.execute('UPDATE budgets SET started=1')
    stage='H/proposal/trial-001';attempt=s.store.begin(stage);s.store.attach_job(attempt,'original-job')
    s.store.db.execute('UPDATE attempts SET started=1')
    context={'arm':'H','trials':[],'baseline_metrics':s.task['baseline_metrics'],'remaining_budget':{'tokens':1000000,'wall_seconds':21600}}
    s.store.seal(stage+'/context',context)
    output=s.root/'proposals/H/trial-001';json_write(output/'context.json',context)
    native=tmp_path/'native';json_write(native/'launch.json',{'submitted_at_unix':2})
    class Runtime:
        def record(self,*args):return json.dumps({'results_dir':str(native),'backend_ref':'original-container'})
        def logs(self,*args):return ''
        def submit(self,*args,**kwargs):raise AssertionError('must not launch during reconciliation')
    def proposal(runtime,mounts,auth,model,effort,context,output,wall,*,job,on_open=None):
        assert job['job_id']=='original-job' and wall<=98
        return ({'action':'trial','policy':policy()} if complete else None),{
            'job':job,'usage':{'input_tokens':123,'output_tokens':45},
            'backend':{'state':'COMPLETE' if complete else 'CANCELED','native':{}},
            'started_at_unix':2,'ended_at_unix':100,'budget_exhausted':not complete}
    with patch.object(m,'Docker',return_value=Runtime()),patch('astra_isolation.generic_mounts',return_value=[]),patch('policy_agent.propose',side_effect=proposal):
        if complete:s.reconcile_proposals()
        else:
            with pytest.raises(TimeoutError):s.reconcile_proposals()
    assert s.store.remaining('H')=={'wall_seconds':0,'tokens':999832}
    assert s.store.db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]==1
    assert s.store.stage(stage)['status']==('complete' if complete else 'incomplete')


def test_freeze_refuses_every_pending_native_attempt(tmp_path):
    store,task=make_store(tmp_path)
    attempt=store.begin('H/proxy/trial-002');store.attach_job(attempt,'original-native')
    with pytest.raises(ValueError,match='pending|running|active'):
        freeze(store,task)
    assert not store.get('freeze')


@pytest.mark.parametrize('stage,complete,late',[('H/proxy/trial-002',True,False),('H/proxy/trial-002',False,False),
                                         ('retrieval/embed-proxy',True,False),('retrieval/embed-mining',False,False),
                                         ('H/proxy/trial-002',True,True)])
def test_native_reconciliation_preserves_handles_deadlines_answers_costs(tmp_path,stage,complete,late):
    m=api('icl_driver');s=object.__new__(m.ICLSuite);s.store,s.task=make_store(tmp_path);s.root=s.store.root
    s.deadline=100;s.budget_scope=None
    s.task.update(framework_image='fixture',data_services_image='fixture')
    s.store.start_budget('H',21600,1000000);s.store.db.execute('UPDATE budgets SET started=1')
    attempt=s.store.begin(stage);s.store.attach_job(attempt,'original-native');s.store.db.execute('UPDATE attempts SET started=1')
    native=tmp_path/'native';native.mkdir()
    answer={'id':'q','raw_answer':'INVALID','completed':True,'invocation_id':'query-original',
            'inference_input_tokens':11,'inference_output_tokens':2,'started_at_unix':2,'ended_at_unix':90}
    if stage.startswith('H/'):
        (native/'raw_answers.jsonl').write_text(json.dumps(answer)+'\n')
        json_write(native/'completion.json',{'count':1})
    else:
        output=s.root/'retrieval'/stage.split('/')[1]/'embeddings.parquet';output.parent.mkdir(parents=True);output.write_bytes(b'fixture')
    class Runtime:
        def record(self,*args):return json.dumps({'results_dir':str(native),'backend_ref':'original-container'})
        def wait(self,job,deadline):
            assert job=='original-native' and deadline==100
            return {'state':'COMPLETE' if complete else 'CANCELED','native':{'FinishedAt':'1970-01-01T00:01:45Z' if late else '1970-01-01T00:01:35Z'}}
        def logs(self,*args):return ''
        def submit(self,*args,**kwargs):raise AssertionError('must not launch during reconciliation')
    s.runtime=Runtime()
    def cost(runtime,job,gpu,stage):
        s.store.record_usage(job['job_id'],{'stage':stage,'gpu_seconds':93,'invocation_id':job['job_id']})
    s._native_cost=cost
    with patch.object(m,'Docker',return_value=s.runtime):
        if complete and not late:s.reconcile_native()
        else:
            with pytest.raises(TimeoutError):s.reconcile_native()
    assert s.store.db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]==1
    assert s.store.db.execute("SELECT COUNT(*) FROM usage WHERE invocation='original-native'").fetchone()[0]==1
    if stage.startswith('H/'):
        assert s.store.answers('H','proxy','trial-002')==[answer]
        assert s.store.db.execute("SELECT COUNT(*) FROM usage WHERE invocation='query-original'").fetchone()[0]==1
    assert s.store.stage(stage)['status']==('complete' if complete and not late else 'incomplete')


def test_completed_native_trial_is_recovered_before_search_admission(tmp_path):
    m=api('icl_driver');s=object.__new__(m.ICLSuite);s.store,s.task=make_store(tmp_path);s.root=s.store.root
    s.proxy=[{'id':'q'}];s.deadline=10**20;s.budget_scope=None
    s.store.start_budget('H',21600,1000000)
    import time
    s.store.db.execute('UPDATE budgets SET started=?',(time.time()-21000,))
    a=s.store.begin('H/proxy/trial-002');s.store.finish(a,'complete',{'job':{'job_id':'finished'}})
    json_write(s.root/'policies/H/trial-002.json',policy())
    json_write(s.root/'views/H/proxy/trial-002/requests.json',[{'messages':[{'content':[{'type':'text','text':'exact prompt'}]}]}])
    def infer(arm,phase,candidate,model,**kwargs):
        assert s.store.stage('H/proxy/trial-002')['status']=='complete'
        return {'metrics':{'n':1,'mcc':0.6}}
    s.infer=infer
    s.recover_trials()
    assert s.store.get('H/trial/trial-002')['mcc']==0.6
    s.store.seal('H/trial/trial-001',dict(s.store.get('H/trial/trial-002'),trial_id='trial-001',mcc=0.2))
    with patch('suite_driver.generic_mounts',return_value=[]),patch('suite_driver.Docker'):
        selected=s.search('H',s.task['fixed_checkpoint'])
    assert selected['trial_id']=='trial-002' and selected['search_stop_reason']=='search_budget'


def test_source_checkpoint_must_match_original_g_journal_metadata(tmp_path):
    m=api('icl_preflight');root=tmp_path/'g';store=SuiteStore(root)
    task={'base_fingerprint':'base','public_package':'fixture-public','private_package':'fixture-private',
          'prompt_sha256':'prompt','framework_image':'framework','data_services_image':'embedding',
          'siglip_revision':'encoder','statistics':config(tmp_path)['statistics']}
    selection=root/'selections/G.json';manifest=root/'frozen-model.json'
    json_write(selection,{'checkpoint':'selected'});json_write(manifest,[])
    source={'checkpoint':'selected','selection':binding(selection),'model_manifest':binding(manifest)}
    store.seal('protocol','generator-addendum-v1');store.seal('config',task)
    settings={'G':{'checkpoint':'selected','model_manifest':{'path':str(manifest),'sha256':sha256(manifest)}}}
    store.seal('freeze',{'settings':settings,'sha256':json_hash(settings)})
    a=store.begin('G/select');store.finish(a,'complete',{'checkpoint':'selected'})
    store.artifact('G/select',selection);store.artifact('G/frozen-model',manifest);store.db.close()
    before=sha256(root/'suite.sqlite')
    proof=m.source_lineage(source,root,task)
    assert proof['checkpoint']=='selected' and sha256(root/'suite.sqlite')==before
    with pytest.raises(ValueError):m.source_lineage({**source,'checkpoint':'another'},root,task)
    alternate=tmp_path/'alternate-selection.json';json_write(alternate,{'checkpoint':'selected'})
    with pytest.raises(ValueError):m.source_lineage({**source,'selection':binding(alternate)},root,task)
    with pytest.raises(ValueError):m.source_lineage(source,root,{**task,'base_fingerprint':'another-base'})


def test_host_runtime_distinguishes_venv_and_dependency_bytes(tmp_path,monkeypatch):
    m=api('icl_preflight');import types
    module=tmp_path/'module.py';module.write_text('fixture module')
    class Distribution:
        version='1.0';files=[Path('module.py')]
        def locate_file(self,path):return tmp_path/path
    monkeypatch.setattr(m.importlib.metadata,'distribution',lambda name:Distribution())
    monkeypatch.setattr(m.importlib.util,'find_spec',lambda name:types.SimpleNamespace(origin=str(module)))
    monkeypatch.setattr(m.sys,'executable',str(tmp_path/'venv/bin/python'))
    monkeypatch.setattr(m.sys,'prefix',str(tmp_path/'venv'))
    first=m.host_runtime()
    assert first['executable']==str(tmp_path/'venv/bin/python') and first['prefix']==str(tmp_path/'venv')
    monkeypatch.setattr(m.sys,'prefix',str(tmp_path/'other-venv'))
    assert first!=m.host_runtime()
    monkeypatch.setattr(m.sys,'prefix',str(tmp_path/'venv'))
    module.write_text('changed module')
    assert first!=m.host_runtime()


def test_run_reconciles_active_trial_below_reserve_before_freezing(tmp_path):
    import time,datetime
    m=api('icl_driver');s=object.__new__(m.ICLSuite);s.task=config(tmp_path)
    manifest=tmp_path/'manifest.json';json_write(manifest,[])
    s.task.update(source_model={'model_manifest':binding(manifest)},model_path='fixture-base',
                  framework_image='fixture',data_services_image='fixture')
    s.store=api('icl_store').ICLStore(tmp_path/'run');s.root=s.store.root
    s.store.initialize(s.task,{'preflight_sha256':json_hash(s.task),'approved_by':'test','approved_at':'test'})
    s.proxy=[{'id':'q'}];s.deadline=time.time()+5000;s.budget_scope=None
    s.store.start_budget('H',21600,1000000);s.store.db.execute('UPDATE budgets SET started=?',(time.time()-21000,))
    first={'phase':'proxy','status':'complete','n':1,'expected_n':1,'mcc':0.2,'trial_id':'trial-001',
           'images_per_query':2,'calls_per_query':1,'prompt_characters':5,'policy':policy(),'policy_sha256':json_hash(policy())}
    s.store.seal('H/trial/trial-001',first)
    stage='H/proxy/trial-002';a=s.store.begin(stage);s.store.attach_job(a,'already-running')
    native=tmp_path/'native';json_write(native/'completion.json',{'count':1})
    json_write(s.root/'policies/H/trial-002.json',policy())
    json_write(s.root/'views/H/proxy/trial-002/requests.json',[{'messages':[{'content':[{'type':'text','text':'exact prompt'}]}]}])
    class Runtime:
        def record(self,*args):return json.dumps({'results_dir':str(native),'backend_ref':'already-running'})
        def wait(self,job,deadline):
            assert job=='already-running' and 590<deadline-time.time()<610
            return {'state':'COMPLETE','native':{'FinishedAt':datetime.datetime.now(datetime.timezone.utc).isoformat()}}
        def submit(self,*args,**kwargs):raise AssertionError('new model calls are forbidden')
    s.runtime=Runtime()
    s._native_cost=lambda runtime,job,gpu,stage:s.store.record_usage(job['job_id'],{'gpu_seconds':1,'stage':stage})
    s.infer=lambda *args,**kwargs:{'metrics':{'n':1,'mcc':0.6}}
    s.benchmark=lambda:{'pending':s.store.db.execute("SELECT COUNT(*) FROM attempts WHERE status IN ('pending','running')").fetchone()[0],
                        'selected':s.result('H/select')['trial_id']}
    with patch.object(m,'Docker',return_value=s.runtime),patch('suite_driver.Docker',return_value=s.runtime),\
         patch('suite_driver.generic_mounts',return_value=[]),patch('icl_preflight.verify_checkpoint'):
        result=s.run()
    assert result=={'pending':0,'selected':'trial-002'}
    assert s.store.get('freeze') and s.store.db.execute('SELECT COUNT(*) FROM usage').fetchone()[0]==1
    assert s.store.db.execute("SELECT COUNT(*) FROM attempts WHERE stage LIKE 'H/proposal/%'").fetchone()[0]==0


def test_terminal_program_failure_cannot_be_skipped_on_later_resume(tmp_path):
    m=api('icl_driver');s=object.__new__(m.ICLSuite);s.store,s.task=make_store(tmp_path)
    attempt=s.store.begin('H/proxy/trial-002');s.store.finish(attempt,'program_error',{'reason':'deterministic fixture'})
    with patch.object(s,'reconcile_native',side_effect=AssertionError('terminal failure must stop before work')):
        with pytest.raises(RuntimeError,match='program|deterministic'):
            s.run()
