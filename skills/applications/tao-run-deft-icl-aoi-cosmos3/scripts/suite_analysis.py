# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Brev-side scientific aggregation/export. Never runs an inference or search.

Metrics and resampled metrics delegate to the frozen external scorer. The
renderer receives measured aggregates, not responsibility for scoring data.
"""
from __future__ import annotations
import datetime
import json
import random
import shutil
import socket
from pathlib import Path
from tht_contract import CLASSES,external_evaluator,load_jsonl,sha256,json_hash
from suite_data import identity_from_query
from suite_costs import aggregate,interval_usage,FIELDS
from suite_runtime import json_write

METRICS=('mcc','accuracy','macro_f1','balanced_accuracy')
PAIRS=(('C','B'),('D','B'),('E','D'),('F','B'),('D','F'))
NAMES={'A':'Astra + Codex','B':'Nano zero-shot','C':'Nano + ICL','F':'Full Mining LoRA','D':'DEFT LoRA','E':'DEFT LoRA + ICL'}

def quantile(values,q):
    if not values: return None
    values=sorted(values);position=(len(values)-1)*q;lo=int(position);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(position-lo)

def paired_bootstrap(predictions,identities,evaluator,config):
    if config['method']!='paired_joint_percentile' or config['grouping']!='object_name+normalized_scan_id':
        raise ValueError('unsupported statistical protocol')
    if not predictions: return {'status':'unavailable','reason':'no completed arms'}
    ids=set(next(iter(predictions.values())))
    if any(set(rows)!=ids for rows in predictions.values()) or set(identities)!=ids:
        raise ValueError('paired bootstrap requires identical complete query coverage')
    groups={}
    for q in sorted(ids):
        ident=identities[q];groups.setdefault((ident['object_name'],ident['scan_id']),[]).append(q)
    keys=sorted(groups);rng=random.Random(config['seed']);draws={a:{m:[] for m in METRICS} for a in predictions}
    for _ in range(config['replicates']):
        sample=[q for _ in keys for q in groups[rng.choice(keys)]]
        for arm,rows in predictions.items():
            result=evaluator.score([rows[q] for q in sample])
            for metric in METRICS: draws[arm][metric].append(result[metric])
    alpha=(1-config['confidence'])/2
    actual={a:evaluator.score(list(rows.values())) for a,rows in predictions.items()}
    differences={}
    for a,b in PAIRS:
        if a not in predictions or b not in predictions: continue
        differences[a+'-'+b]={}
        for metric in METRICS:
            delta=[x-y for x,y in zip(draws[a][metric],draws[b][metric])]
            differences[a+'-'+b][metric]={'estimate':actual[a][metric]-actual[b][metric],
                                        'lower':quantile(delta,alpha),'upper':quantile(delta,1-alpha)}
    missing=[a+'-'+b for a,b in PAIRS if a not in predictions or b not in predictions]
    return {'status':'incomplete' if missing else 'complete','available_arms':sorted(predictions),
            'missing_pairs':missing,'config':config,'joint_groups':len(groups),'queries':len(ids),'pairs':differences,
            'limits':'Cluster percentile intervals preserve within-joint dependence; retained Mining/Benchmark overlaps remain a training leakage limitation, not repaired by bootstrap.'}

def pass_fail(rows,mapping):
    if mapping is None: return {'status':'unavailable','reason':'historical mapping source not supplied'}
    fail=set(mapping['fail']);passed=set(CLASSES)-fail
    if not fail or not fail<set(CLASSES): raise ValueError('invalid descriptive Fail/Pass mapping')
    return {'status':'complete','mapping':{'Fail':sorted(fail),'Pass':sorted(passed)},'provenance':mapping['provenance'],
            'fail_to_pass':sum(r['truth'] in fail and r['prediction'] in passed for r in rows),
            'fail_denominator':sum(r['truth'] in fail for r in rows),
            'pass_to_fail':sum(r['truth'] in passed and r['prediction'] in fail for r in rows),
            'pass_denominator':sum(r['truth'] in passed for r in rows),
            'invalid':sum(r['prediction']=='INVALID' for r in rows),
            'purpose':'descriptive historical paired-reference mapping; eight-class MCC remains primary'}

def setup_usage(file,cutoff):
    """A missing/bad setup log cannot erase otherwise valid experiment evidence."""
    header={};digest=None
    try:
        digest=sha256(file);events=load_jsonl(file);header=events[0]
        usage=interval_usage(events,header['interval_start'],cutoff,header['session_id'],header.get('session_started_at'))
    except (OSError,ValueError,TypeError,KeyError,IndexError) as error:
        usage={**{k:None for k in FIELDS},'reason':f'{type(error).__name__}: {error}'}
    return {'path':str(file),'sha256':digest,'usage':usage,'phase':header.get('phase') if isinstance(header,dict) else None,
            'coverage':'snapshot cutoff; refresh source exports for final reporting'}

def training_prefix_cost(usage,cutoff):
    if cutoff is None: return {'status':'unavailable','reason':'F checkpoint selection incomplete'}
    rows=[];unknown=0
    for row in usage:
        if not row.get('stage','').startswith('F/train/'): continue
        start,end=row.get('started_at_unix'),row.get('ended_at_unix')
        if start is None or end is None:
            unknown+=1;continue
        if start>=cutoff: continue
        clipped=dict(row,ended_at_unix=min(end,cutoff))
        if row.get('gpu_seconds') is not None:
            clipped['gpu_seconds']=row['gpu_seconds']*(min(end,cutoff)-start)/(end-start) if end>start else 0
        rows.append(clipped)
    return {'cutoff':cutoff,'costs':aggregate(rows),'unknown_intervals':unknown,
            'method':'Native F training allocations clipped at selected checkpoint ready time, including prior failed attempts and held allocations during earlier epoch validation. Child export/evaluation jobs excluded; not kernel-active time.'}

def export(store,task,*,failure=None):
    root=store.root;now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    frozen=store.get('freeze')
    if frozen: store.verify_freeze()
    expected_ids={q['id'] for q in load_jsonl(Path(task['public_package'])/'benchmark/queries.jsonl')} if frozen else set()
    usage=[json.loads(r['payload']) for r in store.db.execute('SELECT payload FROM usage')]
    attempts=[dict(r) for r in store.db.execute('SELECT * FROM attempts ORDER BY id')]
    stages=[dict(r) for r in store.db.execute('SELECT * FROM stages ORDER BY name')]
    artifacts=[dict(r) for r in store.db.execute('SELECT * FROM artifacts ORDER BY name')]
    predictions={};metrics={};arms=[];latencies={};tools=[];commands=[];training_telemetry=[]
    for attempt in attempts:
        job=attempt.get('job_id')
        if not job: continue
        # Job results are local application-owned files; no credential files.
        for directory in ('jobs','astra-jobs','policy-jobs'):
            launch=root/directory/job/'launch.json'
            if launch.exists():
                commands.append({'stage':attempt['stage'],'attempt':attempt['id'],'job_id':job,
                                 'status':attempt['status'],'launch':json.loads(launch.read_text()),
                                 'launch_path':str(launch),'launch_sha256':sha256(launch),
                                 'log_path':str(launch.parent/'container.log')})
                if '/train/' in attempt['stage']:
                    for file in launch.parent.rglob('training_telemetry.jsonl'):
                        training_telemetry.append({'stage':attempt['stage'],'job_id':job,'attempt_status':attempt['status'],
                                                   'path':str(file),'sha256':sha256(file),'events':load_jsonl(file)})
    for arm in 'ABCFDE':
        rows=store.answers(arm,'benchmark','final') if frozen else []
        metric_path=root/'metrics'/arm/'benchmark/final.json'
        metric=None
        if len(rows)==271 and metric_path.is_file():
            scored=json.loads(metric_path.read_text());metric=scored['metrics'];metrics[arm]=metric
            if {r['id'] for r in rows}!=expected_ids or {r['id'] for r in scored['predictions']}!=expected_ids or metric['n']!=271:
                raise ValueError('complete-arm export requires exact Benchmark identity coverage')
            predictions[arm]={r['id']:r for r in scored['predictions']}
        arm_usage=[u for u in usage if u.get('arm')==arm or u.get('stage','').startswith(arm+'/')]
        benchmark_usage=[u for u in arm_usage if u.get('phase')=='benchmark' or '/benchmark/' in u.get('stage','')]
        one_time_usage=[u for u in arm_usage if u not in benchmark_usage]
        durations=[r.get('wall_seconds',r.get('usage',{}).get('wall_seconds')) for r in rows]
        durations=[v for v in durations if v is not None]
        # A stores invocation timestamps in its durable run-agent result.
        if arm=='A':
            durations=[]
            for record in store.db.execute("SELECT result FROM stages WHERE name LIKE 'A/benchmark/%' AND status='complete'"):
                value=json.loads(record['result']);durations.append(value['ended_at_unix']-value['started_at_unix'])
                for t in value['usage']['tool_calls']:
                    tools.append({'job_id':value['job']['job_id'],'tool':t,'invocation_path':str(Path(value['job']['results_dir'])/'invocation.json')})
        latencies[arm]={'completed':len(rows),'measured':len(durations),'planned':271,
                        'p50_seconds':quantile(durations,.5),'p95_seconds':quantile(durations,.95),
                        'tokens_and_costs':aggregate(benchmark_usage),'one_time_costs':aggregate(one_time_usage),
                        'attribution':'Benchmark final inference only; Proxy/training/policy costs separated; shared retrieval in stage_costs'}
        selected=store.stage(arm+'/select');selection=json.loads(selected['result']) if selected['result'] else None
        arms.append({'id':arm,'name':NAMES[arm],'run_id':str(root),'planned_queries':271,'completed_queries':len(rows),
                     'status':'complete' if metric else 'incomplete','metrics':metric,'costs':aggregate(arm_usage),
                     'autonomy_status':('assisted' if store.get('interventions',[]) else 'unassisted')+('-complete' if metric else '-incomplete'),
                     'checkpoint':frozen['settings'][arm].get('checkpoint') if frozen else (selection or {}).get('checkpoint'),
                     'policy':frozen['settings'][arm].get('policy') if frozen else (selection or {}).get('policy')})
    paired={'status':'unavailable','reason':'Benchmark arms not complete'}
    if predictions:
        public=Path(task['public_package']);queries=load_jsonl(public/'benchmark/queries.jsonl')
        paired=paired_bootstrap(predictions,{q['id']:identity_from_query(q) for q in queries},external_evaluator(public/'evaluator'),task['statistics'])
    f=[store.get(f'F/epoch/{i}') for i in range(1,21) if store.get(f'F/epoch/{i}')]
    d=[store.get(f'D/round/{i}') for i in range(1,6) if store.get(f'D/round/{i}')]
    exposure={}
    selected_f=store.stage('F/select')['result']
    selected_cutoff=json.loads(selected_f).get('checkpoint_ready_at') if selected_f else None
    for arm in 'DF':
        batches=[e for entry in training_telemetry if entry['stage'].startswith(arm+'/') for e in entry['events'] if e['event']=='batch_complete']
        def totals(rows):
            return {'batches':len(rows),'actual_exposures':sum(r['examples'] for r in rows if r.get('examples') is not None) if rows else None,
                    'input_tokens':sum(r['input_tokens'] for r in rows if r.get('input_tokens') is not None) if rows else None,
                    'supervised_tokens':sum(r['supervised_tokens'] for r in rows if r.get('supervised_tokens') is not None) if any(r.get('supervised_tokens') is not None for r in rows) else None,
                    'supervised_token_coverage':sum(r.get('supervised_tokens') is not None for r in rows),
                    'batch_wall_seconds':sum(r['batch_seconds'] for r in rows) if rows else None,
                    'peak_allocated_bytes':max((r['peak_allocated_bytes'] for r in rows),default=None)}
        exposure[arm]={'all_attempts':totals(batches),'selected_checkpoint_prefix':totals([r for r in batches if r['ended_at_unix']<=selected_cutoff]) if arm=='F' and selected_cutoff else None}
    setup=[setup_usage(file,now) for file in task.get('setup_token_exports',[])]
    sample=None
    if predictions:
        arm=next(a for a in 'BCFDEA' if a in predictions);row=store.answers(arm,'benchmark','final')[0]
        request_file=root/'views'/arm/'benchmark/final/requests.json'
        request=None;assets=[]
        if request_file.is_file():
            request=next(r for r in json.loads(request_file.read_text()) if r['id']==row['id'])
            destination=root/'presentation/sample';destination.mkdir(parents=True,exist_ok=True)
            for message in request['messages']:
                for content in message['content']:
                    if content['type']!='image': continue
                    name=Path(content['image']).name;source=request_file.parent/name
                    target=destination/name;shutil.copyfile(source,target)
                    assets.append({'path':str(target.relative_to(root)),'sha256':sha256(target),
                                   'model_input_path':content['image']})
        sample={'arm':arm,'id':row['id'],'raw_answer':row['raw_answer'],'parsed_and_graded':predictions[arm][row['id']],
                'input_artifact':str(request_file) if request_file.is_file() else None,
                'actual_model_input':request,'permitted_assets':assets,
                'input_status':'complete' if request is not None else 'unavailable',
                'input_reason':None if request is not None else 'No native-worker request artifact available for this completed arm.',
                'boundary':'query/reference view -> isolated worker -> immutable raw response -> trusted external scorer'}
    from tool_evidence import summarize
    astra_invocations=[u for u in usage if u.get('stage','').startswith('A/benchmark/')]
    tool_summary=summarize(astra_invocations)
    tool_summary['completed_queries']=next(a['completed_queries'] for a in arms if a['id']=='A')
    json_write(root/'astra_trace_analysis.json',tool_summary)
    report={'schema_version':'tht-report/1','frozen':bool(frozen),'provenance':{'export_version':1,'cutoff':now,'host':socket.gethostname(),
             'run_id':str(root),'task':task,'code':task.get('code'),'artifacts':artifacts,'statistical_config':task['statistics'],
             'suite_db':str(root/'suite.sqlite')},'arms':arms,'results':{
        'environment':{'host_snapshot':task.get('environment'),'code':task.get('code'),
                       'runtime_images':{k:task.get(k) for k in ('framework_image','data_services_image')},
                       'astra':{k:v for k,v in task.get('astra',{}).items() if k!='auth_file'},
                       'model_path':task.get('model_path'),'base_fingerprint':task.get('base_fingerprint'),
                       'siglip_revision':task.get('siglip_revision'),
                       'embedding_preprocess':'RGB ImageOps.pad 256x256 NEAREST PNG, then pinned native SigLIP processor',
                       'provenance':task.get('provenance'),'m3_evidence':task.get('m3_evidence')},
        'execution_commands':commands,'preflight':{'approval':store.get('approval'),'effective':task},
        'freeze':frozen,'paired_differences':paired,
        'per_class':{a:{'classes':CLASSES,'metrics':m['per_class'],'errors':{c:m['per_class'][c]['support']-m['confusion_matrix'][i][i] for i,c in enumerate(CLASSES)},
                        'pass_fail':pass_fail(list(predictions[a].values()),task.get('pass_fail_mapping'))} for a,m in metrics.items()},
        'confusion_matrices':{a:{'labels':m['confusion_labels'],'matrix':m['confusion_matrix'],'axes':'rows truth; columns prediction'} for a,m in metrics.items()},
        'f_learning_curve':{'epochs':f,'selection':store.stage('F/select'),'attempts':[a for a in attempts if a['stage'].startswith('F/')],
                            'total_search_cost':aggregate([u for u in usage if u.get('stage','').startswith('F/') and '/benchmark/' not in u.get('stage','')]),
                            'selected_checkpoint_training_cost':training_prefix_cost(usage,selected_cutoff)},
        'd_iterations':d,'training_exposure':{'F':f,'D':d,'actual':exposure,'attempt_telemetry':training_telemetry,'shared_settings':task['training'],
              'limits':'native valid-label token counts are not full input-token exposure; per-job allocated GPU intervals are not kernel-active time'},
        'icl_trials':{a:{'trials':[store.get(f'{a}/trial/trial-{i:03d}') for i in range(1,13)],
                           'attempts':[t for t in attempts if t['stage'].startswith(a+'/')],'selection':store.stage(a+'/select')} for a in 'CE'},
        'stage_costs':{'formal':usage,'formal_aggregate':aggregate(usage),
                       'shared':[u for u in usage if not u.get('arm') and not any(u.get('stage','').startswith(a+'/') for a in 'ABCDEF')],
                       'setup':setup,'price':None},
        'inference_costs':latencies,'autonomy':{'stages':stages,'first_blocker':failure,'interventions':store.get('interventions',[]),
                       'completed_queries':sum(a['completed_queries'] for a in arms),'planned_queries':1626},
        'recovery':{'attempts':attempts,'state':str(root/'suite.sqlite'),'max_attempts_per_stage':2,
                    'command':'python scripts/suite.py run --preflight PREFLIGHT.json --approval APPROVAL.json',
                    'policy':'infrastructure only; no completed INVALID retry; original budgets persist'},
        'astra_tool_calls':tool_summary,'astra_trace':tool_summary['representative_traces'],'sample_walkthrough':sample,
        'wsp':{'status':'not_applicable','reason':'separate bounded experiment, outside this six-arm implementation/run'},
        'handoff':{'code':task.get('code'),'artifact_inventory':artifacts,'runtime':'native Cosmos Framework; Docker four verbs',
                   'integration_status':'evidence export awaiting coordinator review'}}}
    json_write(root/'report_data.json',report)
    return report
