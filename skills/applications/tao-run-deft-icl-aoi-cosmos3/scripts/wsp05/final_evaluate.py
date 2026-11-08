# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owner-only once-per-frozen-config B evaluation; expose aggregates only."""
from pathlib import Path
import json,re,subprocess,time,traceback
import oracle
from wsp03_statistics import paired
from tht_contract import external_evaluator

def terminal_state(container):
    state=json.loads(subprocess.check_output(['docker','inspect','--format','{{json .State}}',container],text=True))
    if state.get('Running') or state.get('Status') not in ['exited','dead']:
        raise ValueError('scientist must be stopped before Bench-B')
    return {'container':container,'state':state['Status'],'exit_code':state.get('ExitCode')}

def freeze_gate(root):
    work=root/'work';private=root/'private'
    freeze=oracle.read(work/'FREEZE.json')
    if freeze.get('schema')!='wsp05-autonomous-freeze/1' or freeze.get('status') not in ['FROZEN','NO_VALID_PATCH']:
        raise ValueError('terminal scientific freeze required')
    if oracle.sha(work/'PLAN.json')!=freeze.get('plan_sha256'):raise ValueError('final plan binding changed')
    selected=freeze.get('finalists')
    if not isinstance(selected,list) or len(selected)>4 or (freeze['status']=='FROZEN' and not selected):
        raise ValueError('one to four frozen finalists required')
    if freeze['status']=='NO_VALID_PATCH' and (selected or not freeze.get('reason')):
        raise ValueError('explicit terminal negative evidence required')
    run=oracle.read(root/'runtime/run.json')
    state=terminal_state(run['job']['container_id'])
    old_run=root.parent/'W5/runtime/run.json'
    old_state=terminal_state(oracle.read(old_run)['job']['container_id'])
    if not (private/'oracle-terminal.json').exists():raise ValueError('oracle must stop before B')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('GPU must be empty before final evaluator')
    entries=oracle.ledger_entries(private)
    if len(entries)>20:raise ValueError('A query cap violated')
    records={};pairs=[];names=set();configurations=set()
    def load_query(q):
        if not isinstance(q,str) or not re.fullmatch(r'Q(?:0[1-9]|1[0-9]|20)',q):
            raise ValueError('frozen query must name an oracle identifier')
        path=private/'queries'/q
        response=oracle.read(path/'response.json')
        if response['status']!='COMPLETE':raise ValueError('final query must be complete')
        real=response.get('reused_query_id',q);actual=private/'queries'/real
        binding=oracle.read(actual/'snapshot/configuration.json')
        if oracle.sha(actual/'snapshot/configuration.json')!=response['configuration_sha256']:
            raise ValueError('frozen query configuration changed')
        return {'query_id':real,'configuration_sha256':response['configuration_sha256'],
                'config':binding,'snapshot':str(actual/'snapshot'),
                'a_evaluation':str(actual/'evaluation')}
    for row in selected:
        name=row['name']
        if not isinstance(name,str) or not name or name in names:raise ValueError('unique finalist names')
        names.add(name);candidate=load_query(row['query_id']);base=load_query(row['baseline_query_id'])
        if candidate['configuration_sha256'] in configurations:raise ValueError('duplicate finalist configuration')
        configurations.add(candidate['configuration_sha256'])
        a,b=candidate['config'],base['config']
        if b['patch_files'] is not None or any(a[k]!=b[k] for k in ['parent','profile','model_files','max_new_tokens','backend','worker_sha256','replay_sha256']):
            raise ValueError('matched unedited baseline required')
        baseline_name='BASE_'+b['parent']+'_'+b['profile']
        if baseline_name in records and records[baseline_name]['configuration_sha256']!=base['configuration_sha256']:
            raise ValueError('baseline config conflict')
        records[name]=candidate;records[baseline_name]=base;pairs.append((name,baseline_name))
    audit=oracle.read(root/'setup/development-audit.json')
    if audit.get('status')!='PASS' or audit.get('freeze_sha256')!=oracle.sha(work/'FREEZE.json'):
        raise ValueError('independent development evidence must bind this freeze')
    return {'freeze':freeze,'freeze_sha256':oracle.sha(work/'FREEZE.json'),'editor_states':[state,old_state],
            'development_audit_sha256':oracle.sha(root/'setup/development-audit.json'),
            'queries_used':len(entries),'records':records,'pairs':pairs,'checked_at':time.time()}

def run(root=oracle.ROOT):
    # Until the gate returns, no partition identities, B requests, labels or images are read.
    gate=freeze_gate(root);out=root/'private/final';out.mkdir(exist_ok=False)
    oracle.write(out/'GATE.json',gate)
    if gate['freeze']['status']=='NO_VALID_PATCH':
        result={'status':'NO_VALID_PATCH','reason':gate['freeze']['reason'],'queries_used':gate['queries_used'],
                'benchmark_b_passes':0,'interpretation':'finite study produced no eligible edited finalist; no model success claim'}
        oracle.write(out/'public-results.json',result);return result
    evaluated={};arm_b={};arm_full={};costs={}
    for name,binding in gate['records'].items():
        key=binding['configuration_sha256']
        if key not in evaluated:
            directory=out/('config-'+key[:16])
            result=oracle.evaluate(binding['config'],Path(binding['snapshot']),'B',directory)
            evaluated[key]=directory;costs[key]=result['cost']
        bscore=oracle.read(evaluated[key]/'private-score.json')
        ascore=oracle.read(Path(binding['a_evaluation'])/'private-score.json')
        arm_b[name]=bscore
        joined=ascore['predictions']+bscore['predictions']
        if len(joined)!=271 or len({r['id'] for r in joined})!=271:raise ValueError('A/B union differs')
        evaluator=external_evaluator(oracle.read(root/'private/upstream-bindings.json')['evaluator_directory'])
        arm_full[name]={'predictions':joined,'metrics':evaluator.score(joined)}
    oracle.write(out/'ALL_INFERENCE_COMPLETE.json',{'completed_at':time.time(),'unique_configurations':len(evaluated),
                 'all_first_attempts_only':True,'configurations':list(evaluated)})
    partition=oracle.read(root/'private/partition.json')
    identities={q:{'scan_id':joint[0],'object_name':joint[1]} for q,joint in partition['joints'].items()}
    bident={q:identities[q] for q in partition['ids']['B']}
    stats_b=paired(arm_b,bident,evaluator,expected_groups=128,pairs=gate['pairs'])
    stats_full=paired(arm_full,identities,evaluator,expected_groups=255,pairs=gate['pairs'])
    oracle.write(out/'private-all-scores.json',{'B':arm_b,'full':arm_full})
    result={'schema':'wsp05-autonomous-results/1','status':'COMPLETE','freeze_sha256':gate['freeze_sha256'],
       'Attention':['This Benchmark cohort was used previously. Bench-B was withheld only from this development, not globally unused.',
                    'Bench-A aggregates informed selection; full271 scores are Benchmark-informed and optimistically selected.',
                    'Bench-B contains136 images and128 joints; confidence intervals are wide.'],
       'Bench-B':stats_b,'Benchmark-informed':stats_full,'query_budget':{'used':gate['queries_used'],'limit':20},
       'finalists':gate['freeze']['finalists'],'configurations':{n:{'parent':v['config']['parent'],'profile':v['config']['profile'],
         'query_id':v['query_id'],'configuration_sha256':v['configuration_sha256']} for n,v in gate['records'].items()},
       'independent_b_costs':costs,'completed_at':time.time(),'unique_b_passes':len(evaluated)}
    oracle.write(out/'public-results.json',result)
    oracle.write(out/'COMPLETE.json',{'public_results_sha256':oracle.sha(out/'public-results.json'),'completed_at':time.time()})
    return {'status':'COMPLETE','public_results_sha256':oracle.sha(out/'public-results.json'),'unique_b_passes':len(evaluated)}

if __name__=='__main__':
    try:print(json.dumps(run()))
    except BaseException as error:
        # Public diagnostics contain source locations only, never row-bearing error values.
        print(json.dumps({'status':'FAILED','error_type':type(error).__name__,
              'locations':[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(error.__traceback__)]}))
        raise SystemExit(1)
