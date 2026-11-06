# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measured addendum export; old D is read only after completed frozen evaluation."""
import datetime
import json
import sqlite3
from pathlib import Path
from generator_contract import file_binding, verify_file, validate_admission, immutable_json
from suite_analysis import paired_bootstrap, setup_usage
from suite_costs import aggregate
from suite_data import identity_from_query
from tht_contract import load_jsonl, external_evaluator, json_hash, sha256


def old_d(root, task, expected):
    root=Path(root).resolve(strict=True)
    with sqlite3.connect('file:'+str(root/'suite.sqlite')+'?mode=ro',uri=True) as db:
        meta={k:json.loads(v) for k,v in db.execute("SELECT key,value FROM meta WHERE key IN ('config','freeze','protocol')")}
        if meta.get('protocol')!='tht-six-arm-v1': raise ValueError('comparison must use the original six-arm run')
        frozen=meta['freeze']
        if json_hash(frozen['settings'])!=frozen['sha256']: raise ValueError('old D freeze hash changed')
        if meta['config']['base_fingerprint']!=task['base_fingerprint'] or meta['config']['public_package']!=task['public_package']:
            raise ValueError('comparison source data/base differs')
        path=root/'metrics/D/benchmark/final.json'
        binding=db.execute('SELECT path,sha256 FROM artifacts WHERE name=?',('D/benchmark/final/metrics',)).fetchone()
        if not binding or binding[0]!=str(path) or sha256(path)!=binding[1]: raise ValueError('original D metric artifact differs')
        stage=db.execute('SELECT status FROM stages WHERE name=?',('D/benchmark/final',)).fetchone()
        if not stage or stage[0]!='complete': raise ValueError('old D evaluation incomplete')
    scored=json.loads(path.read_text())
    if {r['id'] for r in scored['predictions']}!=expected or len(scored['predictions'])!=len(expected): raise ValueError('old D coverage differs')
    return scored,{'metrics':file_binding(path),'freeze':frozen,'run':str(root)}


def export(store, task, *, failure=None, comparison_run=None):
    root=store.root; frozen=store.get('freeze')
    if frozen: store.verify_freeze()
    expected={q['id'] for q in load_jsonl(Path(task['public_package'])/'benchmark/queries.jsonl')} if frozen else set()
    complete=bool(frozen) and store.completed_benchmark(expected)
    if comparison_run and not complete: raise ValueError('old D comparison forbidden before frozen final evaluation completes')
    metrics_path=root/'metrics/G/benchmark/final.json'
    scored=json.loads(metrics_path.read_text()) if complete and metrics_path.is_file() else None
    if scored and (len(scored['predictions'])!=len(expected) or {r['id'] for r in scored['predictions']}!=expected): raise ValueError('metric coverage differs')
    usage=[json.loads(r['payload']) for r in store.db.execute('SELECT payload FROM usage')]
    attempts=[dict(r) for r in store.db.execute('SELECT * FROM attempts ORDER BY id')]
    stages=[dict(r) for r in store.db.execute('SELECT * FROM stages ORDER BY name')]
    events=[dict(r) for r in store.db.execute('SELECT * FROM events ORDER BY seq')]
    rounds=[store.get(f'G/round/{n}') for n in range(1,6) if store.get(f'G/round/{n}')]
    admissions=[]; commands=[]; telemetry=[]; generator_attempts=[]
    for number in range(1,6):
        binding=store.get(f'G/admission/{number}')
        if binding:
            admission=json.loads(verify_file(binding).read_text())
            validate_admission(admission,task['generator']['prepared_package']);admissions.append(admission)
    for attempt in attempts:
        if not attempt['job_id']: continue
        directory=root/'jobs'/attempt['job_id'];launch=directory/'launch.json'
        if launch.exists():
            commands.append({'stage':attempt['stage'],'attempt':attempt['id'],'status':attempt['status'],
                             'launch':json.loads(launch.read_text()),'launch_binding':file_binding(launch),
                             'native_commands':load_jsonl(directory/'commands.jsonl') if (directory/'commands.jsonl').exists() else []})
        if attempt['stage'].startswith('G/generate/'):
            import csv
            observed=[]
            for ledger in sorted((directory/'native').rglob('*.csv')):
                if ledger.name not in ('texture_ft_generation_result.csv','guardrail_blocked.csv'): continue
                try:
                    with ledger.open() as stream: records=list(csv.DictReader(stream))
                    observed.append({'file':file_binding(ledger),'rows':len(records),'identities':[{k:r.get(k) for k in ('index','output_idx','anomaly_type')} for r in records]})
                except (OSError,ValueError,csv.Error) as error:
                    observed.append({'path':str(ledger),'error':str(error),'rows':None})
            generator_attempts.append({'stage':attempt['stage'],'job_id':attempt['job_id'],'status':attempt['status'],
                                       'observed_ledgers':observed,'counting_scope':'raw observations including failed attempts; selected output is not an extra render; only sealed final admission counts toward logical generation'})
        for file in directory.rglob('training_telemetry.jsonl'):
            telemetry.append({'stage':attempt['stage'],'attempt_status':attempt['status'],'file':file_binding(file),'events':load_jsonl(file)})
    batches=[e for t in telemetry for e in t['events'] if e.get('event')=='batch_complete']
    actual={}
    for source,dest in [('examples','exposures'),('input_tokens','input_tokens'),('supervised_tokens','supervised_tokens'),('batch_seconds','batch_wall_seconds')]:
        known=[r[source] for r in batches if r.get(source) is not None]
        actual[dest]={'value':sum(known) if known else None,'known_batches':len(known),'total_batches':len(batches)}
    comparison={'status':'not_requested'}
    if comparison_run:
        old,provenance=old_d(comparison_run,task,expected)
        predictions={'G':{r['id']:r for r in scored['predictions']},'D':{r['id']:r for r in old['predictions']}}
        queries=load_jsonl(Path(task['public_package'])/'benchmark/queries.jsonl')
        comparison=paired_bootstrap(predictions,{q['id']:identity_from_query(q) for q in queries},
                                    external_evaluator(Path(task['public_package'])/'evaluator'),task['statistics'],pairs=(('G','D'),))
        comparison['original_d']=provenance
    operator=[e for e in events if e['kind']=='resume' and json.loads(e['payload'])['origin']=='operator']
    cutoff=datetime.datetime.now(datetime.timezone.utc).isoformat()
    report={'schema':'generator-addendum-report/1','scope':'engineering-addendum','arm':'G',
            'status':'complete' if scored else 'incomplete','metrics':scored,'freeze':frozen,
            'preflight':{'approval':store.get('approval'),'effective':task},
            'rounds':rounds,'admissions':admissions,'comparison':comparison,
            'costs':{'usage':usage,'aggregate':aggregate(usage),
                     'stages':{name:aggregate([u for u in usage if u.get('stage')==name]) for name in sorted({u.get('stage','unknown') for u in usage})},
                     'benchmark':aggregate([u for u in usage if u.get('phase')=='benchmark']),
                     'setup':[setup_usage(p,cutoff) for p in task.get('setup_token_exports',[])],
                     'price':None,'coverage':'Unknown costs remain null. Native GPU intervals measure allocation, not kernel-active time.'},
            'training':{'settings':task['training'],'epochs_per_round':7,'telemetry':telemetry,'actual_all_attempts':actual,
                        'planned_successful_round_exposure':{'real':sum(r['real_exposures'] for r in rounds),'synthetic':sum(r['synthetic_exposures'] for r in rounds)},
                        'limits':'Per-source real/synthetic exposures use verified admitted population times epochs; native telemetry measures actual total exposure including failed attempts.'},
            'generator_attempts':generator_attempts,
            'execution':{'commands':commands,'attempts':attempts,'stages':stages,'events':events,'failure':failure},
            'autonomy':{'operator_resumes':len(operator),'status':'assisted' if operator else 'unassisted',
                        'policy_intervention':'forbidden after approved preflight; all resumes retain sealed policy'},
            'limitations':{'generator_training_benchmark_exposure':'unknown',
                          'comparison':'Engineering addendum, not a leakage-free causal estimate of augmentation benefit.',
                          'synthetic_labels':'Intended generator classes; independently verified visual class correctness is not established.',
                          'known_source_overlap':task['overlap_disclosure']},
            'exported_at':cutoff}
    output=root/'report_data.json'
    if not scored or output.exists(): output=root/'exports'/('report-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    report['artifact']=str(output)
    immutable_json(output,report)
    from suite_runtime import json_write
    json_write(root/'latest-report.json',file_binding(output))
    return report
