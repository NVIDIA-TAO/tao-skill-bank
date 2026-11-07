# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Trusted postcompletion export. No historical Benchmark enters policy search.
import datetime
import json
import sqlite3
from pathlib import Path

from generator_contract import file_binding, immutable_json
from suite_analysis import paired_bootstrap, setup_usage
from suite_costs import aggregate
from suite_data import identity_from_query
from tht_contract import external_evaluator, json_hash, load_jsonl, sha256


def read_comparator(root, arm, task, expected):
    if arm not in ('G','E','F'):
        raise ValueError('unsupported comparator')
    root = Path(root).resolve(strict=True)
    with sqlite3.connect((root/'suite.sqlite').as_uri()+'?mode=ro',uri=True) as db:
        meta = {k:json.loads(v) for k,v in db.execute(
            "SELECT key,value FROM meta WHERE key IN ('config','freeze','protocol','completed')")}
        if meta.get('protocol') != ('generator-addendum-v1' if arm == 'G' else 'tht-six-arm-v1'):
            raise ValueError('comparator protocol differs')
        frozen = meta['freeze']
        if json_hash(frozen['settings']) != frozen['sha256'] or not meta.get('completed'):
            raise ValueError('comparator is not frozen and complete')
        old = meta['config']
        if (old['base_fingerprint'] != task['base_fingerprint'] or
                old['public_package'] != task['public_package'] or old['statistics'] != task['statistics']):
            raise ValueError('comparator source/base/statistics differs')
        if arm == 'G' and frozen['settings']['G']['checkpoint'] != task['fixed_checkpoint']:
            raise ValueError('primary comparator checkpoint differs from H weights')
        path = root/'metrics'/arm/'benchmark/final.json'
        row = db.execute('SELECT path,sha256 FROM artifacts WHERE name=?',(arm+'/benchmark/final/metrics',)).fetchone()
        stage = db.execute('SELECT status FROM stages WHERE name=?',(arm+'/benchmark/final',)).fetchone()
        if not row or row[0] != str(path) or sha256(path) != row[1] or not stage or stage[0] != 'complete':
            raise ValueError('comparator final metrics not sealed/complete')
        ids = {r[0] for r in db.execute("SELECT query_id FROM answers WHERE arm=? AND phase='benchmark' AND candidate='final'",(arm,))}
        if ids != expected:
            raise ValueError('comparator final answer coverage differs')
    score = json.loads(path.read_text())
    rows = score['predictions']
    if len(rows) != 271 or {r['id'] for r in rows} != expected or score['metrics']['n'] != 271:
        raise ValueError('comparator metric coverage differs')
    return score, {'metrics':file_binding(path),'freeze_sha256':frozen['sha256'],'run':str(root)}


def export(store, task):
    # Check before even opening label-free Benchmark manifests or old journals.
    if not store.get('freeze') or not store.get('completed'):
        raise ValueError('comparison export forbidden before H frozen final evaluation completes')
    frozen = store.verify_freeze()
    queries = load_jsonl(Path(task['public_package'])/'benchmark/queries.jsonl')
    expected = {q['id'] for q in queries}
    if len(queries) != 271 or not store.completed_benchmark(expected):
        raise ValueError('H final coverage is incomplete')
    completed = store.get('completed')
    path = Path(completed['metrics'])
    if sha256(path) != completed['sha256']:
        raise ValueError('H final metric snapshot changed')
    scored = json.loads(path.read_text())
    if len(scored['predictions']) != 271 or {r['id'] for r in scored['predictions']} != expected:
        raise ValueError('H scored coverage differs')
    predictions = {'H':{r['id']:r for r in scored['predictions']}}
    comparators = {}
    for arm in ('G','E','F'):
        old,comparators[arm] = read_comparator(task['comparison_runs'][arm],arm,task,expected)
        predictions[arm] = {r['id']:r for r in old['predictions']}
        if any(predictions[arm][q]['truth'] != predictions['H'][q]['truth'] for q in expected):
            raise ValueError('paired comparator truth differs')
    identities = {q['id']:identity_from_query(q) for q in queries}
    if len({(i['object_name'],i['scan_id']) for i in identities.values()}) != 255:
        raise ValueError('paired bootstrap requires frozen255 joints')
    comparison = paired_bootstrap(predictions,identities,external_evaluator(Path(task['public_package'])/'evaluator'),
                                  task['statistics'],pairs=(('H','G'),('H','E'),('H','F')))
    comparison.update(primary='H-G',secondary=['H-E','H-F'],provenance=comparators)
    usage = [json.loads(r['payload']) for r in store.db.execute('SELECT payload FROM usage')]
    attempts = [dict(r) for r in store.db.execute('SELECT * FROM attempts ORDER BY id')]
    events = [dict(r) for r in store.db.execute('SELECT * FROM events ORDER BY seq')]
    commands = []
    for attempt in attempts:
        if not attempt['job_id']:
            continue
        for kind in ('jobs','policy-jobs'):
            launch = store.root/kind/attempt['job_id']/'launch.json'
            if launch.is_file():
                commands.append({'stage':attempt['stage'],'attempt':attempt['id'],'status':attempt['status'],
                                 'launch':json.loads(launch.read_text()),'binding':file_binding(launch)})
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report = {'schema':'icl-addendum-report/1','scope':'engineering-addendum','arm':'H','status':'complete',
              'metrics':scored,'comparison':comparison,'freeze':frozen,
              'preflight':{'effective':task,'approval':store.get('approval')},
              'selection':json.loads(store.stage('H/select')['result']),
              'trials':[store.get(f'H/trial/trial-{i:03d}') for i in range(1,13) if store.get(f'H/trial/trial-{i:03d}')],
              'retrieval_cache':store.get('retrieval_cache_origin'),
              'costs':{'usage':usage,'aggregate':aggregate(usage),
                       'stages':{stage:aggregate([u for u in usage if u.get('stage')==stage])
                                 for stage in sorted({u.get('stage','unknown') for u in usage})},
                       'benchmark':aggregate([u for u in usage if u.get('phase')=='benchmark']),
                       'setup':[setup_usage(p,now) for p in task.get('setup_token_exports',[])],
                       'price':None,'controller_usage':'external control-role exports; unknown unless supplied',
                       'scope':'All H native attempts and fresh retrieval setup; policy and inference usage retained. '
                               'GPU time is allocated execution time, not kernel-active time. '
                               'G training is inherited provenance, excluded from incremental H cost; unknown costs remain null.'},
              'execution':{'started_at_unix':store.get('started_at'),'exported_at':now,'attempts':attempts,
                           'stages':[dict(r) for r in store.db.execute('SELECT * FROM stages ORDER BY name')],
                           'events':events,'commands':commands},
              'limitations':{'generator_training_benchmark_exposure':'unknown',
                            'known_source_overlap':task['overlap_disclosure'],
                            'interpretation':'Engineering addendum; paired intervals do not resolve unknown generator training exposure.'}}
    output = store.root/'exports'/('report-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    report['artifact'] = str(output)
    immutable_json(output,report)
    from suite_runtime import json_write
    json_write(store.root/'latest-report.json',file_binding(output))
    return report
