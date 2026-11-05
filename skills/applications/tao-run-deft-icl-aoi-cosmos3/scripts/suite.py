#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""THT suite entry point. Preflight never starts model or agent jobs."""
import argparse
import fcntl
import json
import sys
import time
from pathlib import Path
from tht_contract import json_hash
from suite_runtime import json_write

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='verb',required=True)
    p=sub.add_parser('preflight');p.add_argument('--task',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('run');p.add_argument('--preflight',required=True);p.add_argument('--approval',required=True)
    p.add_argument('--resume-origin',choices=['operator','autonomous-recovery'],default='operator')
    p=sub.add_parser('status');p.add_argument('--results-dir',required=True)
    p=sub.add_parser('export');p.add_argument('--results-dir',required=True)
    args=parser.parse_args()
    if args.verb=='preflight':
        from suite_preflight import validate
        result=validate(json.loads(Path(args.task).read_text()));json_write(args.output,result)
        print(json.dumps({'ready':result['ready'],'preflight_sha256':result['preflight_sha256'],'path':args.output}));return
    if args.verb in ('status','export'):
        from suite_store import SuiteStore
        root=Path(args.results_dir)
        if not (root/'suite.sqlite').exists(): raise ValueError('suite state does not exist')
        store=SuiteStore(root)
        if args.verb=='status':
            print(json.dumps({'frozen':bool(store.get('freeze')),'completed':store.get('completed'),
                  'stages':[dict(r) for r in store.db.execute('SELECT * FROM stages ORDER BY name')]}));return
        from suite_analysis import export
        export(store,store.get('config'),failure=store.get('first_blocker'));print(root/'report_data.json');return
    preflight=json.loads(Path(args.preflight).read_text());approval=json.loads(Path(args.approval).read_text())
    task=preflight['effective']
    if not preflight.get('ready') or approval.get('preflight_sha256')!=json_hash(task) or not approval.get('approved_by') or not approval.get('approved_at'):
        raise ValueError('explicit approval must identify approver/time and bind the reviewed effective preflight checksum')
    # Revalidate immutable inputs but keep the approved resource snapshot stable.
    from suite_preflight import validate
    current=validate(task)['effective']
    for key in ('code','provenance','base_fingerprint','prompt_sha256'):
        if current[key]!=task[key]: raise ValueError('approved input changed: '+key)
    root=Path(task['results_dir']);root.mkdir(parents=True,exist_ok=True)
    with (root/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from suite_driver import Suite
        resumed=(root/'suite.sqlite').exists()
        suite=Suite(task,approval)
        if resumed and not suite.store.get('completed'):
            detail={'time':time.time(),'operator':args.resume_origin,'reason':'supervisor process restarted','impact':'reattach existing jobs; preserve answers and budgets'}
            suite.store.event('supervisor_resumed',detail)
            if args.resume_origin=='operator':
                history=suite.store.get('interventions',[])+[detail]
                suite.store.db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)',('interventions',json.dumps(history)))
        try: suite.run()
        except BaseException as error:
            failure={'time':time.time(),'type':type(error).__name__,'message':str(error)}
            if suite.store.get('first_blocker') is None: suite.store.seal('first_blocker',failure)
            from suite_analysis import export
            export(suite.store,task,failure=failure)
            raise

if __name__=='__main__': main()
