#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
'Preflight, run/resume, status and trusted export for fixed-checkpoint H.'
import argparse
import fcntl
import json
import sqlite3
from pathlib import Path

from generator_contract import file_binding, immutable_json
from icl_store import ICLStore, PROTOCOL, check_root
from tht_contract import json_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    pre = sub.add_parser('preflight');pre.add_argument('--task',required=True);pre.add_argument('--output',required=True)
    run = sub.add_parser('run');run.add_argument('--preflight',required=True);run.add_argument('--approval',required=True)
    run.add_argument('--resume-origin',choices=['operator','autonomous-recovery'],default='operator')
    for verb in ('status','export'):
        p = sub.add_parser(verb);p.add_argument('--results-dir',required=True)
    args = parser.parse_args()
    if args.command == 'preflight':
        from icl_preflight import validate
        request = json.loads(Path(args.task).read_text())
        request['task_binding'] = file_binding(args.task)
        result = validate(request)
        output = Path(args.output).resolve()
        protected = [Path(p).resolve() for p in result['effective']['protected_roots']]
        if any(output.is_relative_to(p) for p in protected):
            raise ValueError('preflight output cannot enter source or historical results')
        immutable_json(output,result)
        print(json.dumps({'ready':True,'preflight_sha256':result['preflight_sha256'],'output':str(output)}))
        return
    if args.command == 'run':
        preflight = json.loads(Path(args.preflight).read_text());task = preflight['effective']
        approval = json.loads(Path(args.approval).read_text())
        if (not preflight.get('ready') or preflight['preflight_sha256'] != json_hash(task) or
                approval.get('preflight_sha256') != json_hash(task) or not approval.get('approved_by') or
                not approval.get('approved_at')):
            raise ValueError('explicit approval must bind the exact reviewed H preflight')
        from icl_preflight import validate_output_root
        root = Path(task['results_dir'])
        validate_output_root(root,task['protected_roots'])
        check_root(root,resume=True)
        root.mkdir(parents=True,exist_ok=True)
        with (root/'driver.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            from icl_driver import ICLSuite
            suite = ICLSuite(task,approval,resume_origin=args.resume_origin)
            try:
                report = suite.run()
            except BaseException as error:
                failure = {'type':type(error).__name__,'message':str(error)}
                suite.store.event('orchestrator_stopped',failure)
                if suite.store.get('first_blocker') is None:
                    suite.store.seal('first_blocker',failure)
                raise
            print(json.dumps({'status':report['status'],'results_dir':str(root)}))
        return
    root = Path(args.results_dir).absolute()
    check_root(root,resume=True)
    if not (root/'suite.sqlite').is_file():
        raise ValueError('no H journal exists')
    if args.command == 'status':
        with sqlite3.connect((root/'suite.sqlite').as_uri()+'?mode=ro',uri=True) as db:
            db.row_factory = sqlite3.Row
            meta = {r['key']:json.loads(r['value']) for r in db.execute('SELECT * FROM meta')}
            print(json.dumps({'protocol':meta['protocol'],'frozen':bool(meta.get('freeze')),
                  'completed':meta.get('completed'),'first_blocker':meta.get('first_blocker'),
                  'started_at':meta.get('started_at'),
                  'stages':[dict(r) for r in db.execute('SELECT * FROM stages ORDER BY name')],
                  'attempts':[dict(r) for r in db.execute('SELECT * FROM attempts ORDER BY id')],
                  'budgets':[dict(r) for r in db.execute('SELECT * FROM budgets')],
                  'status_source':'durable journal; native job handles retained for backend observation'},indent=2))
        return
    with (root/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        store = ICLStore(root)
        from icl_preflight import verify_resume
        verify_resume(store.get('config'))
        from icl_analysis import export
        report = export(store,store.get('config'))
        print(json.dumps({'status':report['status'],'artifact':report['artifact']}))


if __name__ == '__main__':
    main()
