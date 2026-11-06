#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preflight/run/status/export for the isolated native generator addendum."""
import argparse
import fcntl
import json
from pathlib import Path
from generator_contract import immutable_json
from generator_store import GeneratorStore
from tht_contract import json_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    pre=sub.add_parser('preflight');pre.add_argument('--task',required=True);pre.add_argument('--output',required=True)
    run=sub.add_parser('run');run.add_argument('--preflight',required=True);run.add_argument('--approval',required=True)
    run.add_argument('--resume-origin',choices=['operator','autonomous-recovery'],default='operator')
    status=sub.add_parser('status');status.add_argument('--results-dir',required=True)
    export=sub.add_parser('export');export.add_argument('--results-dir',required=True);export.add_argument('--comparison-run')
    args=parser.parse_args()
    if args.command=='preflight':
        from generator_preflight import validate
        result=validate(json.loads(Path(args.task).read_text()));immutable_json(args.output,result)
        print(json.dumps({'ready':True,'preflight_sha256':result['preflight_sha256'],'output':args.output}));return
    if args.command=='run':
        preflight=json.loads(Path(args.preflight).read_text());task=preflight['effective']
        if not preflight.get('ready') or preflight['preflight_sha256']!=json_hash(task): raise ValueError('preflight changed')
        approval=json.loads(Path(args.approval).read_text());root=Path(task['results_dir']);root.mkdir(parents=True,exist_ok=True)
        with (root/'driver.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            from generator_driver import GeneratorSuite
            suite=GeneratorSuite(task,approval,resume_origin=args.resume_origin)
            try: report=suite.run()
            except Exception as error:
                suite.store.event('orchestrator_stopped',{'type':type(error).__name__,'message':str(error)})
                from generator_analysis import export
                try:
                    export(suite.store,task,failure={'type':type(error).__name__,'message':str(error)})
                except Exception as export_error:
                    suite.store.event('failure_export_blocked',{'type':type(export_error).__name__,'message':str(export_error)})
                raise
            print(json.dumps({'status':report['status'],'results_dir':str(root)}))
        return
    root=Path(args.results_dir)
    if not (root/'suite.sqlite').is_file(): raise ValueError('no addendum journal exists')
    store=GeneratorStore(root)
    if store.get('protocol')!='generator-addendum-v1': raise ValueError('refuse non-addendum result store')
    if args.command=='status':
        print(json.dumps({'protocol':store.get('protocol'),'frozen':bool(store.get('freeze')),
                          'completed':bool(store.get('completed')),'started_at':store.get('started_at'),
                          'stages':[dict(r) for r in store.db.execute('SELECT name,status FROM stages ORDER BY name')]},indent=2));return
    from generator_analysis import export
    report=export(store,store.get('config'),comparison_run=args.comparison_run)
    print(json.dumps({'status':report['status'],'results_dir':str(root),'comparison':report['comparison']['status']}))

if __name__=='__main__': main()
