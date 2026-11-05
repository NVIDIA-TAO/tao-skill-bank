# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic orchestration/statistics tests; no model-capacity claims."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from tht_contract import CLASSES,external_evaluator,grade,json_hash
from suite_store import SuiteStore
from suite_costs import codex_usage,interval_usage,aggregate
from suite_analysis import export,paired_bootstrap,pass_fail,setup_usage
from suite_data import identity_from_query,validate_identity_grammar
from tht_rcca import route
from suite_driver import Suite
from commit_stage import _apply_success
from tool_evidence import observations,summarize

class Boundaries(unittest.TestCase):
    def test_missing_malformed_setup_usage_stays_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'events.jsonl'
            for contents in (None,'','{broken',json.dumps({'type':'export_metadata','session_id':'s','interval_start':'2026-01-01T00:00:00Z'})):
                if contents is not None: file.write_text(contents)
                result=setup_usage(file,'2026-01-01T00:01:00Z')
                self.assertIsNone(result['usage']['input_tokens'])
            bad=[{'timestamp':'2026-01-01T00:00:01Z','type':'token_count','info':{'total_token_usage':{'input_tokens':10}}}]
            with self.assertRaises(ValueError): interval_usage(bad,'2026-01-01T00:00:00Z','2026-01-01T00:01:00Z','s')
    def test_tool_timing_includes_failed_attempts_and_unknowns(self):
        events=[{'type':'item.started','_observed_at':1,'item':{'id':'a','type':'command_execution'}},
                {'type':'item.completed','_observed_at':3,'item':{'id':'a','type':'command_execution','exit_code':1}},
                {'type':'item.completed','item':{'id':'b','type':'image_view'}}]
        calls=observations(events)
        result=summarize([{'stage':'A/benchmark/0000','invocation_id':'failed','tool_observations':calls}])
        self.assertEqual(result['planned_queries'],271);self.assertEqual(result['call_count'],2)
        self.assertEqual(result['failure'],1);self.assertEqual(result['unknown_outcome'],1)
        self.assertEqual(result['elapsed_tool_seconds'],2);self.assertEqual(result['timing_coverage']['unknown'],1)
    def test_explicit_sdg_skip_preserves_false_accept_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'false_accepts.json';source.write_text('[{"id":"synthetic"}]')
            before=source.read_bytes();phase={};args=SimpleNamespace(stage='anomalygen',skip=True,iter_label='iter1',summary='explicit skip')
            iterations={'baseline':{'false_accepts_json':str(source)}}
            with self.assertRaises(ValueError): _apply_success({},args,root,iterations)
            _apply_success(phase,args,root,iterations,skip_sdg=True)
            self.assertEqual(phase['anomalygen_skip_reason'],'user-requested');self.assertEqual(source.read_bytes(),before)
    def test_physical_gpu_union_and_measurement_populations(self):
        rows=[{'invocation_id':'train','gpu_seconds':10,'gpu_resource':'one','started_at_unix':0,'ended_at_unix':10},
              {'invocation_id':'export','gpu_seconds':3,'gpu_resource':'one','started_at_unix':5,'ended_at_unix':8},
              {'invocation_id':'agent','input_tokens':20,'output_tokens':2,'cached_input_tokens':0}]
        result=aggregate(rows)
        self.assertEqual(result['gpu_seconds'],13);self.assertEqual(result['physical_device_occupied_seconds'],10)
        self.assertEqual(result['coverage']['input_tokens']['total'],1)
    def test_equal_usage_and_explicit_replay(self):
        one={'type':'turn.completed','_source_file':'session1','_source_line':10,'usage':{'input_tokens':10,'output_tokens':2,'cached_input_tokens':0}}
        two={**one,'_source_line':20}
        self.assertEqual(codex_usage([one,two,one])['input_tokens'],20)
    def test_bounded_session_delta_and_reset(self):
        def event(sec,n): return {'timestamp':f'2026-01-01T00:00:{sec:02d}Z','type':'token_count','info':{'total_token_usage':{'input_tokens':n,'output_tokens':n//10,'cached_input_tokens':n//2}}}
        e=[event(0,100),event(5,200),event(5,200),event(9,250),event(20,500)]
        result=interval_usage(e,'2026-01-01T00:00:01Z','2026-01-01T00:00:10Z','one')
        self.assertEqual(result['input_tokens'],150)
        birth=interval_usage(e[1:],'2026-01-01T00:00:01Z','2026-01-01T00:00:10Z','new','2026-01-01T00:00:02Z')
        self.assertEqual(birth['input_tokens'],250);self.assertTrue(birth['zero_baseline_from_session_birth'])
        missing=interval_usage(e[1:],'2026-01-01T00:00:01Z','2026-01-01T00:00:10Z','old','2025-12-31T23:00:00Z')
        self.assertIsNone(missing['input_tokens'])
        with self.assertRaises(ValueError): interval_usage(e[:2]+[event(9,1)],'2026-01-01T00:00:01Z','2026-01-01T00:00:10Z','one')
    def test_identity_projection_and_no_private_rows(self):
        q={'id':'images/prefix__SCN_scan_1_MBO_part_2_A1_abc123.jpg','image_sha256':'x'}
        identity=identity_from_query(q)
        self.assertEqual((identity['scan_id'],identity['object_name']),('scan_1','MBO_part_2'))
        validate_identity_grammar([{**q,'object_name':'MBO_part_2','scan_id':'SCN_scan_1'}],[q])
        with self.assertRaises(ValueError): validate_identity_grammar([{**q,'object_name':'other','scan_id':'scan_1'}],[q])
    def test_full_identity_rcca_and_zero_errors(self):
        rows=[{'id':'a','truth':'Good','prediction':'INVALID'}]
        result=route({'predictions':rows,'metrics':{}},[{'id':'a'}],phase='proxy')
        self.assertEqual(result['mining_targets'],['a'])
        self.assertEqual(result['sdg']['reason'],'user-requested')
        with self.assertRaises(ValueError): route({'predictions':rows,'metrics':{}},[{'id':'b'}],phase='proxy')
        with self.assertRaises(ValueError): route({'predictions':rows,'metrics':{}},[{'id':'a'}],phase='benchmark')
    def test_expired_retry_never_launches(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite=Suite.__new__(Suite);suite.root=Path(tmp);suite.store=SuiteStore(tmp)
            suite.task={'framework_image':'x@sha256:fake','budgets':{'stage_wall_seconds':1}}
            suite.deadline=10**12;suite.budget_scope=None
            attempt=suite.store.begin('B/proxy/base');suite.store.finish(attempt,'infrastructure_error',{})
            suite.store.db.execute('UPDATE attempts SET started=1')
            with patch('suite_driver.Docker.submit') as submit:
                with self.assertRaises(TimeoutError): suite.run_job('B/proxy/base',[])
                submit.assert_not_called()

@unittest.skipUnless(os.environ.get('THT_EVALUATOR_DIR'),'external scorer required')
class ExportAcceptance(unittest.TestCase):
    def test_complete_incomplete_exports_and_cost_separation(self):
        evaluator=external_evaluator(os.environ['THT_EVALUATOR_DIR'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);public=root/'public';(public/'benchmark').mkdir(parents=True)
            (public/'evaluator').symlink_to(Path(os.environ['THT_EVALUATOR_DIR']).resolve(),target_is_directory=True)
            queries=[{'id':f'images/SCN_scan_{i//2}_MBO_component_A{i%2+1}.png','image_sha256':str(i),'image':'unused'} for i in range(271)]
            (public/'benchmark/queries.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in queries))
            task={'public_package':str(public),'training':{},'statistics':{'method':'paired_joint_percentile','grouping':'object_name+normalized_scan_id','seed':42,'replicates':20,'confidence':.95}}
            store=SuiteStore(root/'run');store.initialize(task,{'preflight_sha256':json_hash(task)})
            incomplete=export(store,task,failure={'reason':'synthetic'})
            self.assertFalse(incomplete['frozen']);self.assertEqual(len(incomplete['arms']),6)
            self.assertTrue(all(a['metrics'] is None and a['status']=='incomplete' for a in incomplete['arms']))
            settings={a:{'checkpoint':'base','base_fingerprint':'base'} for a in 'ABCDEF'}
            for a in 'ACDEF':
                attempt=store.begin(a+'/select');store.finish(attempt,'complete',{'checkpoint_ready_at':50} if a=='F' else {})
            store.freeze(settings)
            for arm in 'ABCDEF':
                raw=[{'id':q['id'],'raw_answer':json.dumps({'labels':[CLASSES[i%8]]}),'completed':True,'wall_seconds':i/100} for i,q in enumerate(queries)]
                gold=[{'id':q['id'],'labels':[CLASSES[i%8]]} for i,q in enumerate(queries)]
                for row in raw: store.record_answer(arm,'benchmark','final',row)
                directory=store.root/'metrics'/arm/'benchmark';directory.mkdir(parents=True)
                (directory/'final.json').write_text(json.dumps(grade(raw,gold,evaluator)))
            store.record_usage('train',{'invocation_id':'train','stage':'F/train/0','arm':'F','gpu_seconds':100,'started_at_unix':0,'ended_at_unix':100})
            store.record_usage('infer',{'invocation_id':'infer','stage':'F/benchmark/final','arm':'F','phase':'benchmark','gpu_seconds':10})
            store.record_usage('shared',{'invocation_id':'shared','stage':'retrieval/embed-mining','gpu_seconds':17})
            view=store.root/'views/B/benchmark/final';view.mkdir(parents=True)
            (view/'q-00000.png').write_bytes(b'synthetic-copy-fixture')
            request={'id':queries[0]['id'],'messages':[{'role':'user','content':[{'type':'image','image':'/input/q-00000.png'},{'type':'text','text':'synthetic'}]}]}
            (view/'requests.json').write_text(json.dumps([request]))
            report=export(store,task)
            self.assertTrue(report['frozen']);self.assertTrue(all(a['status']=='complete' and a['completed_queries']==271 for a in report['arms']))
            self.assertEqual(report['results']['inference_costs']['F']['tokens_and_costs']['gpu_seconds'],10)
            self.assertEqual(report['results']['inference_costs']['F']['one_time_costs']['gpu_seconds'],100)
            self.assertEqual(report['results']['f_learning_curve']['selected_checkpoint_training_cost']['costs']['gpu_seconds'],50)
            self.assertEqual(len(report['results']['stage_costs']['shared']),1)
            sample=report['results']['sample_walkthrough']
            self.assertEqual(sample['arm'],'B');self.assertEqual(sample['actual_model_input'],request)
            self.assertEqual((store.root/sample['permitted_assets'][0]['path']).read_bytes(),b'synthetic-copy-fixture')
            for pair in report['results']['paired_differences']['pairs'].values():
                for metric in pair.values(): self.assertEqual(metric,{'estimate':0,'lower':0,'upper':0})

if __name__=='__main__': unittest.main()
