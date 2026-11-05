# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from tht_contract import CLASSES, external_evaluator, grade, json_hash, validate_sharegpt
from suite_selection import epoch_decision, select_round, select_icl, deft_input
from suite_store import SuiteStore
from suite_costs import codex_usage, aggregate

def evidence(number,score,kind='epoch',**extra):
    return dict(phase='proxy',status='complete',n=74,expected_n=74,mcc=score,**{kind:number},**extra)

class SelectionTests(unittest.TestCase):
    def test_minimum_patience_ties_and_missing_epoch(self):
        rows=[evidence(i,x,checkpoint=f'cp{i}') for i,x in enumerate([.3,.3,.2,.1,.3],1)]
        self.assertFalse(epoch_decision(rows[:4])['stop'])
        decision=epoch_decision(rows)
        self.assertTrue(decision['stop'])
        self.assertEqual(decision['best']['epoch'],1)
        self.assertEqual(decision['non_improving_epochs'],4)
        with self.assertRaises(ValueError): epoch_decision(rows[1:])
        rows[-1]['phase']='benchmark'
        with self.assertRaises(ValueError): epoch_decision(rows)

    def test_epoch_cap_and_independent_d_lineage(self):
        rows=[evidence(i,i/30,checkpoint=f'F-{i}') for i in range(1,21)]
        best=epoch_decision(rows)['best']
        self.assertEqual(epoch_decision(rows)['reason'],'epoch_limit')
        self.assertEqual(deft_input('base',[],best)['load_checkpoint'],None)
        d=[evidence(1,.7,'round',checkpoint='D-1')]
        self.assertEqual(deft_input('base',d,best)['load_checkpoint'],'D-1')
        self.assertEqual(deft_input('base',d,best)['epochs'],20)
        self.assertEqual(select_round(d+[evidence(2,.7,'round',checkpoint='D-2')])['round'],1)

    def test_icl_tolerance_complexity_and_full_coverage(self):
        rows=[evidence('t1',.8,'trial_id',images_per_query=5,calls_per_query=1,prompt_characters=100),
              evidence('t2',.79,'trial_id',images_per_query=2,calls_per_query=1,prompt_characters=120),
              evidence('t3',.789,'trial_id',images_per_query=1,calls_per_query=1,prompt_characters=80)]
        self.assertEqual(select_icl(rows)['trial_id'],'t2')
        rows[1]['n']=73
        with self.assertRaises(ValueError): select_icl(rows)

class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.s=SuiteStore(self.tmp.name)
        config={'skip_sdg':True}
        self.s.initialize(config,{'preflight_sha256':json_hash(config),'approved_by':'test'})
    def tearDown(self):
        self.s.db.close()
        self.tmp.cleanup()
    def test_no_benchmark_before_freeze_and_no_answer_repair(self):
        answer={'id':'x','raw_answer':'not JSON','completed':True,'finish_reason':'length'}
        with self.assertRaises(ValueError): self.s.record_answer('B','benchmark','final',answer)
        self.s.record_answer('B','proxy','base',answer)
        self.s.record_answer('B','proxy','base',answer)
        with self.assertRaises(ValueError): self.s.record_answer('B','proxy','base',dict(answer,raw_answer='{"labels":["Good"]}'))
        self.assertEqual(len(self.s.answers('B','proxy','base')),1)
    def test_recovery_requires_native_reconciliation_and_is_bounded(self):
        with self.assertRaises(ValueError): self.s.begin('D/train',['F/select'])
        attempt=self.s.begin('F/train')
        self.s.attach_job(attempt,'job1')
        with self.assertRaises(ValueError): self.s.begin('F/train')
        self.s.finish(attempt,'infrastructure_error',{'exit':137})
        retry=self.s.begin('F/train')
        self.s.finish(retry,'infrastructure_error',{'exit':137})
        with self.assertRaises(ValueError): self.s.begin('F/train')
        a=self.s.begin('C/trial1')
        self.s.finish(a,'program_error',{})
        with self.assertRaises(ValueError): self.s.begin('C/trial1')
    def test_budget_survives_resume_and_usage_is_not_double_counted(self):
        self.s.start_budget('C',1000,100)
        self.s.record_usage('call1',{'input_tokens':50,'output_tokens':10,'cached_input_tokens':40},budget='C')
        self.s.record_usage('call1',{'input_tokens':50,'output_tokens':10,'cached_input_tokens':40},budget='C')
        other=SuiteStore(self.tmp.name)
        other.start_budget('C',1000,100)
        self.assertEqual(other.remaining('C')['tokens'],40)
        with self.assertRaises(ValueError): other.start_budget('C',1000,101)
        with self.assertRaises(ValueError): other.record_usage('unknown',{'input_tokens':None,'output_tokens':None},budget='C')
        other.db.close()
    def test_freeze_and_artifact_integrity(self):
        settings={a:{'base_fingerprint':'base','checkpoint':'base'} for a in 'ABCDEF'}
        settings['D']['checkpoint']='d-selected'
        settings['E']['checkpoint']='d-selected'
        for arm in 'ACDEF':
            a=self.s.begin(arm+'/select'); self.s.finish(a,'complete',{})
        p=Path(self.tmp.name)/'policy.json'; p.write_text('{}')
        self.s.artifact('policy',p)
        self.s.freeze(settings)
        self.s.verify_freeze()
        p.write_text('{"changed":true}')
        with self.assertRaises(ValueError): self.s.verify_freeze()

class CostTests(unittest.TestCase):
    def test_cumulative_and_turn_usage(self):
        cumulative=lambda i,o,c:{'type':'event_msg','payload':{'type':'token_count','info':{'total_token_usage':{'input_tokens':i,'output_tokens':o,'cached_input_tokens':c}}}}
        self.assertEqual(codex_usage([cumulative(10,2,3),cumulative(20,4,8),cumulative(20,4,8)])['input_tokens'],20)
        turn={'type':'turn.completed','turn_id':'one','usage':{'input_tokens':20,'output_tokens':4,'cached_input_tokens':8}}
        self.assertEqual(codex_usage([turn,turn,cumulative(20,4,8)])['input_tokens'],20)
        self.assertIsNone(codex_usage([])['input_tokens'])
    def test_shared_calls_parallel_wall_and_unknown_cost(self):
        row={'invocation_id':'A-query','input_tokens':100,'cached_input_tokens':50,'output_tokens':10,'started_at_unix':1,'ended_at_unix':11}
        second={'invocation_id':'other','started_at_unix':5,'ended_at_unix':15}
        result=aggregate([row,row,second,{'invocation_id':'parent','children':['A-query','other']}])
        self.assertEqual(result['input_tokens'],100)
        self.assertEqual(result['active_wall_seconds'],14)
        self.assertEqual(result['invocations'],2)
        self.assertIsNone(result['gpu_seconds'])

class AnnotationTests(unittest.TestCase):
    def test_single_and_paired(self):
        def row(images,answer): return {'images':images,'conversations':[{'from':'human','value':'classify'},{'from':'gpt','value':answer}]}
        validate_sharegpt([row(['query.png'],'{"labels":["Good"]}')],'tht_single')
        validate_sharegpt([row(['query.png','golden.png'],'NG')],'bare_okng')
        for answer in ['Good','```json\n{"labels":["Good"]}\n```','{"labels":["Good","Bright"]}','{"labels":["A"]}']:
            with self.assertRaises(ValueError): validate_sharegpt([row(['query.png'],answer)],'tht_single')
        with self.assertRaises(ValueError): validate_sharegpt([row(['query.png'],'NG')],'bare_okng')

@unittest.skipUnless(os.environ.get('THT_EVALUATOR_DIR'),'frozen external evaluator dependency not configured')
class ExternalParityTests(unittest.TestCase):
    def test_predictions_invalid_and_every_metric(self):
        evaluator=external_evaluator(os.environ['THT_EVALUATOR_DIR'])
        texts=[json.dumps({'labels':[label]}) for label in CLASSES]+[
            '', 'Good','{"labels":["Good","Bright"]}', '{"labels":[]}',
            '{"labels":["A"]}', '```json\n{"labels":["Good"]}\n```',
            '{"labels":["Good"],"why":"x"}', 'prefix {"labels":["Good"]}',
            '{"labels":"Good"}', 'null', '{"labels":[null]}']
        # Missing-class, empty, zero-denominator and all-invalid fixture sets.
        for selected in [texts,[],['{"labels":["Good"]}']*3,['bad']*4]:
            raw=[{'id':str(i),'raw_answer':t} for i,t in enumerate(selected)]
            gold=[{'id':str(i),'labels':['Good' if len(selected)<8 else CLASSES[i%8]]} for i in range(len(selected))]
            direct=[{'truth':g['labels'][0],'prediction':evaluator.parse(r['raw_answer'])} for r,g in zip(raw,gold)]
            result=grade(raw,gold,evaluator)
            self.assertEqual([r['prediction'] for r in result['predictions']],[r['prediction'] for r in direct])
            self.assertEqual(result['metrics'],evaluator.score(direct))
            self.assertEqual(len(result['metrics']['confusion_matrix']),9)

if __name__=='__main__': unittest.main()
