# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

class OperationTests(unittest.TestCase):
    def setUp(self):
        import torch
        from safetensors.torch import save_file
        from wsp03_kernel import TensorStore
        self.t=tempfile.TemporaryDirectory();self.root=Path(self.t.name);base=self.root/'base';base.mkdir()
        save_file({'model.layers.0.mlp.weight':torch.eye(4,dtype=torch.bfloat16)},base/'model.safetensors')
        (base/'config.json').write_text('{"synthetic":true}')
        self.store=TensorStore(base,self.root/'edits');self.ops=importlib.import_module('wsp03_ops').Operations(self.store)
    def tearDown(self):self.t.cleanup()
    def test_closed_form_rank_one_receipt_replay_and_hf_export(self):
        self.ops.call('edit.family',{'id':'memory','operation':'edit.memit','mechanism':'key-value','tensors':['model.layers.0.mlp.weight'],'lineage':['synthetic'],'configurations':[{'alpha':1,'rank':1}]})
        result=self.ops.call('edit.memit',{'name':'model.layers.0.mlp.weight','family':'memory','configuration':0,'keys':[[1],[0],[0],[0]],'delta_values':[[0],[.5],[0],[0]],'covariance':[0,0,0,0],'covariance_source':'synthetic-dev-moments'})
        self.assertTrue(result['ok']);self.assertEqual(result['operation'],'edit.memit')
        manifest=self.ops.call('edit.commit',{'id':'candidate','hypothesis':'synthetic memory','method':'algebraic_closed_form'})['result']
        self.assertEqual(manifest['changed_elements'],1)
        self.assertTrue(self.ops.call('edit.replay',{'candidate':'candidate'})['result']['passed'])
        export=self.ops.call('export.checkpoint',{'candidate':'candidate'})['result']
        from safetensors import safe_open
        with safe_open(Path(export['directory'])/'model.safetensors',framework='pt') as file:
            self.assertEqual(float(file.get_tensor('model.layers.0.mlp.weight')[1,0]),.5)
        self.assertTrue((Path(export['directory'])/'config.json').exists())
    def test_family_rename_cannot_reset_five_configuration_limit(self):
        params={'id':'one','operation':'edit.direction','mechanism':'offset','tensors':['model.layers.0.mlp.weight'],'lineage':['same'],'configurations':[{'alpha':1}]}
        self.ops.call('edit.family',params)
        with self.assertRaises(ValueError):self.ops.call('edit.family',dict(params,id='renamed',mechanism='different prose'))
        with self.assertRaises(ValueError):self.ops.call('edit.family',dict(params,id='six',lineage=['other'],configurations=[{'alpha':i} for i in range(6)]))
    def test_reference_attribution_reports_actual_paired_delta(self):
        out=self.ops.call('reference.attribute',{'records':[{'query':'q','reference':'r','zero_shot_logprob':-3.,'reference_logprob':-1.,'read_ids':['read0','read1']} ]})['result']
        self.assertEqual(out['references']['r']['mean_delta'],2.)

    def test_memit_preservation_covariance_changes_closed_form_update(self):
        self.ops.call('edit.family',{'id':'regularized','operation':'edit.memit','mechanism':'key-value','tensors':['model.layers.0.mlp.weight'],'lineage':['dev-cov'],'configurations':[{'alpha':1,'rank':1}]})
        self.ops.call('edit.memit',{'name':'model.layers.0.mlp.weight','family':'regularized','configuration':0,'keys':[[1],[0],[0],[0]],'delta_values':[[0],[1],[0],[0]],'covariance':[3,3,3,3],'covariance_source':'synthetic-dev-diagonal'})
        self.assertEqual(float(self.store.staged['model.layers.0.mlp.weight'][1,0]),.25)
    def test_failed_candidate_attempt_cannot_reset_on_restart(self):
        from wsp03_kernel import TensorStore
        self.store.max_candidate_attempts=1
        with self.assertRaises(ValueError):self.store.commit_candidate('unsafe/id','invalid fixture')
        restarted=TensorStore(self.store.base,self.store.output,max_candidate_attempts=1)
        with self.assertRaisesRegex(ValueError,'attempt budget'):restarted.commit_candidate('safe','invalid fixture')

    def test_commit_consumes_configuration_and_revert_clears_it(self):
        import torch
        family={'id':'direction','operation':'edit.direction','mechanism':'offset','tensors':['model.layers.0.mlp.weight'],'lineage':['fresh'],'configurations':[{'alpha':1}]}
        self.ops.family(family);self.ops.config({'family':'direction','configuration':0,'name':'model.layers.0.mlp.weight'})
        w=self.store.load_tensor('model.layers.0.mlp.weight');w[0,1]=.25;self.store.save_edit('model.layers.0.mlp.weight',w)
        self.store.commit_candidate('first','fixture')
        w[0,2]=.25;self.store.save_edit('model.layers.0.mlp.weight',w)
        with self.assertRaisesRegex(ValueError,'configuration'):self.store.commit_candidate('stale','must fail')
        self.ops.call('edit.revert',{})
        self.assertIsNone(getattr(self.store,'edit_configuration',None))
    def test_tensor_inverse_marks_algebraic_even_through_alias(self):
        from wsp03_kernel import Kernel
        kernel=Kernel(self.store,operations=self.ops)
        kernel.execute('x=torch.eye(4); inverse=x.inverse; y=inverse()')
        self.assertTrue(self.store.requires_algebraic)

    def test_scientific_family_requires_bound_numeric_evidence_not_renamed_text(self):
        self.ops.policy={'configurations':{'edit.direction':[{'alpha':1}]},'allowed_tensors':['model.layers.0.mlp.weight']}
        p={'id':'first','operation':'edit.direction','mechanism':'offset','tensors':['model.layers.0.mlp.weight'],'lineage':['real-read'],'configurations':[{'alpha':1}]}
        with self.assertRaisesRegex(ValueError,'lineage'):self.ops.family(p)
        self.ops.trusted_lineage={'real-read':['a'*64],'renamed-read':['a'*64],'different-read':['b'*64]}
        self.ops.family(p)
        with self.assertRaises(ValueError):self.ops.family(dict(p,id='renamed',mechanism='new prose',lineage=['renamed-read']))
        self.ops.family(dict(p,id='new-evidence',lineage=['different-read']))

    def test_family_content_identity_ignores_read_numbering_order_and_batch_split(self):
        from wsp03_service import read_evidence_records
        rows=[{'id':'q@0','query_id':'q','references':[],'correct_class':{'mean_logprob':-2}}, {'id':'r@1','query_id':'r','references':['q'],'correct_class':{'mean_logprob':-1}}]
        joined=read_evidence_records({'candidate':'base','raw':rows})
        reordered=read_evidence_records({'candidate':'base','raw':[dict(rows[1],id='r@0'),dict(rows[0],id='q@1')]})
        self.assertEqual(joined,reordered)
        self.assertEqual(joined,read_evidence_records({'candidate':'renamed-equivalent-state','raw':rows}))
        self.ops.policy={'configurations':{'edit.direction':[{'alpha':1}]},'allowed_tensors':['model.layers.0.mlp.weight']}
        self.ops.trusted_lineage={'joined':joined,'part0':read_evidence_records({'candidate':'base','raw':rows[:1]}),'part1':read_evidence_records({'candidate':'base','raw':rows[1:]})}
        family={'id':'one','operation':'edit.direction','mechanism':'text','tensors':['model.layers.0.mlp.weight'],'lineage':['joined'],'configurations':[{'alpha':1}]}
        self.ops.family(family)
        with self.assertRaises(ValueError):self.ops.family(dict(family,id='split',lineage=['part1','part0'],mechanism='changed prose'))
