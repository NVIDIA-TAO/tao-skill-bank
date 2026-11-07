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
        self.ops.call('edit.family',{'id':'memory','mechanism':'key-value','tensors':['model.layers.0.mlp.weight'],'lineage':['synthetic'],'configurations':[{'alpha':1,'rank':1}]})
        result=self.ops.call('edit.memit',{'name':'model.layers.0.mlp.weight','family':'memory','configuration':0,'keys':[[1],[0],[0],[0]],'values':[[1],[.5],[0],[0]]})
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
        params={'id':'one','mechanism':'offset','tensors':['model.layers.0.mlp.weight'],'lineage':['same'],'configurations':[{'alpha':1}]}
        self.ops.call('edit.family',params)
        with self.assertRaises(ValueError):self.ops.call('edit.family',dict(params,id='renamed'))
        with self.assertRaises(ValueError):self.ops.call('edit.family',dict(params,id='six',lineage=['other'],configurations=[{'alpha':i} for i in range(6)]))
    def test_reference_attribution_reports_actual_paired_delta(self):
        out=self.ops.call('reference.attribute',{'records':[{'query':'q','reference':'r','zero_shot_logprob':-3.,'reference_logprob':-1.,'read_ids':['read0','read1']} ]})['result']
        self.assertEqual(out['references']['r']['mean_delta'],2.)
