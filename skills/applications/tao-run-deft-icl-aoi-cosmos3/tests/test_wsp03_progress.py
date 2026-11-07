# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
class ProgressTests(unittest.TestCase):
 def test_layer_tie_uses_frozen_numeric_architecture_order(self):
  from wsp03_progress import choose_layers
  layers=['model.layers.'+str(i) for i in range(12)]
  values={name:{'q':0} for name in layers};values[layers[2]]={'q':1};values[layers[10]]={'q':1}
  self.assertEqual(choose_layers(layers,{'q':0},values,['q']),[layers[2],layers[10]])

 def test_cue_reads_cannot_satisfy_original_prompt_phase2(self):
  import tempfile,types
  from wsp03_state import Ledger
  from wsp03_progress import progress
  for baseline_profile,source_profile,wanted in [('B04','P00',False),('P00','P10',False),('P00','P00',True)]:
   with tempfile.TemporaryDirectory() as tmp:
    l=Ledger(Path(tmp)/'db',track='R3',caps={'probe':0,'baseline':0,'read':10,'full':3,'trusted':2,'explore':1,'python':5,'candidate':5},full_size=2,explore_size=1);l.start_clock()
    def add(identity,profile,conditions,rows):
     l.admit('read',identity,{'candidate':'base','profile':profile,'conditions':conditions});l.finish(identity,{'complete':True,'raw':rows})
    add('source',source_profile,[],[{'id':'q','references':['ref']}])
    add('zero',baseline_profile,[{'query_id':'q','references':[],'capture':{}}],[{'correct_class':{'mean_logprob':-2}}])
    layers=['layer2','layer10']
    for layer in layers:
     patch={'module':layer,'kind':'output','positions':['pre_label'],'mode':'replace','source_capture':{'request_id':'source','record_id':'q','key':layer+':output'}}
     add(layer,'P00',[{'query_id':'q','references':[],'capture':{'patches':[patch]}}],[{'correct_class':{'mean_logprob':-1}}])
    service=types.SimpleNamespace(ledger=l,contract={'splits':{'explore':['q']},'diagnostic_contract':{'query_ids':['q'],'layers':layers,'head_modules':{},'head_count':1,'head_dimension':2}},request_factory=types.SimpleNamespace(attribution_candidates=lambda q:[],references=lambda q,p:['ref']))
    self.assertEqual(progress(service)['phase2_layer_sweep_complete'],wanted);l.close()
