# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from wsp03_state import Ledger,AdmissionError
from wsp03_service import Service
from wsp03_final import register_reference_profiles,finalize_development,evaluate_final_arms
from wsp03_worker import digest
class Factory:
 def references(self,qid,profile):return [] if profile in ('P00','B04') else ['r']
 def render(self,qid,refs,mode,profile):return {'id':qid,'messages':[],'references':refs}
class Worker:
 def __init__(self):self.calls=[]
 def checkpoint_identity(self,candidate):return 'exact-frozen-r3-bf16'
 def image_identity(self,rows):return digest([(r['id'],r['references']) for r in rows])
 def __call__(self,kind,rows,candidate,split,rid):
  self.calls.append((kind,candidate,split));return {'complete':True,'ids':[r['id'] for r in rows],'requests_sha256':digest(rows),'raw':[{'id':r['id']} for r in rows]}
def grader(result,split):return {'complete':True,'n':len(result['ids']),'mcc':.2,'macro_f1':.3,'invalid':0}
class RoutingTests(unittest.TestCase):
 def make(self,root,track):
  c={'splits':{'development':['d','r'],'explore':['d'],'holdout':['h'],'benchmark':['b']},'max_references':4,'benchmark_reference_ids':['d','r']}
  if track=='R4':c.update(weights_frozen=True,profile_registry=['P%02d'%i for i in range(12)],trusted_ablation_profiles=['P01'],fixed_worker_candidate='edited')
  caps={'probe':0,'baseline':5 if track=='R3' else 0,'explore':12,'full':3,'trusted':2 if track=='R3' else 4,'read':512,'python':500,'candidate':98}
  ledger=Ledger(root/(track+'.db'),track=track,caps=caps,full_size=2,explore_size=1);ledger.start_clock();ledger.set_finalist_reserve(1)
  worker=Worker();s=Service(ledger=ledger,contract=c,kernel=None,request_factory=Factory(),worker=worker,grader=grader,edit_root=root,approval={'scientific_launch_approved':True,'contract_sha256':digest(c)})
  if track=='R3':ledger.candidate('edited',{'valid':True,'changed_elements':1})
  else:register_reference_profiles(ledger,checkpoint_identity='exact-frozen-r3-bf16')
  return s
 def exercise(self,chosen):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);r3=self.make(root,'R3');r4=self.make(root,'R4')
   try:
    r3.forward('explore',{'candidate':'edited','ids':['d']});self.assertEqual(finalize_development(r3),'edited')
    r4.reuse_source=r3;r4.forward('explore',{'candidate':chosen,'profile':chosen,'ids':['d']});self.assertEqual(finalize_development(r4),chosen)
    if chosen=='P00':self.assertEqual(r4.worker.calls,[]);self.assertEqual(r4.ledger.status()['used']['reuse'],2)
    attached=[];results=evaluate_final_arms(r3,r4,lambda *_:attached.append(True),root)
    self.assertEqual(attached,[True]);self.assertEqual(len(results),6)
    self.assertEqual(sum(k=='trusted' for k,_,_ in r3.worker.calls),2)
    self.assertEqual(sum(k=='trusted' for k,_,_ in r4.worker.calls),2)
    if chosen=='P01':self.assertEqual(results['R4-C:benchmark']['alias_of'],'R4:benchmark')
    else:self.assertEqual(results['R4:benchmark']['reused_kind'],'trusted')
    with self.assertRaises((ValueError,AdmissionError)):r4.forward('trusted',{'candidate':chosen,'profile':chosen,'split':'benchmark','ids':['b']},trusted=True)
   finally:r3.ledger.close();r4.ledger.close()
 def test_p00_reuses_exact_r3_at_all_three_stages(self):self.exercise('P00')
 def test_p01_primary_reuses_fixed_c_ablation(self):self.exercise('P01')
 def test_cues_only_baseline_is_not_a_thirteenth_r4_profile(self):
  with tempfile.TemporaryDirectory() as t:
   r3=self.make(Path(t),'R3');r4=self.make(Path(t),'R4')
   try:
    r3.forward('baseline',{'candidate':'base','profile':'B04','split':'explore','ids':['d']},trusted=True)
    with self.assertRaises(AdmissionError):r4.forward('explore',{'candidate':'B04','profile':'B04','ids':['d']})
   finally:r3.ledger.close();r4.ledger.close()
