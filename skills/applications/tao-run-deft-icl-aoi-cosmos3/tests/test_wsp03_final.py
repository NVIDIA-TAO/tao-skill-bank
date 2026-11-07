# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import tempfile,unittest,sys,types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
class FinalGateTests(unittest.TestCase):
 def test_no_final_source_is_opened_before_both_freezes(self):
  from wsp03_final import after_both_freeze
  class L:
   def __init__(self,p):self.p=p
   def status(self):return {'phase':self.p}
   def get(self,key):return 'frozen-choice'
   def remaining_joint_seconds(self):return 100
  called=[]
  with self.assertRaises(RuntimeError):after_both_freeze(L('frozen'),L('explore'),lambda:called.append(True))
  self.assertEqual(called,[])
  self.assertEqual(after_both_freeze(L('frozen'),L('frozen'),lambda:'attached'),'attached')
 def test_reference_registry_is_not_a_weight_edit(self):
  from wsp03_final import register_reference_profiles
  from wsp03_state import Ledger
  with tempfile.TemporaryDirectory() as t:
   caps={'probe':0,'baseline':0,'explore':12,'full':3,'trusted':4,'read':512,'python':500,'candidate':98}
   l=Ledger(Path(t)/'state.db',track='R4',caps=caps);l.start_clock();register_reference_profiles(l,checkpoint_identity='frozen-r3-checkpoint')
   self.assertEqual(l.db.execute('SELECT count(*) FROM candidates').fetchone()[0],12)
   self.assertEqual(l.status()['used'].get('candidate',0),0)
   self.assertEqual(l.db.execute("SELECT json_extract(payload,'$.changed_elements') FROM candidates WHERE id='P00'").fetchone()[0],0)
   l.close()

 def test_expired_joint_clock_blocks_source_callback(self):
  from wsp03_final import after_both_freeze
  class L:
   def status(self):return {'phase':'frozen'}
   def get(self,k):return 'chosen'
   def remaining_joint_seconds(self):return 0
  called=[]
  with self.assertRaises(RuntimeError):after_both_freeze(L(),L(),lambda:called.append(True))
  self.assertEqual(called,[])
