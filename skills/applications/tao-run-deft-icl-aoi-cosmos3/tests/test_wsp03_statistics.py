# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from wsp03_statistics import paired,percentile
class Evaluator:
 @staticmethod
 def score(rows):
  mean=sum(r['truth']==r['prediction'] for r in rows)/len(rows)
  return {'mcc':mean,'macro_f1':mean,'balanced_accuracy':mean}
class PairedTests(unittest.TestCase):
 def test_paired_alias_zero_and_explicit_baseline_control_names(self):
  rows=[{'id':'x','truth':'a','prediction':'a'},{'id':'y','truth':'b','prediction':'INVALID'}]
  score=Evaluator.score(rows);arms={name:{'predictions':rows,'metrics':score} for name in ('R3','R4','R4-C','C')};ids={q:{'object_name':'joint','scan_id':'SCN_one'} for q in ('x','y')}
  result=paired(arms,ids,Evaluator,expected_groups=1,aliases={'R4':'R3'})
  for pair in result['pairs'].values():
   for metric in ('mcc','macro_f1','balanced_accuracy'):self.assertEqual(pair[metric],{'estimate':0.,'lower':0.,'upper':0.})
  self.assertEqual(result['pairs']['R4-C__minus__C']['left_arm'],'R4-C');self.assertEqual(result['pairs']['R4__minus__C']['right_arm'],'C')
  self.assertEqual(percentile([0,10],.25),2.5)
 def test_rejects_changed_gold_groups_or_protocol(self):
  rows=[{'id':'x','truth':'a','prediction':'a'}];arms={name:{'predictions':rows,'metrics':Evaluator.score(rows)} for name in ('R3','C')};ids={'x':{'object_name':'o','scan_id':'s'}}
  for kwargs in ({'expected_groups':2},{'expected_groups':1,'seed':99}):
   with self.assertRaises(ValueError):paired(arms,ids,Evaluator,**kwargs)
  arms['C']={'predictions':[{'id':'x','truth':'b','prediction':'b'}],'metrics':Evaluator.score(rows)}
  with self.assertRaises(ValueError):paired(arms,ids,Evaluator,expected_groups=1)
