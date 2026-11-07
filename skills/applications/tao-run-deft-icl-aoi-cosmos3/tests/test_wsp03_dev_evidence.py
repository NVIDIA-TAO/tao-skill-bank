# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys,tempfile,json,types,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from wsp03_state import Ledger
from wsp03_dev_evidence import publish_baselines
class EvidenceTests(unittest.TestCase):
 def test_five_completed_baselines_publish_without_new_forward(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);l=Ledger(root/'ledger',track='R3',caps={'probe':0,'baseline':5,'full':3,'trusted':2,'explore':1,'read':5,'python':5,'candidate':5},full_size=2,explore_size=1);l.start_clock()
   for profile in ('P00','P01','P02','P03','B04'):
    out=root/profile;out.mkdir();(out/'trusted-score.json').write_text(json.dumps({'metrics':{'mcc':0,'macro_f1':0,'confusion':[[0,1]]},'predictions':[{'id':'dev','truth':'Good','prediction':'INVALID'}]}))
    l.admit('baseline',profile,{'profile':profile});l.finish(profile,{'complete':True,'output_directory':str(out),'raw':[{'id':'dev','inference_output_tokens':4096,'finish_reason':'length','raw_answer':'x'*600}]})
   service=types.SimpleNamespace(ledger=l,result_assets={});before=l.status()['used'];result=publish_baselines(service,root/'evidence')
   self.assertEqual(l.status()['used'],before);self.assertEqual(len(service.result_assets),6)
   summary=json.loads(Path(service.result_assets[result['baseline_summary_asset']]['path']).read_text())
   self.assertEqual(set(summary),{'P00','P01','P02','P03','B04'});self.assertEqual(summary['B04']['length_limit_count'],1)
   case=json.loads(Path(service.result_assets[summary['P00']['wrong_case_asset']]['path']).read_text());self.assertEqual(len(case['raw_prefix']),512);self.assertTrue(case['raw_prefix_truncated']);l.close()
