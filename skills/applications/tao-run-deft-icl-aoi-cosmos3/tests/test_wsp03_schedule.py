# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from wsp03_state import Ledger,AdmissionError
class ScheduleTests(unittest.TestCase):
 def test_slow_edited_timing_inherits_without_resetting_joint_deadline(self):
  caps={'probe':0,'baseline':0,'explore':12,'full':3,'trusted':4,'read':512,'python':500,'candidate':98}
  with tempfile.TemporaryDirectory() as tmp:
   now=[100.];ledgers=[]
   try:
    for name in ('R3','R4'):
     l=Ledger(Path(tmp)/(name+'.db'),track=name,caps=caps,clock=lambda:now[0],full_size=100,explore_size=10,total_wall=1000,development_wall=900);ledgers.append(l);l.start_clock();l.set_finalist_reserve(10,1);l.bind_joint_schedule(deadline=600,downstream_seconds=20,evidence='fixture');l.set_schedule_model(queries=100,passes=3,fixed_seconds=10)
    r3,r4=ledgers;r3.observe_generation_timing(query_seconds=[1.,2.,3.],setup_seconds=4,evidence='completed edited explore');now[0]=200;r4.inherit_timing(r3)
    self.assertEqual(r4.get('joint_schedule')['deadline'],600);self.assertEqual(r4.remaining_joint_seconds(),400)
    self.assertGreaterEqual(r4.get('joint_schedule')['downstream_seconds'],322)
    self.assertEqual(r4.get('full_reserve_seconds'),304)
    with self.assertRaises(AdmissionError):r4.admit('editor','late')
   finally:
    for l in ledgers:l.close()

 def test_slow_generation_adds_cpu_and_r4_local_reserves_without_double_charge(self):
  caps={'probe':0,'baseline':0,'explore':12,'full':3,'trusted':4,'read':512,'python':500,'candidate':98}
  with tempfile.TemporaryDirectory() as tmp:
   ledgers=[]
   try:
    for name in ('R3','R4'):
     l=Ledger(Path(tmp)/(name+'.db'),track=name,caps=caps,full_size=100,explore_size=10);ledgers.append(l);l.start_clock();l.set_finalist_reserve(10,1);l.bind_joint_schedule(deadline=l.clock()+1000,downstream_seconds=36,evidence='fixed CPU and GPU');l.set_schedule_model(queries=100,cpu_queries=80,passes=1,fixed_seconds=10,initial_gpu_rate=.1,cpu_seconds_per_query=.2)
    r3,r4=ledgers
    self.assertEqual(r3.get('full_reserve_seconds'),10);self.assertEqual(r4.get('full_reserve_seconds'),30);self.assertEqual(r4.get('explore_reserve_seconds'),3)
    r3.observe_generation_timing(query_seconds=[.6]*10,setup_seconds=0,preparation_seconds=0,evidence='edited ZS slower')
    r4.inherit_timing(r3)
    self.assertAlmostEqual(r4.estimate_seconds(100,1,cpu_queries=80,fixed_seconds=10),86)
    self.assertAlmostEqual(r4.get('full_reserve_seconds'),80)
    r4.observe_generation_timing(query_seconds=[.5]*10,setup_seconds=0,preparation_seconds=3,evidence='measured larger reference preparation')
    self.assertAlmostEqual(r4.estimate_seconds(100,1,cpu_queries=80,fixed_seconds=10),94)
   finally:
    for l in ledgers:l.close()
