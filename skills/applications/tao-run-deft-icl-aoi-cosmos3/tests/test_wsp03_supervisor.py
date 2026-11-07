# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib,json,tempfile,unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

class SupervisorTests(unittest.TestCase):
 def setUp(self):self.m=importlib.import_module('wsp03_supervisor')
 def test_strict_single_response_usage_and_no_builtins(self):
  events=[{'type':'thread.started','thread_id':'one'},{'type':'turn.started'},{'type':'item.completed','item':{'type':'agent_message','text':'```python\nprint(1)\n```'}},{'type':'turn.completed','usage':{'input_tokens':50,'cached_input_tokens':10,'output_tokens':5}}]
  value=self.m.parse_events(events);self.assertEqual(value['usage'],{'input_tokens':50,'cached_input_tokens':10,'output_tokens':5});self.assertIn('print(1)',value['text'])
  for bad in [events+events[-1:],events[:-1],events[:2]+[{'type':'item.completed','item':{'type':'command_execution','command':'id'}}]+events[2:]]:
   with self.assertRaises(ValueError):self.m.parse_events(bad)
 def test_round_admission_persistent_unknown_and_finalization(self):
  with tempfile.TemporaryDirectory() as t:
   book=self.m.RoundBook(Path(t)/'events.jsonl',soft_limit=100,hard_limit=120)
   book.admit('first',input_bound=40,output_bound=30)
   with self.assertRaises(RuntimeError):book.admit('second',input_bound=1,output_bound=1)
   book.complete('first',{'input_tokens':40,'cached_input_tokens':10,'output_tokens':20},events_sha256='f'*64)
   restarted=self.m.RoundBook(Path(t)/'events.jsonl',soft_limit=100,hard_limit=120)
   self.assertEqual(restarted.status()['used'],60)
   with self.assertRaises(RuntimeError):restarted.admit('second',input_bound=20,output_bound=21)
   restarted.admit('second',input_bound=10,output_bound=30);restarted.interrupt('second','missing terminal usage')
   self.assertFalse(restarted.status()['provider_usage_complete']);self.assertEqual(restarted.status()['used'],60)
   with self.assertRaises(RuntimeError):restarted.admit('third',input_bound=1,output_bound=1)
 def test_ledger_mutation_and_duplicate_completion_failclosed(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'ledger.jsonl';b=self.m.RoundBook(p,soft_limit=100,hard_limit=120);b.admit('x',1,1);b.complete('x',{'input_tokens':1,'cached_input_tokens':0,'output_tokens':1},events_sha256='a'*64)
   with self.assertRaises(RuntimeError):b.complete('x',{'input_tokens':1,'cached_input_tokens':0,'output_tokens':1},events_sha256='a'*64)
   p.write_text(p.read_text().replace('"input_tokens":1','"input_tokens":0'))
   with self.assertRaises(ValueError):self.m.RoundBook(p,soft_limit=100,hard_limit=120)
 def test_block_allowlist_and_fortieth_handoff(self):
  self.assertEqual(self.m.parse_blocks('```python\nx=1\n```')[0],{'name':'python','body':'x=1'})
  with self.assertRaises(ValueError):self.m.parse_blocks('```shell\ncat /secrets\n```')
  with self.assertRaises(ValueError):self.m.parse_blocks('```python\nx=1\n```',require_handoff=True)
  self.assertEqual(self.m.parse_blocks('```handoff\nsummary\n```',require_handoff=True)[0]['name'],'handoff')

class DedicatedIdentityTests(unittest.TestCase):
 def test_reject_supervisor_or_root_as_tool_identity(self):
  import os
  from wsp03_runtime import DedicatedDocker
  for uid in [0,os.getuid()]:
   with self.assertRaises(ValueError):DedicatedDocker('/tmp/jobs','example@sha256:'+'a'*64,uid=uid,gid=1)

class RequestShapeTests(unittest.TestCase):
 def test_hidden_lite_tools_are_rejected(self):
  from wsp03_supervisor import validate_request_shape
  good={'model':'fixture','input':[{'type':'message','content':[]}],'reasoning':{'effort':'xhigh'}}
  self.assertTrue(validate_request_shape(good,'fixture','xhigh')['no_tools'])
  for bad in [dict(good,tools=[{'name':'shell'}]),dict(good,input=[{'type':'additional_tools','tools':[{'name':'apply_patch'}]}])]:
   with self.assertRaises(ValueError):validate_request_shape(bad,'fixture','xhigh')

class TerminalPolicyTests(unittest.TestCase):
 def test_false_bound_charges_usage_and_closes_future_admission(self):
  from wsp03_supervisor import RoundBook
  with tempfile.TemporaryDirectory() as t:
   b=RoundBook(Path(t)/'ledger.jsonl',soft_limit=100,hard_limit=120);b.admit('x',10,10);b.complete('x',{'input_tokens':11,'output_tokens':1,'cached_input_tokens':0},events_sha256='c'*64)
   self.assertEqual(b.status()['used'],12);self.assertTrue(b.status()['closed'])
   with self.assertRaises(RuntimeError):b.admit('y',1,1)
 def test_freeze_cannot_be_followed_by_execution(self):
  from wsp03_supervisor import parse_blocks
  with self.assertRaises(ValueError):parse_blocks('```freeze\n{}\n```\n```python\nx=1\n```')
 def test_code_mode_notice_is_not_clean_response(self):
  from wsp03_supervisor import parse_events,COMPATIBILITY_NOTICE
  e=[{'type':'thread.started'},{'type':'item.completed','item':{'type':'error','message':COMPATIBILITY_NOTICE}},{'type':'turn.started'},{'type':'item.completed','item':{'type':'agent_message','text':'OK'}},{'type':'turn.completed','usage':{'input_tokens':1,'cached_input_tokens':0,'output_tokens':1}}]
  with self.assertRaises(ValueError):parse_events(e)

class EditorLoopTests(unittest.TestCase):
 def test_fortieth_handoff_preserves_global_round_and_deadline(self):
  import contextlib,types,os
  from wsp03_editor import run_editor
  class Ledger:
   def __init__(self):self.values={};self.events=[]
   def status(self):return {'remaining_development_seconds':90,'remaining_wall_seconds':100,'observed_tokens':123,'soft_stopped':False}
   def editing_seconds(self):return 60
   def tx(self):return contextlib.nullcontext()
   def set(self,k,v):self.values[k]=v
   def event(self,k,v):self.events.append((k,v))
  class Book:
   def __init__(self):self.n=39;self.closed=False
   def status(self):return {'rounds':self.n,'used':123,'closed':self.closed}
   def close(self,reason):self.closed=True
  class Supervisor:
   def __init__(self,root):self.book=Book();self.output=root;self.prompts=[]
   def round(self,prompt,**kwargs):
    self.prompts.append(prompt);self.book.n+=1
    return {'text':'```read\n{"ids":["synthetic-dev"]}\n```\n```handoff\ncompact trusted progress\n```' if self.book.n==40 else '```freeze\n{}\n```'}
  with tempfile.TemporaryDirectory() as t:
   supervisor=Supervisor(Path(t));ledger=Ledger()
   service=types.SimpleNamespace(ledger=ledger,kernel=types.SimpleNamespace(runtime=types.SimpleNamespace(uid=os.getuid()+1)),worker=types.SimpleNamespace(runtime=types.SimpleNamespace(uid=os.getuid()+1)),require_launch=lambda:None,dispatch=lambda action,params:{'request_id':'fresh-round40-read-id'})
   result=run_editor(service=service,supervisor=supervisor,immutable_task='FIXED_TASK',context='DEV_ONLY')
   self.assertTrue(result['trusted_finalization_available']);self.assertEqual(supervisor.book.n,41)
   self.assertIn('compact trusted progress',supervisor.prompts[-1]);self.assertNotIn('start_clock',ledger.values);self.assertIn('fresh-round40-read-id',supervisor.prompts[-1]);self.assertIn('fresh-round40-read-id',(Path(t)/'dispatch-receipts.jsonl').read_text())

class JointDeadlineTests(unittest.TestCase):
 def test_joint_reserve_and_clock_survive_restart(self):
  from wsp03_state import Ledger,AdmissionError
  caps={'probe':0,'baseline':0,'explore':1,'full':3,'trusted':4,'read':5,'python':5,'candidate':5}
  with tempfile.TemporaryDirectory() as t:
   clock=lambda:100
   ledger=Ledger(Path(t)/'state.db',track='R4',caps=caps,clock=clock,total_wall=1000,development_wall=900);ledger.start_clock(100);ledger.set_finalist_reserve(10,2)
   ledger.bind_joint_schedule(deadline=200,downstream_seconds=65,evidence='frozen-schedule')
   self.assertEqual(ledger.editing_seconds(),3)
   ledger.close();ledger=Ledger(Path(t)/'state.db',track='R4',caps=caps,clock=clock,total_wall=1000,development_wall=900)
   self.assertEqual(ledger.editing_seconds(),3)
   with self.assertRaises(ValueError):ledger.bind_joint_schedule(deadline=201,downstream_seconds=65,evidence='extend-forbidden')
   ledger.clock=lambda:104
   with self.assertRaises(AdmissionError):ledger.admit('editor','too-late')
   ledger.close()

class DocumentedOutputBudgetTests(unittest.TestCase):
 def test_128k_admission_uses_remaining_budget_without_reset(self):
  from wsp03_supervisor import RoundBook
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'usage.jsonl';b=RoundBook(p);b.admit('prior',4200000,128000);b.complete('prior',{'input_tokens':4200000,'output_tokens':100000,'cached_input_tokens':1000000},events_sha256='d'*64)
   b=RoundBook(p)
   with self.assertRaises(RuntimeError):b.admit('oversized-old',10000,872000)
   b.admit('documented-new',10000,128000)
   self.assertEqual(b.status()['used'],4300000)
   self.assertEqual(b.status()['rounds'],2)

class AdmissionAndCancellationTests(unittest.TestCase):
 def supervisor(self,root):
  from wsp03_supervisor import Supervisor,RoundBook,sha
  value=Supervisor.__new__(Supervisor);value.home=root;value.workdir=root;value.output=root/'out';value.output.mkdir();value.binary=root/'fake-codex'
  value.binary.write_text('#!/usr/bin/env python3\nimport time\ntime.sleep(30)\n');value.binary.chmod(0o700)
  value.binary_hash=sha(value.binary.read_bytes());value.bound_files={};value.config_hash='a'*64;value.model='synthetic';value.effort='xhigh';value.provider='test';value.relay_fixture_upstream=None;value.guard=None;value.overhead=32768;value.output_reservation=128000;value.bound_evidence='fixture';value.book=RoundBook(value.output/'usage.jsonl')
  return value
 def test_preprovider_budget_rejection_is_known_zero_not_unknown(self):
  from unittest.mock import Mock,patch
  from wsp03_state import AdmissionError
  with tempfile.TemporaryDirectory() as tmp:
   sup=self.supervisor(Path(tmp));ledger=Mock();ledger.admit.side_effect=AdmissionError('reserve expired')
   with patch('wsp03_supervisor.subprocess.Popen') as popen:
    with self.assertRaises(AdmissionError):sup.round('no dispatch',ledger=ledger)
    popen.assert_not_called()
   status=sup.book.status();self.assertEqual(status['unresolved'],[]);self.assertTrue(status['provider_usage_complete']);self.assertEqual(status['used'],0)
 def test_resource_guard_cancels_owned_process_and_keeps_usage_unknown(self):
  import time
  with tempfile.TemporaryDirectory() as tmp:
   sup=self.supervisor(Path(tmp));calls=[]
   def guard():
    calls.append(1)
    if len(calls)>1:raise RuntimeError('synthetic storage stop')
   sup.guard=guard;start=time.monotonic()
   with self.assertRaises(ValueError):sup.round('no external request',timeout=10)
   self.assertLess(time.monotonic()-start,3)
   self.assertFalse(sup.book.status()['provider_usage_complete'])
   receipt=json.loads(next(sup.output.glob('*/relay-receipt.json')).read_text())
   self.assertEqual(receipt['upstream_requests_dispatched'],0);self.assertEqual(receipt['guard_failures'],['RuntimeError'])
