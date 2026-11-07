# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys,tempfile,time,unittest,threading
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from wsp03_limits import AggregateLimits
class LimitsTests(unittest.TestCase):
 def test_live_registry_discards_hundreds_of_completed_jobs(self):
  with tempfile.TemporaryDirectory() as tmp:
   limits=AggregateLimits(tmp);seen=[]
   for i in range(500):
    limits.register_job(str(i),lambda i=i:seen.append(i));limits.unregister_job(str(i))
   limits.register_job('active',lambda:seen.append('active'))
   limits.cancel_owned()
   self.assertEqual(seen,['active']);self.assertLess(limits.callback_dispatch_seconds,1)
 def test_slow_completed_cleanup_does_not_delay_active_cancel(self):
  with tempfile.TemporaryDirectory() as tmp:
   limits=AggregateLimits(tmp);hit=threading.Event();release=threading.Event()
   limits.register_job('slow',lambda:release.wait(.3));limits.register_job('active',hit.set)
   thread=threading.Thread(target=limits.cancel_owned);thread.start()
   self.assertTrue(hit.wait(.1));release.set();thread.join()
 def test_storage_counts_symlink_itself_and_separates_accounts(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);shared=root/'shared';shared.write_bytes(b'x'*20)
   r3=root/'R3';r4=root/'R4';r3.mkdir();r4.mkdir();(r3/'old').write_bytes(b'a'*30)
   (r4/'native-codex-sandbox').symlink_to('/not/available/codex')
   limits=AggregateLimits(root/'out',accounts={'R3':[r3,shared],'R4':[r4,shared]},storage_bytes=100)
   self.assertEqual(limits.retained_bytes('R3'),50)
   self.assertEqual(limits.retained_bytes('R4'),20+len('/not/available/codex'))
   group=root/'cgroup';group.mkdir();(group/'memory.max').write_text(str(limits.memory_bytes));limits.group=group
   with limits.reserve(30,'write','R3'):
    with self.assertRaises(RuntimeError):
     with limits.reserve(21,'too large','R3'):pass
   with limits.reserve(40,'independent','R4'):pass
   (r3/'over').write_bytes(b'!'*80)
   with self.assertRaises(RuntimeError):limits.check()
   with self.assertRaises(RuntimeError):limits.available('R4')
   self.assertGreater(limits.account_peaks['R3'],limits.storage_bytes)

 def test_native_status_and_cancel_are_serialized_for_same_job(self):
  import os,json
  from unittest.mock import patch
  from wsp03_runtime import DedicatedDocker,Docker
  with tempfile.TemporaryDirectory() as tmp:
   runtime=DedicatedDocker(tmp,'fixture@sha256:'+'a'*64,uid=os.getuid()+1,gid=1);entered=threading.Event();release=threading.Event();calls=[]
   def status(*args):entered.set();release.wait(1);calls.append('status');return {'state':'RUNNING'}
   with patch.object(Docker,'status',side_effect=status),patch.object(Docker,'cancel',side_effect=lambda *_:calls.append('cancel')),patch.object(runtime,'record',return_value=json.dumps({'terminal_state':None})):
    first=threading.Thread(target=runtime.status,args=('job',));second=threading.Thread(target=runtime.cancel,args=('job',));first.start();self.assertTrue(entered.wait(.2));second.start();time.sleep(.02)
    self.assertEqual(calls,[]);release.set();first.join();second.join();self.assertEqual(calls,['status','cancel'])

 def test_final_scan_includes_bytes_written_after_first_violation(self):
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);data=root/'data';data.mkdir();limits=AggregateLimits(root/'out',roots=[data],storage_bytes=10);group=root/'group';group.mkdir();(group/'memory.max').write_text(str(limits.memory_bytes));limits.group=group
   (data/'before').write_bytes(b'a'*20)
   with self.assertRaises(RuntimeError):limits.check()
   (data/'after').write_bytes(b'b'*30)
   with patch('wsp03_limits.subprocess.run'):receipt=limits.close()
   self.assertEqual(receipt['final_nofollow_storage_scan']['values'],{'default':50});self.assertEqual(receipt['account_overshoot_bytes']['default'],40)

 def test_only_current_lease_uuid_files_are_charged_to_both_accounts(self):
  from wsp03_resources import GPULease
  from wsp03_launch import register_lease_storage
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);shared=root/'shared';shared.mkdir();(shared/'old-observations').write_bytes(b'x'*100000)
   limits=AggregateLimits(root/'out',accounts={'R3':[],'R4':[]});group=root/'group';group.mkdir();(group/'memory.max').write_text(str(limits.memory_bytes));limits.group=group
   with GPULease(shared/'gpu.lock','synthetic',query=lambda:[],snapshot=lambda groups:[],poll_seconds=10) as lease:
    paths=register_lease_storage(limits,lease);lease.sample()
   expected=sum(Path(p).stat().st_size for p in paths if Path(p).is_file())
   self.assertEqual(limits.retained_bytes('R3'),expected);self.assertEqual(limits.retained_bytes('R4'),expected);self.assertLess(expected,100000)
   self.assertNotIn(shared,limits.accounts['R3']);self.assertEqual(len(paths),5)
