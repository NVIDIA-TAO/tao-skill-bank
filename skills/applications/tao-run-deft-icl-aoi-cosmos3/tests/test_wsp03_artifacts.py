# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json,os,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from wsp03_artifacts import child_json
class ChildReadTests(unittest.TestCase):
 def test_valid_bounded_regular_read(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);(p/'reply.json').write_text('{"ok":true}')
   self.assertEqual(child_json(p,'reply.json',owner=os.getuid()),{'ok':True})
   with self.assertRaises(ValueError):child_json(p,'reply.json',owner=os.getuid(),max_bytes=1)
   with self.assertRaises(ValueError):child_json(p,'reply.json',owner=os.getuid()+1)
 def test_host_private_symlink_and_parent_symlink_are_denied(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);(p/'host-private.json').write_text('{"secret":"synthetic"}');(p/'exchange').mkdir()
   (p/'exchange'/'reply.json').symlink_to(p/'host-private.json');(p/'exchange'/'nested').symlink_to(p,target_is_directory=True)
   for relative in ('reply.json','nested/host-private.json'):
    with self.assertRaises(OSError):child_json(p/'exchange',relative,owner=os.getuid())
   with self.assertRaises(ValueError):child_json(p/'exchange','../host-private.json',owner=os.getuid())
