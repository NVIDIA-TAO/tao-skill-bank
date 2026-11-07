# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import tempfile,unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from wsp03_io import json_write
class JsonModes(unittest.TestCase):
 def test_private_default_and_explicit_mailbox_group_read(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'private.json';m=Path(t)/'mailbox.json'
   json_write(p,{'value':1});json_write(m,{'value':2},mode=0o640)
   self.assertEqual(p.stat().st_mode&0o777,0o600);self.assertEqual(m.stat().st_mode&0o777,0o640)
   json_write(m,{'value':3},mode=0o640);self.assertEqual(m.stat().st_mode&0o777,0o640)
   with self.assertRaises(ValueError):json_write(m,{},mode=0o666)
