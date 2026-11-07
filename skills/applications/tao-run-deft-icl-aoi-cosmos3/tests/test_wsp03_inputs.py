# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys,tempfile,json,types,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from wsp03_inputs import InputGuard
from wsp03_bindings import file_hash
class InputTests(unittest.TestCase):
 def fixture(self,root):
  expected={}
  for name in ('base_model','siglip','evaluator'):
   folder=root/name;folder.mkdir();f=folder/'weights.bin';f.write_bytes(b'fixture');item={'path':str(f),'sha256':file_hash(f),'bytes':f.stat().st_size}
   expected[name]={'files':{f.name:item}} if name!='evaluator' else {f.name:{'path':str(f),'sha256':file_hash(f)}}
  binary=root/'codex';binary.write_bytes(b'not executable');expected['codex_binary']={'path':str(binary),'sha256':file_hash(binary)}
  asset=root/'expected.json';asset.write_text(json.dumps(expected))
  args={'asset':{'path':str(asset),'sha256':file_hash(asset)},'recipe':types.SimpleNamespace(config={'model_directory':str(root/'base_model'),'evaluator_directory':str(root/'evaluator')}),'editor':{'binary':str(binary)},'final_inputs':{'siglip_path':str(root/'siglip')}}
  return args
 def test_expected_bytes_and_later_metadata_are_bound(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);args=self.fixture(root);guard=InputGuard(**args);guard.check();(root/'siglip/weights.bin').write_bytes(b'changed-bytes')
   with self.assertRaises(RuntimeError):guard.check()
   with self.assertRaises(ValueError):InputGuard(**args)
 def test_extra_model_file_and_world_write_are_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);args=self.fixture(root);extra=root/'base_model/unapproved';extra.write_bytes(b'!')
   with self.assertRaises(ValueError):InputGuard(**args)
   extra.unlink();(root/'siglip/weights.bin').chmod(0o666)
   with self.assertRaises(ValueError):InputGuard(**args)
