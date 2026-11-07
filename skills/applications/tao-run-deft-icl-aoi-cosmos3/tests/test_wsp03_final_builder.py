# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib,json,os,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from wsp03_final_builder import FinalInputBuilder
from wsp03_bindings import Recipe,file_hash,image_name
class FinalInputSafety(unittest.TestCase):
 def test_source_staging_excludes_gold_and_unreferenced_files(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);public=p/'public';(public/'mining').mkdir(parents=True);(public/'benchmark').mkdir();(public/'images').mkdir()
   image=public/'images/x.png';image.write_bytes(b'fixture-image');row={'id':'images/SCN_fixture_MBO_item_A1.jpg','image':'images/x.png','image_sha256':file_hash(image)}
   (public/'mining/samples.jsonl').write_text(json.dumps(dict(row,labels=['fixture']))+'\n');(public/'benchmark/queries.jsonl').write_text(json.dumps(dict(row,id='images/SCN_fixture_MBO_other_A1.jpg'))+'\n');(public/'benchmark/gold.jsonl').write_text('must never be staged')
   output=p/'out';output.mkdir();builder=FinalInputBuilder({},runtime=None,data_runtime=None,lease=None,output=output)
   cfg={'public_package':str(public),'source_owner':os.getuid(),'prepare':{'mining_sha256':file_hash(public/'mining/samples.jsonl'),'benchmark_queries_sha256':file_hash(public/'benchmark/queries.jsonl')}}
   staged=builder.stage_sources(cfg)
   self.assertEqual(sorted(str(f.relative_to(staged)) for f in staged.rglob('*') if f.is_file()),['benchmark/queries.jsonl','images/x.png','mining/samples.jsonl'])
 def test_actual_shaped_ids_use_contained_image_names(self):
  self.assertEqual(len(image_name('images/SCN_fixture_MBO_item_A1.jpg')),68)
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);(p/'image.png').write_bytes(b'pixels');recipe=object.__new__(Recipe);recipe.images={'images/SCN_fixture_MBO_item_A1.jpg':p/'image.png'};recipe.cached_c_images={}
   path='/input/'+image_name(next(iter(recipe.images)));rows=[{'messages':[{'content':[{'type':'image','image':path}]}]}]
   self.assertEqual(list(recipe.request_images(rows)),[Path(path).name])
   rows[0]['messages'][0]['content'][0]['image']='/input/../host-private'
   with self.assertRaises(ValueError):recipe.request_images(rows)
   (p/'alias').symlink_to(p/'image.png');recipe.images[next(iter(recipe.images))]=p/'alias';rows[0]['messages'][0]['content'][0]['image']=path
   with self.assertRaises(ValueError):recipe.request_images(rows)
