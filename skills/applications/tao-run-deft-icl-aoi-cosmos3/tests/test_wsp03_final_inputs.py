# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json,tempfile,unittest,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
class FinalPreparationTests(unittest.TestCase):
 def test_unfrozen_preparation_does_not_touch_sources(self):
  from wsp03_final_inputs import prepare
  with self.assertRaisesRegex(ValueError,'freeze'):prepare({'public_package':'/not-present'},'/also-not-present')
 def test_label_free_queries_and_only_missing_embeddings(self):
  from PIL import Image
  import pyarrow.parquet as pq
  from wsp03_final_inputs import prepare
  from wsp03_bindings import file_hash
  from wsp03_worker import digest
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);public=root/'public';(public/'mining/images').mkdir(parents=True);(public/'benchmark/images').mkdir(parents=True)
   rows=[]
   for i,label in [(1,'Good'),(2,'Bright')]:
    path=public/('mining/images/'+str(i)+'.png');Image.new('RGB',(8,6),(i,0,0)).save(path)
    rows.append({'id':'mining__SCN_'+str(i)+'_MBO_TEST_A1','image':str(path.relative_to(public)),'image_sha256':file_hash(path),'object_name':'MBO_TEST','scan_id':'SCN_'+str(i),'labels':[label]})
   (public/'mining/samples.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
   image=public/'benchmark/images/3.png';Image.new('RGB',(8,6),(3,0,0)).save(image)
   query={'id':'benchmark__SCN_3_MBO_TEST_A1','image':'benchmark/images/3.png','image_sha256':file_hash(image)};(public/'benchmark/queries.jsonl').write_text(json.dumps(query)+'\n')
   cache=root/'dev.jsonl';cache.write_text(json.dumps({'id':rows[0]['id'],'embedding':[1.,0.]})+'\n')
   plan={'both_freeze_receipts':{'R3':'a','R4':'b'},'public_package':str(public),'mining_sha256':file_hash(public/'mining/samples.jsonl'),'benchmark_queries_sha256':file_hash(public/'benchmark/queries.jsonl'),'mining_count':2,'benchmark_count':1,'dev_embeddings':str(cache),'dev_embeddings_sha256':file_hash(cache),'dev_count':1,'excluded_mining_ids':[],'holdout_count':1,'holdout_ids_sha256':digest([rows[1]['id']])}
   result=prepare(plan,root/'out');self.assertEqual(result['missing_embeddings'],2);self.assertFalse(result['gold_read'])
   prepared=json.loads((root/'out/prepared.json').read_text());self.assertNotIn('labels',prepared['rows'][query['id']])
   native=pq.read_table(root/'out/embedding-input.parquet').to_pylist();self.assertTrue(all('labels' not in r for r in native));self.assertEqual({r['id'] for r in native},{rows[1]['id'],query['id']})
