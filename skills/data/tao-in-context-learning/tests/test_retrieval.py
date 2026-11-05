# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from retrieval import allowed_references,order_mined,validate_policy,messages

class RetrievalTests(unittest.TestCase):
    def test_excludes_identity_bytes_and_normalized_joint(self):
        q={'id':'q','object_name':'joint1','scan_id':'SCN_board1','image_sha256':'bytes1'}
        def row(i,**kw): return dict(id=i,object_name='other',scan_id='board2',image_sha256=i,labels=['Good'],**kw)
        refs=[dict(row('q')),dict(row('same-bytes'),image_sha256='bytes1'),dict(row('same-joint'),object_name='joint1',scan_id='board1'),row('allowed'),row('allowed')]
        self.assertEqual([r['id'] for r in allowed_references(q,refs)],['allowed'])
        with self.assertRaises(ValueError): allowed_references(dict(q,scan_id=''),refs)
    def test_tmm_boundary_and_presentation_order(self):
        p={'toolset':'siglip-retrieval-v1','version':1,'rationale':'synthetic fixture','max_calls':1,'parameters':{'reference_count':2,'ordering':'farthest_first','instruction':''}}
        validate_policy(p,4)
        refs=[dict(id='a',filepath='/a',embedding=[1.,0.],labels=['Good'],view_path='/input/a.png'),dict(id='b',filepath='/b',embedding=[0.8,0.2],labels=['Bright'],view_path='/input/b.png')]
        out=order_mined(p,[1.,0.],[{'filepath':'/a'},{'filepath':'/b'}],refs)
        self.assertEqual([r['id'] for r in out],['b','a'])
        parts=messages(p,'/input/q.png',out,'classify')[0]['content']
        self.assertEqual([c['image'] for c in parts if c['type']=='image'],['/input/b.png','/input/a.png','/input/q.png'])
        with self.assertRaises(ValueError): order_mined(p,[1.,0.],[{'filepath':'/forbidden'}],refs)

if __name__=='__main__': unittest.main()
