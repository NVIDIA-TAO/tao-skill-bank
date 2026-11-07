# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib,json,tempfile,unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

class RetrievalBindingTests(unittest.TestCase):
    def setUp(self):
        self.mod=importlib.import_module('wsp03_bindings')
        self.rows={'q':{'id':'q','labels':['A'],'component':'q','image_sha256':'q'},'a':{'id':'a','labels':['A'],'component':'a','image_sha256':'a'},'b':{'id':'b','labels':['B'],'component':'b','image_sha256':'b'},'dup':{'id':'dup','labels':['A'],'component':'q','image_sha256':'dup'}}
        self.emb={'q':[1.,0.],'a':[1.,0.],'b':[.8,.6],'dup':[1.,0.]}
    def test_fixed_strategies_exclude_component_and_no_gold_decision(self):
        selector=self.mod.ReferenceSelector(self.rows,self.emb,helpfulness={'a':-2.,'b':2.},atlas=['a','b'])
        self.assertEqual(selector.select('q','P01'),['a'])
        self.assertEqual(selector.select('q','P05'),['b'])
        self.assertEqual(selector.select('q','P00'),[])
        self.rows['q']['labels']=['different-query-gold']
        self.assertEqual(selector.select('q','P01'),['a'])
        with self.assertRaises(ValueError):selector.select('q','P12')
    def test_insufficient_reference_policy_is_explicit(self):
        selector=self.mod.ReferenceSelector(self.rows,self.emb,helpfulness={},atlas=['b'])
        self.assertEqual(selector.select('q','P08'),['b'])
        with self.assertRaises(ValueError):selector.select('q','P03')

class SafeInputNames(unittest.TestCase):
    def test_source_ids_never_become_paths(self):
        from wsp03_bindings import image_name
        a=image_name('images/SCN_fixture_MBO_x_A1.jpg');self.assertNotIn('/',a);self.assertTrue(a.endswith('.png'))
        self.assertNotEqual(a,image_name('../SCN_fixture_MBO_x_A1.jpg'))

class MetadataBoundary(unittest.TestCase):
    def test_metadata_lists_only_development_and_allowlisted_context(self):
        import types
        from wsp03_service import Service
        from wsp03_bindings import file_hash
        from wsp03_state import AdmissionError
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'model-index.json';p.write_text('{"synthetic":true}')
            service=object.__new__(Service);service.approved=True
            service.contract={'splits':{'development':['images/q.jpg'],'explore':['images/q.jpg'],'holdout':['withheld']},'context_assets':{'model-index.json':{'path':str(p),'sha256':file_hash(p)}}}
            service.request_factory=types.SimpleNamespace(rows={'images/q.jpg':{'labels':['A']}})
            self.assertEqual(service.dispatch('data.list',{'split':'explore'})['rows'][0]['id'],'images/q.jpg')
            with self.assertRaises(AdmissionError):service.dispatch('data.list',{'split':'holdout'})
            self.assertIn('synthetic',service.dispatch('context.read',{'name':'model-index.json'})['text'])
            with self.assertRaises(ValueError):service.dispatch('context.read',{'name':'../secret'})

class CaptureReservationTests(unittest.TestCase):
 def test_unique_source_capture_copy_sizes_are_reserved_before_execute(self):
  import os,types,contextlib
  from wsp03_bindings import NativeWorker
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);rid='a'*32;(root/rid).mkdir();(root/rid/'one').write_bytes(b'x'*100);(root/rid/'two').write_bytes(b'y'*200)
   rows=[{'capture':{'patches':[{'source_capture':{'request_id':rid,'file':name}} for name in ['one','two','one']]}}]
   reservations=[]
   class Limits:
    @contextlib.contextmanager
    def reserve(self,amount,label,account):reservations.append(amount);yield
   worker=NativeWorker.__new__(NativeWorker);worker.read_root=root;worker.runtime=types.SimpleNamespace(resources=Limits(),resource_account='R3');worker.recipe=types.SimpleNamespace(request_images=lambda rows:{})
   worker.execute=lambda *args:{'complete':True};worker('read',rows,'base','explore','new')
   self.assertEqual(reservations,[2*1024**3+300])
