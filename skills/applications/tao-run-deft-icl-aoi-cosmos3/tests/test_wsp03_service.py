# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from wsp03_service import Service
from wsp03_state import Ledger,AdmissionError
from wsp03_worker import digest

class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.ledger=Ledger(self.root/'ledger.db',track='R3',caps={'probe':2,'baseline':5,'explore':88,'full':3,'trusted':2,'read':512,'python':500,'candidate':98},full_size=2,explore_size=1)
        self.contract={'splits':{'development':['d','r'],'explore':['d'],'holdout':['h'],'benchmark':['b']},'joints':{'d':'a','r':'b','h':'c','b':'a'},'benchmark_reference_ids':['d','r','h'],'max_references':4}
        self.calls=[]
        def render(q,refs,mode):
            row={'id':q,'messages':[],'references':refs}
            if mode=='read':row['forced_text']='{"class":"Good"}'
            return row
        def worker(kind,rows,candidate,split,rid):
            self.calls.append((kind,split))
            return {'complete':True,'ids':[r['id'] for r in rows],'requests_sha256':digest(rows)}
        self.service=Service(ledger=self.ledger,contract=self.contract,kernel=None,request_factory=render,worker=worker,grader=lambda result,split:{'complete':True,'n':1,'mcc':.2,'macro_f1':.3,'invalid':0},edit_root=self.root,approval={'scientific_launch_approved':True,'contract_sha256':digest(self.contract)})
    def tearDown(self):self.ledger.close();self.temp.cleanup()
    def test_gate_split_reference_and_trusted_operations(self):
        self.service.approved=False
        with self.assertRaises(AdmissionError):self.service.dispatch('read',{'ids':['d']})
        self.service.approved=True
        for params in ({'ids':['h'],'split':'holdout'},{'ids':['d'],'references':{'d':['h']}},{'ids':['d'],'references':{'d':['d']}}):
            with self.assertRaises((ValueError,AdmissionError)):self.service.dispatch('read',params)
        with self.assertRaises(AdmissionError):self.service.dispatch('freeze',{})
        self.assertEqual(self.calls,[])
    def test_read_exposure_debits_and_restart_cannot_reset(self):
        self.service.dispatch('read',{'ids':['d'],'references':{'d':['r']},'capture':{'patches':[]}})
        self.assertEqual(self.ledger.status()['used']['read'],1)
        self.ledger.close()
        self.ledger=Ledger(self.root/'ledger.db',track='R3',caps={'probe':2,'baseline':5,'explore':88,'full':3,'trusted':2,'read':512,'python':500,'candidate':98},full_size=2,explore_size=1)
        self.assertEqual(self.ledger.status()['used']['read'],1)
    def test_soft_stop_blocks_reads_but_allows_trusted_finalization(self):
        self.ledger.candidate('c',{'valid':True,'changed_elements':1})
        self.ledger.result('c','explore',{'complete':True,'n':1,'mcc':.1,'macro_f1':.2,'invalid':0})
        self.ledger.admit('editor','response');self.ledger.usage('response',4499999,1,0)
        with self.assertRaises(AdmissionError):self.service.dispatch('read',{'ids':['d']})
        self.assertEqual(self.ledger.nominate(),['c'])
        self.ledger.result('c','full',{'complete':True,'n':2,'mcc':.1,'macro_f1':.2,'invalid':0})
        self.assertEqual(self.ledger.freeze(),'c')
        self.service.forward('trusted',{'ids':['h'],'split':'holdout','candidate':'c'},trusted=True)
        with self.assertRaises(AdmissionError):self.service.forward('trusted',{'ids':['h'],'split':'holdout','candidate':'c'},trusted=True)
