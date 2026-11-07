# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

class WorkerContractTests(unittest.TestCase):
    def test_generation_rejects_hooks_and_target_before_model_load(self):
        m=importlib.import_module('wsp03_worker')
        for extra in ({'capture':{}},{'forced_text':'Good'},{'patches':[]}):
            with self.assertRaises(ValueError): m.validate_requests([dict(id='x',messages=[],**extra)],'generation','explore')
    def test_read_rejects_trusted_split_and_oversize_exposure(self):
        m=importlib.import_module('wsp03_worker')
        r={'id':'x','messages':[],'forced_text':'Good','capture':{}}
        with self.assertRaises(ValueError): m.validate_requests([r],'read','holdout')
        with self.assertRaises(ValueError): m.validate_requests([dict(r,id=str(i)) for i in range(65)],'read','explore')
    def test_uncertain_attempt_cannot_be_repeated(self):
        m=importlib.import_module('wsp03_worker')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'attempts.jsonl').write_text(json.dumps({'id':'x','status':'running'})+'\n')
            with self.assertRaises(RuntimeError): m.recovery(p,[{'id':'x'}])
    def test_completed_invalid_is_retained_and_changed_prompt_refused(self):
        m=importlib.import_module('wsp03_worker')
        r={'id':'x','messages':[]}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'raw_answers.jsonl').write_text(json.dumps({'id':'x','request_sha256':m.digest(r),'raw_answer':'INVALID'})+'\n')
            self.assertEqual(m.recovery(p,[r])['x']['raw_answer'],'INVALID')
            with self.assertRaises(ValueError):m.recovery(p,[dict(r,messages=[1])])
