# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU boundary tests for the isolated generator arm; no model or gold data."""
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from tht_contract import CLASSES, json_hash
from suite_store import SuiteStore


def api(name):
    assert (SCRIPTS / (name + '.py')).is_file(), 'generator addendum API is not implemented'
    return importlib.import_module(name)


class AllocationTests(unittest.TestCase):
    def mapping(self):
        return {f'Widget+defect_{i}': c for i, c in enumerate(CLASSES[1:])}

    def test_zero_allocation_is_retained_in_vector(self):
        c = api('generator_contract')
        counts = {key: (1 if i == 0 else 0) for i,key in enumerate(self.mapping())}
        row = {'anomaly_type':next(iter(counts)), 'num_generated_images':1,
               'guidance':3.0, 'num_steps':35, 'crop_ratio':2.0, 'poisson_blend':False}
        c.validate_cases([row], counts)
        with self.assertRaises(ValueError): c.validate_cases([row,row], counts)

    def test_exact_mapping_and_realized_vector(self):
        c = api('generator_contract'); mapping = self.mapping()
        bad = dict(mapping); bad[next(iter(bad))] = 'Good'
        with self.assertRaises(ValueError): c.validate_mapping(bad)
        counts = {key: 4 for key in mapping}
        rows = [{'anomaly_type': key, 'num_generated_images': 1, 'guidance': 3.0,
                 'num_steps': 35, 'crop_ratio': 2.0, 'poisson_blend': False}
                for key in mapping for _ in range(4)]
        c.validate_cases(rows, counts)
        rows[0]['anomaly_type'] = list(mapping)[1]
        with self.assertRaises(ValueError): c.validate_cases(rows, counts)
        rows[0]['anomaly_type'] = list(mapping)[0]; rows[0]['num_steps'] = 34
        with self.assertRaises(ValueError): c.validate_cases(rows, counts)


class StoreTests(unittest.TestCase):
    def test_isolated_protocol_and_one_final_gate(self):
        m = api('generator_store')
        with tempfile.TemporaryDirectory() as tmp:
            task = {'max_deft_rounds': 5, 'base_fingerprint': 'base'}
            store = m.GeneratorStore(Path(tmp) / 'g')
            store.initialize(task, {'preflight_sha256': json_hash(task)})
            self.assertEqual(store.get('protocol'), 'generator-addendum-v1')
            answer = {'id': 'q', 'raw_answer': 'INVALID', 'completed': True}
            with self.assertRaises(ValueError): store.record_answer('G', 'benchmark', 'final', answer)
            for i in range(1, 6): store.seal(f'G/round/{i}', {'round': i})
            a = store.begin('G/select'); store.finish(a, 'complete', {'checkpoint': 'selected'})
            settings = {'G': {'checkpoint': 'selected', 'base_fingerprint': 'base'}}
            store.freeze(settings)
            store.record_answer('G', 'benchmark', 'final', answer)
            store.record_answer('G', 'benchmark', 'final', answer)
            self.assertTrue(store.completed_benchmark(['q']))
            with self.assertRaises(ValueError): store.record_answer('G', 'benchmark', 'again', answer)
            with self.assertRaises(ValueError): store.record_answer('G', 'benchmark', 'final', {**answer, 'raw_answer': 'Good'})
            with self.assertRaises(ValueError): store.record_answer('D', 'proxy', 'base', answer)
            original = SuiteStore(Path(tmp) / 'six'); original.initialize({}, {'preflight_sha256': json_hash({})})
            with self.assertRaises(ValueError): original.record_answer('G', 'proxy', 'base', answer)
            with self.assertRaises(ValueError): original.freeze(settings)

    def test_resume_cannot_change_inputs_or_reset_generation_deadline(self):
        m = api('generator_store')
        with tempfile.TemporaryDirectory() as tmp:
            store = m.GeneratorStore(tmp); task = {'max_deft_rounds': 5}
            store.initialize(task, {'preflight_sha256': json_hash(task)})
            with self.assertRaises(ValueError): store.initialize({'max_deft_rounds': 4}, {'preflight_sha256': json_hash({'max_deft_rounds': 4})})
            a = store.begin('G/generate/1'); store.finish(a, 'infrastructure_error', {})
            store.db.execute('UPDATE attempts SET started=1')
            self.assertEqual(store.stage_deadline('G/generate/1', 7200, 10000), 7201)
            b = store.begin('G/generate/1'); store.finish(b, 'program_error', {})
            with self.assertRaises(ValueError): store.begin('G/generate/1')
            self.assertEqual(store.stage_deadline('G/generate/1', 7200, 10000), 7201)


if __name__ == '__main__':
    unittest.main()
