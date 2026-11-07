# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic WSP03 boundary/regression tests; no real model or GPU needed."""
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


class StateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.mod = importlib.import_module('wsp03_state')

    def ledger(self, track='T1', **kw):
        ledger=self.mod.Ledger(self.path / 'state.sqlite', track=track, clock=lambda: 1000, caps={'probe':2,'baseline':5,'explore':88,'full':3,'trusted':2,'read':512,'python':500,'candidate':98}, **kw)
        if ledger.get('started_at') is None:ledger.start_clock(1000)
        return ledger

    def test_soft_stop_survives_reopen_and_keeps_trusted_path(self):
        x = self.ledger()
        x.admit('editor', 'turn1')
        x.usage('turn1', 4490000, 10000, 4000000)
        with self.assertRaises(self.mod.AdmissionError):
            x.admit('editor', 'turn2')
        x.close()
        y = self.ledger()
        self.assertEqual(y.status()['input_tokens'], 4490000)
        self.assertEqual(y.status()['cached_tokens'], 4000000)
        y.nominate()
        y.freeze()
        y.admit('trusted', 'eval1')
        y.finish('eval1', {'complete': True})
        y.close()

    def test_usage_dedup_and_overrun(self):
        x = self.ledger()
        x.admit('editor', 'turn1')
        x.usage('turn1', 4950000, 80000, 4700000)
        x.usage('turn1', 4950000, 80000, 4700000)
        self.assertEqual(x.status()['observed_tokens'], 5030000)
        self.assertTrue(x.status()['budget_overrun'])
        with self.assertRaises(ValueError):
            x.usage('turn1', 1, 1, 0)
        x.close()

    def test_probe_debit_not_reset_or_double_counted(self):
        debit = {'identity': 'probe-v1', 'generation_passes': 2, 'wall_seconds': 20,
                 'gpu_allocated_seconds': 12, 'input_tokens': 0, 'output_tokens': 0}
        x = self.ledger(probe_debit=debit)
        self.assertEqual(x.status()['used']['probe'], 2)
        self.assertEqual(x.status()['remaining_wall_seconds'], 86380)
        x.close()
        x = self.ledger(probe_debit=debit)
        self.assertEqual(x.status()['used']['probe'], 2)
        x.close()
        with self.assertRaises(ValueError):
            self.ledger(probe_debit=dict(debit, wall_seconds=0))

    def test_single_inflight_failed_request_consumes_budget(self):
        x = self.ledger()
        x.admit('explore', 'p1')
        with self.assertRaises(self.mod.AdmissionError):
            x.admit('read', 'p2')
        x.finish('p1', {'complete': False, 'error': 'fixture'})
        self.assertEqual(x.status()['used']['explore'], 1)
        with self.assertRaises(self.mod.AdmissionError):
            x.admit('explore', 'p1')
        x.close()

    def test_full_dev_reserve_blocks_late_explore(self):
        x = self.ledger()
        x.start_clock(1000)
        x.set_finalist_reserve(300)
        x.clock = lambda: 1000 + 79200 - 899
        with self.assertRaises(self.mod.AdmissionError):
            x.admit('explore', 'late')
        x.close()

    def test_shortlist_immutable_max_three_final_rank_uses_full_only(self):
        x = self.ledger()
        for i, score in enumerate([.9, .8, .7, .6]):
            x.candidate(str(i), {'valid': True, 'changed_elements': i + 1})
            x.result(str(i), 'explore', {'n': 512, 'complete': True, 'mcc': score, 'macro_f1': .5, 'invalid': 0})
        self.assertEqual(x.nominate(), ['0', '1', '2'])
        with self.assertRaises(self.mod.AdmissionError):
            x.candidate('4', {'valid': True, 'changed_elements': 1})
        for i, score in enumerate([.1, .8, .2]):
            x.result(str(i), 'full', {'n': 2917, 'complete': True, 'mcc': score, 'macro_f1': .5, 'invalid': 0})
        self.assertEqual(x.freeze(), '1')
        x.close()

    def test_partial_full_dev_cannot_freeze(self):
        x = self.ledger()
        x.candidate('a', {'valid': True, 'changed_elements': 1})
        x.result('a', 'explore', {'n': 512, 'complete': True, 'mcc': .2, 'macro_f1': .1, 'invalid': 0})
        x.nominate()
        with self.assertRaises(ValueError):
            x.result('a', 'full', {'n': 2916, 'complete': True, 'mcc': .9, 'macro_f1': .9, 'invalid': 0})
        with self.assertRaises(self.mod.AdmissionError):x.freeze()
        x.admit('full','failed-finalist',{'candidate':'a'});x.finish('failed-finalist',{'complete':False})
        self.assertIsNone(x.freeze())
        x.close()

    def test_full_pass_must_fit_measured_remaining_wall(self):
        ledger=self.ledger();ledger.set_finalist_reserve(10)
        ledger.candidate('a',{'valid':True,'changed_elements':1})
        ledger.result('a','explore',{'complete':True,'n':512,'mcc':.2,'macro_f1':.2,'invalid':0})
        ledger.nominate();ledger.clock=lambda:1000+79200-1
        with self.assertRaises(self.mod.AdmissionError):ledger.admit('full','too-late',{'candidate':'a'})
        ledger.close()
    def test_unstarted_clock_fails_closed(self):
        ledger=self.mod.Ledger(self.path/'unstarted.sqlite',track='R4',caps={'probe':0,'baseline':0,'explore':12,'full':3,'trusted':4,'read':512,'python':0,'candidate':12})
        with self.assertRaises(self.mod.AdmissionError):ledger.admit('read','unstarted')
        ledger.close()


class ResourceTests(unittest.TestCase):
    def test_lease_conflict_live_owner_and_release(self):
        m = importlib.import_module('wsp03_resources')
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / 'gpu.lock'
            with m.GPULease(p, 'first', query=lambda: []):
                with self.assertRaises(m.LeaseError):
                    with m.GPULease(p, 'second', query=lambda: []):
                        pass
            with m.GPULease(p, 'second', query=lambda: []):
                pass

    def test_foreign_gpu_process_refused(self):
        m = importlib.import_module('wsp03_resources')
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(m.LeaseError):
                with m.GPULease(Path(t) / 'gpu.lock', 'a', query=lambda: [{'pid': 123, 'memory_bytes': 1}]):
                    pass


class DataTests(unittest.TestCase):
    def test_grouped_split_preserves_rare_and_rejects_size_drift(self):
        m = importlib.import_module('wsp03_data')
        classes = ['Good', 'Bright', 'Interrupted', 'Trapped', 'Hidden', 'Bridge', 'Dry Contact', 'Less Solder']
        rows = [{'id': str(i), 'object_name': 'j' + str(i), 'scan_id': 's', 'labels': [classes[i % 8]], 'image_sha256': str(i).zfill(64)} for i in range(80)]
        result, summary = m.partition(rows, [], explore_size=32)
        self.assertEqual(len(result['explore']), 32)
        dev_rare = {r['id'] for r in result['development'] if r['labels'] == ['Less Solder']}
        self.assertLessEqual(dev_rare, {r['id'] for r in result['explore']})
        self.assertFalse({r['id'] for r in result['development']} & {r['id'] for r in result['holdout']})
        with self.assertRaises(ValueError):
            m.partition(rows, [], explore_size=1)

class SourceGuardTests(unittest.TestCase):
    def test_symlink_parent_cannot_create_inside_source(self):
        m = importlib.import_module('wsp03_data')
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            public = root / 'public'
            public.mkdir()
            (public / 'mining').mkdir()
            (root / 'alias').symlink_to(public / 'mining', target_is_directory=True)
            # Guard must execute before attempting source reads.
            with self.assertRaisesRegex(ValueError, 'outside source'):
                m.prepare(public, root / 'exclusions.json', root / 'alias' / 'derived')
            self.assertFalse((public / 'mining' / 'derived').exists())

    def test_source_digest_mismatch_fails_before_output(self):
        m = importlib.import_module('wsp03_data')
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); (root / 'mining').mkdir()
            (root / 'mining/samples.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError, 'source digest'):
                m.prepare(root, root / 'missing', root.parent / 'unused-wsp03-fixture', expected_source_sha256='0' * 64)

if __name__ == '__main__':unittest.main()
