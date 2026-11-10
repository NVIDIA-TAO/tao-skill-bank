# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only application routing and freeze gates; no model or Docker calls."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ENTRY = Path(__file__).resolve().parents[1] / 'scripts/simple_harness.py'
SPEC = importlib.util.spec_from_file_location('simple_harness_application', ENTRY)
APP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(APP)


class EntryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.config = self.root / 'config.json'
        self.preflight = self.root / 'preflight.json'
        self.approval = self.root / 'approval.json'
        self.output = self.root / 'results'
        self.output.mkdir()
        self.put(self.config, {'profile': APP.PROFILES[0], 'query_split': 'val', 'output': str(self.output)})
        self.calls = []
        self.mock = patch.object(APP, 'adapter', self.adapter)
        self.mock.start()
        self.addCleanup(self.mock.stop)

    def put(self, path, obj):
        path.write_text(json.dumps(obj))

    def adapter(self, action, path):
        self.calls.append(action)
        if action == 'prepare':
            self.put(self.output / 'prepared.json', {
                'query_count': 2, 'tool_round_limit': 10, 'decision_owner': 'CosmosBackend',
                'max_new_tokens': 4096, 'generation_deviation': None})
        else:
            self.put(self.output / 'complete.json', {'status': 'COMPLETE', 'completed': 2})

    def freeze(self):
        APP.preflight(self.config, self.preflight)
        self.put(self.approval, {'preflight_sha256': APP.sha(self.preflight),
                                'approved_by': 'synthetic-test-only', 'approved_at': 'fixture'})

    def test_preflight_never_infers_and_declares_cosmos_decisions(self):
        APP.preflight(self.config, self.preflight)
        record = APP.read(self.preflight)
        self.assertEqual(self.calls, ['prepare'])
        self.assertEqual(record['decision_owner'], 'Cosmos')
        self.assertEqual(record['max_model_turns'], 11)
        self.assertIn('not performed', record['gpu_smoke'])

    def test_frozen_approved_run_delegates_only_to_native_adapter(self):
        self.freeze()
        result = APP.run(self.preflight, self.approval, 'fixture-job')
        self.assertEqual(self.calls, ['prepare', 'run'])
        self.assertEqual(result['completed'], 2)
        self.assertEqual(len(list((self.output / 'application-launches').glob('*.json'))), 1)

    def test_wrong_approval_blocks_before_model(self):
        self.freeze()
        self.put(self.approval, {'preflight_sha256': 'wrong'})
        with self.assertRaises(ValueError): APP.run(self.preflight, self.approval, 'fixture')
        self.assertEqual(self.calls, ['prepare'])

    def test_changed_inputs_block_before_model(self):
        self.freeze()
        self.config.write_text(self.config.read_text() + '\n')
        with self.assertRaises(ValueError): APP.run(self.preflight, self.approval, 'fixture')
        self.assertEqual(self.calls, ['prepare'])

    def test_changed_preparation_blocks_before_model(self):
        self.freeze()
        self.put(self.output / 'prepared.json', {'changed': True})
        with self.assertRaises(ValueError): APP.run(self.preflight, self.approval, 'fixture')
        self.assertEqual(self.calls, ['prepare'])

    def test_no_platform_job_record_blocks_before_model(self):
        self.freeze()
        with self.assertRaises(ValueError): APP.run(self.preflight, self.approval, '')
        self.assertEqual(self.calls, ['prepare'])

    def test_static_grid_is_not_mislabeled_as_agentic_loop(self):
        self.put(self.config, {'profile': 'simple-harness-grid32-cap128'})
        with self.assertRaises(ValueError): APP.preflight(self.config, self.preflight)
        self.assertEqual(self.calls, [])

    def test_preflight_and_resume_receipts_do_not_overwrite_history(self):
        self.freeze()
        APP.preflight(self.config, self.preflight)
        APP.run(self.preflight, self.approval, 'fixture-job-1')
        APP.run(self.preflight, self.approval, 'fixture-job-2')
        self.assertEqual(len(list((self.output / 'application-launches').glob('*.json'))), 2)
        with self.assertRaises(ValueError): APP.seal(self.preflight, {'different': True})


if __name__ == '__main__':
    unittest.main()
