# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from simple_harness import cap_backend, within, identity, verify_dataset_view, sha


class NativeProfileTests(unittest.TestCase):
    def test_dataset_view_rejects_extra_or_inventoried_gold_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'dataset'; root.mkdir()
            manifest = Path(directory) / 'inventory.json'
            for name in ['train.jsonl', 'val.jsonl']:
                (root / name).write_text('[]')
            inventory = {name: sha(root / name) for name in ['train.jsonl', 'val.jsonl']}
            manifest.write_text(json.dumps(inventory))
            self.assertEqual(verify_dataset_view(manifest, root, 'val'), inventory)
            (root / 'benchmark-gold.json').write_text('{}')
            with self.assertRaises(ValueError): verify_dataset_view(manifest, root, 'val')
            inventory['benchmark-gold.json'] = sha(root / 'benchmark-gold.json')
            manifest.write_text(json.dumps(inventory))
            with self.assertRaises(ValueError): verify_dataset_view(manifest, root, 'val')

    def test_native_decoder_receives_capped_generation_and_receipt_is_honest(self):
        observed = []
        model = SimpleNamespace(generate=lambda **kw: observed.append(kw))
        backend = SimpleNamespace(model=model)
        def generate(messages, tools, root, request):
            request.write_text(json.dumps({'generation': {'max_new_tokens': 4096, 'do_sample': False}}))
            backend.model.generate(max_new_tokens=4096, do_sample=False)
            return {'usage': {'completion_tokens': 128}, 'token_limit_reached': False, 'calls': [], 'text': 'INVALID'}
        backend.generate = generate; bounded = cap_backend(backend)
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'request.json'; result = bounded.generate([], [], Path(directory), p)
            receipt = json.loads(p.read_text())
        self.assertEqual(observed, [{'max_new_tokens': 128, 'do_sample': False}])
        self.assertTrue(result['token_limit_reached'])
        self.assertEqual(receipt['generation']['max_new_tokens'], 128)
        self.assertEqual(receipt['native_declared_generation']['max_new_tokens'], 4096)

    def test_missing_identity_and_path_escape_are_blocked(self):
        with self.assertRaises(ValueError): identity({'object_name': 'x', 'scan_id': ''})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'sub').mkdir(); (root / 'outside').write_text('fixture')
            with self.assertRaises(ValueError): within(root / 'sub', '../outside')


if __name__ == '__main__': unittest.main()
