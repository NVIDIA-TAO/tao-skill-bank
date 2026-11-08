# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only integrity regression fixtures; no private assets or model imports.

Run: PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_native_integrity_review.py
The adapter defaults to the sibling maxwell_profile.py; override its path with
ICL_PROFILE_UNDER_TEST. Native interfaces and trust pins use synthetic fixtures.
An attempted backend allocation always raises before any model can be imported.
"""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


PROFILE_PATH = Path(os.environ.get('ICL_PROFILE_UNDER_TEST', Path(__file__).resolve().parents[1] / 'scripts' / 'maxwell_profile.py'))
SPEC = importlib.util.spec_from_file_location('reviewed_maxwell_profile', PROFILE_PATH)
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def inventory(root):
    return {str(path.relative_to(root)): PROFILE.sha(path)
            for path in root.rglob('*') if path.is_file()}


class UnexpectedBackendAllocation(AssertionError):
    pass


class IntegrityReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='icl-review-synthetic-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / 'dataset'
        (self.dataset / 'images').mkdir(parents=True)
        self.native = self.root / 'native'
        self.native.mkdir()
        (self.native / 'synthetic.py').write_text('# Synthetic source, never imported.\n')
        self.selected = []
        for index, label in enumerate(PROFILE.CLASSES):
            for sample in range(4):
                filename = f'images/mining-{index}-{sample}.png'
                (self.dataset / filename).write_bytes(f'synthetic-{index}-{sample}'.encode())
                self.selected.append({'filename': filename, 'object_name': f'mining-{index}-{sample}',
                                      'scan_id': 'mine', 'labels': [label],
                                      'sha256': PROFILE.sha(self.dataset / filename)})
        self.selection = {'source_split': 'train', 'samples': self.selected}
        (self.dataset / 'train.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in self.selected))
        (self.dataset / 'images/query.png').write_bytes(b'synthetic-query')
        self.query = {'filename': 'images/query.png', 'object_name': 'synthetic-query',
                      'scan_id': 'query', 'image_sha256': PROFILE.sha(self.dataset / 'images/query.png')}
        (self.dataset / 'val.jsonl').write_text(json.dumps(self.query) + '\n')
        (self.root / 'queries.jsonl').write_text(json.dumps(self.query) + '\n')
        (self.root / 'atlas.png').write_bytes(b'synthetic-source-atlas')
        put(self.root / 'selection.json', self.selection)
        put(self.root / 'source-inventory.json', inventory(self.native))
        put(self.root / 'dataset-inventory.json', inventory(self.dataset))
        caches = {}
        for name in ('training-cache', 'query-cache'):
            directory = self.root / name
            directory.mkdir()
            put(directory / 'metadata.json', {'synthetic': True})
            (directory / 'features.npz').write_bytes(b'synthetic-features-not-a-real-array')
            manifest = self.root / (name + '-inventory.json')
            put(manifest, inventory(directory))
            caches[name] = {'path': str(directory), 'manifest': str(manifest)}
        model = self.root / 'model'
        model.mkdir()
        put(model / 'config.json', {'synthetic': True})
        put(model / 'tokenizer_config.json', {'synthetic': True})
        (model / 'model.safetensors').write_bytes(b'synthetic-not-model-weights')
        put(self.root / 'model-inventory.json', inventory(model))
        self.config = {
            'schema': 'tao-icl-maxwell-native/1', 'profile': 'maxwell-grid32-cap128',
            'query_split': 'val', 'output': str(self.root / 'output'),
            'native': {'revision': PROFILE.PIN, 'checkout': str(self.native),
                       'source_manifest': str(self.root / 'source-inventory.json'),
                       'prompt_config': 'synthetic'},
            'assets': {
                'dataset_view': str(self.dataset),
                'dataset_manifest': str(self.root / 'dataset-inventory.json'),
                **{key: {'path': str(self.root / name), 'sha256': PROFILE.sha(self.root / name)}
                   for key, name in [('atlas', 'atlas.png'), ('selection', 'selection.json'),
                                     ('queries', 'queries.jsonl')]},
                'training_cache': caches['training-cache'], 'query_cache': caches['query-cache']},
            'model': {'path': str(model), 'inventory': str(self.root / 'model-inventory.json')},
            'protocol': {'examples_per_class': 4, 'seed': 0, 'reasoning': 'off',
                         'label_mode': 'names', 'execution_policy': 'tht_ablation_no_format_repair_v1',
                         'format_retries': 0, 'max_new_tokens': 128}}
        self.config_path = self.root / 'config.json'
        put(self.config_path, self.config)
        self.output = self.root / 'output'
        self.case = self.output / 'cases' / hashlib.sha256(self.query['filename'].encode()).hexdigest()
        # Trust anchors are synthetic only so verify_config can run without
        # copying private files or depending on the external native checkout.
        patches = [
            mock.patch.object(PROFILE, 'PINNED', inventory(self.native)),
            mock.patch.object(PROFILE, 'NATIVE_ATLAS_SHA256', PROFILE.sha(self.root / 'atlas.png')),
            mock.patch.object(PROFILE, 'NATIVE_SELECTION_SHA256', PROFILE.sha(self.root / 'selection.json'))]
        class Mode:
            def __init__(self, arm): pass
            def prepare(self, *args):
                return types.SimpleNamespace(label_mode='names', user='synthetic-prompt', system=None,
                    bank=None, settings={'model': 'synthetic-model', 'split': 'val', 'max_tool_rounds': 0})
            def schemas(self, *args): return []
            def prepare_output(self, output, *args):
                (output / 'reference_atlas.png').write_bytes(b'synthetic-rendered-atlas')
        class NeverBackend:
            def __init__(self, **kwargs):
                raise UnexpectedBackendAllocation('No model import or call is allowed in integrity tests')
        mode_module = types.ModuleType('aoi_harness.analysis.tht_ablation')
        mode_module.Mode = Mode
        runtime_module = types.ModuleType('aoi_harness.analysis.tht_ablation_runtime')
        def never_run(*args, **kwargs):
            raise AssertionError('Native inference must not run in integrity tests')
        runtime_module.run_case = never_run
        backend_module = types.ModuleType('aoi_harness.harness_backends')
        backend_module.CosmosBackend = NeverBackend
        modules = {'aoi_harness': types.ModuleType('aoi_harness'),
                   'aoi_harness.analysis': types.ModuleType('aoi_harness.analysis'),
                   mode_module.__name__: mode_module, runtime_module.__name__: runtime_module,
                   backend_module.__name__: backend_module}
        patches.append(mock.patch.dict(sys.modules, modules))
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def invoke(self, action):
        captured = io.StringIO()
        with mock.patch.object(sys, 'argv', ['maxwell_profile.py', action, '--config', str(self.config_path)]):
            with mock.patch.object(sys, 'path', list(sys.path)), contextlib.redirect_stdout(captured):
                PROFILE.main()
        return json.loads(captured.getvalue().splitlines()[-1])

    def seal_case(self, trace):
        put(self.case / 'trace.json', trace)
        binding = PROFILE.binding_digest(PROFILE.read(self.output / 'prepared.json'))
        row = {k: value for k, value in self.query.items() if k != 'image_sha256'}
        put(self.case / 'adapter-receipt.json', {
            'binding_sha256': binding, 'query_sha256': PROFILE.binding_digest(row),
            'trace_sha256': PROFILE.sha(self.case / 'trace.json'),
            'request_sha256': {turn['request']: PROFILE.sha(self.output / turn['request'])
                               for turn in trace.get('turns', [])}})

    def terminal_case(self, prediction=None):
        if prediction is None:
            prediction = PROFILE.CLASSES[0]
        self.case.mkdir(parents=True, exist_ok=True)
        (self.case / 'query_view.png').write_bytes(b'synthetic-query-view')
        put(self.case / 'request_00.json', {'generation': {'max_new_tokens': 128},
                                          'native_declared_generation': {'max_new_tokens': 4096}})
        trace = {'execution_policy': 'tht_ablation_no_format_repair_v1', 'schema_version': 1,
                 'prediction': prediction, 'status': 'invalid_final' if prediction == 'INVALID' else 'complete',
                 'seconds': 1, 'assets': {
                     'reference_atlas.png': {'sha256': PROFILE.sha(self.output / 'reference_atlas.png')},
                     str((self.case / 'query_view.png').relative_to(self.output)):
                         {'sha256': PROFILE.sha(self.case / 'query_view.png')}},
                 'turns': [{'round': 0, 'final_only': True,
                            'request': str((self.case / 'request_00.json').relative_to(self.output)),
                            'model_response': {'calls': [], 'usage': {'input_tokens': 12, 'completion_tokens': 5}},
                            'tool_results': []}]}
        self.seal_case(trace)
        return trace

    def test_valid_config_and_terminal_case_can_complete_and_resume(self):
        self.invoke('prepare')
        self.terminal_case()
        result = self.invoke('run')
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertEqual(result['completed'], 1)
        self.assertEqual(self.invoke('run'), result)

    def test_terminal_invalid_is_reused_without_generation(self):
        self.invoke('prepare')
        self.terminal_case('INVALID')
        self.assertEqual(self.invoke('run')['invalid'], 1)
        self.assertEqual(self.invoke('run')['invalid'], 1)

    def test_changed_rendered_atlas_is_rejected(self):
        self.invoke('prepare')
        self.terminal_case()
        (self.output / 'reference_atlas.png').write_bytes(b'changed')
        with self.assertRaises(ValueError): self.invoke('run')

    def test_malformed_terminal_trace_is_rejected(self):
        self.invoke('prepare')
        self.case.mkdir(parents=True)
        self.seal_case({'status': 'complete', 'seconds': 0, 'turns': []})
        with self.assertRaises(ValueError): self.invoke('run')

    def test_changed_trace_is_rejected(self):
        self.invoke('prepare')
        trace = self.terminal_case()
        trace['seconds'] = 999
        put(self.case / 'trace.json', trace)
        with self.assertRaises(ValueError): self.invoke('run')

    def test_wrong_query_binding_is_rejected(self):
        self.invoke('prepare')
        self.terminal_case()
        seal = PROFILE.read(self.case / 'adapter-receipt.json')
        seal['query_sha256'] = 'wrong-query'
        put(self.case / 'adapter-receipt.json', seal)
        with self.assertRaises(ValueError): self.invoke('run')

    def test_unrelated_completion_marker_is_rejected(self):
        self.invoke('prepare')
        self.terminal_case()
        self.invoke('run')
        put(self.output / 'complete.json', {'status': 'UNRELATED', 'config_sha256': 'wrong'})
        with self.assertRaises(ValueError): self.invoke('run')

    def test_request_without_trace_blocks_before_backend_allocation(self):
        self.invoke('prepare')
        put(self.case / 'request_00.json', {'synthetic': 'ambiguous-previous-request'})
        with self.assertRaises(ValueError): self.invoke('run')

    def test_existing_empty_case_blocks_before_backend_allocation(self):
        self.invoke('prepare')
        self.case.mkdir(parents=True)
        with self.assertRaises(ValueError): self.invoke('run')

    def test_updated_dataset_inventory_cannot_reuse_preparation(self):
        self.invoke('prepare')
        (self.dataset / 'images/extra.png').write_bytes(b'new-synthetic-image')
        put(self.root / 'dataset-inventory.json', inventory(self.dataset))
        with self.assertRaises(ValueError): self.invoke('run')

    def test_updated_cache_inventories_cannot_reuse_preparation(self):
        self.invoke('prepare')
        for name in ('training-cache', 'query-cache'):
            with self.subTest(cache=name):
                directory = self.root / name
                path = directory / 'features.npz'
                original = path.read_bytes()
                path.write_bytes(b'changed-synthetic-features')
                put(self.root / (name + '-inventory.json'), inventory(directory))
                with self.assertRaises(ValueError): self.invoke('run')
                path.write_bytes(original)
                put(self.root / (name + '-inventory.json'), inventory(directory))

    def test_incomplete_model_inventory_is_rejected(self):
        value = PROFILE.read(self.root / 'model-inventory.json')
        del value['model.safetensors']
        put(self.root / 'model-inventory.json', value)
        with self.assertRaises(ValueError): PROFILE.verify_config(self.config_path)

    def test_required_cache_file_is_enforced(self):
        root = self.root / 'training-cache'
        (root / 'features.npz').unlink()
        put(self.root / 'training-cache-inventory.json', inventory(root))
        with self.assertRaises(ValueError): PROFILE.verify_config(self.config_path)

    def test_atlas_provenance_rejects_wrong_split_class_missing_member_and_bytes(self):
        self.assertEqual(PROFILE.verify_atlas_selection(self.selection, self.dataset), self.selected)
        invalid = []
        value = copy.deepcopy(self.selection); value['source_split'] = 'not-train'; invalid.append(value)
        value = copy.deepcopy(self.selection); value['samples'][0]['labels'] = ['unknown']; invalid.append(value)
        value = copy.deepcopy(self.selection); value['samples'][0]['filename'] = 'images/missing.png'; invalid.append(value)
        value = copy.deepcopy(self.selection); value['samples'][0]['sha256'] = 'wrong'; invalid.append(value)
        for index, value in enumerate(invalid):
            with self.subTest(invalid=index):
                with self.assertRaises(ValueError): PROFILE.verify_atlas_selection(value, self.dataset)

    def test_original_atlas_pair_pin_is_enforced(self):
        (self.root / 'atlas.png').write_bytes(b'other-atlas')
        self.config['assets']['atlas']['sha256'] = PROFILE.sha(self.root / 'atlas.png')
        put(self.config_path, self.config)
        with self.assertRaises(ValueError): PROFILE.verify_config(self.config_path)

    def test_changed_request_receipt_is_rejected_on_completed_resume(self):
        self.invoke('prepare')
        self.terminal_case()
        self.invoke('run')
        put(self.case / 'request_00.json', {'generation': {'max_new_tokens': 9999}})
        with self.assertRaises(ValueError): self.invoke('run')

    def test_missing_request_receipt_is_rejected_on_completed_resume(self):
        self.invoke('prepare')
        self.terminal_case()
        self.invoke('run')
        (self.case / 'request_00.json').unlink()
        with self.assertRaises((ValueError, FileNotFoundError)): self.invoke('run')

    def test_completion_is_bound_to_the_exact_sealed_terminal_case_set(self):
        self.invoke('prepare')
        self.terminal_case()
        self.assertEqual(self.invoke('run')['invalid'], 0)
        # Simulate explicit reconciliation/import of a separately sealed terminal
        # case under the same run identity. The old aggregate must not survive.
        self.terminal_case('INVALID')
        with self.assertRaises(ValueError): self.invoke('run')

    def test_changed_native_settings_receipt_is_rejected(self):
        self.invoke('prepare')
        self.terminal_case()
        self.invoke('run')
        put(self.output / 'native-settings.private.json', {'synthetic': 'changed-settings'})
        with self.assertRaises(ValueError): self.invoke('run')

    def test_missing_native_settings_receipt_is_rejected(self):
        self.invoke('prepare')
        self.terminal_case()
        self.invoke('run')
        (self.output / 'native-settings.private.json').unlink()
        with self.assertRaises((ValueError, FileNotFoundError)): self.invoke('run')

    def test_changed_prepared_settings_cannot_reuse_preparation(self):
        self.invoke('prepare')
        mode = sys.modules['aoi_harness.analysis.tht_ablation'].Mode
        original = mode.prepare
        def changed_settings(instance, *args):
            prepared = original(instance, *args)
            prepared.settings['initial_query_view'] = {'size': [999, 999]}
            return prepared
        with mock.patch.object(mode, 'prepare', changed_settings):
            with self.assertRaises(ValueError): self.invoke('run')


if __name__ == '__main__':
    unittest.main()
