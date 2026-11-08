# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic CPU fixtures only: no target weights, data or GPU."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import wsp03_fast as fast
import wsp03_direction as direction
import wsp03_light_final as final


class Evaluator:
    CLASSES = ['A', 'B']
    @staticmethod
    def parse(text):
        return text if text in ('A', 'B') else 'INVALID'
    @staticmethod
    def score(rows):
        # This fixture score tests control flow only; production loads the pinned
        # external MCC implementation and never uses this synthetic substitute.
        value = sum(row['truth'] == row['prediction'] for row in rows)/len(rows)
        return {'mcc': value, 'macro_f1': value}


def rows(n=64, valid=64):
    return [{'id': str(i), 'raw_answer': 'A' if i < valid else '?',
             'finish_reason': 'stop', 'inference_output_tokens': 3,
             'request_sha256': str(i), 'rendered_prompt_sha256': str(i)}
            for i in range(n)]


def asset(path, value):
    path.write_text(json.dumps(value))
    return {'path': str(path), 'sha256': final.file_hash(path)}


def final_config(root):
    r3 = asset(root/'r3.json', {'status': 'FROZEN'})
    r4 = asset(root/'r4.json', {'status': 'FROZEN'})
    ref = asset(root/'reference.json', {'helpfulness': {}, 'atlas': []})
    cfg = {'freezes': {'R3': r3, 'R4': r4}, 'reference_state': ref,
           'r4_selection': {'freeze_sha256': r4['sha256'], 'profile': 'P09', 'patch_manifest_sha256': 'original'},
           'r4_c_freeze': asset(root/'control.json', {'status': 'FROZEN', 'profile': 'P01', 'patch_manifest_sha256': 'original'}),
           'editor_containers': {key: [key] for key in ('R3', 'R4', *final.EXTENSIONS)}}
    for arm, key in final.EXTENSIONS.items():
        prereg = asset(root/(key+'-pre.json'), {'frozen_before_diagnostics': True})
        value = {'status': 'FROZEN', 'preregistration_sha256': prereg['sha256']}
        if arm == 'R3-ext2':
            value['parent'] = 'base'
        if arm == 'R4-ext':
            value.update(parent='R4', parent_freeze_sha256=r4['sha256'], profile='P09',
                         reference_state_sha256=ref['sha256'])
        cfg[key] = {'freeze': asset(root/(key+'-freeze.json'), value), 'preregistration': prereg}
    cfg['extension_statistics'] = asset(root/'pairs.json', {
        'R3-ext': [['R3-ext', 'C'], ['R3-ext', 'B'], ['R3-ext', 'R3']],
        'R3-ext2': [['R3-ext2', 'B'], ['R3-ext2', 'C'], ['R3-ext2', 'R3-ext']],
        'R4-ext': [['R4-ext', 'C'], ['R4-ext', 'R4']]})
    return cfg


class Efficient(unittest.TestCase):
    def test_union_counts_overlap_once_and_minimum32(self):
        value = rows(32, 16)
        for item in value[:10]:
            item['finish_reason'] = 'length'
        self.assertTrue(fast.degenerate(value, Evaluator))
        self.assertEqual(fast.summary(value, Evaluator)['degenerate_union_count'], 26)
        self.assertFalse(fast.degenerate(value[:31], Evaluator))
        value[0]['finish_reason'] = 'stop'
        self.assertFalse(fast.degenerate(value, Evaluator))
        for item in value[16:]:
            item['finish_reason'] = 'length'
        self.assertEqual(fast.summary(value, Evaluator)['degenerate_union_count'], 25)

    def test_batch_eos_is_not_truncation_or_padding_tokens(self):
        self.assertEqual(fast.trim_completion([1, 2, 9, 0, 0], [9], 5), ([1, 2, 9], 'stop'))
        self.assertEqual(fast.trim_completion([1, 2, 3], [9], 3), ([1, 2, 3], 'length'))
        self.assertEqual(fast.trim_completion([1, 2, 9], [9], 3), ([1, 2, 9], 'stop'))

    def test_s1_strict_gate_s2_promotion_and_nomination(self):
        truth = [{'id': str(i), 'labels': ['A']} for i in range(512)]
        baseline, candidate = rows(valid=40), rows(valid=52)
        dev = [str(i) for i in range(512)]
        receipt = fast.paired_screen('S1', baseline, candidate, truth[:64], Evaluator,
            development_ids=dev, candidate_id='x', baseline_id='base')
        self.assertTrue(receipt['advance'])
        no_gain = fast.paired_screen('S1', candidate, candidate, truth[:64], Evaluator,
            development_ids=dev, candidate_id='x', baseline_id='base')
        self.assertFalse(no_gain['advance'])
        invalid = fast.paired_screen('S1', rows(valid=20), rows(valid=51), truth[:64], Evaluator,
            development_ids=dev, candidate_id='x', baseline_id='base')
        self.assertFalse(invalid['advance'])
        with self.assertRaises(ValueError):
            fast.paired_screen('S2', rows(512, 300), rows(512, 400), truth, Evaluator,
                development_ids=dev, candidate_id='x', baseline_id='base', s1=no_gain)
        s2 = fast.paired_screen('S2', rows(512, 300), rows(512, 400), truth, Evaluator,
            development_ids=dev, candidate_id='x', baseline_id='base', s1=receipt)
        nominees = fast.nominate([dict(s2, candidate_id=str(i)) for i in range(5)])
        self.assertEqual(len(nominees['nominees']), 3)
        self.assertFalse(nominees['reuse_fast_predictions'])
        bad = rows()
        bad[0]['request_sha256'] = 'changed'
        with self.assertRaises(ValueError):
            fast.paired_screen('S1', baseline, bad, truth[:64], Evaluator,
                development_ids=dev, candidate_id='x', baseline_id='base')

    def test_parity_binds_same_candidate_and_keeps_full_hf(self):
        frozen = rows()
        profile = {'schema': 'wsp03-worker/1', 'batch_size': 1, 'split': 'development',
                   'mode': 'generation', 'max_new_tokens': 4096, 'do_sample': False,
                   'candidate_manifest_sha256': 'patch', 'chat_template_sha256': 'template'}
        fast_profile = dict(profile, base_inventory_sha256='base')
        for valid, passed in [(61, True), (60, False)]:
            result = fast.parity64(rows(valid=valid), frozen, Evaluator,
                development_ids=[str(i) for i in range(64)], fast_profile=fast_profile,
                frozen_profile=profile, base_inventory_sha256='base')
            self.assertEqual(result['passed'], passed)
            self.assertFalse(result['reuse_fast_predictions_for_hf'])
        with self.assertRaises(ValueError):
            fast.parity64(rows(), frozen, Evaluator, development_ids=[str(i) for i in range(64)],
                fast_profile=dict(fast_profile, candidate_manifest_sha256='wrong'),
                frozen_profile=profile, base_inventory_sha256='base')

    def test_atomic_degenerate_pass_retains32_and_refuses_retry(self):
        class FakeBatch:
            calls = 0
            processor = type('Processor', (), {'chat_template': 'fixture'})
            def __init__(self, cfg):
                pass
            def generate(self, batch):
                FakeBatch.calls += 1
                return [dict(rows(1, 0)[0], id=item['id'], rendered_prompt='fixture') for item in batch]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            requests = [{'id': str(i), 'messages': [], 'references': []} for i in range(64)]
            (root/'requests.json').write_text(json.dumps(requests))
            (root/'ids.json').write_text(json.dumps([str(i) for i in range(64)]))
            cfg = {'split': 'development', 'stage': 'S1', 'requests': str(root/'requests.json'),
                   'development_ids': str(root/'ids.json'), 'output': str(root/'pass'),
                   'track_started_at': time.time(), 'candidate_id': 'x'}
            receipt = fast.generation(cfg, backend_factory=FakeBatch, evaluator=Evaluator)
            self.assertEqual(receipt['status'], 'ABANDONED_DEGENERATE')
            self.assertEqual(receipt['n'], 32)
            self.assertEqual(FakeBatch.calls, 1)
            self.assertEqual(len(fast.read_rows(root/'pass/raw_answers.jsonl')), 32)
            with self.assertRaises(FileExistsError):
                fast.generation(cfg, backend_factory=FakeBatch, evaluator=Evaluator)
            cfg.update(output=str(root/'late'), track_started_at=time.time()-21601)
            receipt = fast.generation(cfg, backend_factory=FakeBatch, evaluator=Evaluator)
            self.assertEqual(receipt['status'], 'ABANDONED_WALL')
            self.assertEqual(FakeBatch.calls, 1)

    def test_six_arm_gate_before_protected_reads_and_parent_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = final_config(root)
            stopped = '{"Running":false,"Status":"exited"}'
            with patch.object(final.subprocess, 'check_output', return_value=stopped) as inspect:
                freezes, states = final.freeze_gate(cfg)
                self.assertEqual(len(freezes), 6)
                self.assertEqual(inspect.call_count, 5)
            for key in ('r3_ext', 'r3_ext2', 'r4_ext', 'r4_c_freeze'):
                incomplete = copy.deepcopy(cfg)
                del incomplete[key]
                with patch.object(final, 'read', side_effect=AssertionError('must reject before reading')):
                    with self.assertRaises(ValueError):
                        final.freeze_gate(incomplete)
            with patch.object(final.subprocess, 'check_output', return_value='{"Running":true,"Status":"running"}'):
                with self.assertRaises(RuntimeError):
                    final.freeze_gate(cfg)
            cfg['r4_selection']['profile'] = 'P01'
            with self.assertRaises(ValueError):
                final.freeze_gate(cfg)

    def test_six_outputs_keep_original_results_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = final_config(root)
            calls = []
            def evaluate(cfg, recipe, runtime, patch_dir, manifest, output, arm, profile, split):
                calls.append((arm, profile, patch_dir))
                return {'arm': arm, 'generation_debit': 1}
            patches = {arm: (Path(arm), {}) for arm in final.EXTENSIONS}
            with patch.object(final, 'evaluate', side_effect=evaluate):
                result = final.evaluate_arms(cfg, None, None, Path('original'), {}, root,
                                             extension_patches=patches)
            self.assertEqual(len(result), 12)
            self.assertEqual(len(json.loads((root/'results.json').read_text())), 6)
            for arm in final.EXTENSIONS:
                self.assertEqual(len(json.loads((root/('results-'+arm.lower()+'.json')).read_text())), 2)
            self.assertTrue(all(profile == 'P09' and patch_dir == Path('R4-ext')
                                for arm, profile, patch_dir in calls if arm == 'R4-ext'))

    def test_no_valid_patch_is_terminal_and_never_inferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = final_config(root)
            path = Path(cfg['r3_ext2']['freeze']['path'])
            freeze = json.loads(path.read_text())
            freeze.update(status='NO_VALID_PATCH', reason='No candidate passed S1',
                          counts={'S1': 3, 'S2': 0, 'FULL2917': 0})
            cfg['r3_ext2']['freeze'] = asset(path, freeze)
            with patch.object(final.subprocess, 'check_output', return_value='{"Running":false,"Status":"exited"}'):
                freezes, _ = final.freeze_gate(cfg)
            calls = []
            def evaluate(cfg, recipe, runtime, patch_dir, manifest, output, arm, profile, split):
                calls.append(arm)
                return {'arm': arm, 'generation_debit': 1}
            patches = {arm: (Path(arm), {}) for arm in ('R3-ext', 'R4-ext')}
            with patch.object(final, 'evaluate', side_effect=evaluate):
                result = final.evaluate_arms(cfg, None, None, Path('original'), {}, root,
                    extension_patches=patches, terminal_extensions=freezes)
            self.assertNotIn('R3-ext2', calls)
            self.assertEqual(result['R3-ext2:benchmark']['generation_debit'], 0)
            self.assertFalse(result['R3-ext2:benchmark']['available'])

    def test_no_valid_patch_statistics_explain_skipped_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = final_config(root)
            cfg['evaluator_directory'] = '/unused-fixture'
            scored = {'predictions': [{'id': 'x', 'truth': 'A', 'prediction': 'A'}],
                      'metrics': {'mcc': 1, 'macro_f1': 1, 'balanced_accuracy': 1}}
            cfg['historical_predictions'] = {arm: asset(root/(arm+'.json'), scored) for arm in ('B', 'C')}
            cfg['final_inputs'] = {split+'_gold': asset(root/(split+'.json'), [{'id': 'x', 'labels': ['A']}])
                                   for split in ('holdout', 'benchmark')}
            results = {}
            for split in ('holdout', 'benchmark'):
                for arm in ('R3', 'R4', 'R4-C', *final.EXTENSIONS):
                    (root/'arms'/split/arm).mkdir(parents=True)
                    results[arm+':'+split] = (
                        {'status': 'NO_VALID_PATCH', 'reason': 'No eligible candidate'}
                        if arm in final.EXTENSIONS else {'raw_directory': '/synthetic'})
            recipe = type('RecipeFixture', (), {
                'members': {'holdout': ['x'], 'benchmark': ['x']},
                'rows': {'x': {'object_name': 'part', 'scan_id': 'SCN_1'}}})()
            with patch.object(final, 'external_evaluator', return_value=Evaluator), patch.object(final.Recipe, 'jsonl', return_value=[]), patch.object(final, 'grade', return_value=scored) as grade, patch.object(final, 'statistics', return_value={'original': True}), patch.object(final, 'paired', side_effect=AssertionError('unavailable arms must not bootstrap')):
                result = final.score_all(cfg, recipe, results, root)
            self.assertEqual(result, {'original': True})
            self.assertEqual(grade.call_count, 6)
            result = json.loads((root/'statistics-r4-ext.json').read_text())['splits']['benchmark']
            self.assertEqual(result['status'], 'UNAVAILABLE')
            self.assertEqual(result['pairs'], {})
            self.assertIn('NO_VALID_PATCH', result['skipped_pairs'][0]['reason'])

    def test_dev_hf_wrapper_terminates_only_its_degenerate_child(self):
        import wsp03_dev_hf as hf
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            requests = [{'id': str(i), 'messages': [], 'references': []} for i in range(64)]
            (root/'requests.json').write_text(json.dumps(requests))
            (root/'ids.json').write_text(json.dumps([str(i) for i in range(64)]))
            cfg = {'split': 'explore', 'stage': 'PARITY64', 'requests': str(root/'requests.json'),
                   'development_ids': str(root/'ids.json'), 'output': str(root/'pass'),
                   'model': '/model', 'candidate_id': 'fixture', 'track_started_at': time.time()}
            class Child:
                returncode = None
                terminated = False
                def __init__(self, command, **kwargs):
                    out = root/'pass'/'worker'
                    out.mkdir()
                    (out/'raw_answers.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows(32, 0)))
                def poll(self):
                    return self.returncode
                def terminate(self):
                    Child.terminated = True
                    self.returncode = -15
                def wait(self, **kwargs):
                    return self.returncode
            with patch.object(hf, 'evaluator_for', return_value=Evaluator), patch.object(hf, 'verify_model', return_value={}), patch.object(hf.subprocess, 'Popen', Child):
                receipt = hf.run(cfg)
            self.assertTrue(Child.terminated)
            self.assertEqual(receipt['status'], 'ABANDONED_DEGENERATE')
            self.assertEqual(receipt['n'], 32)

    def test_nomination_preserves_original_tie_order(self):
        s1 = fast.paired_screen('S1', rows(valid=20), rows(valid=60),
            [{'id': str(i), 'labels': ['A']} for i in range(64)], Evaluator,
            development_ids=[str(i) for i in range(512)], candidate_id='x', baseline_id='base')
        value = fast.paired_screen('S2', rows(512, 100), rows(512, 500),
            [{'id': str(i), 'labels': ['A']} for i in range(512)], Evaluator,
            development_ids=[str(i) for i in range(512)], candidate_id='x', baseline_id='base', s1=s1)
        high_f1 = copy.deepcopy(value)
        high_f1.update(candidate_id='z', changed_elements=100)
        high_f1['candidate']['metrics']['macro_f1'] = 1.
        smaller = dict(value, candidate_id='b', changed_elements=5)
        larger = dict(value, candidate_id='a', changed_elements=10)
        self.assertEqual(fast.nominate([larger, smaller, high_f1])['nominees'], ['z', 'b', 'a'])


class Numeric(unittest.TestCase):
    def test_rank_one_and_random_delta_norm_without_gradients(self):
        import torch
        weight = torch.arange(1, 25, dtype=torch.float32).reshape(4, 6).requires_grad_()
        u = torch.tensor([1., -1., 0., 0.])
        u = u/u.norm()
        edited, detail = direction.rank_one_removal(weight, u, k=2, alpha=2)
        expected = weight.detach()-2*torch.outer(u, (u@weight.detach())*detail['mask'])
        torch.testing.assert_close(edited, expected)
        self.assertFalse(edited.requires_grad)
        self.assertIsNone(weight.grad)
        random, control = direction.random_same_norm(weight, u, k=2, alpha=2, seed=42)
        torch.testing.assert_close(control['delta'].norm(), detail['delta'].norm())
        self.assertFalse(random.requires_grad)
        self.assertTrue(torch.equal(detail['mask'].nonzero().flatten(), torch.tensor([0, 1])))

    def test_contrast_orientation_and_parent_composition(self):
        import torch
        from safetensors.torch import save_file
        from wsp03_kernel import TensorStore
        u = direction.unit_direction(torch.tensor([[2., 0.], [4., 0.]]),
                                     torch.tensor([[0., 0.], [0., 0.]]))
        torch.testing.assert_close(u, torch.tensor([1., 0.]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root/'base'
            base.mkdir()
            name = 'model.layers.0.mlp.down_proj.weight'
            other = 'model.layers.1.mlp.down_proj.weight'
            w = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=torch.bfloat16)
            save_file({name: w, other: w.clone()}, base/'model.safetensors')
            store = TensorStore(base, root/'patches')
            parent = w.clone()
            parent[1, 0] += .5
            store.save_edit(other, parent)
            manifest = store.commit_candidate('original', 'synthetic parent')
            cfg = {'track': 'R4-ext', 'parent': 'R4', 'parent_patch': str(root/'patches/original'),
                   'parent_patch_manifest_sha256': fast.digest(manifest), 'profile': 'P09',
                   'parent_freeze_sha256': 'frozen-r4', 'reference_state_sha256': 'fixed-p09',
                   'layers': [name], 'k': 1, 'alpha': 1, 'writer_mode': 'absolute',
                   'control': 'single_layer', 'candidate_id': 'extension', 'hypothesis': 'fixture',
                   'preregistration_sha256': 'frozen-before-diagnostics'}
            edited = direction.edit_candidate(store, cfg, {name: u})
            self.assertEqual({item['name'] for item in edited['tensors']}, {name, other})
            torch.testing.assert_close(store.load_tensor(other, 'extension'), parent)
            torch.testing.assert_close(store.load_tensor(name, 'extension'), torch.tensor([[1., 2., 0.], [4., 5., 6.]], dtype=torch.bfloat16))
            self.assertTrue(store.verify('extension')['passed'])


if __name__ == '__main__':
    unittest.main()
