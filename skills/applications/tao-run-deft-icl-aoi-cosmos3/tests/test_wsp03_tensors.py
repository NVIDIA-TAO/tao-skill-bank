# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tensor/capture fixtures executed only in the pinned framework image."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch
from safetensors.torch import save_file
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


class TensorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.base = self.root / 'base'; self.base.mkdir()
        self.name = 'model.language_model.layers.0.mlp.down_proj.weight'
        values = {self.name: torch.arange(1, 17, dtype=torch.bfloat16).reshape(4, 4),
                  'lm_head.weight': torch.ones((4, 4), dtype=torch.bfloat16),
                  'model.language_model.norm.weight': torch.ones(4, dtype=torch.bfloat16)}
        save_file(values, self.base / 'model.safetensors')
        self.mod = importlib.import_module('wsp03_kernel')
        self.store = self.mod.TensorStore(self.base, self.root / 'edits')

    def test_numeric_read_nonzero_edit_bit_exact_replay(self):
        x = self.store.load_tensor(self.name)
        self.assertEqual(x[0, 0].item(), 1)
        x[0, 0] += 2
        self.store.save_edit(self.name, x)
        m = self.store.commit_candidate('a', 'one-entry fixture')
        self.assertEqual(m['changed_elements'], 1)
        self.assertEqual(self.store.load_tensor(self.name, 'a')[0, 0].item(), 3)
        self.assertEqual(self.store.load_tensor(self.name)[0, 0].item(), 1)
        self.assertTrue(self.store.verify('a')['passed'])

    def test_reject_protected_scaling_nonfinite_shape_and_new_parameter(self):
        for name, val in [('lm_head.weight', torch.zeros(4, 4)),
                          ('model.language_model.norm.weight', torch.zeros(4)),
                          ('new.bias', torch.zeros(4))]:
            with self.assertRaises(ValueError): self.store.save_edit(name, val)
        x = self.store.load_tensor(self.name)
        for val in [x * 2, x.reshape(16), torch.full_like(x, float('nan'))]:
            with self.assertRaises(ValueError): self.store.save_edit(self.name, val)

    def test_whole_scaling_detected_across_slice_operations(self):
        x = self.store.load_tensor(self.name); x[:2] *= 2
        self.store.save_edit(self.name, x)
        x[2:] *= 2
        with self.assertRaises(ValueError): self.store.save_edit(self.name, x)

    def test_kernel_persists_numeric_state_and_blocks_training_or_host_access(self):
        kernel = self.mod.Kernel(self.store)
        a = kernel.execute('x = load_tensor(' + repr(self.name) + ')\ny = x[0,0].item()\nprint(y)')
        self.assertIn('1.0', a['stdout'])
        b = kernel.execute('print(y + 2)')
        self.assertIn('3.0', b['stdout'])
        for code in ['import os', 'open("/etc/passwd")', 'x.backward()', 'x.__class__', 'torch.optim.Adam([])', 'while True: pass']:
            with self.assertRaises(ValueError): kernel.execute(code)

    def test_closed_form_lstsq_allowed_without_autograd(self):
        kernel = self.mod.Kernel(self.store)
        r = kernel.execute('a = torch.eye(2)\nb = torch.ones(2,1)\nsolution = torch.linalg.lstsq(a,b).solution\nprint(solution.sum().item())')
        self.assertIn('2.0', r['stdout'])


class CaptureTests(unittest.TestCase):
    def test_activation_capture_and_diagnostic_patch_removed_after_context(self):
        m = importlib.import_module('wsp03_capture')
        model = torch.nn.Sequential(torch.nn.Linear(3, 3, bias=False), torch.nn.ReLU())
        x = torch.ones(1, 2, 3)
        base = model(x).detach().clone()
        spec = {'points': [{'module': '0', 'kind': 'output', 'positions': [1]}],
                'patches': [{'module': '0', 'positions': [1], 'mode': 'replace', 'values': [[0., 0., 0.]]}]}
        with m.ActivationCapture(model, spec, max_bytes=4096) as cap:
            result = model(x)
        self.assertTrue(torch.equal(result[:, 1], torch.zeros_like(result[:, 1])))
        self.assertTrue(torch.equal(model(x), base))
        self.assertIn('0:output', cap.values)
        self.assertTrue(cap.diagnostic_only)

    def test_attention_reconstruction_causal_mask_and_grouped_heads(self):
        m = importlib.import_module('wsp03_capture')
        q = torch.ones(1, 4, 3, 2); k = torch.ones(1, 2, 3, 2)
        out = m.attention_rows(q, k, None, rows=[0, 2], heads=[0, 3], scaling=2**-.5)
        self.assertEqual(tuple(out.shape), (1, 2, 2, 3))
        self.assertTrue(torch.equal(out[..., 0, :], torch.tensor([[[1., 0., 0.], [1., 0., 0.]]])))
        self.assertTrue(torch.allclose(out[..., 1, :], torch.full((1, 2, 3), 1/3)))

    def test_canonical_label_logprobs_are_teacher_forced_not_generated(self):
        m = importlib.import_module('wsp03_capture')
        logits = torch.zeros(1, 4, 5); ids = torch.tensor([[1, 2, 3, 4]])
        r = m.target_logprobs(logits, ids, [2, 3])
        self.assertAlmostEqual(r['sum_logprob'], -2 * __import__('math').log(5), places=5)
        self.assertEqual(r['target_token_ids'], [3, 4])


if __name__ == '__main__': unittest.main()
