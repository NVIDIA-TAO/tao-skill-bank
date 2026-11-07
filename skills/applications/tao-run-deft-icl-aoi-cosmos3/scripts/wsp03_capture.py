# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Budgeted teacher-forced numeric capture and dev-only intervention hooks.

Attention probabilities are diagnostic FP32 reconstruction from the same pass's
normalized Q/K with that pass's RoPE/mask, not probabilities exposed by SDPA.
"""
from __future__ import annotations
import math


def target_logprobs(logits, token_ids, positions):
    import torch
    if not positions or any(type(p) is not int or p < 1 or p >= token_ids.shape[1] for p in positions):
        raise ValueError('valid teacher-forced target token positions required')
    ids = token_ids[0, positions]
    selected = logits[0, [p-1 for p in positions]].float()
    values = selected.log_softmax(dim=-1).gather(1, ids[:, None]).squeeze(1)
    return {'target_token_ids': ids.detach().cpu().tolist(), 'positions': positions,
            'logprobs': values.detach().cpu().tolist(), 'sum_logprob': values.sum().item(),
            'mean_logprob': values.mean().item(), 'semantics': 'teacher-forced conditional target log-probability; not generated output'}


def attention_rows(q, k, mask, *, rows, heads, scaling, causal=True):
    import torch
    if q.ndim != 4 or k.ndim != 4 or q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError('expected batch,heads,sequence,head_dim')
    if q.shape[1] % k.shape[1] or not rows or not heads or len(rows) > 64:
        raise ValueError('invalid grouped heads/capture rows')
    if any(type(i) is not int or i < 0 or i >= q.shape[2] for i in rows) or any(type(i) is not int or i < 0 or i >= q.shape[1] for i in heads):
        raise ValueError('attention index outside pass')
    # Select KV head for each query head; never materialize all attention heads.
    kv = [h // (q.shape[1] // k.shape[1]) for h in heads]
    selected_q = q[:, heads][:, :, rows].float()
    selected_k = k[:, kv].float()
    scores = torch.matmul(selected_q, selected_k.transpose(-1, -2)) * scaling
    if mask is None and causal:
        future = torch.arange(k.shape[2], device=q.device)[None, :] > torch.tensor(rows, device=q.device)[:, None]
        scores = scores.masked_fill(future[None, None], -float('inf'))
    elif mask is not None:
        if mask.ndim != 4:
            raise ValueError('actual four-dimensional attention mask required')
        chosen_mask = mask[:, :, rows, :k.shape[2]]
        if chosen_mask.shape[1] != 1:
            chosen_mask = chosen_mask[:, heads]
        scores = scores.masked_fill(~chosen_mask, -float('inf')) if chosen_mask.dtype == torch.bool else scores + chosen_mask.float()
    return scores.softmax(dim=-1)


class ActivationCapture:
    def __init__(self, model, spec, *, max_bytes=256*1024**2, allow_patches=True):
        self.model, self.spec, self.max_bytes = model, spec, max_bytes
        self.values, self.metadata, self.handles, self.bytes = {}, {}, [], 0
        self.diagnostic_only = bool(spec.get('patches'))
        if self.diagnostic_only and not allow_patches:
            raise ValueError('activation injection forbidden in final generation')
        if set(spec) - {'points', 'attention', 'patches'}:
            raise ValueError('unknown capture fields')
        self.modules = dict(model.named_modules())
        self.attn_state = {}

    @staticmethod
    def tensor_output(output):
        import torch
        value = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(value, torch.Tensor):
            raise ValueError('hook must expose a tensor')
        return value

    def keep(self, key, value, metadata):
        amount = value.numel() * 4
        if self.bytes + amount > self.max_bytes:
            raise ValueError('capture storage bound exceeded; no silent truncation')
        if key in self.values:
            raise ValueError('module executed repeatedly in a single capture')
        self.bytes += amount
        self.values[key] = value.detach().float().cpu().contiguous()
        self.metadata[key] = dict(metadata, source_dtype=str(value.dtype), shape=list(value.shape), bytes=amount)

    def point_hook(self, item):
        def hook(module, args, output):
            value = self.tensor_output(args if item['kind'] == 'input' else output)
            positions = item['positions']
            if value.ndim < 3 or len(positions) > 64 or not positions or any(type(i) is not int or i < 0 or i >= value.shape[1] for i in positions):
                raise ValueError('capture positions must match batch,sequence,features')
            self.keep(item['module'] + ':' + item['kind'], value[:, positions], {'positions': positions, 'kind': item['kind']})
        return hook

    def patch_hook(self, item):
        def hook(module, args, output):
            import torch
            value = self.tensor_output(output)
            positions = item['positions']
            if value.ndim != 3 or not positions or len(positions) > 64 or any(type(i) is not int or i < 0 or i >= value.shape[1] for i in positions):
                raise ValueError('invalid diagnostic patch positions')
            patch = torch.tensor(item['values'], dtype=value.dtype, device=value.device)
            if tuple(patch.shape) != (len(positions), value.shape[-1]) or not bool(torch.isfinite(patch).all()):
                raise ValueError('diagnostic patch shape/finiteness mismatch')
            changed = value.clone()
            if item['mode'] == 'replace': changed[:, positions] = patch
            elif item['mode'] == 'add': changed[:, positions] += patch
            else: raise ValueError('unsupported diagnostic intervention')
            if isinstance(output, tuple): return (changed, *output[1:])
            if isinstance(output, list): return [changed, *output[1:]]
            return changed
        return hook

    def attention_hooks(self, item):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
        name = item['module']; module = self.modules[name]; state = {}
        for field in ('q_norm', 'k_norm', 'head_dim', 'scaling'):
            if not hasattr(module, field):
                raise ValueError('attention capture requires the pinned Qwen3VL attention architecture')
        def before(mod, args, kwargs):
            if kwargs.get('past_key_values') is not None:
                raise ValueError('teacher-forced read must disable KV cache')
            state['position_embeddings'] = kwargs.get('position_embeddings', args[1] if len(args)>1 else None)
            state['mask'] = kwargs.get('attention_mask', args[2] if len(args)>2 else None)
            if state['position_embeddings'] is None:
                raise ValueError('same-pass RoPE evidence unavailable')
        def normalized(which):
            def capture(mod, args, output): state[which] = output.detach()
            return capture
        def after(mod, args, output):
            if not all(k in state for k in ('q', 'k', 'position_embeddings')):
                raise ValueError('same-pass Q/K capture missing')
            q, k = state['q'].transpose(1, 2), state['k'].transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, *state['position_embeddings'])
            rows = attention_rows(q, k, state['mask'], rows=item['rows'], heads=item['heads'], scaling=mod.scaling)
            self.keep(name + ':attention', rows, {'rows': item['rows'], 'heads': item['heads'],
                      'method': 'FP32 QK+same-pass RoPE/mask diagnostic reconstruction; generation stays SDPA'})
            state.clear()
        self.handles += [module.register_forward_pre_hook(before, with_kwargs=True), module.q_norm.register_forward_hook(normalized('q')),
                         module.k_norm.register_forward_hook(normalized('k')), module.register_forward_hook(after)]

    def __enter__(self):
        try:
            for item in self.spec.get('points', []):
                if item['module'] not in self.modules or item['kind'] not in ('input', 'output'):
                    raise ValueError('unknown hook point')
                self.handles.append(self.modules[item['module']].register_forward_hook(self.point_hook(item)))
            for item in self.spec.get('attention', []):
                if item['module'] not in self.modules: raise ValueError('unknown attention module')
                self.attention_hooks(item)
            for item in self.spec.get('patches', []):
                if item['module'] not in self.modules: raise ValueError('unknown diagnostic patch module')
                self.handles.append(self.modules[item['module']].register_forward_hook(self.patch_hook(item)))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles: handle.remove()
        self.handles.clear()
        self.attn_state.clear()
