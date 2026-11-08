# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dev-only contrast capture, writer ranking and gradient-free rank-one removal.

Scientific choices (pairs, residual layers, position, dose, k and controls)
belong to each editor's frozen preregistration. No default dominant class.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from wsp03_worker import canonical_target, apply_candidate, digest
from wsp03_io import json_write
from wsp03_bindings import file_hash
from wsp03_fast import render_batch, require_dev, read_rows


def unit_direction(undesired, desired):
    import torch
    with torch.inference_mode():
        bad = torch.as_tensor(undesired).detach().float()
        good = torch.as_tensor(desired).detach().float()
        if bad.shape != good.shape or bad.ndim != 2 or not bad.shape[0]:
            raise ValueError('paired observations x residual width required')
        difference = (bad-good).mean(dim=0)
        norm = torch.linalg.vector_norm(difference)
        if not bool(torch.isfinite(difference).all()) or norm <= 0:
            raise ValueError('nonzero finite development contrast required')
        return difference/norm


def writer_ranking(weight, direction, *, mode='absolute'):
    import torch
    with torch.inference_mode():
        w = weight.detach().float()
        u = direction.detach().float()
        if w.ndim != 2 or u.shape != (w.shape[0],) or not torch.isclose(u.norm(), u.new_tensor(1.), atol=1e-5):
            raise ValueError('writer weight [residual, neuron] and unit direction required')
        projection = u @ w
        scores = {'absolute': projection.abs(), 'positive': projection,
                  'negative': -projection}.get(mode)
        if scores is None or not bool(torch.isfinite(scores).all()):
            raise ValueError('finite supported writer scores required')
        # Stable sort freezes index order for ties.
        return torch.argsort(scores, descending=True, stable=True), projection


def rank_one_removal(weight, direction, *, k, alpha, mode='absolute', mask=None):
    import torch
    with torch.inference_mode():
        w = weight.detach().float()
        order, projection = writer_ranking(w, direction, mode=mode)
        if type(k) is not int or not 1 <= k <= w.shape[1] or not 0 < alpha < float('inf'):
            raise ValueError('positive finite dose and valid top-k required')
        if mask is None:
            mask = torch.zeros(w.shape[1], dtype=torch.bool, device=w.device)
            mask[order[:k]] = True
        else:
            mask = torch.as_tensor(mask, dtype=torch.bool, device=w.device)
            if mask.shape != (w.shape[1],) or int(mask.sum()) != k:
                raise ValueError('fixed top-k mask shape/count differs')
        delta = -float(alpha)*torch.outer(direction.detach().float(), projection*mask)
        return w+delta, {'mask': mask, 'order': order, 'projection': projection,
                        'delta': delta, 'alpha': float(alpha)}


def random_same_norm(weight, direction, *, k, alpha, seed, mode='absolute', mask_policy='own'):
    import torch
    with torch.inference_mode():
        _, target = rank_one_removal(weight, direction, k=k, alpha=alpha, mode=mode)
        generator = torch.Generator(device='cpu').manual_seed(seed)
        random = torch.randn(direction.shape, generator=generator).to(weight.device)
        random /= random.norm()
        if mask_policy not in ('own', 'target'):
            raise ValueError('random mask policy must be preregistered as own or target')
        edited, control = rank_one_removal(weight, random, k=k, alpha=1, mode=mode,
            mask=target['mask'] if mask_policy == 'target' else None)
        norm = control['delta'].norm()
        if norm <= 0:
            raise ValueError('random control has zero writer projection')
        scale = target['delta'].norm()/norm
        delta = control['delta']*scale
        return weight.detach().float()+delta, dict(control, delta=delta,
            alpha=float(scale), direction=random, random_seed=seed,
            norm_target=float(target['delta'].norm()), mask_policy=mask_policy)


def capture_contrast(model, processor, request, undesired_text, desired_text, modules,
                     *, image_root, position='label_end'):
    """Teacher-force two answers on the same image/prompt with zero gradients."""
    import torch
    from wsp03_capture import ActivationCapture
    if not modules or len(set(modules)) != len(modules) or position not in ('label_end', 'target_end'):
        raise ValueError('unique residual modules and registered continuation position required')
    if undesired_text == desired_text:
        raise ValueError('contrast answers must differ')
    texts, groups, paths = render_batch(processor, [request], image_root)
    captures, metadata = [], []
    model.eval()
    model.requires_grad_(False)
    device = next(model.parameters()).device
    try:
        for forced in (undesired_text, desired_text):
            inputs = processor(text=texts, images=groups[0], return_tensors='pt').to(device)
            prompt_length = inputs['input_ids'].shape[1]
            target, class_positions, span = canonical_target(processor.tokenizer, forced)
            if not target or len(target) > 64:
                raise ValueError('bounded canonical contrast target required')
            end = class_positions[-1] if position == 'label_end' else len(target)-1
            ids = torch.tensor([target], device=device, dtype=inputs['input_ids'].dtype)
            inputs['input_ids'] = torch.cat([inputs['input_ids'], ids], dim=1)
            inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.ones_like(ids)], dim=1)
            spec = {'points': [{'module': name, 'kind': 'output', 'positions': [prompt_length+end]}
                               for name in modules]}
            with torch.inference_mode(), ActivationCapture(model, spec, allow_patches=False) as capture:
                result = model(**inputs, use_cache=False, return_dict=True)
                del result
            captures.append({name: capture.values[name+':output'][0, 0] for name in modules})
            metadata.append({'forced_text': forced, 'target_token_ids': target,
                             'class_span': span, 'position': prompt_length+end})
        return {name: (captures[0][name], captures[1][name]) for name in modules}, {
            'request_sha256': digest(request), 'continuations': metadata,
            'position_semantics': position, 'image_order': paths[0], 'gradients': False}
    finally:
        for image in groups[0]:
            image.close()


def stage_parent(store, directory):
    """Copy every original R4 replacement into a new base-relative patch."""
    from safetensors import safe_open
    from wsp03_kernel import digest_tensor
    store.reset()
    if directory is None:
        return None
    directory = Path(directory).resolve(strict=True)
    manifest = json.loads((directory/'manifest.json').read_text())
    if not manifest.get('valid'):
        raise ValueError('valid frozen parent patch required')
    for item in manifest['tensors']:
        base = store.load_tensor(item['name'])
        if digest_tensor(base) != item['before_sha256']:
            raise ValueError('parent must replay from the original pristine base')
        path = (directory/item['file']).resolve(strict=True)
        if path.parent != directory:
            raise ValueError('parent patch path escaped')
        with safe_open(path, framework='pt', device='cpu') as handle:
            value = handle.get_tensor('value')
        if digest_tensor(value) != item['after_sha256']:
            raise ValueError('parent patch tensor changed')
        store.save_edit(item['name'], value)
    return digest(manifest)


def edit_candidate(store, cfg, directions):
    """Compose selected layer edits onto base or original R4, then export replay."""
    import torch
    from wsp03_kernel import digest_tensor
    track = cfg['track']
    if track not in ('R3-ext2', 'R4-ext'):
        raise ValueError('new extension track required')
    if (track == 'R3-ext2') != (cfg.get('parent_patch') is None):
        raise ValueError('R3-ext2 starts pristine; R4-ext requires original R4 patch')
    if track == 'R4-ext' and (cfg.get('profile') != 'P09' or not cfg.get('parent_freeze_sha256')
                             or not cfg.get('reference_state_sha256')):
        raise ValueError('original R4 parent/P09/reference bindings required')
    parent_hash = stage_parent(store, cfg.get('parent_patch'))
    if cfg.get('parent_patch_manifest_sha256') != parent_hash:
        raise ValueError('frozen parent manifest differs')
    layers = cfg['layers']
    if not layers or len(set(layers)) != len(layers):
        raise ValueError('unique preregistered layer names required')
    if cfg.get('control') == 'single_layer' and len(layers) != 1:
        raise ValueError('single-layer control must isolate exactly one layer')
    records = []
    with torch.inference_mode():
        for name in layers:
            if not name.endswith('.mlp.down_proj.weight'):
                raise ValueError('writer removal edits only MLP down_proj weights')
            parent = store.staged.get(name)
            if parent is None:
                parent = store.load_tensor(name)
            direction = directions[name].detach().float()
            kwargs = {'k': cfg['k'], 'alpha': cfg['alpha'], 'mode': cfg['writer_mode']}
            if cfg['control'] == 'random_same_norm':
                edited, detail = random_same_norm(parent, direction, seed=cfg['seed'],
                    mask_policy=cfg['random_mask_policy'], **kwargs)
            elif cfg['control'] in ('direction', 'single_layer'):
                edited, detail = rank_one_removal(parent, direction, **kwargs)
            else:
                raise ValueError('unknown preregistered control')
            store.save_edit(name, edited.cpu())
            rounded = store.staged[name]
            records.append({'name': name, 'parent_weight_sha256': digest_tensor(parent),
                'direction_sha256': digest_tensor(direction), 'k': cfg['k'],
                'selected_neurons': detail['mask'].nonzero().flatten().cpu().tolist(),
                'alpha': detail['alpha'], 'prequant_delta_l2': float(detail['delta'].norm()),
                'actual_parent_delta_l2': float((rounded.float()-parent.float()).norm()),
                'after_sha256': digest_tensor(rounded)})
        store.edit_configuration = {'track': track, 'parent': cfg.get('parent', 'base'),
            'parent_manifest_sha256': parent_hash, 'preregistration_sha256': cfg['preregistration_sha256'],
            'configuration': cfg, 'operations': records, 'gradients': False}
        result = store.commit_candidate(cfg['candidate_id'], cfg['hypothesis'], method='strict_wsp')
    store.verify(cfg['candidate_id'])
    return result


def run_capture(cfg):
    import torch
    from safetensors.torch import save_file
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    require_dev([pair['request']['id'] for pair in cfg['pairs']], read_rows(cfg['development_ids']))
    prereg = Path(cfg['preregistration'])
    if file_hash(prereg) != cfg['preregistration_sha256']:
        raise ValueError('capture must bind immutable preregistration before diagnostics')
    output = Path(cfg['output'])
    output.mkdir(parents=True, exist_ok=False)
    processor = AutoProcessor.from_pretrained(cfg['model'], local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(cfg['model'], torch_dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True).eval()
    if cfg.get('parent_patch'):
        apply_candidate(model, cfg['parent_patch'])
    model.requires_grad_(False)
    model.to('cuda')
    bad = {name: [] for name in cfg['modules']}
    good = {name: [] for name in cfg['modules']}
    for pair in cfg['pairs']:
        values, receipt = capture_contrast(model, processor, pair['request'], pair['undesired_text'],
            pair['desired_text'], cfg['modules'], image_root=cfg['image_root'], position=cfg['position'])
        for name, (negative, positive) in values.items():
            bad[name].append(negative)
            good[name].append(positive)
        with (output/'pairs.jsonl').open('a') as stream:
            stream.write(json.dumps(receipt)+'\n')
    directions = {}
    for module in cfg['modules']:
        directions[module] = unit_direction(torch.stack(bad[module]), torch.stack(good[module]))
    save_file(directions, output/'directions.safetensors')
    receipt = {'schema': 'wsp03-dev-directions/1', 'preregistration_sha256': cfg['preregistration_sha256'],
               'configuration_sha256': digest(cfg), 'pairs': len(cfg['pairs']),
               'orientation': 'mean(undesired - desired), normalized per layer',
               'file_sha256': file_hash(output/'directions.safetensors'), 'gradients': False,
               'development_only': True}
    json_write(output/'receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('capture', 'edit'))
    parser.add_argument('config')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.action == 'capture':
        result = run_capture(cfg)
    else:
        from safetensors.torch import load_file
        from wsp03_kernel import TensorStore
        if file_hash(cfg['preregistration']) != cfg['preregistration_sha256']:
            raise ValueError('edit preregistration changed')
        raw = load_file(cfg['directions'])
        directions = {weight: raw[module] for weight, module in cfg['direction_mapping'].items()}
        store = TensorStore(cfg['model'], cfg['patches'])
        result = edit_candidate(store, cfg, directions)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
