# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional adapter to an independently supplied, pinned Maxwell checkout.

No internal harness source is vendored. This is a capped128 adaptation of the
screenshot ablation executor, not historical R97 with formatting repair.
Run inside the application's isolated platform job with only label-free queries.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import fcntl
import hashlib
import json
import sys
import time

PROFILES = {'maxwell-grid32-cap128': 'E_atlas_only',
            'maxwell-grid32-tools-guidance-cap128': 'A_full'}
PIN = '1711959a40f89cbf4072ea343de318a204903dfd'
PINNED = {'aoi_harness/analysis/tht_ablation.py': '26f56278291e10842661f3c73cf10f0d0730fd3946c0bbe28025b9e2e86412f0',
          'aoi_harness/analysis/tht_ablation_runtime.py': 'e1227fd0958c53b602f8a2401522217bc5012fa8a53c3c3bfca48b14e3f3ad1b'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8388608), b''): h.update(b)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f: json.dump(value, f, indent=2); f.write('\n')


def within(root, relative):
    path = (Path(root) / relative).resolve(strict=True)
    if Path(relative).is_absolute() or not path.is_relative_to(Path(root).resolve()):
        raise ValueError('path escapes configured root')
    return path


def verify_inventory(path, root):
    inventory = read(path)
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError('nonempty path-to-hash inventory required')
    for name, expected in inventory.items():
        if sha(within(root, name)) != expected:
            raise ValueError('inventory binding changed')
    return inventory


def identity(row):
    if not row.get('object_name') or not row.get('scan_id'):
        raise ValueError('missing trusted joint identity')
    return row['object_name'], row['scan_id'].removeprefix('SCN_')


def verify_dataset_view(manifest, root, split):
    """The native dataset view may contain Mining labels, never query gold."""
    inventory = verify_inventory(manifest, root)
    metadata = {'train.jsonl', split + '.jsonl'}
    if not metadata <= set(inventory):
        raise ValueError('native Mining and projected query manifests required')
    for name in inventory:
        if name not in metadata and Path(name).suffix.lower() not in ('.png', '.jpg', '.jpeg', '.bmp', '.webp'):
            raise ValueError('dataset view contains unneeded metadata or results')
    actual = {str(p.relative_to(root)) for p in Path(root).rglob('*') if p.is_file()}
    if actual != set(inventory):
        raise ValueError('dataset view contains unbound files')
    return inventory


def verify_config(path):
    c = read(path)
    if c.get('schema') != 'tao-icl-maxwell-native/1' or c.get('profile') not in PROFILES:
        raise ValueError('unknown explicitly versioned profile')
    if c['native']['revision'] != PIN or c['query_split'] not in ('val', 'test'):
        raise ValueError('source revision or query split differs')
    if c['protocol'] != {'examples_per_class': 4, 'seed': 0, 'reasoning': 'off', 'label_mode': 'names',
                          'execution_policy': 'tht_ablation_no_format_repair_v1', 'format_retries': 0, 'max_new_tokens': 128}:
        raise ValueError('protocol differs; use a separately registered profile')
    checkout = Path(c['native']['checkout']).resolve()
    sources = verify_inventory(c['native']['source_manifest'], checkout)
    for name, expected in PINNED.items():
        if sources.get(name) != expected:
            raise ValueError('native executor source differs')
    for p in checkout.rglob('*'):
        if p.is_file() and p.suffix in ('.py', '.txt', '.json') and '.git' not in p.parts:
            # The supplied checkout is a code/prompt-only view, not historical results.
            rel = str(p.relative_to(checkout))
            if rel not in sources:
                raise ValueError('unbound file in native source view')
    assets = c['assets']; dataset = Path(assets['dataset_view']).resolve()
    verify_dataset_view(assets['dataset_manifest'], dataset, c['query_split'])
    for key in ('atlas', 'selection', 'queries'):
        if sha(assets[key]['path']) != assets[key]['sha256']:
            raise ValueError('asset binding changed')
    for key in ('training_cache', 'query_cache'):
        verify_inventory(assets[key]['manifest'], assets[key]['path'])
    verify_inventory(c['model']['inventory'], c['model']['path'])
    rows = [json.loads(line) for line in Path(assets['queries']['path']).read_text().splitlines() if line.strip()]
    if not rows or len({r['filename'] for r in rows}) != len(rows):
        raise ValueError('nonempty unique query identities required')
    for row in rows:
        if set(row) != {'filename', 'object_name', 'scan_id', 'image_sha256'}:
            raise ValueError('queries must be label-free identity/hash projections')
        identity(row)
        if sha(within(dataset, row['filename'])) != row['image_sha256']:
            raise ValueError('query bytes changed')
    split_rows = [json.loads(line) for line in (dataset / (c['query_split'] + '.jsonl')).read_text().splitlines() if line.strip()]
    if split_rows != rows:
        raise ValueError('native split must match the supplied label-free projection')
    selected = read(assets['selection']['path'])['samples']
    if len(selected) != 32 or len({r['labels'][0] for r in selected}) != 8:
        raise ValueError('fixed eight-class four-per-class atlas required')
    for label in {r['labels'][0] for r in selected}:
        group = [r for r in selected if r['labels'] == [label]]
        if len(group) != 4 or len({identity(r) for r in group}) != 4:
            raise ValueError('atlas class quota or joint diversity differs')
    if any(identity(q) == identity(r) or q['image_sha256'] == r['sha256'] or q['filename'] == r['filename']
           for q in rows for r in selected):
        raise ValueError('fixed-atlas overlap blocks this study; never silently replace tiles')
    return c, rows


def cap_backend(native, max_tokens=128):
    """Intercept only generation cap; retain native rendering/decoder/tool semantics."""
    original_generate = native.model.generate
    def bounded_generate(*args, **kwargs):
        if kwargs.get('max_new_tokens') != 4096:
            raise ValueError('native backend contract changed')
        kwargs['max_new_tokens'] = max_tokens
        return original_generate(*args, **kwargs)
    native.model.generate = bounded_generate
    original_request = native.generate
    def request(messages, tools, root, request_path):
        try:
            result = original_request(messages, tools, root, request_path)
        finally:
            if request_path.exists():
                evidence = read(request_path)
                evidence['native_declared_generation'] = dict(evidence['generation'])
                evidence['generation']['max_new_tokens'] = max_tokens
                evidence['adapter_deviation'] = 'Explicit cap128; native decoding and tool executor retained'
                request_path.write_text(json.dumps(evidence, indent=2) + '\n')
        result['token_limit_reached'] = result['usage']['completion_tokens'] >= max_tokens
        return result
    native.generate = request
    return native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--config', required=True)
    a = parser.parse_args(); config, rows = verify_config(a.config)
    output = Path(config['output']); output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'adapter.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sys.path.insert(0, str(Path(config['native']['checkout']).resolve()))
    from aoi_harness.analysis.tht_ablation import Mode
    from aoi_harness.analysis.tht_ablation_runtime import run_case
    mode = Mode(PROFILES[config['profile']]); mode.split = config['query_split']
    assets = config['assets']; mode.dataset = Path(assets['dataset_view'])
    mode.atlas_path = Path(assets['atlas']['path']); mode.selection_path = Path(assets['selection']['path'])
    mode.train_cache = Path(assets['training_cache']['path']); mode.query_cache = Path(assets['query_cache']['path'])
    args = SimpleNamespace(model='cosmos_reason3_8b', cosmos_reasoning='off',
        prompt_config=config['native']['prompt_config'], examples_per_class=4, max_rounds=10, seed=0, limit=None, workers=1)
    native_rows = [{k: v for k, v in row.items() if k != 'image_sha256'} for row in rows]
    prepared = mode.prepare(args, native_rows)
    if prepared.label_mode != 'names': raise ValueError('original class names required')
    receipt = {'schema': 'tao-icl-maxwell-receipt/1', 'status': 'PREPARED', 'config_sha256': sha(a.config),
               'profile': config['profile'], 'query_count': len(rows), 'source_revision': PIN,
               'native_source_manifest_sha256': sha(config['native']['source_manifest']),
               'model_inventory_sha256': sha(config['model']['inventory']),
               'native_declared_model': prepared.settings['model'], 'actual_model': config['model']['path'],
               'native_declared_split': prepared.settings['split'], 'actual_split': config['query_split'],
               'max_new_tokens': 128, 'native_max_new_tokens': 4096, 'format_retries': 0,
               'atlas_overlap_audit': 'PASS; stricter adapter gate than native fixed atlas',
               'prompt_sha256': hashlib.sha256(prepared.user.encode()).hexdigest(),
               'tool_schema_sha256': hashlib.sha256(json.dumps(mode.schemas('names'), sort_keys=True).encode()).hexdigest(),
               'tool_round_limit': prepared.settings['max_tool_rounds']}
    marker = output / 'prepared.json'
    if marker.exists():
        if read(marker) != receipt: raise ValueError('prepared configuration changed')
    else:
        mode.prepare_output(output, 'names')
        write(output / 'native-settings.private.json', prepared.settings); write(marker, receipt)
    if a.action == 'prepare': print(json.dumps(receipt)); return
    if (output / 'complete.json').exists(): print(json.dumps(read(output / 'complete.json'))); return
    completed = []; pending = []
    for row in native_rows:
        case = output / 'cases' / hashlib.sha256(row['filename'].encode()).hexdigest()
        trace = case / 'trace.json'
        if trace.exists():
            saved = read(trace)
            if saved['status'] not in ('complete', 'invalid_final'):
                raise ValueError('ambiguous previous generation; explicit reconciliation required')
            completed.append(saved)
        else: pending.append((row, case))
    peak_allocated = peak_reserved = None
    if pending:
        from aoi_harness.harness_backends import CosmosBackend
        backend = cap_backend(CosmosBackend(model_path=config['model']['path']))
        backend.torch.cuda.reset_peak_memory_stats()
        for row, case in pending:
            completed.append(run_case(backend, prepared.bank, mode.query_path(row), mode.query_identity(row), output,
                case, prepared.system, prepared.user, prepared.settings['max_tool_rounds'], reasoning=False, label_mode='names', mode=mode))
            print(json.dumps({'operation': 'maxwell-profile-progress', 'completed': len(completed), 'planned': len(rows)}), flush=True)
        peak_allocated = backend.torch.cuda.max_memory_allocated()
        peak_reserved = backend.torch.cuda.max_memory_reserved()
    result = {**receipt, 'status': 'COMPLETE', 'completed': len(completed),
              'invalid': sum(t['status'] == 'invalid_final' for t in completed),
              'model_calls': sum(len(t['turns']) for t in completed),
              'tool_calls': sum(len(turn['model_response']['calls']) for t in completed for turn in t['turns']),
              'source_atlas_sha256': sha(mode.atlas_path), 'rendered_atlas_sha256': sha(output / 'reference_atlas.png'),
              'case_wall_seconds': sum(t['seconds'] for t in completed),
              'input_tokens': sum(turn['model_response']['usage']['input_tokens'] for t in completed for turn in t['turns']),
              'output_tokens': sum(turn['model_response']['usage']['completion_tokens'] for t in completed for turn in t['turns']),
              'kernel_active_seconds': None, 'currency': None,
              'peak_allocated_bytes': peak_allocated, 'peak_reserved_bytes': peak_reserved,
              'cost_quality': 'native trace timing/usage; kernel time and currency not measured', 'finished_at': time.time()}
    write(output / 'complete.json', result); print(json.dumps(result))


if __name__ == '__main__':
    try: main()
    except Exception as error:
        print(json.dumps({'operation': 'maxwell-profile', 'status': 'BLOCKED', 'error_type': type(error).__name__}))
        raise SystemExit(1)
