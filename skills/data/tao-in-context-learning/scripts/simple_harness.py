# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional adapter to an independently supplied, pinned Maxwell checkout.

No internal harness source is vendored. The native4096 profile preserves the
screenshot ablation executor; cap128 is an explicitly separate adaptation.
Neither profile is historical R97 with formatting repair.
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

PROFILES = {'simple-harness-grid32-cap128': 'E_atlas_only',
            'simple-harness-grid32-tools-guidance-cap128': 'A_full',
            'simple-harness-grid32-tools-guidance-native4096': 'A_full',
            'simple-harness-grid32-explore-cap128-v1': 'A_full'}
CAPS = {profile: 4096 if profile.endswith('native4096') else 128 for profile in PROFILES}
EXPLORE_PROFILE = 'simple-harness-grid32-explore-cap128-v1'
EXPLORE_GUIDANCE = Path(__file__).resolve().parents[1] / 'references/simple-harness-explore-v1.txt'
PIN = '1711959a40f89cbf4072ea343de318a204903dfd'
NATIVE_ATLAS_SHA256 = '09b29726dfadbecdbd58c49ba74c5dd3aa8e7312d770732d8ab1fa30db61dddd'
NATIVE_SELECTION_SHA256 = '54e2f494aff841189efefe13095d483fd6ba0da0f0d83b179bcfaf1735138a44'
CLASSES = ('Good', 'Bright', 'Interrupted', 'Trapped', 'Hidden', 'Bridge', 'Dry Contact', 'Less Solder')
PINS_PATH = Path(__file__).resolve().parents[1] / 'references/simple-harness-source-pins.json'
PINNED = json.loads(PINS_PATH.read_text())['files']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8388608), b''): h.update(b)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def apply_guidance(prepared, profile):
    """Opt-in prompt extension; never force, synthesize or repair model actions."""
    if profile != EXPLORE_PROFILE:
        return None
    guidance = EXPLORE_GUIDANCE.read_text().strip()
    if not guidance:
        raise ValueError('exploration guidance is empty')
    prepared.user += '\n\n' + guidance
    prepared.settings['user_prompt'] = prepared.user
    deviation = {'kind': 'appended_exploration_guidance', 'version': 1,
                 'sha256': sha(EXPLORE_GUIDANCE)}
    prepared.settings['prompt_deviation'] = deviation
    return deviation


def write(path, value):
    with Path(path).open('x') as f: json.dump(value, f, indent=2); f.write('\n')


def within(root, relative):
    path = (Path(root) / relative).resolve(strict=True)
    if Path(relative).is_absolute() or not path.is_relative_to(Path(root).resolve()):
        raise ValueError('path escapes configured root')
    return path


def verify_inventory(path, root, complete=False, required=()):
    inventory = read(path)
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError('nonempty path-to-hash inventory required')
    for name, expected in inventory.items():
        if sha(within(root, name)) != expected:
            raise ValueError('inventory binding changed')
    if not set(required) <= set(inventory):
        raise ValueError('required inventory files missing')
    if complete and set(inventory) != {str(p.relative_to(root)) for p in Path(root).rglob('*') if p.is_file()}:
        raise ValueError('inventory does not cover the complete snapshot')
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


def verify_atlas_selection(selection, dataset):
    if selection.get('source_split') != 'train':
        raise ValueError('atlas must come from Mining/train')
    selected = selection['samples']
    rows = [json.loads(line) for line in (Path(dataset) / 'train.jsonl').read_text().splitlines() if line.strip()]
    mining = {r['filename']: r for r in rows}
    if len(mining) != len(rows) or len(selected) != 32 or len({r['filename'] for r in selected}) != 32:
        raise ValueError('unique original Mining and atlas identities required')
    if {tuple(r['labels']) for r in selected} != {(c,) for c in CLASSES}:
        raise ValueError('atlas requires the canonical eight single-label classes')
    for row in selected:
        original = mining.get(row['filename'])
        if original is None or any(row[k] != original[k] for k in ('labels', 'object_name', 'scan_id')):
            raise ValueError('atlas selection differs from Mining provenance')
        if sha(within(dataset, row['filename'])) != row['sha256']:
            raise ValueError('atlas source image differs from Mining bytes')
    for label in CLASSES:
        group = [r for r in selected if r['labels'] == [label]]
        if len(group) != 4 or len({identity(r) for r in group}) != 4:
            raise ValueError('atlas class quota or joint diversity differs')
    return selected


def binding_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def case_seal(output, trace_path, row, binding):
    trace = read(trace_path)
    requests = [turn['request'] for turn in trace['turns']]
    if not requests or len(set(requests)) != len(requests):
        raise ValueError('unique native requests required')
    return {'binding_sha256': binding, 'query_sha256': binding_digest(row),
            'trace_sha256': sha(trace_path),
            'request_sha256': {name: sha(within(output, name)) for name in requests}}


def complete_seal(output, binding):
    cases = {str(p.relative_to(output)): sha(p) for p in (output / 'cases').glob('*/adapter-receipt.json')}
    return {'binding_sha256': binding, 'complete_sha256': sha(output / 'complete.json'),
            'case_receipts_sha256': binding_digest(cases)}


def validate_case(output, case, row, binding):
    trace_path, seal_path = case / 'trace.json', case / 'adapter-receipt.json'
    if not trace_path.is_file() or not seal_path.is_file():
        raise ValueError('partial case without terminal binding; explicit reconciliation required')
    trace, seal = read(trace_path), read(seal_path)
    if seal != case_seal(output, trace_path, row, binding):
        raise ValueError('saved case belongs to different input or trace changed')
    if trace.get('status') not in ('complete', 'invalid_final') or not trace.get('turns') or not trace.get('assets'):
        raise ValueError('terminal trace is incomplete')
    if trace.get('prediction') not in (*CLASSES, 'INVALID'):
        raise ValueError('saved prediction is outside the native vocabulary')
    for name, asset in trace['assets'].items():
        if sha(within(output, name)) != asset['sha256']:
            raise ValueError('saved case image artifact changed')
    return trace


def verify_config(path):
    c = read(path)
    if c.get('schema') != 'tao-icl-simple-harness-native/1' or c.get('profile') not in PROFILES:
        raise ValueError('unknown explicitly versioned profile')
    if c['native']['revision'] != PIN or c['query_split'] not in ('val', 'test'):
        raise ValueError('source revision or query split differs')
    if c['protocol'] != {'examples_per_class': 4, 'seed': 0, 'reasoning': 'off', 'label_mode': 'names',
                          'execution_policy': 'tht_ablation_no_format_repair_v1', 'format_retries': 0,
                          'max_new_tokens': CAPS[c['profile']]}:
        raise ValueError('protocol differs; use a separately registered profile')
    if c['native']['prompt_config'] != 'config/current/tht-inspection.json':
        raise ValueError('native profile requires its pinned prompt configuration')
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
        verify_inventory(assets[key]['manifest'], assets[key]['path'], complete=True, required=('metadata.json', 'features.npz'))
    verify_inventory(c['model']['inventory'], c['model']['path'], complete=True, required=('config.json', 'tokenizer_config.json'))
    if not any(name.endswith('.safetensors') for name in read(c['model']['inventory'])):
        raise ValueError('model inventory must bind the actual weight files')
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
    if assets['atlas']['sha256'] != NATIVE_ATLAS_SHA256 or assets['selection']['sha256'] != NATIVE_SELECTION_SHA256:
        raise ValueError('this native profile requires the pinned original atlas/selection pair')
    selected = verify_atlas_selection(read(assets['selection']['path']), dataset)
    if any(identity(q) == identity(r) or q['image_sha256'] == r['sha256'] or q['filename'] == r['filename']
           for q in rows for r in selected):
        raise ValueError('fixed-atlas overlap blocks this study; never silently replace tiles')
    output = Path(c['output']).resolve()
    roots = [checkout, dataset, Path(c['model']['path']).resolve(),
             *(Path(assets[key]['path']).resolve() for key in ('training_cache', 'query_cache'))]
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in roots):
        raise ValueError('results must be separate from source, dataset, model and cache roots')
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
                evidence['adapter_deviation'] = f'Explicit cap{max_tokens}; native decoding and tool executor retained'
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
    with (output / 'adapter.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
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
        prompt_deviation = apply_guidance(prepared, config['profile'])
        receipt = {'schema': 'tao-icl-simple-harness-receipt/2', 'status': 'PREPARED', 'config_sha256': sha(a.config),
                   'adapter_sha256': sha(__file__), 'source_pins_sha256': sha(PINS_PATH),
                   'profile': config['profile'], 'query_count': len(rows), 'source_revision': PIN,
                   'native_source_manifest_sha256': sha(config['native']['source_manifest']),
                   'model_inventory_sha256': sha(config['model']['inventory']),
                   'dataset_inventory_sha256': sha(assets['dataset_manifest']),
                   'training_cache_inventory_sha256': sha(assets['training_cache']['manifest']),
                   'query_cache_inventory_sha256': sha(assets['query_cache']['manifest']),
                   'native_settings_sha256': binding_digest(prepared.settings),
                   'native_declared_model': prepared.settings['model'], 'actual_model': config['model']['path'],
                   'native_declared_split': prepared.settings['split'], 'actual_split': config['query_split'],
                   'max_new_tokens': CAPS[config['profile']], 'native_max_new_tokens': 4096, 'format_retries': 0,
                   'generation_deviation': None if CAPS[config['profile']] == 4096 else 'max_new_tokens: 4096 -> 128',
                   'prompt_deviation': prompt_deviation,
                   'decision_owner': 'CosmosBackend', 'tool_sequence': 'model_selected',
                   'atlas_overlap_audit': 'PASS; stricter adapter gate than native fixed atlas',
                   'prompt_sha256': hashlib.sha256(prepared.user.encode()).hexdigest(),
                   'tool_schema_sha256': hashlib.sha256(json.dumps(mode.schemas('names'), sort_keys=True).encode()).hexdigest(),
                   'tool_round_limit': prepared.settings['max_tool_rounds']}
        marker = output / 'prepared.json'
        if marker.exists():
            receipt['rendered_atlas_sha256'] = sha(output / 'reference_atlas.png')
            if read(marker) != receipt: raise ValueError('prepared configuration changed')
            if binding_digest(read(output / 'native-settings.private.json')) != receipt['native_settings_sha256']:
                raise ValueError('stored native settings changed')
        else:
            if (output / 'cases').exists() or (output / 'complete.json').exists():
                raise ValueError('run artifacts exist without a preparation binding')
            mode.prepare_output(output, 'names')
            receipt['rendered_atlas_sha256'] = sha(output / 'reference_atlas.png')
            write(output / 'native-settings.private.json', prepared.settings); write(marker, receipt)
        if a.action == 'prepare': print(json.dumps(receipt)); return
        binding = binding_digest(receipt)
        completed = []; pending = []
        for row in native_rows:
            case = output / 'cases' / hashlib.sha256(row['filename'].encode()).hexdigest()
            trace = case / 'trace.json'
            if case.exists():
                completed.append(validate_case(output, case, row, binding))
            else: pending.append((row, case))
        if (output / 'complete.json').exists():
            result = read(output / 'complete.json')
            seal = read(output / 'complete-seal.json')
            if pending or seal != complete_seal(output, binding):
                raise ValueError('completed run seal or case coverage differs')
            if result.get('status') != 'COMPLETE' or result.get('completed') != len(rows) or any(result.get(k) != v for k, v in receipt.items() if k != 'status'):
                raise ValueError('completed run belongs to a different configuration')
            print(json.dumps(result)); return
        peak_allocated = peak_reserved = None
        if pending:
            from aoi_harness.harness_backends import CosmosBackend
            backend = CosmosBackend(model_path=config['model']['path'])
            if CAPS[config['profile']] != 4096:
                backend = cap_backend(backend, CAPS[config['profile']])
            backend.torch.cuda.reset_peak_memory_stats()
            for row, case in pending:
                run_case(backend, prepared.bank, mode.query_path(row), mode.query_identity(row), output,
                    case, prepared.system, prepared.user, prepared.settings['max_tool_rounds'], reasoning=False, label_mode='names', mode=mode)
                write(case / 'adapter-receipt.json', case_seal(output, case / 'trace.json', row, binding))
                completed.append(validate_case(output, case, row, binding))
                print(json.dumps({'operation': 'simple-harness-progress', 'completed': len(completed), 'planned': len(rows)}), flush=True)
            peak_allocated = backend.torch.cuda.max_memory_allocated()
            peak_reserved = backend.torch.cuda.max_memory_reserved()
        result = {**receipt, 'status': 'COMPLETE', 'completed': len(completed),
                  'invalid': sum(t['status'] == 'invalid_final' for t in completed),
                  'model_calls': sum(len(t['turns']) for t in completed),
                  'tool_calls': sum(len(turn['tool_results']) for t in completed for turn in t['turns']),
                  'tool_calls_requested': sum(len(turn['model_response']['calls']) for t in completed for turn in t['turns']),
                  'tool_errors': sum(bool(result['result'].get('error')) for t in completed
                                     for turn in t['turns'] for result in turn['tool_results']),
                  'source_atlas_sha256': sha(mode.atlas_path), 'rendered_atlas_sha256': sha(output / 'reference_atlas.png'),
                  'case_wall_seconds': sum(t['seconds'] for t in completed),
                  'input_tokens': sum(turn['model_response']['usage']['input_tokens'] for t in completed for turn in t['turns']),
                  'output_tokens': sum(turn['model_response']['usage']['completion_tokens'] for t in completed for turn in t['turns']),
                  'kernel_active_seconds': None, 'currency': None,
                  'peak_allocated_bytes': peak_allocated, 'peak_reserved_bytes': peak_reserved,
                  'cost_quality': 'native trace timing/usage; kernel time and currency not measured', 'finished_at': time.time()}
        write(output / 'complete.json', result)
        write(output / 'complete-seal.json', complete_seal(output, binding))
        print(json.dumps(result))


if __name__ == '__main__':
    try: main()
    except Exception as error:
        print(json.dumps({'operation': 'simple-harness', 'status': 'BLOCKED', 'error_type': type(error).__name__}))
        raise SystemExit(1)
