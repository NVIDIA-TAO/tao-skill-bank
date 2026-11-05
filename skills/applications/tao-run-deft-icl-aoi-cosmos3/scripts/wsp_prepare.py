#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trusted CPU-only WSP preparation; Mining and identity-only exclusions only."""
from __future__ import annotations

import collections
import hashlib
import json
import os
import shutil
from pathlib import Path

from tht_contract import CLASSES, EXTERNAL_HASHES, json_hash, load_jsonl, preprocess, sha256, within
from suite_data import identity_from_query
from suite_runtime import json_write
from wsp_weights import catalog, regular_file, strict_json
from wsp_usage import RESPONSE_POLICY, runtime_identity

PROMPT_SHA = 'c2b3b9aeb4c31b6b699dad32ea6f4c00081c7aadf2a8d65384f0bba1423326f6'
FRAMEWORK_IMAGE = 'nvcr.io/nvidia/tao/tao-toolkit@sha256:897c7cb2088e761bb48003029f1f5ed4fac32997388efe4d158e93dd8d36ac8d'
GENERIC_IMAGE = 'ubuntu@sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852'
BUDGETS = {
    'editor_agent_tokens': 200000, 'editor_wall_seconds': 2400,
    'experiment_wall_seconds': 5400, 'candidate_attempts': 3,
    'tensors_per_candidate': 4, 'elements_per_candidate': 65536,
    'edited_elements_total': 196608, 'inspection_requests': 64,
    'inspection_elements': 65536, 'elements_per_inspection': 1024,
    'development_requests': 4, 'holdout_requests': 2,
    'images_per_request': 32, 'forward_images': 192,
    'forward_output_tokens': 786432, 'max_new_tokens': 4096,
    'forward_wall_seconds': 600, 'gpu_wall_seconds': 3600,
    'storage_bytes': 80 * 1024**3, 'editor_writable_bytes': 256 * 1024**2,
    'tool_requests': 160, 'request_bytes': 4 * 1024**2,
    'editor_cpus': 2, 'editor_memory_bytes': 4 * 1024**3, 'editor_log_bytes': 64 * 1024**2,
}
EDITOR_PROMPT = '''This is one bounded Weight-Space Programming feasibility experiment on a prepared Cosmos3-Nano model for eight-class, single-image THT classification.
Use only the supplied development images/labels, model metadata/value tools and bounded development forward feedback. Identify one observable behavior, state a short testable hypothesis, and implement a replayable direct weight patch. You choose the behavior, hypothesis, tensors and values. No preferred change or expected improvement is supplied.
Do not use autograd, backward passes, optimizers, gradient estimation or an iterative fitting/training loop. No prompts, preprocessing, tokenizer, model configuration or generation settings may change. A prompt-only or generation-only proposal is not a weight edit. No real-model execution is available except the forward tool.
Tools: python3 /tools/wsp_client.py reads a JSON request on stdin. See /task/tool_contract.json for schemas, the exact frozen budgets and base fingerprint. Metadata and value inspection concern the original base. Each candidate is independently patched from that same base; failures consume budgets. Forward requests always use the complete fixed development view. Completed INVALID/length-limited outputs are final, not repaired or retried.
Keep the Python code you use to construct each patch in /task and provide its literal source in the candidate request. The trusted service executes only the declarative sparse assignments, never your Python source. Keep a brief development-evidence rationale. Freeze at most one validated, development-evaluated candidate. Freeze ends all editor access; no holdout feedback will be returned. A failed or inconclusive attempt is valid evidence; do not reset or extend any cap to seek improvement.
You have a fresh context, generic Python/image tools, this development view and the listed service capabilities. Do not seek other experiments, holdout data, host resources, credentials, external tools or other agents. Report limitations and actual work honestly.
'''


def joint(row):
    if not isinstance(row.get('object_name'), str) or not row['object_name'] or not isinstance(row.get('scan_id'), str) or not row['scan_id']:
        raise ValueError('Mining joint metadata required')
    return row['object_name'], row['scan_id'].removeprefix('SCN_')


def grouped_split(rows, exclusions, *, seed=42, development_groups=16, holdout_groups=16, image_cap=32):
    """Hash-ranked joint components; no class- or feedback-dependent selection.

    Joint groups with duplicate image bytes are joined, preventing a byte leak
    across roles. An exclusion removes its whole connected component.
    """
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('nonempty unique Mining identities required')
    if type(seed) is not int or min(development_groups, holdout_groups, image_cap) < 1:
        raise ValueError('invalid split parameters')
    if not isinstance(exclusions, list) or any(set(x) != {'object_name', 'normalized_scan_id'} for x in exclusions):
        raise ValueError('exclusions must be an identity-only projection')
    excluded = {(x['object_name'], x['normalized_scan_id'].removeprefix('SCN_')) for x in exclusions}
    if len(excluded) != 2 or len(exclusions) != 2:
        raise ValueError('exactly the two declared overlapping joints are required')
    parents = {}

    def find(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            key = parents[key]
        return key

    seen_bytes = {}
    for row in rows:
        if set(row) - {'id', 'image', 'image_sha256', 'labels', 'object_name', 'scan_id', 'angle', 'role'}:
            raise ValueError('private/unexpected Mining fields')
        if row.get('role', 'mining') != 'mining' or row['labels'] not in [[c] for c in CLASSES]:
            raise ValueError('eligible single-label Mining required')
        key = joint(row)
        identity = identity_from_query(row)
        if (identity['object_name'], identity['scan_id']) != key:
            raise ValueError('Mining metadata/identity grammar disagreement')
        if len(row['image_sha256']) != 64 or any(c not in '0123456789abcdef' for c in row['image_sha256']):
            raise ValueError('image hash required')
        find(key)
        previous = seen_bytes.setdefault(row['image_sha256'], key)
        a, b = find(key), find(previous)
        if a != b:
            parents[max(a, b)] = min(a, b)
    if not excluded <= set(parents):
        raise ValueError('declared exclusions must occur in Mining')
    groups = collections.defaultdict(list)
    for row in rows:
        groups[find(joint(row))].append(row)
    forbidden = {find(key) for key in excluded}
    eligible = [key for key in groups if key not in forbidden]
    ranked = sorted(eligible, key=lambda key: (json_hash({'seed': seed, 'joint_component': key}), key))
    if len(ranked) < development_groups + holdout_groups:
        raise ValueError('insufficient eligible joint groups')
    dev_keys = ranked[:development_groups]
    hold_keys = ranked[development_groups:development_groups + holdout_groups]
    selected = {}
    for role, keys in [('development', dev_keys), ('holdout', hold_keys)]:
        selected[role] = sorted((r for k in keys for r in groups[k]), key=lambda r: r['id'])
        if len(selected[role]) > image_cap:
            raise ValueError('fixed group draw exceeds image cap; do not redraw')
    selected_ids = {r['id'] for rows_ in selected.values() for r in rows_}
    summary = {'algorithm': 'sha256-ranked-normalized-joint-components-v1', 'seed': seed,
               'grouping': 'object_name + scan_id.removeprefix(SCN_); union duplicate image hashes',
               'stratified': False, 'source_count': len(rows), 'exclusions': exclusions,
               'excluded_images': sum(len(groups[k]) for k in forbidden),
               'eligible_images': sum(len(groups[k]) for k in eligible),
               'eligible_components': len(eligible),
               'unused_eligible_images': sum(len(groups[k]) for k in eligible) - len(selected_ids),
               'roles': {role: {'images': len(items), 'joints': len({joint(r) for r in items}),
                                'class_counts': dict(collections.Counter(r['labels'][0] for r in items)),
                                'ids_sha256': json_hash([r['id'] for r in items]),
                                'source_rows_sha256': json_hash(items)} for role, items in selected.items()}}
    return selected, summary


def write_jsonl(path, rows):
    path = Path(path)
    with path.open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def prepare_views(public, exclusion_file, output, *, seed=42, development_groups=16, holdout_groups=16):
    public, output = Path(public).resolve(strict=True), Path(output).absolute()
    # Never walk/query the main Proxy/Benchmark or private source rows.
    source = public / 'mining/samples.jsonl'
    rows = load_jsonl(source)
    exclusions = strict_json(Path(exclusion_file).read_bytes())
    selected, summary = grouped_split(rows, exclusions, seed=seed,
                                      development_groups=development_groups, holdout_groups=holdout_groups)
    if output.exists() or output.is_relative_to(public):
        raise ValueError('new separate WSP view directory required')
    output.mkdir(parents=True, mode=0o700)
    summary.update(source_manifest_sha256=sha256(source), exclusions_sha256=sha256(exclusion_file))
    for role, records in selected.items():
        directory = output / role
        directory.mkdir(mode=0o700)
        queries, gold, provenance = [], [], []
        for index, row in enumerate(records):
            if not row['image'].startswith('mining/images/'):
                raise ValueError('only Mining image paths accepted')
            image = within(public / 'mining', str(Path(row['image']).relative_to('mining')))
            if sha256(image) != row['image_sha256']:
                raise ValueError('Mining source image changed')
            name = f'image-{index:04d}.png'
            info = preprocess(image, directory / name)
            queries.append({'id': row['id'], 'image': name, 'image_sha256': info['view_sha256']})
            gold.append({'id': row['id'], 'labels': row['labels']})
            provenance.append({'id': row['id'], 'joint': joint(row), **info})
        write_jsonl(directory / 'queries.jsonl', queries)
        write_jsonl(directory / 'gold.jsonl', gold)
        write_jsonl(directory / 'provenance.jsonl', provenance)
        summary['roles'][role]['files'] = {p.name: sha256(p) for p in sorted(directory.iterdir())}
        summary['roles'][role]['view_sha256'] = json_hash(summary['roles'][role]['files'])
        for path in directory.iterdir():
            path.chmod(0o400)
    summary['split_sha256'] = json_hash(summary)
    json_write(output / 'split.json', summary)
    return summary


def proposed_preflight(*, model, views, direct_profile, evaluator, output, run_dir, seed=42):
    """No tensor payload reads, agents, model copies, containers or GPU calls."""
    model, views = Path(model).resolve(strict=True), Path(views).resolve(strict=True)
    provenance = strict_json(regular_file(model, 'tao_conversion_provenance.json').read_bytes())
    declared = provenance['prepared']
    if declared['model_type'] != 'qwen3_vl':
        raise ValueError('prepared Qwen3-VL Nano required')
    files = declared['files']
    if len({f['name'] for f in files}) != len(files):
        raise ValueError('duplicate prepared files')
    observed = catalog(model)
    for file in files:
        path = regular_file(model, file['name'])
        if path.stat().st_size != file['size']:
            raise ValueError('prepared file size mismatch')
        if path.suffix != '.safetensors' and sha256(path) != file['sha256']:
            raise ValueError('non-weight prepared file hash mismatch')
    if set(observed['files']) != {f['name'] for f in files if f['name'].endswith('.safetensors')}:
        raise ValueError('prepared shard inventory mismatch')
    profile = strict_json(Path(direct_profile).read_bytes())
    if profile['prompt_sha256'] != PROMPT_SHA or hashlib.sha256(profile['prompt'].encode()).hexdigest() != PROMPT_SHA:
        raise ValueError('frozen THT prompt mismatch')
    evaluator = Path(evaluator).resolve(strict=True)
    for name, digest in EXTERNAL_HASHES.items():
        if sha256(evaluator / name) != digest:
            raise ValueError('frozen external evaluator changed')
    split = strict_json((views / 'split.json').read_bytes())
    check = dict(split)
    split_hash = check.pop('split_sha256')
    if json_hash(check) != split_hash:
        raise ValueError('split record changed')
    for role, detail in split['roles'].items():
        for name, digest in detail['files'].items():
            if sha256(regular_file(views / role, name)) != digest:
                raise ValueError('prepared view changed')
    code = Path(__file__).parent
    code_files = ['wsp.py', 'wsp_prepare.py', 'wsp_weights.py', 'wsp_boundary.py', 'wsp_runtime.py',
                  'wsp_client.py', 'wsp_isolation.py', 'wsp_usage.py', 'tht_contract.py', 'inference_worker.py',
                  'suite_runtime.py', 'codex_capture.py', 'suite_costs.py', 'tool_evidence.py', 'astra_isolation.py']
    budgets = dict(BUDGETS)
    budgets['forward_images'] = (budgets['development_requests'] * split['roles']['development']['images']
                                 + budgets['holdout_requests'] * split['roles']['holdout']['images'])
    budgets['forward_output_tokens'] = budgets['forward_images'] * budgets['max_new_tokens']
    task = {'schema': 'wsp-preflight/1', 'status': 'PROPOSED_CPU_PREPARATION',
            'platform': 'docker', 'backend': 'cosmos-framework',
            'backend_reason': 'explicit existing Nano prepared checkpoint and M3 inference contract',
            'model': str(model), 'declared_base': declared,
            'conversion_provenance_sha256': sha256(model / 'tao_conversion_provenance.json'),
            'header_catalog': observed, 'full_base_hash_verification': 'required_after_approval',
            'views': str(views), 'split_sha256': split_hash,
            'view_hashes': {role: info['view_sha256'] for role, info in split['roles'].items()},
            'role_counts': {role: info['images'] for role, info in split['roles'].items()},
            'run_dir': str(Path(run_dir).absolute()), 'evaluator': str(evaluator),
            'evaluator_hashes': EXTERNAL_HASHES, 'seed': seed,
            'inference': {'prompt': profile['prompt'], 'prompt_sha256': PROMPT_SHA,
                          'preprocess': 'RGB/ImageOps.pad/256x256/NEAREST/PNG',
                          'do_sample': False, 'max_new_tokens': 4096, 'dtype': 'bfloat16', 'attention': 'sdpa'},
            'framework_image': FRAMEWORK_IMAGE, 'generic_image': GENERIC_IMAGE,
            'native_runtime': {'cpus': 4, 'memory_bytes': 64 * 1024**3, 'gpu_ids': ['0'], 'gpu_count': 1, 'network': 'none'},
            'editor': {'model': 'gpt-6-astra', 'effort': 'xhigh', 'codex_version': 'codex-cli 0.166.0',
                       'prompt': EDITOR_PROMPT, 'prompt_sha256': hashlib.sha256(EDITOR_PROMPT.encode()).hexdigest(),
                       'fresh_session': True, 'tools': 'generic Python/image + wsp-sparse-v1',
                       'network': 'none; optional provider-only CONNECT relay via Unix socket',
                       'api_hosts': ['chatgpt.com', 'api.openai.com'],
                       'token_cap_semantics': RESPONSE_POLICY['accounting'] + '; ' + RESPONSE_POLICY['inflight'],
                       'token_enforcement_status': 'UNVALIDATED: requires reviewed live cached-response admission diagnostic and trusted per-invocation chronology',
                       'token_enforcement_limit': 'response-boundary threshold, not a zero-overshoot spend ceiling; watchdog/CONNECT traffic cannot prove next-response denial'},
            'response_budget_policy': RESPONSE_POLICY,
            'budgets': budgets, 'retry_policy': 'none; failures and interrupted reservations consume caps',
            'budget_origin': {'kind': 'engineering proposal, not paper reproduction',
                              'source': 'coordination/M3_HANDOFF.md, small-sample mechanical observations',
                              'one_image_generation_seconds': 2.622, 'peak_allocated_GB': 17.72,
                              'arithmetic': f"{budgets['forward_images']} * 2.622 = {budgets['forward_images'] * 2.622:.3f} generation seconds, excluding loads/long outputs; reserve 6 * 600 = 3600 allocated GPU seconds",
                              'storage': 'prepared base plus 3 independent candidates approximately 65.4 GiB; 80 GiB ceiling includes retained logs/views/failed copies',
                              'uncertainty': 'not corpus throughput; 4096-token outputs can exhaust the caps'},
            'code_hashes': {name: sha256(code / name) for name in code_files},
            'stopping': ['any wall/token/storage cap', '3 candidate attempts', '4 development requests',
                         'one successful freeze closes editor tools', 'at most one base and one edited holdout request',
                         'no valid development-tested candidate means no holdout comparison'],
            'approval_required': ['coordinator review of this exact preflight', 'primary GPU release',
                                  'runtime access probe', 'full base hash/finiteness and independent-copy validation',
                                  'reviewed live cached-response admission diagnostic and trusted per-invocation chronology for isolated codex exec'],
            'prices': {'agent_usd': None, 'gpu_usd': None, 'reason': 'no price supplied'}}
    if sum(f['size'] for f in files) * 4 + 1024**3 > BUDGETS['storage_bytes']:
        raise ValueError('four retained model copies exceed storage proposal')
    codex_binary = shutil.which('codex')
    if codex_binary is None:
        raise ValueError('installed Codex binary required to seal diagnostic runtime identity')
    task['editor']['runtime_identity'] = runtime_identity(task, codex_binary)
    result = {'task': task, 'preflight_sha256': json_hash(task)}
    json_write(output, result)
    return result
