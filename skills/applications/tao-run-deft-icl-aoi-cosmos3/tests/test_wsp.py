# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only WSP tests. All tensor payloads and predictions here are synthetic."""
import copy
import hashlib
import json
import math
import os
import socket
import struct
import sys
import threading
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from tht_contract import CLASSES, external_evaluator, json_hash, sha256
from wsp_boundary import Boundary, Journal, LimitError
from wsp_client import call
from wsp_isolation import connect_target
from wsp_prepare import BUDGETS, grouped_split, prepare_views
from wsp_runtime import CapabilityServer, approval_gate, sandbox_mounts
from wsp_weights import (apply_patch, catalog, clone, finite_tensor, header, inventory, pack,
                         read_values, strict_json, unpack, verify_inventory)
from wsp_usage import RESPONSE_POLICY, actual_usage, response_completion, validate_live_proof


def write_tensors(root, tensors=None):
    root.mkdir(parents=True, exist_ok=True)
    tensors = tensors or {'layer.weight': ('F32', [2, 3], [1, 2, 3, 4, 5, 6]),
                          'norm.weight': ('BF16', [2], [1, 2])}
    metadata, body = {}, b''
    for name, (dtype, shape, values) in tensors.items():
        raw = b''.join(pack(dtype, value) for value in values)
        metadata[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [len(body), len(body) + len(raw)]}
        body += raw
    raw = json.dumps(metadata, separators=(',', ':')).encode()
    raw += b' ' * ((-len(raw)) % 8)
    path = root / 'model.safetensors'
    path.write_bytes(struct.pack('<Q', len(raw)) + raw + body)
    return path


def patch_for(manifest):
    return {'schema': 'wsp-patch/1', 'base_fingerprint': manifest['fingerprint'],
            'hypothesis': 'Synthetic boundary test only, no scientific hypothesis.',
            'edits': [{'tensor': 'layer.weight', 'indices': [0, 4], 'values': [2.5, 3.0]}]}


def mining_row(n, angle=1, label=None):
    identity = f'SCN_scan{n}_MBO_component{n}_A{angle}.png'
    return {'id': identity, 'image': f'mining/images/{identity}',
            'image_sha256': hashlib.sha256(identity.encode()).hexdigest(),
            'labels': [label or CLASSES[n % len(CLASSES)]],
            'object_name': f'MBO_component{n}', 'scan_id': f'SCN_scan{n}', 'angle': str(angle)}


EXCLUSIONS = [{'object_name': f'MBO_component{n}', 'normalized_scan_id': f'scan{n}'} for n in (0, 1)]


class SyntheticEvaluator:
    """Only used for control-flow unit tests, never a runtime scoring fallback."""
    @staticmethod
    def parse(raw):
        try:
            value = json.loads(raw)
            if set(value) == {'labels'} and value['labels'] in [[c] for c in CLASSES]:
                return value['labels'][0]
        except (ValueError, TypeError):
            pass
        return 'INVALID'

    @staticmethod
    def score(rows):
        return {'synthetic_only': True, 'n': len(rows),
                'correct': sum(r['truth'] == r['prediction'] for r in rows),
                'invalid': sum(r['prediction'] == 'INVALID' for r in rows)}


def synthetic_boundary(tmp_path, *, evaluator=None, budget_changes=None, now=None):
    root = tmp_path / 'run'
    base = root / 'base'
    write_tensors(base)
    manifest = inventory(base)
    views = tmp_path / 'views'
    for role in ('development', 'holdout'):
        directory = views / role
        directory.mkdir(parents=True)
        (directory / 'queries.jsonl').write_text(json.dumps({'id': role + '-synthetic', 'image': 'image.png', 'image_sha256': '0' * 64}) + '\n')
        (directory / 'gold.jsonl').write_text(json.dumps({'id': role + '-synthetic', 'labels': ['Good']}) + '\n')
    task = {'budgets': {**BUDGETS, **(budget_changes or {})}, 'split_sha256': 'f' * 64,
            'inference': {'max_new_tokens': 4096, 'do_sample': False}, 'synthetic_only': True}
    calls = []
    def forward(model, role, directory, deadline):
        calls.append((str(model), role, str(directory)))
        return {'complete': True, 'profile_sha256': json_hash(task['inference']),
                'runtime_profile_sha256': 'a' * 64, 'gpu_wall_seconds': 0.0,
                'raw': [{'id': role + '-synthetic', 'raw_answer': '{"labels":["Good"]}', 'completed': True,
                         'inference_input_tokens': 10, 'inference_output_tokens': 6}]}
    kwargs = {'now': now} if now else {}
    boundary = Boundary(root, task, base, manifest, views, forward, evaluator or SyntheticEvaluator(), **kwargs)
    return boundary, calls


@pytest.fixture
def boundary(tmp_path):
    item, calls = synthetic_boundary(tmp_path)
    yield item, calls
    item.close()


def submit_candidate(boundary):
    return boundary.handle({'op': 'candidate', 'patch': patch_for(boundary.base_manifest), 'code': '# Synthetic construction\n'})


def freeze_valid(boundary):
    assert boundary.handle({'op': 'forward', 'candidate': 'base'})['ok']
    candidate = submit_candidate(boundary)['result']['candidate']
    assert boundary.handle({'op': 'forward', 'candidate': candidate})['ok']
    assert boundary.handle({'op': 'freeze', 'candidate': candidate, 'rationale': 'Synthetic development evidence.'})['ok']
    return candidate


def test_tensor_inspection_and_exact_replay(tmp_path):
    base = tmp_path / 'base'
    write_tensors(base)
    original = inventory(base)
    tensors = catalog(base)['tensors']
    assert tensors['layer.weight']['shape'] == [2, 3]
    assert read_values(base, tensors, 'layer.weight', [0, 4])['values'] == [1, 5]
    patch = patch_for(original)
    a = apply_patch(base, tmp_path / 'a', original, patch, max_tensors=2, max_elements=10)
    b = apply_patch(base, tmp_path / 'b', original, patch, max_tensors=2, max_elements=10)
    assert a == b
    assert a['changed_tensors'] == 1 and a['changed_elements'] == 2
    assert a['delta_norms'] == {'l1': 3.5, 'l2': 2.5, 'linf': 2.0}
    assert inventory(base) == original
    assert a['manifest']['fingerprint'] != original['fingerprint']
    assert (base / 'model.safetensors').stat().st_ino != (tmp_path / 'a/model.safetensors').stat().st_ino
    assert read_values(tmp_path / 'a', tensors, 'layer.weight', [0, 4])['values'] == [2.5, 3.0]
    assert a['loadability'] == 'pending_native_development_forward'


@pytest.mark.parametrize('mutation', ['wrong_lineage', 'zero', 'nan', 'inf', 'duplicate_index', 'duplicate_tensor',
                                      'too_many_elements', 'too_many_tensors', 'bad_index', 'negative_index',
                                      'bool_index', 'code_field', 'wrong_tensor', 'dtype_overflow', 'empty'])
def test_patch_rejections_leave_base_unchanged(tmp_path, mutation):
    base = tmp_path / 'base'
    write_tensors(base)
    manifest = inventory(base)
    patch = patch_for(manifest)
    e = patch['edits'][0]
    max_tensors, max_elements = 2, 10
    if mutation == 'wrong_lineage': patch['base_fingerprint'] = 'bad'
    elif mutation == 'zero': e['values'][0] = 1.0
    elif mutation == 'nan': e['values'][0] = float('nan')
    elif mutation == 'inf': e['values'][0] = float('inf')
    elif mutation == 'duplicate_index': e['indices'] = [0, 0]
    elif mutation == 'duplicate_tensor': patch['edits'].append(copy.deepcopy(e))
    elif mutation == 'too_many_elements': max_elements = 1
    elif mutation == 'too_many_tensors': max_tensors = 0
    elif mutation == 'bad_index': e['indices'][0] = 99
    elif mutation == 'negative_index': e['indices'][0] = -1
    elif mutation == 'bool_index': e['indices'][0] = True
    elif mutation == 'code_field': patch['code'] = 'anything'
    elif mutation == 'wrong_tensor': e['tensor'] = '../../holdout'
    elif mutation == 'dtype_overflow': e['values'][0] = 1e100
    elif mutation == 'empty': patch['edits'] = []
    with pytest.raises(ValueError):
        apply_patch(base, tmp_path / 'edited', manifest, patch, max_tensors=max_tensors, max_elements=max_elements)
    assert inventory(base) == manifest


def test_bf16_rounding_and_zero_after_cast(tmp_path):
    assert unpack('BF16', pack('BF16', 1.00390625)) == 1.0
    assert unpack('BF16', pack('BF16', 1.01171875)) == 1.015625
    base = tmp_path / 'base'
    write_tensors(base)
    manifest = inventory(base)
    patch = patch_for(manifest)
    patch['edits'] = [{'tensor': 'norm.weight', 'indices': [0], 'values': [1.001]}]
    with pytest.raises(ValueError, match='rounded-away'):
        apply_patch(base, tmp_path / 'edit', manifest, patch, max_tensors=1, max_elements=2)


def test_nonfinite_untouched_part_of_edited_tensor_rejected(tmp_path):
    base = tmp_path / 'base'
    path = write_tensors(base)
    meta = catalog(base)['tensors']['layer.weight']
    with path.open('r+b') as stream:
        stream.seek(meta['offset'] + 8)
        stream.write(struct.pack('<f', float('nan')))
    manifest = inventory(base)
    with pytest.raises(ValueError, match='nonfinite tensor'):
        apply_patch(base, tmp_path / 'edit', manifest, patch_for(manifest), max_tensors=1, max_elements=2)


def test_header_and_path_negative_cases(tmp_path):
    base = tmp_path / 'base'
    path = write_tensors(base)
    original = path.read_bytes()
    assert header(path)['header_bytes'] < len(original)
    path.write_bytes(struct.pack('<Q', 2**63))
    with pytest.raises(ValueError): header(path)
    path.write_bytes(original[:-1])
    with pytest.raises(ValueError): header(path)
    path.write_bytes(original)
    (base / 'escape.safetensors').symlink_to(path)
    with pytest.raises(ValueError): catalog(base)
    (base / 'escape.safetensors').unlink()
    (base / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'layer.weight': '../outside.safetensors'}}))
    with pytest.raises(ValueError): catalog(base)
    with pytest.raises(ValueError): strict_json('{"x":1,"x":2}')
    with pytest.raises(ValueError): strict_json('{"x":NaN}')


def test_clone_rejects_inplace_and_symlink_parent(tmp_path):
    base = tmp_path / 'base'
    write_tensors(base)
    manifest = inventory(base)
    with pytest.raises(ValueError): clone(base, base, manifest)
    alias = tmp_path / 'alias'
    alias.symlink_to(base, target_is_directory=True)
    with pytest.raises(ValueError): clone(base, alias / 'child', manifest)


def test_grouped_split_deterministic_disjoint_excluded():
    rows = [mining_row(n, a) for n in range(40) for a in (1, 2)]
    first, summary = grouped_split(rows, EXCLUSIONS, development_groups=8, holdout_groups=8)
    second, summary2 = grouped_split(list(reversed(rows)), list(reversed(EXCLUSIONS)), development_groups=8, holdout_groups=8)
    assert first == second
    assert summary['roles'] == summary2['roles']
    dev = {(r['object_name'], r['scan_id']) for r in first['development']}
    hold = {(r['object_name'], r['scan_id']) for r in first['holdout']}
    assert not dev & hold
    assert not {f'MBO_component{x}' for x in (0, 1)} & {r['object_name'] for rows_ in first.values() for r in rows_}
    assert summary['excluded_images'] == 4
    assert summary['unused_eligible_images'] == 44
    # Labels cannot affect selection/rank (they remain recorded as gold only).
    relabeled = [{**r, 'labels': ['Good']} for r in rows]
    third, _ = grouped_split(relabeled, EXCLUSIONS, development_groups=8, holdout_groups=8)
    assert {role: [r['id'] for r in items] for role, items in first.items()} == {role: [r['id'] for r in items] for role, items in third.items()}


def test_duplicate_image_components_and_forbidden_projections():
    rows = [mining_row(n) for n in range(20)]
    rows[2]['image_sha256'] = rows[0]['image_sha256']
    selected, info = grouped_split(rows, EXCLUSIONS, development_groups=4, holdout_groups=4)
    assert info['excluded_images'] == 3
    assert rows[2] not in selected['development'] + selected['holdout']
    with pytest.raises(ValueError): grouped_split(rows, [{**EXCLUSIONS[0], 'labels': ['Good']}, EXCLUSIONS[1]], development_groups=4, holdout_groups=4)
    with pytest.raises(ValueError): grouped_split(rows + [rows[0]], EXCLUSIONS)
    bad = copy.deepcopy(rows); bad[3]['scan_id'] = 'incorrect'
    with pytest.raises(ValueError): grouped_split(bad, EXCLUSIONS)


def test_real_prepare_views_uses_only_mining(tmp_path):
    from PIL import Image
    public = tmp_path / 'public'
    images = public / 'mining/images'
    images.mkdir(parents=True)
    rows = [mining_row(n) for n in range(12)]
    for n, row in enumerate(rows):
        image = public / row['image']
        Image.new('RGB', (13, 21), (n, 2*n, 3*n)).save(image)
        row['image_sha256'] = sha256(image)
    (public / 'mining/samples.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    exclusions = tmp_path / 'exclusions.json'
    exclusions.write_text(json.dumps(EXCLUSIONS))
    output = tmp_path / 'views'
    result = prepare_views(public, exclusions, output, development_groups=3, holdout_groups=3)
    assert result['roles']['development']['images'] == 3
    assert not (public / 'proxy').exists() and not (public / 'benchmark').exists()
    assert set(p.name for p in output.iterdir()) == {'development', 'holdout', 'split.json'}
    assert not (output / 'development/holdout').exists()
    with pytest.raises(ValueError): prepare_views(public, exclusions, output, development_groups=3, holdout_groups=3)
    for path in (output / 'development').glob('*.png'):
        with Image.open(path) as image:
            assert image.size == (256, 256) and image.mode == 'RGB'


def test_end_to_end_synthetic_boundary_and_single_holdout(boundary):
    boundary, calls = boundary
    candidate = freeze_valid(boundary)
    assert len(calls) == 2
    assert all(call_[1] == 'development' for call_ in calls)
    assert not boundary.handle({'op': 'status'})['ok']
    assert not boundary.handle({'op': 'forward', 'candidate': candidate})['ok']
    assert not boundary.handle({'op': 'holdout'})['ok']
    result = boundary.holdout()
    assert len(calls) == 4
    assert [c[1] for c in calls] == ['development', 'development', 'holdout', 'holdout']
    assert result['before']['metrics']['correct'] == result['after']['metrics']['correct'] == 1
    with pytest.raises(LimitError): boundary.holdout()
    export = boundary.export()
    assert export['status'] == 'COMPLETE'
    assert export['budget_reservations']['forward_images'] == 4
    assert export['budget_reservations']['forward_output_tokens'] == 4 * 4096
    assert export['known_complete_forward_usage']['output_tokens'] == 24
    assert export['prices_usd']['agent'] is None
    assert export['agent_usage'] is None


def test_failed_candidates_spend_cap_and_remain_on_disk(boundary):
    boundary, _ = boundary
    patch = patch_for(boundary.base_manifest)
    patch['base_fingerprint'] = 'wrong'
    for _ in range(3):
        result = boundary.handle({'op': 'candidate', 'patch': patch, 'code': '# wrong lineage\n'})
        assert not result['ok']
    assert not submit_candidate(boundary)['ok']
    assert boundary.journal.state()['used']['candidate_attempts'] == 3
    assert len(list((boundary.root / 'candidates').glob('*/failure.json'))) == 3
    assert inventory(boundary.base) == boundary.base_manifest


def test_failed_forward_spends_all_reservations_and_never_retries(boundary):
    boundary, calls = boundary
    def fail(*args):
        calls.append('failed')
        raise RuntimeError('synthetic engine error')
    boundary.forward = fail
    assert not boundary.handle({'op': 'forward', 'candidate': 'base'})['ok']
    state = boundary.journal.state()
    assert state['used']['development_requests'] == 1
    assert state['used']['forward_output_tokens'] == 4096
    assert state['used']['gpu_wall_seconds'] == 600
    assert not boundary.handle({'op': 'forward', 'candidate': 'base'})['ok']
    assert calls == ['failed']
    assert boundary.export()['failed_forward_costs']['development:base']['incomplete_request_output_tokens'] is None


def test_invalid_output_is_complete_and_immutable(boundary):
    boundary, _ = boundary
    original = boundary.forward
    def invalid(*args):
        result = original(*args)
        result['raw'][0]['raw_answer'] = 'not a JSON answer'
        result['raw'][0]['inference_output_tokens'] = 4096
        result['raw'][0]['finish_reason'] = 'length'
        return result
    boundary.forward = invalid
    result = boundary.handle({'op': 'forward', 'candidate': 'base'})
    assert result['ok'] and result['result']['metrics']['invalid'] == 1
    assert not boundary.handle({'op': 'forward', 'candidate': 'base'})['ok']


def test_freeze_requires_loadability_and_base_feedback(boundary):
    boundary, _ = boundary
    candidate = submit_candidate(boundary)['result']['candidate']
    freeze = {'op': 'freeze', 'candidate': candidate, 'rationale': 'synthetic'}
    assert not boundary.handle(freeze)['ok']
    assert boundary.handle({'op': 'forward', 'candidate': candidate})['ok']
    assert not boundary.handle(freeze)['ok']
    assert boundary.handle({'op': 'forward', 'candidate': 'base'})['ok']
    assert boundary.handle(freeze)['ok']
    assert not boundary.handle(freeze)['ok']


def test_candidate_tamper_cannot_freeze(boundary):
    boundary, _ = boundary
    assert boundary.handle({'op': 'forward', 'candidate': 'base'})['ok']
    candidate = submit_candidate(boundary)['result']['candidate']
    assert boundary.handle({'op': 'forward', 'candidate': candidate})['ok']
    model = Path(boundary.journal.state()['candidates'][candidate]['model'])
    path = model / 'model.safetensors'
    path.chmod(0o600)
    with path.open('ab') as stream: stream.write(b'x')
    assert not boundary.handle({'op': 'freeze', 'candidate': candidate, 'rationale': 'tampered'})['ok']


@pytest.mark.parametrize('tool_request', [
    {'op': 'holdout'}, {'op': 'forward', 'candidate': 'base', 'prompt': 'changed'},
    {'op': 'forward', 'candidate': '../holdout'}, {'op': 'values', 'tensor': '/etc/passwd', 'indices': [0]},
    {'op': 'metadata', 'offset': 0, 'limit': 100000}, {'op': 'exec', 'code': 'import torch'},
    {'op': 'forward', 'candidate': 'base', 'images': ['../holdout/image.png']},
])
def test_narrow_tool_schema_denies_paths_and_holdout(boundary, tool_request):
    boundary, calls = boundary
    result = boundary.handle(tool_request)
    assert not result['ok'] and 'holdout-synthetic' not in json.dumps(result)
    assert not calls


def test_native_profile_change_is_failure(boundary):
    boundary, _ = boundary
    assert boundary.handle({'op': 'forward', 'candidate': 'base'})['ok']
    candidate = submit_candidate(boundary)['result']['candidate']
    original = boundary.forward
    def changed(*args):
        result = original(*args); result['runtime_profile_sha256'] = 'b' * 64; return result
    boundary.forward = changed
    assert not boundary.handle({'op': 'forward', 'candidate': candidate})['ok']
    assert not boundary.journal.state()['candidates'][candidate].get('development_complete')


def test_resume_keeps_caps_freeze_and_inflight_reservations(tmp_path):
    item, _ = synthetic_boundary(tmp_path)
    item.journal.reserve('synthetic_crash_reservation', {'candidate_attempts': 2}, {})
    state = item.journal.state()
    task, root = item.task, item.root
    item.close()
    reopened = Journal(root, task)
    assert reopened.state() == state
    with pytest.raises(LimitError): reopened.reserve('too_many', {'candidate_attempts': 2}, {})
    with pytest.raises(ValueError): Journal(root, task)
    reopened.close()
    changed = copy.deepcopy(task); changed['budgets']['candidate_attempts'] += 1
    with pytest.raises(ValueError, match='reset'): Journal(root, changed)


def test_wall_cap_and_token_snapshots_no_reset(tmp_path):
    clock = [100.0]
    item, calls = synthetic_boundary(tmp_path, now=lambda: clock[0], budget_changes={'experiment_wall_seconds': 5})
    try:
        item.record_agent_usage({'input_tokens': 100, 'output_tokens': 7}, 'synthetic')
        item.record_agent_usage({'input_tokens': 110, 'output_tokens': 9}, 'synthetic')
        with pytest.raises(ValueError): item.record_agent_usage({'input_tokens': 1, 'output_tokens': 1}, 'synthetic')
        assert item.export()['agent_usage']['input_tokens'] == 110
        clock[0] = 105
        assert not item.handle({'op': 'forward', 'candidate': 'base'})['ok']
        assert not calls
    finally:
        item.close()


def test_holdout_failure_is_terminal_and_private(boundary):
    boundary, _ = boundary
    freeze_valid(boundary)
    original = boundary.forward
    def broken(model, role, *rest):
        if role == 'holdout': raise RuntimeError('synthetic private failure')
        return original(model, role, *rest)
    boundary.forward = broken
    with pytest.raises(RuntimeError): boundary.holdout()
    assert boundary.journal.state()['phase'] == 'ERROR'
    with pytest.raises(LimitError): boundary.holdout()
    result = boundary.handle({'op': 'status'})
    assert not result['ok'] and 'private' not in json.dumps(result)


def test_unix_capability_protocol(boundary, tmp_path):
    boundary, _ = boundary
    path = tmp_path / 'wsp.sock'
    server = CapabilityServer(path, boundary)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        assert call({'op': 'metadata', 'offset': 0, 'limit': 2}, str(path))['ok']
        assert not call({'op': 'holdout'}, str(path))['ok']
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(str(path))
            connection.sendall(b'{"op":"status","op":"holdout"}\n')
            assert not json.loads(connection.recv(1024))['ok']
    finally:
        server.shutdown(); server.server_close(); worker.join()


@pytest.mark.parametrize('authority', ['127.0.0.1:443', '169.254.169.254:80', 'chatgpt.com:80',
                                       'chatgpt.com.evil.test:443', 'api.openai.com@host:443', 'host.docker.internal:443'])
def test_provider_relay_denies_nonprovider_targets(authority):
    with pytest.raises(ValueError): connect_target(f'CONNECT {authority} HTTP/1.1\r\n\r\n'.encode(), ['chatgpt.com', 'api.openai.com'])
    assert connect_target(b'CONNECT chatgpt.com:443 HTTP/1.1\r\n\r\n', ['chatgpt.com']) == ('chatgpt.com', 443)


def test_approval_and_separate_release_required(tmp_path):
    p = {'preflight_sha256': 'a' * 64}
    approval, release = tmp_path / 'approval.json', tmp_path / 'release.json'
    base = {'preflight_sha256': p['preflight_sha256'], 'approved_by': 'synthetic test fixture', 'approved_at': '2026-01-01T00:00:00Z'}
    approval.write_text(json.dumps({**base, 'approved_for_real_weights': True}))
    release.write_text(json.dumps({**base, 'primary_gpu_released': False, 'host': socket.gethostname(), 'gpu_ids': ['0']}))
    with pytest.raises(ValueError): approval_gate(p, approval, release)
    release.write_text(json.dumps({**base, 'primary_gpu_released': True, 'host': socket.gethostname(), 'gpu_ids': ['0']}))
    assert approval_gate(p, approval, release)['approval_sha256'] == sha256(approval)
    with pytest.raises(ValueError): approval_gate({'preflight_sha256': 'b' * 64}, approval, release)


def test_pinned_external_evaluator_on_synthetic_boundary(tmp_path):
    directory = os.environ.get('THT_EVALUATOR_DIR')
    if not directory:
        pytest.skip('set THT_EVALUATOR_DIR for frozen external scorer parity')
    evaluator = external_evaluator(directory)
    item, _ = synthetic_boundary(tmp_path, evaluator=evaluator)
    try:
        freeze_valid(item)
        comparison = item.holdout()
        assert comparison['before']['metrics'] == comparison['after']['metrics']
        assert item.export()['status'] == 'COMPLETE'
    finally:
        item.close()


def usage_event(inputs, outputs, cached, source='one_fresh_session'):
    return {'_source_file': source, 'timestamp': '2026-11-05T00:00:00Z', 'payload': {
        'type': 'token_count', 'info': {'total_token_usage': {
            'input_tokens': inputs, 'output_tokens': outputs, 'cached_input_tokens': cached}}}}


def test_actual_cache_accounting_is_not_uncached_or_repeated_sum():
    events = [usage_event(1000, 20, 900), usage_event(1000, 20, 900), usage_event(2000, 40, 1900)]
    result = actual_usage(events)
    assert result['actual_input_plus_output_tokens'] == 2040
    assert result['distinct_snapshots'] == 2
    assert result['cached_input_tokens'] == 1900
    assert actual_usage([])['actual_input_plus_output_tokens'] is None


@pytest.mark.parametrize('bad', [usage_event(999, 30, 900), usage_event(2000, 10, 1800),
                                usage_event(2000, 40, 2001), usage_event(2000, 40, 1800, 'second_session')])
def test_actual_usage_rejects_resets_cache_double_counting_and_second_session(bad):
    with pytest.raises(ValueError): actual_usage([usage_event(1000, 20, 900), bad])


def test_actual_cached_tokens_close_tools_at_cap(boundary):
    item, calls = boundary
    item.record_agent_usage(actual_usage([usage_event(199999, 1, 199000)]), 'synthetic raw fixture')
    assert not item.handle({'op': 'forward', 'candidate': 'base'})['ok']
    assert not calls


def test_missing_live_cached_proof_cannot_open_real_gate(tmp_path):
    proof = tmp_path / 'proof.json'
    proof.write_text(json.dumps({'schema': 'wsp-live-actual-usage-proof/1', 'synthetic': True, 'case': 'tiny_100_token_stop'}))
    task = {'editor': {'model': 'gpt-6-astra', 'effort': 'xhigh', 'codex_version': 'codex-cli 0.166.0'}, 'budgets': BUDGETS}
    with pytest.raises(ValueError): validate_live_proof(proof, {}, task)
    with pytest.raises(ValueError): validate_live_proof(proof, {'actual_usage_proof_sha256': sha256(proof)}, task)


def test_real_holdout_gate_requires_final_known_actual_usage(boundary):
    item, _ = boundary
    freeze_valid(item)
    item.task['synthetic_only'] = False
    with pytest.raises(LimitError, match='final actual editor usage'):
        item.holdout()
    assert item.journal.state()['phase'] == 'FROZEN'


def test_header_only_catalog_never_reads_tensor_payload(tmp_path, monkeypatch):
    base = tmp_path / 'base'
    path = write_tensors(base)
    allowed = header(path)['header_bytes']
    original_open = Path.open
    reads = []
    class Reader:
        def __init__(self, stream): self.stream = stream
        def __enter__(self): return self
        def __exit__(self, *args): self.stream.close()
        def read(self, count=-1):
            assert count >= 0 and self.stream.tell() + count <= allowed
            reads.append(count)
            return self.stream.read(count)
    def guarded_open(p, *args, **kwargs):
        stream = original_open(p, *args, **kwargs)
        return Reader(stream) if p == path else stream
    monkeypatch.setattr(Path, 'open', guarded_open)
    assert catalog(base)['tensors']['layer.weight']['elements'] == 6
    assert sum(reads) == allowed


def test_export_schema_complete_and_incomplete(boundary):
    import subprocess
    # Reuse an installed system helper when the existing test venv omits it;
    # never install a dependency or change the shared experiment environment.
    validator = sys._base_executable
    probe = subprocess.run([validator, '-c', 'import jsonschema'], capture_output=True)
    if probe.returncode:
        pytest.skip('no existing system JSON Schema validator')
    def validate(document, schema):
        subprocess.run([validator, '-c', 'import json,sys,jsonschema; x=json.load(sys.stdin); jsonschema.validate(x["document"],x["schema"])'],
                       input=json.dumps({'document': document, 'schema': schema}), text=True, capture_output=True, check=True)
    item, _ = boundary
    schema = json.loads((SCRIPTS.parent / 'references/wsp-export.schema.json').read_text())
    validate(item.export(), schema)
    freeze_valid(item)
    item.holdout()
    result = item.export()
    validate(result, schema)
    assert result['paired_behavior']['gains'] == result['paired_behavior']['regressions'] == 0
    assert result['holdout_metrics']['mcc_delta'] is None  # synthetic scorer has no MCC


def test_expired_native_deadline_does_not_start_model(tmp_path):
    import time
    from wsp_runtime import NativeForward
    class NoCalls:
        def submit(self, *args, **kwargs):
            raise AssertionError('must never submit')
    with pytest.raises(ValueError, match='deadline exhausted'):
        NativeForward({}, NoCalls())(tmp_path, 'development', tmp_path, time.time())


def response_trace(events, threshold, *, denied=False, exit_code=0, reason='normal'):
    """Synthetic native capture fixture; never a live admission trace."""
    import tempfile
    from wsp_fixtures import native_fixture
    with tempfile.TemporaryDirectory() as root:
        _, evidence = native_fixture(root, events, threshold, native_stop=denied, exit_code=exit_code,
                                     error='unrelated error' if reason == 'error' else None)
    return evidence


@pytest.mark.parametrize('final_total', [200000, 200027])
def test_frozen_exact_boundary_and_final_overrun_admit_only_trusted_holdout(boundary, final_total):
    item, calls = boundary
    freeze_valid(item)
    item.task['synthetic_only'] = False  # Exercise the actual production holdout gate.
    events = [usage_event(99990, 10, 80000), usage_event(final_total - 20, 20, 160000)]
    trace = response_trace(events, 200000, denied=True, exit_code=1, reason='response_threshold')
    completion = item.accept_editor_completion(events, trace, exit_code=1)
    assert completion['status'] == 'HOLDOUT_ELIGIBLE'
    assert completion['exit_code'] == 1  # Never invent a successful editor exit.
    assert completion['actual_overrun_tokens'] == final_total - 200000
    assert not item.handle({'op': 'status'})['ok']
    comparison = item.holdout()
    assert [row[1] for row in calls] == ['development', 'development', 'holdout', 'holdout']
    assert comparison['before']['metrics']['n'] == comparison['after']['metrics']['n'] == 1
    report = item.export()
    assert report['status'] == 'COMPLETE'
    assert report['editor_completion']['actual_overrun_tokens'] == final_total - 200000
    assert report['agent_usage']['actual_input_plus_output_tokens'] == final_total
    assert report['budget_reservations']['holdout_requests'] == 2


def test_normal_final_response_at_boundary_is_accepted_without_invented_next_attempt(boundary):
    item, calls = boundary
    freeze_valid(item)
    item.task['synthetic_only'] = False
    events = [usage_event(199999, 1, 100000)]
    completion = item.accept_editor_completion(events, response_trace(events, 200000), exit_code=0)
    assert completion['exit_reason'] == 'normal' and completion['additional_completed_responses_after_boundary'] == 0
    item.holdout()
    assert len(calls) == 4


def test_token_stop_without_freeze_is_incomplete(boundary):
    item, calls = boundary
    events = [usage_event(199999, 5, 100000)]
    completion = item.accept_editor_completion(events, response_trace(events, 200000, denied=True, exit_code=1, reason='response_threshold'), exit_code=1)
    assert completion['status'] == 'INCOMPLETE_NO_FREEZE'
    with pytest.raises(LimitError): item.holdout()
    assert not calls


@pytest.mark.parametrize('failure', ['extra_response', 'missing_snapshot', 'reset', 'unrelated_error',
                                     'invented_exit', 'no_native_stop', 'forced_wall', 'false_cached_accounting', 'missing_evidence'])
def test_response_completion_rejects_nonpolicy_failures(failure):
    events = [usage_event(100, 5, 60), usage_event(220, 10, 150)]
    trace = response_trace(events, 200, denied=True, exit_code=1)
    exit_code, forced = 1, None
    if failure == 'extra_response':
        events.append(usage_event(300, 20, 200)); trace = response_trace(events, 200, denied=True, exit_code=1)
    elif failure == 'missing_snapshot': events.pop(0)
    elif failure == 'reset': events[1] = usage_event(99, 10, 60)
    elif failure == 'unrelated_error': trace['native_events'][-1]['event']['error']['message'] = 'unrelated failure'
    elif failure == 'invented_exit': exit_code = 0
    elif failure == 'no_native_stop': trace['native_events'] = trace['native_events'][:2]
    elif failure == 'forced_wall': forced = 'wall_limit'
    elif failure == 'false_cached_accounting': events[1]['payload']['info']['total_token_usage']['input_tokens'] = 70
    elif failure == 'missing_evidence': trace = {}
    with pytest.raises(ValueError): response_completion(events, trace, 200, frozen=True, exit_code=exit_code, forced_stop=forced)


def test_trusted_holdout_still_obeys_independent_budgets(boundary):
    item, _ = boundary
    freeze_valid(item)
    item.task['synthetic_only'] = False
    events = [usage_event(199999, 1, 0)]
    item.accept_editor_completion(events, response_trace(events, 200000), exit_code=0)
    item.journal.change('synthetic exhausted holdout reservation', {}, lambda state: state['used'].update(holdout_requests=2))
    with pytest.raises(LimitError, match='holdout_requests'): item.holdout()


def diagnostic_fixture(tmp_path):
    """Synthetic schema test input; never emitted as a live proof artifact."""
    from wsp_fixtures import native_fixture
    from wsp_usage import diagnostic_differences
    events = [usage_event(24990, 10, 10000), usage_event(50020, 10, 30000)]
    _, trace = native_fixture(tmp_path, events, 50000, native_stop=True, exit_code=1)
    invocation = json.loads((tmp_path / 'editor_invocation.json').read_text())
    identity = invocation['runtime_identity']
    task = {'editor': {'runtime_identity': identity, 'model': 'fixture-only', 'effort': 'xhigh',
                       'prompt': 'Different scientific prompt.'}, 'budgets': BUDGETS}
    proof = {'schema': 'wsp-live-actual-usage-proof/3', 'runtime': 'isolated_codex_exec', 'synthetic': False,
             'case': 'cached_multiresponse_actual_total', 'runtime_identity': identity, 'policy_sha256': json_hash(RESPONSE_POLICY),
             'scientific_threshold_tokens': 200000, 'diagnostic_threshold_tokens': 50000,
             'threshold_rationale': 'Synthetic validator test for the declared lower diagnostic threshold.',
             'diagnostic_differences': diagnostic_differences(task, invocation),
             'evidence_directory': str(tmp_path), 'native_evidence_sha256': json_hash(trace), 'observed_actual_usage': actual_usage(events)}
    path = tmp_path / 'fixture-proof.json'; path.write_text(json.dumps(proof))
    return task, proof, path, tmp_path / 'editor_response_evidence.json'


def test_lower_diagnostic_threshold_binds_runtime_and_charges_overrun(tmp_path):
    task, proof, path, _ = diagnostic_fixture(tmp_path)
    result = validate_live_proof(path, {'actual_usage_proof_sha256': sha256(path)}, task)
    assert result['diagnostic_threshold_tokens'] == 50000
    assert result['scientific_threshold_tokens'] == 200000
    assert result['completion']['actual_overrun_tokens'] == 30
    assert result['completion']['exit_code'] == 1
    assert result['usage']['actual_input_plus_output_tokens'] == 50030


@pytest.mark.parametrize('change', ['identity', 'under_threshold', 'no_native_stop', 'opaque_boolean', 'collector', 'threshold_rationale'])
def test_live_proof_rejects_unbound_or_non_native_evidence(tmp_path, change):
    task, proof, path, gate = diagnostic_fixture(tmp_path)
    if change == 'identity': proof['runtime_identity'] = {'model': 'different-runtime'}
    elif change == 'threshold_rationale': proof['threshold_rationale'] = ''
    elif change == 'under_threshold': proof['diagnostic_threshold_tokens'] = 60000
    else:
        trace = json.loads(gate.read_text())
        if change == 'no_native_stop': trace['native_events'] = trace['native_events'][:2]
        elif change == 'opaque_boolean': trace = {'actual_ceiling_enforced': True, 'followup_response_denied': True}
        elif change == 'collector': trace['collector_sha256'] = 'd' * 64
        gate.write_text(json.dumps(trace)); proof['native_evidence_sha256'] = json_hash(trace)
    path.write_text(json.dumps(proof))
    with pytest.raises(ValueError): validate_live_proof(path, {'actual_usage_proof_sha256': sha256(path)}, task)
