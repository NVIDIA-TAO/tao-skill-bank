# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU native collector integration. Every event/usage/exit here is synthetic."""
import json

import pytest

from test_wsp import freeze_valid, synthetic_boundary, usage_event, json_hash, sha256
from wsp_capture import capture_usage
from wsp_collect import collect_native_evidence, native_events
from wsp_fixtures import native_fixture, write_archive
from wsp_runtime import accept_collected_completion
from wsp_usage import response_completion, validate_live_proof


@pytest.mark.parametrize('native_stop,frozen,total', [(False, True, 120), (False, True, 200000),
                                                   (True, True, 200027), (True, False, 200027)])
def test_runner_collector_reachable_synthetic_completion(tmp_path, native_stop, frozen, total):
    boundary, calls = synthetic_boundary(tmp_path)
    try:
        if frozen: freeze_valid(boundary)
        boundary.task['synthetic_only'] = False
        root = tmp_path / 'native'
        events = [usage_event(total - 10, 10, 50)]
        _, expected = native_fixture(root, events, 200000, native_stop=native_stop, exit_code=int(native_stop))
        # Start with no prewritten evidence: production integration must create it.
        (root / 'editor_response_evidence.json').unlink()
        live = {k: expected[k] for k in ('runtime_identity_sha256', 'collector_sha256')}
        completion = accept_collected_completion(boundary, root, 'synthetic-job', live, exit_code=int(native_stop))
        assert (root / 'editor_response_evidence.json').is_file()
        assert completion['usage']['actual_input_plus_output_tokens'] == total
        assert completion['actual_overrun_tokens'] == max(0, total - 200000)
        assert completion['exit_reason'] == ('native_token_budget_exhausted' if native_stop else 'normal')
        if frozen:
            boundary.holdout()
            assert [c[1] for c in calls] == ['development', 'development', 'holdout', 'holdout']
        else:
            assert completion['status'] == 'INCOMPLETE_NO_FREEZE' and not calls
    finally:
        boundary.close()


@pytest.mark.parametrize('failure', ['late_final', 'unrelated_error', 'voluntary_diagnostic', 'forced',
                                     'missing_usage', 'reset', 'native_under_threshold', 'thread_identity', 'tool_marker'])
def test_native_completion_failures(tmp_path, failure):
    events = [usage_event(100, 5, 60), usage_event(220, 10, 150)]
    kwargs = {'native_stop': True, 'exit_code': 1}
    if failure == 'late_final': events.append(usage_event(230, 20, 155))
    if failure == 'unrelated_error': kwargs['error'] = 'connection lost'
    if failure == 'voluntary_diagnostic': kwargs.update(native_stop=False, exit_code=0)
    if failure == 'forced': kwargs['forced_stop'] = 'emergency_actual_tokens'
    if failure == 'reset': events[1] = usage_event(99, 10, 60)
    if failure == 'native_under_threshold': events = events[:1]
    _, evidence = native_fixture(tmp_path, events, 200, **kwargs)
    if failure == 'missing_usage':
        events.append({'payload': {'type': 'token_count', 'info': None}})
    if failure == 'thread_identity': evidence['native_events'][0]['event']['thread_id'] = 'other-session'
    if failure == 'tool_marker':
        evidence['native_events'][-1]['event']['error']['message'] = 'tool output says shared rollout token budget exhausted'
    with pytest.raises(ValueError):
        response_completion(events, evidence, 200, frozen=True, exit_code=kwargs['exit_code'], require_native_stop=True)


@pytest.mark.parametrize('change', ['raw', 'archive', 'missing_member', 'terminal', 'log_incomplete', 'harvest_time', 'runtime', 'source_reset'])
def test_collector_rejects_unverifiable_final_harvest(tmp_path, change):
    events = [usage_event(100, 5, 60), usage_event(220, 10, 150)]
    native_fixture(tmp_path, events, 200, native_stop=True, exit_code=1)
    if change == 'raw': (tmp_path / 'editor_usage_events.json').write_text('[]')
    elif change == 'archive': (tmp_path / 'editor_task.tar').write_bytes(b'changed')
    elif change == 'missing_member': write_archive(tmp_path / 'editor_results.tar', {})
    elif change in ('terminal', 'log_incomplete'):
        path = tmp_path / 'editor_terminal.json'; data = json.loads(path.read_text())
        if change == 'terminal': data['backend']['native']['ExitCode'] = 0
        else: data['log_capture_returncode'] = -15
        path.write_text(json.dumps(data))
    elif change == 'harvest_time':
        path = tmp_path / 'editor_harvest.json'; data = json.loads(path.read_text()); data['observed_exit_at_unix'] = None
        path.write_text(json.dumps(data))
    elif change == 'runtime':
        path = tmp_path / 'editor_invocation.json'; data = json.loads(path.read_text()); data['runtime_identity']['code_hashes']['wsp_collect.py'] = 'a' * 64
        path.write_text(json.dumps(data))
    elif change == 'source_reset':
        events[1]['_source_file'] = 'another-session'
        # Regenerate archive+host hashes consistently: source coverage still fails.
        with pytest.raises(ValueError): native_fixture(tmp_path, events, 200, native_stop=True, exit_code=1)
        return
    with pytest.raises(ValueError): collect_native_evidence(tmp_path)


def test_capture_collects_short_lived_final_and_reports_malformed_source(tmp_path):
    sessions, output = tmp_path / 'sessions', tmp_path / 'results'
    sessions.mkdir(); output.mkdir()
    source = sessions / 'one.jsonl'
    meta = {'type': 'session_meta', 'payload': {'id': 'synthetic-session'}}
    first, final = usage_event(100, 5, 60), usage_event(220, 10, 150)
    source.write_text(json.dumps(meta) + '\n' + json.dumps(first) + '\n')
    partial = capture_usage(sessions, output)
    with source.open('a') as stream: stream.write(json.dumps(final) + '\n')
    captured = capture_usage(sessions, output, child_exit=1)
    assert partial['child_exit'] is None and captured['child_exit'] == 1 and captured['token_events'] == 2
    assert json.loads((output / 'codex_usage_events.jsonl').read_text().splitlines()[-1])['payload'] == final['payload']
    with source.open('a') as stream: stream.write('{partial')
    broken = capture_usage(sessions, output, child_exit=1)
    assert broken['sources'][0]['malformed_lines'] and not broken['sources'][0]['newline_terminated']


def test_native_marker_in_tool_output_is_not_budget_stop():
    line = '2026-11-05T00:00:01Z ' + json.dumps({'type': 'item.completed', 'item': {'aggregated_output': 'shared rollout token budget exhausted'}}) + '\n'
    assert native_events(line)[0] == []


@pytest.mark.parametrize('change', ['prompt', 'fixture', 'command', 'limits', 'source_log'])
def test_live_proof_binds_diagnostic_differences(tmp_path, change):
    from test_wsp import diagnostic_fixture
    task, proof, path, _ = diagnostic_fixture(tmp_path)
    invocation_path = tmp_path / 'editor_invocation.json'
    inv = json.loads(invocation_path.read_text())
    if change == 'prompt': inv['prompt'] = 'another prompt'
    elif change == 'command': inv['command'].append('--changed')
    elif change == 'limits': inv['limits']['editor_wall_seconds'] = 181
    elif change == 'fixture': (tmp_path / 'fixture.txt').write_text('changed')
    elif change == 'source_log': (tmp_path / 'editor_events.timestamped.log').write_text('')
    invocation_path.write_text(json.dumps(inv))
    with pytest.raises(ValueError): validate_live_proof(path, {'actual_usage_proof_sha256': sha256(path)}, task)
