#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual token accounting, independent of a CLI's configured budget semantics."""
import hashlib
from pathlib import Path

from tht_contract import json_hash, sha256
from wsp_weights import strict_json


def actual_usage(events):
    """One fresh session, monotone cumulative totals; cache is a subset of input.

    Never subtract cached tokens, sum repeated snapshots, take maxima to hide a
    reset, or replace raw total_token_usage with an undocumented budget counter.
    """
    fields = ('input_tokens', 'output_tokens', 'cached_input_tokens')
    previous = None
    snapshots = 0
    sources = set()
    last_at = None
    for event in events:
        payload = event.get('payload', event)
        if payload.get('type') != 'token_count':
            continue
        usage = (payload.get('info') or {}).get('total_token_usage')
        if usage is None:
            continue
        if any(type(usage.get(k)) is not int or usage[k] < 0 for k in fields):
            raise ValueError('malformed actual token counters')
        if usage['cached_input_tokens'] > usage['input_tokens']:
            raise ValueError('cached input is not additional or subtractable input')
        if previous and any(usage[k] < previous[k] for k in fields):
            raise ValueError('cumulative reset; cannot reset a WSP budget')
        source = event.get('_source_file') or event.get('source_file')
        if source:
            sources.add(source)
        if len(sources) > 1:
            raise ValueError('fresh editor must have exactly one session')
        if previous is None or any(usage[k] != previous[k] for k in fields):
            snapshots += 1
        previous = {k: usage[k] for k in fields}
        last_at = event.get('timestamp')
    if previous is None:
        return {k: None for k in (*fields, 'actual_input_plus_output_tokens')}
    return {**previous, 'actual_input_plus_output_tokens': previous['input_tokens'] + previous['output_tokens'],
            'source': 'raw_session_total_token_usage', 'distinct_snapshots': snapshots, 'last_event_at': last_at,
            'cached_semantics': 'subset of input; no subtraction and no double counting'}


RESPONSE_POLICY = {
    'schema': 'wsp-response-budget/2',
    'accounting': 'cumulative raw input_tokens + output_tokens; cached input included once; no reset',
    'admission': 'native response-boundary stopping must be established by a separately reviewed live cached-response diagnostic',
    'inflight': 'charge and disclose the complete first threshold-crossing response; reject any later completed-response usage',
    'completion': 'valid freeze plus complete native final usage/exit/harvest permits holdout; token stop without freeze is incomplete',
    'failures': 'missing final usage/evidence, reset, later completed response, unrelated error, watchdog cancellation or another exceeded budget fails',
    'observability': 'native budget-stop and completed usage evidence, not per-request admission telemetry or a strict spend ceiling',
    'spend_ceiling': False,
    'diagnostic_variations': ['prompt', 'synthetic text fixture/tools', 'output directory', '50000 threshold and derived reminder',
                              '180s wall / 100000 observed emergency threshold / 10 tool requests / 128MiB retained evidence'],
}


def editor_command(task, threshold=None):
    """Shared actual stopping command; diagnostic prompt substitution is explicit."""
    from astra_isolation import token_cap_config
    budget = '<threshold-and-derived-reminder>' if threshold is None else token_cap_config(threshold)
    return ['/tools/wsp_isolation.py', 'editor-entry', '/usr/bin/python3', '/runtime/wsp_capture.py',
            '/usr/local/bin/codex', 'exec', '--ignore-user-config', '--ignore-rules', '--skip-git-repo-check',
            '--dangerously-bypass-approvals-and-sandbox', '--json', '--color', 'never',
            '-m', task['editor']['model'], '-c', 'model_reasoning_effort="' + task['editor']['effort'] + '"',
            '-c', 'developer_instructions=""', '-c', 'features.plugins=false', '-c', 'features.multi_agent=false',
            '-c', budget, '-C', '/task', '-o', '/results/final.txt', task['editor']['prompt']]


def runtime_identity(task, codex_binary):
    """Invariant stopping mechanism; invocation-specific inputs are bound separately."""
    binary = Path(codex_binary).resolve(strict=True)
    binaries = [binary, binary.parent / 'codex-code-mode-host', binary.parent / 'codex-exec-server', Path('/usr/bin/python3.12')]
    command = editor_command(task)
    command[-1] = '<invocation-prompt>'
    return {'schema': 'wsp-editor-runtime/2', 'model': task['editor']['model'], 'effort': task['editor']['effort'],
            'codex_version': task['editor']['codex_version'], 'image': task['generic_image'],
            'api_hosts': task['editor']['api_hosts'], 'command_template': command,
            'binary_files': {str(p): sha256(p) for p in binaries}, 'code_hashes': task['code_hashes'],
            'sandbox': {'cpus': 2, 'memory_bytes': 4 * 1024**3, 'network': 'none; provider-only CONNECT relay',
                        'credential_mount_target': '/codex/auth.json', 'gpu': False},
            'permitted_diagnostic_variations': RESPONSE_POLICY['diagnostic_variations']}


def invocation_manifest(task, job_id, output, *, kind='scientific', fixture_hashes=None):
    """Written before/at launch; binds the actual prompt, flags and fixture changes."""
    threshold = task['budgets']['editor_agent_tokens']
    return {'schema': 'wsp-editor-invocation/1', 'kind': kind, 'job_id': job_id,
            'runtime_identity': task['editor']['runtime_identity'], 'output_directory': str(Path(output).resolve()),
            'threshold_tokens': threshold, 'prompt': task['editor']['prompt'],
            'prompt_sha256': hashlib.sha256(task['editor']['prompt'].encode()).hexdigest(),
            'command': editor_command(task, threshold), 'fixture_hashes': fixture_hashes or {},
            'log_limit_bytes': task['budgets']['editor_log_bytes'], 'limits': task['budgets'],
            'model_context_sha256': task.get('model_context', {}).get('bundle_sha256')}


def diagnostic_differences(task, invocation):
    """Exact disclosed differences; a text diagnostic never equals a WSP prompt."""
    return {'scientific_prompt_sha256': hashlib.sha256(task['editor']['prompt'].encode()).hexdigest(),
            'diagnostic_prompt_sha256': invocation['prompt_sha256'],
            'scientific_model_context_sha256': task.get('model_context', {}).get('bundle_sha256'),
            'scientific_development_view_sha256': task.get('view_hashes', {}).get('development'),
            'diagnostic_fixture_hashes': invocation['fixture_hashes'],
            'scientific_output_directory': task.get('run_dir'), 'diagnostic_output_directory': invocation['output_directory'],
            'scientific_threshold_tokens': task['budgets']['editor_agent_tokens'],
            'diagnostic_threshold_tokens': invocation['threshold_tokens'], 'diagnostic_limits': invocation['limits'],
            'shared_runtime_identity_sha256': json_hash(task['editor']['runtime_identity'])}


def response_snapshots(events):
    actual_usage(events)
    result, previous = [], None
    for event in events:
        payload = event.get('payload', event)
        usage = (payload.get('info') or {}).get('total_token_usage') if payload.get('type') == 'token_count' else None
        if usage is None:
            continue
        key = tuple(usage[k] for k in ('input_tokens', 'output_tokens', 'cached_input_tokens'))
        if key != previous:
            result.append((json_hash(event), usage['input_tokens'] + usage['output_tokens']))
            previous = key
    return result


def response_completion(events, evidence, threshold, *, frozen, exit_code, forced_stop=None, require_native_stop=False):
    """Validate native observable completion; never fabricate request chronology."""
    from wsp_collect import NATIVE_BUDGET_ERROR, epoch
    usage = actual_usage(events)
    snapshots = response_snapshots(events)
    if not snapshots or usage['actual_input_plus_output_tokens'] is None:
        raise ValueError('complete actual usage is required')
    if not (events[-1].get('payload', events[-1]).get('info') or {}).get('total_token_usage'):
        raise ValueError('final raw usage is missing')
    if forced_stop is not None or evidence.get('forced_stop') is not None:
        raise ValueError('watchdog/forced stop is not native completion')
    if (evidence.get('schema') != 'wsp-native-evidence/1' or evidence.get('threshold_tokens') != threshold or
            evidence.get('raw_events_sha256') != json_hash(events) or evidence.get('exit_code') != exit_code or type(exit_code) is not int):
        raise ValueError('native evidence threshold/usage/exit mismatch')
    if any(total >= threshold for _, total in snapshots[:-1]):
        raise ValueError('additional completed model response after actual threshold')
    capture = evidence.get('capture', {})
    if capture.get('events_sha256') != json_hash(events) or capture.get('child_exit') != exit_code:
        raise ValueError('final capture evidence missing')
    rows = evidence.get('native_events', [])
    if not rows or any(epoch(a['timestamp']) > epoch(b['timestamp']) for a, b in zip(rows, rows[1:])):
        raise ValueError('native event chronology missing or reversed')
    native = [r['event'] for r in rows]
    threads = [r for r in native if r.get('type') == 'thread.started']
    if (len(threads) != 1 or threads[0].get('thread_id') != capture['sources'][0]['session_ids'][0] or
            sum(r.get('type') == 'turn.started' for r in native) != 1):
        raise ValueError('fresh native/session identity mismatch')
    errors = [r for r in native if r.get('type') in ('error', 'turn.failed')]
    terminals = [r for r in native if r.get('type') in ('turn.completed', 'turn.failed')]
    if len(terminals) != 1 or native[-1] != terminals[0]:
        raise ValueError('single complete native turn termination required')
    def message(row):
        return row.get('message') if row['type'] == 'error' else (row.get('error') or {}).get('message')
    native_stop = bool(errors) and all(message(r) == NATIVE_BUDGET_ERROR for r in errors)
    if native_stop:
        if terminals[0]['type'] != 'turn.failed' or usage['actual_input_plus_output_tokens'] < threshold or exit_code < 0:
            raise ValueError('native budget marker without actual threshold/failed-turn completion')
        boundary_hash = snapshots[-1][0]
        crossing = next(e for e in events if json_hash(e) == boundary_hash)
        if epoch(rows[-1]['timestamp']) < epoch(crossing.get('timestamp')):
            raise ValueError('native stop predates raw threshold crossing')
        reason = 'native_token_budget_exhausted'
    else:
        if errors or exit_code != 0 or terminals[0]['type'] != 'turn.completed':
            raise ValueError('unrelated editor error or missing normal termination')
        observed = terminals[0].get('usage') or {}
        if any(observed.get(k) != usage[k] for k in ('input_tokens', 'output_tokens', 'cached_input_tokens')):
            raise ValueError('native final usage disagrees with raw capture')
        reason = 'normal'
    if require_native_stop and not native_stop:
        raise ValueError('diagnostic needs native budget exhaustion, not voluntary completion')
    total = usage['actual_input_plus_output_tokens']
    return {'schema': 'wsp-editor-completion/2', 'status': 'HOLDOUT_ELIGIBLE' if frozen else 'INCOMPLETE_NO_FREEZE',
            'usage': usage, 'threshold_tokens': threshold, 'actual_overrun_tokens': max(0, total - threshold),
            'first_crossing_token_event_sha256': snapshots[-1][0] if total >= threshold else None,
            'completed_usage_snapshots': len(snapshots), 'additional_completed_responses_after_boundary': 0,
            'exit_code': exit_code, 'exit_reason': reason, 'response_evidence_sha256': json_hash(evidence),
            'policy_sha256': json_hash(RESPONSE_POLICY),
            'admission_observation': 'not directly exposed; native stopping semantics require separate reviewed live proof'}


def validate_live_proof(path, approval, task):
    """Replay the actual collector from hashed native sources; no proof Booleans."""
    from wsp_collect import collect_native_evidence
    path = Path(path)
    if approval.get('actual_usage_proof_sha256') != sha256(path):
        raise ValueError('coordinator has not approved this actual-usage proof')
    proof = strict_json(path.read_bytes())
    if proof.get('schema') != 'wsp-live-actual-usage-proof/3' or proof.get('synthetic') is not False:
        raise ValueError('reviewed live native cached-response proof v3 required')
    expected = {'schema': 'wsp-live-actual-usage-proof/3', 'runtime': 'isolated_codex_exec', 'synthetic': False,
                'case': 'cached_multiresponse_actual_total', 'runtime_identity': task['editor']['runtime_identity'],
                'policy_sha256': json_hash(RESPONSE_POLICY), 'scientific_threshold_tokens': task['budgets']['editor_agent_tokens'],
                'diagnostic_threshold_tokens': 50000}
    if any(proof.get(k) != value for k, value in expected.items()):
        raise ValueError('reviewed live native cached-response proof v3 required for this runtime')
    root = Path(proof['evidence_directory'])
    events, trace = collect_native_evidence(root, write=False)
    saved = strict_json((root / 'editor_response_evidence.json').read_bytes())
    if trace != saved or json_hash(trace) != proof.get('native_evidence_sha256'):
        raise ValueError('live native evidence changed or was not produced by collector')
    invocation = strict_json((root / 'editor_invocation.json').read_bytes())
    diagnostic_task = {'editor': {**task['editor'], 'prompt': invocation['prompt']}}
    limits = invocation['limits']
    if (invocation['kind'] != 'text_diagnostic' or invocation['threshold_tokens'] != 50000 or
            invocation['model_context_sha256'] is not None or not invocation['fixture_hashes'] or
            invocation['runtime_identity'] != task['editor']['runtime_identity'] or
            invocation['command'] != editor_command(diagnostic_task, 50000) or
            invocation['prompt_sha256'] != hashlib.sha256(invocation['prompt'].encode()).hexdigest() or
            invocation['output_directory'] != str(root.resolve()) or
            limits.get('editor_wall_seconds') != 180 or limits.get('emergency_actual_tokens') != 100000 or
            limits.get('tool_requests') != 10 or limits.get('retained_bytes') != 128 * 1024**2):
        raise ValueError('diagnostic prompt/fixture/limits or stopping mechanism not explicitly bound')
    if not proof.get('threshold_rationale') or proof.get('diagnostic_differences') != diagnostic_differences(task, invocation):
        raise ValueError('diagnostic differences and threshold rationale required')
    # Check the exact fixture files, not an arbitrary matching field. The later
    # diagnostic launcher must keep these inert text fixtures in its own root.
    for name, digest in invocation['fixture_hashes'].items():
        from wsp_weights import regular_file
        if sha256(regular_file(root, name)) != digest:
            raise ValueError('diagnostic fixture changed')
    usage = actual_usage(events)
    if (usage.get('distinct_snapshots', 0) < 2 or not usage.get('cached_input_tokens') or
            proof.get('observed_actual_usage') != usage):
        raise ValueError('multiple genuine response snapshots, cached input and exact actual usage required')
    completion = response_completion(events, trace, 50000, frozen=False,
                                     exit_code=trace['exit_code'], require_native_stop=True)
    return {'proof_sha256': sha256(path), 'usage': usage, 'completion': completion,
            'runtime_identity_sha256': json_hash(invocation['runtime_identity']), 'collector_sha256': trace['collector_sha256'],
            'diagnostic_threshold_tokens': 50000, 'scientific_threshold_tokens': task['budgets']['editor_agent_tokens'],
            'review_note': 'Native budget exhaustion correlated with cache-inclusive crossing and no later completed usage. Does not directly expose attempted-request admission; coordinator must review the live diagnostic to establish native stopping semantics.'}
