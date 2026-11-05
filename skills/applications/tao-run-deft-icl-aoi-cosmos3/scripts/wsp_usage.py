#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual token accounting, independent of a CLI's configured budget semantics."""
import hashlib
from pathlib import Path

from tht_contract import json_hash, sha256
from wsp_weights import strict_json


from wsp_stop import STOP_POLICY, actual_usage, response_snapshots


RESPONSE_POLICY = {
    'schema': 'wsp-response-budget/3', 'mode': 'observed_raw_usage',
    'accounting': 'cumulative raw input_tokens + output_tokens; cached input included once; no reset',
    'stop': STOP_POLICY,
    'admission': 'close WSP capabilities on first observed crossing; supervisor terminates owned child tree',
    'inflight': 'already-admitted work remains charged; disclose all completed overrun and unknown interrupted provider usage',
    'completion': 'only valid freeze plus normal child exit and complete final native usage permits holdout; all stops are incomplete',
    'failures': 'missing final capture, excessive polling/termination, reset, failed cleanup or another exceeded limit cannot pass',
    'observability': 'observed raw stopping, not native threshold semantics or API request admission',
    'spend_ceiling': False,
    'diagnostic_variations': ['prompt', 'synthetic text fixture/tools', 'output directory', '50000 threshold and derived reminder',
                              '180s wall / 100000 observed emergency threshold / 10 tool requests / 128MiB retained evidence'],
}

def editor_command(task, threshold=None):
    """Shared actual stopping command; diagnostic prompt substitution is explicit."""
    from astra_isolation import token_cap_config
    budget = '<threshold-and-derived-reminder>' if threshold is None else token_cap_config(threshold)
    return ['/tools/wsp_isolation.py', 'editor-entry', '/usr/bin/python3', '/runtime/wsp_capture.py',
            '--threshold', str(threshold) if threshold is not None else '<observed-threshold>', '--', '/usr/local/bin/codex', 'exec', '--ignore-user-config', '--ignore-rules', '--skip-git-repo-check',
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
    return {'schema': 'wsp-editor-runtime/3', 'model': task['editor']['model'], 'effort': task['editor']['effort'],
            'codex_version': task['editor']['codex_version'], 'image': task['generic_image'],
            'api_hosts': task['editor']['api_hosts'], 'command_template': command,
            'binary_files': {str(p): sha256(p) for p in binaries}, 'code_hashes': task['code_hashes'],
            'sandbox': {'cpus': 2, 'memory_bytes': 4 * 1024**3, 'network': 'none; provider-only CONNECT relay',
                        'credential_mount_target': '/codex/auth.json', 'gpu': False},
            'stop_policy': STOP_POLICY, 'response_policy_sha256': json_hash(RESPONSE_POLICY),
            'permitted_diagnostic_variations': RESPONSE_POLICY['diagnostic_variations']}


def invocation_manifest(task, job_id, output, *, kind='scientific', fixture_hashes=None):
    """Written before/at launch; binds the actual prompt, flags and fixture changes."""
    threshold = task['budgets']['editor_agent_tokens']
    return {'schema': 'wsp-editor-invocation/2', 'kind': kind, 'job_id': job_id,
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


def response_completion(events, evidence, threshold, *, frozen, exit_code, forced_stop=None, require_observed_stop=False):
    """Only normal, complete native usage can admit holdout; stops stay incomplete."""
    from wsp_collect import epoch
    usage = actual_usage(events)
    snapshots = response_snapshots(events)
    if not snapshots or usage['actual_input_plus_output_tokens'] is None:
        raise ValueError('measured actual usage is required')
    if not (events[-1].get('payload', events[-1]).get('info') or {}).get('total_token_usage'):
        raise ValueError('final raw snapshot is missing')
    if (evidence.get('schema') != 'wsp-native-evidence/2' or evidence.get('mode') != 'observed_raw_usage' or
            evidence.get('threshold_tokens') != threshold or evidence.get('raw_events_sha256') != json_hash(events) or
            evidence.get('exit_code') != exit_code or type(exit_code) is not int):
        raise ValueError('observed evidence threshold/usage/exit mismatch')
    capture, supervision = evidence.get('capture', {}), evidence.get('supervision', {})
    if capture.get('events_sha256') != json_hash(events) or capture.get('child_exit') != exit_code:
        raise ValueError('final capture evidence missing')
    if (supervision.get('policy') != STOP_POLICY or supervision.get('threshold_tokens') != threshold or
            supervision.get('child_returncode') != exit_code or supervision.get('cleanup_complete') is not True or
            supervision.get('supervisor_returncode') != 0 or supervision.get('errors') != [] or
            supervision.get('measured_completed_usage') != usage or
            not 0 <= supervision.get('max_poll_gap_seconds', float('inf')) <= STOP_POLICY['max_poll_gap_seconds']):
        raise ValueError('complete bounded supervision required')
    rows = evidence.get('native_events', [])
    if not rows or any(epoch(a['timestamp']) > epoch(b['timestamp']) for a, b in zip(rows, rows[1:])):
        raise ValueError('native event chronology missing or reversed')
    native = [r['event'] for r in rows]
    threads = [r for r in native if r.get('type') == 'thread.started']
    if (len(threads) != 1 or threads[0].get('thread_id') != capture['sources'][0]['session_ids'][0] or
            sum(r.get('type') == 'turn.started' for r in native) != 1):
        raise ValueError('fresh native/session identity mismatch')
    notice = supervision.get('notice')
    total = usage['actual_input_plus_output_tokens']
    reason = forced_stop or evidence.get('forced_stop')
    observed_stop = notice is not None and notice.get('reason') == 'observed_raw_threshold'
    if notice is not None:
        count = notice.get('token_events')
        if type(count) is not int or not 0 <= count <= len(events):
            raise ValueError('stop observation source extent invalid')
        prefix = events[:count]
        if (notice.get('events_sha256') != json_hash(prefix) or notice.get('usage') != actual_usage(prefix) or
                notice.get('threshold_tokens') != threshold):
            raise ValueError('stop observation disagrees with captured raw prefix')
        ack = evidence.get('host_stop_ack', {})
        detected = notice.get('detected_at_unix')
        duration = supervision.get('termination_seconds')
        if (type(detected) not in (int, float) or type(duration) not in (int, float) or
                not 0 <= duration <= STOP_POLICY['max_stop_seconds'] or
                ack != supervision.get('host_ack') or ack.get('notice_sha256') != json_hash(notice) or
                ack.get('admissions_closed') is not True or
                not detected <= ack.get('closed_at_unix', 0) <= detected + STOP_POLICY['notify_timeout_seconds'] or
                not detected <= supervision.get('terminated_at_unix', 0) <= capture['captured_at_unix'] or
                abs(supervision['terminated_at_unix'] - detected - duration) > .1):
            raise ValueError('bounded observed termination/host closure is missing')
        if observed_stop:
            observations = supervision.get('observations', [])
            if not observations:
                raise ValueError('raw observation chronology missing')
            for index, row in enumerate(observations):
                n = row.get('token_events')
                if type(n) is not int or not 0 <= n <= count or row.get('events_sha256') != json_hash(events[:n]):
                    raise ValueError('raw observation prefix mismatch')
                measured = actual_usage(events[:n])['actual_input_plus_output_tokens']
                if row.get('actual_input_plus_output_tokens') != measured:
                    raise ValueError('raw observation counter mismatch')
                if index < len(observations) - 1 and measured is not None and measured >= threshold:
                    raise ValueError('stop not at first observed crossing')
                if index and row['at_unix'] < observations[index - 1]['at_unix']:
                    raise ValueError('observations reversed')
            if (observations[-1]['events_sha256'] != notice['events_sha256'] or
                    not observations[-1]['at_unix'] <= detected <= observations[-1]['at_unix'] + STOP_POLICY['max_poll_gap_seconds']):
                raise ValueError('stop detection timestamp/prefix mismatch')
            observed_snapshots = response_snapshots(prefix)
            crossing = next((h for h, n in observed_snapshots if n >= threshold), None)
            if crossing is None or notice.get('first_crossing_event_sha256') != crossing:
                raise ValueError('actual raw crossing required')
            if not supervision.get('signals') or supervision['signals'][0].get('signal') != 'SIGTERM':
                raise ValueError('observed child termination path missing')
        reason = reason or notice['reason']
    terminals = [r for r in native if r.get('type') in ('turn.completed', 'turn.failed')]
    errors = [r for r in native if r.get('type') in ('error', 'turn.failed')]
    normal = (not reason and notice is None and exit_code == 0 and len(terminals) == 1 and
              terminals[0]['type'] == 'turn.completed' and native[-1] == terminals[0] and not errors)
    if normal:
        if total >= threshold:
            raise ValueError('normal completion crossed threshold without observed stopping')
        if any((terminals[0].get('usage') or {}).get(k) != usage[k] for k in ('input_tokens', 'output_tokens', 'cached_input_tokens')):
            raise ValueError('native final usage disagrees with raw capture')
    elif notice is None and not reason:
        raise ValueError('nonzero/error editor exit is not normal completion')
    if require_observed_stop and (not observed_stop or reason != 'observed_raw_threshold'):
        raise ValueError('diagnostic requires observed raw threshold termination')
    first_crossing = next((h for h, n in snapshots if n >= threshold), None)
    after = 0 if first_crossing is None else len(snapshots) - next(i for i, (h, _) in enumerate(snapshots) if h == first_crossing) - 1
    return {'schema': 'wsp-editor-completion/3', 'mode': 'observed_raw_usage',
            'status': ('HOLDOUT_ELIGIBLE' if frozen else 'INCOMPLETE_NO_FREEZE') if normal else 'INCOMPLETE_STOP',
            'usage': usage, 'provider_usage_complete': normal, 'interrupted_provider_usage': None if normal else 'unknown',
            'provider_total_tokens': total if normal else None,
            'threshold_tokens': threshold, 'actual_overrun_tokens': max(0, total - threshold),
            'first_crossing_token_event_sha256': first_crossing, 'completed_usage_snapshots': len(snapshots),
            'additional_completed_responses_after_boundary': after,
            'exit_code': exit_code, 'exit_reason': 'normal' if normal else reason,
            'response_evidence_sha256': json_hash(evidence), 'policy_sha256': json_hash(RESPONSE_POLICY),
            'admission_observation': 'host WSP closure recorded; provider request admission unobserved; not a strict spend ceiling'}


def validate_live_proof(path, approval, task):
    """Replay the actual collector from hashed native sources; no proof Booleans."""
    from wsp_collect import collect_native_evidence
    path = Path(path)
    if approval.get('actual_usage_proof_sha256') != sha256(path):
        raise ValueError('coordinator has not approved this actual-usage proof')
    proof = strict_json(path.read_bytes())
    if proof.get('schema') != 'wsp-live-observed-usage-proof/4' or proof.get('synthetic') is not False:
        raise ValueError('reviewed live observed cached-response proof v4 required')
    expected = {'schema': 'wsp-live-observed-usage-proof/4', 'runtime': 'isolated_codex_exec', 'synthetic': False,
                'case': 'cached_multiresponse_observed_stop', 'runtime_identity': task['editor']['runtime_identity'],
                'policy_sha256': json_hash(RESPONSE_POLICY), 'scientific_threshold_tokens': task['budgets']['editor_agent_tokens'],
                'diagnostic_threshold_tokens': 50000}
    if any(proof.get(k) != value for k, value in expected.items()):
        raise ValueError('reviewed live observed cached-response proof v4 required for this runtime')
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
                                     exit_code=trace['exit_code'], require_observed_stop=True)
    lifecycle = validate_diagnostic_cleanup(root, completion)
    if proof.get('lifecycle_source_hashes') != lifecycle:
        raise ValueError('diagnostic lifecycle evidence changed')
    return {'proof_sha256': sha256(path), 'usage': usage, 'completion': completion,
            'runtime_identity_sha256': json_hash(invocation['runtime_identity']), 'collector_sha256': trace['collector_sha256'],
            'diagnostic_threshold_tokens': 50000, 'scientific_threshold_tokens': task['budgets']['editor_agent_tokens'],
            'review_note': 'Observed cache-inclusive threshold, host admission closure and bounded termination/cleanup. Completed overrun is charged; interrupted provider usage remains unknown. Native accounting and API request admission are not established.'}


def validate_diagnostic_cleanup(root, completion):
    """Proof replay also binds actual host lifecycle/resource closeout, not flags."""
    root = Path(root)
    cleanup = strict_json((root / 'cleanup.json').read_bytes())
    fixture = strict_json((root / 'fixture_state.json').read_bytes())
    attempt = strict_json((root / 'attempt.json').read_bytes())
    outcome = strict_json((root / 'outcome.json').read_bytes())
    if (cleanup.get('errors') != [] or not cleanup.get('closed_jobs') or
            any(j.get('state') not in ('COMPLETE', 'ERROR') for j in cleanup['closed_jobs']) or
            cleanup.get('fixture_admissions') != fixture.get('admitted') or
            type(fixture.get('admitted')) is not int or not 0 <= fixture['admitted'] <= 10 or
            fixture.get('closed') is not True or
            completion['usage']['actual_input_plus_output_tokens'] >= 100000 or
            outcome.get('error') is not None or outcome.get('outer_stop') not in (None, 'observed_raw_threshold') or
            not 0 <= outcome.get('ended_at_unix', float('inf')) - attempt['started_at_unix'] <= 180):
        raise ValueError('diagnostic cleanup/known usage/wall/admission bounds failed')
    hashes = {name: sha256(root / name) for name in ('cleanup.json', 'fixture_state.json', 'attempt.json')}
    hashes['outcome_basis_sha256'] = json_hash({k: v for k, v in outcome.items() if k not in ('status', 'proof_sha256')})
    if sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink()) > 92 * 1024**2:
        raise ValueError('diagnostic retained host evidence exceeds bound')
    return hashes
