#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual token accounting, independent of a CLI's configured budget semantics."""
import math
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
    'schema': 'wsp-response-budget/1',
    'accounting': 'cumulative raw input_tokens + output_tokens; cached input included once; no reset',
    'admission': 'deny a new model response once completed actual usage reaches the threshold',
    'inflight': 'one serial response admitted below the threshold may complete; charge and disclose its entire overrun',
    'completion': 'valid freeze plus complete final usage and trusted admission/exit evidence permits holdout; token stop without freeze is incomplete',
    'failures': 'missing usage/evidence, counter reset, later response admission, unrelated errors or any other exceeded budget fail',
    'spend_ceiling': False,
    'diagnostic_threshold': 'a lower positive disclosed threshold is allowed; only limit_tokens and its derived reminder change',
}


def editor_command(task, threshold=None):
    """One actual command renderer, also sealed in the diagnostic identity."""
    from astra_isolation import token_cap_config
    budget = '<threshold-and-derived-reminder>' if threshold is None else token_cap_config(threshold)
    return ['/tools/wsp_isolation.py', 'editor-entry', '/usr/bin/python3', '/runtime/codex_capture.py',
            '/usr/local/bin/codex', 'exec', '--ignore-user-config', '--ignore-rules', '--skip-git-repo-check',
            '--dangerously-bypass-approvals-and-sandbox', '--json', '--color', 'never',
            '-m', task['editor']['model'], '-c', 'model_reasoning_effort="' + task['editor']['effort'] + '"',
            '-c', 'developer_instructions=""', '-c', 'features.plugins=false', '-c', 'features.multi_agent=false',
            '-c', budget, '-C', '/task', '-o', '/results/final.txt', task['editor']['prompt']]


def runtime_identity(task, codex_binary):
    """Nonsecret runtime identity. No agent execution, credential or model read."""
    binary = Path(codex_binary).resolve(strict=True)
    binaries = [binary, binary.parent / 'codex-code-mode-host', binary.parent / 'codex-exec-server', Path('/usr/bin/python3.12')]
    return {'schema': 'wsp-editor-runtime/1', 'model': task['editor']['model'], 'effort': task['editor']['effort'],
            'codex_version': task['editor']['codex_version'], 'image': task['generic_image'],
            'api_hosts': task['editor']['api_hosts'], 'command_template': editor_command(task),
            'binary_files': {str(p): sha256(p) for p in binaries}, 'code_hashes': task['code_hashes'],
            'budgets_except_editor_tokens': {k: v for k, v in task['budgets'].items() if k != 'editor_agent_tokens'},
            'credential_mount_target': '/codex/auth.json',
            'fixture_policy': 'diagnostic uses declared synthetic development/tool payloads, no real weights or GPU; same prompt, tools, wrapper, route, flags and code',
            'threshold_variation': ['limit_tokens', 'reminder_at_remaining_tokens=threshold//2']}


def response_snapshots(events):
    actual_usage(events)  # Validate every cumulative counter, including resets.
    result = []
    previous = None
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


def response_completion(events, evidence, threshold, *, frozen, exit_code, forced_stop=None, require_denial=False):
    """Validate trusted serial admission chronology against the actual raw events.

    This validates evidence; it cannot create it from a watchdog or CONNECT log.
    The separately reviewed live collector must supply authentic admission,
    completion, denial and final-exit events outside editor-writable mounts.
    """
    usage = actual_usage(events)
    snapshots = response_snapshots(events)
    if not snapshots or usage['actual_input_plus_output_tokens'] is None:
        raise ValueError('complete actual usage is required')
    if forced_stop is not None:
        raise ValueError('non-token forced stop is not successful completion')
    if evidence.get('schema') != 'wsp-response-evidence/1' or evidence.get('threshold_tokens') != threshold:
        raise ValueError('response evidence threshold/schema mismatch')
    chronology = evidence.get('events')
    if not isinstance(chronology, list) or not chronology or chronology[-1].get('kind') != 'editor_exit':
        raise ValueError('complete admission and exit chronology required')
    current = 0
    active = None
    seen = set()
    completed = []
    denied = 0
    boundary_response = None
    last_time = -math.inf
    final = None
    for index, row in enumerate(chronology):
        at = row.get('at_unix')
        if type(at) not in (int, float) or not math.isfinite(at) or at < last_time:
            raise ValueError('invalid admission chronology time')
        last_time = at
        kind = row.get('kind')
        if kind == 'response_admitted':
            identity = row.get('response_id')
            if not isinstance(identity, str) or not identity or identity in seen or active is not None:
                raise ValueError('responses must have unique serial admissions')
            if current >= threshold or row.get('actual_before') != current:
                raise ValueError('model response admitted after actual threshold or against false usage')
            seen.add(identity); active = identity
        elif kind == 'response_completed':
            if active is None or row.get('response_id') != active or len(completed) >= len(snapshots):
                raise ValueError('completion without matching admission')
            event_hash, total = snapshots[len(completed)]
            if row.get('token_event_sha256') != event_hash:
                raise ValueError('response chronology omits/reorders actual usage')
            completed.append(event_hash)
            current = total
            if current >= threshold:
                boundary_response = active
            active = None
        elif kind == 'response_denied':
            if active is not None or current < threshold or row.get('reason') != 'response_threshold' or row.get('actual_before') != current:
                raise ValueError('denial does not establish actual-threshold admission control')
            identity = row.get('response_id')
            if not isinstance(identity, str) or not identity or identity in seen:
                raise ValueError('denial must identify a distinct attempted next response')
            seen.add(identity); denied += 1
        elif kind == 'editor_exit':
            if index != len(chronology) - 1 or active is not None or len(completed) != len(snapshots):
                raise ValueError('incomplete final response/usage collection')
            if type(row.get('exit_code')) is not int or row['exit_code'] != exit_code:
                raise ValueError('actual editor exit must be retained, not invented')
            reason = row.get('reason')
            if reason == 'normal':
                if exit_code != 0:
                    raise ValueError('unrelated nonzero editor exit')
            elif reason == 'response_threshold':
                if boundary_response is None or not denied or exit_code < 0:
                    raise ValueError('token stop lacks denied followup evidence')
            else:
                raise ValueError('unrelated editor error')
            final = row
        else:
            raise ValueError('unknown admission evidence event')
    if final is None or current != usage['actual_input_plus_output_tokens']:
        raise ValueError('missing complete final usage')
    if require_denial and (boundary_response is None or denied == 0):
        raise ValueError('diagnostic must reach its threshold and deny the attempted next response')
    return {'schema': 'wsp-editor-completion/1', 'status': 'HOLDOUT_ELIGIBLE' if frozen else 'INCOMPLETE_NO_FREEZE',
            'usage': usage, 'threshold_tokens': threshold, 'actual_overrun_tokens': max(0, current - threshold),
            'boundary_response': boundary_response, 'admitted_responses': len(completed), 'denied_responses': denied,
            'exit_code': exit_code, 'exit_reason': final['reason'], 'response_evidence_sha256': json_hash(evidence),
            'policy_sha256': json_hash(RESPONSE_POLICY)}


def validate_live_proof(path, approval, task):
    """A lower diagnostic threshold is explicit; every other runtime binding matches."""
    path = Path(path)
    if approval.get('actual_usage_proof_sha256') != sha256(path):
        raise ValueError('coordinator has not approved this actual-usage proof')
    proof = strict_json(path.read_bytes())
    if proof.get('schema') != 'wsp-live-actual-usage-proof/2' or proof.get('synthetic') is not False:
        raise ValueError('reviewed live response-boundary proof v2 required')
    expected = {'schema': 'wsp-live-actual-usage-proof/2', 'runtime': 'isolated_codex_exec',
                'synthetic': False, 'case': 'cached_multiresponse_actual_total',
                'runtime_identity': task['editor']['runtime_identity'],
                'policy_sha256': json_hash(RESPONSE_POLICY),
                'scientific_threshold_tokens': task['budgets']['editor_agent_tokens']}
    if any(proof.get(k) != value for k, value in expected.items()):
        raise ValueError('live cached-response enforcement is not established for this exact editor')
    source = Path(proof['token_events_path'])
    gate_evidence = Path(proof['request_gate_evidence_path'])
    if sha256(source) != proof['token_events_sha256'] or sha256(gate_evidence) != proof['request_gate_evidence_sha256']:
        raise ValueError('live proof source changed')
    events = [strict_json(line) for line in source.read_bytes().splitlines() if line.strip()]
    usage = actual_usage(events)
    previous_cached = 0
    cached_responses = 0
    for event in events:
        payload = event.get('payload', event)
        value = (payload.get('info') or {}).get('total_token_usage') if payload.get('type') == 'token_count' else None
        if value:
            cached_responses += value['cached_input_tokens'] > previous_cached
            previous_cached = value['cached_input_tokens']
    if not usage.get('actual_input_plus_output_tokens') or usage.get('distinct_snapshots', 0) < 2 or cached_responses < 2:
        raise ValueError('proof must contain multiple actual cached-response snapshots')
    threshold = proof.get('diagnostic_threshold_tokens')
    if type(threshold) is not int or not 2 <= threshold <= task['budgets']['editor_agent_tokens'] or not proof.get('threshold_rationale'):
        raise ValueError('disclose the lower diagnostic threshold and its relationship to the scientific threshold')
    if proof.get('observed_actual_usage') != usage:
        raise ValueError('proof misstates actual usage')
    trace = strict_json(gate_evidence.read_bytes())
    identity_hash = json_hash(task['editor']['runtime_identity'])
    collector = proof.get('collector_sha256')
    if not isinstance(collector, str) or len(collector) != 64 or any(c not in '0123456789abcdef' for c in collector):
        raise ValueError('reviewed admission collector code hash required')
    if sha256(Path(proof['collector_path'])) != collector:
        raise ValueError('admission collector code changed')
    if trace.get('runtime_identity_sha256') != identity_hash or trace.get('collector_sha256') != collector:
        raise ValueError('diagnostic chronology runtime/collector identity mismatch')
    completion = response_completion(events, trace, threshold, frozen=False,
                                     exit_code=proof.get('observed_exit_code'), require_denial=True)
    return {'proof_sha256': sha256(path), 'usage': usage, 'completion': completion,
            'runtime_identity_sha256': identity_hash, 'collector_sha256': collector,
            'diagnostic_threshold_tokens': threshold, 'scientific_threshold_tokens': task['budgets']['editor_agent_tokens'],
            'review_note': 'authentic admission chronology and collector provenance require coordinator review; no inference from watchdog, CONNECT traffic or proof Booleans'}
