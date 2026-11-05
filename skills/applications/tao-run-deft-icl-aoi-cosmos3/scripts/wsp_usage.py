#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual token accounting, independent of a CLI's configured budget semantics."""
from pathlib import Path

from tht_contract import sha256
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


def validate_live_proof(path, approval, task):
    """Review-bound live evidence is mandatory; synthetic/tiny-stop proof fails."""
    path = Path(path)
    if approval.get('actual_usage_proof_sha256') != sha256(path):
        raise ValueError('coordinator has not approved this actual-usage proof')
    proof = strict_json(path.read_bytes())
    expected = {'schema': 'wsp-live-actual-usage-proof/1', 'runtime': 'isolated_codex_exec',
                'synthetic': False, 'case': 'cached_multiresponse_actual_total',
                'model': task['editor']['model'], 'effort': task['editor']['effort'],
                'codex_version': task['editor']['codex_version'],
                'limit_tokens': task['budgets']['editor_agent_tokens'],
                'actual_ceiling_enforced': True, 'followup_response_denied': True}
    if any(proof.get(k) != value for k, value in expected.items()):
        raise ValueError('live cached-response enforcement is not established for this exact editor')
    source = Path(proof['token_events_path'])
    gate_evidence = Path(proof['request_gate_evidence_path'])
    if sha256(source) != proof['token_events_sha256'] or sha256(gate_evidence) != proof['request_gate_evidence_sha256']:
        raise ValueError('live proof source changed')
    events = [strict_json(line) for line in source.read_bytes().splitlines() if line.strip()]
    usage = actual_usage(events)
    if not usage.get('actual_input_plus_output_tokens') or usage.get('distinct_snapshots', 0) < 2 or usage.get('cached_input_tokens', 0) <= 0:
        raise ValueError('proof must contain multiple actual cached-response snapshots')
    if usage['actual_input_plus_output_tokens'] > proof['limit_tokens'] or proof['observed_actual_usage'] != usage:
        raise ValueError('actual ceiling violated or proof misstates usage')
    return {'proof_sha256': sha256(path), 'usage': usage,
            'review_note': 'request-gate evidence is an explicit coordinator review responsibility; this check does not infer enforcement from a CLI flag'}
