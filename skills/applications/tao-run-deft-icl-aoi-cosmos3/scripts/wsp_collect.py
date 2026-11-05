#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trusted host collector for obtainable native events; no admission invention.

Only the runner writes the input/harvest/terminal records outside editor mounts.
Archives are inspected as bytes, never extracted. CPU fixtures establish parser
and lifecycle behavior, not native cached-response stopping semantics.
"""
import datetime
import math
import tarfile
from pathlib import Path, PurePosixPath

from tht_contract import json_hash, sha256
from suite_runtime import json_write
from wsp_weights import regular_file, strict_json

NATIVE_BUDGET_ERROR = 'shared rollout token budget exhausted'
SOURCE_FILES = ('editor_usage_events.json', 'editor_events.timestamped.log', 'editor_results.tar',
                'editor_task.tar', 'editor_harvest.json', 'editor_terminal.json', 'editor_invocation.json', 'editor_stop_ack.json')


def epoch(value):
    if not isinstance(value, str):
        raise ValueError('timestamp required')
    result = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('timezone-aware timestamp required')
    return result.timestamp()


def native_events(raw):
    """Only top-level native event fields count; tool-output marker text cannot."""
    result, other = [], []
    if raw and not raw.endswith('\n'):
        raise ValueError('incomplete final native log line')
    for index, line in enumerate(raw.splitlines()):
        timestamp, separator, body = line.partition(' ')
        if not separator:
            raise ValueError('timestamped native log required')
        epoch(timestamp)
        try:
            row = strict_json(body)
        except (ValueError, UnicodeError):
            if body.lstrip().startswith(('{', '[')):
                raise ValueError('malformed native JSON log line')
            other.append({'line': index, 'sha256': json_hash(body)})
            continue  # Plain native stderr is retained and hashed, never a stop marker.
        if not isinstance(row, dict):
            raise ValueError('invalid native JSON event')
        if row.get('type') in ('thread.started', 'turn.started', 'turn.completed', 'turn.failed', 'error'):
            result.append({'timestamp': timestamp, 'line': index, 'event': row})
        elif row.get('type') not in ('item.started', 'item.updated', 'item.completed') and row.get('schema') != 'wsp-access-probe/1':
            raise ValueError('unknown top-level native event')
    return result, other


def archive_files(path, names):
    found = {}
    with tarfile.open(path, 'r:*') as archive:
        for member in archive:
            name = str(PurePosixPath(member.name))
            # docker cp CONTAINER:/results/. - includes the literal results/
            # directory prefix. Accept only that known prefix, never basename
            # matching or archive extraction that could conceal other paths.
            if name.startswith('results/'):
                name = name.removeprefix('results/')
            if name not in names:
                continue
            if not member.isfile() or member.issym() or member.islnk() or name in found or member.size > 16 * 1024**2:
                raise ValueError('invalid/duplicate final capture archive member')
            found[name] = archive.extractfile(member).read()
    if set(found) != set(names):
        raise ValueError('missing final capture archive member')
    return found


def collect_native_evidence(root, *, write=True):
    """Called by the real runner after final observation, harvest and native exit.

    Replayed read-only by the live-proof validator against the exact saved files.
    Host paths stay private; the evidence is never mounted into the editor.
    """
    root = Path(root)
    paths = {name: regular_file(root, name) for name in SOURCE_FILES}
    invocation = strict_json(paths['editor_invocation.json'].read_bytes())
    limit = invocation['log_limit_bytes']
    if type(limit) is not int or not 0 < limit <= 128 * 1024**2 or paths['editor_events.timestamped.log'].stat().st_size >= limit:
        raise ValueError('native log overflow or invalid bound')
    events = strict_json(paths['editor_usage_events.json'].read_bytes())
    native, non_json = native_events(paths['editor_events.timestamped.log'].read_text())
    harvest = strict_json(paths['editor_harvest.json'].read_bytes())
    terminal = strict_json(paths['editor_terminal.json'].read_bytes())
    files = archive_files(paths['editor_results.tar'], {'codex_usage_events.jsonl', 'codex_usage_capture.json', 'editor_exit.json', 'editor_supervision.json'})
    archived_events = [strict_json(line) for line in files['codex_usage_events.jsonl'].splitlines() if line.strip()]
    capture = strict_json(files['codex_usage_capture.json'])
    child_exit = strict_json(files['editor_exit.json'])
    supervision = strict_json(files['editor_supervision.json'])
    ack = strict_json(paths['editor_stop_ack.json'].read_bytes())
    if events != archived_events or capture.get('events_sha256') != json_hash(events) or capture.get('token_events') != len(events):
        raise ValueError('final usage observation/harvest mismatch')
    if capture.get('schema') != 'wsp-final-capture/2' or type(capture.get('child_exit')) is not int or capture['child_exit'] != child_exit.get('child_returncode'):
        raise ValueError('missing final child-exit capture')
    if (supervision.get('schema') != 'wsp-supervision/1' or supervision.get('child_returncode') != capture['child_exit'] or
            supervision.get('supervisor_returncode') != child_exit.get('supervisor_returncode') or
            supervision.get('final_capture_sha256') != json_hash(capture)):
        raise ValueError('supervision/final capture/returncode mismatch')
    if supervision.get('notice') is not None and (supervision.get('host_ack') != ack or ack.get('notice_sha256') != json_hash(supervision['notice'])):
        raise ValueError('host admission closure evidence mismatch')
    if child_exit != terminal.get('editor_exit'):
        raise ValueError('actual wrapper exit/harvest mismatch')
    sources = capture.get('sources', [])
    if (len(sources) != 1 or type(sources[0].get('bytes')) is not int or sources[0]['bytes'] <= 0 or
            not isinstance(sources[0].get('sha256'), str) or len(sources[0]['sha256']) != 64 or
            not sources[0].get('newline_terminated') or sources[0].get('malformed_lines') != [] or
            len(sources[0].get('session_ids', [])) != 1 or not sources[0]['session_ids'][0]):
        raise ValueError('fresh session final source coverage missing or malformed')
    if not events or any(e.get('_source_file') != sources[0]['file'] for e in events):
        raise ValueError('final source identity mismatch')
    lines = [e.get('_source_line') for e in events]
    if any(type(n) is not int or n < 0 for n in lines) or lines != sorted(set(lines)):
        raise ValueError('raw token event source order invalid')
    for name in ('editor_results.tar', 'editor_task.tar', 'editor_usage_events.json'):
        if harvest.get('files', {}).get(name) != sha256(paths[name]):
            raise ValueError('final harvest source changed')
    times = [capture['captured_at_unix'], harvest.get('observed_exit_at_unix'),
             harvest.get('final_observation_at_unix'), harvest.get('archives_completed_at_unix')]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in times) or times != sorted(times):
        raise ValueError('final capture/exit/observation/harvest ordering missing')
    backend = terminal.get('backend', {})
    native_exit = backend.get('native', {}).get('ExitCode')
    if (backend.get('state') not in ('COMPLETE', 'ERROR') or native_exit != child_exit.get('container_returncode') or
            terminal.get('log_capture_returncode') != 0 or terminal.get('log_thread_completed') is not True or
            terminal.get('log_limit_hit') is not False):
        raise ValueError('terminal native exit or complete log harvest missing')
    collector = sha256(Path(__file__))
    if invocation.get('runtime_identity', {}).get('code_hashes', {}).get('wsp_collect.py') != collector:
        raise ValueError('collector is not bound to invocation runtime')
    evidence = {'schema': 'wsp-native-evidence/2', 'mode': 'observed_raw_usage', 'invocation_id': invocation['job_id'],
                'threshold_tokens': invocation['threshold_tokens'], 'collector_sha256': collector,
                'runtime_identity_sha256': json_hash(invocation['runtime_identity']),
                'invocation_sha256': json_hash(invocation), 'raw_events_sha256': json_hash(events),
                'native_events': native, 'non_json_log_lines': non_json,
                'capture': capture, 'supervision': supervision, 'host_stop_ack': ack,
                'exit_code': child_exit['child_returncode'], 'supervisor_returncode': child_exit['supervisor_returncode'],
                'container_returncode': native_exit, 'forced_stop': terminal.get('stop_reason'),
                'source_hashes': {name: sha256(path) for name, path in paths.items()},
                'observability': 'Raw cumulative completed usage, observed threshold notice and host closure acknowledgement, bounded child-tree termination, distinct child/supervisor/container exits and local final capture. Interrupted provider usage remains unknown. No API request-admission chronology is inferred.'}
    if write:
        json_write(root / 'editor_response_evidence.json', evidence)
    return events, evidence
