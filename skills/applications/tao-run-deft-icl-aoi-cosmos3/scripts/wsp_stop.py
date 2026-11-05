#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WSP-only raw accounting and close-only host control, all standard library."""
import hashlib
import json
import os
import socket
import socketserver
import time
from pathlib import Path

STOP_POLICY = {'schema': 'wsp-observed-stop-policy/1', 'mode': 'observed_raw_usage',
               'poll_seconds': .1, 'max_poll_gap_seconds': 1.0,
               'term_grace_seconds': 3.0, 'kill_reap_seconds': 2.0,
               'notify_timeout_seconds': 1.0, 'max_stop_seconds': 7.0}


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True) + '\n')
    temporary.replace(path)


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


class StopHandler(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(STOP_POLICY['notify_timeout_seconds'])
        try:
            raw = self.rfile.readline(16385)
            if len(raw) > 16384 or not raw.endswith(b'\n'):
                raise ValueError('stop notification framing')
            notice = json.loads(raw)
            if notice.get('schema') != 'wsp-stop-notice/1' or not notice.get('reason'):
                raise ValueError('stop notification schema')
            # This endpoint can only close editing. It cannot launch work or
            # reopen/refund any admission. Actual notice content is replayed
            # against captured raw events by the post-run collector/validator.
            self.server.close_admissions(notice)
            ack = {'schema': 'wsp-stop-ack/1', 'notice_sha256': json_hash(notice),
                   'closed_at_unix': time.time(), 'admissions_closed': True}
            atomic_json(self.server.record, ack)
            self.wfile.write(json.dumps(ack).encode() + b'\n')
        except (ValueError, OSError):
            pass


class StopServer(socketserver.UnixStreamServer):
    def __init__(self, path, record, close_admissions):
        self.record, self.close_admissions = Path(record), close_admissions
        atomic_json(self.record, {'schema': 'wsp-stop-ack/1', 'status': 'not_requested'})
        super().__init__(str(path), StopHandler)
        os.chmod(path, 0o600)


def notify_stop(path, notice):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(STOP_POLICY['notify_timeout_seconds'])
        connection.connect(str(path))
        connection.sendall(json.dumps(notice).encode() + b'\n')
        with connection.makefile('rb') as stream:
            ack = json.loads(stream.readline(16385))
        if ack.get('notice_sha256') != json_hash(notice) or ack.get('admissions_closed') is not True:
            raise ValueError('host admission closure unconfirmed')
        return ack
