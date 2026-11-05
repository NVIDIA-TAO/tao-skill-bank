#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WSP child capture: raw token events plus final source/exit coverage.

Runs inside the isolated container. Never reads auth or exports session text.
Native Codex JSON stdout/stderr is inherited and captured by the trusted host.
"""
import argparse
import ctypes
import hashlib
import os
import signal
import json
import subprocess
import sys
import time
from pathlib import Path

from wsp_stop import STOP_POLICY, actual_usage, response_snapshots, notify_stop, atomic_json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def capture_usage(session_root=Path('/codex/sessions'), output=Path('/results'), *, child_exit=None):
    rows, sources = [], []
    for path in sorted(session_root.rglob('*.jsonl')):
        if path.is_symlink() or path.stat().st_size > 64 * 1024**2:
            raise ValueError('invalid or oversized private session source')
        data = path.read_bytes()
        malformed, identities = [], []
        for ordinal, line in enumerate(data.splitlines()):
            try:
                event = json.loads(line)
            except (ValueError, UnicodeError):
                malformed.append(ordinal)
                continue
            if event.get('type') == 'session_meta':
                identities.append(event.get('payload', {}).get('id'))
            if event.get('payload', {}).get('type') == 'token_count':
                event['_source_file'] = str(path)
                event['_source_line'] = ordinal
                rows.append(event)
        sources.append({'file': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                        'session_ids': identities, 'malformed_lines': malformed,
                        'newline_terminated': data.endswith(b'\n')})
    coverage = {'schema': 'wsp-final-capture/2', 'captured_at_unix': time.time(), 'child_exit': child_exit,
                'sources': sources, 'token_events': len(rows), 'events_sha256': digest(rows)}
    for name, data in (('codex_usage_events.jsonl', ''.join(json.dumps(e) + '\n' for e in rows)),
                       ('codex_usage_capture.json', json.dumps(coverage) + '\n')):
        temporary = output / (name + '.tmp')
        temporary.write_text(data)
        temporary.replace(output / name)
    return coverage



def owned_descendants(pid=None):
    """Only this supervisor's descendants, never a global process listing."""
    pid = os.getpid() if pid is None else pid
    try:
        children = [int(x) for x in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]
    except FileNotFoundError:
        return set()
    result = set(children)
    for child in children:
        result.update(owned_descendants(child))
    return result


def reap_helpers():
    while True:
        try:
            if os.waitpid(-1, os.WNOHANG)[0] == 0:
                break
        except ChildProcessError:
            break


def supervise(command, *, threshold, session_root=Path('/codex/sessions'), output=Path('/results'),
              stop_file=Path('/tmp/editor-stop'), control=Path('/channel/control.sock')):
    """Survives TERM/INT; child has a separate group. Final capture follows reap.

    Native CLI configuration is secondary only. Poll interval is a scheduling
    target; actual gaps and stop duration are recorded and bounded by acceptance.
    Interrupted provider usage is NEVER made complete by a successful local flush.
    """
    if type(threshold) is not int or threshold <= 0:
        raise ValueError('positive observed threshold required')
    # Reap orphaned grandchildren (including helpers that create their own group).
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise OSError('cannot become child subreaper')
    interrupted = []
    previous_handlers = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.signal(signum, lambda sig, frame: interrupted.append(sig))
    process = None
    state = {'schema': 'wsp-supervision/1', 'policy': STOP_POLICY, 'threshold_tokens': threshold,
             'started_at_unix': time.time(), 'notice': None, 'host_ack': None,
             'signals': [], 'observations': [], 'errors': [], 'max_poll_gap_seconds': 0.0, 'polls': 0,
             'child_returncode': None, 'supervisor_returncode': None, 'cleanup_complete': False,
             'provider_interrupted_usage': 'unknown', 'provider_total_tokens': None}
    atomic_json(output / 'editor_supervision.json', state)
    last_poll = time.monotonic()
    def signal_owned(sig):
        sent = set()
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, sig); sent.add(process.pid)
            except ProcessLookupError:
                pass
        for pid in owned_descendants():
            try:
                os.kill(pid, sig); sent.add(pid)
            except ProcessLookupError:
                pass
        state['signals'].append({'signal': sig.name, 'at_unix': time.time(), 'owned_pids': sorted(sent)})
    def stop(reason, events):
        if state['notice'] is not None:
            return
        usage = actual_usage(events)
        snapshots = response_snapshots(events)
        crossing = next((h for h, total in snapshots if total >= threshold), None)
        notice = {'schema': 'wsp-stop-notice/1', 'reason': reason, 'detected_at_unix': time.time(),
                  'threshold_tokens': threshold, 'usage': usage, 'events_sha256': digest(events),
                  'token_events': len(events), 'first_crossing_event_sha256': crossing}
        state['notice'] = notice
        atomic_json(output / 'editor_supervision.json', state)
        # TERM immediately. Host acknowledgement closes future capability
        # admissions independently of any already-admitted forward/copy.
        signal_owned(signal.SIGTERM)
        try:
            state['host_ack'] = notify_stop(control, notice)
        except (ValueError, OSError) as error:
            state['errors'].append('host closure: ' + str(error))
        until = time.monotonic() + max(0, STOP_POLICY['term_grace_seconds'] - (time.time() - notice['detected_at_unix']))
        while time.monotonic() < until:
            process.poll()
            if process.returncode is not None:
                reap_helpers()
            if process.returncode is not None and not owned_descendants():
                break
            time.sleep(.02)
        if process.poll() is None or owned_descendants():
            signal_owned(signal.SIGKILL)
        try:
            process.wait(timeout=STOP_POLICY['kill_reap_seconds'])
        except subprocess.TimeoutExpired:
            state['errors'].append('child reap timeout')
        until = time.monotonic() + STOP_POLICY['kill_reap_seconds']
        while time.monotonic() < until:
            reap_helpers()
            if not owned_descendants():
                break
            for pid in owned_descendants():
                try: os.kill(pid, signal.SIGKILL)
                except ProcessLookupError: pass
            time.sleep(.02)
        state['termination_seconds'] = time.time() - notice['detected_at_unix']
        state['terminated_at_unix'] = time.time()
    events = []
    try:
        process = subprocess.Popen(command, start_new_session=True)
        state['child_pid'] = process.pid
        while True:
            capture_usage(session_root, output)
            events = [json.loads(line) for line in (output / 'codex_usage_events.jsonl').read_text().splitlines()]
            now = time.monotonic()
            state['max_poll_gap_seconds'] = max(state['max_poll_gap_seconds'], now - last_poll)
            state['polls'] += 1
            last_poll = now
            usage = actual_usage(events)
            snapshot = {'at_unix': time.time(), 'events_sha256': digest(events), 'token_events': len(events),
                        'actual_input_plus_output_tokens': usage['actual_input_plus_output_tokens']}
            if not state['observations'] or snapshot['events_sha256'] != state['observations'][-1]['events_sha256']:
                state['observations'].append(snapshot)
            if (usage['actual_input_plus_output_tokens'] or 0) >= threshold:
                stop('observed_raw_threshold', events); break
            if state['max_poll_gap_seconds'] > STOP_POLICY['max_poll_gap_seconds']:
                stop('poll_deadline_exceeded', events); break
            if interrupted or stop_file.exists():
                stop('external_signal' if interrupted else 'external_stop', events); break
            if process.poll() is not None:
                if owned_descendants():
                    stop('orphan_helper_cleanup', events)
                break
            time.sleep(STOP_POLICY['poll_seconds'])
    except BaseException as error:
        state['errors'].append(type(error).__name__ + ': ' + str(error))
        if process is not None:
            try: stop('capture_error', [])
            except BaseException as cleanup_error: state['errors'].append(str(cleanup_error))
    finally:
        if process is not None:
            if process.poll() is None or owned_descendants():
                signal_owned(signal.SIGKILL)
                try: process.wait(timeout=STOP_POLICY['kill_reap_seconds'])
                except subprocess.TimeoutExpired: state['errors'].append('final reap timeout')
            reap_helpers()
            state['child_returncode'] = process.returncode
        state['cleanup_complete'] = process is not None and process.returncode is not None and not owned_descendants()
        try:
            coverage = capture_usage(session_root, output, child_exit=state['child_returncode'])
            state['final_capture_sha256'] = digest(coverage)
            final_events = [json.loads(line) for line in (output / 'codex_usage_events.jsonl').read_text().splitlines()]
            state['measured_completed_usage'] = actual_usage(final_events)
        except BaseException as error:
            state['errors'].append('final capture: ' + str(error))
        state['finished_at_unix'] = time.time()
        # This is the supervisor's status, never the child's negative signal.
        state['supervisor_returncode'] = 0 if state['cleanup_complete'] and not state['errors'] else 70
        atomic_json(output / 'editor_supervision.json', state)
        for signum, previous in previous_handlers.items(): signal.signal(signum, previous)
    return state['supervisor_returncode']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=int, required=True)
    parser.add_argument('--session-root', type=Path, default=Path('/codex/sessions'))
    parser.add_argument('--output', type=Path, default=Path('/results'))
    parser.add_argument('--stop-file', type=Path, default=Path('/tmp/editor-stop'))
    parser.add_argument('--control', type=Path, default=Path('/channel/control.sock'))
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    sys.exit(supervise(command, threshold=args.threshold, session_root=args.session_root,
                       output=args.output, stop_file=args.stop_file, control=args.control))
