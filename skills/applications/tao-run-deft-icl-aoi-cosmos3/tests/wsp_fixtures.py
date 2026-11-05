# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicitly synthetic native records for CPU parser/lifecycle tests only."""
import io
import json
import tarfile
from pathlib import Path

from tht_contract import json_hash, sha256
from suite_runtime import json_write
from wsp_collect import NATIVE_BUDGET_ERROR, collect_native_evidence, epoch
from wsp_usage import editor_command, actual_usage, response_snapshots
from wsp_stop import STOP_POLICY


def native_fixture(root, events, threshold, *, native_stop=False, exit_code=0, error=None, forced_stop=None, observed_stop=False):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    for i, event in enumerate(events):
        event.setdefault('_source_line', i + 1)
    usage = events[-1]['payload']['info']['total_token_usage']
    started = [{'type': 'thread.started', 'thread_id': 'synthetic-session'}, {'type': 'turn.started'}]
    if observed_stop:
        rows = started
    elif native_stop or error:
        message = error or NATIVE_BUDGET_ERROR
        rows = started + [{'type': 'error', 'message': message}, {'type': 'turn.failed', 'error': {'message': message}}]
    else:
        rows = started + [{'type': 'turn.completed', 'usage': usage}]
    (root / 'editor_events.timestamped.log').write_text(''.join('2026-11-05T00:00:05Z ' + json.dumps(r) + '\n' for r in rows))
    json_write(root / 'editor_usage_events.json', events)
    capture = {'schema': 'wsp-final-capture/2', 'captured_at_unix': epoch('2026-11-05T00:00:06Z'),
               'child_exit': exit_code, 'events_sha256': json_hash(events), 'token_events': len(events),
               'sources': [{'file': events[0]['_source_file'], 'session_ids': ['synthetic-session'],
                            'bytes': 1, 'sha256': 'f' * 64, 'malformed_lines': [], 'newline_terminated': True}]}
    exit_record = {'schema': 'wsp-editor-exit/2', 'child_returncode': exit_code,
                   'supervisor_returncode': 0, 'container_returncode': 0}
    supervision = {'schema': 'wsp-supervision/1', 'policy': STOP_POLICY, 'threshold_tokens': threshold,
                   'child_returncode': exit_code, 'supervisor_returncode': 0, 'cleanup_complete': True,
                   'errors': [], 'max_poll_gap_seconds': .11, 'measured_completed_usage': actual_usage(events),
                   'notice': None, 'host_ack': None, 'final_capture_sha256': json_hash(capture)}
    ack = {'schema': 'wsp-stop-ack/1', 'status': 'not_requested'}
    if observed_stop:
        notice = {'schema': 'wsp-stop-notice/1', 'reason': 'observed_raw_threshold',
                  'detected_at_unix': epoch('2026-11-05T00:00:04Z'), 'threshold_tokens': threshold,
                  'events_sha256': json_hash(events), 'token_events': len(events), 'usage': actual_usage(events),
                  'first_crossing_event_sha256': next((h for h, n in response_snapshots(events) if n >= threshold), None)}
        ack = {'schema': 'wsp-stop-ack/1', 'notice_sha256': json_hash(notice),
               'closed_at_unix': epoch('2026-11-05T00:00:04Z') + .01, 'admissions_closed': True}
        supervision.update(notice=notice, host_ack=ack, termination_seconds=.1,
                           observations=[{'at_unix': notice['detected_at_unix'], 'events_sha256': json_hash(events),
                                          'token_events': len(events), 'actual_input_plus_output_tokens': actual_usage(events)['actual_input_plus_output_tokens']}],
                           terminated_at_unix=epoch('2026-11-05T00:00:04Z') + .1,
                           signals=[{'signal': 'SIGTERM', 'at_unix': notice['detected_at_unix']}])
    json_write(root / 'editor_stop_ack.json', ack)
    members = {'codex_usage_events.jsonl': ''.join(json.dumps(e) + '\n' for e in events).encode(),
               'codex_usage_capture.json': json.dumps(capture).encode(),
               'editor_supervision.json': json.dumps(supervision).encode(), 'editor_exit.json': json.dumps(exit_record).encode()}
    write_archive(root / 'editor_results.tar', members)
    write_archive(root / 'editor_task.tar', {})
    json_write(root / 'editor_harvest.json', {
        'observed_exit_at_unix': epoch('2026-11-05T00:00:07Z'), 'final_observation_at_unix': epoch('2026-11-05T00:00:08Z'),
        'archives_completed_at_unix': epoch('2026-11-05T00:00:09Z'),
        'files': {n: sha256(root / n) for n in ('editor_results.tar', 'editor_task.tar', 'editor_usage_events.json')}})
    json_write(root / 'editor_terminal.json', {'backend': {'state': 'COMPLETE', 'native': {'ExitCode': 0}},
               'editor_exit': exit_record, 'stop_reason': forced_stop, 'log_limit_hit': False,
               'log_thread_completed': True, 'log_capture_returncode': 0})
    collector = Path(__file__).resolve().parents[1] / 'scripts/wsp_collect.py'
    identity = {'model': 'fixture-only', 'code_hashes': {'wsp_collect.py': sha256(collector)}}
    editor = {'model': 'fixture-only', 'effort': 'xhigh', 'prompt': 'Synthetic text diagnostic prompt, no WSP capabilities.'}
    (root / 'fixture.txt').write_text('Deterministic inert text fixture; not live evidence.\n')
    json_write(root / 'editor_invocation.json', {'schema': 'wsp-editor-invocation/1', 'job_id': 'synthetic-job',
               'kind': 'text_diagnostic', 'runtime_identity': identity, 'threshold_tokens': threshold,
               'prompt': editor['prompt'], 'prompt_sha256': __import__('hashlib').sha256(editor['prompt'].encode()).hexdigest(),
               'command': editor_command({'editor': editor}, threshold), 'fixture_hashes': {'fixture.txt': sha256(root / 'fixture.txt')},
               'model_context_sha256': None, 'output_directory': str(root), 'log_limit_bytes': 64 * 1024**2,
               'limits': {'editor_wall_seconds': 180, 'emergency_actual_tokens': 100000,
                          'tool_requests': 10, 'retained_bytes': 128 * 1024**2}})
    return collect_native_evidence(root)


def write_archive(path, members):
    with tarfile.open(path, 'w') as archive:
        for name, data in members.items():
            member = tarfile.TarInfo('./' + name); member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
