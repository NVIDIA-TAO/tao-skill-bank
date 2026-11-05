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
from wsp_usage import editor_command


def native_fixture(root, events, threshold, *, native_stop=False, exit_code=0, error=None, forced_stop=None):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    for i, event in enumerate(events):
        event.setdefault('_source_line', i + 1)
    usage = events[-1]['payload']['info']['total_token_usage']
    started = [{'type': 'thread.started', 'thread_id': 'synthetic-session'}, {'type': 'turn.started'}]
    if native_stop or error:
        message = error or NATIVE_BUDGET_ERROR
        rows = started + [{'type': 'error', 'message': message}, {'type': 'turn.failed', 'error': {'message': message}}]
    else:
        rows = started + [{'type': 'turn.completed', 'usage': usage}]
    (root / 'editor_events.timestamped.log').write_text(''.join('2026-11-05T00:00:05Z ' + json.dumps(r) + '\n' for r in rows))
    json_write(root / 'editor_usage_events.json', events)
    capture = {'schema': 'wsp-final-capture/1', 'captured_at_unix': epoch('2026-11-05T00:00:06Z'),
               'child_exit': exit_code, 'events_sha256': json_hash(events), 'token_events': len(events),
               'sources': [{'file': events[0]['_source_file'], 'session_ids': ['synthetic-session'],
                            'bytes': 1, 'sha256': 'f' * 64, 'malformed_lines': [], 'newline_terminated': True}]}
    members = {'codex_usage_events.jsonl': ''.join(json.dumps(e) + '\n' for e in events).encode(),
               'codex_usage_capture.json': json.dumps(capture).encode(), 'editor_exit.json': json.dumps({'exit_code': exit_code}).encode()}
    write_archive(root / 'editor_results.tar', members)
    write_archive(root / 'editor_task.tar', {})
    json_write(root / 'editor_harvest.json', {
        'observed_exit_at_unix': epoch('2026-11-05T00:00:07Z'), 'final_observation_at_unix': epoch('2026-11-05T00:00:08Z'),
        'archives_completed_at_unix': epoch('2026-11-05T00:00:09Z'),
        'files': {n: sha256(root / n) for n in ('editor_results.tar', 'editor_task.tar', 'editor_usage_events.json')}})
    json_write(root / 'editor_terminal.json', {'backend': {'state': 'COMPLETE' if exit_code == 0 else 'ERROR', 'native': {'ExitCode': exit_code}},
               'editor_exit': {'exit_code': exit_code}, 'stop_reason': forced_stop, 'log_limit_hit': False,
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
