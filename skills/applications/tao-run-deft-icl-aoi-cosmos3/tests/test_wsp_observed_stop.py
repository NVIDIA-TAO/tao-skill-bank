# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real small CPU subprocesses, never a model/API/container/GPU invocation."""
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import wsp_capture
from wsp_stop import STOP_POLICY, StopServer, json_hash, actual_usage, notify_stop
from wsp_diagnostic import EvidenceBudget, FixtureLedger
from wsp_usage import response_completion
from wsp_fixtures import native_fixture
from test_wsp import synthetic_boundary, usage_event

CHILD = '''import json,signal,sys,time,subprocess,os
from pathlib import Path
mode,root=sys.argv[1],Path(sys.argv[2])
source=root/'sessions'/'one.jsonl'
source.write_text(json.dumps({'type':'session_meta','payload':{'id':'synthetic-session'}})+'\\n')
def usage(i,o,c):
 with source.open('a') as f:
  f.write(json.dumps({'timestamp':'2026-11-05T00:00:00Z','payload':{'type':'token_count','info':{'total_token_usage':{'input_tokens':i,'output_tokens':o,'cached_input_tokens':c}}}})+'\\n');f.flush()
def finish(sig,frame):
 usage(225,15,180)
 (root/'flushed').write_text('yes')
 sys.exit(0)
if mode=='graceful': signal.signal(signal.SIGTERM,finish)
if mode=='stubborn':
 signal.signal(signal.SIGTERM,signal.SIG_IGN)
 helper=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)'],start_new_session=True)
 (root/'helper.pid').write_text(str(helper.pid))
usage(90,10,70)
(root/'ready').write_text('ready')
if mode in ('graceful','stubborn','negative'):
 time.sleep(.15);usage(210,10,170)
if mode=='normal':
 usage(110,10,80);sys.exit(0)
if mode=='nonzero':sys.exit(7)
time.sleep(30)
'''


def launch(tmp_path, mode, *, with_server=True):
    (tmp_path / 'sessions').mkdir(); (tmp_path / 'results').mkdir()
    child = tmp_path / 'child.py'; child.write_text(CHILD)
    closed = []
    def close(notice): closed.append(notice)
    control = tmp_path / 'control.sock'
    server = StopServer(control, tmp_path / 'ack.json', close) if with_server else None
    thread = None
    if server:
        thread = threading.Thread(target=server.serve_forever); thread.start()
    process = subprocess.Popen([sys.executable, wsp_capture.__file__, '--threshold', '200',
        '--session-root', str(tmp_path / 'sessions'), '--output', str(tmp_path / 'results'),
        '--stop-file', str(tmp_path / 'stop'), '--control', str(control), '--', sys.executable, str(child), mode, str(tmp_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process, server, thread, closed


def finish(tmp_path, process, server, thread):
    try:
        stdout, stderr = process.communicate(timeout=12)
        assert not stdout and not stderr, (stdout, stderr)
        return json.loads((tmp_path / 'results/editor_supervision.json').read_text())
    finally:
        if process.poll() is None: process.kill(); process.wait()
        if server: server.shutdown(); server.server_close(); thread.join()


@pytest.mark.parametrize('mode,expected', [('graceful', 0), ('negative', -15), ('stubborn', -9), ('normal', 0), ('nonzero', 7)])
def test_real_supervisor_flushes_reaps_and_separates_returncodes(tmp_path, mode, expected):
    process, server, thread, closed = launch(tmp_path, mode)
    state = finish(tmp_path, process, server, thread)
    assert process.returncode == state['supervisor_returncode'] == 0
    assert state['child_returncode'] == expected and state['cleanup_complete'] and state['errors'] == []
    capture = json.loads((tmp_path / 'results/codex_usage_capture.json').read_text())
    events = [json.loads(line) for line in (tmp_path / 'results/codex_usage_events.jsonl').read_text().splitlines()]
    assert capture['child_exit'] == expected and capture['events_sha256'] == json_hash(events)
    assert state['final_capture_sha256'] == json_hash(capture)
    if mode in ('normal', 'nonzero'):
        assert not closed and state['notice'] is None
    else:
        assert len(closed) == 1 and closed[0]['usage']['actual_input_plus_output_tokens'] == 220
        assert closed[0]['usage']['cached_input_tokens'] == 170
        assert state['notice']['reason'] == 'observed_raw_threshold'
        assert state['termination_seconds'] <= STOP_POLICY['max_stop_seconds']
        assert state['provider_total_tokens'] is None and state['provider_interrupted_usage'] == 'unknown'
    if mode == 'graceful':
        assert (tmp_path / 'flushed').is_file() and actual_usage(events)['actual_input_plus_output_tokens'] == 240
    if mode == 'stubborn':
        assert any(row['signal'] == 'SIGKILL' for row in state['signals'])
        assert not Path('/proc/' + (tmp_path / 'helper.pid').read_text()).exists()


def test_external_signal_leaves_supervisor_alive_for_final_capture(tmp_path):
    process, server, thread, closed = launch(tmp_path, 'external')
    until = time.monotonic() + 3
    while not (tmp_path / 'ready').exists() and time.monotonic() < until: time.sleep(.01)
    process.send_signal(signal.SIGTERM)
    state = finish(tmp_path, process, server, thread)
    assert process.returncode == 0 and state['child_returncode'] == -15
    assert state['notice']['reason'] == 'external_signal' and state['cleanup_complete']
    assert closed and state['final_capture_sha256']


def test_missing_host_ack_cannot_claim_complete_supervision(tmp_path):
    process, server, thread, _ = launch(tmp_path, 'negative', with_server=False)
    state = finish(tmp_path, process, server, thread)
    assert process.returncode == 70 and state['cleanup_complete'] and state['errors']
    assert state['final_capture_sha256'] and state['host_ack'] is None


def test_stop_closes_fixture_admissions_without_reset(tmp_path):
    ledger = FixtureLedger(EvidenceBudget(tmp_path))
    assert ledger.request({'op': 'fixture', 'sequence': 1})['ok']
    ledger.close_admissions({'reason': 'observed_raw_threshold'})
    assert not ledger.request({'op': 'fixture', 'sequence': 2})['ok']
    assert ledger.count == 1
    state = json.loads((tmp_path / 'fixture_state.json').read_text())
    assert state['closed'] and state['admitted'] == 1


def test_closure_does_not_wait_on_already_admitted_forward_operation_lock(tmp_path):
    boundary, calls = synthetic_boundary(tmp_path)
    reached = threading.Event()
    try:
        boundary.mutex.acquire()
        worker = threading.Thread(target=lambda: (boundary.close_editor_admissions({'reason': 'observed_raw_threshold'}), reached.set()))
        worker.start()
        assert reached.wait(1)
        boundary.mutex.release(); worker.join()
        assert not boundary.handle({'op': 'status'})['ok']
        assert not calls and boundary.journal.state()['editor_admissions_closed']
    finally:
        boundary.close()


def test_extra_completed_usage_is_charged_and_unknown_interruption_not_filled_in(tmp_path):
    events = [usage_event(100, 10, 70), usage_event(210, 10, 150), usage_event(240, 20, 180)]
    _, evidence = native_fixture(tmp_path, events, 200, observed_stop=True, exit_code=-15)
    completion = response_completion(events, evidence, 200, frozen=True, exit_code=-15, require_observed_stop=True)
    assert completion['status'] == 'INCOMPLETE_STOP'
    assert completion['additional_completed_responses_after_boundary'] == 1
    assert completion['actual_overrun_tokens'] == 60 and completion['provider_total_tokens'] is None


def test_old_native_proof_schema_is_never_observed_proof(tmp_path):
    from wsp_usage import validate_live_proof
    from tht_contract import sha256
    proof = tmp_path / 'old.json'; proof.write_text(json.dumps({'schema': 'wsp-live-actual-usage-proof/3', 'synthetic': False}))
    with pytest.raises(ValueError, match='v4'): validate_live_proof(proof, {'actual_usage_proof_sha256': sha256(proof)}, {})


def test_real_supervisor_final_flush_replays_through_actual_collector(tmp_path):
    from wsp_collect import collect_native_evidence
    from wsp_fixtures import write_archive
    from tht_contract import sha256
    process, server, thread, closed = launch(tmp_path, 'graceful')
    state = finish(tmp_path, process, server, thread)
    results = tmp_path / 'results'
    events = [json.loads(line) for line in (results / 'codex_usage_events.jsonl').read_text().splitlines()]
    native_fixture(tmp_path, events, 200, observed_stop=True, exit_code=0)
    exit_record = {'schema': 'wsp-editor-exit/2', 'child_returncode': state['child_returncode'],
                   'supervisor_returncode': process.returncode, 'container_returncode': 0}
    members = {name: (results / name).read_bytes() for name in
               ('codex_usage_events.jsonl', 'codex_usage_capture.json', 'editor_supervision.json')}
    members['editor_exit.json'] = json.dumps(exit_record).encode()
    write_archive(tmp_path / 'editor_results.tar', members)
    (tmp_path / 'editor_stop_ack.json').write_bytes((tmp_path / 'ack.json').read_bytes())
    terminal = json.loads((tmp_path / 'editor_terminal.json').read_text()); terminal['editor_exit'] = exit_record
    (tmp_path / 'editor_terminal.json').write_text(json.dumps(terminal))
    now = time.time()
    harvest = {'observed_exit_at_unix': now, 'final_observation_at_unix': now,
               'archives_completed_at_unix': now, 'files': {name: sha256(tmp_path / name) for name in
               ('editor_usage_events.json', 'editor_results.tar', 'editor_task.tar')}}
    (tmp_path / 'editor_harvest.json').write_text(json.dumps(harvest))
    raw, evidence = collect_native_evidence(tmp_path)
    completion = response_completion(raw, evidence, 200, frozen=True, exit_code=0, require_observed_stop=True)
    assert completion['usage']['actual_input_plus_output_tokens'] == 240
    assert completion['status'] == 'INCOMPLETE_STOP' and not completion['provider_usage_complete']
    assert evidence['container_returncode'] == 0 and completion['provider_total_tokens'] is None


@pytest.mark.parametrize('change', ['cleanup', 'wall', 'closed', 'outcome_hash'])
def test_proof_rejects_unbound_or_failed_host_lifecycle(tmp_path, change):
    from test_wsp import diagnostic_fixture
    from wsp_usage import validate_live_proof
    from tht_contract import sha256
    task, proof, path, _ = diagnostic_fixture(tmp_path)
    if change == 'cleanup':
        p = tmp_path / 'cleanup.json'; value = json.loads(p.read_text()); value['errors'] = ['owned helper remains']
    elif change == 'closed':
        p = tmp_path / 'fixture_state.json'; value = json.loads(p.read_text()); value['closed'] = False
    else:
        p = tmp_path / 'outcome.json'; value = json.loads(p.read_text()); value['ended_at_unix'] += 200 if change == 'wall' else .1
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError): validate_live_proof(path, {'actual_usage_proof_sha256': sha256(path)}, task)
