# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Driver CPU fakes/small Python subprocesses only; never Codex, API or GPU."""
import json
import sys
import threading

import pytest

from test_wsp import usage_event, json_hash, sha256, diagnostic_fixture as raw_diagnostic_fixture
from wsp_client import call
from wsp_diagnostic import (EvidenceBudget, EvidenceLimit, BoundedStream, FixtureLedger, FixtureServer,
                            LIMITS, HOST_LIMIT, guard_reason, drive, assess, prepare, approve_manifest, run)
from wsp_fixtures import native_fixture
from wsp_usage import validate_live_proof


def diagnostic_fixture(root):
    return raw_diagnostic_fixture(root)


def test_host_fixture_counter_is_bounded_and_not_resettable(tmp_path):
    ledger = FixtureLedger(EvidenceBudget(tmp_path))
    server = FixtureServer(tmp_path / 'socket', ledger)
    thread = threading.Thread(target=server.serve_forever); thread.start()
    try:
        for n in range(1, 11):
            value = call({'op': 'fixture', 'sequence': n}, path=str(tmp_path / 'socket'))
            assert value['ok'] and value['sequence'] == n and len(json.dumps(value)) < 512
        assert not call({'op': 'fixture', 'sequence': 11}, path=str(tmp_path / 'socket'))['ok']
        assert json.loads((tmp_path / 'fixture_state.json').read_text())['admitted'] == 10
        with pytest.raises(ValueError, match='no reset'): FixtureLedger(EvidenceBudget(tmp_path))
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_invalid_fixture_requests_spend_admission(tmp_path):
    ledger = FixtureLedger(EvidenceBudget(tmp_path))
    assert not ledger.request({'op': 'forward', 'sequence': 1})['ok']
    assert ledger.request({'op': 'fixture', 'sequence': 2})['ok']
    assert ledger.count == 2


def test_retained_stream_is_truncated_and_final_record_reserved(tmp_path):
    budget = EvidenceBudget(tmp_path, limit=4096, reserve=1024)
    stream = BoundedStream([sys.executable, '-c', 'import sys;sys.stdout.write("x"*10000)'], tmp_path / 'raw', budget, 10000)
    stream.finish()
    assert stream.error and (tmp_path / 'raw').stat().st_size == 3072
    budget.json('failure.json', {'truncated': True}, final=True)
    assert budget.size() <= 4096
    with pytest.raises(EvidenceLimit): budget.write(tmp_path / 'extra', b'x')


def test_stream_timeout_does_not_leave_child(tmp_path):
    stream = BoundedStream([sys.executable, '-c', 'import time;time.sleep(30)'], tmp_path / 'raw', EvidenceBudget(tmp_path), 1000)
    stream.finish(timeout=.02)
    assert stream.error and stream.process.poll() is not None and not stream.thread.is_alive()


@pytest.mark.parametrize('events,elapsed,extra,expected', [
    ([usage_event(90000, 10000, 80000)], 1, {}, 'emergency_actual_tokens'),
    ([usage_event(90000, 9999, 80000)], 1, {}, None),
    ([usage_event(49990, 10, 20000), usage_event(51000, 20, 21000)], 1, {}, None),
    ([], 180, {}, 'wall_limit'), ([], 0, {'stream_error': 'truncated'}, 'retained_evidence_limit'),
    ([], 0, {'retained': HOST_LIMIT}, 'retained_evidence_limit')])
def test_outer_guards_count_cache_once(events, elapsed, extra, expected):
    assert guard_reason(events, elapsed, **extra) == expected


class FakeBackend:
    def __init__(self, root, *, start_error=False, observe_error=False, finalize_error=False, cleanup_error=False, tokens=50030):
        self.budget = EvidenceBudget(root)
        self.start_error, self.observe_error = start_error, observe_error
        self.finalize_error, self.cleanup_error = finalize_error, cleanup_error
        self.tokens, self.closed, self.finalized, self.started = tokens, False, None, 0
    def start(self):
        self.started += 1
        if self.start_error: raise ValueError('synthetic setup failure after owned job allocation')
    def observe(self):
        if self.observe_error: raise ValueError('synthetic observation failure')
        return {'events': [usage_event(self.tokens - 10, 10, 30000)], 'exit': {'exit_code': 1}}
    def stream_error(self): return None
    def finalize(self, reason):
        self.finalized = reason or 'normal'
        if self.finalize_error: raise ValueError('synthetic harvest failure')
        native_fixture(self.budget.root, [usage_event(24990, 10, 10000), usage_event(self.tokens - 10, 10, 30000)],
                       50000, observed_stop=True, exit_code=-15, forced_stop=reason)
    def close(self):
        self.closed = True
        if self.cleanup_error: raise ValueError('synthetic cleanup failure')
        self.budget.json('fixture_state.json', {'admitted': 2, 'limit': 10, 'closed': True})
        self.budget.json('cleanup.json', {'errors': [], 'fixture_admissions': 2,
                  'closed_jobs': [{'job_id': 'synthetic-job', 'state': 'ERROR'}]})


@pytest.mark.parametrize('failure', ['start_error', 'observe_error', 'finalize_error', 'cleanup_error', 'emergency', 'wall', None])
def test_outer_loop_always_finalizes_and_closes_one_attempt(tmp_path, failure):
    kwargs = {failure: True} if failure in ('start_error', 'observe_error', 'finalize_error', 'cleanup_error') else {}
    backend = FakeBackend(tmp_path, tokens=100001 if failure == 'emergency' else 50030, **kwargs)
    times = iter([0, 181 if failure == 'wall' else 1])
    reason, error = drive(backend, clock=lambda: next(times), sleep=lambda _: None)
    assert backend.closed and backend.finalized is not None and backend.started == 1
    assert bool(reason or error) == (failure is not None)


def test_proof_proposal_comes_from_collector_and_passes_reviewed_validator(tmp_path):
    task, _, _, _ = diagnostic_fixture(tmp_path)
    # No approval is created by the driver. This synthetic test invokes the
    # reviewed validator with a synthetic approval solely to test compatibility.
    result = assess(tmp_path, task, EvidenceBudget(tmp_path), None, None)
    assert result['status'] == 'OBSERVED_STOP_REVIEW_REQUIRED'
    proof = tmp_path / 'live_proof.proposed.json'
    accepted = validate_live_proof(proof, {'actual_usage_proof_sha256': sha256(proof)}, task)
    assert accepted['usage']['actual_input_plus_output_tokens'] == 50030


@pytest.mark.parametrize('kind', ['voluntary', 'no_cache', 'old_native', 'outer_stop', 'driver_error'])
def test_bad_outcomes_never_propose_live_proof(tmp_path, kind):
    task, _, _, _ = diagnostic_fixture(tmp_path)
    if kind != 'driver_error':
        events = [usage_event(24990, 10, 0 if kind == 'no_cache' else 10000), usage_event(50020, 10, 0 if kind == 'no_cache' else 30000)]
        if kind == 'old_native': events.append(usage_event(55000, 20, 32000))
        native_fixture(tmp_path, events, 50000, observed_stop=kind not in ('voluntary', 'old_native'), native_stop=kind == 'old_native', exit_code=0 if kind == 'voluntary' else -15)
    result = assess(tmp_path, task, EvidenceBudget(tmp_path), 'emergency_actual_tokens' if kind == 'outer_stop' else None,
                    'error' if kind == 'driver_error' else None)
    assert result['status'] == 'INCONCLUSIVE' and not (tmp_path / 'live_proof.proposed.json').exists()


def test_run_directory_cannot_resume_or_overwrite(tmp_path, monkeypatch):
    import wsp_diagnostic
    root = tmp_path / 'attempt'; root.mkdir()
    monkeypatch.setattr(wsp_diagnostic, 'approve_manifest', lambda *args: ({'run_dir': str(root)}, {}))
    with pytest.raises(FileExistsError): run('unused', 'unused', backend_factory=lambda *_: pytest.fail('must not launch'))


def test_full_one_shot_driver_with_fake_backend(tmp_path, monkeypatch):
    import wsp_diagnostic
    source = tmp_path / 'source'; source.mkdir()
    science, _, _, _ = diagnostic_fixture(source)
    root = tmp_path / 'attempt'
    approval = tmp_path / 'synthetic-approval.json'; approval.write_text('{"synthetic_test_only":true}')
    monkeypatch.setattr(wsp_diagnostic, 'approve_manifest', lambda *args: ({'run_dir': str(root)}, science))
    made = []
    def factory(manifest, budget):
        backend = FakeBackend(root); made.append(backend); return backend
    result = run('synthetic-manifest', approval, backend_factory=factory)
    assert result['status'] == 'OBSERVED_STOP_REVIEW_REQUIRED'
    assert made[0].closed and made[0].started == 1 and result['fixture_admissions'] == 2
    assert result['wall_seconds_including_cleanup'] >= 0
    with pytest.raises(FileExistsError): run('synthetic-manifest', approval, backend_factory=factory)
    assert len(made) == 1


def test_manifest_prepare_seals_without_credential_read_or_launch(tmp_path, monkeypatch):
    import wsp_diagnostic
    from pathlib import Path
    identity = {'fixed': 'synthetic runtime'}
    science = {'editor': {'runtime_identity': identity, 'model': 'gpt-6-astra', 'effort': 'xhigh', 'prompt': 'synthetic scientific prompt'},
               'budgets': {'editor_agent_tokens': 200000}, 'generic_image': 'synthetic-pinned-image',
               'run_dir': str(tmp_path / 'scientific-never-create'), 'response_budget_policy': __import__('wsp_usage').RESPONSE_POLICY,
               'code_hashes': {'wsp_diagnostic.py': sha256(Path(wsp_diagnostic.__file__))}}
    source = tmp_path / 'science.json'; source.write_text(json.dumps({'task': science, 'preflight_sha256': json_hash(science)}))
    monkeypatch.setattr(wsp_diagnostic, 'runtime_identity', lambda *_: identity)
    monkeypatch.setattr(wsp_diagnostic.subprocess, 'Popen', lambda *_a, **_k: pytest.fail('prepare must never launch'))
    auth = tmp_path / 'synthetic-private-auth'; auth.write_text('fixture only, never opened')
    original = Path.read_bytes
    def protected(path):
        assert path != auth
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', protected)
    sealed = prepare(source, tmp_path / 'manifest.json', tmp_path / 'diagnostic', tmp_path / 'fake-codex', auth)
    assert sealed['manifest']['task']['budgets'] == LIMITS
    assert not (tmp_path / 'diagnostic').exists() and not Path(science['run_dir']).exists()
    approval = tmp_path / 'approval.json'; approval.write_text('{}')
    with pytest.raises(ValueError, match='approval'): approve_manifest(tmp_path / 'manifest.json', approval)


def test_truncated_archive_retains_inconclusive_outcome(tmp_path):
    task, _, _, _ = diagnostic_fixture(tmp_path)
    (tmp_path / 'editor_results.tar').write_bytes(b'truncated archive')
    result = assess(tmp_path, task, EvidenceBudget(tmp_path), 'retained_evidence_limit', None)
    assert result['status'] == 'INCONCLUSIVE'
    assert result['outer_stop'] == 'retained_evidence_limit'
    assert result['assessment_error']
    assert json.loads((tmp_path / 'outcome.json').read_text()) == result
    assert not (tmp_path / 'live_proof.proposed.json').exists()
