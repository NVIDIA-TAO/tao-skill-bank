#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare/seal a text-only diagnostic; run only with separate exact approval.

One attempt, no scientific runner/model/data access, no resume or retry. Native
stopping remains a measured outcome, not a claim implied by configuration.
"""
import argparse
import copy
import hashlib
import json
import os
import socket
import socketserver
import subprocess
import threading
import time
from pathlib import Path

from suite_runtime import Docker
from tht_contract import json_hash, sha256
from wsp_collect import collect_native_evidence
from wsp_isolation import UnixProvider
from wsp_runtime import archive_command, sandbox_mounts, submit_sandbox
from wsp_usage import (RESPONSE_POLICY, actual_usage, diagnostic_differences, editor_command,
                       invocation_manifest, response_completion, response_snapshots, runtime_identity)
from wsp_weights import regular_file, strict_json

HERE = Path(__file__).resolve().parent
MIB = 1024**2
LIMITS = {'editor_agent_tokens': 50000, 'editor_wall_seconds': 180, 'emergency_actual_tokens': 100000,
          'tool_requests': 10, 'retained_bytes': 128 * MIB, 'editor_log_bytes': 16 * MIB,
          'editor_cpus': 2, 'editor_memory_bytes': 4 * 1024**3}
# Reserve32MiB for two native Docker log files (16MiB each), plus4MiB
# headroom for their framing/rotation and2MiB host final-status bookkeeping.
HOST_LIMIT = 92 * MIB
NATIVE_LOG_LIMIT = 16 * MIB
FIXTURE = 'Synthetic diagnostic fixture. Sequence payload: alpha beta gamma delta. No images, labels, model or weights.\n'
PROMPT = '''This is one bounded pure-text runtime diagnostic, not a scientific experiment.
Use only the generic fixture service: run python3 /tools/wsp_client.py with JSON on stdin, starting with {"op":"fixture","sequence":1}.
Make exactly one fixture request per tool call. Read that returned fixture, then make the next sequential request in a later tool call, up to sequence10. Do not batch, loop in a script, pad output, create other agents, call APIs or use any other endpoint. Each result is small deterministic text; do not invent results, usage, timestamps or native stop events.
Continue these sequential requests until the native runtime stops or all ten permitted results have been read. If ten complete, give one brief factual final answer. An early finish is inconclusive evidence, never a reason to retry. Do not seek any model, data, images, labels, host paths or credentials. /development/fixture.txt and /task/tool_contract.json contain only the synthetic fixture and this limited protocol.
'''


class EvidenceLimit(ValueError):
    pass


class EvidenceBudget:
    def __init__(self, root, limit=HOST_LIMIT, reserve=2 * MIB):
        self.root, self.limit, self.reserve = Path(root), limit, reserve
        self.lock = threading.RLock()

    def size(self):
        return sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file() and not p.is_symlink())

    def write(self, path, data, *, final=False):
        path = Path(path)
        with self.lock:
            old = path.stat().st_size if path.exists() else 0
            if self.size() - old + len(data) > self.limit - (0 if final else self.reserve):
                raise EvidenceLimit('retained evidence bound')
            path.write_bytes(data)

    def json(self, name, value, *, final=False):
        self.write(self.root / name, (json.dumps(value, sort_keys=True, indent=2) + '\n').encode(), final=final)

    def append(self, stream, data, local_remaining):
        with self.lock:
            available = min(local_remaining, self.limit - self.reserve - self.size())
            kept = max(0, min(len(data), available))
            stream.write(data[:kept]); stream.flush()
            if kept != len(data):
                raise EvidenceLimit('stream truncated at retained evidence bound')
            return kept


class BoundedStream:
    """Only this host subprocess is terminated; its partial bytes stay retained."""
    def __init__(self, command, path, budget, limit):
        self.error = None
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        def pump():
            size = 0
            try:
                with Path(path).open('xb') as output:
                    while data := self.process.stdout.read1(65536):
                        size += budget.append(output, data, limit - size)
            except BaseException as error:
                self.error = str(error)
                if self.process.poll() is None:
                    self.process.terminate()
        self.thread = threading.Thread(target=pump, daemon=True)
        self.thread.start()

    def finish(self, timeout=5):
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.error = self.error or 'stream collection timeout'
            self.process.terminate()
            try: self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait(timeout=2)
        self.thread.join(timeout=2)
        self.process.stdout.close()
        if self.thread.is_alive(): self.error = self.error or 'stream thread did not exit'
        if self.process.returncode != 0: self.error = self.error or 'stream subprocess exit ' + str(self.process.returncode)
        return self.process.returncode


class FixtureLedger:
    """Admission count is host-owned and written before returning any payload."""
    def __init__(self, budget):
        self.budget, self.count, self.lock = budget, 0, threading.Lock()
        if (budget.root / 'fixture_state.json').exists(): raise ValueError('fixture ledger exists; no reset')
        self.budget.json('fixture_state.json', {'admitted': 0, 'limit': 10})

    def request(self, request):
        with self.lock:
            if self.count >= 10:
                return {'ok': False, 'error': 'fixture_limit'}
            self.count += 1  # Invalid requests also spend a host admission.
            self.budget.json('fixture_state.json', {'admitted': self.count, 'limit': 10})
            if request != {'op': 'fixture', 'sequence': self.count}:
                return {'ok': False, 'error': 'fixture_schema_or_sequence'}
            return {'ok': True, 'sequence': self.count, 'fixture': FIXTURE.strip()}


class FixtureHandler(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(2)
        try:
            raw = self.rfile.readline(1025)
            value = strict_json(raw) if len(raw) <= 1024 and raw.endswith(b'\n') else None
        except (ValueError, OSError): value = None
        try:
            response = self.server.ledger.request(value)
            self.wfile.write(json.dumps(response).encode() + b'\n')
        except (OSError, EvidenceLimit): pass


class FixtureServer(socketserver.UnixStreamServer):
    def __init__(self, path, ledger):
        self.ledger = ledger
        super().__init__(str(path), FixtureHandler)
        os.chmod(path, 0o600)


class ClosingProvider(UnixProvider):
    """Track and close only this invocation's incoming provider connections."""
    def __init__(self, *args):
        self.connections, self.connection_lock = set(), threading.Lock()
        super().__init__(*args)

    def get_request(self):
        request, address = super().get_request()
        with self.connection_lock: self.connections.add(request)
        return request, address

    def close_request(self, request):
        with self.connection_lock: self.connections.discard(request)
        super().close_request(request)

    def disconnect(self):
        with self.connection_lock:
            for request in list(self.connections):
                try: request.shutdown(socket.SHUT_RDWR)
                except OSError: pass
                request.close()


class DiagnosticDocker(Docker):
    """Shared lifecycle without the default unbounded terminal-log duplication."""
    def status(self, job):
        result = subprocess.run(['docker', 'inspect', '--format', '{{json .State}}', job],
                                capture_output=True, text=True, timeout=5)
        if result.returncode: return {'state': 'UNKNOWN', 'reason': 'owned backend missing'}
        native = strict_json(result.stdout)
        state = {'running': 'RUNNING', 'created': 'PENDING', 'exited': 'COMPLETE' if native['ExitCode'] == 0 else 'ERROR'}.get(native['Status'], 'UNKNOWN')
        if state in ('COMPLETE', 'ERROR'):
            record = strict_json(self.record('show', job))
            if record['terminal_state'] is None: self.record('mark', job, '--state', state, '--message', 'diagnostic native exit ' + str(native['ExitCode']))
        return {'state': state, 'native': native}


def prepare(scientific_file, output, run_dir, codex_binary, auth_file):
    scientific_file, output, run_dir = Path(scientific_file).resolve(), Path(output).resolve(), Path(run_dir).resolve()
    record = strict_json(scientific_file.read_bytes()); science = record['task']
    if record['preflight_sha256'] != json_hash(science): raise ValueError('scientific preflight hash changed')
    if (run_dir.exists() or output.exists() or run_dir == output or run_dir in output.parents or
            run_dir == Path(science['run_dir']) or Path(science['run_dir']) in run_dir.parents):
        raise ValueError('new diagnostic paths required; never the scientific directory')
    for name, digest in science['code_hashes'].items():
        if sha256(regular_file(HERE, name)) != digest: raise ValueError('pinned code changed: ' + name)
    identity = runtime_identity(science, codex_binary)
    if identity != science['editor']['runtime_identity']: raise ValueError('scientific runtime identity changed')
    if science['code_hashes'].get('wsp_diagnostic.py') != sha256(Path(__file__)):
        raise ValueError('regenerate proposed scientific identity with diagnostic driver pinned')
    task = {'editor': {**copy.deepcopy(science['editor']), 'prompt': PROMPT,
                       'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest()},
            'budgets': dict(LIMITS), 'generic_image': science['generic_image']}
    fixture_hash = hashlib.sha256(FIXTURE.encode()).hexdigest()
    template = invocation_manifest(task, '<minted-at-launch>', run_dir, kind='text_diagnostic', fixture_hashes={'fixture.txt': fixture_hash})
    manifest = {'schema': 'wsp-diagnostic-manifest/1', 'scientific_preflight': str(scientific_file),
                'scientific_preflight_file_sha256': sha256(scientific_file), 'scientific_preflight_sha256': record['preflight_sha256'],
                'reviewed_base_commit': '2e8a90e1e4ee0aa061ec937719357567729cad1a', 'run_dir': str(run_dir),
                'codex_binary': str(Path(codex_binary).resolve()), 'auth_file': str(Path(auth_file).absolute()),
                'credential_policy': 'existing private mount only; contents never read/exported/hashed',
                'task': task, 'invocation_template': template, 'fixture_text': FIXTURE,
                'invocation_template_sha256': json_hash(template),
                'diagnostic_differences': diagnostic_differences(science, template),
                'retention': {'total_bytes': 128 * MIB, 'host_bytes': HOST_LIMIT, 'native_logs_bytes_each': NATIVE_LOG_LIMIT,
                              'native_jobs': 2, 'native_framing_reserve_bytes': 4 * MIB, 'host_closeout_reserve_bytes': 2 * MIB},
                'sandbox_variation': 'native log rotation16MiB per owned probe/editor job; other sandbox controls unchanged',
                'execution': 'one attempt; no retry/resume; explicit manifest approval required; no launch during preparation'}
    sealed = {'manifest': manifest, 'manifest_sha256': json_hash(manifest)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream: json.dump(sealed, stream, indent=2, sort_keys=True); stream.write('\n')
    output.chmod(0o444)
    return sealed


def approve_manifest(path, approval_path):
    sealed = strict_json(Path(path).read_bytes()); m = sealed['manifest']
    if sealed['manifest_sha256'] != json_hash(m): raise ValueError('manifest hash changed')
    approval = strict_json(Path(approval_path).read_bytes())
    if (approval.get('diagnostic_manifest_sha256') != sealed['manifest_sha256'] or approval.get('approved_for_live_text_diagnostic') is not True or
            not approval.get('approved_by') or not approval.get('approved_at')):
        raise ValueError('separate exact-manifest live diagnostic approval required')
    if sha256(m['scientific_preflight']) != m['scientific_preflight_file_sha256']: raise ValueError('scientific preflight changed')
    science = strict_json(Path(m['scientific_preflight']).read_bytes())['task']
    if json_hash(science) != m['scientific_preflight_sha256']: raise ValueError('scientific identity changed')
    for name, digest in science['code_hashes'].items():
        if sha256(regular_file(HERE, name)) != digest: raise ValueError('pinned code changed: ' + name)
    if runtime_identity(science, m['codex_binary']) != m['task']['editor']['runtime_identity']: raise ValueError('runtime changed')
    if m['task']['budgets'] != LIMITS or m['fixture_text'] != FIXTURE or m['task']['editor']['prompt'] != PROMPT:
        raise ValueError('diagnostic contract changed')
    expected_editor = {**science['editor'], 'prompt': PROMPT, 'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest()}
    if (set(m['task']) != {'editor', 'budgets', 'generic_image'} or m['task']['editor'] != expected_editor or
            m['task']['generic_image'] != science['generic_image']): raise ValueError('shared runtime configuration changed')
    template = invocation_manifest(m['task'], '<minted-at-launch>', m['run_dir'], kind='text_diagnostic',
                                   fixture_hashes={'fixture.txt': hashlib.sha256(FIXTURE.encode()).hexdigest()})
    if (template != m['invocation_template'] or json_hash(template) != m['invocation_template_sha256'] or
            diagnostic_differences(science, template) != m['diagnostic_differences']): raise ValueError('sealed invocation changed')
    if not Path(m['auth_file']).is_file(): raise ValueError('existing private auth mount absent')
    return m, science


def guard_reason(events, elapsed, *, stream_error=None, retained=0):
    if stream_error or retained >= HOST_LIMIT - 2 * MIB: return 'retained_evidence_limit'
    if elapsed >= 180: return 'wall_limit'
    if events and (actual_usage(events)['actual_input_plus_output_tokens'] or 0) >= 100000: return 'emergency_actual_tokens'
    if any(total >= 50000 for _, total in response_snapshots(events)[:-1]): return 'native_boundary_late_response'
    return None


def drive(backend, *, clock=time.monotonic, sleep=time.sleep):
    """Testable outer resource/exit loop; all early failures go through finalize."""
    started = clock(); reason = None; error = None
    try:
        backend.start()
        while True:
            observation = backend.observe()
            reason = guard_reason(observation['events'], clock() - started,
                                  stream_error=backend.stream_error(), retained=backend.budget.size())
            if reason or observation['exit'] is not None: break
            sleep(.2)
    except BaseException as caught:
        reason, error = 'driver_error', type(caught).__name__ + ': ' + str(caught)
    try:
        backend.finalize(reason)
    except BaseException as caught:
        error = error or type(caught).__name__ + ': ' + str(caught)
        reason = reason or 'collection_error'
    finally:
        try: backend.close()
        except BaseException as caught:
            error = error or type(caught).__name__ + ': ' + str(caught)
            reason = reason or 'cleanup_error'
    return reason, error


class HostBackend:
    def __init__(self, manifest, budget):
        self.m, self.budget, self.root = manifest, budget, budget.root
        self.runtime = DiagnosticDocker(self.root / 'jobs', manifest['task']['generic_image'])
        self.jobs, self.services = [], []
        self.job, self.log, self.last, self.observed_exit_at = None, None, None, None
        self.ledger = FixtureLedger(budget)

    def start(self):
        development = self.root / 'development'; development.mkdir()
        self.budget.write(self.root / 'fixture.txt', FIXTURE.encode())
        self.budget.write(development / 'fixture.txt', FIXTURE.encode())
        self.budget.json('tool_contract.json', {'schema': 'wsp-text-fixture/1', 'request': {'op': 'fixture', 'sequence': 'integer1..10'},
                                              'admission_limit': 10, 'host_owned_counter': True, 'real_wsp_capabilities': False})
        channel = self.root / 'channel'; channel.mkdir(mode=0o700)
        for factory in (lambda: FixtureServer(channel / 'wsp.sock', self.ledger),
                        lambda: ClosingProvider(channel / 'provider.sock', self.m['task']['editor']['api_hosts'])):
            service = factory()
            thread = threading.Thread(target=service.serve_forever, daemon=True); thread.start()
            self.services.append((service, thread))
        self.mounts = sandbox_mounts(development, self.root / 'tool_contract.json', channel / 'wsp.sock',
                     codex_binary=self.m['codex_binary'], auth_file=self.m['auth_file'], provider_socket=channel / 'provider.sock')
        def opened(job_id, kind):
            self.jobs.append(job_id); self.budget.json(kind + '_job.json', {'job_id': job_id})
            if kind == 'editor':
                self.job = {'job_id': job_id}  # Retain the minted ID even if submission fails.
                invocation = copy.deepcopy(self.m['invocation_template']); invocation['job_id'] = job_id
                self.budget.json('editor_invocation.json', invocation)
        probe = submit_sandbox(self.runtime, 'wsp-text-access', ['/tools/wsp_isolation.py', 'probe'], self.mounts,
                               on_open=lambda job_id: opened(job_id, 'probe'), log_max_bytes=NATIVE_LOG_LIMIT)
        terminal = self.runtime.wait(probe['job_id'], time.time() + 20)
        self.budget.json('probe_terminal.json', terminal)
        stream = BoundedStream(['docker', 'logs', '--timestamps', probe['job_id']], self.root / 'probe.log', self.budget, MIB)
        stream.finish()
        if terminal['state'] != 'COMPLETE' or stream.error: raise ValueError('access probe failed')
        self.job = submit_sandbox(self.runtime, 'wsp-text-diagnostic', editor_command(self.m['task'], 50000), self.mounts,
                                 on_open=lambda job_id: opened(job_id, 'editor'), log_max_bytes=NATIVE_LOG_LIMIT)
        controls = strict_json(subprocess.check_output(['docker', 'inspect', self.job['job_id']], text=True, timeout=5))[0]
        self.budget.json('native_controls.json', {k: controls[k] for k in ('HostConfig', 'Mounts', 'Config')})
        self.log = BoundedStream(['docker', 'logs', '--follow', '--timestamps', self.job['job_id']],
                                 self.root / 'editor_events.timestamped.log', self.budget, LIMITS['editor_log_bytes'])

    def observe(self):
        code = ('import json;from pathlib import Path; p=Path("/results/codex_usage_events.jsonl");e=Path("/results/editor_exit.json");'
                'print(json.dumps({"events":[json.loads(x) for x in p.read_text().splitlines()] if p.is_file() else [],'
                '"exit":json.loads(e.read_text()) if e.is_file() else None}))')
        record = strict_json(subprocess.check_output(['docker', 'exec', self.job['job_id'], '/usr/bin/python3', '-c', code], text=True, timeout=5))
        self.budget.json('editor_usage_events.json', record['events'])
        self.last = record
        if record['exit'] is not None and self.observed_exit_at is None: self.observed_exit_at = time.time()
        return record

    def stream_error(self):
        return self.log.error if self.log else None

    def signal(self, name):
        code = 'from pathlib import Path;Path(' + repr('/tmp/' + name) + ').touch()'
        subprocess.run(['docker', 'exec', self.job['job_id'], '/usr/bin/python3', '-c', code], check=True, timeout=5)

    def finalize(self, reason):
        if not self.job: return
        if reason or not self.last or self.last['exit'] is None:
            self.signal('editor-stop')
            until = time.monotonic() + 5
            while time.monotonic() < until:
                if self.observe()['exit'] is not None: break
                time.sleep(.1)
        record = self.observe(); final_at = time.time()
        errors = []
        for name in ('results', 'task'):
            stream = BoundedStream(archive_command(self.job['job_id'], name), self.root / ('editor_' + name + '.tar'),
                                   self.budget, 24 * MIB if name == 'task' else 18 * MIB)
            stream.finish(timeout=10)
            if stream.error: errors.append(name + ': ' + stream.error)
        self.budget.json('editor_harvest.json', {'observed_exit_at_unix': self.observed_exit_at,
            'final_observation_at_unix': final_at, 'archives_completed_at_unix': time.time(), 'archive_errors': errors,
            'files': {n: sha256(self.root / n) for n in ('editor_results.tar', 'editor_task.tar', 'editor_usage_events.json')}}, final=True)
        self.signal('harvested')
        terminal = self.runtime.wait(self.job['job_id'], time.time() + 5)
        self.log.finish()
        self.budget.json('editor_terminal.json', {'backend': terminal, 'editor_exit': record['exit'],
              'stop_reason': reason or ('archive_limit_or_error' if errors else None),
              'log_limit_hit': bool(self.log.error), 'log_capture_returncode': self.log.process.returncode,
              'log_thread_completed': not self.log.thread.is_alive()}, final=True)
        if errors: raise EvidenceLimit('; '.join(errors))

    def close(self):
        errors, closed_jobs = [], []
        for job_id in self.jobs:
            try:
                state = self.runtime.status(job_id)
                if state['state'] in ('RUNNING', 'PENDING'): self.runtime.cancel(job_id)
                final = self.runtime.status(job_id)
                closed_jobs.append({'job_id': job_id, 'state': final['state']})
                if final['state'] not in ('COMPLETE', 'ERROR'): errors.append('owned job has no verified terminal state')
            except BaseException as error: errors.append(str(error))
        if self.log and self.log.process.poll() is None: self.log.finish(timeout=.1)
        for service, thread in self.services:
            try:
                if isinstance(service, ClosingProvider): service.disconnect()
                service.shutdown(); service.server_close(); thread.join(timeout=3)
                if thread.is_alive(): errors.append('owned service thread did not exit')
            except BaseException as error: errors.append(str(error))
        self.budget.json('cleanup.json', {'owned_job_ids': self.jobs, 'errors': errors,
                                        'closed_jobs': closed_jobs, 'fixture_admissions': self.ledger.count}, final=True)
        if errors: raise ValueError('owned cleanup failed: ' + '; '.join(errors))


def assess(root, science, budget, reason, error):
    result = {'schema': 'wsp-diagnostic-outcome/1', 'status': 'INCONCLUSIVE', 'outer_stop': reason, 'error': error,
              'ended_at_unix': time.time(),
              'observability': 'Native stop and completed-usage evidence, not direct request-admission telemetry; review required.'}
    if (root / 'attempt.json').is_file():
        attempt = strict_json((root / 'attempt.json').read_bytes())
        result['wall_seconds_including_cleanup'] = result['ended_at_unix'] - attempt['started_at_unix']
    try:
        cleanup = strict_json((root / 'cleanup.json').read_bytes())
        fixture = strict_json((root / 'fixture_state.json').read_bytes())
        result['fixture_admissions'] = fixture.get('admitted')
        if (cleanup.get('errors') != [] or type(fixture.get('admitted')) is not int or not 0 <= fixture['admitted'] <= 10 or
                cleanup.get('fixture_admissions') != fixture['admitted'] or not cleanup.get('closed_jobs') or
                any(j['state'] not in ('COMPLETE', 'ERROR') for j in cleanup['closed_jobs'])):
            raise ValueError('owned cleanup or host fixture count not established')
        events, trace = collect_native_evidence(root, write=False)
        budget.json('editor_response_evidence.json', trace)
        result['completion'] = response_completion(events, trace, 50000, frozen=False,
                              exit_code=trace['exit_code'], forced_stop=reason, require_native_stop=True)
        usage = actual_usage(events)
        if error or usage.get('distinct_snapshots', 0) < 2 or not usage.get('cached_input_tokens'):
            raise ValueError('missing genuine cached multiresponse evidence or driver error')
        invocation = strict_json((root / 'editor_invocation.json').read_bytes())
        proof = {'schema': 'wsp-live-actual-usage-proof/3', 'runtime': 'isolated_codex_exec', 'synthetic': False,
                 'case': 'cached_multiresponse_actual_total', 'runtime_identity': invocation['runtime_identity'],
                 'policy_sha256': json_hash(RESPONSE_POLICY), 'scientific_threshold_tokens': science['budgets']['editor_agent_tokens'],
                 'diagnostic_threshold_tokens': 50000, 'threshold_rationale': 'One reviewed50k text diagnostic of the same native mechanism proposed for200k scientific use.',
                 'diagnostic_differences': diagnostic_differences(science, invocation), 'evidence_directory': str(root),
                 'native_evidence_sha256': json_hash(trace), 'observed_actual_usage': usage}
        budget.json('live_proof.proposed.json', proof)
        result.update(status='NATIVE_STOP_OBSERVED_REVIEW_REQUIRED', proof_sha256=sha256(root / 'live_proof.proposed.json'))
    except (ValueError, KeyError, OSError) as caught:
        result['assessment_error'] = str(caught)
    budget.json('outcome.json', result, final=True)
    return result


def run(manifest_file, approval_file, *, backend_factory=HostBackend):
    manifest, science = approve_manifest(manifest_file, approval_file)
    root = Path(manifest['run_dir']); root.mkdir(mode=0o700, parents=False, exist_ok=False)
    budget = EvidenceBudget(root)
    budget.json('attempt.json', {'attempt': 1, 'started_at_unix': time.time(), 'manifest_sha256': json_hash(manifest),
                               'approval_sha256': sha256(approval_file), 'retries': 0})
    os.environ['TAO_STATE_DIR'] = str(root / 'job-records')
    backend = backend_factory(manifest, budget)
    reason, error = drive(backend)
    return assess(root, science, budget, reason, error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='verb', required=True)
    prep = commands.add_parser('prepare')
    for name in ('scientific-preflight', 'output', 'run-dir', 'codex-binary', 'auth-file'): prep.add_argument('--' + name, required=True)
    execute = commands.add_parser('run')
    execute.add_argument('--manifest', required=True); execute.add_argument('--approval', required=True)
    args = parser.parse_args()
    if args.verb == 'prepare':
        result = prepare(args.scientific_preflight, args.output, args.run_dir, args.codex_binary, args.auth_file)
        print(json.dumps({'manifest_sha256': result['manifest_sha256'], 'execution_authorized': False}))
    else:
        result = run(args.manifest, args.approval)
        print(json.dumps(result)); raise SystemExit(0 if result['status'] == 'NATIVE_STOP_OBSERVED_REVIEW_REQUIRED' else 1)


if __name__ == '__main__':
    main()
