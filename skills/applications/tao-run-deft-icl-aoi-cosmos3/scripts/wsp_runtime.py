#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WSP native inference adapter and restricted editor/container capabilities.

Nothing launches at import. Real execution requires both a hash-bound review
approval and a separate primary-GPU-release record. CPU probes use fake tensors
and the same sandbox flags, with no agent invocation or GPU device requests.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import shlex
import shutil
import socket
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path

from suite_runtime import BANK, Docker, json_write
from tht_contract import external_evaluator, json_hash, load_jsonl, sha256
from wsp_boundary import Boundary
from wsp_weights import catalog, clone, finite_tensor, inventory, regular_file, strict_json
from wsp_isolation import UnixProvider
from wsp_context import verify_context
from wsp_usage import RESPONSE_POLICY, actual_usage, editor_command, response_snapshots, runtime_identity, validate_live_proof

HERE = Path(__file__).parent


def load_preflight(path):
    preflight = strict_json(Path(path).read_bytes())
    task = preflight['task']
    if task['schema'] != 'wsp-preflight/1' or preflight['preflight_sha256'] != json_hash(task):
        raise ValueError('preflight hash mismatch')
    for name, digest in task['code_hashes'].items():
        if sha256(regular_file(HERE, name)) != digest:
            raise ValueError('reviewed code changed: ' + name)
    verify_context(task)
    return preflight


def approval_gate(preflight, approval_file, release_file):
    digest = preflight['preflight_sha256']
    approval, release = (strict_json(Path(p).read_bytes()) for p in (approval_file, release_file))
    for record in (approval, release):
        if record.get('preflight_sha256') != digest or not record.get('approved_by') or not record.get('approved_at'):
            raise ValueError('explicit review/release for this exact proposal required')
    if approval.get('approved_for_real_weights') is not True or release.get('primary_gpu_released') is not True:
        raise ValueError('CPU preparation is not real-experiment approval')
    if release.get('host') != socket.gethostname() or release.get('gpu_ids') != ['0']:
        raise ValueError('release must identify this host and device0')
    return {'approval_sha256': sha256(approval_file), 'release_sha256': sha256(release_file)}


def verify_views(task):
    views = Path(task['views'])
    split = strict_json((views / 'split.json').read_bytes())
    raw = dict(split)
    digest = raw.pop('split_sha256')
    if digest != task['split_sha256'] or json_hash(raw) != digest:
        raise ValueError('split changed')
    for role in ('development', 'holdout'):
        detail = split['roles'][role]
        if detail['view_sha256'] != task['view_hashes'][role] or json_hash(detail['files']) != detail['view_sha256']:
            raise ValueError('role fingerprint mismatch')
        if {p.name for p in (views / role).iterdir()} != set(detail['files']):
            raise ValueError('unexpected role files')
        for name, expected in detail['files'].items():
            if sha256(regular_file(views / role, name)) != expected:
                raise ValueError('role bytes changed')


class NativeForward:
    """Reuse the M3 model-owned inference mechanics and pinned external scorer."""
    def __init__(self, task, runtime):
        self.task, self.runtime = task, runtime

    def __call__(self, model, role, directory, deadline):
        if time.time() + 15 >= deadline:
            raise ValueError('forward deadline exhausted before native launch')
        verify_views(self.task)
        inputs = directory / 'input'
        inputs.mkdir()
        queries = load_jsonl(Path(self.task['views']) / role / 'queries.jsonl')
        requests = []
        for query in queries:
            source = regular_file(Path(self.task['views']) / role, query['image'])
            if sha256(source) != query['image_sha256']:
                raise ValueError('image view changed')
            shutil.copyfile(source, inputs / query['image'])
            requests.append({'id': query['id'], 'messages': [{'role': 'user', 'content': [
                {'type': 'image', 'image': '/input/' + query['image']},
                {'type': 'text', 'text': self.task['inference']['prompt']}]}], 'references': []})
        json_write(inputs / 'requests.json', requests)
        worker = HERE / 'inference_worker.py'
        if sha256(worker) != self.task['code_hashes']['inference_worker.py']:
            raise ValueError('frozen worker changed')
        started = time.time()
        job = self.runtime.submit('wsp-' + role, ['/runtime/inference_worker.py', '--model', '/model',
                                  '--requests', '/input/requests.json', '--output', '/results', '--seed', str(self.task['seed'])],
                                  mounts=[(model, '/model', True), (inputs, '/input', True), (worker, '/runtime/inference_worker.py', True)],
                                  gpu=True, network='none', extra_args=['--read-only', '--tmpfs', '/tmp:rw,size=1g,mode=1777',
                                                                     '--cpus', str(self.task['native_runtime']['cpus']),
                                                                     '--memory', str(self.task['native_runtime']['memory_bytes']), '--pids-limit', '256'],
                                  on_open=lambda job_id: json_write(directory / 'job.json', {'job_id': job_id}),
                                  launch_timeout=max(1, min(60, int(deadline - time.time() - 15))))
        json_write(directory / 'job.json', job)
        # Reserve15s for the2s polling interval and Docker's10s stop grace.
        terminal = self.runtime.wait(job['job_id'], deadline - 15)
        result_dir = Path(job['results_dir'])
        json_write(directory / 'terminal.json', terminal)
        raw_file = result_dir / 'raw_answers.jsonl'
        raw = load_jsonl(raw_file) if raw_file.exists() else []
        native = terminal.get('native') or self.runtime.status(job['job_id']).get('native', {})
        allocated = None
        try:
            def timestamp(value):
                # Docker emits nanoseconds; datetime accepts a truncated fraction.
                return datetime.datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
            allocated = timestamp(native['FinishedAt']) - timestamp(native['StartedAt'])
        except (KeyError, ValueError):
            pass
        profile_path = result_dir / 'runtime_profile.json'
        profile = strict_json(profile_path.read_bytes()) if profile_path.exists() else {}
        comparable = {k: v for k, v in profile.items() if k not in ('model_path', 'model_load_seconds')}
        required = {'dtype': 'bfloat16', 'attention': 'sdpa', 'do_sample': False, 'max_new_tokens': 4096, 'seed': self.task['seed']}
        profile_ok = all(profile.get(k) == v for k, v in required.items())
        complete = terminal['state'] == 'COMPLETE' and (result_dir / 'completion.json').is_file() and profile_ok and allocated is not None
        return {'raw': raw, 'complete': complete, 'profile_sha256': json_hash(self.task['inference']),
                'runtime_profile_sha256': json_hash(comparable), 'runtime_profile': comparable,
                'gpu_wall_seconds': allocated, 'wall_seconds': time.time() - started,
                'job': job, 'backend_state': terminal['state'],
                'gpu_time_method': 'one allocated GPU times Docker StartedAt-to-FinishedAt wall interval; not kernel-active time'}


class CapabilityHandler(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(15)
        limit = self.server.boundary.b['request_bytes']
        raw = self.rfile.readline(limit + 1)
        try:
            if len(raw) > limit or not raw.endswith(b'\n'):
                raise ValueError('request framing/size limit')
            request = strict_json(raw)
        except (ValueError, UnicodeError):
            boundary = self.server.boundary
            boundary.journal.change('rpc_rejected', {'bytes_read': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
            try:
                boundary.journal.reserve('rejected_rpc_admission', {'tool_requests': 1}, {})
            except ValueError:
                pass
            result = {'ok': False, 'error': 'invalid_request'}
        else:
            result = self.server.boundary.handle(request)
        try:
            self.wfile.write(json.dumps(result, allow_nan=False).encode() + b'\n')
        except (BrokenPipeError, ConnectionResetError):
            # Work stays spent and its result remains in the trusted journal.
            pass


class CapabilityServer(socketserver.UnixStreamServer):
    def __init__(self, path, boundary):
        self.boundary = boundary
        super().__init__(str(path), CapabilityHandler)
        os.chmod(path, 0o600)


def sandbox_mounts(development, tool_contract, capability_socket, *, model_context=None, codex_binary=None, auth_file=None, provider_socket=None, probe_file=None):
    mounts = [(Path('/usr/bin/python3.12'), '/usr/bin/python3', True),
              (Path('/usr/lib/python3.12'), '/usr/lib/python3.12', True),
              (Path('/usr/lib/x86_64-linux-gnu'), '/usr/lib/x86_64-linux-gnu', True),
              (Path('/etc/ssl/certs'), '/etc/ssl/certs', True),
              (Path(development), '/development', True),
              (Path(tool_contract), '/task/tool_contract.json', True),
              (HERE / 'wsp_client.py', '/tools/wsp_client.py', True),
              (HERE / 'wsp_isolation.py', '/tools/wsp_isolation.py', True),
              (HERE / 'codex_capture.py', '/runtime/codex_capture.py', True),
              (Path(capability_socket), '/channel/wsp.sock', True)]
    if model_context is not None:
        mounts.append((Path(model_context), '/model_context', True))
    # Only generic image packages, never the host venv or model libraries.
    for entry in sys.path:
        path = Path(entry)
        if path.name == 'site-packages':
            for name in ('PIL', 'pillow.libs'):
                if (path / name).exists():
                    mounts.append((path / name, '/opt/generic/' + name, True))
    if codex_binary:
        binary = Path(codex_binary).resolve(strict=True)
        mounts.append((binary, '/usr/local/bin/codex', True))
        for name in ('codex-code-mode-host', 'codex-exec-server'):
            mounts.append((binary.parent / name, '/usr/local/bin/' + name, True))
    if auth_file:
        mounts.append((Path(auth_file), '/codex/auth.json', True))
    if provider_socket:
        mounts.append((Path(provider_socket), '/channel/provider.sock', True))
    if probe_file:
        mounts.append((Path(probe_file), '/probe.py', True))
    return mounts


def submit_sandbox(runtime, stage, command, mounts):
    """CPU-only Docker four-verb consumer with NO writable host bind mounts.

    Uses the shared job-record before launch. Only named, read-only capabilities
    enter; all editor writes live in bounded tmpfs and are exported as tar bytes.
    """
    if os.getuid() == 0:
        raise ValueError('non-root submitting uid required')
    subprocess.run(['docker', 'image', 'inspect', runtime.image], check=True, stdout=subprocess.DEVNULL)
    job_id = runtime.record('open', '--platform', 'docker', '--image', runtime.image, '--network-arch', 'cosmos-reason',
                            '--action', stage, '--storage-tier', 'A', '--results-root', str(runtime.root))
    record = json.loads(runtime.record('show', job_id))
    out = Path(record['results_dir'])
    out.mkdir(parents=True, exist_ok=True)
    cmd = ['docker', 'run', '-d', '--pull=never', '--runtime=runc', '--name', job_id, '--label', 'tao-job=' + job_id,
           '--user', f'{os.getuid()}:{os.getgid()}', '--network=none', '--read-only', '--cap-drop=ALL',
           '--security-opt=no-new-privileges', '--pids-limit=128', '--memory=4g', '--memory-swap=4g', '--cpus=2',
           '--shm-size=16m', '--ipc=private', '--ulimit', 'core=0',
           '--tmpfs', '/tmp:rw,nosuid,size=64m,mode=1777', '--tmpfs', '/codex:rw,nosuid,size=64m,mode=1777',
           '--tmpfs', '/task:rw,nosuid,size=256m,mode=1777', '--tmpfs', '/results:rw,nosuid,size=16m,mode=1777',
           '--workdir=/task', '-e', 'HOME=/tmp/home', '-e', 'CODEX_HOME=/codex', '-e', 'USER=tao', '-e', 'LOGNAME=tao',
           '-e', 'PYTHONPATH=/opt/generic', '-e', 'PYTHONNOUSERSITE=1', '-e', 'PYTHONDONTWRITEBYTECODE=1',
           '-e', 'NVIDIA_VISIBLE_DEVICES=void', '--log-driver=json-file', '--log-opt', 'max-size=128m', '--log-opt', 'max-file=1']
    targets = set()
    allowed_targets = {'/usr/bin/python3', '/usr/lib/python3.12', '/usr/lib/x86_64-linux-gnu', '/etc/ssl/certs',
                       '/development', '/model_context', '/task/tool_contract.json', '/tools/wsp_client.py', '/tools/wsp_isolation.py',
                       '/runtime/codex_capture.py', '/channel/wsp.sock', '/channel/provider.sock', '/codex/auth.json',
                       '/usr/local/bin/codex', '/usr/local/bin/codex-code-mode-host', '/usr/local/bin/codex-exec-server',
                       '/opt/generic/PIL', '/opt/generic/pillow.libs', '/probe.py'}
    for source, target, readonly in mounts:
        source = Path(source).resolve(strict=True)
        if not readonly or target not in allowed_targets or target in targets or str(source) in ('/', '/home', '/home/ubuntu', '/home/ubuntu/workspace', '/var/run/docker.sock'):
            runtime.record('mark', job_id, '--state', 'ERROR', '--err-class', 'ERR_PROGRAM', '--message', 'sandbox mount rejected')
            raise ValueError('sandbox mount rejected')
        targets.add(target)
        cmd += ['--mount', f'type=bind,src={source},dst={target},readonly']
    cmd += ['--entrypoint', '/usr/bin/python3', runtime.image, *command]
    lint_cmd = [s.replace('prefill_token_weight=1,', 'prefill_weight=1,').replace('sampling_token_weight=1}', 'sampling_weight=1}') for s in cmd]
    subprocess.run([sys.executable, str(BANK / 'scripts/redact_secrets.py'), 'lint', '-'], input=shlex.join(lint_cmd), text=True, check=True)
    json_write(out / 'launch.json', {'job_id': job_id, 'argv': cmd, 'submitted_at_unix': time.time()})
    try:
        cid = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE, timeout=60).strip()
    except BaseException as error:
        # Reconcile only this minted ID. Never touch another container/session.
        observed = runtime.status(job_id)
        if observed['state'] in ('RUNNING', 'PENDING'):
            runtime.cancel(job_id)
        else:
            runtime.record('mark', job_id, '--state', 'ERROR', '--err-class', 'ERR_PROGRAM', '--message', 'sandbox launch failed')
        (out / 'launch_error.txt').write_text(str(error) + '\n' + str(getattr(error, 'stderr', '') or ''))
        raise
    runtime.record('mark', job_id, '--state', 'RUNNING', '--backend-ref', cid)
    return {'job_id': job_id, 'container_id': cid, 'results_dir': str(out)}


def tool_contract(boundary):
    return {'schema': 'wsp-toolset/1', 'base_fingerprint': boundary.base_manifest['fingerprint'],
            'model_context': {'path': '/model_context', 'readonly': True,
                              'bundle_sha256': boundary.task.get('model_context', {}).get('bundle_sha256'),
                              'permission': 'Factual context only; all configuration and inference settings stay frozen.'},
            'response_budget_policy': RESPONSE_POLICY,
            'budgets': boundary.b, 'operations': {
                'metadata': {'op': 'metadata', 'offset': 0, 'limit': 64},
                'values': {'op': 'values', 'tensor': '<exact name>', 'indices': ['<unique integer flat index>']},
                'candidate': {'op': 'candidate', 'code': '<actual Python construction source; captured, not executed by service>',
                              'patch': {'schema': 'wsp-patch/1', 'base_fingerprint': boundary.base_manifest['fingerprint'],
                                        'hypothesis': '<editor chooses>', 'edits': [{'tensor': '<editor chooses>', 'indices': ['<flat index>'], 'values': ['<finite replacement>'] }]}},
                'forward': {'op': 'forward', 'candidate': 'base or returned cNNNN'},
                'freeze': {'op': 'freeze', 'candidate': '<returned cNNNN>', 'rationale': '<development only>'},
                'status': {'op': 'status'}},
            'semantics': 'All forwards use the same complete development view. No holdout capability. No arbitrary path, code execution, model loading or inference setting is accepted.'}


def initialize_run(task, approval_evidence):
    """Only after approval: verify tensor bytes, independent copy, full finiteness."""
    root = Path(task['run_dir'])
    if root.exists():
        raise ValueError('run directory exists; do not reset caps or start another editor')
    root.mkdir(parents=True, mode=0o700)
    json_write(root / 'initialization.json', {'status': 'STARTED', 'started_at_unix': time.time(),
                                             'host': socket.gethostname(), 'task_sha256': json_hash(task)})
    json_write(root / 'approval_evidence.json', approval_evidence)
    json_write(root / 'preflight.json', {'task': task, 'preflight_sha256': json_hash(task)})
    verify_views(task)
    source = Path(task['model'])
    if sha256(source / 'tao_conversion_provenance.json') != task['conversion_provenance_sha256']:
        raise ValueError('base provenance changed')
    expected_names = {f['name'] for f in task['declared_base']['files']} | {'tao_conversion_provenance.json'}
    if {p.name for p in source.iterdir()} != expected_names:
        raise ValueError('base inventory changed')
    for file in task['declared_base']['files']:
        path = regular_file(source, file['name'])
        if path.stat().st_size != file['size'] or sha256(path) != file['sha256']:
            raise ValueError('base content fingerprint mismatch')
    if shutil.disk_usage(root.parent).free < task['budgets']['storage_bytes']:
        raise ValueError('insufficient free disk for frozen retained-artifact ceiling')
    manifest = inventory(source)
    clone(source, root / 'base', manifest)
    for meta in catalog(root / 'base')['tensors'].values():
        finite_tensor(root / 'base', meta)
    for path in (root / 'base').iterdir():
        path.chmod(0o400)
    json_write(root / 'base_manifest.json', manifest)
    return root, manifest


def run_approved(preflight_file, approval_file, release_file, *, usage_proof, codex_binary, auth_file):
    preflight = load_preflight(preflight_file)
    evidence = approval_gate(preflight, approval_file, release_file)
    task = preflight['task']
    if runtime_identity(task, codex_binary) != task['editor']['runtime_identity']:
        raise ValueError('actual editor runtime/binary/helper/code identity changed')
    evidence['live_actual_usage_proof'] = validate_live_proof(usage_proof, strict_json(Path(approval_file).read_bytes()), task)
    # Read-only resource admission; do not communicate with the primary harness.
    occupancy = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    if occupancy:
        raise ValueError('released GPU is occupied; stop without launch')
    for image in (task['framework_image'], task['generic_image']):
        subprocess.run(['docker', 'image', 'inspect', image], check=True, stdout=subprocess.DEVNULL)
    version = subprocess.check_output([codex_binary, '--version'], text=True).strip()
    if version != task['editor']['codex_version']:
        raise ValueError('Codex version needs renewed native-cap validation')
    if not Path(auth_file).is_file():
        raise ValueError('approved private auth file absent; never provide credentials in chat')
    if Path(task['run_dir']).exists():
        raise ValueError('run directory exists; preserve evidence and do not reset caps')
    try:
        root, manifest = initialize_run(task, evidence)
    except BaseException as error:
        root = Path(task['run_dir'])
        if (root / 'initialization.json').is_file() and not (root / 'journal.sqlite').exists():
            json_write(root / 'initialization_failure.json', {'error_type': type(error).__name__, 'error': str(error), 'ended_at_unix': time.time()})
        raise
    os.environ['TAO_STATE_DIR'] = str(root / 'job-records')
    native = Docker(root / 'jobs', task['framework_image'])
    editor_runtime = Docker(root / 'editor-jobs', task['generic_image'])
    boundary = Boundary(root, task, root / 'base', manifest, task['views'], NativeForward(task, native), external_evaluator(task['evaluator']))
    channel = root / 'channel'
    channel.mkdir(mode=0o700)
    server = CapabilityServer(channel / 'wsp.sock', boundary)
    provider = UnixProvider(channel / 'provider.sock', task['editor']['api_hosts'])
    for service in (server, provider):
        threading.Thread(target=service.serve_forever, daemon=True).start()
    contract_file = root / 'tool_contract.json'
    json_write(contract_file, tool_contract(boundary))
    mounts = sandbox_mounts(Path(task['views']) / 'development', contract_file, channel / 'wsp.sock',
                            model_context=verify_context(task),
                            codex_binary=codex_binary, auth_file=auth_file, provider_socket=channel / 'provider.sock')
    command = editor_command(task, task['budgets']['editor_agent_tokens'])
    job = None
    log_process = None
    log_thread = None
    try:
        # Separate real-runtime access check before the fresh editor starts.
        probe = submit_sandbox(editor_runtime, 'wsp-access-probe', ['/tools/wsp_isolation.py', 'probe'], mounts)
        if editor_runtime.wait(probe['job_id'], time.time() + 45)['state'] != 'COMPLETE':
            raise ValueError('actual editor sandbox access probe failed')
        job = submit_sandbox(editor_runtime, 'wsp-editor', command, mounts)
        json_write(root / 'editor_job.json', job)
        editor_deadline = time.time() + task['budgets']['editor_wall_seconds']
        editor_exit = None
        observed_exit_at = None
        stop_reason = None
        log_limit_hit = threading.Event()
        log_process = subprocess.Popen(['docker', 'logs', '--follow', '--timestamps', job['job_id']], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        def capture_log():
            size = 0
            with (root / 'editor_events.timestamped.log').open('wb') as stream:
                while True:
                    data = log_process.stdout.read(65536)
                    if not data:
                        return
                    allowed = min(len(data), task['budgets']['editor_log_bytes'] - size)
                    stream.write(data[:allowed]); stream.flush(); size += allowed
                    if allowed < len(data) or size == task['budgets']['editor_log_bytes']:
                        log_limit_hit.set()
                        return
        log_thread = threading.Thread(target=capture_log, daemon=True)
        log_thread.start()
        def observe_editor(final=False):
            # The reader uses only this fresh container's token events. Capture
            # again after child exit so a short-lived final response is included.
            code = ('import json,runpy; from pathlib import Path; '
                    'runpy.run_path("/runtime/codex_capture.py")["capture_usage"](); '
                    'p=Path("/results/codex_usage_events.jsonl"); e=Path("/results/editor_exit.json"); '
                    'print(json.dumps({"events":[json.loads(x) for x in p.read_text().splitlines()] if p.is_file() else [],'
                    '"exit":json.loads(e.read_text()) if e.is_file() else None}))')
            observed = subprocess.run(['docker', 'exec', job['job_id'], '/usr/bin/python3', '-c', code], capture_output=True, text=True, timeout=15)
            if observed.returncode:
                raise ValueError('actual editor usage observation failed')
            record = strict_json(observed.stdout)
            if record['events']:
                usage = actual_usage(record['events'])
                boundary.record_agent_usage(usage, 'fresh_editor_raw_token_events', final=final)
                json_write(root / 'editor_usage_events.json', record['events'])
                snapshots = response_snapshots(record['events'])
                if any(total >= task['budgets']['editor_agent_tokens'] for _, total in snapshots[:-1]):
                    raise ValueError('additional completed model response after actual threshold')
            return record['exit']
        while editor_runtime.status(job['job_id'])['state'] in ('RUNNING', 'PENDING'):
            editor_exit = observe_editor()
            if editor_exit is not None:
                observed_exit_at = time.time()
                break
            state = boundary.journal.state()
            usage = state['agent_usage']
            if (state['phase'] not in ('DEVELOPMENT', 'FROZEN') or time.time() >= editor_deadline or
                    time.time() >= state['started_at_unix'] + task['budgets']['experiment_wall_seconds'] or log_limit_hit.is_set()):
                stop_reason = 'wall_log_or_state_limit'
                subprocess.run(['docker', 'exec', job['job_id'], '/usr/bin/python3', '-c', 'from pathlib import Path; Path("/tmp/editor-stop").touch()'], check=True, timeout=10)
                for _ in range(10):
                    editor_exit = observe_editor()
                    if editor_exit is not None:
                        observed_exit_at = time.time()
                        break
                    time.sleep(.5)
                break
            boundary.check_storage()
            time.sleep(1)
        # A threshold stop may have a nonzero native exit. Retain it; only trusted
        # response-admission/exit evidence can classify it as an accepted stop.
        observe_editor(final=editor_exit is not None and stop_reason is None)
        # Preserve raw tar bytes. Never extract editor-created paths/symlinks on
        # the trusted host. Both sources are bounded tmpfs, no host bind output.
        for name in ('task', 'results'):
            with (root / ('editor_' + name + '.tar')).open('wb') as stream:
                subprocess.run(['docker', 'cp', job['job_id'] + ':/' + name + '/.', '-'], stdout=stream, check=True)
        subprocess.run(['docker', 'exec', job['job_id'], '/usr/bin/python3', '-c', 'from pathlib import Path; Path("/tmp/harvested").touch()'], check=True, timeout=10)
        terminal = editor_runtime.wait(job['job_id'], time.time() + 10)
        log_thread.join(timeout=5)
        log_process.terminate(); log_process.wait(timeout=5)
        json_write(root / 'editor_terminal.json', {'backend': terminal, 'editor_exit': editor_exit, 'stop_reason': stop_reason,
                                                   'log_limit_hit': log_limit_hit.is_set()})
        if not editor_exit:
            raise ValueError('editor final exit is missing')
        if (observed_exit_at is None or observed_exit_at > editor_deadline or
                time.time() >= boundary.journal.state()['started_at_unix'] + task['budgets']['experiment_wall_seconds'] or log_limit_hit.is_set()):
            stop_reason = stop_reason or 'final_wall_or_log_limit'
        # Required trusted-host collector output. Neither the editor's writable
        # files, our polling loop nor CONNECT proxy can manufacture this proof.
        trace = strict_json((root / 'editor_response_evidence.json').read_bytes())
        live = evidence['live_actual_usage_proof']
        if (trace.get('invocation_id') != job['job_id'] or trace.get('runtime_identity_sha256') != live['runtime_identity_sha256'] or
                trace.get('collector_sha256') != live['collector_sha256']):
            raise ValueError('scientific admission trace is not bound to this invocation/runtime/collector')
        completion = boundary.accept_editor_completion(strict_json((root / 'editor_usage_events.json').read_bytes()), trace,
                                                       exit_code=editor_exit['exit_code'], forced_stop=stop_reason)
        if completion['status'] == 'HOLDOUT_ELIGIBLE':
            # Stop serving editor capabilities before any private evaluation.
            server.shutdown()
            boundary.holdout()
        else:
            boundary.journal.change('editor_incomplete', {'terminal': terminal}, lambda state: state.update(phase='ERROR'))
        return boundary.export()
    except BaseException as error:
        boundary.journal.change('run_failed', {'error_type': type(error).__name__, 'error': str(error)}, lambda state: state.update(phase='ERROR'))
        boundary.export()
        raise
    finally:
        # Preserve partial stdout and bounded tmpfs evidence on any failure.
        # No extraction of untrusted archives; no deletion or retry of an editor.
        if job and editor_runtime.status(job['job_id'])['state'] in ('RUNNING', 'PENDING'):
            for name in ('task', 'results'):
                destination = root / ('editor_' + name + '.tar')
                if not destination.exists():
                    with destination.open('xb') as stream:
                        subprocess.run(['docker', 'cp', job['job_id'] + ':/' + name + '/.', '-'], stdout=stream, stderr=subprocess.DEVNULL, timeout=20)
        if job and editor_runtime.status(job['job_id'])['state'] in ('RUNNING', 'PENDING'):
            editor_runtime.cancel(job['job_id'])
        if log_process and log_process.poll() is None:
            log_process.terminate()
            log_process.wait(timeout=5)
        if log_thread:
            log_thread.join(timeout=5)
        server.shutdown()
        server.server_close()
        provider.shutdown()
        provider.server_close()
        boundary.close()
