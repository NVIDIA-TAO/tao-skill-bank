#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit opt-in CPU Docker access probe. Tiny synthetic tensors/output only.

Uses an already cached Ubuntu image, two CPU/four GiB, network none and runc.
No agent, GPU, real model, package install, image pull or main-suite path read.
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from suite_runtime import Docker, json_write
from wsp_prepare import GENERIC_IMAGE
from wsp_runtime import CapabilityServer, sandbox_mounts, submit_sandbox, tool_contract
from test_wsp import synthetic_boundary
from test_wsp_context import context_task
from wsp_context import build_context, verify_context

PROBE = '''import json, pathlib, sys
sys.path.insert(0, '/tools')
from wsp_client import call
from wsp_isolation import probe
checks = probe()
context = pathlib.Path('/model_context')
manifest = json.loads((context / 'manifest.json').read_text())
assert manifest['schema'] == 'wsp-model-context/1'
assert json.loads((context / 'tokenizer.json').read_text())['vocabulary']['a'] == 0
assert json.loads((context / 'inference.json').read_text())['frozen_profile']['prompt'] == 'Frozen synthetic prompt'
for name in [*manifest['files'], 'manifest.json', 'forbidden-addition']:
    try:
        (context / name).write_text('forbidden')
        raise AssertionError('model context writable')
    except OSError:
        pass
assert not any(p.suffix == '.safetensors' for p in context.iterdir())
assert '/home/secret' not in ''.join(p.read_text() for p in context.iterdir())
checks['context_readable_readonly_no_payload_or_host_paths'] = True
assert pathlib.Path('/development/gold.jsonl').is_file()
try:
    pathlib.Path('/development/forbidden-write').write_text('x')
    raise AssertionError('development writable')
except OSError:
    checks['development_readonly'] = True
metadata = call({'op':'metadata','offset':0,'limit':10})
assert metadata['ok']
values = call({'op':'values','tensor':'layer.weight','indices':[0,4]})
assert values['result']['values'] == [1.0,5.0]
checks['actual_synthetic_tensor_read'] = True
for request in [
    {'op':'holdout'},
    {'op':'forward','candidate':'base','prompt':'change'},
    {'op':'forward','candidate':'../holdout'},
    {'op':'values','tensor':'/etc/passwd','indices':[0]},
    {'op':'exec','code':'import torch'},
]:
    response = call(request)
    assert not response['ok']
    assert 'holdout-synthetic' not in json.dumps(response)
checks['capability_negatives'] = True
patch = {'schema':'wsp-patch/1','base_fingerprint':metadata['result']['base_fingerprint'],
         'hypothesis':'Synthetic CPU boundary check, not a scientific weight candidate.',
         'edits':[{'tensor':'layer.weight','indices':[0,4],'values':[2.5,3.0]}]}
candidate = call({'op':'candidate','patch':patch,'code':pathlib.Path('/probe.py').read_text()})
assert candidate['ok'] and candidate['result']['changed_elements'] == 2
identity = candidate['result']['candidate']
assert call({'op':'forward','candidate':'base'})['ok']
assert call({'op':'forward','candidate':identity})['ok']
assert not call({'op':'forward','candidate':identity})['ok']
checks['bounded_synthetic_forwards'] = True
assert call({'op':'freeze','candidate':identity,'rationale':'Synthetic CPU boundary proof.'})['ok']
assert not call({'op':'status'})['ok']
checks['freeze_closes_editor'] = True
print(json.dumps({'schema':'wsp-cpu-boundary-probe/1','checks':checks,'passed':all(checks.values()),
                  'real_model_forwards':0,'real_weight_patches':0,'gpu_launches':0,'agent_launches':0}), flush=True)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='new, dedicated evidence directory')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.environ['TAO_STATE_DIR'] = str(output / 'job-records')
    boundary, calls = synthetic_boundary(output)
    context = context_task(output)
    context['model_context'] = build_context(context, output / 'model-context')
    boundary.task['model_context'] = context['model_context']
    channel = output / 'channel'
    channel.mkdir(mode=0o700)
    server = CapabilityServer(channel / 'wsp.sock', boundary)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    contract = output / 'tool_contract.json'
    json_write(contract, tool_contract(boundary))
    probe_file = output / 'probe.py'
    probe_file.write_text(PROBE)
    runtime = Docker(output / 'jobs', GENERIC_IMAGE)
    mounts = sandbox_mounts(boundary.views / 'development', contract, channel / 'wsp.sock',
                            model_context=verify_context(context), probe_file=probe_file)
    job = None
    harvest_job = None
    try:
        job = submit_sandbox(runtime, 'wsp-cpu-boundary-probe', ['/probe.py'], mounts)
        json_write(output / 'job.json', job)
        terminal = runtime.wait(job['job_id'], time.time() + 45)
        text = runtime.logs(job['job_id'], 10000, timestamps=True)
        (output / 'probe.timestamped.log').write_text(text)
        json_write(output / 'terminal.json', terminal)
        # Record actual native controls without importing unrelated containers.
        import subprocess
        inspect = json.loads(subprocess.check_output(['docker', 'inspect', job['job_id']], text=True))[0]
        controls = {k: inspect['HostConfig'].get(k) for k in ('NetworkMode', 'ReadonlyRootfs', 'Memory', 'MemorySwap', 'NanoCpus',
                                                            'Runtime', 'CapDrop', 'SecurityOpt', 'DeviceRequests', 'PidsLimit', 'Tmpfs')}
        controls['mounts'] = [{k: row.get(k) for k in ('Type', 'Source', 'Destination', 'RW')} for row in inspect['Mounts']]
        json_write(output / 'native_controls.json', controls)
        # No Codex binary or auth is mounted. This ordinary Python subprocess
        # verifies the tmpfs/exit/harvest lifecycle used by a future editor.
        dummy = 'from pathlib import Path; Path("/results/synthetic.txt").write_text("synthetic CPU child only"); print("synthetic child complete",flush=True)'
        harvest_job = submit_sandbox(runtime, 'wsp-cpu-harvest-probe',
                                     ['/tools/wsp_isolation.py', 'editor-entry', '/usr/bin/python3', '-c', dummy], mounts)
        json_write(output / 'harvest_job.json', harvest_job)
        child_exit = None
        deadline = time.time() + 30
        while time.time() < deadline:
            state = runtime.status(harvest_job['job_id'])
            if state['state'] not in ('RUNNING', 'PENDING'):
                break
            check = subprocess.run(['docker', 'exec', harvest_job['job_id'], '/usr/bin/python3', '-c',
                                    'from pathlib import Path; p=Path("/results/editor_exit.json"); print(p.read_text() if p.exists() else "null")'], capture_output=True, text=True)
            if check.returncode == 0:
                child_exit = json.loads(check.stdout)
            if child_exit is not None:
                break
            time.sleep(.25)
        if child_exit != {'exit_code': 0}:
            raise RuntimeError('synthetic subprocess harvest sentinel missing')
        assert runtime.status(harvest_job['job_id'])['state'] == 'RUNNING'
        with (output / 'synthetic_results.tar').open('wb') as stream:
            subprocess.run(['docker', 'cp', harvest_job['job_id'] + ':/results/.', '-'], stdout=stream, check=True)
        subprocess.run(['docker', 'exec', harvest_job['job_id'], '/usr/bin/python3', '-c',
                        'from pathlib import Path; Path("/tmp/harvested").touch()'], check=True)
        harvest_terminal = runtime.wait(harvest_job['job_id'], time.time() + 10)
        json_write(output / 'harvest_terminal.json', harvest_terminal)
        (output / 'harvest.timestamped.log').write_text(runtime.logs(harvest_job['job_id'], 10000, timestamps=True))
        boundary.export()
        success = terminal['state'] == 'COMPLETE' and harvest_terminal['state'] == 'COMPLETE' and len(calls) == 2 and all(c[1] == 'development' for c in calls)
        summary = {'passed': success, 'job_id': job['job_id'], 'backend_state': terminal['state'],
                   'synthetic_forward_calls': len(calls), 'real_model_forwards': 0, 'real_weight_patches': 0,
                   'gpu_launches': 0, 'agent_launches': 0, 'native_controls': 'native_controls.json',
                   'harvest_job_id': harvest_job['job_id'], 'synthetic_subprocess_archive_before_exit': True,
                   'scope': 'actual sandbox boundary; API relay/editor execution and real model remain unvalidated'}
        json_write(output / 'summary.json', summary)
        print(json.dumps(summary))
        if not success:
            print(text)
            raise SystemExit(1)
    finally:
        if job and runtime.status(job['job_id'])['state'] in ('RUNNING', 'PENDING'):
            runtime.cancel(job['job_id'])
        if harvest_job and runtime.status(harvest_job['job_id'])['state'] in ('RUNNING', 'PENDING'):
            runtime.cancel(harvest_job['job_id'])
        server.shutdown(); server.server_close(); thread.join()
        boundary.close()


if __name__ == '__main__':
    main()
