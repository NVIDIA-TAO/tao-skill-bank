#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Application entry for Maxwell's Cosmos-driven, bounded THT tool loop.

Preflight prepares native inputs without a model. Run is an in-container
worker, dispatched by the existing platform skill after its job record opens.
The original six-arm suite and generator/ICL addenda are not invoked.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

DATA_SKILL = Path(__file__).resolve().parents[3] / 'data/tao-in-context-learning'
ADAPTER = DATA_SKILL / 'scripts/simple_harness.py'
PROFILES = ('simple-harness-grid32-tools-guidance-native4096',
            'simple-harness-grid32-tools-guidance-cap128',
            'simple-harness-grid32-explore-cap128-v1')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal(path, value):
    """Never overwrite an existing review or launch receipt."""
    target = Path(path)
    if target.exists():
        if read(target) != value:
            raise ValueError('existing receipt differs; use a separate run directory')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def config(path):
    value = read(path)
    if value.get('profile') not in PROFILES:
        raise ValueError('this application entry requires the Cosmos grid + tools profile')
    return value


def adapter(action, path):
    subprocess.run([sys.executable, str(ADAPTER), action, '--config', str(path)], check=True)


def preflight(config_path, output):
    path = Path(config_path).resolve(strict=True)
    value = config(path)
    adapter('prepare', path)
    prepared_path = Path(value['output']) / 'prepared.json'
    prepared = read(prepared_path)
    if prepared['tool_round_limit'] != 10 or prepared['decision_owner'] != 'CosmosBackend':
        raise ValueError('native preparation differs from the Cosmos tool-loop contract')
    receipt = {
        'schema': 'tao-cosmos-simple-harness-preflight/1',
        'status': 'READY_FOR_REVIEW', 'config': str(path), 'config_sha256': sha(path),
        'prepared': str(prepared_path), 'prepared_sha256': sha(prepared_path),
        'entrypoint_sha256': sha(__file__), 'adapter_sha256': sha(ADAPTER),
        'profile': value['profile'], 'query_split': value['query_split'],
        'query_count': prepared['query_count'], 'decision_owner': 'Cosmos',
        'tools': ['get_class_examples', 'search_similar_examples'],
        'tool_policy': 'optional; chosen by Cosmos; one call per turn',
        'max_tool_rounds': 10, 'max_model_turns': 11,
        'generation': {'do_sample': False, 'max_new_tokens': prepared['max_new_tokens'], 'seed': 0},
        'format_retries': 0, 'generation_deviation': prepared['generation_deviation'],
        'prompt_deviation': prepared.get('prompt_deviation'),
        'gpu_smoke': 'not performed by preflight',
        'platform': 'docker; narrow mounts and record-before-launch are platform-owned',
    }
    seal(output, receipt)
    return {'status': receipt['status'], 'output': str(output), 'preflight_sha256': sha(output)}


def run(preflight_path, approval_path, job_id):
    frozen = read(preflight_path)
    approval = read(approval_path)
    if (frozen.get('schema') != 'tao-cosmos-simple-harness-preflight/1'
            or frozen.get('status') != 'READY_FOR_REVIEW'
            or approval.get('preflight_sha256') != sha(preflight_path)
            or not approval.get('approved_by') or not approval.get('approved_at')):
        raise ValueError('approval must bind the exact preflight; never manufacture approval')
    if not job_id or not job_id.strip():
        raise ValueError('platform job-record ID is required before worker execution')
    for path, expected in [(frozen['config'], frozen['config_sha256']),
                           (frozen['prepared'], frozen['prepared_sha256']),
                           (__file__, frozen['entrypoint_sha256']),
                           (ADAPTER, frozen['adapter_sha256'])]:
        if sha(path) != expected:
            raise ValueError('frozen input or implementation changed')
    value = config(frozen['config'])
    # Adapter run revalidates every source, model, cache, image and case seal.
    # Job ID is provenance supplied by the platform, not an isolation proof.
    output = Path(value['output'])
    launch = {
        'schema': 'tao-cosmos-simple-harness-launch/1',
        'preflight_sha256': sha(preflight_path),
        'approval_sha256': sha(approval_path), 'job_id': job_id,
    }
    key = hashlib.sha256(job_id.encode()).hexdigest()
    seal(output / 'application-launches' / (key + '.json'), launch)
    adapter('run', frozen['config'])
    return read(output / 'complete.json')


def status(results_dir):
    root = Path(results_dir)
    if (root / 'complete.json').is_file():
        value = read(root / 'complete.json')
        return {key: value.get(key) for key in
                ('status', 'profile', 'completed', 'invalid', 'model_calls', 'tool_calls')}
    return {'status': 'PREPARED' if (root / 'prepared.json').is_file() else 'NOT_PREPARED',
            'note': 'Artifact status only. Poll Docker status/logs for the live job.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    prepare = commands.add_parser('preflight')
    prepare.add_argument('--config', required=True)
    prepare.add_argument('--output', required=True)
    execute = commands.add_parser('run')
    execute.add_argument('--preflight', required=True)
    execute.add_argument('--approval', required=True)
    execute.add_argument('--job-id', required=True)
    inspect = commands.add_parser('status')
    inspect.add_argument('--results-dir', required=True)
    args = parser.parse_args()
    if args.action == 'preflight':
        result = preflight(args.config, args.output)
    elif args.action == 'run':
        result = run(args.preflight, args.approval, args.job_id)
    else:
        result = status(args.results_dir)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
