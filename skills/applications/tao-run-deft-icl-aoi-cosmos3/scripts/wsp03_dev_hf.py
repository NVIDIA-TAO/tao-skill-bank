# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One dev-only frozen-HF subprocess with partial-answer degeneration stop.

No generation setting changes, patched model code or answer substitution.
The same frozen worker is used for PARITY64 and nominated FULL2917 passes.
This small local wrapper has no service, Docker access, retries or job ledger.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from wsp03_fast import read_rows, require_dev, evaluator_for, degenerate, summary, verify_model
from wsp03_worker import digest, validate_requests
from wsp03_bindings import file_hash
from wsp03_io import json_write

APP = Path(__file__).resolve().parent


def run(cfg):
    if cfg.get('split') not in ('development', 'explore'):
        raise ValueError('development-only frozen HF wrapper')
    sizes = {'PARITY64': 64, 'S1': 64, 'S2': 512, 'FULL2917': 2917}
    requests = read_rows(cfg['requests'])
    validate_requests(requests, 'generation', 'development')
    require_dev([row['id'] for row in requests], read_rows(cfg['development_ids']))
    if cfg['stage'] not in sizes or len(requests) != sizes[cfg['stage']]:
        raise ValueError('exact preregistered development stage size required')
    if cfg.get('wall_hours', 6) != 6 or cfg.get('token_target') is not None:
        raise ValueError('six-hour tracks and unlimited editor tokens required')
    if cfg['stage'] == 'FULL2917':
        receipt = json.loads(Path(cfg['nominations']).read_text())
        if (cfg['candidate_id'] not in receipt['nominees'] or len(receipt['nominees']) > 3
                or receipt.get('required_backend') != 'original_frozen_hf'
                or receipt.get('reuse_fast_predictions') is not False):
            raise ValueError('FULL2917 requires top-three S2 nomination')
    if cfg['stage'] == 'S2':
        receipt = json.loads(Path(cfg['s1_receipt']).read_text())
        if not receipt.get('advance') or receipt.get('candidate_id') != cfg['candidate_id']:
            raise ValueError('S2 requires this candidate S1 promotion')
    output = Path(cfg['output'])
    output.mkdir(parents=True, exist_ok=False)
    evaluator = evaluator_for(cfg)
    identity = verify_model(cfg)
    if cfg['stage'] == 'FULL2917':
        binding = receipt.get('candidate_bindings', {}).get(cfg['candidate_id'])
        if not binding or any(binding.get(key) != identity[key] for key in identity):
            raise ValueError('nominated candidate/base identity changed before FULL2917')
    worker = APP/'wsp03_worker.py'
    command = [cfg.get('hf_python', '/workspace/.venv/bin/python'), str(worker),
               '--model', cfg['model'], '--requests', cfg['requests'],
               '--output', str(output/'worker'), '--mode', 'generation', '--split', cfg['split']]
    if cfg.get('candidate'):
        command += ['--candidate', cfg['candidate']]
    profile = {'schema': 'wsp03-dev-hf/1', 'backend': 'original_frozen_hf',
               'stage': cfg['stage'], 'candidate_id': cfg['candidate_id'],
               'requests_sha256': digest(requests), **identity,
               'worker_sha256': file_hash(worker), 'command': command,
               'config_sha256': digest(cfg), 'generation_settings_changed': False, 'max_new_tokens': 4096, 'do_sample': False,
               'track_started_at': cfg['track_started_at'], 'retry_allowed': False}
    json_write(output/'attempt.json', profile)
    rows, status, proc = [], 'FAILED', None
    started = time.time()
    deadline = float(cfg['track_started_at'])+6*3600
    cursor = 0
    try:
        if started >= deadline:
            status = 'ABANDONED_WALL'
        else:
            with (output/'stdout.log').open('x') as stdout, (output/'stderr.log').open('x') as stderr:
                proc = subprocess.Popen(command, stdout=stdout, stderr=stderr)
                while True:
                    path = output/'worker'/'raw_answers.jsonl'
                    if path.exists():
                        with path.open() as stream:
                            stream.seek(cursor)
                            while True:
                                before = stream.tell()
                                line = stream.readline()
                                if not line.endswith('\n'):
                                    cursor = before
                                    break
                                rows.append(json.loads(line))
                                cursor = stream.tell()
                    if degenerate(rows, evaluator):
                        status = 'ABANDONED_DEGENERATE'
                        break
                    if time.time() >= deadline:
                        status = 'ABANDONED_WALL'
                        break
                    code = proc.poll()
                    if code is not None:
                        if code == 0 and len(rows) == len(requests):
                            status = 'COMPLETE'
                        break
                    time.sleep(.1)
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        result = dict(profile, status=status, n=len(rows), planned_n=len(requests),
                      raw_directory=str(output/'worker'), returncode=proc.returncode if proc else None,
                      summary=summary(rows, evaluator), ended_at=time.time(),
                      wall_seconds=time.time()-started, partial_results_preserved=True,
                      in_flight_answer_at_abandonment='not retried')
        json_write(output/'receipt.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    args = parser.parse_args()
    print(json.dumps(run(json.loads(Path(args.config).read_text())), indent=2))
