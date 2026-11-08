# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owner-only paired64 preparation/run. No protected paths or accuracy report.

prepare stages dev requests/CPU exports only; run is a separate explicit GPU
action after the owner releases the existing GPU lease. Native editors do not
need this Docker-capable helper.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
import threading
from wsp03_bindings import Recipe, ReferenceSelector, pinned, file_hash
from wsp03_worker import digest
from wsp03_io import json_write

APP = Path(__file__).resolve().parent


def select64(recipe, classes):
    """Frozen deterministic eight-per-class subset of the fixed512 explore."""
    if len(recipe.members['explore']) != 512 or len(classes) != 8:
        raise ValueError('probe requires fixed512 explore and eight configured classes')
    selected = []
    for label in classes:
        pool = [qid for qid in recipe.members['explore'] if recipe.rows[qid]['labels'] == [label]]
        pool.sort(key=lambda qid: (digest({'seed': 42, 'purpose': 'efficient-paired64', 'id': qid}), qid))
        if len(pool) < 8:
            raise ValueError('insufficient explore coverage for frozen stratification')
        selected.extend(pool[:8])
    return selected


def docker(cfg, command, mounts, *, gpu=False, python='/workspace/.venv/bin/python', log):
    args = ['docker', 'run', '--rm', '--network', 'none', '--shm-size', '8g',
            '--env', 'PYTHONPATH=/app']
    if gpu:
        args += ['--gpus', 'device='+str(cfg.get('gpu', 0))]
    for source, target, readonly in [(APP, '/app', True), *mounts]:
        args += ['--mount', 'type=bind,source='+str(source)+',target='+target+(',readonly' if readonly else '')]
    args += ['--entrypoint', python, cfg['image'], *command]
    with Path(log).open('x') as stream:
        proc = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT, timeout=cfg.get('timeout_seconds', 7200))
    if proc.returncode:
        raise RuntimeError('probe subprocess failed; preserve '+str(log))
    return args


def export(cfg):
    from wsp03_kernel import TensorStore
    from wsp03_ops import Operations
    work = Path(cfg['output'])
    work.mkdir(parents=True, exist_ok=False)
    shutil.copytree(cfg['candidate'], work/'candidate')
    store = TensorStore(cfg['model'], work)
    return Operations(store).export('candidate')


def prepare(cfg):
    output = Path(cfg['output']).resolve()
    output.mkdir(parents=True, exist_ok=False)
    recipe = Recipe(cfg['recipe']['path'], cfg['recipe']['sha256'])
    reference = json.loads(pinned(**cfg['reference_state']).read_text())
    recipe.selector = ReferenceSelector(recipe.rows, recipe.selector.embeddings,
        helpfulness=reference['helpfulness'], atlas=reference['atlas'], pool=recipe.members['development'])
    ids = select64(recipe, cfg['classes'])
    json_write(output/'development-ids.json', recipe.members['development'])
    json_write(output/'paired64-ids.json', ids)
    (output/'input').mkdir()
    seen = set()
    cases = []
    for case in cfg['cases']:
        name = case['name']
        if name in seen or not name.replace('-', '').isalnum() or case['profile'] not in ('P00', 'P09'):
            raise ValueError('unique safe case name and fixed P00/P09 profile required')
        seen.add(name)
        directory = output/'cases'/name
        directory.mkdir(parents=True)
        requests = [recipe.render(qid, recipe.references(qid, case['profile']), 'generation', case['profile']) for qid in ids]
        for relative, source in recipe.request_images(requests).items():
            target = output/'input'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and file_hash(target) != file_hash(source):
                raise ValueError('probe image name collision')
            if not target.exists():
                shutil.copyfile(source, target)
        json_write(directory/'requests.json', requests)
        shutil.copytree(case['candidate'], directory/'patch')
        manifest = json.loads((directory/'patch/manifest.json').read_text())
        candidate_hash = digest(manifest)
        if candidate_hash != case['candidate_manifest_sha256']:
            raise ValueError('probe must bind exact selected candidate manifest')
        export_cfg = {'model': '/model', 'candidate': '/probe/cases/'+name+'/patch',
                      'output': '/probe/cases/'+name+'/export-work'}
        json_write(directory/'export-config.json', export_cfg)
        docker(cfg, ['/app/wsp03_probe.py', 'export', '/probe/cases/'+name+'/export-config.json'],
            [(cfg['model'], '/model', True), (output, '/probe', False)],
            log=directory/'export.log')
        common = {'model': '/model', 'model_files': cfg['model_files'],
            'candidate': '/probe/cases/'+name+'/patch', 'candidate_id': case['candidate_id'],
            'requests': '/probe/cases/'+name+'/requests.json', 'development_ids': '/probe/development-ids.json',
            'image_root': '/input', 'classes': cfg['classes'], 'evaluator_directory': '/evaluator',
            'split': 'explore', 'stage': 'PARITY64', 'batch_size': 32, 'max_new_tokens': 4096,
            'wall_hours': 6, 'token_target': None, 'track_started_at': None}
        fast = dict(common, backend=cfg.get('backend', 'vllm'),
            exported_model='/probe/cases/'+name+'/export-work/exports/candidate',
            max_model_len=cfg.get('max_model_len', 16384), max_images=5,
            output='/probe/cases/'+name+'/fast')
        hf = dict(common, output='/probe/cases/'+name+'/hf')
        json_write(directory/'fast-config.json', fast)
        json_write(directory/'hf-config.json', hf)
        parity = {'fast': '/probe/cases/'+name+'/fast/raw_answers.jsonl',
            'frozen': '/probe/cases/'+name+'/hf/worker/raw_answers.jsonl',
            'fast_profile': '/probe/cases/'+name+'/fast/runtime_profile.json',
            'frozen_profile': '/probe/cases/'+name+'/hf/worker/runtime_profile.json',
            'base_inventory_sha256': digest(cfg['model_files']), 'development_ids': '/probe/development-ids.json',
            'classes': cfg['classes'], 'evaluator_directory': '/evaluator',
            'output': '/probe/cases/'+name+'/parity.json'}
        json_write(directory/'parity-config.json', parity)
        cases.append(dict(case, requests_sha256=digest(requests),
                          candidate_manifest_sha256=candidate_hash))
    result = {'schema': 'wsp03-probe-preparation/1', 'status': 'PREPARED', 'dev_only': True,
        'gpu_used': False, 'ids_sha256': digest(ids), 'cases': cases,
        'config_sha256': digest(cfg), 'image': cfg['image'], 'created_at': time.time()}
    json_write(output/'prepared.json', result)
    return result



def sample_gpu(path, stop):
    while not stop.is_set():
        try:
            proc = subprocess.run(['nvidia-smi', '--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total',
                                   '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
            item = {'time': time.time(), 'csv': proc.stdout.strip(), 'returncode': proc.returncode}
        except (OSError, subprocess.TimeoutExpired) as error:
            item = {'time': time.time(), 'error': repr(error)}
        with path.open('a') as stream:
            stream.write(json.dumps(item)+'\n')
        stop.wait(10)


def run(cfg):
    output = Path(cfg['output']).resolve(strict=True)
    prepared = json.loads((output/'prepared.json').read_text())
    if prepared['config_sha256'] != digest(cfg):
        raise ValueError('probe config changed after preparation')
    # An explicit separate owner action; never called by prepare.
    with (output/'RUN_ATTEMPT.json').open('x') as stream:
        stream.write(json.dumps({'started_at': time.time(), 'retry': False, 'config_sha256': digest(cfg)}))
    results = {}
    started = time.time()
    mounts = [(cfg['model'], '/model', True), (cfg['evaluator_directory'], '/evaluator', True),
              (output, '/probe', False), (output/'input', '/input', True)]
    status = 'INCOMPLETE'
    stop = threading.Event()
    sampler = threading.Thread(target=sample_gpu, args=(output/'gpu-samples.jsonl', stop), daemon=True)
    sampler.start()
    try:
        for case in prepared['cases']:
            name = case['name']
            directory = output/'cases'/name
            if digest(json.loads((directory/'requests.json').read_text())) != case['requests_sha256']:
                raise ValueError('prepared paired requests changed')
            for filename in ('fast-config.json', 'hf-config.json'):
                path = directory/filename
                value = json.loads(path.read_text())
                value['track_started_at'] = started
                json_write(path, value)
            interpreter = '/opt/wsp-vllm/bin/python' if cfg.get('backend', 'vllm') == 'vllm' else '/workspace/.venv/bin/python'
            docker(cfg, ['/app/wsp03_fast.py', 'generate', '/probe/cases/'+name+'/fast-config.json'],
                mounts, gpu=True, python=interpreter, log=directory/'fast.log')
            docker(cfg, ['/app/wsp03_dev_hf.py', '/probe/cases/'+name+'/hf-config.json'],
                mounts, gpu=True, log=directory/'hf.log')
            for backend in ('fast', 'hf'):
                value = json.loads((directory/backend/'receipt.json').read_text())
                if value['status'] != 'COMPLETE' or value['n'] != 64:
                    raise RuntimeError('paired64 incomplete: '+name+'/'+backend+' '+value['status'])
            docker(cfg, ['/app/wsp03_fast.py', 'parity', '/probe/cases/'+name+'/parity-config.json'],
                mounts, log=directory/'parity.log')
            parity = json.loads((directory/'parity.json').read_text())
            fast = json.loads((directory/'fast/receipt.json').read_text())
            hf = json.loads((directory/'hf/receipt.json').read_text())
            results[name] = {'profile': case['profile'], 'candidate_id': case['candidate_id'],
                'candidate_manifest_sha256': case['candidate_manifest_sha256'],
                'requests_sha256': case['requests_sha256'], 'actual_checkpoint_support': True,
                'agreement': parity, 'fast_backend': fast['backend'],
                'fast_summary': fast['summary'], 'hf_summary': hf['summary'],
                'fast_wall_seconds': fast['wall_seconds'], 'hf_wall_seconds': hf['wall_seconds'],
                'fast_answers_per_second_including_setup': 64/fast['wall_seconds'],
                'hf_answers_per_second_including_setup': 64/hf['wall_seconds']}
        status = 'PASS' if all(value['agreement']['passed'] for value in results.values()) else 'COARSE_ONLY'
    finally:
        stop.set()
        sampler.join(timeout=6)
        receipt = {'schema': 'wsp03-backend-probe/1', 'status': status, 'dev_only': True,
            'cases': results, 'image': cfg['image'], 'base_inventory_sha256': digest(cfg['model_files']),
            'ids_sha256': prepared['ids_sha256'], 'ended_at': time.time(),
            'wall_seconds': time.time()-started, 'new_track_clocks_started': False,
            'generation_settings_tuned': False, 'full_dev_backend': 'original_frozen_hf',
            'fast_predictions_substitute_full_hf': False, 'dev_accuracy_reported': False, 'cost_scope': 'pretrack engineering probe',
            'gpu_samples': str(output/'gpu-samples.jsonl'), 'gpu_sample_interval_seconds': 10}
        json_write(output/'backend-probe.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'export'))
    parser.add_argument('config')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    print(json.dumps({'prepare': prepare, 'run': run, 'export': export}[args.action](cfg), indent=2))
