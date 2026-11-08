# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare frozen static-reference selections from hash-bound label-free inputs.

Output contains private reference identities, so mount only rendered messages in
the classifier. This command performs no VLM calls or scoring.
"""
import argparse
import hashlib
import json
from pathlib import Path
from reference_strategies import select, fixed_atlas, check_fixed_atlas


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def asset(item):
    if sha(item['path']) != item['sha256']: raise ValueError('input hash mismatch')
    return json.loads(Path(item['path']).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', required=True)
    a = p.parse_args(); config = json.loads(Path(a.config).read_text())
    if config['schema'] != 'tao-reference-preparation/1': raise ValueError('unknown configuration')
    queries = asset(config['queries']); references = asset(config['references'])
    if not queries or len({q['id'] for q in queries}) != len(queries): raise ValueError('nonempty unique queries required')
    for q in queries:
        if set(q) - {'id', 'source_id', 'image_sha256', 'object_name', 'scan_id', 'embedding', 'image', 'view_path'}:
            raise ValueError('query includes forbidden fields')
    if config['mode'] == 'fixed_atlas':
        selected, manifest = fixed_atlas(references, config['classes'], config['per_class'], config.get('salt', 'atlas-v1:'))
        overlap = check_fixed_atlas(queries, selected)
        if overlap['status'] != 'PASS': raise ValueError('fixed atlas overlap; do not silently replace tiles')
        result = {'manifest': manifest, 'overlap': overlap, 'references': selected}
    elif config['mode'] == 'similarity':
        result = {}
        for q in queries:
            selected, receipt = select(config['policy'], q, q['embedding'], references, config['classes'], config['reference_capacity'])
            result[q['id']] = {'references': [{k: v for k, v in r.items() if k != 'embedding'} for r in selected], 'receipt': receipt}
    else: raise ValueError('unknown mode')
    output = Path(config['output']); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f: json.dump(result, f, indent=2)
    print(json.dumps({'operation': 'prepare-references', 'status': 'COMPLETE', 'queries': len(queries),
                      'mode': config['mode'], 'config_sha256': sha(a.config), 'output_sha256': sha(output), 'model_calls': 0}))


if __name__ == '__main__':
    try: main()
    except Exception as e:
        print(json.dumps({'operation': 'prepare-references', 'status': 'BLOCKED', 'error_type': type(e).__name__}))
        raise SystemExit(1)
