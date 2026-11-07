# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic WSP views. Reads Mining and identity-only exclusions, never gold outside Mining."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

from tht_contract import CLASSES, json_hash, sha256, preprocess, within, load_jsonl
from wsp03_io import json_write


def joint(row):
    return row['object_name'], row['scan_id'].removeprefix('SCN_')


def components(rows):
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('nonempty unique IDs required')
    parents = {joint(r): joint(r) for r in rows}
    def find(k):
        while parents[k] != k:
            parents[k] = parents[parents[k]]
            k = parents[k]
        return k
    seen = {}
    for r in rows:
        if r['labels'] not in [[c] for c in CLASSES] or r.get('role', 'mining') != 'mining':
            raise ValueError('single-label real Mining only')
        if not r['object_name'] or not r['scan_id'] or not r['image_sha256']:
            raise ValueError('identity/hash required')
        a, b = find(joint(r)), find(seen.setdefault(r['image_sha256'], joint(r)))
        if a != b:
            parents[max(a, b)] = min(a, b)
    groups = defaultdict(list)
    for r in rows:
        groups[find(joint(r))].append(r)
    return groups, {k: find(k) for k in parents}


def partition(rows, exclusions, *, seed=42, explore_size=512):
    groups, component_of = components(rows)
    forbidden = {component_of[(r['object_name'], r['normalized_scan_id'].removeprefix('SCN_'))] for r in exclusions}
    rank = sorted((k for k in groups if k not in forbidden), key=lambda k: (json_hash({'seed': seed, 'joint_component': k}), k))
    cut = len(rank) * 4 // 5
    dev_keys = rank[:cut]
    erank = sorted(dev_keys, key=lambda k: (json_hash({'seed': seed, 'purpose': 'wsp03-dev-explore', 'joint_component': k}), k))
    selected = {k for k in dev_keys if any(r['labels'] == ['Less Solder'] for r in groups[k])}
    for label in CLASSES:
        if not any(r['labels'] == [label] for k in selected for r in groups[k]):
            available = [k for k in erank if any(r['labels'] == [label] for r in groups[k])]
            if not available:
                raise ValueError('development has no representation for ' + label)
            selected.add(available[0])
    n = sum(len(groups[k]) for k in selected)
    if n > explore_size:
        raise ValueError('mandatory components exceed explore size')
    for k in erank:
        if k not in selected and n + len(groups[k]) <= explore_size:
            selected.add(k)
            n += len(groups[k])
    if n != explore_size:
        raise ValueError('whole-component selection cannot fill exact size; no redraw')
    roles = {name: sorted([r for k in keys for r in groups[k]], key=lambda r: r['id']) for name, keys in [('development', dev_keys), ('holdout', rank[cut:]), ('explore', selected)]}
    roles['probe'] = roles['development'][:64]
    def meta(rs):
        return {'images': len(rs), 'components': len({component_of[joint(r)] for r in rs}),
                'class_counts': dict(Counter(r['labels'][0] for r in rs)),
                'ids_sha256': json_hash([r['id'] for r in rs]), 'source_rows_sha256': json_hash(rs)}
    summary = {'schema': 'wsp03-split/1', 'seed': seed, 'source_count': len(rows),
               'excluded_images': sum(len(groups[k]) for k in forbidden), 'excluded_components': len(forbidden),
               'eligible_components': len(rank), 'roles': {name: meta(rs) for name, rs in roles.items()},
               'excluded_ids': sorted(r['id'] for k in forbidden for r in groups[k])}
    return roles, summary


def write_jsonl(path, rows):
    with Path(path).open('x') as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def prepare(public, exclusions, output, expected=None, *, expected_source_sha256=None, expected_exclusions_sha256=None):
    public, output = Path(public).resolve(strict=True), Path(output).resolve(strict=False)
    if output.exists() or output.is_relative_to(public):
        raise ValueError('new output outside source required')
    source = public / 'mining/samples.jsonl'
    if expected_source_sha256 and sha256(source) != expected_source_sha256:
        raise ValueError('source digest mismatch')
    if expected_exclusions_sha256 and sha256(exclusions) != expected_exclusions_sha256:
        raise ValueError('exclusions digest mismatch')
    rows = load_jsonl(source)
    raw = Path(exclusions).read_text()
    try:
        excluded = json.loads(raw)
    except json.JSONDecodeError:
        excluded = [json.loads(line) for line in raw.splitlines() if line.strip()]
    excluded = [{'object_name': r['object_name'], 'normalized_scan_id': r['normalized_scan_id']} for r in excluded]
    roles, summary = partition(rows, excluded)
    if expected:
        for role, values in expected.items():
            for key, value in values.items():
                if summary['roles'][role].get(key) != value:
                    raise ValueError(f'frozen {role}/{key} mismatch')
    if output.exists() or output.is_relative_to(public):
        raise ValueError('new output outside source required')
    output.mkdir(parents=True, mode=0o700)
    summary.update(source_sha256=sha256(source), exclusions_sha256=sha256(exclusions))
    for role, records in roles.items():
        dst = output / role
        dst.mkdir(mode=0o700)
        queries, gold, provenance = [], [], []
        for i, row in enumerate(records):
            if not row['image'].startswith('mining/images/'):
                raise ValueError('only Mining image paths accepted')
            image = within(public / 'mining', str(Path(row['image']).relative_to('mining')))
            if sha256(image) != row['image_sha256']:
                raise ValueError('Mining bytes changed')
            name = f'image-{i:04d}.png'
            info = preprocess(image, dst / name)
            queries.append({'id': row['id'], 'image': name, 'image_sha256': info['view_sha256']})
            gold.append({'id': row['id'], 'labels': row['labels']})
            provenance.append({'id': row['id'], 'joint': joint(row), **info})
        for name, data in [('queries.jsonl', queries), ('gold.jsonl', gold), ('provenance.jsonl', provenance), ('source_rows.jsonl', records)]:
            write_jsonl(dst / name, data)
        summary['roles'][role]['files'] = {f.name: sha256(f) for f in sorted(dst.iterdir())}
        summary['roles'][role]['view_sha256'] = json_hash(summary['roles'][role]['files'])
        for f in dst.iterdir():
            f.chmod(0o400)
    summary['split_sha256'] = json_hash(summary)
    json_write(output / 'split.json', summary)
    return summary


def validate(output):
    output = Path(output).resolve(strict=True)
    summary = json.loads((output / 'split.json').read_text())
    copy = dict(summary)
    expected = copy.pop('split_sha256')
    if json_hash(copy) != expected:
        raise ValueError('split manifest changed')
    for role, record in summary['roles'].items():
        actual = {p.name: sha256(p) for p in sorted((output / role).iterdir())}
        if actual != record['files']:
            raise ValueError('view membership/bytes changed: ' + role)
        rows = load_jsonl(output / role / 'source_rows.jsonl')
        if json_hash(rows) != record['source_rows_sha256'] or json_hash([r['id'] for r in rows]) != record['ids_sha256']:
            raise ValueError('source membership changed: ' + role)
    return {'passed': True, 'split_sha256': expected}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'validate'])
    p.add_argument('--public')
    p.add_argument('--exclusions')
    p.add_argument('--expected', help='JSON role→frozen fields mapping')
    p.add_argument('--output', required=True)
    p.add_argument('--source-sha256')
    p.add_argument('--exclusions-sha256')
    a = p.parse_args()
    result = validate(a.output) if a.command == 'validate' else prepare(a.public, a.exclusions, a.output, json.loads(Path(a.expected).read_text()) if a.expected else None, expected_source_sha256=a.source_sha256, expected_exclusions_sha256=a.exclusions_sha256)
    print(json.dumps({'passed': True, 'split_sha256': result['split_sha256']}))
