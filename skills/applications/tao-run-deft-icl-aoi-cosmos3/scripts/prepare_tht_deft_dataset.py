#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the single-label THT DEFT package on Brev, preserving the source bundle.

Public queries and private grading/provenance are separate directory trees.
Future experiment containers must mount only their approved views; Unix modes
alone do not isolate another process running as the same user.
"""
import argparse
import collections
import hashlib
import json
import re
import shutil
import socket
import time
from pathlib import Path

CLASSES = ['Good', 'Bright', 'Interrupted', 'Trapped', 'Hidden', 'Bridge', 'Dry Contact', 'Less Solder']
EVALUATOR_SHA = '17047385eca86fd66d67967704cd457a713a370cf86e9ce85eafb67fe61fdde6'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')


def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows))


def safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f'Invalid source path: {relative}')
    return path


def verify_source(root):
    entries = []
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        if digest(safe_path(root, relative)) != expected:
            raise ValueError(f'Source checksum mismatch: {relative}')
        entries.append((relative, expected))
    return entries


def joint(row):
    return row['object_name'], row['scan_id'].removeprefix('SCN_')


def filename_metadata(row):
    prefix = 'SCN_' + row.get('scan_id', '').removeprefix('SCN_') + '_'
    stem = Path(row['filename']).stem
    if not stem.startswith(prefix):
        return None
    match = re.fullmatch(r'(MBO_.+)_A([12])(?:_[0-9a-f]+)?', stem[len(prefix):])
    return {'object_name': match.group(1), 'angle': 'A' + match.group(2)} if match else None


def source_rows(root, split):
    if split == 'train':
        records = load_lines(root / 'references/source-train/source.jsonl')
        index = {r['id']: r for r in load_lines(root / 'references/source-train/labels.jsonl')}
        paths = {k: root / 'references/source-train' / v['image'] for k, v in index.items()}
        for row in records:
            assert index[row['filename']]['labels'] == row['labels']
    else:
        records = load_lines(root / f'evaluation/{split}-source.jsonl')
        requests = load_lines(root / f'{split}/requests.jsonl')
        paths = {}
        for request in requests:
            images = [part['image'] for msg in request['messages'] for part in msg['content'] if part['type'] == 'image']
            assert len(images) == 1
            paths[request['id']] = root / split / images[0]
    assert len(records) == len(paths) == len({r['filename'] for r in records})
    assert set(paths) == {r['filename'] for r in records}
    return [dict(source_split=split, source=row, path=paths[row['filename']], image_sha256=digest(paths[row['filename']])) for row in records]


def checksum_tree(root):
    entries = [(p.relative_to(root).as_posix(), digest(p)) for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS']
    (root / 'SHA256SUMS').write_text(''.join(f'{sha}  {name}\n' for name, sha in entries))
    return len(entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--private-output', type=Path, required=True)
    parser.add_argument('--preserve-source-joint-overlaps', action='store_true', help='Explicitly preserve Sean-approved original train/test joint overlaps; never permits new overlaps')
    args = parser.parse_args()
    start = time.monotonic()
    source, output, private = args.source.resolve(), args.output.resolve(), args.private_output.resolve()
    assert not output.exists() and not private.exists(), 'Refuse to overwrite an existing frozen package'
    assert not any(a == b or a.is_relative_to(b) for a, b in [(output, source), (source, output), (private, source), (source, private), (output, private), (private, output)])
    verified_before = verify_source(source)
    original = {s: source_rows(source, s) for s in ('train', 'val', 'test')}
    assert {s: len(v) for s, v in original.items()} == {'train': 3952, 'val': 74, 'test': 271}
    assert digest(source/'evaluation/source-contract/evaluation.py') == EVALUATOR_SHA
    metadata_candidates, excluded, eligible = [], [], []
    audit = collections.Counter()
    # Validate filename decoding against existing metadata before recovering it.
    for record in original['train']:
        row = record['source']
        decoded = filename_metadata(row)
        if row.get('object_name') and decoded:
            assert decoded['object_name'] == row['object_name']
            assert not row.get('angle') or decoded['angle'] == row['angle']
            audit['filename_decoder_matches_existing_metadata'] += 1
    for split, records in original.items():
        for record in records:
            row = dict(record['source'])
            reasons = []
            if split == 'train':
                if len(row['labels']) != 1:
                    reasons.append('multi_label_out_of_scope')
                if not row.get('object_name'):
                    audit['original_missing_object_name'] += 1
                    decoded = filename_metadata(row)
                    if decoded:
                        metadata_candidates.append(dict(source_id=row['filename'], original_values={'object_name':row.get('object_name'), 'angle':row.get('angle')}, candidate_values=decoded, applied=False, reason='Sean requested exact Maxwell original eligibility; exclude even recoverable records', evidence='provenance filename grammar + exact known scan_id prefix + decoder validation against populated source metadata'))
                    reasons.append('missing_original_object_name')
            assert isinstance(row['labels'], list) and row['labels'] and all(label in CLASSES for label in row['labels'])
            if not row.get('scan_id') or (not row.get('object_name') and 'missing_original_object_name' not in reasons):
                if 'unresolved_joint_identity' not in reasons:
                    reasons.append('unresolved_joint_identity')
            record['effective'] = row
            if reasons:
                assert split == 'train', 'Never silently remove val/test cases'
                excluded.append(dict(source_split=split, source_id=row['filename'], source_row=record['source'], reasons=reasons, image_sha256=record['image_sha256']))
            else:
                assert len(row['labels']) == 1
                eligible.append(record)
    train = [r for r in eligible if r['source_split'] == 'train']
    # Compare source eligibility exactly, without inferring or modifying metadata.
    maxwell_ids = {r['source']['filename'] for r in original['train'] if len(r['source']['labels']) == 1 and r['source']['labels'][0] in CLASSES and r['source'].get('object_name')}
    assert {r['source']['filename'] for r in train} == maxwell_ids and len(maxwell_ids) == 3642
    source_joint_rows = {split: collections.defaultdict(list) for split in ('train','val','test')}
    for record in eligible:
        source_joint_rows[record['source_split']][joint(record['effective'])].append(record['source']['filename'])
    known_joint_overlap = set(source_joint_rows['train']) & set(source_joint_rows['test'])
    assert not set(source_joint_rows['train']) & set(source_joint_rows['val'])
    assert not set(source_joint_rows['val']) & set(source_joint_rows['test'])
    if known_joint_overlap:
        assert args.preserve_source_joint_overlaps, 'Source train/test joint overlap requires an explicit dataset policy'
        assert len(known_joint_overlap) == 2, 'Only the two reviewed original overlaps are authorized'
    selection = dict(algorithm='preserve_original_Maxwell_splits', source_split_seed=json.loads((source/'bundle.json').read_text())['split_seed'], original_train_to_proxy=0)
    by_role = {'mining': [], 'proxy': [], 'benchmark': []}
    for record in eligible:
        role = {'train':'mining', 'val':'proxy', 'test':'benchmark'}[record['source_split']]
        record['role'] = role
        by_role[role].append(record)
    for records in by_role.values():
        records.sort(key=lambda r: r['source']['filename'])
    overlap_counts = {}
    for key, fn in [('joint', lambda r: joint(r['effective'])), ('source_id', lambda r: r['source']['filename']), ('image_bytes', lambda r: r['image_sha256'])]:
        sets = {role: {fn(r) for r in records} for role, records in by_role.items()}
        for a, b in [('mining','proxy'), ('mining','benchmark'), ('proxy','benchmark')]:
            overlap = sets[a] & sets[b]
            overlap_counts[f'{key}:{a}:{b}'] = len(overlap)
            if key == 'joint' and (a,b) == ('mining','benchmark'):
                assert overlap == known_joint_overlap, 'Preserve only the reviewed original overlaps in Mining'
            else:
                assert not overlap, f'New cross-role overlap: {key} {a} {b}'
    output.mkdir(parents=True)
    private.mkdir(parents=True, mode=0o700)
    mapping, reference_rows = [], []
    for role, records in by_role.items():
        public_rows, gold_rows = [], []
        for record in records:
            row = record['effective']
            relative = f"{role}/images/{record['image_sha256']}{record['path'].suffix}"
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copyfile(record['path'], destination)
            assert digest(destination) == record['image_sha256']
            view = dict(id=row['filename'], image=relative, image_sha256=record['image_sha256'])
            public_rows.append(dict(view, labels=row['labels'], object_name=row['object_name'], scan_id=row['scan_id'].removeprefix('SCN_'), angle=row.get('angle','')) if role == 'mining' else view)
            gold_rows.append(dict(id=row['filename'], labels=row['labels']))
            mapping.append(dict(source_split=record['source_split'], role=role, source_id=row['filename'], source_row=record['source'], effective_row=row, image=relative, image_sha256=record['image_sha256']))
            if record['source_split'] == 'train':
                reference_rows.append(dict(view, labels=row['labels'], object_name=row['object_name'], scan_id=row['scan_id'].removeprefix('SCN_'), angle=row.get('angle',''), role=role))
        write_lines(output / role / ('samples.jsonl' if role == 'mining' else 'queries.jsonl'), public_rows)
        if role != 'mining':
            assert all(set(r) == {'id','image','image_sha256'} for r in public_rows)
            write_lines(private / 'evaluation' / f'{role}-gold.jsonl', gold_rows)
    assert len(by_role['benchmark']) == 271
    write_lines(private/'manifests/source_to_role.jsonl', mapping)
    write_lines(private/'manifests/excluded.jsonl', excluded)
    write_lines(private/'manifests/metadata_candidates_not_applied.jsonl', metadata_candidates)
    write_lines(private/'manifests/known_source_joint_overlaps.jsonl', [dict(object_name=k[0], normalized_scan_id=k[1], original_train_ids=source_joint_rows['train'][k], original_test_ids=source_joint_rows['test'][k], decision='preserve_original_Maxwell_population', authorized_by='Sean, 2026-11-05', risk='Different views of the same physical joint remain in Mining and Benchmark; training results are not a joint-disjoint holdout estimate') for k in sorted(known_joint_overlap)])
    write_lines(private/'reference_pools/proxy.jsonl', [r for r in reference_rows if r['role'] == 'mining'])
    write_lines(private/'reference_pools/benchmark.jsonl', reference_rows)
    write_json(private/'manifests/proxy_selection.json', selection)
    evaluator = output/'evaluator'
    evaluator.mkdir()
    for name in ('evaluation.py','definition.py'):
        shutil.copyfile(source/'evaluation/source-contract'/name, evaluator/name)
    (evaluator/'__init__.py').write_text('')
    script = output/'scripts/prepare_tht_deft_dataset.py'
    script.parent.mkdir()
    shutil.copyfile(Path(__file__), script)
    reasons = collections.Counter(reason for row in excluded for reason in row['reasons'])
    original_multilabel = sum(len(r['source']['labels']) != 1 for r in original['train'])
    summary = dict(name='tht-deft-v1', task='single_image_single_label_8_class', split_policy='preserve_original_Maxwell_splits', source_split_seed=selection['source_split_seed'],
        source_revision=json.loads((source/'bundle.json').read_text())['source_revision'],
        source_sha256sums_sha256=digest(source/'SHA256SUMS'), preparation_script_sha256=digest(script),
        source_counts={k:len(v) for k,v in original.items()}, role_counts={k:len(v) for k,v in by_role.items()},
        eligible_original_train=len(train), original_multilabel=original_multilabel,
        original_missing_object_name=audit['original_missing_object_name'], metadata_repaired=0,
        metadata_recovery_candidates_not_applied=len(metadata_candidates),
        eligibility='Maxwell original single-label and populated object_name; no metadata repair',
        excluded_unique=len(excluded), exclusion_reason_counts=dict(reasons),
        proxy_original_val=74, proxy_from_original_train=0,
        reference_pool_counts={'proxy':len(by_role['mining']), 'benchmark':len(train)},
        evaluator_sha256=digest(evaluator/'evaluation.py'), definition_sha256=digest(evaluator/'definition.py'),
        known_source_joint_overlaps=len(known_joint_overlap),
        overlap_policy='preserve_original_Maxwell_population; no_new_cross_role_overlap',
        checks={'all_active_rows_single_label': True, 'cross_role_overlap_counts':overlap_counts,
                'exact_original_Maxwell_eligible_ids':True,
                'benchmark_membership_unchanged': True, 'copied_image_hashes_verified':True,
                'query_manifests_have_no_gold':True, 'private_artifacts_outside_public_tree':True},
        status='prepared_and_verified; runtime_gold_access_isolation_requires_M3')
    summary['checks']['source_unchanged_after_preparation'] = verify_source(source) == verified_before
    assert summary['checks']['source_unchanged_after_preparation']
    write_json(output/'bundle.json', summary)
    write_json(private/'verification.json', dict(summary, filename_audit=dict(audit), source_files_verified=len(verified_before), host=socket.gethostname(), python_runtime=__import__('sys').version, preparation_seconds=time.monotonic()-start))
    (output/'README.md').write_text('''# tht-deft-v1 — single-image, eight-choice, one-answer dataset

Use this package for the A–F DEFT/ICL study. The original bundle remains intact.
All active labels have exactly one original class; no multi-label target was
truncated or relabeled. Multi-label records are excluded from training AND ICL.
Missing ORIGINAL object_name excludes a record even when the filename could
restore it, matching Maxwell's original eligibility exactly. No metadata repair
is applied. See bundle.json for verified counts.

Paths in all manifests are relative to THIS package root.
- mining/samples.jsonl: training labels and images.
- proxy/queries.jsonl: exactly the 74 original val queries, no gold.
- benchmark/queries.jsonl: all 271 original test queries, no gold.
- evaluator/: unchanged Maxwell parser/scorer and class definition.
- scripts/: deterministic preparation script; Python standard library only.
- SHA256SUMS: checksums of every delivered file except this checksum list.

Gold, source mapping, exclusions, unused metadata candidates and references live
in the SEPARATE tht-deft-v1-private package. Never mount that whole package, the
original source, or this entire public package into classification sessions.
A gets query-only access. C/E get Mining references during Proxy; the private
full eligible original-train reference view is released only after freeze for
Benchmark. D/F train only on Mining. Runtime access isolation is an M3 gate;
directory names and file modes do not isolate processes of the same user.

No source rows are moved across splits. TWO ORIGINAL
train/test joints remain as different A1/A2 views in Mining and Benchmark,
explicitly preserved by Sean to match Maxwell's source population. Therefore
the package is NOT a joint-disjoint training/test split. No source IDs or exact
image bytes overlap across the three roles. Report the two-joint limitation in
all experiment results; ICL must still exclude same-joint references per query.

Source membership is unchanged for original val/test: train -> Mining (3,642
eligible), val -> Proxy (74), test -> Benchmark (271). The original split seed
is 1; this preparation performs NO random split and draws NO extra Proxy rows.
No image transformation, label adjudication, model training or inference ran.

Rebuild on Brev into fresh paths (never overwrite a frozen package):
```sh
python3 scripts/prepare_tht_deft_dataset.py --source SOURCE_BUNDLE \\
  --output NEW_PUBLIC_PACKAGE --private-output NEW_PRIVATE_PACKAGE \\
  --preserve-source-joint-overlaps
```
''')
    (private/'README.md').write_text('''# Private THT DEFT grading and provenance

Owner-only preparation archive, kept separately from the public dataset.
evaluation/: Proxy/Benchmark gold for an isolated grader only.
manifests/: source mapping, exclusions (original labels preserved), unused
metadata recovery candidates, split evidence. Excluded images stay in the source;
they are NOT copied into the public dataset.
reference_pools/proxy.jsonl: Mining only.
reference_pools/benchmark.jsonl: the SAME 3,642 eligible original-train rows as
the Proxy reference pool. No Proxy/Benchmark queries enter either pool.
All image paths resolve against the paired PUBLIC package root.
Do not push this package, gold, images or manifests to the code branch.
File permissions alone are not sufficient for a same-user classification agent;
mount separate approved views in isolated runtime environments at M3.
''')
    checksum_tree(output)
    checksum_tree(private)
    for path in private.rglob('*'):
        path.chmod(0o700 if path.is_dir() else 0o600)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
