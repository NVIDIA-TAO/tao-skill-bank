#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, streaming safetensors inspection and declarative sparse weight patches.

No model import, tensor library, pickle, executable patch, training or GPU API.
Header inspection is safe during preparation; all other operations belong to the
later approved experiment. BF16 writes use round-to-nearest, ties-to-even.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import struct
from pathlib import Path

from tht_contract import json_hash, sha256

DTYPES = {'F64': (8, 'd'), 'F32': (4, 'f'), 'F16': (2, 'e'), 'BF16': (2, None),
          'I64': (8, 'q'), 'I32': (4, 'i'), 'I16': (2, 'h'), 'I8': (1, 'b'),
          'U64': (8, 'Q'), 'U32': (4, 'I'), 'U16': (2, 'H'), 'U8': (1, 'B'),
          'BOOL': (1, '?')}
FLOATS = {'F64', 'F32', 'F16', 'BF16'}
MAX_HEADER = 16 * 1024 * 1024


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('duplicate JSON key')
        value[key] = item
    return value


def strict_json(data):
    def invalid(value):
        raise ValueError('nonfinite JSON number: ' + value)
    return json.loads(data, object_pairs_hook=unique_object, parse_constant=invalid)


def regular_file(root, name):
    """One flat checkpoint file, no symlinks, devices or path expressions."""
    if not isinstance(name, str) or Path(name).name != name or name in ('', '.', '..'):
        raise ValueError('flat file name required')
    path = Path(root) / name
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError('regular non-symlink file required')
    return path


def header(path):
    path = Path(path)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError('regular safetensors file required')
    size = path.stat().st_size
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('truncated safetensors prefix')
        length = struct.unpack('<Q', prefix)[0]
        if not 2 <= length <= min(MAX_HEADER, size - 8):
            raise ValueError('invalid safetensors header size')
        raw = stream.read(length)
    obj = strict_json(raw)
    if not isinstance(obj, dict):
        raise ValueError('safetensors header must be an object')
    tensors = {}
    intervals = []
    for name, item in obj.items():
        if name == '__metadata__':
            if not isinstance(item, dict) or any(not isinstance(v, str) for v in item.values()):
                raise ValueError('invalid safetensors metadata')
            continue
        if not isinstance(item, dict) or set(item) != {'dtype', 'shape', 'data_offsets'}:
            raise ValueError('invalid tensor header')
        dtype, shape, offsets = item['dtype'], item['shape'], item['data_offsets']
        if dtype not in DTYPES or not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape):
            raise ValueError('unsupported dtype or shape')
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int for x in offsets):
            raise ValueError('invalid tensor offsets')
        start, end = offsets
        elements = math.prod(shape)
        if start < 0 or end < start or end - start != elements * DTYPES[dtype][0] or end > size - 8 - length:
            raise ValueError('tensor bounds disagree with shape/file')
        if end > start:
            intervals.append((start, end))
        tensors[name] = {**item, 'elements': elements, 'offset': 8 + length + start}
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError('overlapping tensors or holes in data')
        cursor = end
    if cursor != size - 8 - length:
        raise ValueError('unclaimed bytes in safetensors file')
    return {'header_sha256': hashlib.sha256(prefix + raw).hexdigest(),
            'header_bytes': 8 + length, 'size': size, 'tensors': tensors}


def catalog(root):
    """Read headers/index only; never read real tensor values here."""
    root = Path(root).resolve(strict=True)
    index = root / 'model.safetensors.index.json'
    expected = None
    if index.exists():
        expected = strict_json(regular_file(root, index.name).read_bytes())['weight_map']
        if not isinstance(expected, dict) or not expected:
            raise ValueError('empty weight map')
        names = sorted(set(expected.values()))
    else:
        names = sorted(p.name for p in root.glob('*.safetensors'))
    if not names or set(names) != {p.name for p in root.glob('*.safetensors')}:
        raise ValueError('shard inventory mismatch')
    tensors, files = {}, {}
    for name in names:
        info = header(regular_file(root, name))
        files[name] = {k: v for k, v in info.items() if k != 'tensors'}
        for tensor, meta in info['tensors'].items():
            if tensor in tensors or (expected is not None and expected.get(tensor) != name):
                raise ValueError('tensor/index mismatch')
            tensors[tensor] = {'file': name, **meta}
    if expected is not None and set(expected) != set(tensors):
        raise ValueError('missing indexed tensor')
    return {'schema': 'wsp-tensors/1', 'files': files, 'tensors': tensors}


def inventory(root):
    """Full content inventory. ONLY synthetic tests or later approved runtime."""
    root = Path(root)
    files = []
    for path in sorted(root.iterdir()):
        regular_file(root, path.name)
        files.append({'name': path.name, 'size': path.stat().st_size, 'sha256': sha256(path)})
    return {'files': files, 'fingerprint': json_hash(files)}


def verify_inventory(root, manifest):
    if inventory(root) != manifest:
        raise ValueError('base/candidate content lineage mismatch')


def clone(root, destination, manifest):
    """An independent byte copy. Never hardlink or write shared checkpoint files."""
    root = Path(root).resolve(strict=True)
    destination = Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ValueError('copy destination must be new and separate')
    verify_inventory(root, manifest)
    destination.mkdir(parents=True, mode=0o700)
    for row in manifest['files']:
        source = regular_file(root, row['name'])
        target = destination / row['name']
        shutil.copyfile(source, target)
        target.chmod(0o600)
        if target.stat().st_ino == source.stat().st_ino and target.stat().st_dev == source.stat().st_dev:
            raise ValueError('copy must not share an inode')
    verify_inventory(destination, manifest)


def unpack(dtype, raw):
    if dtype == 'BF16':
        return struct.unpack('<f', b'\0\0' + raw)[0]
    return struct.unpack('<' + DTYPES[dtype][1], raw)[0]


def pack(dtype, value):
    if dtype not in FLOATS or type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('finite floating point replacement required')
    try:
        if dtype == 'BF16':
            bits = struct.unpack('<I', struct.pack('<f', value))[0]
            rounded = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) & 0xffff
            raw = struct.pack('<H', rounded)
        else:
            raw = struct.pack('<' + DTYPES[dtype][1], value)
    except (OverflowError, struct.error) as error:
        raise ValueError('replacement overflows dtype') from error
    if not math.isfinite(unpack(dtype, raw)):
        raise ValueError('replacement becomes nonfinite after dtype conversion')
    return raw


def read_values(root, tensors, name, indices, limit=1024):
    if name not in tensors or not isinstance(indices, list) or not 1 <= len(indices) <= limit:
        raise ValueError('unknown tensor or inspection count exceeded')
    meta = tensors[name]
    if any(type(i) is not int or not 0 <= i < meta['elements'] for i in indices) or len(set(indices)) != len(indices):
        raise ValueError('invalid/duplicate flat indices')
    width = DTYPES[meta['dtype']][0]
    values = []
    with regular_file(root, meta['file']).open('rb') as stream:
        for index in indices:
            stream.seek(meta['offset'] + index * width)
            raw = stream.read(width)
            value = unpack(meta['dtype'], raw)
            if not math.isfinite(value):
                raise ValueError('nonfinite inspected base value')
            values.append(value)
    return {'tensor': name, 'dtype': meta['dtype'], 'shape': meta['shape'],
            'indices': indices, 'values': values}


def finite_tensor(root, meta):
    """Scan an edited tensor in bounded chunks, including its untouched values."""
    width, code = DTYPES[meta['dtype']]
    remaining = meta['elements'] * width
    with regular_file(root, meta['file']).open('rb') as stream:
        stream.seek(meta['offset'])
        while remaining:
            raw = stream.read(min(1024 * 1024, remaining))
            if not raw:
                raise ValueError('truncated tensor')
            if meta['dtype'] == 'BF16':
                # Exponent all ones is NaN/inf, independent of sign/mantissa.
                valid = all((value & 0x7f80) != 0x7f80 for (value,) in struct.iter_unpack('<H', raw))
            else:
                valid = all(math.isfinite(value) for (value,) in struct.iter_unpack('<' + code, raw))
            if not valid:
                raise ValueError('nonfinite tensor')
            remaining -= len(raw)


def apply_patch(base, destination, manifest, patch, *, max_tensors, max_elements):
    """Validate every cast, copy once, apply deterministic set operations, audit.

    Patch JSON is the replayable program; arbitrary Python is never executed by
    the trusted service. Failed work is retained by its caller, never overwritten.
    """
    if set(patch) != {'schema', 'base_fingerprint', 'hypothesis', 'edits'} or patch['schema'] != 'wsp-patch/1':
        raise ValueError('invalid patch schema')
    if patch['base_fingerprint'] != manifest['fingerprint']:
        raise ValueError('wrong patch base lineage')
    if not isinstance(patch['hypothesis'], str) or not 1 <= len(patch['hypothesis'].strip()) <= 8192:
        raise ValueError('bounded testable hypothesis required')
    edits = patch['edits']
    if not isinstance(edits, list) or not 1 <= len(edits) <= max_tensors:
        raise ValueError('tensor budget exceeded')
    names = [e.get('tensor') for e in edits if isinstance(e, dict)]
    if len(names) != len(edits) or any(not isinstance(n, str) for n in names) or len(set(names)) != len(names):
        raise ValueError('duplicate/invalid tensor edit')
    tensors = catalog(base)['tensors']
    replacements, diffs = {}, []
    total = 0
    for edit in edits:
        if set(edit) != {'tensor', 'indices', 'values'} or not isinstance(edit['indices'], list) or not isinstance(edit['values'], list):
            raise ValueError('explicit sparse assignments required')
        name, indices, values = edit['tensor'], edit['indices'], edit['values']
        if name not in tensors or tensors[name]['dtype'] not in FLOATS or len(values) != len(indices) or not indices:
            raise ValueError('invalid floating tensor assignments')
        total += len(indices)
        if total > max_elements:
            raise ValueError('element budget exceeded')
        old = read_values(base, tensors, name, indices, max_elements)['values']
        meta = tensors[name]
        finite_tensor(base, meta)
        changes = []
        for index, before, requested in zip(indices, old, values):
            encoded = pack(meta['dtype'], requested)
            after = unpack(meta['dtype'], encoded)
            # Count numerical changes after quantization, not requested edits.
            if after == before:
                raise ValueError('zero/rounded-away element change')
            delta = after - before
            if not math.isfinite(delta):
                raise ValueError('nonfinite delta')
            changes.append((index, encoded, before, after, delta))
        replacements[name] = changes
        deltas = [x[4] for x in changes]
        norms = {'l1': math.fsum(abs(v) for v in deltas), 'l2': math.hypot(*deltas),
                 'linf': max(abs(v) for v in deltas)}
        if not all(math.isfinite(v) for v in norms.values()):
            raise ValueError('nonfinite delta norm')
        diffs.append({'tensor': name, 'dtype': meta['dtype'], 'shape': meta['shape'],
                      'changed_elements': len(changes), 'delta_norms': norms,
                      'actual_assignments': [{'index': i, 'before': b, 'after': a} for i, _, b, a, _ in changes]})
    clone(base, destination, manifest)
    for name, changes in replacements.items():
        meta = tensors[name]
        with regular_file(destination, meta['file']).open('r+b') as stream:
            for index, encoded, *_ in changes:
                stream.seek(meta['offset'] + index * DTYPES[meta['dtype']][0])
                stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    audit_sparse_diff(base, destination, tensors, replacements)
    for name in replacements:
        finite_tensor(destination, tensors[name])
    result_manifest = inventory(destination)
    for row in result_manifest['files']:
        regular_file(destination, row['name']).chmod(0o400)
    return {'schema': 'wsp-candidate/1', 'base_fingerprint': manifest['fingerprint'],
            'patch_sha256': json_hash(patch), 'changed_tensors': len(diffs),
            'changed_elements': total, 'tensor_diffs': diffs,
            'delta_norms': {'l1': math.fsum(d['delta_norms']['l1'] for d in diffs),
                            'l2': math.hypot(*(d['delta_norms']['l2'] for d in diffs)),
                            'linf': max(d['delta_norms']['linf'] for d in diffs)},
            'finite': True, 'finite_scope': 'all values of every edited tensor',
            'manifest': result_manifest, 'loadability': 'pending_native_development_forward'}


def audit_sparse_diff(base, edited, tensors, replacements):
    """Byte-for-byte proof that only the declared, cast assignments changed."""
    allowed = {}
    for name, changes in replacements.items():
        meta = tensors[name]
        for index, raw, *_ in changes:
            offset = meta['offset'] + index * DTYPES[meta['dtype']][0]
            for k, value in enumerate(raw):
                allowed.setdefault(meta['file'], {})[offset + k] = value
    if {p.name for p in Path(base).iterdir()} != {p.name for p in Path(edited).iterdir()}:
        raise ValueError('candidate file set changed')
    for source in sorted(Path(base).iterdir()):
        target = regular_file(edited, source.name)
        if source.stat().st_size != target.stat().st_size:
            raise ValueError('candidate file size changed')
        if source.name not in allowed:
            if sha256(source) != sha256(target):
                raise ValueError('undeclared candidate change')
            continue
        positions = sorted(allowed[source.name].items())
        cursor = 0
        with source.open('rb') as a, target.open('rb') as b:
            while True:
                before, after = a.read(1024 * 1024), b.read(1024 * 1024)
                if not before:
                    break
                expected = bytearray(before)
                lo = cursor
                # Process each patch byte once, without a full tensor allocation.
                if cursor == 0:
                    iterator = iter(positions)
                    point = next(iterator, None)
                while point is not None and point[0] < lo + len(before):
                    expected[point[0] - lo] = point[1]
                    point = next(iterator, None)
                if bytes(expected) != after:
                    raise ValueError('candidate bytes differ from deterministic patch')
                cursor += len(before)
