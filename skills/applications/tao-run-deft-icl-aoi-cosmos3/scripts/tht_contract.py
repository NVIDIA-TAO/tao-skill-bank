#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""THT task adapter. Parsing and scoring belong to the frozen external evaluator."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

CLASSES = ('Good', 'Bright', 'Interrupted', 'Trapped', 'Hidden', 'Bridge', 'Dry Contact', 'Less Solder')
EXTERNAL_HASHES = {
    'evaluation.py': '17047385eca86fd66d67967704cd457a713a370cf86e9ce85eafb67fe61fdde6',
    'definition.py': 'e0bb0405793353a87cd3a79a34e3a9d9fcb79c2d78a2a6afc675d85a8624d19c',
}
SOURCE_REVISION = '1711959a40f89cbf4072ea343de318a204903dfd'
# The exact source string is an external, hash-pinned task input. The readable
# plan transcription is deliberately not silently promoted to source parity.

def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()

def external_evaluator(root: str | Path):
    """Load only declared files after verifying bytes, without writing pycache."""
    root = Path(root).resolve(strict=True)
    for name, expected in EXTERNAL_HASHES.items():
        if sha256(root / name) != expected:
            raise ValueError(f'frozen evaluator hash mismatch: {name}')
    namespace = '_tht_external_' + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    if namespace + '.evaluation' in sys.modules:
        return sys.modules[namespace + '.evaluation']
    package = types.ModuleType(namespace)
    package.__path__ = [str(root)]
    sys.modules[namespace] = package
    for name in ('definition', 'evaluation'):
        module = types.ModuleType(namespace + '.' + name)
        module.__file__ = str(root / (name + '.py'))
        module.__package__ = namespace
        sys.modules[module.__name__] = module
        exec(compile((root / (name + '.py')).read_bytes(), module.__file__, 'exec'), module.__dict__)
    evaluator = sys.modules[namespace + '.evaluation']
    if tuple(evaluator.CLASSES) != CLASSES:
        raise ValueError('external class order changed')
    return evaluator

def grade(raw_rows: list[dict], gold_rows: list[dict], evaluator) -> dict:
    """Trusted grading boundary; require exactly the declared query set."""
    raw = {r['id']: r for r in raw_rows}
    gold = {r['id']: r for r in gold_rows}
    if len(raw) != len(raw_rows) or len(gold) != len(gold_rows) or raw.keys() != gold.keys():
        raise ValueError('duplicate or incomplete prediction/gold identities')
    scored = []
    for identity, row in raw.items():
        labels = gold[identity]['labels']
        if not isinstance(labels, list) or len(labels) != 1 or labels[0] not in CLASSES:
            raise ValueError('gold must have exactly one original class')
        scored.append({'id': identity, 'truth': labels[0], 'prediction': evaluator.parse(row['raw_answer'])})
    return {'predictions': scored, 'metrics': evaluator.score(scored)}

def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def within(root: str | Path, relative: str) -> Path:
    root = Path(root).resolve(strict=True)
    result = (root / relative).resolve(strict=True)
    if not result.is_relative_to(root) or not result.is_file():
        raise ValueError(f'path escapes approved role view: {relative}')
    return result

def validate_queries(rows: list[dict], *, count: int | None = None):
    if not rows or (count is not None and len(rows) != count):
        raise ValueError('query count mismatch')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate query identities')
    if any(set(r) != {'id', 'image', 'image_sha256'} for r in rows):
        raise ValueError('query manifest must contain only id/image/image_sha256')

def validate_sharegpt(rows: list[dict], mode: str):
    if mode not in ('tht_single', 'bare_okng'):
        raise ValueError('unsupported annotation mode; no multi-label mode')
    if not isinstance(rows, list) or not rows:
        raise ValueError('expected nonempty ShareGPT array')
    count = 1 if mode == 'tht_single' else 2
    for row in rows:
        images, turns = row.get('images'), row.get('conversations')
        if not isinstance(images, list) or len(images) != count or not all(isinstance(x, str) and x for x in images):
            raise ValueError(f'{mode} requires {count} images in declared order')
        if not isinstance(turns, list) or len(turns)<2 or turns[0].get('from')!='human' or turns[-1].get('from')!='gpt':
            raise ValueError('training requires a human prompt and final assistant target')
        if mode=='tht_single' and (len(turns)!=2 or [t.get('from') for t in turns]!=['human','gpt']):
            raise ValueError('THT requires exactly one human turn and one target')
        if any(t.get('from') not in ('human','gpt') for t in turns):
            raise ValueError('unsupported ShareGPT role')
        if not all(isinstance(t.get('value'), str) for t in turns):
            raise ValueError('training text must be strings')
        answer = turns[-1]['value']
        if mode == 'bare_okng':
            if answer not in ('OK', 'NG'):
                raise ValueError('paired target must be exactly OK or NG')
        else:
            try:
                obj = json.loads(answer)
            except (TypeError, ValueError) as e:
                raise ValueError('training target must be exact one-label JSON') from e
            if not isinstance(obj, dict) or set(obj) != {'labels'} or not isinstance(obj['labels'], list) or len(obj['labels']) != 1 or obj['labels'][0] not in CLASSES:
                raise ValueError('training target must be exact one-label JSON')

def preprocess(source: str | Path, destination: str | Path):
    from PIL import Image, ImageOps
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        ImageOps.pad(image.convert('RGB'), (256, 256), method=Image.Resampling.NEAREST).save(destination, format='PNG')
    return {'source_sha256': sha256(source), 'view_sha256': sha256(destination),
            'profile': 'RGB/ImageOps.pad/256x256/NEAREST/PNG'}

def load_prompt(path: str | Path, expected_sha256: str) -> str:
    if sha256(path) != expected_sha256:
        raise ValueError('exact baseline prompt hash mismatch')
    text = Path(path).read_text()
    if not text or any(label not in text for label in CLASSES):
        raise ValueError('baseline prompt lacks the complete class names')
    return text
