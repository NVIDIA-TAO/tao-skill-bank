#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packaged Cosmos tool loop over user-owned references and TAO SigLIP embeddings.

This is a separate protocol from the externally pinned historical THT adapter.
No evaluator, target labels, external source checkout, or encoder lives here.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import time

PROFILE = 'simple-harness-siglip-v1'
SCRIPTS = Path(__file__).resolve().parent
DEFAULTS = dict(examples_per_class=4, tool_examples_per_class=4,
                max_tool_rounds=10, max_new_tokens=4096, seed=0)
TOOLS = [
    {'type': 'function', 'function': {
        'name': 'search_similar_examples',
        'description': 'Find references similar to the query. Page 0 returns the two '
                       'nearest classes; page 1 the next two classes, and so on.',
        'parameters': {'type': 'object', 'properties': {
            'page': {'type': 'integer', 'minimum': 0}}, 'required': ['page'],
            'additionalProperties': False}}},
    {'type': 'function', 'function': {
        'name': 'get_class_examples',
        'description': 'Inspect more examples from one or two named classes, ranked '
                       'by similarity to the query. Increase page for new examples.',
        'parameters': {'type': 'object', 'properties': {
            'classes': {'type': 'array', 'items': {'type': 'string'},
                        'minItems': 1, 'maxItems': 2},
            'page': {'type': 'integer', 'minimum': 0}},
            'required': ['classes', 'page'], 'additionalProperties': False}}},
]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    """Atomic progress write. Immutable artifacts use seal instead."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def seal(path, value):
    path = Path(path)
    if path.exists():
        verify_seal(path)
        if read(path) != value:
            raise ValueError(f'Existing frozen receipt differs: {path}; use a new directory')
        return
    write(path, value)
    path.with_suffix(path.suffix + '.sha256').write_text(sha(path) + '\n')


def verify_seal(path):
    path = Path(path)
    witness = path.with_suffix(path.suffix + '.sha256')
    if not witness.is_file() or witness.read_text().strip() != sha(path):
        raise ValueError(f'Artifact changed or missing seal: {path}')


def hashes(paths):
    return {str(Path(p)): sha(p) for p in sorted(set(map(str, paths)))}


def verify_files(files):
    for path, expected in files.items():
        if not Path(path).is_file() or sha(path) != expected:
            raise ValueError(f'Frozen input changed or missing: {path}')


def implementation_files():
    return [Path(__file__).resolve(), SCRIPTS / 'simple_harness_backend.py']


def embedding_template():
    # Both repository layout and flattened plugin skill installs are supported.
    for root in (SCRIPTS.parents[1], SCRIPTS.parents[2]):
        candidate = root / 'tao-generate-image-embeddings/assets/default_image_embeddings.yaml'
        if candidate.is_file():
            return candidate
    raise ValueError('Install companion skill tao-generate-image-embeddings')


def load_config(path):
    c = read(path)
    required = {'schema', 'profile', 'classes', 'task', 'image_root', 'references',
                'queries', 'query_split', 'model', 'siglip_model', 'staging', 'output'}
    if not isinstance(c, dict) or set(c) - (required | {'protocol'}) or required - set(c):
        raise ValueError('Unknown or missing config keys; see packaged example')
    if c['schema'] != 'tao-simple-harness/1' or c['profile'] != PROFILE:
        raise ValueError('Unsupported packaged profile/schema')
    classes = c['classes']
    if (not isinstance(classes, list) or not 2 <= len(classes) <= 32
            or any(not isinstance(x, str) or not x.strip() or len(x) > 64 for x in classes)
            or len(set(classes)) != len(classes)):
        raise ValueError('classes must contain 2–32 unique, nonempty names (max 64 characters)')
    if not isinstance(c['task'], str) or not c['task'].strip():
        raise ValueError('task must be nonempty text')
    if c['query_split'] not in {'proxy', 'validation', 'benchmark', 'test', 'inference'}:
        raise ValueError('Unknown query_split')
    if not isinstance(c['model'], dict) or set(c['model']) != {'path'}:
        raise ValueError('model must name a prepared local Cosmos snapshot path')
    for key in ('image_root', 'references', 'queries', 'staging', 'output'):
        if not isinstance(c[key], str) or not Path(c[key]).is_absolute():
            raise ValueError(f'{key} must be an absolute path, stable inside the container')
        c[key] = str(Path(c[key]).resolve())
    if not isinstance(c['model']['path'], str) or not Path(c['model']['path']).is_absolute():
        raise ValueError('model.path must be absolute')
    c['model']['path'] = str(Path(c['model']['path']).resolve())
    if not isinstance(c['siglip_model'], str) or not Path(c['siglip_model']).is_absolute():
        raise ValueError('siglip_model must name the same pinned local snapshot for both jobs')
    if c['staging'] == c['output']:
        raise ValueError('staging and output must be different directories')
    protocol = c.get('protocol', {})
    if not isinstance(protocol, dict) or set(protocol) - set(DEFAULTS):
        raise ValueError('Unknown protocol settings')
    protocol = {**DEFAULTS, **protocol}
    bounds = dict(examples_per_class=(1, 16), tool_examples_per_class=(1, 16),
                  max_tool_rounds=(1, 10), max_new_tokens=(32, 4096), seed=(0, 0))
    for key, (low, high) in bounds.items():
        if type(protocol[key]) is not int or not low <= protocol[key] <= high:
            raise ValueError(f'protocol.{key} must be an integer in [{low}, {high}]')
    c['protocol'] = protocol
    return c


def inventories(c):
    from PIL import Image
    root = Path(c['image_root']).resolve(strict=True)
    result = {}
    for role in ('references', 'queries'):
        rows, ids, paths = [], set(), set()
        for line in Path(c[role]).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            allowed = {'id', 'image', 'group_id'} | ({'label'} if role == 'references' else set())
            if (not isinstance(row, dict) or set(row) - allowed
                    or not {'id', 'image'} <= set(row)):
                raise ValueError(f'{role}: only id/image/group_id allowed; no query labels or gold')
            for key in ('id', 'image', 'group_id'):
                if key in row and (not isinstance(row[key], str) or not row[key].strip()):
                    raise ValueError(f'{role}: {key} must be nonempty text')
            if 'group_id' in row and row['group_id'] != row['group_id'].strip():
                raise ValueError('group_id must not have surrounding whitespace')
            if row['id'] in ids:
                raise ValueError(f'{role}: duplicate ID')
            ids.add(row['id'])
            if role == 'references' and row.get('label') not in c['classes']:
                raise ValueError('Reference label not in configured classes')
            image = (root / row['image']).resolve(strict=True)
            if not image.is_relative_to(root) or not image.is_file():
                raise ValueError('Image outside image_root')
            if str(image) in paths:
                raise ValueError(f'{role}: duplicate image path')
            paths.add(str(image))
            with Image.open(image) as img:
                img.verify()
            rows.append({**row, 'image': str(image), 'image_sha256': sha(image)})
        if not rows:
            raise ValueError(f'{role} is empty')
        result[role] = rows
    all_rows = result['references'] + result['queries']
    if any('group_id' in r for r in all_rows) and not all('group_id' in r for r in all_rows):
        raise ValueError('Supply group_id for every reference and query, or omit it everywhere')
    return result


def input_hashes(c, path, rows):
    return hashes([Path(path).resolve(), c['references'], c['queries'],
                   embedding_template(), *implementation_files(), *snapshot_files(c['siglip_model']),
                   *(r['image'] for role in rows.values() for r in role)])


def stage(config_path):
    """Create label-free input parquets/specs; embedding jobs remain platform-owned."""
    import pandas as pd
    import yaml
    c = load_config(config_path)
    rows = inventories(c)
    binding = input_hashes(c, config_path, rows)
    root = Path(c['staging'])
    receipt_path = root / 'staged.json'
    if receipt_path.exists():
        verify_seal(receipt_path)
        old = read(receipt_path)
        if old['inputs'] != binding:
            raise ValueError('Embedding stage inputs changed; use new staging/output directories')
        verify_files(old['files'])
        return old
    if root.exists() and any(root.iterdir()):
        raise ValueError('Incomplete or foreign staging directory; use a new directory')
    root.mkdir(parents=True, exist_ok=True)
    files = []
    receipt = dict(schema='tao-simple-harness-stage/1', profile=PROFILE,
                   embedding_skill='tao-generate-image-embeddings', inputs=binding)
    for role in ('references', 'queries'):
        spec = yaml.safe_load(embedding_template().read_text())
        spec.update(model='SigLIP', model_path=c['siglip_model'],
                    input_parquet=str(root / f'{role}.input.parquet'),
                    output_parquet=str(root / f'{role}.embeddings.parquet'))
        pd.DataFrame({'filepath': [r['image'] for r in rows[role]]}).to_parquet(
            spec['input_parquet'], index=False)
        spec_path = root / f'{role}.yaml'
        spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
        files.extend([spec_path, spec['input_parquet']])
        receipt[role + '_spec'] = str(spec_path)
    receipt['files'] = hashes(files)
    seal(receipt_path, receipt)
    return receipt


def snapshot_files(model):
    root = Path(model)
    if not root.is_dir() or not (root / 'config.json').is_file():
        raise ValueError('Missing local model snapshot/config.json')
    files = sorted(p for p in root.rglob('*')
                   if p.is_file() and '.cache' not in p.relative_to(root).parts)
    if not any(p.suffix == '.safetensors' for p in files):
        raise ValueError('Expected local safetensors model snapshot')
    return files


def collision(a, b):
    return (a['id'] == b['id'] or a['image_sha256'] == b['image_sha256']
            or ('group_id' in a and a['group_id'] == b.get('group_id')))


def draw_atlas(rows, classes, path, per_class):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    # User class labels are text in the prompt too; cell IDs identify each image.
    font = ImageFont.load_default()
    cell, caption = 192, 28
    canvas = Image.new('RGB', (cell * per_class, (cell + caption) * len(classes)), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(rows):
        x, y = (i % per_class) * cell, (i // per_class) * (cell + caption)
        with Image.open(row['image']) as img:
            tile = ImageOps.contain(img.convert('RGB'), (cell, cell))
            canvas.paste(tile, (x + (cell - tile.width) // 2, y))
        draw.text((x + 5, y + cell + 5), f'Class {i // per_class + 1} / example {i % per_class + 1}',
                  font=font, fill='black')
    canvas.save(path)


def prepare(config_path):
    import numpy as np
    import pandas as pd
    import yaml
    c = load_config(config_path)
    root = Path(c['output'])
    target = root / 'prepared.json'
    if target.exists():
        return checked_preparation(config_path)
    staged = stage(config_path)
    rows = inventories(c)
    embedding_files, dimension = [], None
    for role in ('references', 'queries'):
        spec = yaml.safe_load(Path(staged[role + '_spec']).read_text())
        path = Path(spec['output_parquet'])
        if not path.is_file():
            raise ValueError(f'Run tao-generate-image-embeddings with {staged[role + "_spec"]} first')
        table = pd.read_parquet(path)
        if set(table.columns) != {'filepath', 'embedding'} or table.filepath.duplicated().any():
            raise ValueError('Embedding parquet must contain unique filepath and embedding only')
        lookup = dict(zip(table.filepath, table.embedding))
        if set(lookup) != {r['image'] for r in rows[role]}:
            raise ValueError('Embedding paths do not match the staged inventory')
        for row in rows[role]:
            try:
                vector = np.asarray(lookup[row['image']], dtype=np.float64)
            except (ValueError, TypeError) as exc:
                raise ValueError('Malformed embedding vector') from exc
            if (vector.ndim != 1 or not vector.size or not np.isfinite(vector).all()
                    or not math.isfinite(float(np.linalg.norm(vector)))
                    or np.linalg.norm(vector) == 0):
                raise ValueError('Every embedding must be a finite, nonzero vector')
            dimension = dimension or len(vector)
            if len(vector) != dimension:
                raise ValueError('Embedding dimensions differ')
            row['embedding'] = (vector / np.linalg.norm(vector)).tolist()
        embedding_files.append(path)
    # One fixed atlas safe for every query, chosen without query labels or scores.
    eligible = [r for r in rows['references'] if not any(collision(r, q) for q in rows['queries'])]
    atlas, seen, seen_groups = [], set(), set()
    quota = c['protocol']['examples_per_class']
    for label in c['classes']:
        selected = []
        for row in sorted(eligible, key=lambda r: r['id']):
            if (row['label'] == label and row['image_sha256'] not in seen
                    and row.get('group_id', row['id']) not in seen_groups):
                seen.add(row['image_sha256'])
                seen_groups.add(row.get('group_id', row['id']))
                selected.append(row)
                if len(selected) == quota:
                    break
        if len(selected) < quota:
            raise ValueError(f'Insufficient eligible references for atlas class {label!r}')
        atlas.extend(selected)
    model_files = snapshot_files(c['model']['path'])
    if read(Path(c['model']['path']) / 'config.json').get('model_type') != 'qwen3_vl':
        raise ValueError('Prepare the Cosmos3-Nano VLM checkpoint (model_type=qwen3_vl) first')
    if root.exists() and any(root.iterdir()):
        raise ValueError('Nonempty unprepared output directory; use a new directory')
    root.mkdir(parents=True, exist_ok=True)
    atlas_path = root / 'atlas.png'
    draw_atlas(atlas, c['classes'], atlas_path, quota)
    prepared = dict(schema='tao-simple-harness-prepared/1', profile=PROFILE, config=c,
                    references=rows['references'], queries=rows['queries'],
                    query_count=len(rows['queries']), atlas=str(atlas_path),
                    atlas_ids=[r['id'] for r in atlas], embedding_dimension=dimension,
                    tool_round_limit=c['protocol']['max_tool_rounds'],
                    decision_owner='CosmosBackend', max_new_tokens=c['protocol']['max_new_tokens'],
                    generation_deviation='Packaged SigLIP protocol; not historical C-RADIO reproduction',
                    isolation=('id + image bytes + supplied group_id' if 'group_id' in rows['queries'][0]
                               else 'id + image bytes only; physical-group isolation NOT established'),
                    model_files=[str(p) for p in model_files],
                    encoder_files=[str(p) for p in snapshot_files(c['siglip_model'])],
                    files=hashes([*staged['inputs'], *staged['files'],
                                  Path(c['staging']) / 'staged.json', atlas_path,
                                  *embedding_files, *model_files]))
    seal(target, prepared)
    return prepared


def checked_preparation(config_path):
    c = load_config(config_path)
    path = Path(c['output']) / 'prepared.json'
    verify_seal(path)
    prepared = read(path)
    if prepared['config'] != c:
        raise ValueError('Frozen config changed')
    if [str(p) for p in snapshot_files(c['model']['path'])] != prepared['model_files']:
        raise ValueError('Frozen model file inventory changed')
    if [str(p) for p in snapshot_files(c['siglip_model'])] != prepared['encoder_files']:
        raise ValueError('Frozen encoder file inventory changed')
    verify_files(prepared['files'])
    return prepared


class ReferenceBank:
    def __init__(self, prepared):
        self.p = prepared
        self.c = prepared['config']
        self.queries = {q['id']: q for q in prepared['queries']}

    def select(self, query_id, name, arguments):
        if not isinstance(arguments, dict):
            raise ValueError('Tool arguments must be an object')
        expected = {'page'} if name == 'search_similar_examples' else {'page', 'classes'}
        if name not in {'search_similar_examples', 'get_class_examples'} or set(arguments) != expected:
            raise ValueError('Unknown tool or unexpected/missing arguments')
        page = arguments['page']
        if type(page) is not int or not 0 <= page <= 10000:
            raise ValueError('page must be a nonnegative integer <= 10000')
        query = self.queries[query_id]
        atlas_ids = set(self.p['atlas_ids'])
        atlas_hashes = {r['image_sha256'] for r in self.p['references'] if r['id'] in atlas_ids}
        atlas_groups = {r['group_id'] for r in self.p['references']
                        if r['id'] in atlas_ids and 'group_id' in r}
        ranked, seen = [], set(atlas_hashes)
        for row in sorted(self.p['references'], key=lambda r: (
                -sum(a * b for a, b in zip(r['embedding'], query['embedding'])), r['id'])):
            if (row['id'] in atlas_ids or collision(row, query) or row['image_sha256'] in seen
                    or row.get('group_id') in atlas_groups):
                continue
            seen.add(row['image_sha256'])
            ranked.append(row)
        classes = arguments.get('classes')
        if name == 'search_similar_examples':
            nearest = list(dict.fromkeys(r['label'] for r in ranked))
            classes = nearest[page * 2:page * 2 + 2]
            offset = 0
        else:
            if (not isinstance(classes, list) or not 1 <= len(classes) <= 2
                    or any(not isinstance(x, str) or x not in self.c['classes'] for x in classes)
                    or len(set(classes)) != len(classes)):
                raise ValueError('classes must name one or two distinct configured classes')
            offset = page * self.c['protocol']['tool_examples_per_class']
        n = self.c['protocol']['tool_examples_per_class']
        return [r for label in classes for r in
                [r for r in ranked if r['label'] == label][offset:offset + n]]


def parse_reply(raw, classes):
    """One tool call or strict singleton final JSON; invalid finals are terminal."""
    if not isinstance(raw, str):
        return None, None
    raw = raw.strip()
    # Cosmos may emit its native reasoning/answer wrappers. Removing complete
    # wrappers is decoding, not asking the model to repair an invalid answer.
    if raw.startswith('<think>'):
        match = re.fullmatch(r'<think>.*?</think>\s*(.*)', raw, re.S)
        if not match:
            return None, None
        raw = match.group(1).strip()
    if raw.startswith('<answer>') and raw.endswith('</answer>'):
        raw = raw[len('<answer>'):-len('</answer>')].strip()
    matches = list(re.finditer(r'<tool_call>(.*?)</tool_call>', raw, re.S))
    if '<tool_call>' in raw or '</tool_call>' in raw:
        if len(matches) != 1 or raw.count('<tool_call>') != 1 or raw.count('</tool_call>') != 1:
            return None, None
        if raw[:matches[0].start()].strip() or raw[matches[0].end():].strip():
            return None, None
        try:
            tool = json.loads(matches[0].group(1))
            if (isinstance(tool, dict) and set(tool) == {'name', 'arguments'}
                    and isinstance(tool['name'], str) and isinstance(tool['arguments'], dict)):
                return tool, None
        except (ValueError, TypeError):
            pass
        return None, None
    try:
        value = json.loads(raw)
        if (isinstance(value, dict) and set(value) == {'labels'}
                and isinstance(value['labels'], list) and len(value['labels']) == 1
                and isinstance(value['labels'][0], str) and value['labels'][0] in classes):
            return None, value['labels'][0]
    except (ValueError, TypeError):
        pass
    return None, None


def initial_messages(prepared, query):
    c = prepared['config']
    text = (c['task'] + '\nClass names: ' + json.dumps(c['classes']) +
            '\nThe first image is a fixed reference grid. Rows follow the class order above; '
            f'each row contains {c["protocol"]["examples_per_class"]} labeled examples. '
            'The second image is the query to classify.\n'
            'Use search_similar_examples to compare visually similar references when useful. '
            'Narrow down to one or two classes, then use get_class_examples if more evidence '
            'is needed. Search pages advance to other classes; class-example pages advance '
            'within a class. You may search again when uncertain; do not repeat an unchanged '
            'call. Tools are optional. You decide when to answer.\n'
            'For a tool use exactly <tool_call>{"name":"TOOL","arguments":{...}}</tool_call>. '
            'Use one tool per turn. Otherwise output only {"labels":["CLASS_NAME"]}, '
            'with exactly one of the class names. No prose or markdown.')
    return [{'role': 'user', 'content': [{'type': 'text', 'text': text},
                                        {'type': 'image', 'image': prepared['atlas']},
                                        {'type': 'image', 'image': query['image']}]}]


@contextlib.contextmanager
def exclusive_run(root):
    with (root / '.run.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another worker owns this output directory') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def run(config_path, backend=None):
    p = checked_preparation(config_path)
    c, root = p['config'], Path(p['config']['output'])
    with exclusive_run(root):
        return run_locked(p, root, backend)


def run_locked(p, root, backend):
    c, bank = p['config'], ReferenceBank(p)
    completed = root / 'complete.json'
    traces = []
    # Inspect all interrupted/sealed cases before allocating the model or continuing.
    for query in p['queries']:
        path = root / 'cases' / hashlib.sha256(query['id'].encode()).hexdigest() / 'trace.json'
        if path.exists():
            trace = read(path)
            if trace.get('status') != 'COMPLETE':
                raise ValueError(f'Interrupted/incomplete case {query["id"]}; retained, not retried')
            verify_seal(path)
            if (trace.get('query_id') != query['id']
                    or trace.get('prepared_sha256') != sha(root / 'prepared.json')):
                raise ValueError('Case belongs to a different frozen preparation')
        traces.append((query, path))
    if completed.exists():
        verify_seal(completed)
        summary = read(completed)
        verify_files(summary['artifacts'])
        return summary
    if backend is None and any(not path.exists() for _, path in traces):
        from simple_harness_backend import CosmosBackend
        backend = CosmosBackend(c)
    totals = dict(completed=0, invalid=0, model_calls=0, tool_calls=0, tool_errors=0,
                  input_tokens=0, output_tokens=0, generation_seconds=0.0)
    predictions = []
    for query, path in traces:
        if path.exists():
            trace = read(path)
        else:
            messages = initial_messages(p, query)
            trace = dict(status='RUNNING', query_id=query['id'], profile=PROFILE,
                         prepared_sha256=sha(root / 'prepared.json'), turns=[], label=None)
            write(path, trace)
            limit = c['protocol']['max_tool_rounds']
            for turn in range(limit + 1):
                if turn == limit:
                    messages.append({'role': 'user', 'content': 'Tool budget exhausted. '
                                     'Return only the final {"labels":["CLASS_NAME"]} now.'})
                item = dict(request=json.loads(json.dumps({
                    'messages': messages, 'tools': TOOLS if turn < limit else []})))
                trace['turns'].append(item)
                write(path, trace)
                started = time.monotonic()
                reply = backend.generate(messages, TOOLS if turn < limit else [])
                tool, label = parse_reply(reply['raw'], c['classes'])
                item.update(response=reply, wall_seconds=time.monotonic() - started)
                if reply.get('finish_reason') == 'length':
                    tool, label = None, None
                if tool and turn < limit:
                    item['tool_call'] = tool
                    messages.append({'role': 'assistant', 'content': '', 'tool_calls': [
                        {'type': 'function', 'function': tool}]})
                    try:
                        rows = bank.select(query['id'], tool['name'], tool['arguments'])
                        result = {'reference_ids': [r['id'] for r in rows],
                                  'labels': [r['label'] for r in rows],
                                  'images': [{'path': r['image'], 'sha256': r['image_sha256']}
                                             for r in rows],
                                  'empty': not bool(rows)}
                        messages.append({'role': 'tool', 'name': tool['name'],
                                         'content': json.dumps(result)})
                        # Qwen chat templates render images in user messages, not tool strings.
                        # Keep the tool receipt AND append the actual evidence for the next turn.
                        if rows:
                            content = [{'type': 'text', 'text': 'Tool reference images, in result order:'}]
                            for row in rows:
                                content.extend([{'type': 'text', 'text': 'Class: ' + row['label']},
                                                {'type': 'image', 'image': row['image']}])
                            messages.append({'role': 'user', 'content': content})
                    except ValueError as exc:
                        result = {'error': str(exc), 'reference_ids': []}
                        messages.append({'role': 'tool', 'name': tool['name'],
                                         'content': json.dumps(result)})
                    item['tool_result'] = result
                    write(path, trace)
                    continue
                trace['label'] = label
                trace['invalid'] = label is None
                break
            trace['status'] = 'COMPLETE'
            # Replace only this worker's RUNNING progress; publish a separate immutable seal.
            write(path, trace)
            path.with_suffix('.json.sha256').write_text(sha(path) + '\n')
        totals['completed'] += 1
        totals['invalid'] += int(trace['invalid'])
        totals['model_calls'] += len(trace['turns'])
        totals['tool_calls'] += sum('tool_call' in t for t in trace['turns'])
        totals['tool_errors'] += sum('error' in t.get('tool_result', {}) for t in trace['turns'])
        for t in trace['turns']:
            for name in ('input_tokens', 'output_tokens'):
                totals[name] += t['response'].get(name, 0)
            totals['generation_seconds'] += t['response'].get('seconds', t['wall_seconds'])
        predictions.append({'id': query['id'], 'labels': [trace['label']] if trace['label'] else [],
                            'invalid': trace['invalid']})
    seal(root / 'predictions.json', predictions)
    summary = dict(status='COMPLETE', profile=PROFILE, **totals,
                   artifacts=hashes([path for _, path in traces] + [root / 'predictions.json']))
    seal(completed, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['stage', 'prepare', 'run'])
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    result = {'stage': stage, 'prepare': prepare, 'run': run}[args.action](args.config)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in {'files', 'inputs', 'references', 'queries', 'model_files'}}, indent=2))


if __name__ == '__main__':
    main()
