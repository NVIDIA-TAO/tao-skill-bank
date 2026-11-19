# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone user-data contract; all images and model responses are synthetic."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import shutil
import subprocess

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def test_runtime_is_packaged():
    assert (SCRIPTS / "simple_harness_runtime.py").is_file(), "Missing packaged runtime"


@pytest.fixture
def runtime():
    path = SCRIPTS / 'simple_harness_runtime.py'
    assert path.is_file(), 'The standalone runtime must ship in the skill bank'
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location('packaged_icl_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def dataset(tmp_path):
    classes = ['Normal', 'Scratch', 'Missing', 'Bridge']
    refs = []
    for c, label in enumerate(classes):
        for i in range(7):
            name = f'ref-{c}-{i}.png'
            Image.new('RGB', (24, 24), (c * 50, i * 30, 10)).save(tmp_path / name)
            refs.append({'id': name, 'image': name, 'label': label, 'group_id': name})
    Image.new('RGB', (24, 24), (23, 43, 53)).save(tmp_path / 'query.png')
    queries = [{'id': 'query', 'image': 'query.png', 'group_id': 'query-joint'}]
    for name, rows in [('refs.jsonl', refs), ('queries.jsonl', queries)]:
        (tmp_path / name).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text('{"model_type":"qwen3_vl"}')
    (model / 'model.safetensors').write_bytes(b'synthetic weights: never loaded')
    (model / 'tokenizer_config.json').write_text('{}')
    encoder = tmp_path / 'siglip'
    encoder.mkdir()
    (encoder / 'config.json').write_text('{"model_type":"siglip"}')
    (encoder / 'model.safetensors').write_bytes(b'synthetic encoder: never loaded')
    config = {'schema': 'tao-simple-harness/1', 'profile': 'simple-harness-siglip-v1',
              'classes': classes, 'task': 'Classify the visible manufacturing defect.',
              'image_root': str(tmp_path), 'references': str(tmp_path / 'refs.jsonl'),
              'queries': str(tmp_path / 'queries.jsonl'), 'query_split': 'validation',
              'model': {'path': str(model)}, 'siglip_model': str(encoder),
              'staging': str(tmp_path / 'staging'), 'output': str(tmp_path / 'results'),
              'protocol': {'examples_per_class': 1, 'tool_examples_per_class': 2,
                           'max_tool_rounds': 4, 'max_new_tokens': 128, 'seed': 0}}
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    return path, config, refs, queries


def supply_embeddings(runtime, dataset):
    path, config, refs, queries = dataset
    staged = runtime.stage(path)
    for role, rows in [('references', refs), ('queries', queries)]:
        spec = yaml.safe_load(Path(staged[role + '_spec']).read_text())
        table = pd.read_parquet(spec['input_parquet'])
        vectors = []
        for row in rows:
            c = config['classes'].index(row['label']) if 'label' in row else 0
            v = np.ones(4, dtype=float) * .1
            v[c] = 1
            vectors.append(v.tolist())
        table['embedding'] = vectors
        table.to_parquet(spec['output_parquet'], index=False)
    return staged


class ScriptedBackend:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def generate(self, messages, tools):
        self.requests.append(copy.deepcopy({'messages': messages, 'tools': tools}))
        raw = next(self.replies)
        return {'raw': raw, 'input_tokens': 20, 'output_tokens': 6,
                'seconds': .01, 'finish_reason': 'stop'}


def call(name, **arguments):
    return '<tool_call>' + json.dumps({'name': name, 'arguments': arguments}) + '</tool_call>'


def test_stage_uses_existing_siglip_skill_and_strips_reference_labels(runtime, dataset):
    staged = runtime.stage(dataset[0])
    a = yaml.safe_load(Path(staged['references_spec']).read_text())
    b = yaml.safe_load(Path(staged['queries_spec']).read_text())
    assert a['model'] == b['model'] == 'SigLIP'
    assert a['model_path'] == b['model_path'] == dataset[1]['siglip_model']
    assert list(pd.read_parquet(a['input_parquet']).columns) == ['filepath']
    assert list(pd.read_parquet(b['input_parquet']).columns) == ['filepath']
    assert staged['embedding_skill'] == 'tao-generate-image-embeddings'


def test_custom_classes_run_model_selected_repeated_search_and_images(runtime, dataset):
    supply_embeddings(runtime, dataset)
    prepared = runtime.prepare(dataset[0])
    assert prepared['query_count'] == 1
    backend = ScriptedBackend([call('search_similar_examples', page=0),
                               call('search_similar_examples', page=1),
                               call('get_class_examples', classes=['Scratch'], page=1),
                               '{"labels":["Scratch"]}'])
    result = runtime.run(dataset[0], backend=backend)
    assert result['completed'] == 1 and result['invalid'] == 0
    assert result['model_calls'] == 4 and result['tool_calls'] == 3
    assert len(backend.requests) == 4
    first = backend.requests[0]['messages'][0]['content']
    assert sum(p['type'] == 'image' for p in first) == 2  # atlas + query
    for turn in (1, 2, 3):
        assert any(m['role'] == 'tool' for m in backend.requests[turn]['messages'])
        assert sum(p['type'] == 'image' for m in backend.requests[turn]['messages']
                   if isinstance(m['content'], list) for p in m['content']) > 2
    traces = list((Path(dataset[1]['output']) / 'cases').glob('*/trace.json'))
    trace = json.loads(traces[0].read_text())
    pages = [t['tool_result']['reference_ids'] for t in trace['turns'][:2]]
    assert set(pages[0]).isdisjoint(pages[1])
    assert runtime.run(dataset[0], backend=ScriptedBackend([])) == result


@pytest.mark.parametrize('answer,invalid', [('{"labels":["Normal"]}', 0),
                                         ('not json', 1)])
def test_direct_answer_is_terminal_and_no_tool_forced(runtime, dataset, answer, invalid):
    supply_embeddings(runtime, dataset)
    runtime.prepare(dataset[0])
    backend = ScriptedBackend([answer])
    result = runtime.run(dataset[0], backend=backend)
    assert result['model_calls'] == 1 and result['tool_calls'] == 0
    assert result['invalid'] == invalid


def test_round_bound_adds_exactly_one_final_only_turn(runtime, dataset):
    supply_embeddings(runtime, dataset)
    runtime.prepare(dataset[0])
    backend = ScriptedBackend([call('search_similar_examples', page=0)] * 5)
    result = runtime.run(dataset[0], backend=backend)
    assert len(backend.requests) == 5
    assert backend.requests[-1]['tools'] == []
    assert result['invalid'] == 1


def test_tool_errors_return_to_model_and_are_not_final_classification(runtime, dataset):
    supply_embeddings(runtime, dataset)
    runtime.prepare(dataset[0])
    backend = ScriptedBackend([call('search_similar_examples', page=-1),
                               call('get_class_examples', classes=['invented'], page=0),
                               '{"labels":["Bridge"]}'])
    result = runtime.run(dataset[0], backend=backend)
    assert result['invalid'] == 0 and result['tool_errors'] == 2
    assert 'error' in backend.requests[1]['messages'][-1]['content']


def test_query_gold_is_rejected_before_staging(runtime, dataset):
    path, config, refs, queries = dataset
    queries[0]['label'] = 'Normal'
    Path(config['queries']).write_text(json.dumps(queries[0]) + '\n')
    with pytest.raises(ValueError, match='query|label'):
        runtime.stage(path)


def test_image_changes_invalidate_embedding_stage(runtime, dataset):
    supply_embeddings(runtime, dataset)
    Image.new('RGB', (24, 24), (1, 2, 3)).save(Path(dataset[1]['image_root']) / 'query.png')
    with pytest.raises(ValueError, match='changed|stage'):
        runtime.prepare(dataset[0])


def test_embedding_mismatch_and_nonfinite_vectors_fail(runtime, dataset):
    staged = supply_embeddings(runtime, dataset)
    spec_path = Path(staged['queries_spec'])
    spec = yaml.safe_load(spec_path.read_text())
    table = pd.read_parquet(spec['output_parquet'])
    table['embedding'] = [[float('nan')] * 4]
    table.to_parquet(spec['output_parquet'], index=False)
    with pytest.raises(ValueError, match='finite|embedding'):
        runtime.prepare(dataset[0])


def test_modified_prepared_run_cannot_resume_or_start(runtime, dataset):
    supply_embeddings(runtime, dataset)
    runtime.prepare(dataset[0])
    model = Path(dataset[1]['model']['path']) / 'model.safetensors'
    model.write_bytes(b'changed')
    backend = ScriptedBackend([])
    with pytest.raises(ValueError, match='changed|frozen'):
        runtime.run(dataset[0], backend=backend)
    assert backend.requests == []


def test_query_joint_and_atlas_members_excluded_from_tools(runtime, dataset):
    path, config, refs, queries = dataset
    refs[-1]['group_id'] = queries[0]['group_id']
    Path(config['references']).write_text(''.join(json.dumps(x)+'\n' for x in refs))
    supply_embeddings(runtime, dataset)
    prepared = runtime.prepare(path)
    bank = runtime.ReferenceBank(prepared)
    for name, args in [('search_similar_examples', {'page': 1}),
                       ('get_class_examples', {'classes': ['Bridge'], 'page': 0})]:
        rows = bank.select(queries[0]['id'], name, args)
        ids = [r['id'] for r in rows]
        assert refs[-1]['id'] not in ids
        assert not set(ids) & set(prepared['atlas_ids'])


def test_interrupted_case_is_preserved_and_not_retried(runtime, dataset):
    supply_embeddings(runtime, dataset)
    runtime.prepare(dataset[0])
    backend = ScriptedBackend([])
    with pytest.raises(StopIteration):
        runtime.run(dataset[0], backend=backend)
    with pytest.raises(ValueError, match='interrupted|incomplete'):
        runtime.run(dataset[0], backend=ScriptedBackend(['{"labels":["Normal"]}']))


def test_modified_terminal_trace_is_rejected(runtime, dataset):
    supply_embeddings(runtime, dataset)
    runtime.prepare(dataset[0])
    runtime.run(dataset[0], backend=ScriptedBackend(['{"labels":["Normal"]}']))
    trace = next((Path(dataset[1]['output']) / 'cases').glob('*/trace.json'))
    trace.write_text(trace.read_text() + '\n')
    with pytest.raises(ValueError, match='changed|seal'):
        runtime.run(dataset[0], backend=ScriptedBackend([]))


@pytest.mark.parametrize('vector', [[0, 0, 0, 0], [1, 2], [[1, 2]], [float('inf')]])
def test_bad_embedding_vectors_rejected(runtime, dataset, vector):
    staged = supply_embeddings(runtime, dataset)
    spec = yaml.safe_load(Path(staged['queries_spec']).read_text())
    table = pd.read_parquet(spec['output_parquet'])
    table['embedding'] = [vector]
    table.to_parquet(spec['output_parquet'], index=False)
    with pytest.raises(ValueError, match='embedding|Embedding'):
        runtime.prepare(dataset[0])


def test_external_image_path_rejected(runtime, dataset):
    path, c, _, queries = dataset
    outside = Path(c['image_root']).parent / 'outside.png'
    Image.new('RGB', (2, 2)).save(outside)
    queries[0]['image'] = str(outside)
    Path(c['queries']).write_text(json.dumps(queries[0]) + '\n')
    with pytest.raises(ValueError, match='outside'):
        runtime.stage(path)


def test_encoder_change_invalidates_stage(runtime, dataset):
    supply_embeddings(runtime, dataset)
    (Path(dataset[1]['siglip_model']) / 'config.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='changed'):
        runtime.prepare(dataset[0])


def test_query_overlap_is_excluded_from_fixed_atlas(runtime, dataset):
    path, c, refs, queries = dataset
    queries[0]['image'] = refs[0]['image']
    queries[0]['group_id'] = refs[0]['group_id']
    Path(c['queries']).write_text(json.dumps(queries[0]) + '\n')
    supply_embeddings(runtime, dataset)
    prepared = runtime.prepare(path)
    assert refs[0]['id'] not in prepared['atlas_ids']
    selected = runtime.ReferenceBank(prepared).select('query', 'search_similar_examples', {'page': 0})
    assert refs[0]['id'] not in [r['id'] for r in selected]


@pytest.mark.parametrize('raw', [
    '<tool_call>{bad}</tool_call>',
    call('search_similar_examples', page=0) + call('search_similar_examples', page=1),
    '{"labels":["Unknown"]}', '{"labels":["Normal","Scratch"]}',
    '{"labels":["Normal"],"extra":true}', '```json\n{"labels":["Normal"]}\n```',
])
def test_invalid_protocol_outputs_are_terminal(runtime, raw):
    assert runtime.parse_reply(raw, ['Normal', 'Scratch']) == (None, None)


def test_cosmos_complete_reasoning_and_answer_wrappers(runtime):
    assert runtime.parse_reply('<think>Compare the image.</think><answer>{"labels":["Normal"]}</answer>',
                               ['Normal', 'Scratch']) == (None, 'Normal')
    tool, label = runtime.parse_reply('<think>Need evidence.</think>' + call('search_similar_examples', page=0),
                                      ['Normal', 'Scratch'])
    assert tool['name'] == 'search_similar_examples' and label is None
    assert runtime.parse_reply('<think>unclosed' + call('search_similar_examples', page=0), ['Normal']) == (None, None)
    assert runtime.parse_reply('Extra text ' + call('search_similar_examples', page=0), ['Normal']) == (None, None)


def test_valid_looking_truncated_answer_is_invalid(runtime, dataset):
    supply_embeddings(runtime, dataset)
    runtime.prepare(dataset[0])
    class Truncated:
        def generate(self, messages, tools):
            return {'raw': '{"labels":["Normal"]}', 'finish_reason': 'length'}
    result = runtime.run(dataset[0], backend=Truncated())
    assert result['invalid'] == 1 and result['model_calls'] == 1


def test_requests_are_recorded_and_final_answer_never_repaired(runtime, dataset):
    supply_embeddings(runtime, dataset)
    runtime.prepare(dataset[0])
    backend = ScriptedBackend([call('search_similar_examples', page=0), 'not json'])
    runtime.run(dataset[0], backend=backend)
    trace = json.loads(next(Path(dataset[1]['output']).glob('cases/*/trace.json')).read_text())
    assert [t['request'] for t in trace['turns']] == backend.requests
    assert trace['invalid'] and len(trace['turns']) == 2
    assert trace['turns'][0]['tool_result']['images'][0]['sha256']


def test_packaged_application_preflight_subprocess(runtime, dataset):
    supply_embeddings(runtime, dataset)
    entry = SCRIPTS.parents[2] / 'applications/tao-run-deft-icl-aoi-cosmos3/scripts/simple_harness.py'
    receipt = dataset[0].parent / 'preflight.json'
    result = subprocess.run([sys.executable, str(entry), 'preflight', '--config', str(dataset[0]),
                             '--output', str(receipt)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    obj = json.loads(receipt.read_text())
    assert obj['profile'] == 'simple-harness-siglip-v1'
    assert obj['max_model_turns'] == 5
    assert obj['adapter_sha256'] == runtime.sha(SCRIPTS / 'simple_harness_runtime.py')


def test_flattened_install_has_no_private_source_dependency(dataset, tmp_path):
    flat = tmp_path / 'installed-skills'
    data = flat / 'tao-in-context-learning'
    app = flat / 'tao-run-deft-icl-aoi-cosmos3'
    embedding = flat / 'tao-generate-image-embeddings'
    (data / 'scripts').mkdir(parents=True)
    (app / 'scripts').mkdir(parents=True)
    (embedding / 'assets').mkdir(parents=True)
    for name in ['simple_harness_runtime.py', 'simple_harness_backend.py']:
        shutil.copy2(SCRIPTS / name, data / 'scripts' / name)
    shutil.copy2(SCRIPTS.parents[2] / 'applications/tao-run-deft-icl-aoi-cosmos3/scripts/simple_harness.py',
                 app / 'scripts/simple_harness.py')
    shutil.copy2(SCRIPTS.parents[1] / 'tao-generate-image-embeddings/assets/default_image_embeddings.yaml',
                 embedding / 'assets/default_image_embeddings.yaml')
    spec = importlib.util.spec_from_file_location('flat_runtime', data / 'scripts/simple_harness_runtime.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    supply_embeddings(module, dataset)
    result = subprocess.run([sys.executable, str(app / 'scripts/simple_harness.py'), 'preflight',
                             '--config', str(dataset[0]), '--output', str(tmp_path / 'flat-preflight.json')],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    module.run(dataset[0], backend=ScriptedBackend(['{"labels":["Normal"]}']))


def test_model_and_encoder_snapshots_under_huggingface_cache(runtime, dataset):
    path, c, _, _ = dataset
    cache = path.parent / '.cache/huggingface/hub'
    for key in ('model', 'siglip_model'):
        source = Path(c['model']['path'] if key == 'model' else c[key])
        snapshot = cache / ('models--' + key) / 'snapshots/pinned-revision'
        shutil.copytree(source, snapshot)
        if key == 'model':
            c['model']['path'] = str(snapshot)
        else:
            c[key] = str(snapshot)
    path.write_text(json.dumps(c))
    supply_embeddings(runtime, dataset)
    runtime.prepare(path)
    result = runtime.run(path, backend=ScriptedBackend(['{"labels":["Normal"]}']))
    assert result['completed'] == 1
