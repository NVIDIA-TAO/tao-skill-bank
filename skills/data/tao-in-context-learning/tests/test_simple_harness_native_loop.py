# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the external native loop with synthetic evidence and no model calls.

Set SIMPLE_HARNESS_NATIVE_ROOT to the independently supplied native checkout.
No upstream implementation, real dataset, checkpoint or credentials are bundled.
These tests prove conversation control flow, not Cosmos classification quality.
"""
from __future__ import annotations

import copy
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope='module')
def native():
    source = os.environ.get('SIMPLE_HARNESS_NATIVE_ROOT')
    if not source:
        pytest.skip('Set SIMPLE_HARNESS_NATIVE_ROOT to test the external native runner')
    root = Path(source).resolve(strict=True)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(root))
        runtime = importlib.import_module('aoi_harness.analysis.tht_ablation_runtime')
        backends = importlib.import_module('aoi_harness.harness_backends')
        schemas = importlib.import_module('modes.tht-inspection.tools')
        evaluation = importlib.import_module('modes.tht-inspection.evaluation')
        # Reject a previously imported, unrelated checkout instead of testing it.
        for module in (runtime, backends, schemas, evaluation):
            assert Path(module.__file__).resolve().is_relative_to(root)
        yield SimpleNamespace(run_case=runtime.run_case,
                              assistant_message=runtime.assistant_message,
                              decode=backends.decode_local,
                              render=backends.render_local,
                              schemas=schemas.tool_schemas, parse=evaluation.parse)


def call(name, **arguments):
    return '<tool_call>' + json.dumps({'name': name, 'arguments': arguments}) + '</tool_call>'


class ScriptedBackend:
    """Only the replies are synthetic; native decoding is exercised each turn."""
    def __init__(self, native, replies):
        self.native = native
        self.replies = list(replies)
        self.requests = []

    def generate(self, messages, tools, root, request_path):
        request = copy.deepcopy({'messages': messages, 'tools': tools})
        self.requests.append(request)
        request_path.write_text(json.dumps(request))
        assert self.replies, 'The runner generated an unexpected extra model turn'
        raw = self.replies.pop(0)
        return {**self.native.decode(raw), 'raw': raw, 'seconds': 0.0,
                'usage': {'input_tokens': 10, 'completion_tokens': 3}}


class SyntheticSession:
    def __init__(self, images):
        self.images = images
        self.calls = []

    def execute(self, name, arguments):
        assert name in self.images, 'Unexpected tool reached the test executor'
        self.calls.append((name, copy.deepcopy(arguments)))
        return {'images': [str(self.images[name])], 'labels': ['Good'],
                'source_filename': 'private-mining-provenance.png'}


class SyntheticMode:
    name = 'tht-inspection'

    def __init__(self, session, native):
        self.session = session
        self.native = native

    def schemas(self, label_mode):
        assert label_mode == 'names'
        return self.native.schemas(label_mode)

    def prepare_case(self, bank, query, query_joint, root, case_dir, user, label_mode):
        return self.session, [{'type': 'text', 'text': user},
                              {'type': 'image', 'image': str(query.relative_to(root))}]

    def final_instruction(self, reasoning):
        return 'Tool budget exhausted. Return the final classification only.'

    def public_tool_result(self, output):
        return {k: v for k, v in output.items() if k not in ('images', 'source_filename')}

    def parse_answer(self, text, label_mode):
        return self.native.parse(text)


@pytest.fixture
def execute(native, tmp_path):
    from PIL import Image
    root = tmp_path / 'run'
    root.mkdir()
    query = tmp_path / 'query.png'
    Image.new('RGB', (16, 16), 'white').save(query)
    images = {}
    for name, color in [('search_similar_examples', 'red'), ('get_class_examples', 'blue')]:
        images[name] = root / (name + '.png')
        Image.new('RGB', (16, 16), color).save(images[name])

    def run(replies, max_rounds=10):
        session = SyntheticSession(images)
        backend = ScriptedBackend(native, replies)
        case = root / 'case'
        trace = native.run_case(backend, None, query, ('synthetic', 'query'), root,
                                case, None, 'Classify the original query.', max_rounds,
                                reasoning=False, label_mode='names', mode=SyntheticMode(session, native))
        persisted = json.loads((case / 'trace.json').read_text())
        assert persisted == trace
        return backend, session, trace, root
    return run


def test_direct_answer_does_not_force_a_tool(execute):
    backend, session, trace, _ = execute(['{"labels":["Good"]}'])
    assert session.calls == []
    assert len(backend.requests) == 1
    assert backend.requests[0]['tools']
    assert trace['status'] == 'complete'
    assert trace['prediction'] == 'Good'
    assert trace['execution_policy'] == 'tht_ablation_no_format_repair_v1'


def test_search_then_examples_is_model_selected_and_multiturn(execute):
    replies = [call('search_similar_examples', page=0),
               call('get_class_examples', classes=['Good'], page=1), '{"labels":["Good"]}']
    backend, session, trace, root = execute(replies)
    assert session.calls == [('search_similar_examples', {'page': 0}),
                             ('get_class_examples', {'classes': ['Good'], 'page': 1})]
    assert len(backend.requests) == len(trace['turns']) == 3
    assert trace['prediction'] == 'Good'
    # The second turn actually contains search evidence; the third adds examples.
    for turn, tool in [(1, 'search_similar_examples'), (2, 'get_class_examples')]:
        messages = backend.requests[turn]['messages']
        assistant, result, evidence = messages[-3:]
        assert assistant['role'] == 'assistant' and assistant['content'] == ''
        native_call = assistant['tool_calls'][0]
        assert native_call['function']['name'] == tool
        assert result['role'] == 'tool' and result['tool_call_id'] == native_call['id']
        assert 'source_filename' not in json.loads(result['content'])
        assert evidence['role'] == 'user'
        assert 'not a new query' in evidence['content'][0]['text']
        assert evidence['content'][1] == {'type': 'image', 'image': tool + '.png'}
        assert (root / evidence['content'][1]['image']).is_file()
        assert tool + '.png' in trace['assets']
    ids = [turn['model_response']['calls'][0]['id'] for turn in trace['turns'][:2]]
    assert len(set(ids)) == 2


def test_examples_can_be_requested_before_search(execute):
    backend, session, trace, _ = execute([
        call('get_class_examples', classes=['Good'], page=0), '{"labels":["Good"]}'])
    assert session.calls == [('get_class_examples', {'classes': ['Good'], 'page': 0})]
    assert len(backend.requests) == 2 and trace['prediction'] == 'Good'


def test_model_can_search_another_page_after_class_examples(execute):
    backend, session, trace, _ = execute([
        call('search_similar_examples', page=0),
        call('get_class_examples', classes=['Good', 'Bright'], page=0),
        call('search_similar_examples', page=1),
        call('get_class_examples', classes=['Interrupted'], page=1),
        '{"labels":["Interrupted"]}'])
    assert session.calls == [
        ('search_similar_examples', {'page': 0}),
        ('get_class_examples', {'classes': ['Good', 'Bright'], 'page': 0}),
        ('search_similar_examples', {'page': 1}),
        ('get_class_examples', {'classes': ['Interrupted'], 'page': 1})]
    assert len(backend.requests) == 5
    assert trace['prediction'] == 'Interrupted'
    # Every executed tool's image envelope is part of the next model request.
    for request in backend.requests[1:]:
        assert request['messages'][-1]['role'] == 'user'
        assert request['messages'][-1]['content'][1]['type'] == 'image'


def test_budget_adds_one_final_only_turn(execute):
    backend, session, trace, _ = execute([
        call('search_similar_examples', page=0), '{"labels":["Good"]}'], max_rounds=1)
    assert len(session.calls) == 1 and len(backend.requests) == 2
    assert backend.requests[-1]['tools'] == []
    assert 'final classification' in backend.requests[-1]['messages'][-1]['content']
    assert trace['turns'][-1]['final_only'] is True
    assert trace['prediction'] == 'Good'


def test_tool_request_in_final_only_turn_is_not_executed(execute):
    backend, session, trace, _ = execute([
        call('search_similar_examples', page=0),
        call('get_class_examples', classes=['Good'], page=0)], max_rounds=1)
    assert len(backend.requests) == 2
    assert session.calls == [('search_similar_examples', {'page': 0})]
    assert trace['status'] == 'invalid_final' and trace['prediction'] == 'INVALID'
    assert trace['turns'][-1]['tool_results'] == []


@pytest.mark.parametrize('raw', [
    '<tool_call>{broken}</tool_call>',
    call('search_similar_examples', page=0) + call('get_class_examples', classes=['Good']),
    'Extra text ' + call('search_similar_examples', page=0),
    '<think>unfinished ' + call('search_similar_examples', page=0),
    '{"labels": ["unknown"]}',
])
def test_malformed_or_multiple_actions_end_without_tool_execution(execute, raw):
    backend, session, trace, _ = execute([raw])
    assert session.calls == [] and len(backend.requests) == 1
    assert trace['status'] == 'invalid_final'
    assert trace['turns'][0]['model_response']['raw'] == raw


def test_native_assistant_call_renders_without_spurious_text(native, tmp_path):
    result = native.decode(call('get_class_examples', classes=['Good'], page=0))
    message = native.assistant_message(result, 4)
    assert message['content'] == ''
    assert message['tool_calls'][0]['id'] == 'call_4_0'
    rendered = native.render([message], tmp_path)
    assert rendered[0]['content'] == []
    assert json.loads(rendered[0]['tool_calls'][0]['function']['arguments']) == {
        'classes': ['Good'], 'page': 0}
