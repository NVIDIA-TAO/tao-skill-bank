# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt isolation and observable exploration accounting; synthetic only."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from simple_harness import apply_guidance, EXPLORE_PROFILE, PROFILES, binding_digest, case_seal, complete_seal, sha
from simple_harness_audit import summarize, audit_run


def turn(page, refs=('r1',), error=False):
    output = {'error': 'invalid page', 'images': []} if error else {
        'images': ['synthetic.png'], 'references': [{'reference_id': r} for r in refs]}
    return {'model_response': {'calls': [{'id': 'c', 'name': 'search_similar_examples',
                                        'arguments': {'page': page}}]},
            'tool_results': [{'call_id': 'c', 'name': 'search_similar_examples', 'result': output}]}


def trace(*turns):
    return {'status': 'complete', 'prediction': 'Good', 'turns': list(turns)}


def test_existing_profiles_keep_exact_prompt_and_settings():
    for profile in set(PROFILES) - {EXPLORE_PROFILE}:
        p = SimpleNamespace(user='original prompt', settings={'user_prompt': 'original prompt'})
        before = copy.deepcopy(vars(p))
        assert apply_guidance(p, profile) is None
        assert vars(p) == before


def test_exploration_prompt_and_deviation_are_bound_together():
    p = SimpleNamespace(user='original prompt', settings={})
    receipt = apply_guidance(p, EXPLORE_PROFILE)
    assert p.user.startswith('original prompt\n\n')
    assert p.settings['user_prompt'] == p.user
    assert p.settings['prompt_deviation'] == receipt
    assert len(receipt['sha256']) == 64


def test_retry_and_duplicate_search_do_not_count_as_new_exploration():
    r = summarize([trace(turn(0), turn(0), turn(1, error=True))])['counts']
    assert r['search_similar_examples'] == 3
    assert r['tool_errors'] == 1
    assert r['queries_with_multiple_search_attempts'] == 1
    assert r['queries_with_multiple_successful_search_pages'] == 0
    assert r['repeated_successful_requests'] == 1
    assert r['new_references_from_additional_search_pages'] == 0


def test_new_search_page_counts_only_new_reference_ids():
    r = summarize([trace(turn(0, ('r1', 'r2')), turn(1, ('r2', 'r3')))])['counts']
    assert r['queries_with_multiple_successful_search_pages'] == 1
    assert r['additional_successful_search_pages'] == 1
    assert r['new_references_from_additional_search_pages'] == 1


def test_incomplete_trace_is_not_completed_evidence():
    with pytest.raises(ValueError, match='terminal'):
        summarize([{'status': 'running'}])


@pytest.fixture
def completed_run(tmp_path):
    root = tmp_path / 'results'
    case = root / 'cases/q'
    case.mkdir(parents=True)
    put = lambda path, value: path.write_text(json.dumps(value))
    image = case / 'query.png'
    image.write_bytes(b'synthetic image bytes')
    request = case / 'request_00.json'
    put(request, {'messages': ['synthetic']})
    t = trace(turn(0))
    t['turns'][0]['request'] = str(request.relative_to(root))
    t['assets'] = {str(image.relative_to(root)): {'sha256': sha(image)}}
    put(case / 'trace.json', t)
    prepared = {'status': 'PREPARED', 'profile': EXPLORE_PROFILE, 'query_count': 1,
                'prompt_sha256': 'synthetic', 'prompt_deviation': {'kind': 'synthetic'}}
    put(root / 'prepared.json', prepared)
    binding = binding_digest(prepared)
    put(case / 'adapter-receipt.json', case_seal(root, case / 'trace.json', {'filename': 'q'}, binding))
    put(root / 'complete.json', {**prepared, 'status': 'COMPLETE', 'completed': 1})
    put(root / 'complete-seal.json', complete_seal(root, binding))
    return root


def test_audit_verifies_and_reports_frozen_prompt_provenance(completed_run):
    result = audit_run(completed_run)
    assert result['counts']['queries'] == 1
    assert result['profile'] == EXPLORE_PROFILE
    assert result['prepared_sha256'] == sha(completed_run / 'prepared.json')
    assert result['prompt_deviation'] == {'kind': 'synthetic'}


@pytest.mark.parametrize('relative', [
    'prepared.json', 'complete.json', 'cases/q/adapter-receipt.json',
    'cases/q/trace.json', 'cases/q/request_00.json', 'cases/q/query.png'])
def test_audit_rejects_changed_evidence(completed_run, relative):
    path = completed_run / relative
    if relative == 'prepared.json':
        value = json.loads(path.read_text())
        value['query_count'] = 2
        path.write_text(json.dumps(value))
    else:
        with path.open('ab') as f:
            f.write(b' ')
    # Preparation uses a canonical value digest; sealed artifacts use byte hashes.
    with pytest.raises(ValueError):
        audit_run(completed_run)
