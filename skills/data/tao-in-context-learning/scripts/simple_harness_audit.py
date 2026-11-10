#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit completed tool traces without gold, images or model calls.

Repeated search means successful searches on distinct pages, not retries.
New references are relative to earlier tool results for the same query.
This establishes observable evidence gathering, never accuracy or reasoning.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from simple_harness import binding_digest, complete_seal, read, sha, within


def summarize(traces):
    counts = Counter()
    model_turns, search_pages, examples_calls = Counter(), Counter(), Counter()
    for trace in traces:
        if trace.get('status') not in ('complete', 'invalid_final'):
            raise ValueError('audit requires terminal traces')
        counts['queries'] += 1
        counts['invalid'] += trace['prediction'] == 'INVALID'
        model_turns[len(trace['turns'])] += 1
        counts['model_calls'] += len(trace['turns'])
        seen_pages, seen_requests, seen_references = set(), set(), set()
        search_attempts = example_attempts = new_searches = new_references = 0
        for turn in trace['turns']:
            requests = {call['id']: call for call in turn['model_response']['calls']}
            counts['length_limited_model_calls'] += bool(turn['model_response'].get('token_limit_reached'))
            for result in turn['tool_results']:
                counts['tool_calls'] += 1
                call = requests[result['call_id']]
                name, arguments, output = result['name'], call['arguments'], result['result']
                counts[name] += 1
                search_attempts += name == 'search_similar_examples'
                example_attempts += name == 'get_class_examples'
                if output.get('error'):
                    counts['tool_errors'] += 1
                    continue
                key = (name, json.dumps(arguments, sort_keys=True))
                counts['repeated_successful_requests'] += key in seen_requests
                seen_requests.add(key)
                references = {str(r['reference_id']) for r in output.get('references', []) if 'reference_id' in r}
                fresh = references - seen_references
                if name == 'search_similar_examples' and output.get('images'):
                    page = arguments.get('page', 0)
                    if seen_pages and page not in seen_pages:
                        new_searches += 1
                        new_references += len(fresh)
                    seen_pages.add(page)
                seen_references.update(references)
        counts['queries_with_multiple_search_attempts'] += search_attempts >= 2
        counts['queries_with_multiple_successful_search_pages'] += len(seen_pages) >= 2
        counts['additional_successful_search_pages'] += new_searches
        counts['new_references_from_additional_search_pages'] += new_references
        counts['queries_with_new_references_from_additional_search'] += new_references > 0
        counts['queries_with_multiple_example_calls'] += example_attempts >= 2
        search_pages[len(seen_pages)] += 1
        examples_calls[example_attempts] += 1
    # Stable zero-valued fields are useful when a policy produces no exploration.
    for key in ('queries', 'invalid', 'model_calls', 'tool_calls', 'tool_errors',
                'search_similar_examples', 'get_class_examples', 'length_limited_model_calls',
                'repeated_successful_requests', 'queries_with_multiple_search_attempts',
                'queries_with_multiple_successful_search_pages', 'additional_successful_search_pages',
                'new_references_from_additional_search_pages',
                'queries_with_new_references_from_additional_search', 'queries_with_multiple_example_calls'):
        counts.setdefault(key, 0)
    return {'schema': 'tao-simple-harness-exploration-audit/1', 'counts': dict(counts),
            'model_turn_histogram': dict(sorted(model_turns.items())),
            'successful_search_page_histogram': dict(sorted(search_pages.items())),
            'example_call_histogram': dict(sorted(examples_calls.items())),
            'interpretation': 'Observed tool behavior only; no gold access, accuracy claim or hidden reasoning inference.'}


def audit_run(root):
    """Reject stale or altered saved evidence before counting any behavior."""
    root = Path(root)
    prepared = read(root / 'prepared.json')
    binding = binding_digest(prepared)
    complete = read(root / 'complete.json')
    if read(root / 'complete-seal.json') != complete_seal(root, binding):
        raise ValueError('completed run seal changed')
    if any(complete.get(k) != v for k, v in prepared.items() if k != 'status'):
        raise ValueError('completed run differs from preparation')
    files = sorted((root / 'cases').glob('*/trace.json'))
    if complete.get('status') != 'COMPLETE' or len(files) != complete['completed'] or len(files) != prepared['query_count']:
        raise ValueError('complete query coverage required')
    traces = []
    for path in files:
        trace = read(path)
        seal = read(path.parent / 'adapter-receipt.json')
        requests = [turn['request'] for turn in trace['turns']]
        if (seal['binding_sha256'] != binding or seal['trace_sha256'] != sha(path)
                or len(set(requests)) != len(requests)
                or seal['request_sha256'] != {name: sha(within(root, name)) for name in requests}):
            raise ValueError('saved trace or request binding changed')
        for name, asset in trace['assets'].items():
            if sha(within(root, name)) != asset['sha256']:
                raise ValueError('saved image artifact changed')
        traces.append(trace)
    result = summarize(traces)
    result.update(profile=complete['profile'], prepared_sha256=sha(root / 'prepared.json'),
                  complete_sha256=sha(root / 'complete.json'),
                  complete_seal_sha256=sha(root / 'complete-seal.json'),
                  prompt_sha256=prepared['prompt_sha256'],
                  prompt_deviation=prepared.get('prompt_deviation'))
    result['trace_inventory_sha256'] = hashlib.sha256(''.join(sha(p) for p in files).encode()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit_run(args.results_dir)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
