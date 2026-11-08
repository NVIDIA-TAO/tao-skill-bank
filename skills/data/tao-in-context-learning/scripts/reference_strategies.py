# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit reference selection over a frozen embedding pool; no target labels.

This is a separate toolset. Legacy siglip-retrieval-v1/TMM semantics are unchanged.
Model invocation, preprocessing and scoring belong to the application.
"""
from __future__ import annotations
import hashlib
import json
import math
from retrieval import allowed_references, cosine, joint, policy_hash

TOOLSET = 'similarity-reference-v2'
PARAMETERS = {'strategy', 'reference_count', 'class_count', 'per_class',
              'distinct_joints', 'ordering', 'instruction'}


def validate(policy, capacity, classes):
    if policy.get('toolset') != TOOLSET or policy.get('version') != 2:
        raise ValueError('unknown reference toolset/version')
    if policy.get('max_calls') != 1 or not str(policy.get('rationale', '')).strip():
        raise ValueError('one-call policy and rationale required')
    if not classes or len(set(classes)) != len(classes):
        raise ValueError('unique configured class names required')
    p = policy.get('parameters', {})
    if set(p) != PARAMETERS or p['strategy'] not in ('nearest', 'different_classes'):
        raise ValueError('unsupported or incomplete reference policy')
    if type(p['reference_count']) is not int or not 1 <= p['reference_count'] <= capacity:
        raise ValueError('reference count exceeds declared/validated capacity')
    if type(p['distinct_joints']) is not bool or not isinstance(p['instruction'], str):
        raise ValueError('invalid diversity/instruction')
    if p['ordering'] not in ('nearest_first', 'class_ranked'):
        raise ValueError('unsupported reference ordering')
    if p['strategy'] == 'nearest':
        if p['class_count'] is not None or p['per_class'] is not None:
            raise ValueError('nearest selection does not use class quotas')
    elif (type(p['class_count']) is not int or not 1 <= p['class_count'] <= len(classes)
          or type(p['per_class']) is not int or p['per_class'] < 1
          or p['reference_count'] != p['class_count'] * p['per_class']):
        raise ValueError('class quotas do not match the reference count')
    return policy


def select(policy, query, query_embedding, references, classes, capacity):
    """Exclude before ranking. Input references are in frozen manifest order.

    A class is ranked by its best eligible cosine, canonical class-order ties.
    No query label is accepted; class names on Mining references are permitted.
    Insufficient class quota fails instead of silently changing the policy.
    """
    validate(policy, capacity, classes)
    if any(k in query for k in ('labels', 'truth', 'gold', 'prediction')):
        raise ValueError('query must be label-free')
    p = policy['parameters']; ranks = {name: i for i, name in enumerate(classes)}
    ranked = []
    for reference in allowed_references(query, references):
        if reference['labels'][0] not in ranks:
            raise ValueError('reference has an unknown class')
        similarity = cosine(query_embedding, reference['embedding'])
        if not math.isfinite(similarity):
            raise ValueError('nonfinite similarity')
        ranked.append(dict(reference, similarity=similarity))
    ranked.sort(key=lambda r: (-r['similarity'], r['id']))
    by_class = {}
    for r in ranked:
        by_class.setdefault(r['labels'][0], []).append(r)
    ordered_classes = sorted(by_class, key=lambda c: (-by_class[c][0]['similarity'], ranks[c]))
    chosen_classes = ordered_classes[:p['class_count']] if p['strategy'] == 'different_classes' else []
    if p['strategy'] == 'different_classes' and len(chosen_classes) != p['class_count']:
        raise ValueError('insufficient eligible candidate classes')
    groups = [by_class[c] for c in chosen_classes] if chosen_classes else [ranked]
    count = p['per_class'] if chosen_classes else p['reference_count']
    selected = []
    for group in groups:
        chosen = []; seen = set()
        for r in group:
            key = joint(r)
            if p['distinct_joints'] and key in seen:
                continue
            chosen.append(r); seen.add(key)
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError('insufficient eligible reference quota')
        selected.extend(chosen)
    if p['ordering'] == 'nearest_first':
        selected.sort(key=lambda r: (-r['similarity'], r['id']))
    elif not chosen_classes:
        class_rank = {c: i for i, c in enumerate(ordered_classes)}
        selected.sort(key=lambda r: (class_rank[r['labels'][0]], -r['similarity'], r['id']))
    return selected, {
        'schema': 'reference-selection/2', 'policy_sha256': policy_hash(policy),
        'eligible_count': len(ranked), 'selected_count': len(selected),
        'candidate_classes': chosen_classes, 'ordered_reference_ids': [r['id'] for r in selected],
        'exclusion': 'source-id-image-bytes-normalized-joint-v1',
        'class_rank': 'max-eligible-cosine-then-configured-class-order',
        'target_labels_used': False,
    }


def fixed_atlas(references, classes, per_class=4, salt='atlas-v1:'):
    """Build a query-independent Mining atlas manifest, not a random sample.

    For native Maxwell pixel/prompt parity use its pinned external renderer.
    This generic selector alone is not a reproduction of the full harness.
    """
    if type(per_class) is not int or per_class < 1:
        raise ValueError('positive atlas quota required')
    if not classes or len(set(classes)) != len(classes):
        raise ValueError('unique classes required')
    groups = {c: [] for c in classes}
    for r in references:
        if r.get('role', 'mining') != 'mining' or len(r.get('labels', [])) != 1:
            raise ValueError('single-label Mining atlas required')
        if r['labels'][0] not in groups:
            raise ValueError('unknown atlas class')
        joint(r)
        groups[r['labels'][0]].append(r)
    selected = []
    for c in classes:
        group = sorted(groups[c], key=lambda r: (hashlib.sha256((salt + r['id']).encode()).hexdigest(), r['id']))
        chosen = []; seen = set()
        for r in group:
            if joint(r) in seen:
                continue
            seen.add(joint(r)); chosen.append(r)
            if len(chosen) == per_class:
                break
        if len(chosen) != per_class:
            raise ValueError('insufficient atlas joints for a class')
        selected.extend(chosen)
    manifest = {'schema': 'fixed-atlas-selection/1', 'classes': list(classes), 'per_class': per_class,
                'salt': salt, 'reference_ids': [r['id'] for r in selected],
                'query_dependent': False, 'within_class_distinct_joints': True}
    manifest['selection_sha256'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return selected, manifest


def check_fixed_atlas(queries, selected):
    """Block fixed atlas reuse on overlap; never silently replace tiles per query."""
    bad = 0
    for q in queries:
        if any(k in q for k in ('labels', 'gold', 'truth', 'prediction')):
            raise ValueError('atlas audit uses label-free query identities')
        allowed = allowed_references(q, selected)
        bad += len(allowed) != len(selected)
    return {'schema': 'fixed-atlas-overlap/1', 'status': 'BLOCKED' if bad else 'PASS',
            'query_count': len(queries), 'queries_with_overlap_or_duplicate': bad,
            'tiles_replaced': False}
