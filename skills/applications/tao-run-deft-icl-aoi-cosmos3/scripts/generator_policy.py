# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frozen Proxy-only allocation and clean-image routing for the generator addendum."""
from collections import Counter
import math
from generator_contract import validate_mapping
from tht_contract import CLASSES

POLICY = {
    'version': 'proxy-rate-rarity-decay-anchor-v1', 'rounds': 5,
    'requested_per_round': 28, 'weighted_slots': 25, 'exploration_slots': 3,
    'rarity_power': '-0.5', 'decay_factor': '0.5', 'decay_floor': '0.125',
    'arithmetic': 'binary64', 'remainder_tie': 'class_map_order',
    'exploration_count': 'cumulative_successful_final_logical_before_training_dedup',
    'exploration_remaining': 'separate_largest_remainder_effective_weights',
    'zero_weight_fallback': 'rarity', 'missing_proxy_support': 'preflight_reject',
    'anchors': 'sorted_error_ids_cycle_else_sorted_class_ids_else_native',
    'clean_selection': 'maximum_cosine_same_mining_embeddings_unique_within_round',
    'clean_tie': 'unique_canonical_stem', 'class_order': 'external_class_map',
    'base_seed': 42, 'amp_seed': 42, 'seed_increment_per_round': 1,
    'quality_guidance_range': [1.0, 10.0], 'quality_crop_range': [1.5, 2.5],
    'quality_draw_mode': 'native_per_sample_bayesopt_else_uniform', 'quality_bo_min_obs': 2,
    'quality_bo_candidates': 1024, 'quality_draw_seeds': [1, 2, 3], 'quality_selection': 'best_nn_earliest_tie',
    'quality_search_rounds': 3, 'score': 'nn', 'filter': None, 'top_up': False,
    'guidance': 3.0, 'num_steps': 35, 'crop_ratio': 2.0,
    'poisson_blend': False, 'num_generated_images': 1,
    'placement': 'cad_tht', 'donor_largest_component': True,
}


def _hamilton(n, weights):
    total = sum(weights)
    if n == 0:
        return [0] * len(weights)
    if total <= 0:
        raise ValueError('positive effective allocation weights required')
    quotas = [n * w / total for w in weights]
    out = [math.floor(q) for q in quotas]
    for i in sorted(range(len(out)), key=lambda i: (-(quotas[i] - out[i]), i))[:n-sum(out)]:
        out[i] += 1
    return out


def allocate(mapping, predictions, mining_counts, history, round_number):
    validate_mapping(mapping)
    if round_number not in range(1, 6) or len(history) != round_number - 1 or any(h['round'] != i+1 for i,h in enumerate(history)):
        raise ValueError('allocation history must contain every preceding round in order')
    if set(mining_counts) != set(mapping.values()) or any(type(n) is not int or n <= 0 for n in mining_counts.values()):
        raise ValueError('positive frozen real Mining class counts required; synthetic rows are excluded')
    if len({r['id'] for r in predictions}) != len(predictions) or any(r['truth'] not in CLASSES or r['prediction'] not in (*CLASSES,'INVALID') for r in predictions):
        raise ValueError('invalid or duplicate Proxy predictions')
    totals = Counter(r['truth'] for r in predictions)
    errors = Counter(r['truth'] for r in predictions if r['truth'] != r['prediction'])
    previous = {r['native_type']: r for r in history[-1]['classes']} if history else {}
    cumulative = {k: sum(h['generated_counts'][k] for h in history) for k in mapping}
    classes = []
    weights, rarity = [], []
    for key,label in mapping.items():
        rate = errors[label] / totals[label] if totals[label] else 0.0
        rare = 1.0 / math.sqrt(mining_counts[label])
        decay = previous[key]['decay'] if history else 1.0
        if history and history[-1]['counts'][key] > 0:
            prev = previous[key]
            if errors[label] * prev['proxy_count'] >= prev['errors'] * totals[label]:
                decay = max(.125, decay * .5)
        weight = rate * rare * decay
        weights.append(weight); rarity.append(rare)
        classes.append({'native_type':key, 'label':label, 'proxy_count':totals[label],
                        'errors':errors[label], 'error_rate':rate,
                        'mining_count':mining_counts[label], 'rarity':rare,
                        'decay':decay, 'weight':weight,'cumulative_generated':cumulative[key]})
    fallback = 'rarity' if not any(weights) else None
    effective = rarity if fallback else weights
    base = _hamilton(25,effective)
    eligible = sorted((i for i,r in enumerate(classes) if r['errors'] == 0), key=lambda i:(classes[i]['cumulative_generated'],i))[:3]
    explore = [int(i in eligible) for i in range(len(mapping))]
    remainder = _hamilton(3-len(eligible),effective)
    counts = {}
    for i,r in enumerate(classes):
        r.update(weighted=base[i], exploration=explore[i], exploration_redistributed=remainder[i])
        counts[r['native_type']] = base[i]+explore[i]+remainder[i]
    return {'round':round_number,'policy_version':POLICY['version'],'counts':counts,
            'classes':classes,'fallback':fallback,'total':sum(counts.values())}


def _cosine(a,b):
    if len(a) != len(b) or not a or any(not math.isfinite(float(v)) for v in [*a,*b]):
        raise ValueError('invalid native embedding')
    # Load the existing ICL scalar formula rather than a numerically different
    # vectorized or compensated-sum implementation.
    import suite_data  # establishes the companion module search path
    from retrieval import cosine
    return cosine(a,b)



def select_clean(mapping, allocation, predictions, proxy_vectors, clean_vectors):
    validate_mapping(mapping)
    if set(allocation['counts']) != set(mapping): raise ValueError('allocation class map mismatch')
    clean = sorted(clean_vectors, key=lambda r:(r['stem'],r['id']))
    if len({r['stem'] for r in clean}) != len(clean): raise ValueError('canonical clean stems must be unique')
    if len({r['id'] for r in clean}) != len(clean) or len(clean) < sum(allocation['counts'].values()):
        raise ValueError('insufficient unique clean sources for one round')
    vectors = {r['id']:r['embedding'] for r in proxy_vectors}
    if len(vectors) != len(proxy_vectors): raise ValueError('duplicate Proxy embedding')
    used = set(); work = []
    for native,label in mapping.items():
        anchors = sorted(r['id'] for r in predictions if r['truth']==label and r['prediction']!=label)
        mode = 'errors'
        if not anchors:
            anchors = sorted(r['id'] for r in predictions if r['truth']==label); mode='class_proxy'
        for index in range(allocation['counts'][native]):
            anchor = anchors[index % len(anchors)] if anchors else None
            chosen = None; score = None
            if anchor is not None:
                if anchor not in vectors: raise ValueError('anchor embedding missing')
                ranked = [(_cosine(vectors[anchor],r['embedding']),r) for r in clean if r['id'] not in used]
                score,chosen = min(ranked,key=lambda pair:(-pair[0],pair[1]['stem'],pair[1]['id']))
                used.add(chosen['id'])
            work.append({'slot':len(work),'class_ordinal':index,'round':allocation.get('round'),'anomaly_type':native,'label':label,'anchor_id':anchor,
                         'anchor_mode':mode if anchor else 'native_fallback','cosine':score,
                         'clean':{k:v for k,v in chosen.items() if k!='embedding'} if chosen else None})
    return work
