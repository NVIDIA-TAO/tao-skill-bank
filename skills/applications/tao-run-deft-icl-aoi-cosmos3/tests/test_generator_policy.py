# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy amendment: measured rates, frozen rarity, history and exact clean routing."""
import importlib
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from tht_contract import CLASSES
MAPPING = {f'board+type{i}': c for i, c in enumerate(CLASSES[1:])}


def api():
    return importlib.import_module('generator_policy')


def rows():
    return [{'id': f'{c}-{i}', 'truth': c, 'prediction': 'INVALID' if i == 0 else c}
            for c in MAPPING.values() for i in range(4)]


def test_rates_rarity_decay_and_exploration():
    m = api(); mining = {c: 100 for c in MAPPING.values()}
    first = m.allocate(MAPPING, rows(), mining, [], 1)
    assert sum(first['counts'].values()) == 28
    assert all(r['error_rate'] == .25 and r['decay'] == 1 for r in first['classes'])
    history = []
    current = first
    for n in range(1, 5):
        history.append({**current, 'generated_counts': current['counts']})
        current = m.allocate(MAPPING, rows(), mining, history, n + 1)
    assert all(r['decay'] == .125 for r in current['classes'])
    correct = [{**r, 'prediction': r['truth']} for r in rows()]
    recovered = m.allocate(MAPPING, correct, mining, history, 5)
    assert recovered['fallback'] == 'rarity'
    assert [r['exploration'] for r in recovered['classes']].count(1) == 3
    assert sum(recovered['counts'].values()) == 28
    assert all(r['rarity'] == .1 for r in recovered['classes'])
    with pytest.raises(ValueError, match='history'):
        m.allocate(MAPPING, rows(), mining, [], 2)
    with pytest.raises(ValueError, match='Mining'):
        m.allocate(MAPPING, rows(), {**mining, next(iter(mining)): 0}, [], 1)


def test_unequal_proxy_denominators_and_no_floor():
    m = api(); labels = list(MAPPING.values())
    predictions = [{'id': str(i), 'truth': labels[0], 'prediction': 'INVALID' if i == 0 else labels[0]} for i in range(10)]
    predictions += [{'id': 'short', 'truth': labels[1], 'prediction': 'INVALID'}]
    out = m.allocate(MAPPING, predictions, {c: 100 for c in labels}, [], 1)
    assert [r['error_rate'] for r in out['classes']][:2] == [.1, 1]
    assert list(out['counts'].values()) == [2, 23, 1, 1, 1, 0, 0]
    assert out['classes'][2]['proxy_count'] == 0


def test_cosine_ties_unique_across_classes_and_sorted_cycling_anchors():
    m = api(); types = list(MAPPING); labels = list(MAPPING.values())
    allocation = {'counts': {k: (2 if i < 2 else 0) for i, k in enumerate(types)}}
    predictions = [{'id': i, 'truth': c, 'prediction': 'INVALID'} for c in labels[:2] for i in [c+'-z', c+'-a']]
    vectors = [{'id': r['id'], 'embedding': [1, 0]} for r in predictions]
    clean = [{'id': str(i), 'stem': s, 'embedding': [1, 0], 'prepared_path': s+'.png'} for i, s in enumerate(['d','b','a','c'])]
    work = m.select_clean(MAPPING, allocation, predictions, vectors, clean)
    assert [r['clean']['stem'] for r in work] == ['a','b','c','d']
    assert [r['anchor_id'] for r in work] == [labels[0]+'-a', labels[0]+'-z', labels[1]+'-a', labels[1]+'-z']
    assert all(r['cosine'] == 1 for r in work)
    assert work == m.select_clean(MAPPING, allocation, predictions, vectors, clean)
    with pytest.raises(ValueError, match='unique clean'):
        m.select_clean(MAPPING, allocation, predictions, vectors, clean[:3])


def test_separate_hamilton_pools_and_exploration_cardinality():
    m=api(); mining={c:100 for c in MAPPING.values()}
    errors=[{'id':c,'truth':c,'prediction':'INVALID'} for c in MAPPING.values()]
    out=m.allocate(MAPPING,errors,mining,[],1)
    assert list(out['counts'].values())==[5,5,5,4,3,3,3]
    for eligible in range(4):
        incoming=[{**r,'prediction':r['truth'] if i<eligible else 'INVALID'} for i,r in enumerate(errors)]
        result=m.allocate(MAPPING,incoming,mining,[],1)
        assert sum(r['weighted'] for r in result['classes'])==25
        assert sum(r['exploration'] for r in result['classes'])==eligible
        assert sum(r['exploration_redistributed'] for r in result['classes'])==3-eligible
        assert sum(result['counts'].values())==28


def test_improvement_retains_decay_and_all_blocked_requested_class_still_decays():
    m=api(); mining={c:100 for c in MAPPING.values()};incoming=rows()
    first=m.allocate(MAPPING,incoming,mining,[],1)
    for r in first['classes']: r['decay']=.5
    first['generated_counts']={k:0 for k in MAPPING}
    improved=[{**r,'prediction':r['truth']} for r in incoming]
    current=m.allocate(MAPPING,improved,mining,[first],2)
    assert all(r['decay']==.5 for r in current['classes'])
    unchanged=m.allocate(MAPPING,incoming,mining,[first],2)
    assert all(r['decay']==.25 for r in unchanged['classes'])
    first['counts'][next(iter(MAPPING))]=0
    untouched=m.allocate(MAPPING,incoming,mining,[first],2)
    assert untouched['classes'][0]['decay']==.5


def test_all_zero_fallback_uses_rarity_without_decay():
    m=api(); labels=list(MAPPING.values());mining={c:(1 if i==0 else 10000) for i,c in enumerate(labels)}
    correct=[{'id':c,'truth':c,'prediction':c} for c in labels]
    first=m.allocate(MAPPING,correct,mining,[],1);first['generated_counts']={k:0 for k in MAPPING}
    for i,r in enumerate(first['classes']): r['decay']=.125 if i==0 else 1
    next_round=m.allocate(MAPPING,correct,mining,[first],2)
    assert next_round['fallback']=='rarity'
    assert [r['weighted'] for r in next_round['classes']]==[r['weighted'] for r in first['classes']]


def test_cosine_is_original_scalar_formula_and_rejects_invalid_vectors():
    m=api()
    import suite_data
    from retrieval import cosine
    a=[1e16,1.,-1e16];b=[1.,1.,1.]
    assert m._cosine(a,b)==cosine(a,b)
    for bad in ([0.,0.,0.],[float('nan'),1.,1.],[float('inf'),1.,1.],[]):
        with pytest.raises(ValueError): m._cosine(a,bad)
