# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proxy-only multiclass error analysis and deterministic real-data routing.

This is a descriptive THT adaptation of the paired binary routing contract;
it does not infer visual/physical root causes from class labels.
"""
from collections import Counter

def route(scored,queries,*,phase,neighbors=4):
    if phase!='proxy': raise ValueError('Benchmark errors cannot drive DEFT')
    predictions=scored['predictions']; expected={q['id'] for q in queries}
    if {r['id'] for r in predictions}!=expected or len(predictions)!=len(expected):
        raise ValueError('RCCA requires full Proxy identity coverage')
    errors=[r for r in predictions if r['truth']!=r['prediction']]
    targets=[r['id'] for r in errors] if errors else sorted(expected)
    return {'protocol':'tht-error-neighbors-v1','phase':'proxy','neighbor_count':neighbors,
            'analysis_kind':'descriptive confusion analysis; no visual causal claim',
            'metrics':scored['metrics'],'errors':errors,
            'confusions':[{'truth':a,'prediction':b,'count':n} for (a,b),n in sorted(Counter((r['truth'],r['prediction']) for r in errors).items())],
            'mining_targets':targets,'routing':'all wrong-class and INVALID Proxy responses; uniform top-k cosine neighbors',
            'no_error_policy':'all Proxy queries tie as targets',
            'history':'union unique Mining IDs; previously selected records accumulate once',
            'sdg':{'status':'skipped','reason':'user-requested','synthetic_images':0},
            'adaptations_from_paired':'multiclass errors replace binary false-accept/reject routes; four neighbors; no golden images; Proxy-only selection; five rounds; no Benchmark gate'}
