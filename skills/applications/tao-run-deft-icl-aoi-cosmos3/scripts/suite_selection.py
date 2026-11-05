# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Predeclared Proxy selection/stop rules; no Benchmark input is accepted."""
from __future__ import annotations
import math

def _completed(rows, key):
    seen = set()
    good = []
    for row in rows:
        if row['phase'] != 'proxy':
            raise ValueError('selection accepts Proxy evidence only')
        if row[key] in seen:
            raise ValueError('duplicate selection identity')
        seen.add(row[key])
        if row['status'] != 'complete':
            continue
        if row['n'] != row['expected_n'] or row['n'] <= 0:
            raise ValueError('candidate must cover the entire Proxy')
        if not math.isfinite(row['mcc']) or not -1 <= row['mcc'] <= 1:
            raise ValueError('invalid MCC')
        good.append(row)
    return good

def epoch_decision(rows, *, min_epochs=5, max_epochs=20, patience=3):
    complete = sorted(_completed(rows, 'epoch'), key=lambda r: r['epoch'])
    if [r['epoch'] for r in complete] != list(range(1, len(complete)+1)) or len(complete) > max_epochs:
        raise ValueError('epochs must be contiguous from 1 and within the frozen cap')
    best = None
    stale = 0
    for row in complete:
        if best is None or row['mcc'] > best['mcc']:
            best, stale = row, 0
        else:
            stale += 1
    stop = len(complete) >= max_epochs or (len(complete) >= min_epochs and stale >= patience)
    return {'stop': stop, 'reason': ('epoch_limit' if len(complete) >= max_epochs else 'proxy_patience') if stop else None,
            'best': best, 'non_improving_epochs': stale, 'next_epoch': len(complete)+1}

def select_round(rows):
    good = _completed(rows, 'round')
    if not good:
        raise ValueError('no complete DEFT round')
    return min(good, key=lambda r: (-r['mcc'], r['round']))

def select_icl(rows, *, max_trials=12, tolerance=0.01):
    good = _completed(rows, 'trial_id')
    if not good or len(rows) > max_trials:
        raise ValueError('no complete trial or trial budget exceeded')
    high = max(r['mcc'] for r in good)
    near = [r for r in good if high-r['mcc'] <= tolerance+1e-12]
    return min(near, key=lambda r: (r['images_per_query'], r['calls_per_query'], r['prompt_characters'], r['trial_id']))

def deft_input(base, prior_rounds, selected_f):
    """F supplies epoch count only. D resumes its own immediately prior round."""
    if not selected_f or selected_f['epoch'] < 1:
        raise ValueError('F epoch selection required before D')
    good = sorted(_completed(prior_rounds, 'round'), key=lambda r: r['round'])
    if [r['round'] for r in good] != list(range(1, len(good)+1)):
        raise ValueError('D continuation requires every prior round')
    return {'initial_model':base, 'load_checkpoint': good[-1]['checkpoint'] if good else None,
            'epochs':selected_f['epoch'], 'round':len(good)+1}
