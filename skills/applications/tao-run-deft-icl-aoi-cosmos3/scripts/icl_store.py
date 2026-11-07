# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Dedicated H journal: no historical-store writes or prefreeze Benchmark.
import json
import re
import sqlite3
from pathlib import Path
from suite_store import SuiteStore
from tht_contract import json_hash

PROTOCOL = 'icl-addendum-v1'


def check_root(root, *, resume=False):
    root = Path(root)
    if root.is_symlink():
        raise ValueError('results root must not be a symlink')
    if not root.exists():
        return
    if not root.is_dir():
        raise ValueError('results root must be a directory')
    database = root / 'suite.sqlite'
    if resume and database.is_file() and not database.is_symlink():
        try:
            with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as db:
                row = db.execute("SELECT value FROM meta WHERE key='protocol'").fetchone()
            if row and json.loads(row[0]) == PROTOCOL:
                return
        except (sqlite3.Error, ValueError):
            pass
        raise ValueError('refuse an existing non-ICL-addendum journal')
    allowed = {'driver.lock'} if resume else set()
    if any(p.name not in allowed or not p.is_file() or p.is_symlink() or p.stat().st_size
           for p in root.iterdir()):
        raise ValueError('fresh results root contains unowned artifacts')


class ICLStore(SuiteStore):
    allowed_arms = 'H'

    def __init__(self, root):
        check_root(Path(root).absolute(), resume=True)
        super().__init__(root)

    def initialize(self, config, approval):
        if (config.get('protocol') != PROTOCOL or not approval or
                approval.get('preflight_sha256') != json_hash(config) or
                not approval.get('approved_by') or not approval.get('approved_at')):
            raise ValueError('explicit approval must bind the exact ICL addendum preflight')
        with self.transaction():
            self.seal('config', config)
            self.seal('approval', approval)
            self.seal('protocol', PROTOCOL)

    def guard_stage(self, stage):
        if stage == 'H/benchmark/final' or stage == 'retrieval/embed-benchmark' or re.fullmatch(r'retrieval/benchmark-[0-9a-f]{20}', stage):
            self.verify_freeze()
        elif re.fullmatch(r'H/(proposal|proxy)/trial-0(0[1-9]|1[0-2])', stage) or stage == 'H/select':
            if self.get('freeze'):
                raise ValueError('selection cannot continue after freeze')
        elif not re.fullmatch(r'retrieval/(embed-(mining|proxy)|proxy-[0-9a-f]{20})', stage):
            raise ValueError('stage forbidden in inference-only H addendum: ' + stage)

    def begin(self, stage, dependencies=(), *, max_attempts=2):
        self.guard_stage(stage)
        if max_attempts != 2:
            raise ValueError('H retains the original two-attempt infrastructure envelope')
        return super().begin(stage, dependencies, max_attempts=max_attempts)

    def record_answer(self, arm, phase, candidate, answer):
        if arm != 'H' or phase not in ('proxy', 'benchmark'):
            raise ValueError('only H Proxy/final answers are permitted')
        if phase == 'proxy' and self.get('freeze'):
            raise ValueError('Proxy answers cannot change after freeze')
        if phase == 'benchmark' and candidate != 'final':
            raise ValueError('only one frozen final Benchmark candidate is permitted')
        return super().record_answer(arm, phase, candidate, answer)

    def freeze(self, settings):
        config = self.get('config')
        selection = self.stage('H/select')
        if set(settings) != {'H'} or selection['status'] != 'complete':
            raise ValueError('H must complete independent Proxy selection before freeze')
        selected = json.loads(selection['result'])
        value = settings['H']
        expected = {'checkpoint': config['fixed_checkpoint'],
                    'base_fingerprint': config['base_fingerprint'],
                    'preflight_sha256': json_hash(config), 'prompt_sha256': config['prompt_sha256'],
                    'seed': 42, 'statistics': config['statistics'],
                    'preprocessing': 'RGB/ImageOps.pad/256x256/NEAREST/PNG',
                    'generation': {'max_new_tokens': 4096, 'do_sample': False}}
        if any(value.get(k) != v for k,v in expected.items()):
            raise ValueError('freeze differs from approved checkpoint/protocol')
        if (selected['checkpoint'] != config['fixed_checkpoint'] or
                value.get('policy') != selected['policy'] or
                value.get('policy_sha256') != selected['policy_sha256']):
            raise ValueError('freeze differs from independent Proxy selection')
        payload = {'settings': settings, 'sha256': json_hash(settings)}
        with self.transaction():
            self.seal('freeze', payload)
            self.event('icl_arm_frozen', payload)
        return payload

    def completed_benchmark(self, expected_ids):
        self.verify_freeze()
        rows = self.answers('H', 'benchmark', 'final')
        return (len(expected_ids) == len(set(expected_ids)) == 271 and len(rows) == 271
                and {r['id'] for r in rows} == set(expected_ids))
