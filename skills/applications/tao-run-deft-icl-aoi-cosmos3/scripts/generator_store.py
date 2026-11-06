# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Separate protocol boundary for the single generator addendum arm."""
import time

from suite_store import SuiteStore
from tht_contract import json_hash


class GeneratorStore(SuiteStore):
    allowed_arms = 'G'

    def initialize(self, config, approval):
        if not approval or approval.get('preflight_sha256') != json_hash(config):
            raise ValueError('explicit approval must bind the addendum preflight')
        with self.transaction():
            self.seal('config', config)
            self.seal('approval', approval)
            self.seal('protocol', 'generator-addendum-v1')

    def record_answer(self, arm, phase, candidate, answer):
        if phase == 'benchmark' and candidate != 'final':
            raise ValueError('only one frozen final Benchmark candidate is permitted')
        return super().record_answer(arm, phase, candidate, answer)

    def freeze(self, settings):
        config = self.get('config')
        if set(settings) != {'G'} or self.stage('G/select')['status'] != 'complete':
            raise ValueError('the generator arm must complete selection before freeze')
        if any(self.get(f'G/round/{i}') is None for i in range(1, config['max_deft_rounds'] + 1)):
            raise ValueError('all declared generator rounds must be complete')
        if settings['G']['base_fingerprint'] != config['base_fingerprint']:
            raise ValueError('generator classifier base lineage changed')
        import json
        selection = json.loads(self.stage('G/select')['result'])
        if settings['G']['checkpoint'] != selection['checkpoint']:
            raise ValueError('final checkpoint differs from Proxy selection')
        if 'generator' in config:
            expected={'policy_sha256':json_hash(config['generator']['policy']),
                      'preflight_sha256':json_hash(config),'prompt_sha256':config['prompt_sha256']}
            if any(settings['G'].get(k)!=v for k,v in expected.items()):
                raise ValueError('freeze differs from approved generator/prompt settings')
        payload = {'settings': settings, 'sha256': json_hash(settings)}
        with self.transaction():
            self.seal('freeze', payload)
            self.event('generator_arm_frozen', payload)
        return payload

    def completed_benchmark(self, expected_ids):
        self.verify_freeze()
        rows = self.answers('G', 'benchmark', 'final')
        return len(rows) == len(expected_ids) and {r['id'] for r in rows} == set(expected_ids)

    def stage_deadline(self, stage, cap, run_deadline):
        first = self.db.execute('SELECT MIN(started) FROM attempts WHERE stage=?', (stage,)).fetchone()[0]
        return min(run_deadline, (first if first is not None else time.time()) + cap)
