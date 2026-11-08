# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from reference_strategies import select, fixed_atlas, check_fixed_atlas


class ReferenceStrategiesTests(unittest.TestCase):
    def setUp(self):
        self.q = dict(id='q', object_name='query', scan_id='SCN_x', image_sha256='qbytes')
        self.classes = ['A', 'B', 'C']
        self.refs = [dict(id=str(i), object_name=str(i), scan_id='board', image_sha256=str(i),
                          labels=[c], embedding=[1., i / 20.])
                     for i, c in enumerate(['A', 'A', 'B', 'B', 'B', 'C'], 1)]
        self.p = dict(toolset='similarity-reference-v2', version=2, rationale='synthetic distinct-class check',
                      max_calls=1, parameters=dict(strategy='different_classes', reference_count=2,
                      class_count=2, per_class=1, distinct_joints=False, ordering='class_ranked', instruction=''))

    def test_two_classes_is_not_nearest_two_and_uses_no_query_gold(self):
        selected, receipt = select(self.p, self.q, [1., 0.], self.refs, self.classes, 8)
        self.assertEqual([r['id'] for r in selected], ['1', '3'])
        self.assertFalse(receipt['target_labels_used'])
        nearest = copy.deepcopy(self.p)
        nearest['parameters'].update(strategy='nearest', class_count=None, per_class=None)
        chosen, _ = select(nearest, self.q, [1., 0.], self.refs, self.classes, 8)
        self.assertEqual([r['id'] for r in chosen], ['1', '2'])
        with self.assertRaises(ValueError):
            select(self.p, dict(self.q, labels=['B']), [1., 0.], self.refs, self.classes, 8)

    def test_exclusion_before_class_discovery_and_no_quota_fallback(self):
        refs = copy.deepcopy(self.refs)
        refs[0].update(object_name='query', scan_id='x')
        refs[1]['image_sha256'] = 'qbytes'
        chosen, _ = select(self.p, self.q, [1., 0.], refs, self.classes, 8)
        self.assertEqual([r['labels'][0] for r in chosen], ['B', 'C'])
        p = copy.deepcopy(self.p); p['parameters'].update(reference_count=4, per_class=2, distinct_joints=True)
        refs[3]['object_name'] = refs[2]['object_name']
        refs[4]['object_name'] = refs[2]['object_name']
        with self.assertRaises(ValueError):
            select(p, self.q, [1., 0.], refs, self.classes, 8)

    def test_fixed_atlas_is_stable_and_overlaps_block_without_replacing(self):
        a, m = fixed_atlas(self.refs, self.classes, 1)
        b, n = fixed_atlas(list(reversed(self.refs)), self.classes, 1)
        self.assertEqual(m, n); self.assertEqual(a, b)
        self.assertEqual(check_fixed_atlas([self.q], a)['status'], 'PASS')
        q = {k: v for k, v in a[0].items() if k not in ('labels', 'embedding')}
        result = check_fixed_atlas([q], a)
        self.assertEqual(result['status'], 'BLOCKED'); self.assertFalse(result['tiles_replaced'])


if __name__ == '__main__':
    unittest.main()
