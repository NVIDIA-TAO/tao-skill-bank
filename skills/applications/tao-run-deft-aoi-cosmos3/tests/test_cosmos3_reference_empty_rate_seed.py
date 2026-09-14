"""Regression: the reference empty-rate rule must track the combined (calibration + mined)
target the quota manifest verifies, not round each slice separately.

Observed 2026-09-14 (v12 acquisition run, 3,000 reference calibration pairs + 80 mined
reference rows, KPI empty rate 264/475): round(3000·r) + round(80·r) = 1667 + 44 = 1711
while the manifest requires round(3080·r) = 1712 → ``reference_empty_rate_matched`` False.
"""
import math
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import defect_detection_ablation as dda  # noqa: E402

RATE = 264 / 475
REF = dda.REFERENCE_DEFECT_DETECTION_TASK


def _entry(i: int, empty: bool) -> dict:
    return {
        "task_type": REF,
        "route_tier": "strict",
        "is_replay": False,
        "similarity": 1.0 - i * 1e-4,
        "record_id": f"ref-{'e' if empty else 'p'}-{i:04d}",
        "objects": [] if empty else [{"bbox": [0, 0, 1, 1]}],
    }


def _pool(n_each: int) -> list[dict]:
    return [_entry(i, True) for i in range(n_each)] + [_entry(i, False) for i in range(n_each)]


class ReferenceEmptyRateSeedTests(unittest.TestCase):
    def test_separately_rounded_slices_miss_the_combined_target_by_one(self) -> None:
        cal_total, cal_empty, mined = 3000, math.floor(3000 * RATE + 0.5), 80
        picked = dda._task_balanced(_pool(200), mined, max_novel=10**6, reference_empty_rate=RATE)
        empties = sum(not item["objects"] for item in picked)
        self.assertEqual(len(picked), mined)
        self.assertEqual(cal_empty, 1667)
        self.assertEqual(empties, 44)  # the old per-slice rounding
        self.assertNotEqual(cal_empty + empties, math.floor((cal_total + mined) * RATE + 0.5))  # 1711 != 1712

    def test_seeded_running_total_completes_the_combined_target(self) -> None:
        cal_total, mined = 3000, 80
        cal_empty = math.floor(cal_total * RATE + 0.5)
        picked = dda._task_balanced(
            _pool(200), mined, max_novel=10**6, reference_empty_rate=RATE, reference_seed=(cal_total, cal_empty)
        )
        empties = sum(not item["objects"] for item in picked)
        self.assertEqual(len(picked), mined)
        self.assertEqual(cal_empty + empties, math.floor((cal_total + mined) * RATE + 0.5))  # 1712

    def test_seed_matches_for_many_slice_sizes(self) -> None:
        for cal_total in (500, 1500, 3000):
            cal_empty = math.floor(cal_total * RATE + 0.5)
            for mined in range(0, 120, 7):
                picked = dda._task_balanced(
                    _pool(200), mined, max_novel=10**6, reference_empty_rate=RATE, reference_seed=(cal_total, cal_empty)
                )
                empties = sum(not item["objects"] for item in picked)
                self.assertEqual(
                    cal_empty + empties, math.floor((cal_total + mined) * RATE + 0.5), msg=f"cal={cal_total} mined={mined}"
                )

    def test_no_seed_keeps_the_legacy_single_slice_behaviour(self) -> None:
        picked = dda._task_balanced(_pool(400), 500, max_novel=10**6, reference_empty_rate=RATE)
        self.assertEqual(sum(not item["objects"] for item in picked), math.floor(500 * RATE + 0.5))


if __name__ == "__main__":
    unittest.main()
