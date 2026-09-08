# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import select_detection_calibration  # noqa: E402


def _row(
    record_id: str,
    boxes: list[dict],
    task: str = "Defect Detection",
    *,
    target: str | None = None,
    golden: str | None = None,
) -> dict:
    images = []
    if task.startswith("Ref_based"):
        images.append(
            {
                "type": "image",
                "image": golden or f"images/{record_id}-golden.png",
                "min_pixels": 1,
                "max_pixels": 1,
            }
        )
    images.append(
        {
            "type": "image",
            "image": target or f"images/{record_id}.png",
            "min_pixels": 1,
            "max_pixels": 1,
        }
    )
    return {
        "id": record_id,
        "task_type": task,
        "messages": [
            {
                "role": "user",
                "content": [
                    *images,
                    {"type": "text", "text": "detect"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"```json\n{json.dumps(boxes)}\n```"}
                ],
            },
        ],
    }


def _count_row(record_id: str, answer: int) -> dict:
    row = _row(record_id, [], task="Component Count")
    row["messages"][1]["content"][0]["text"] = str(answer)
    return row


class Cosmos3DetectionCalibrationContractTests(unittest.TestCase):
    def test_proxy_bound_cohort_quotas_keep_reference_pairs_atomic(self) -> None:
        one_box = [{"bbox_2d": [0, 0, 1, 1], "label": "x"}]
        proxy = [
            _row("proxy-single-empty", []),
            _row("proxy-single-few", one_box),
            _row("proxy-ref-empty", [], task="Ref_based Defect Detection"),
            _row("proxy-ref-few0", one_box, task="Ref_based Defect Detection"),
            _row("proxy-ref-few1", one_box, task="Ref_based Defect Detection"),
            _row("proxy-ref-few2", one_box, task="Ref_based Defect Detection"),
        ]
        source = [
            _row("single-empty0", []),
            _row("single-empty1", []),
            _row("single-few0", one_box),
            _row("single-few1", one_box),
            _row(
                "ref-empty",
                [],
                task="Ref_based Defect Detection",
                target="images/shared-target.png",
                golden="images/golden-empty.png",
            ),
            *[
                _row(
                    f"ref-few{index}",
                    one_box,
                    task="Ref_based Defect Detection",
                    target="images/shared-target.png",
                    golden=f"images/golden-{index}.png",
                )
                for index in range(3)
            ],
        ]

        with tempfile.TemporaryDirectory() as temporary:
            media_root = pathlib.Path(temporary)
            images = media_root / "images"
            images.mkdir()
            from PIL import Image

            for name in {
                "single-empty0.png",
                "single-empty1.png",
                "single-few0.png",
                "single-few1.png",
                "shared-target.png",
                "golden-empty.png",
                "golden-0.png",
                "golden-1.png",
                "golden-2.png",
            }:
                Image.new("RGB", (2, 2), color=(len(name), 0, 0)).save(images / name)
            rates = select_detection_calibration.derive_proxy_empty_rates(proxy)
            selected, summary = select_detection_calibration.select_calibration(
                source,
                media_root=media_root,
                cohort_quotas={
                    "non_reference_based": 4,
                    "reference_based": 4,
                },
                cohort_rates=rates,
                pair_assets_dir=media_root / "pair-assets",
                max_boxes=2,
            )

        self.assertEqual(rates["non_reference_based"]["empty_rate"], 0.5)
        self.assertEqual(rates["reference_based"]["empty_rate"], 0.25)
        self.assertEqual(summary["cohorts"]["non_reference_based"]["selected_empty"], 2)
        self.assertEqual(summary["cohorts"]["reference_based"]["selected_empty"], 1)
        reference = [row for row in selected if row["sample_kind"] == "reference_pair"]
        self.assertEqual(len(reference), 4)
        self.assertEqual(len({row["atomic_sample_id"] for row in reference}), 4)
        self.assertTrue(all(len(row["source_image_paths"]) == 2 for row in reference))
        self.assertTrue(all("pair-assets" in row["filepath"] for row in reference))
        negative = next(row for row in reference if row["calibration_box_count"] == 0)
        self.assertIn(
            "calibration_reference_no_change_ground_truth",
            negative["defect_detection_evidence"],
        )

    def test_merge_does_not_collapse_same_target_with_distinct_references(self) -> None:
        rows = [
            {
                "filepath": "/pairs/a.png",
                "atomic_sample_id": "reference_pair:" + "a" * 64,
                "sample_kind": "reference_pair",
                "source_image_paths": ["/golden/a.png", "/target/shared.png"],
                "route_tier": "calibration",
                "routed_task_types": ["Ref_based Defect Detection"],
            },
            {
                "filepath": "/pairs/b.png",
                "atomic_sample_id": "reference_pair:" + "b" * 64,
                "sample_kind": "reference_pair",
                "source_image_paths": ["/golden/b.png", "/target/shared.png"],
                "route_tier": "calibration",
                "routed_task_types": ["Ref_based Defect Detection"],
            },
        ]

        merged, duplicates = select_detection_calibration.merge_candidates(
            rows, [], media_root=pathlib.Path("/data")
        )

        self.assertEqual(len(merged), 2)
        self.assertEqual(duplicates, 0)
        self.assertEqual(
            {row["atomic_sample_id"] for row in merged},
            {"reference_pair:" + "a" * 64, "reference_pair:" + "b" * 64},
        )

    def test_merge_preserves_calibration_and_strict_tiers_per_task(self) -> None:
        atomic_id = "single_image:" + "c" * 64
        calibration = {
            "filepath": "/images/shared.png",
            "atomic_sample_id": atomic_id,
            "route_tier": "calibration",
            "route_tiers": ["calibration"],
            "route_tier_by_task": {"Defect Detection": "calibration"},
            "routed_task_types": ["Defect Detection"],
        }
        strict = {
            "filepath": "/images/shared.png",
            "atomic_sample_id": atomic_id,
            "route_tier": "strict",
            "route_tiers": ["strict"],
            "routed_task_types": ["Component Detection"],
        }

        merged, duplicates = select_detection_calibration.merge_candidates(
            [calibration], [strict], media_root=pathlib.Path("/data")
        )

        self.assertEqual(duplicates, 1)
        self.assertEqual(merged[0]["route_tiers"], ["calibration", "strict"])
        self.assertEqual(
            merged[0]["route_tier_by_task"],
            {
                "Defect Detection": "calibration",
                "Component Detection": "strict",
            },
        )

    def test_selects_bounded_empty_and_few_box_rows_and_excludes_many_boxes(self) -> None:
        rows = [
            _row("empty0", []),
            _row("empty1", []),
            _row("few0", [{"bbox_2d": [0, 0, 1, 1], "label": "x"}]),
            _row(
                "many",
                [
                    {"bbox_2d": [0, 0, 1, 1], "label": "x"},
                    {"bbox_2d": [1, 1, 2, 2], "label": "x"},
                    {"bbox_2d": [2, 2, 3, 3], "label": "x"},
                ],
            ),
        ]

        selected, summary = select_detection_calibration.select_calibration(
            rows,
            media_root=pathlib.Path("/data"),
            max_empty=1,
            max_few=2,
            max_boxes=2,
        )

        self.assertEqual(
            [pathlib.Path(row["filepath"]).name for row in selected],
            ["empty0.png", "few0.png"],
        )
        self.assertEqual(summary["selected_empty"], 1)
        self.assertEqual(summary["selected_few_box"], 1)
        self.assertEqual(summary["excluded_many_box"], 1)
        self.assertTrue(all(row["route_tier"] == "calibration" for row in selected))

    def test_skips_previously_mined_calibration_targets_before_filling_quotas(self) -> None:
        rows = [
            _row("empty0", []),
            _row("empty1", []),
            _row("few0", [{"bbox_2d": [0, 0, 1, 1], "label": "x"}]),
            _row("few1", [{"bbox_2d": [0, 0, 1, 1], "label": "x"}]),
        ]

        selected, summary = select_detection_calibration.select_calibration(
            rows,
            media_root=pathlib.Path("/data"),
            max_empty=1,
            max_few=1,
            max_boxes=2,
            excluded_identities={
                "/data/images/empty0.png",
                "/data/images/few0.png",
            },
        )

        self.assertEqual(
            [pathlib.Path(row["filepath"]).name for row in selected],
            ["empty1.png", "few1.png"],
        )
        self.assertEqual(summary["excluded_previously_mined"], 2)

    def test_selects_component_count_replay_outside_strict_gap_routing(self) -> None:
        selected, summary = select_detection_calibration.select_component_count_replay(
            [_count_row("count0", 2), _count_row("count1", 1)],
            media_root=pathlib.Path("/data"),
            max_count=1,
        )

        self.assertEqual(pathlib.Path(selected[0]["filepath"]).name, "count0.png")
        self.assertEqual(selected[0]["route_tier"], "count_replay")
        self.assertEqual(selected[0]["routed_task_types"], ["Component Count"])
        self.assertEqual(summary["selected_component_count"], 1)

    def test_count_replay_remains_strict_when_all_rows_are_excluded(self) -> None:
        with self.assertRaisesRegex(ValueError, "no eligible Component Count"):
            select_detection_calibration.select_component_count_replay(
                [_count_row("count0", 2)],
                media_root=pathlib.Path("/data"),
                max_count=1,
                excluded_identities={"/data/images/count0.png"},
            )

    def test_count_replay_can_explicitly_allow_an_exhausted_optional_tier(self) -> None:
        selected, summary = select_detection_calibration.select_component_count_replay(
            [_count_row("count0", 2)],
            media_root=pathlib.Path("/data"),
            max_count=1,
            excluded_identities={"/data/images/count0.png"},
            allow_empty=True,
        )

        self.assertEqual(selected, [])
        self.assertEqual(summary["selected_component_count"], 0)
        self.assertEqual(summary["excluded_previously_mined"], 1)
        self.assertEqual(summary["empty_reason"], "eligible_tier_exhausted")


if __name__ == "__main__":
    unittest.main()
