# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import pathlib
import sys
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import select_detection_calibration  # noqa: E402


def _row(record_id: str, boxes: list[dict], task: str = "Defect Detection") -> dict:
    return {
        "id": record_id,
        "task_type": task,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"images/{record_id}.png",
                        "min_pixels": 1,
                        "max_pixels": 1,
                    },
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
