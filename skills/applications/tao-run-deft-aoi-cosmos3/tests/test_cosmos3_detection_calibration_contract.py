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


if __name__ == "__main__":
    unittest.main()
