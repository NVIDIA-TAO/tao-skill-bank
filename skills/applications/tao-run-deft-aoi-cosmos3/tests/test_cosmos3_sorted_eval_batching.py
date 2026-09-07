# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import Counter
import json
import pathlib
import sys
import tempfile
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import cfw_jsonl_runtime  # noqa: E402


def _row(
    record_id: str,
    task_type: str,
    prompt: str,
    *,
    image_count: int = 1,
) -> dict:
    return {
        "id": record_id,
        "task_type": task_type,
        "messages": [
            {
                "role": "user",
                "content": [
                    *(
                        {"type": "image", "image": f"images/{record_id}-{index}.png"}
                        for index in range(image_count)
                    ),
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "A"}]},
        ],
    }


class Cosmos3SortedEvalBatchingTests(unittest.TestCase):
    def test_sort_key_orders_image_count_prompt_length_and_id(self) -> None:
        rows = [
            _row("two-images", "Task A", "x", image_count=2),
            _row("long", "Task A", "longer", image_count=1),
            _row("tie-b", "Task A", "same", image_count=1),
            _row("tie-a", "Task A", "same", image_count=1),
            _row("other-task", "Task B", "x", image_count=1),
        ]

        ordered = sorted(rows, key=cfw_jsonl_runtime.evaluation_row_sort_key)

        self.assertEqual(
            [row["id"] for row in ordered],
            ["tie-a", "tie-b", "long", "two-images", "other-task"],
        )
        self.assertEqual(
            cfw_jsonl_runtime.evaluation_row_sort_key(rows[3]),
            ("Task A", 1, 4, "tie-a"),
        )

    def test_sorted_stride_shards_balance_each_task_and_preserve_coverage(self) -> None:
        task_sizes = {"Task A": 11, "Task B": 7, "Task C": 14}
        rows = [
            _row(f"{task}-{index:02d}", task, "x" * (index + 1))
            for task, size in task_sizes.items()
            for index in range(size)
        ]
        rows = rows[::2] + rows[1::2]

        with tempfile.TemporaryDirectory() as temporary:
            source = pathlib.Path(temporary) / "mixed.jsonl"
            source.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            shards = [
                list(
                    cfw_jsonl_runtime.iter_source_rows(
                        source,
                        num_shards=4,
                        shard_index=rank,
                        row_order="task_length_sorted",
                    )
                )
                for rank in range(4)
            ]
            source_order = list(
                cfw_jsonl_runtime.iter_source_rows(source, row_order="source")
            )

        for task in task_sizes:
            counts = [
                Counter(row["task_type"] for row in shard)[task]
                for shard in shards
            ]
            self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertCountEqual(
            [row["id"] for shard in shards for row in shard],
            [row["id"] for row in rows],
        )
        self.assertEqual(
            [row["id"] for row in source_order],
            [row["id"] for row in rows],
        )


if __name__ == "__main__":
    unittest.main()
