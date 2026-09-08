# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

from PIL import Image


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import atomic_samples  # noqa: E402
import emit_mined_sharegpt  # noqa: E402
import route_selected_gaps  # noqa: E402
import task_mining_router  # noqa: E402


def _row(record_id: str, reference: str, target: str) -> dict:
    return {
        "id": record_id,
        "task_type": "Ref_based Defect Detection",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": reference},
                    {"type": "image", "image": target},
                    {"type": "text", "text": "detect"},
                ],
            },
            {"role": "assistant", "content": "[]"},
        ],
    }


class Cosmos3PairAtomicContractTests(unittest.TestCase):
    def test_gap_routing_keeps_same_test_image_with_different_references_distinct(self) -> None:
        rows = [
            {
                "id": "query-a",
                "evaluation_role": "proxy",
                "task_type": "Ref_based Defect Detection",
                "metric_family": "detection",
                "reference_cohort": "reference_based",
                "dataset": "fixture",
                "target_id": "shared-test",
                "target_path": "images/test.png",
                "reference_path": "images/golden-a.png",
                "image_paths": ["images/golden-a.png", "images/test.png"],
                "atomic_sample_id": "reference_pair:a",
            },
            {
                "id": "query-b",
                "evaluation_role": "proxy",
                "task_type": "Ref_based Defect Detection",
                "metric_family": "detection",
                "reference_cohort": "reference_based",
                "dataset": "fixture",
                "target_id": "shared-test",
                "target_path": "images/test.png",
                "reference_path": "images/golden-b.png",
                "image_paths": ["images/golden-b.png", "images/test.png"],
                "atomic_sample_id": "reference_pair:b",
            },
        ]

        targets, summary = route_selected_gaps.route(rows)

        self.assertEqual(len(targets), 2)
        self.assertEqual(summary["embedding_queries"], 2)
        self.assertEqual(
            [target["image_paths"] for target in targets],
            [
                ["images/golden-a.png", "images/test.png"],
                ["images/golden-b.png", "images/test.png"],
            ],
        )

    def test_router_retrieves_and_emitter_materializes_one_complete_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            for name, color in (
                ("golden-a.png", "white"),
                ("golden-b.png", "gray"),
                ("test.png", "black"),
            ):
                path = root / "images" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), color=color).save(path)
            records = [
                _row("pair-a", "images/golden-a.png", "images/test.png"),
                _row("pair-b", "images/golden-b.png", "images/test.png"),
            ]
            samples = [
                atomic_samples.sample_from_record(
                    record, media_root=root, context=record["id"]
                )
                for record in records
            ]
            source_rows = []
            for index, sample in enumerate(samples):
                source_rows.append(
                    {
                        "filepath": atomic_samples.embedding_filepath(
                            sample, pair_assets_dir=root / "pair-assets"
                        ),
                        "embedding": [float(index == 0), float(index == 1)],
                    }
                )
            target_rows = [
                {
                    "filepath": "query-pair.png",
                    "target_id": "query",
                    "task_types": ["Ref_based Defect Detection"],
                    "defect_detection_evidence": [],
                    "embedding": [0.0, 1.0],
                }
            ]

            selected, summary = task_mining_router.route_candidates(
                target_rows,
                source_rows,
                records,
                media_root=root,
                pair_assets_dir=root / "pair-assets",
                mode="task_strict",
                top_k_per_target=1,
                min_similarity=-1.0,
            )
            emitted, emit_summary = emit_mined_sharegpt.emit_records(
                selected,
                records,
                media_root=root,
                pair_assets_dir=root / "pair-assets",
                relative=True,
            )

            self.assertEqual(summary["source_reference_pairs"], 2)
            self.assertEqual(selected[0]["source_image_paths"], samples[1]["image_paths"])
            self.assertEqual([row["id"] for row in emitted], ["pair-b"])
            self.assertEqual(emit_summary["atomic_reference_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
