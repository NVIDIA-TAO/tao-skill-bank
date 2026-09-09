# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
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
    def test_router_deduplicates_one_atomic_source_but_keeps_ordered_pairs_distinct(self) -> None:
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
                atomic_samples.sample_from_record(record, media_root=root, context=record["id"])
                for record in records
            ]
            paths = [
                atomic_samples.embedding_filepath(
                    sample, pair_assets_dir=root / "pair-assets"
                )
                for sample in samples
            ]
            source_rows = [
                {"filepath": paths[0], "embedding": [1.0, 0.0]},
                {"filepath": paths[0], "embedding": [1.0, 0.0]},
                {"filepath": paths[1], "embedding": [0.0, 1.0]},
            ]
            target_rows = [
                {
                    "filepath": "query.png",
                    "target_id": "query",
                    "task_types": ["Ref_based Defect Detection"],
                    "defect_detection_evidence": [],
                    "embedding": [1.0, 1.0],
                }
            ]

            selected, summary = task_mining_router.route_candidates(
                target_rows,
                source_rows,
                records,
                media_root=root,
                pair_assets_dir=root / "pair-assets",
                mode="task_strict",
                top_k_per_target=2,
                min_similarity=-1.0,
            )

            self.assertEqual(summary["source_images"], 2)
            self.assertEqual(summary["duplicate_source_embeddings_collapsed"], 1)
            self.assertEqual(len(selected), 2)
            self.assertEqual(
                {row["atomic_sample_id"] for row in selected},
                {sample["atomic_sample_id"] for sample in samples},
            )

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
            [target["target_id"] for target in targets],
            ["reference_pair:a", "reference_pair:b"],
        )
        self.assertEqual(
            [target["source_target_ids"] for target in targets],
            [["shared-test"], ["shared-test"]],
        )
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
            self.assertIsNone(selected[0]["sim_golden"])
            self.assertIsNone(selected[0]["sim_test"])
            self.assertEqual(selected[0]["sim_pair"], 1.0)
            self.assertEqual(summary["pair_similarity"], "canvas")
            self.assertEqual([row["id"] for row in emitted], ["pair-b"])
            self.assertEqual(emit_summary["atomic_reference_pairs"], 1)

    def test_two_vector_similarity_prefers_same_board_type_and_audits_combine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            image_dir = root / "images"
            image_dir.mkdir()
            for name in (
                "query__C1036@2_UniformLight.png",
                "board-A__query.png",
                "board-A__candidate.png",
                "candidate__C1036@2_UniformLight.png",
                "candidate__C2048@1_UniformLight.png",
                "board-B__candidate.png",
            ):
                Image.new("RGB", (8, 8), color="white").save(image_dir / name)
            records = [
                _row(
                    "pair-a",
                    "images/candidate__C1036@2_UniformLight.png",
                    "images/board-A__candidate.png",
                ),
                _row(
                    "pair-b",
                    "images/candidate__C2048@1_UniformLight.png",
                    "images/board-B__candidate.png",
                ),
            ]
            target = {
                "filepath": "images/board-A__query.png",
                "target_id": "reference_pair:query-a",
                "atomic_sample_id": "reference_pair:query-a",
                "sample_kind": "reference_pair",
                "image_paths": [
                    "images/query__C1036@2_UniformLight.png",
                    "images/board-A__query.png",
                ],
                "reference_filepath": "images/query__C1036@2_UniformLight.png",
                "target_filepath": "images/board-A__query.png",
                "task_types": ["Ref_based Defect Detection"],
                "defect_detection_evidence": [],
            }
            target_inputs = route_selected_gaps.materialize_embedding_inputs(
                [target],
                media_root=root,
                pair_assets_dir=root / "pair-assets",
                pair_similarity="two_vector",
            )
            embeddings = {
                "query__C1036@2_UniformLight.png": [1.0, 0.0],
                "board-A__query.png": [1.0, 0.0],
                "board-A__candidate.png": [0.6, 0.8],
                "candidate__C1036@2_UniformLight.png": [1.0, 0.0],
                "candidate__C2048@1_UniformLight.png": [0.0, 1.0],
                "board-B__candidate.png": [1.0, 0.0],
            }
            target_rows = [
                {**row, "embedding": embeddings[pathlib.Path(row["filepath"]).name]}
                for row in target_inputs
            ]
            source_rows = [
                {
                    "filepath": str(image_dir / name),
                    "embedding": vector,
                }
                for name, vector in embeddings.items()
                if name not in {"query__C1036@2_UniformLight.png", "board-A__query.png"}
            ]

            mean_selected, mean_summary = task_mining_router.route_candidates(
                target_rows,
                source_rows,
                records,
                media_root=root,
                mode="task_strict",
                top_k_per_target=2,
                min_similarity=-1.0,
                pair_similarity="two_vector",
                pair_similarity_combine="mean",
            )
            min_selected, min_summary = task_mining_router.route_candidates(
                target_rows,
                source_rows,
                records,
                media_root=root,
                mode="task_strict",
                top_k_per_target=2,
                min_similarity=-1.0,
                pair_similarity="two_vector",
                pair_similarity_combine="min",
            )

            self.assertEqual(
                [row["source_record_ids"] for row in mean_selected],
                [["pair-a"], ["pair-b"]],
            )
            self.assertGreater(
                mean_selected[1]["sim_test"], mean_selected[0]["sim_test"]
            )
            self.assertAlmostEqual(mean_selected[0]["sim_golden"], 1.0)
            self.assertAlmostEqual(mean_selected[0]["sim_test"], 0.6)
            self.assertAlmostEqual(mean_selected[0]["sim_pair"], 0.8)
            self.assertAlmostEqual(min_selected[0]["sim_pair"], 0.6)
            self.assertAlmostEqual(min_selected[1]["sim_pair"], 0.0)
            self.assertEqual(
                mean_selected[0]["source_image_paths"],
                [
                    str(
                        (image_dir / "candidate__C1036@2_UniformLight.png").resolve()
                    ),
                    str((image_dir / "board-A__candidate.png").resolve()),
                ],
            )
            emitted, _ = emit_mined_sharegpt.emit_records(
                mean_selected[:1],
                records,
                media_root=root,
                relative=True,
                pair_assets_dir=None,
            )
            self.assertEqual([row["id"] for row in emitted], ["pair-a"])
            self.assertEqual(
                [
                    item["image"]
                    for item in emitted[0]["messages"][0]["content"]
                    if item["type"] == "image"
                ],
                [
                    "images/candidate__C1036@2_UniformLight.png",
                    "images/board-A__candidate.png",
                ],
            )
            for summary, combine in ((mean_summary, "mean"), (min_summary, "min")):
                self.assertEqual(summary["pair_similarity"], "two_vector")
                self.assertEqual(summary["pair_similarity_combine"], combine)
                evidence = summary["targets"][0]
                self.assertEqual(evidence["selected_reference_pairs"], 2)
                self.assertEqual(evidence["same_board_type_hits"], 1)
                self.assertEqual(evidence["same_golden_path_hits"], 0)
                self.assertEqual(evidence["same_board_id_prefix_hits"], 1)
                self.assertEqual(evidence["same_board_type_hit_rate"], 0.5)

    def test_two_vector_query_inputs_embed_a_shared_golden_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            image_dir = root / "images"
            image_dir.mkdir()
            for name in ("golden.png", "test-a.png", "test-b.png"):
                Image.new("RGB", (8, 8), color="white").save(image_dir / name)
            targets = [
                {
                    "filepath": f"images/test-{suffix}.png",
                    "target_id": f"reference_pair:{suffix}",
                    "sample_kind": "reference_pair",
                    "image_paths": [
                        "images/golden.png",
                        f"images/test-{suffix}.png",
                    ],
                    "task_types": ["Ref_based Defect Detection"],
                    "defect_detection_evidence": [],
                }
                for suffix in ("a", "b")
            ]

            rows = route_selected_gaps.materialize_embedding_inputs(
                targets,
                media_root=root,
                pair_assets_dir=root / "pair-assets",
                pair_similarity="two_vector",
            )

            self.assertEqual(len(rows), 3)
            golden_rows = [row for row in rows if row["filepath"].endswith("golden.png")]
            self.assertEqual(len(golden_rows), 1)
            memberships = json.loads(golden_rows[0]["embedding_memberships"])
            self.assertEqual(
                [item["target_id"] for item in memberships],
                ["reference_pair:a", "reference_pair:b"],
            )
            self.assertTrue(all(item["role"] == "golden" for item in memberships))
            self.assertFalse((root / "pair-assets").exists())


if __name__ == "__main__":
    unittest.main()
