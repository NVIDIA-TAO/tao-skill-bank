# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from collections import Counter

from PIL import Image, ImageDraw


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import analyze_gaps  # noqa: E402
import task_mining_router  # noqa: E402
from coverage_stratified_selector import capped_source_waterfill  # noqa: E402


TASK = "Defect Detection"


def _candidate(
    source: str,
    index: int,
    *,
    positive: bool,
    phenotype: str,
    cluster: str,
) -> dict:
    angle = ((index * 37) % 360) * 3.141592653589793 / 180.0
    # Exact trigonometry is unnecessary for the invariant under test; these
    # deterministic non-collinear vectors are normalized by the selector.
    embedding = [1.0, angle / 7.0 + 0.1, (index % 5 + 1) / 9.0]
    group = f"{source}-parent-{index:03d}"
    count_bin = ("1", "2-3", "4+")[index % 3] if positive else "0"
    return {
        "atomic_sample_id": f"atomic:{group}",
        "source_group_id": group,
        "parent_record_id": f"record:{group}",
        "task_type": TASK,
        "source_dataset": source,
        "canonical_phenotype": phenotype if positive else "__empty__",
        "log_bbox_area_quartile": f"Q{index % 4 + 1}" if positive else "NA",
        "local_contrast_quartile": f"Q{(index // 2) % 4 + 1}" if positive else "NA",
        "gt_count_bin": count_bin,
        "gt_count": {"0": 0, "1": 1, "2-3": 3, "4+": 4}[count_bin],
        "embedding": embedding,
        "visual_cluster": cluster,
        "filepath": f"images/{group}.png",
        "source_task_types": [TASK],
        "source_record_ids": [f"record:{group}"],
        "sample_kind": "single_image",
        "source_image_paths": [f"images/{group}.png"],
    }


def _pool() -> list[dict]:
    rows: list[dict] = []
    # Source A is deliberately dominant in the eligible inventory. Every
    # source still has enough positive/negative and phenotype supply for the
    # capped water-filling solution to select five rows from each source.
    for source, count in (("source_a", 56), ("source_b", 20), ("source_c", 16), ("source_d", 12)):
        for index in range(count):
            positive = index % 5 != 0
            rows.append(
                _candidate(
                    source,
                    index,
                    positive=positive,
                    phenotype=("bridge", "missing", "shift")[(index // 3) % 3],
                    cluster=f"cluster_{index % 4}",
                )
            )
    return rows


def _proxy(*, bridge_fn: int = 8, missing_fn: int = 2) -> list[dict]:
    rows = []
    for phenotype, fn in (("bridge", bridge_fn), ("missing", missing_fn), ("shift", 1)):
        rows.append(
            {
                "task_type": TASK,
                "canonical_phenotype": phenotype,
                "log_bbox_area_quartile": "Q1",
                "local_contrast_quartile": "Q1",
                "gt_count_bin": "1",
                "false_negative_count": fn,
                "false_positive_count": 0,
                "query_embedding": [1.0, 0.0, 0.0],
            }
        )
    for cluster in range(4):
        rows.append(
            {
                "task_type": TASK,
                "canonical_phenotype": "__empty__",
                "log_bbox_area_quartile": "NA",
                "local_contrast_quartile": "NA",
                "gt_count_bin": "0",
                "false_negative_count": 0,
                "false_positive_count": 2,
                "visual_cluster": f"cluster_{cluster}",
                "query_embedding": [0.0, 1.0, 0.0],
            }
        )
    return rows


class Cosmos3CoverageStratifiedSelectorTests(unittest.TestCase):
    def test_proxy_detection_contrast_is_computed_from_the_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            image = Image.new("L", (20, 20), color=0)
            ImageDraw.Draw(image).rectangle((5, 5, 15, 15), fill=255)
            image.save(root / "proxy.png")
            source = {
                "id": "proxy-dd",
                "task_type": TASK,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": "proxy.png"},
                            {"type": "text", "text": "detect defects"},
                        ],
                    },
                    {"role": "assistant", "content": "[]"},
                ],
            }
            contrast = analyze_gaps._proxy_local_contrast(
                source,
                [{"label": "bridge", "bbox_2d": [250, 250, 750, 750]}],
                media_root=root,
            )

            self.assertIsNotNone(contrast)
            self.assertGreater(contrast, 0.0)

    def test_proxy_false_positive_cluster_comes_from_cached_query_embedding(self) -> None:
        enriched = task_mining_router._attach_proxy_visual_clusters(
            [
                {
                    "atomic_sample_id": "atomic:fp",
                    "task_type": TASK,
                    "defect_detection_evidence": {"false_positive_count": 2},
                }
            ],
            [{"target_id": "atomic:fp", "embedding": [3.0, 4.0]}],
        )

        self.assertEqual(
            enriched[0]["visual_cluster"],
            task_mining_router._derived_visual_cluster([3.0, 4.0]),
        )

    def test_source_shortage_shrinks_budget_without_dominant_backfill(self) -> None:
        allocation, accepted, reasons = capped_source_waterfill(
            {"dominant": 100, "small_b": 1, "small_c": 1},
            10,
        )

        self.assertEqual(accepted, 3)
        self.assertEqual(allocation, {"dominant": 1, "small_b": 1, "small_c": 1})
        self.assertIn("source_capacity_reduced_budget:10->3", reasons)

    def test_router_builds_classification_inventory_without_query_knn(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            records = []
            source_rows = []
            for index, source in enumerate(("source_a", "source_b", "source_c", "source_d")):
                relative = f"images/{source}.png"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), color=(index * 40, 0, 0)).save(path)
                records.append(
                    {
                        "id": f"record-{source}",
                        "dataset": source,
                        "task_type": "Component Classification",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": relative},
                                    {"type": "text", "text": "A. Resistors"},
                                ],
                            },
                            {"role": "assistant", "content": "A"},
                        ],
                    }
                )
                source_rows.append(
                    {
                        "filepath": relative,
                        "embedding": [float(index == axis) for axis in range(4)],
                    }
                )
            target_rows = [
                {
                    "filepath": "proxy.png",
                    "target_id": "proxy",
                    "task_types": ["Component Classification"],
                    "defect_detection_evidence": [],
                    "embedding": [1.0, 0.0, 0.0, 0.0],
                }
            ]
            proxy_rows = [
                {
                    "task_type": "Component Classification",
                    "canonical_phenotype": "Resistors",
                    "sample_score": 0.0,
                    "query_embedding": [0.0, 0.0, 0.0, 1.0],
                }
            ]

            selected, summary = task_mining_router.route_candidates(
                target_rows,
                source_rows,
                records,
                media_root=root,
                mode="task_strict",
                top_k_per_target=4,
                min_similarity=1.0,
                candidate_selector="coverage_stratified_hardness_v1",
                proxy_rows=proxy_rows,
                round_index=1,
                epochs=5,
                iteration_budget=100,
            )

            self.assertEqual(len(selected), 4)
            self.assertEqual(summary["proxy_role"], "quota_statistics_only")
            self.assertTrue(summary["selector_manifest"]["training_allowed"])
            self.assertTrue(
                summary["selector_manifest"]["query_similarity_used_for_selection"]
                is False
            )

    def test_source_cap_floor_and_parent_diversity_are_enforced(self) -> None:
        selected, manifest = task_mining_router.select_coverage_stratified_candidates(
            _pool(),
            _proxy(),
            budget=20,
            round_index=1,
            epochs=5,
            iteration_budget=100,
        )

        self.assertTrue(manifest["training_allowed"])
        self.assertEqual(len(selected), 20)
        source_counts = Counter(row["source_dataset"] for row in selected)
        self.assertEqual(set(source_counts), {"source_a", "source_b", "source_c", "source_d"})
        for count in source_counts.values():
            self.assertLessEqual(count / len(selected), 0.35)
            self.assertGreaterEqual(count / len(selected), 0.10)
        parents = [row["source_group_id"] for row in selected]
        self.assertEqual(len(parents), len(set(parents)))

        diverse = task_mining_router.farthest_first_parent_order(
            [row for row in _pool() if row["source_dataset"] == "source_b"]
        )
        first_clusters = [row["visual_cluster"] for row in diverse[:4]]
        self.assertEqual(len(set(first_clusters)), 4)

        print("REALIZED_SHARE_TABLE")
        for source in sorted(source_counts):
            print(
                f"{source}\t{source_counts[source]}\t"
                f"{source_counts[source] / len(selected):.3f}"
            )

    def test_proxy_error_mass_changes_cell_quotas_not_query_similarity(self) -> None:
        kwargs = {
            "budget": 20,
            "round_index": 1,
            "epochs": 5,
            "iteration_budget": 100,
        }
        selected_a, manifest_a = task_mining_router.select_coverage_stratified_candidates(
            _pool(), _proxy(bridge_fn=10, missing_fn=1), **kwargs
        )
        query_changed = _proxy(bridge_fn=10, missing_fn=1)
        for row in query_changed:
            row["query_embedding"] = list(reversed(row["query_embedding"]))
        selected_b, manifest_b = task_mining_router.select_coverage_stratified_candidates(
            _pool(), query_changed, **kwargs
        )
        _, manifest_c = task_mining_router.select_coverage_stratified_candidates(
            _pool(), _proxy(bridge_fn=1, missing_fn=10), **kwargs
        )

        self.assertEqual(
            [row["source_group_id"] for row in selected_a],
            [row["source_group_id"] for row in selected_b],
        )
        self.assertEqual(manifest_a["cell_target_rows"], manifest_b["cell_target_rows"])
        bridge_a = sum(
            rows
            for cell, rows in manifest_a["cell_target_rows"].items()
            if "bridge" in cell
        )
        missing_a = sum(
            rows
            for cell, rows in manifest_a["cell_target_rows"].items()
            if "missing" in cell
        )
        bridge_c = sum(
            rows
            for cell, rows in manifest_c["cell_target_rows"].items()
            if "bridge" in cell
        )
        missing_c = sum(
            rows
            for cell, rows in manifest_c["cell_target_rows"].items()
            if "missing" in cell
        )
        self.assertGreater(bridge_a, missing_a)
        self.assertGreater(missing_c, bridge_c)

    def test_share_gap_gate_rejects_infeasible_tier_target(self) -> None:
        positive_only = [row for row in _pool() if row["gt_count_bin"] != "0"]
        _, manifest = task_mining_router.select_coverage_stratified_candidates(
            positive_only,
            _proxy(),
            budget=20,
            round_index=1,
            epochs=5,
            iteration_budget=100,
        )

        self.assertFalse(manifest["training_allowed"])
        self.assertGreater(manifest["max_target_realized_gap"], 0.05)
        with self.assertRaisesRegex(ValueError, "share gap"):
            task_mining_router.require_coverage_training_eligible(manifest)

    def test_inventory_cache_is_reused_only_when_input_hashes_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = pathlib.Path(temporary) / "coverage_inventory.parquet"
            calls = []

            def build() -> list[dict]:
                calls.append("built")
                return _pool()[:8]

            first, first_hashes, first_status = (
                task_mining_router.load_or_build_coverage_inventory(
                    cache,
                    expected_hashes={"annotations_sha256": "a", "embeddings_sha256": "b"},
                    builder=build,
                )
            )
            second, second_hashes, second_status = (
                task_mining_router.load_or_build_coverage_inventory(
                    cache,
                    expected_hashes={"annotations_sha256": "a", "embeddings_sha256": "b"},
                    builder=lambda: self.fail("valid cache should not rebuild"),
                )
            )

            self.assertEqual(calls, ["built"])
            self.assertEqual(first, second)
            self.assertEqual(first_hashes, second_hashes)
            self.assertEqual((first_status, second_status), ("built", "reused"))

            _, changed_hashes, changed_status = (
                task_mining_router.load_or_build_coverage_inventory(
                    cache,
                    expected_hashes={"annotations_sha256": "changed", "embeddings_sha256": "b"},
                    builder=build,
                )
            )
            self.assertEqual(calls, ["built", "built"])
            self.assertEqual(changed_hashes["annotations_sha256"], "changed")
            self.assertEqual(changed_status, "rebuilt")


if __name__ == "__main__":
    unittest.main()
