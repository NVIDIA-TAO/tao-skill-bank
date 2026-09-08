# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

from PIL import Image


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import analyze_gaps  # noqa: E402
import atomic_samples  # noqa: E402
import defect_detection_ablation  # noqa: E402
import route_selected_gaps  # noqa: E402
import task_mining_router  # noqa: E402


def _row(
    record_id: str,
    task_type: str,
    *,
    boxes: list[dict] | None = None,
    dataset: str = "source-a",
) -> dict:
    path = f"images/{record_id}.png"
    images = [
        {
            "type": "image",
            "image": path,
            "min_pixels": 1_048_576,
            "max_pixels": 1_048_576,
        }
    ]
    if task_type.startswith("Ref_based"):
        images.insert(
            0,
            {
                "type": "image",
                "image": f"images/{record_id}-golden.png",
                "min_pixels": 1_048_576,
                "max_pixels": 1_048_576,
            },
        )
    return {
        "id": record_id,
        "dataset": dataset,
        "task_type": task_type,
        "messages": [
            {
                "role": "user",
                "content": [
                    *images,
                    {"type": "text", "text": "official_v1 prompt"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "```json\n"
                        + json.dumps(boxes if boxes is not None else [])
                        + "\n```",
                    }
                ],
            },
        ],
    }


def _candidate(
    record: dict,
    *,
    evidence: list[str] | None = None,
    phash: str,
    is_replay: bool = False,
    route_tier: str = "strict",
) -> dict:
    image = [
        item["image"]
        for item in record["messages"][0]["content"]
        if item.get("type") == "image"
    ][-1]
    sample = atomic_samples.sample_from_record(
        record, media_root=pathlib.Path("/data"), context=str(record.get("id"))
    )
    return {
        "filepath": image,
        "atomic_sample_id": sample["atomic_sample_id"],
        "sample_kind": sample["sample_kind"],
        "source_image_paths": sample["image_paths"],
        # Most contract tests intentionally use synthetic paths; model the
        # trusted identity supplied by the embedding manifest for those rows.
        "content_sha256": hashlib.sha256(
            f"fixture:{sample['atomic_sample_id']}".encode()
        ).hexdigest(),
        "route_tier": route_tier,
        "routed_task_types": [record["task_type"]],
        "defect_detection_evidence": evidence or [],
        "perceptual_hash": phash,
        "local_contrast": (int(phash, 16) % 100) / 100.0,
        "max_cosine_similarity": 0.99,
        "is_replay": is_replay,
    }


class _Evaluator:
    @staticmethod
    def box_iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        ax1, ay1, ax2, ay2 = left
        bx1, by1, bx2, by2 = right
        intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
            0.0, min(ay2, by2) - max(ay1, by1)
        )
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
        return intersection / union if union else 0.0


class Cosmos3DefectDetectionAblationContractTests(unittest.TestCase):
    def test_hybrid_single_calibration_caps_do_not_cap_task_strict_rows(self) -> None:
        rows: list[dict] = []
        candidates: list[dict] = []
        one_box = [{"bbox_2d": [10, 10, 100, 100], "label": "open"}]
        for bucket, boxes, evidence in (
            ("empty", [], ["calibration_empty_ground_truth"]),
            ("few", one_box, ["calibration_few_box_ground_truth"]),
        ):
            for index in range(4):
                row = _row(f"cal-{bucket}-{index}", "Defect Detection", boxes=boxes)
                rows.append(row)
                candidates.append(
                    _candidate(
                        row,
                        evidence=evidence,
                        phash=f"{100 + len(candidates):016x}",
                        route_tier="calibration",
                    )
                )
        for bucket, boxes, evidence in (
            ("empty", [], ["hard_negative_proxy_false_positive"]),
            ("positive", one_box, ["hard_positive_proxy_false_negative"]),
        ):
            for index in range(20):
                row = _row(f"strict-{bucket}-{index}", "Defect Detection", boxes=boxes)
                rows.append(row)
                candidates.append(
                    _candidate(
                        row,
                        evidence=evidence,
                        phash=f"{100 + len(candidates):016x}",
                    )
                )
        for index in range(30):
            task = defect_detection_ablation.MAINTENANCE_TASK_TYPES[index % 5]
            boxes = (
                []
                if task == "Ref_based Defect Detection" and (index // 5) < 3
                else one_box
            )
            row = _row(f"maintenance-{index}", task, boxes=boxes)
            rows.append(row)
            candidates.append(
                _candidate(row, phash=f"{100 + len(candidates):016x}")
            )

        selected, manifest = defect_detection_ablation.materialize(
            candidate_rows=candidates,
            source_records=rows,
            validation_records=[],
            media_root=pathlib.Path("/data"),
            max_rows=60,
            row_multiple=10,
            defect_detection_fraction=0.5,
            proxy_empty_rate=0.9,
            reference_proxy_empty_rate=0.5,
            single_image_calibration_max_empty=2,
            single_image_calibration_max_few=2,
            reference_calibration_total=0,
            epochs=1,
            global_batch=10,
            near_duplicate_hamming_distance=None,
        )

        self.assertEqual(len(selected), 60)
        self.assertTrue(manifest["verified"])
        calibration = manifest["single_image_calibration"]
        self.assertEqual(calibration["selected_empty"], 2)
        self.assertEqual(calibration["selected_few_box"], 2)
        self.assertEqual(manifest["row_counts"]["task_strict_defect_detection"], 26)
        self.assertFalse(
            manifest["empty_ground_truth_by_cohort"]["non_reference_based"][
                "proxy_empty_rate_binding"
            ]
        )

    def test_reference_leakage_uses_ordered_pair_content_not_only_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            for name, color in (
                ("validation-golden.png", (10, 20, 30)),
                ("candidate-golden.png", (10, 20, 30)),
                ("validation-target.png", (40, 50, 60)),
                ("candidate-target.png", (40, 50, 60)),
            ):
                Image.new("RGB", (8, 8), color=color).save(root / name)
            validation = _row(
                "validation-pair", "Ref_based Defect Detection", boxes=[]
            )
            images = [
                item
                for item in validation["messages"][0]["content"]
                if item.get("type") == "image"
            ]
            images[0]["image"] = "validation-golden.png"
            images[1]["image"] = "validation-target.png"
            _, validation_content, _, _ = (
                defect_detection_ablation._validation_identities([validation], root)
            )
            _, candidate_content, _ = (
                defect_detection_ablation._candidate_visual_identity(
                    {
                        "filepath": "candidate-target.png",
                        "sample_kind": "reference_pair",
                        "source_image_paths": [
                            str(root / "candidate-golden.png"),
                            str(root / "candidate-target.png"),
                        ],
                    },
                    media_root=root,
                    compute_perceptual_hash=False,
                )
            )
            self.assertIn(candidate_content, validation_content)

    def test_disabled_near_duplicate_filter_skips_perceptual_decode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image_path = pathlib.Path(temporary) / "candidate.png"
            Image.new("RGB", (8, 8), color=(12, 34, 56)).save(image_path)
            with mock.patch.object(
                defect_detection_ablation,
                "_perceptual_hash",
                side_effect=AssertionError("perceptual hash must not be decoded"),
            ):
                identity, content_sha, phash = (
                    defect_detection_ablation._candidate_visual_identity(
                        {"filepath": str(image_path)},
                        media_root=pathlib.Path(temporary),
                        compute_perceptual_hash=False,
                    )
                )
            self.assertEqual(pathlib.Path(identity).name, image_path.name)
            self.assertEqual(phash, content_sha[:16])

    def test_materialization_matches_empty_rate_in_both_detection_cohorts(self) -> None:
        rows: list[dict] = []
        candidates: list[dict] = []
        one_box = [{"bbox_2d": [10, 10, 100, 100], "label": "open"}]
        for index in range(30):
            empty = index < 12
            row = _row(
                f"single-dd-{index}",
                "Defect Detection",
                boxes=[] if empty else one_box,
            )
            rows.append(row)
            candidates.append(
                _candidate(
                    row,
                    evidence=(
                        ["hard_negative_proxy_false_positive"]
                        if empty
                        else ["hard_positive_proxy_false_negative"]
                    ),
                    phash=f"{index + 1000:016x}",
                )
            )
        maintenance_tasks = defect_detection_ablation.MAINTENANCE_TASK_TYPES
        offset = 2000
        for task in maintenance_tasks:
            for index in range(6):
                is_reference_detection = task == "Ref_based Defect Detection"
                empty = is_reference_detection and index < 3
                row = _row(
                    f"{task.replace(' ', '-')}-{index}",
                    task,
                    boxes=[] if empty else one_box,
                )
                rows.append(row)
                candidates.append(
                    _candidate(
                        row,
                        evidence=(
                            [
                                "calibration_empty_ground_truth",
                                "calibration_reference_no_change_ground_truth",
                            ]
                            if empty
                            else []
                        ),
                        phash=f"{offset:016x}",
                        route_tier="calibration" if empty else "strict",
                    )
                )
                offset += 1

        selected, manifest = defect_detection_ablation.materialize(
            candidate_rows=candidates,
            source_records=rows,
            validation_records=[],
            media_root=pathlib.Path("/data"),
            max_rows=60,
            row_multiple=10,
            defect_detection_fraction=0.5,
            proxy_empty_rate=0.4,
            reference_proxy_empty_rate=0.5,
            epochs=1,
            global_batch=10,
            near_duplicate_hamming_distance=None,
        )

        self.assertEqual(len(selected), 60)
        self.assertTrue(manifest["verified"])
        cohorts = manifest["empty_ground_truth_by_cohort"]
        self.assertEqual(cohorts["non_reference_based"]["selected_empty"], 12)
        self.assertEqual(cohorts["non_reference_based"]["selected_total"], 30)
        self.assertEqual(cohorts["reference_based"]["selected_empty"], 3)
        self.assertEqual(cohorts["reference_based"]["selected_total"], 6)
        self.assertEqual(cohorts["reference_based"]["proxy_rate"], 0.5)
        self.assertEqual(manifest["configuration"]["near_duplicate_filter"], "disabled")

    def test_all_proxy_dd_anchors_are_ordered_by_error_severity(self) -> None:
        selected = [{"id": "maintenance", "task_type": "Component Detection"}]
        all_gaps = [
            {
                "id": "correct",
                "evaluation_role": "proxy",
                "task_type": "Defect Detection",
                "defect_detection_evidence": {
                    "false_negative_count": 0,
                    "best_overlap_0_lt_iou_lte_0p5_count": 0,
                    "false_positive_count": 0,
                    "evidence_types": [],
                },
            },
            {
                "id": "false-positive",
                "evaluation_role": "proxy",
                "task_type": "Defect Detection",
                "defect_detection_evidence": {
                    "false_negative_count": 0,
                    "best_overlap_0_lt_iou_lte_0p5_count": 0,
                    "false_positive_count": 2,
                    "evidence_types": ["hard_negative_proxy_false_positive"],
                },
            },
            {
                "id": "partial-overlap",
                "evaluation_role": "proxy",
                "task_type": "Defect Detection",
                "defect_detection_evidence": {
                    "false_negative_count": 0,
                    "best_overlap_0_lt_iou_lte_0p5_count": 1,
                    "false_positive_count": 0,
                    "evidence_types": [
                        "hard_positive_best_overlap_0_lt_iou_lte_0p5"
                    ],
                },
            },
            {
                "id": "false-negative",
                "evaluation_role": "proxy",
                "task_type": "Defect Detection",
                "defect_detection_evidence": {
                    "false_negative_count": 3,
                    "best_overlap_0_lt_iou_lte_0p5_count": 0,
                    "false_positive_count": 0,
                    "evidence_types": ["hard_positive_proxy_false_negative"],
                },
            },
        ]

        augmented, summary = route_selected_gaps.augment_defect_detection_targets(
            selected,
            all_gaps,
            anchor_policy="all_proxy_severity",
        )

        self.assertEqual(
            [row["id"] for row in augmented],
            [
                "false-negative",
                "partial-overlap",
                "false-positive",
                "correct",
                "maintenance",
            ],
        )
        self.assertEqual(summary["anchor_policy"], "all_proxy_severity")
        self.assertEqual(summary["eligible_defect_detection_rows"], 4)
        self.assertEqual(
            augmented[3]["defect_detection_evidence"]["evidence_types"],
            ["proxy_correct"],
        )
        self.assertEqual(
            summary["severity_counts"],
            {
                "correct": 1,
                "false_negative_or_partial_overlap": 2,
                "false_positive": 1,
            },
        )

    def test_dd_supplement_preserves_selected_rows_and_adds_approved_evidence(self) -> None:
        selected = [
            {"id": "maintenance", "task_type": "Component Detection"},
            {"id": "positive", "task_type": "Defect Detection"},
        ]
        all_gaps = [
            {
                "id": "positive",
                "evaluation_role": "proxy",
                "task_type": "Defect Detection",
                "defect_detection_evidence": {
                    "evidence_types": ["hard_positive_proxy_false_negative"]
                },
            },
            {
                "id": "negative",
                "evaluation_role": "proxy",
                "task_type": "Defect Detection",
                "defect_detection_evidence": {
                    "evidence_types": ["hard_negative_proxy_false_positive"]
                },
            },
            {
                "id": "easy-dd",
                "evaluation_role": "proxy",
                "task_type": "Defect Detection",
                "defect_detection_evidence": {"evidence_types": []},
            },
            {
                "id": "off-task",
                "evaluation_role": "proxy",
                "task_type": "Defect Classification",
            },
        ]

        augmented, summary = route_selected_gaps.augment_defect_detection_targets(
            selected, all_gaps
        )

        self.assertEqual([row["id"] for row in augmented], ["maintenance", "positive", "negative"])
        self.assertEqual(summary["selected_input_rows"], 2)
        self.assertEqual(summary["eligible_defect_detection_rows"], 2)
        self.assertEqual(summary["supplemental_rows"], 1)
        self.assertEqual(summary["output_rows"], 3)

    def test_proxy_detection_evidence_separates_fn_partial_overlap_and_fp(self) -> None:
        evidence = analyze_gaps._detection_evidence(
            _Evaluator(),
            [(0.0, 0.0, 10.0, 10.0)],
            [(5.0, 0.0, 15.0, 10.0)],
            parse_ok=True,
            threshold=0.5,
        )

        self.assertEqual(evidence["false_negative_count"], 1)
        self.assertEqual(evidence["false_positive_count"], 1)
        self.assertEqual(evidence["best_overlap_0_lt_iou_lte_0p5_count"], 1)
        self.assertEqual(
            evidence["evidence_types"],
            [
                "hard_negative_proxy_false_positive",
                "hard_positive_best_overlap_0_lt_iou_lte_0p5",
                "hard_positive_proxy_false_negative",
            ],
        )

    def test_task_strict_router_applies_per_task_top_k(self) -> None:
        dd_records = [
            _row(f"dd-{index}", "Defect Detection", boxes=[])
            for index in range(3)
        ]
        maintenance_records = [
            _row(f"cc-{index}", "Component Classification")
            for index in range(3)
        ]
        source_records = dd_records + maintenance_records
        source_rows = [
            {
                "filepath": record["messages"][0]["content"][0]["image"],
                "embedding": [1.0, index / 100.0],
            }
            for index, record in enumerate(source_records)
        ]
        target_rows = [
            {
                "filepath": "images/proxy.png",
                "target_id": "proxy",
                "task_types": ["Defect Detection", "Component Classification"],
                "defect_detection_evidence": [],
                "embedding": [1.0, 0.0],
            }
        ]

        selected, summary = task_mining_router.route_candidates(
            target_rows,
            source_rows,
            source_records,
            media_root=pathlib.Path("/data"),
            mode="task_strict",
            top_k_per_target=1,
            top_k_by_task={"Defect Detection": 2},
            min_similarity=-1.0,
        )

        self.assertEqual(len(selected), 3)
        routes = summary["targets"][0]["task_routes"]
        self.assertEqual(routes["Defect Detection"]["top_k"], 2)
        self.assertEqual(routes["Defect Detection"]["selected"], 2)
        self.assertEqual(routes["Component Classification"]["top_k"], 1)
        self.assertEqual(routes["Component Classification"]["selected"], 1)
        self.assertEqual(summary["top_k_by_task"], {"Defect Detection": 2})

    def test_materialization_reserves_defect_detection_and_matches_empty_rate(self) -> None:
        rows: list[dict] = []
        candidates: list[dict] = []
        positive_evidence = ["hard_positive_proxy_false_negative"]
        negative_evidence = ["hard_negative_proxy_false_positive"]
        phenotypes = ("open", "short")
        for index in range(8):
            boxes = [
                {
                    "bbox_2d": [10 + index, 20, 110 + index * 10, 120],
                    "label": phenotypes[index % 2],
                }
            ]
            row = _row(
                f"dd-pos-{index}",
                "Defect Detection",
                boxes=boxes,
                dataset=f"source-{index % 2}",
            )
            rows.append(row)
            candidates.append(_candidate(row, evidence=positive_evidence, phash=f"{index + 1:016x}"))
        for index in range(4):
            row = _row(f"dd-empty-{index}", "Defect Detection", boxes=[])
            rows.append(row)
            candidates.append(
                _candidate(
                    row,
                    evidence=(
                        negative_evidence
                        if index < 3
                        else ["calibration_empty_ground_truth"]
                    ),
                    phash=f"{index + 20:016x}",
                    route_tier="strict" if index < 3 else "calibration",
                )
            )

        maintenance_tasks = (
            "Component Classification",
            "Component Detection",
            "Defect Classification",
            "Ref_based Defect Classification",
            "Ref_based Defect Detection",
        )
        for index in range(12):
            row = _row(f"maint-{index}", maintenance_tasks[index % 5])
            rows.append(row)
            candidates.append(_candidate(row, phash=f"{index + 40:016x}"))

        selected, manifest = defect_detection_ablation.materialize(
            candidate_rows=candidates,
            source_records=rows,
            validation_records=[],
            media_root=pathlib.Path("/data"),
            max_rows=24,
            row_multiple=6,
            defect_detection_fraction=0.5,
            proxy_empty_rate=1 / 3,
            epochs=5,
            global_batch=6,
            near_duplicate_hamming_distance=0,
        )

        counts = manifest["row_counts"]
        self.assertTrue(manifest["verified"])
        self.assertEqual(counts["total"], 24)
        self.assertEqual(counts["defect_detection"], 12)
        self.assertEqual(counts["maintenance"], 12)
        self.assertEqual(counts["novel_mining_pool_images"], 24)
        self.assertEqual(counts["replayed_mining_pool_images"], 0)
        self.assertEqual(counts["by_task"]["Component Detection"], 3)
        self.assertEqual(manifest["empty_ground_truth"]["selected_empty"], 4)
        self.assertEqual(manifest["empty_ground_truth"]["selected_non_empty"], 8)
        self.assertEqual(
            manifest["defect_detection_evidence"]["by_type"]["calibration_empty_ground_truth"],
            1,
        )
        self.assertAlmostEqual(manifest["empty_ground_truth"]["selected_rate"], 1 / 3)
        self.assertEqual(manifest["optimizer_schedule"]["expected_optimizer_steps"], 20)
        self.assertEqual(len({row["id"] for row in selected}), 24)

    def test_replay_fills_target_without_counting_as_novel_pool_use(self) -> None:
        rows: list[dict] = []
        candidates: list[dict] = []
        for index in range(5):
            row = _row(
                f"positive-{index}",
                "Defect Detection",
                boxes=[{"bbox_2d": [10, 10, 100, 100], "label": "open"}],
            )
            rows.append(row)
            candidates.append(
                _candidate(
                    row,
                    evidence=["hard_positive_proxy_false_negative"],
                    phash=f"{index + 300:016x}",
                    is_replay=index >= 1,
                )
            )
        for index, task in enumerate(
            (
                "Component Classification",
                "Component Detection",
                "Defect Classification",
                "Ref_based Defect Classification",
                "Ref_based Defect Detection",
            )
        ):
            row = _row(f"maintenance-{index}", task)
            rows.append(row)
            candidates.append(
                _candidate(
                    row,
                    phash=f"{index + 400:016x}",
                    is_replay=index >= 1,
                )
            )

        _, manifest = defect_detection_ablation.materialize(
            candidate_rows=candidates,
            source_records=rows,
            validation_records=[],
            media_root=pathlib.Path("/data"),
            max_rows=10,
            row_multiple=2,
            defect_detection_fraction=0.5,
            proxy_empty_rate=0.0,
            epochs=1,
            global_batch=2,
            near_duplicate_hamming_distance=0,
        )

        self.assertTrue(manifest["verified"])
        self.assertEqual(manifest["row_counts"]["novel_mining_pool_images"], 2)
        self.assertEqual(manifest["row_counts"]["replayed_mining_pool_images"], 8)

    def test_manifest_reports_stratum_shortage_without_off_task_backfill(self) -> None:
        rows: list[dict] = []
        candidates: list[dict] = []
        for index in range(8):
            row = _row(
                f"positive-{index}",
                "Defect Detection",
                boxes=[{"bbox_2d": [10, 10, 100, 100], "label": "rare" if index == 0 else "common"}],
            )
            rows.append(row)
            candidates.append(
                _candidate(
                    row,
                    evidence=["hard_positive_proxy_false_negative"],
                    phash=f"{index + 100:016x}",
                )
            )
        for index in range(8):
            row = _row(f"maintenance-{index}", "Component Detection")
            rows.append(row)
            candidates.append(_candidate(row, phash=f"{index + 200:016x}"))

        _, manifest = defect_detection_ablation.materialize(
            candidate_rows=candidates,
            source_records=rows,
            validation_records=[],
            media_root=pathlib.Path("/data"),
            max_rows=16,
            row_multiple=4,
            defect_detection_fraction=0.5,
            proxy_empty_rate=0.0,
            epochs=1,
            global_batch=4,
            near_duplicate_hamming_distance=0,
        )

        shortages = manifest["positive_marginal_quotas"]["phenotype"]["shortages"]
        self.assertEqual(shortages["rare"], 3)
        self.assertEqual(manifest["row_counts"]["defect_detection"], 8)
        self.assertNotIn("Defect Detection", manifest["maintenance_task_types"])

    def test_batch_aligned_shortfall_is_accepted_above_configured_minimum(self) -> None:
        rows: list[dict] = []
        candidates: list[dict] = []
        for index in range(10):
            row = _row(
                f"positive-shortfall-{index}",
                "Defect Detection",
                boxes=[{"bbox_2d": [10, 10, 100, 100], "label": "open"}],
            )
            rows.append(row)
            candidates.append(
                _candidate(
                    row,
                    evidence=["hard_positive_proxy_false_negative"],
                    phash=f"{index + 500:016x}",
                )
            )
        for index in range(10):
            row = _row(
                f"maintenance-shortfall-{index}",
                defect_detection_ablation.MAINTENANCE_TASK_TYPES[index % 5],
            )
            rows.append(row)
            candidates.append(_candidate(row, phash=f"{index + 600:016x}"))

        selected, manifest = defect_detection_ablation.materialize(
            candidate_rows=candidates,
            source_records=rows,
            validation_records=[],
            media_root=pathlib.Path("/data"),
            max_rows=30,
            minimum_rows=17,
            row_multiple=10,
            defect_detection_fraction=0.5,
            proxy_empty_rate=0.0,
            epochs=2,
            global_batch=10,
            near_duplicate_hamming_distance=0,
        )

        self.assertTrue(manifest["verified"])
        self.assertEqual(len(selected), 20)
        self.assertEqual(manifest["row_counts"]["total"], 20)
        self.assertEqual(manifest["row_counts"]["defect_detection"], 10)
        self.assertEqual(manifest["row_counts"]["maintenance"], 10)
        self.assertEqual(manifest["shortfall"]["requested_rows"], 30)
        self.assertEqual(manifest["shortfall"]["accepted_rows"], 20)
        self.assertTrue(manifest["shortfall"]["accepted"])
        self.assertEqual(manifest["configuration"]["minimum_rows_requested"], 17)
        self.assertEqual(manifest["configuration"]["minimum_rows_batch_aligned"], 20)
        self.assertEqual(manifest["optimizer_schedule"]["expected_optimizer_steps"], 4)

    def test_verified_manifest_is_bound_to_training_jsonl_and_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            training = root / "train.jsonl"
            training.write_text(json.dumps(_row("one", "Defect Detection")) + "\n")
            manifest = defect_detection_ablation.bind_manifest(
                {
                    "schema_version": "defect_detection_quota_manifest_v1",
                    "verified": True,
                    "row_counts": {"total": 1, "defect_detection": 1},
                    "optimizer_schedule": {
                        "epochs": 5,
                        "global_batch": 1,
                        "expected_optimizer_steps": 5,
                    },
                },
                training,
            )
            path = root / "quota_manifest.json"
            path.write_text(json.dumps(manifest))

            defect_detection_ablation.verify_bound_manifest(
                path,
                training_jsonl=training,
                expected_rows=1,
                epochs=5,
                global_batch=1,
            )
            training.write_text(training.read_text() + "{}\n")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                defect_detection_ablation.verify_bound_manifest(
                    path,
                    training_jsonl=training,
                    expected_rows=1,
                    epochs=5,
                    global_batch=1,
                )


if __name__ == "__main__":
    unittest.main()
