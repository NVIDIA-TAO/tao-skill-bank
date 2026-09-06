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

import analyze_gaps  # noqa: E402
import defect_detection_ablation  # noqa: E402


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


def _candidate(record: dict, *, evidence: list[str] | None = None, phash: str) -> dict:
    image = [
        item["image"]
        for item in record["messages"][0]["content"]
        if item.get("type") == "image"
    ][-1]
    return {
        "filepath": image,
        "route_tier": "strict",
        "routed_task_types": [record["task_type"]],
        "defect_detection_evidence": evidence or [],
        "perceptual_hash": phash,
        "local_contrast": (int(phash, 16) % 100) / 100.0,
        "max_cosine_similarity": 0.99,
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
            candidates.append(_candidate(row, evidence=negative_evidence, phash=f"{index + 20:016x}"))

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
        self.assertEqual(counts["by_task"]["Component Detection"], 3)
        self.assertEqual(manifest["empty_ground_truth"]["selected_empty"], 4)
        self.assertEqual(manifest["empty_ground_truth"]["selected_non_empty"], 8)
        self.assertAlmostEqual(manifest["empty_ground_truth"]["selected_rate"], 1 / 3)
        self.assertEqual(manifest["optimizer_schedule"]["expected_optimizer_steps"], 20)
        self.assertEqual(len({row["id"] for row in selected}), 24)

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
