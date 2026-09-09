# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import pathlib
import shutil
import sys
import tempfile
import unittest

from PIL import Image


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import atomic_samples  # noqa: E402
import defect_detection_ablation as ablation  # noqa: E402
import select_detection_calibration as calibration  # noqa: E402


def _row(record_id: str, paths: list[str], *, empty: bool, task: str = "Ref_based Defect Detection") -> dict:
    return {
        "id": record_id,
        "task_type": task,
        "messages": [
            {"role": "user", "content": [
                *[{"type": "image", "image": path} for path in paths],
                {"type": "text", "text": "detect"},
            ]},
            {"role": "assistant", "content": json.dumps(
                [] if empty else [{"bbox_2d": [0, 0, 1, 1], "label": "open"}]
            )},
        ],
    }


def _pool(root: pathlib.Path, *, changed: int = 288) -> list[dict]:
    images = root / "images"
    images.mkdir()
    Image.new("RGB", (2, 2), color=(0, 0, 0)).save(images / "golden.png")
    records = []
    for index in range(212 + changed):
        target = f"target-{index}.png"
        Image.new("RGB", (2, 2), color=(index % 256, index // 256, 123)).save(images / target)
        records.append(_row(
            f"reference-{index}", ["images/golden.png", f"images/{target}"],
            empty=index < 212,
        ))
    # Distinct IDs and paths, identical ordered golden/test constituent bytes.
    shutil.copyfile(images / "golden.png", images / "golden-copy.png")
    shutil.copyfile(images / "target-212.png", images / "target-copy.png")
    records.insert(213, _row(
        "duplicate-changed", ["images/golden-copy.png", "images/target-copy.png"],
        empty=False,
    ))
    return records


def _select(root: pathlib.Path, records: list[dict]) -> tuple[list[dict], dict]:
    return calibration.select_calibration(
        iter(records), media_root=root, pair_assets_dir=root / "pair-assets",
        cohort_quotas={"non_reference_based": 0, "reference_based": 500},
        cohort_rates={
            "non_reference_based": {"empty_rate": 0.418},
            "reference_based": {"empty_rate": 0.424},
        },
    )


def _materialize(root: pathlib.Path, source: list[dict], selected: list[dict]) -> tuple[list[dict], dict]:
    others = [
        _row(f"dd-{index}", [f"images/dd-{index}.png"], empty=True, task="Defect Detection")
        for index in range(520)
    ]
    Image.new("RGB", (2, 2), color=(1, 20, 30)).save(root / "images/maintenance.png")
    for index, task in enumerate(ablation.MAINTENANCE_TASK_TYPES):
        if task == "Ref_based Defect Detection":
            continue
        paths = [f"images/maintenance-{index}.png"]
        if task.startswith("Ref_based"):
            paths = ["images/golden.png", "images/maintenance.png"]
        others.append(_row(f"maintenance-{index}", paths, empty=True, task=task))
    candidates = list(selected)
    for record in others:
        sample = atomic_samples.sample_from_record(record, media_root=root, context=record["id"])
        candidates.append({
            "filepath": sample["target_filepath"],
            "atomic_sample_id": sample["atomic_sample_id"],
            "sample_kind": sample["sample_kind"],
            "source_image_paths": sample["image_paths"],
            "routed_task_types": [record["task_type"]], "route_tier": "strict",
            "defect_detection_evidence": ["hard_negative_proxy_false_positive"],
        })
    return ablation.materialize(
        candidate_rows=candidates, source_records=[*source, *others],
        validation_records=[], media_root=root, max_rows=1024, row_multiple=128,
        defect_detection_fraction=0.5, proxy_empty_rate=0.418,
        reference_proxy_empty_rate=0.424, single_image_calibration_max_empty=0,
        single_image_calibration_max_few=0, reference_calibration_total=500,
        epochs=5, global_batch=128, near_duplicate_hamming_distance=None,
    )


class Cosmos3ReferenceCalibrationContentTests(unittest.TestCase):
    def test_duplicate_contents_are_skipped_until_500_unique_pairs_fill_both_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            records = _pool(root)
            selected, summary = _select(root, records)
            contents = [
                atomic_samples.content_identity_for_paths("reference_pair", row["source_image_paths"])
                for row in selected
            ]
            self.assertEqual(len(selected), 500)
            self.assertEqual(len(set(contents)), 500)
            self.assertEqual(sum(row["calibration_box_count"] == 0 for row in selected), 212)
            self.assertEqual(sum(row["calibration_box_count"] > 0 for row in selected), 288)
            self.assertEqual([row.get("content_sha256") for row in selected], contents)
            self.assertNotIn("duplicate-changed", {row["calibration_record_id"] for row in selected})
            self.assertIn("reference-499", {row["calibration_record_id"] for row in selected})
            self.assertEqual(summary["cohorts"]["reference_based"]["selected_total"], 500)
            self.assertEqual(summary["excluded_duplicate_reference_content"], 1)

    def test_exhausted_content_unique_pool_names_the_changed_bucket_shortfall(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            records = _pool(root, changed=287)
            with self.assertRaisesRegex(
                ValueError,
                "reference calibration content-unique shortfall:.*"
                "changed required=288 available=287 shortfall=1",
            ):
                _select(root, records)

    def test_selector_and_materializer_report_the_same_ordered_content_identities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = _pool(root)
            selected, summary = _select(root, source)
            rows, manifest = _materialize(root, source, selected)
            self.assertTrue(manifest["verified"], manifest["verification"])
            reference = manifest["reference_calibration"]
            self.assertEqual(reference["selected_total"], 500)
            self.assertEqual(reference["selected_no_change"], 212)
            self.assertEqual(reference["selected_changed"], 288)
            self.assertEqual(reference.get("content_sha256_by_record_id"), {
                row["calibration_record_id"]: row["content_sha256"] for row in selected
            })
            self.assertEqual(reference["content_identity"], summary["reference_content_identity"])
            by_id = {row["id"]: row for row in source}
            for row in rows:
                if row["task_type"] == "Ref_based Defect Detection":
                    self.assertEqual(row, by_id[row["id"]])

    def test_gate_rejects_duplicate_content_evidence_even_if_manifest_claims_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = _pool(root)
            selected, _ = _select(root, source)
            rows, manifest = _materialize(root, source, selected)
            self.assertTrue(manifest["verified"], manifest["verification"])
            training = root / "train.jsonl"
            training.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            manifest = ablation.bind_manifest(manifest, training)
            reference = manifest["reference_calibration"]
            reference["content_identity"] = atomic_samples.PAIR_CONTENT_IDENTITY
            reference["content_sha256_by_record_id"] = {
                row["calibration_record_id"]: row["content_sha256"] for row in selected
            }
            manifest["configuration"]["media_root"] = str(root)
            quota = root / "quota.json"
            quota.write_text(json.dumps(manifest), encoding="utf-8")
            ablation.verify_bound_manifest(
                quota, training_jsonl=training, expected_rows=1024, epochs=5, global_batch=128,
            )
            identities = reference["content_sha256_by_record_id"]
            keys = list(identities)
            # A duplicated digest must fail even when all claimed counts and
            # verified flags are intact; a forged but unique digest must also
            # fail against the original ordered image bytes.
            for bad_digest in (identities[keys[-2]], "f" * 64):
                with self.subTest(bad_digest=bad_digest):
                    identities[keys[-1]] = bad_digest
                    quota.write_text(json.dumps(manifest), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "reference calibration.*content"):
                        ablation.verify_bound_manifest(
                            quota, training_jsonl=training, expected_rows=1024,
                            epochs=5, global_batch=128,
                        )


if __name__ == "__main__":
    unittest.main()
