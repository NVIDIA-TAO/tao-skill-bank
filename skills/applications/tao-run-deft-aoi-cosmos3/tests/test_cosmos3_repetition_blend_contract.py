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

import atomic_samples  # noqa: E402
import defect_detection_ablation  # noqa: E402
import validate_split_contract  # noqa: E402
import validate_sharegpt  # noqa: E402


def _row(record_id: str, task_type: str, *, empty: bool = False) -> dict:
    answer: object = [] if empty else [{"bbox_2d": [10, 10, 100, 100], "label": "open"}]
    if "Detection" not in task_type:
        answer = "A"
    images = [
        {
            "type": "image",
            "image": f"images/{record_id}.png",
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
        "dataset": "synthetic",
        "task_type": task_type,
        "messages": [
            {
                "role": "user",
                "content": [*images, {"type": "text", "text": "inspect"}],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "```json\n" + json.dumps(answer) + "\n```"
                            if isinstance(answer, list)
                            else answer
                        ),
                    }
                ],
            },
        ],
    }


def _candidate(record: dict, index: int) -> dict:
    target = [
        item["image"]
        for item in record["messages"][0]["content"]
        if item.get("type") == "image"
    ][-1]
    evidence = []
    route_tier = "strict"
    if record["task_type"] == "Defect Detection":
        if "[]" in record["messages"][1]["content"][0]["text"]:
            evidence = ["calibration_empty_ground_truth"]
            route_tier = "calibration"
        else:
            evidence = ["hard_positive_proxy_false_negative"]
    sample = atomic_samples.sample_from_record(
        record, media_root=pathlib.Path("/data"), context=str(record.get("id"))
    )
    return {
        "filepath": target,
        "atomic_sample_id": sample["atomic_sample_id"],
        "sample_kind": sample["sample_kind"],
        "source_image_paths": sample["image_paths"],
        "route_tier": route_tier,
        "routed_task_types": [record["task_type"]],
        "defect_detection_evidence": evidence,
        "perceptual_hash": f"{index + 1:016x}",
        "local_contrast": (index + 1) / 100.0,
        "max_cosine_similarity": 0.99,
        "is_replay": False,
    }


class Cosmos3RepetitionBlendContractTests(unittest.TestCase):
    def test_deficit_plan_normalizes_weights_and_clamps_rep(self) -> None:
        plan = defect_detection_ablation.plan_repetition(
            available_rows={"Defect Detection": 2, "Component Detection": 8},
            deficit_weights={"Defect Detection": 3.0, "Component Detection": 1.0},
            row_cap=12,
            rep_min=0.5,
            rep_max=3.0,
        )

        self.assertEqual(plan["Defect Detection"]["target_share"], 0.75)
        self.assertEqual(plan["Defect Detection"]["target_rows"], 9.0)
        self.assertEqual(plan["Defect Detection"]["rep"], 3.0)
        self.assertEqual(plan["Component Detection"]["target_share"], 0.25)
        self.assertEqual(plan["Component Detection"]["rep"], 0.5)

    def test_gap_summary_deficits_are_support_weighted_with_equal_fallback(self) -> None:
        tasks = ["Defect Detection", "Component Detection"]
        weights, source = defect_detection_ablation.deficit_weights_from_gap_summary(
            {
                "per_group_mean_weakness": {
                    "task_type=Defect Detection|dataset=a": 1.0,
                    "task_type=Defect Detection|dataset=b": 0.5,
                    "task_type=Component Detection|dataset=a": 0.25,
                },
                "per_group_support": {
                    "task_type=Defect Detection|dataset=a": 1,
                    "task_type=Defect Detection|dataset=b": 3,
                    "task_type=Component Detection|dataset=a": 2,
                },
            },
            tasks,
        )

        self.assertEqual(source, "gap_summary")
        self.assertEqual(weights["Defect Detection"], 0.625)
        self.assertEqual(weights["Component Detection"], 0.25)
        self.assertEqual(
            defect_detection_ablation.deficit_weights_from_gap_summary(None, tasks),
            ({"Component Detection": 1.0, "Defect Detection": 1.0}, "equal_fallback"),
        )

    def test_fractional_explicit_rep_is_seeded_and_deterministic(self) -> None:
        rows = [_row(f"dd-{index}", "Defect Detection") for index in range(4)]
        config = {
            "enabled": True,
            "policy": "explicit",
            "rep_min": 0.5,
            "rep_max": 3.0,
            "never_repeat_empty_gt": True,
            "explicit_multipliers": {"Defect Detection": 1.5},
        }

        first, first_manifest = defect_detection_ablation.apply_repetition_blend(
            rows,
            row_cap=6,
            config=config,
            seed=23,
        )
        second, second_manifest = defect_detection_ablation.apply_repetition_blend(
            rows,
            row_cap=6,
            config=config,
            seed=23,
        )

        self.assertEqual(first, second)
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(len(first), 6)
        self.assertEqual(len({row["id"] for row in first}), 4)
        self.assertEqual(
            first_manifest["tasks"]["Defect Detection"]["repeated_rows"], 6
        )

    def test_empty_ground_truth_rows_stay_at_one_occurrence(self) -> None:
        rows = [
            _row("dd-empty", "Defect Detection", empty=True),
            _row("dd-positive-a", "Defect Detection"),
            _row("dd-positive-b", "Defect Detection"),
        ]
        output, manifest = defect_detection_ablation.apply_repetition_blend(
            rows,
            row_cap=6,
            config={
                "enabled": True,
                "policy": "explicit",
                "rep_min": 0.5,
                "rep_max": 3.0,
                "never_repeat_empty_gt": True,
                "explicit_multipliers": {"Defect Detection": 2.0},
            },
            seed=17,
        )

        self.assertEqual(sum(row["id"] == "dd-empty" for row in output), 1)
        task = manifest["tasks"]["Defect Detection"]
        self.assertEqual(task["empty_rows"], 1)
        self.assertEqual(task["repeated_empty_rows"], 0)
        self.assertEqual(task["repeated_rows"], 6)

    def test_toml_and_json_configs_are_equivalent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            toml_path = root / "repetition.toml"
            toml_path.write_text(
                """
[repetition_blend]
enabled = true
policy = "explicit"
rep_min = 0.5
rep_max = 2.5
never_repeat_empty_gt = true
row_cap = 12000

[repetition_blend.explicit_multipliers]
"Defect Detection" = 2.5
""".strip()
                + "\n",
                encoding="utf-8",
            )
            json_path = root / "repetition.json"
            json_path.write_text(
                json.dumps(
                    {
                        "repetition_blend": {
                            "enabled": True,
                            "policy": "explicit",
                            "rep_min": 0.5,
                            "rep_max": 2.5,
                            "never_repeat_empty_gt": True,
                            "explicit_multipliers": {"Defect Detection": 2.5},
                            "row_cap": 12000,
                        }
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                defect_detection_ablation.load_repetition_config(toml_path),
                defect_detection_ablation.load_repetition_config(json_path),
            )

    def test_tiny_materialization_rebalances_only_accepted_rows(self) -> None:
        task_types = [
            "Defect Detection",
            "Defect Detection",
            "Component Classification",
            "Component Classification",
            "Component Detection",
            "Component Detection",
            "Defect Classification",
            "Defect Classification",
            "Ref_based Defect Classification",
            "Ref_based Defect Detection",
        ]
        records = [_row(f"row-{index}", task) for index, task in enumerate(task_types)]
        candidates = [_candidate(record, index) for index, record in enumerate(records)]
        weights = {task: 1.0 for task in defect_detection_ablation.MAINTENANCE_TASK_TYPES}
        weights["Defect Detection"] = 5.0

        output, quota = defect_detection_ablation.materialize(
            candidate_rows=candidates,
            source_records=records,
            validation_records=[],
            media_root=pathlib.Path("/data"),
            max_rows=10,
            row_multiple=1,
            defect_detection_fraction=0.5,
            proxy_empty_rate=0.0,
            epochs=1,
            global_batch=1,
            near_duplicate_hamming_distance=0,
            repetition_config={
                "enabled": True,
                "policy": "deficit_proportional",
                "rep_min": 0.5,
                "rep_max": 3.0,
                "never_repeat_empty_gt": True,
                "explicit_multipliers": {},
            },
            deficit_weights=weights,
            repetition_seed=17,
        )

        self.assertTrue(quota["verified"])
        self.assertEqual(len(output), 10)
        self.assertEqual(sum(row["task_type"] == "Defect Detection" for row in output), 5)
        repetition = quota["repetition_blend"]
        self.assertEqual(repetition["schema_version"], "repetition_blend_manifest_v1")
        self.assertEqual(repetition["tasks"]["Defect Detection"]["available"], 2)
        self.assertEqual(repetition["tasks"]["Defect Detection"]["target_share"], 0.5)
        self.assertEqual(repetition["tasks"]["Defect Detection"]["rep"], 2.5)
        self.assertEqual(repetition["tasks"]["Defect Detection"]["repeated_rows"], 5)
        source_fingerprints = {
            json.dumps(record, sort_keys=True, separators=(",", ":"))
            for record in records
        }
        self.assertTrue(
            all(
                json.dumps(record, sort_keys=True, separators=(",", ":"))
                in source_fingerprints
                for record in output
            )
        )

    def test_training_validation_allows_only_exact_id_repetitions(self) -> None:
        row = _row("repeat-me", "Component Classification")
        summary = validate_sharegpt.validate_records(
            [row, json.loads(json.dumps(row))],
            media_root=pathlib.Path("/data"),
            require_files=False,
            allow_exact_repetitions=True,
        )

        self.assertEqual(summary["records"], 2)
        self.assertEqual(summary["unique_ids"], 1)
        self.assertEqual(summary["exact_repetitions"], 1)
        changed = json.loads(json.dumps(row))
        changed["messages"][1]["content"][0]["text"] = "B"
        with self.assertRaisesRegex(ValueError, "duplicate id.*different content"):
            validate_sharegpt.validate_records(
                [row, changed],
                media_root=pathlib.Path("/data"),
                require_files=False,
                allow_exact_repetitions=True,
            )

    def test_repetition_binding_remains_compatible_with_quota_gate(self) -> None:
        rows = [_row("dd-a", "Defect Detection"), _row("dd-b", "Defect Detection")]
        output, repetition = defect_detection_ablation.apply_repetition_blend(
            rows,
            row_cap=4,
            config={
                "enabled": True,
                "policy": "explicit",
                "rep_min": 0.5,
                "rep_max": 3.0,
                "never_repeat_empty_gt": True,
                "explicit_multipliers": {"Defect Detection": 2.0},
            },
            seed=17,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            training = root / "train.jsonl"
            training.write_text(
                "".join(json.dumps(row) + "\n" for row in output),
                encoding="utf-8",
            )
            repetition = defect_detection_ablation.bind_repetition_manifest(
                repetition, training
            )
            quota = defect_detection_ablation.bind_manifest(
                {
                    "schema_version": "defect_detection_quota_manifest_v1",
                    "verified": True,
                    "row_counts": {"total": 4, "defect_detection": 4},
                    "optimizer_schedule": {
                        "epochs": 2,
                        "global_batch": 2,
                        "expected_optimizer_steps": 4,
                    },
                    "repetition_blend": repetition,
                },
                training,
            )
            quota_path = root / "quota.json"
            quota_path.write_text(json.dumps(quota), encoding="utf-8")

            verified = defect_detection_ablation.verify_bound_manifest(
                quota_path,
                training_jsonl=training,
                expected_rows=4,
                epochs=2,
                global_batch=2,
            )
            self.assertEqual(
                verified["training_jsonl"]["sha256"],
                repetition["training_jsonl"]["sha256"],
            )

    def test_split_validation_accepts_repeated_training_rows_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)

            def write(name: str, rows: list[dict]) -> pathlib.Path:
                path = root / name
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                return path

            mining_row = _row("mining", "Defect Detection")
            roles = {
                "proxy": write("proxy.jsonl", [_row("proxy", "Defect Detection")]),
                "benchmark": write(
                    "benchmark.jsonl", [_row("benchmark", "Defect Detection")]
                ),
                "mining": write("mining.jsonl", [mining_row]),
                "train": write("train.jsonl", [mining_row, mining_row]),
            }

            summary = validate_split_contract.validate(
                roles,
                media_root=root,
            )
            self.assertEqual(summary["records"]["train"], 2)
            self.assertEqual(summary["unique_targets"]["train"], 1)


if __name__ == "__main__":
    unittest.main()
