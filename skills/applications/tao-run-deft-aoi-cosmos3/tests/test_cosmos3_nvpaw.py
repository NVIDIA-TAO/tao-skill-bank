#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import pathlib
import re
import sys
import tempfile
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
FIXTURES = pathlib.Path(__file__).parent / "fixtures"
sys.path.insert(0, str(SCRIPTS))

import nvpaw_annotations  # noqa: E402
import multitask_metrics  # noqa: E402
import analyze_gaps  # noqa: E402
import init_deft_state  # noqa: E402
import metric_contract  # noqa: E402
import assemble_training_json  # noqa: E402
import commit_stage  # noqa: E402
import emit_mined_sharegpt  # noqa: E402
import emit_sdg_sharegpt  # noqa: E402
import route_selected_gaps  # noqa: E402
import task_mining_router  # noqa: E402
import validate_sharegpt  # noqa: E402
import validate_split_contract  # noqa: E402


EXPECTED_TASKS = {
    "Component Classification",
    "Component Count",
    "Component Detection",
    "Defect Classification",
    "Defect Detection",
    "Ref_based Defect Classification",
    "Ref_based Defect Detection",
}


class NVPawAnnotationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = nvpaw_annotations.load_source_records(
            FIXTURES / "nvpaw_multitask.jsonl"
        )

    def test_materializes_all_tasks_with_explicit_image_roles(self) -> None:
        manifest, sharegpt = nvpaw_annotations.materialize_records(self.source)
        self.assertEqual(len(manifest), 9)
        self.assertEqual(len(sharegpt), 9)
        self.assertEqual({row["task_type"] for row in manifest}, EXPECTED_TASKS)
        self.assertEqual(set(nvpaw_annotations.TASK_SPECS), EXPECTED_TASKS)

        reference = next(
            row
            for row in manifest
            if row["task_type"] == "Ref_based Defect Detection"
        )
        self.assertEqual(
            [image["role"] for image in reference["images"]],
            ["golden", "target"],
        )
        self.assertEqual(reference["images"][-1]["path"], "images/board-6.png")
        self.assertEqual(reference["reference_cohort"], "golden_then_target")

        rendered = next(row for row in sharegpt if row["id"] == reference["id"])
        self.assertEqual(rendered["images"], ["images/golden-6.png", "images/board-6.png"])
        self.assertEqual(rendered["image_roles"], ["golden", "target"])

    def test_component_count_materializes_to_a_typed_nonnegative_integer(self) -> None:
        source = {
            "id": "board-count#component-count",
            "dataset": "fixture-count",
            "split": "proxy",
            "task_type": "Component Count",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "images/board-count.png"},
                        {
                            "type": "text",
                            "text": (
                                "How many components are visible in this PCBA image?\n"
                                "Answer with only the integer count."
                            ),
                        },
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "34"}]},
            ],
        }

        manifest, sharegpt = nvpaw_annotations.materialize_records([source])

        self.assertEqual(manifest[0]["metric_family"], "counting")
        self.assertEqual(manifest[0]["answer"], {"kind": "count", "value": 34})
        self.assertEqual(sharegpt[0]["answer"], {"kind": "count", "value": 34})

        zero_source = copy.deepcopy(source)
        zero_source["id"] = "board-empty#component-count"
        zero_source["messages"][-1]["content"][0]["text"] = "0"
        zero_manifest, _ = nvpaw_annotations.materialize_records([zero_source])
        self.assertEqual(zero_manifest[0]["answer"], {"kind": "count", "value": 0})
        self.assertEqual(sharegpt[0]["image_roles"], ["target"])

        for invalid in ("-1", "3 components", "1.5"):
            malformed = copy.deepcopy(source)
            malformed["id"] = f"bad-count-{invalid.replace(' ', '-')}"
            malformed["messages"][-1]["content"][0]["text"] = invalid
            with self.subTest(answer=invalid), self.assertRaisesRegex(
                ValueError, "count.*non-negative integer"
            ):
                nvpaw_annotations.materialize_records([malformed])

    def test_prompt_catalog_declares_each_supported_task_once(self) -> None:
        catalog = (SKILL_ROOT / "references" / "nvpaw-prompt-formats.md").read_text()
        declared = [
            line.split("|")[1].strip()
            for line in catalog.splitlines()
            if line.startswith("|")
            and any(f"| {task} |" in line for task in EXPECTED_TASKS)
        ]
        self.assertEqual(sorted(declared), sorted(EXPECTED_TASKS))

    def test_shared_target_ids_are_valid_but_record_ids_are_unique(self) -> None:
        manifest, _ = nvpaw_annotations.materialize_records(self.source)
        shared = [row for row in manifest if row["target_id"] == "images/board-1.png"]
        self.assertEqual(
            {row["source_id"] for row in shared},
            {"board-1#component", "board-1#defect"},
        )
        self.assertTrue(
            all(re.fullmatch(r"[A-Za-z0-9._@-]+", row["id"]) for row in manifest)
        )

        duplicate = [*self.source, copy.deepcopy(self.source[0])]
        with self.assertRaisesRegex(ValueError, "duplicate id.*board-1#component"):
            nvpaw_annotations.materialize_records(duplicate)

    def test_empty_answers_and_normalized_labeled_boxes_are_canonical(self) -> None:
        manifest, _ = nvpaw_annotations.materialize_records(self.source)
        empty_choice = next(row for row in manifest if row["source_id"] == "board-7#component-empty")
        empty_detection = next(row for row in manifest if row["source_id"] == "board-4#defect-detection")
        detection = next(row for row in manifest if row["source_id"] == "board-2#component-detection")
        self.assertEqual(empty_choice["answer"], {"kind": "choice_set", "labels": []})
        self.assertEqual(empty_detection["answer"], {"kind": "detections", "objects": []})
        self.assertEqual(
            detection["answer"]["objects"],
            [{"label": "resistor", "bbox_2d": [100, 200, 400, 500]}],
        )

        open_empty = copy.deepcopy(self.source[-1])
        open_empty["id"] = "open-empty"
        open_empty["messages"][1]["content"][-1]["text"] = "List visible component classes."
        open_empty["messages"][-1]["content"][0]["text"] = "[]"
        normalized, _ = nvpaw_annotations.materialize_records([open_empty])
        self.assertEqual(normalized[0]["answer"], {"kind": "choice_set", "labels": []})

        compact = copy.deepcopy(self.source[0])
        compact["id"] = "compact-choice-list"
        compact["messages"][-1]["content"][0]["text"] = "[A,B]"
        normalized, _ = nvpaw_annotations.materialize_records([compact])
        self.assertEqual(
            normalized[0]["answer"]["labels"], ["Resistors", "Capacitors"]
        )

    def test_invalid_detection_coordinates_fail_with_record_id(self) -> None:
        for bad_box in (
            [100.0, 200, 400, 500],
            [-1, 200, 400, 500],
            [100, 200, 100, 500],
            [100, 200, 400, 1001],
        ):
            source = copy.deepcopy(self.source)
            row = next(item for item in source if item["id"] == "board-2#component-detection")
            row["messages"][-1]["content"][0]["text"] = (
                '[{"bbox_2d":' + str(bad_box) + ',"label":"resistor"}]'
            )
            with self.subTest(box=bad_box), self.assertRaisesRegex(
                ValueError, "board-2#component-detection.*bbox_2d"
            ):
                nvpaw_annotations.materialize_records(source)

    def test_materialization_is_deterministic_and_self_contained(self) -> None:
        first = nvpaw_annotations.materialize_records(self.source)
        second = nvpaw_annotations.materialize_records(copy.deepcopy(self.source))
        self.assertEqual(first, second)
        self.assertNotIn("/lustre/", (FIXTURES / "nvpaw_multitask.jsonl").read_text())

    def test_rich_sharegpt_validation_allows_multi_prompt_targets(self) -> None:
        _, sharegpt = nvpaw_annotations.materialize_records(self.source)
        summary = validate_sharegpt.validate_records(
            sharegpt,
            media_root=pathlib.Path("/dataset"),
            require_files=False,
            require_id=True,
            annotation_profile="nvpaw_multitask_v1",
        )
        self.assertEqual(summary["mode"], "nvpaw_multitask_v1")
        self.assertEqual(summary["records"], 9)
        self.assertEqual(summary["unique_target_images"], 8)
        self.assertEqual(set(summary["tasks"]), EXPECTED_TASKS)

    def test_rich_split_contract_uses_target_role_and_allows_prompt_fanout(self) -> None:
        _, proxy = nvpaw_annotations.materialize_records(self.source)

        def relocated(rows: list[dict], prefix: str) -> list[dict]:
            result = copy.deepcopy(rows)
            for row in result:
                row["id"] = f"{prefix}-{row['id']}"
                row["target_id"] = f"{prefix}/{row['target_id']}"
                target_index = row["image_roles"].index("target")
                row["images"][target_index] = f"{prefix}/{row['images'][target_index]}"
            return result

        benchmark = relocated(proxy[:2], "benchmark")
        mining = relocated(proxy[2:4], "mining")
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)

            def write(name: str, rows: list[dict]) -> pathlib.Path:
                path = root / name
                path.write_text(json.dumps(rows) + "\n")
                return path

            paths = {
                "proxy": write("proxy.json", proxy),
                "benchmark": write("benchmark.json", benchmark),
                "mining": write("mining.json", mining),
            }
            summary = validate_split_contract.validate(
                paths,
                media_root=root,
                annotation_profile="nvpaw_multitask_v1",
            )
            self.assertEqual(summary["records"]["proxy"], 9)
            self.assertEqual(summary["unique_targets"]["proxy"], 8)

            leaked = copy.deepcopy(benchmark)
            leaked[0]["target_id"] = proxy[0]["target_id"]
            leaked_target_index = leaked[0]["image_roles"].index("target")
            proxy_target_index = proxy[0]["image_roles"].index("target")
            leaked[0]["images"][leaked_target_index] = proxy[0]["images"][proxy_target_index]
            paths["benchmark"] = write("benchmark.json", leaked)
            with self.assertRaisesRegex(ValueError, "target leakage"):
                validate_split_contract.validate(
                    paths,
                    media_root=root,
                    annotation_profile="nvpaw_multitask_v1",
                )


class NVPawMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        source = nvpaw_annotations.load_source_records(
            FIXTURES / "nvpaw_multitask.jsonl"
        )
        _, self.annotations = nvpaw_annotations.materialize_records(source)

    def perfect_samples(self) -> list[dict]:
        return [
            {
                "id": row["id"],
                "response": row["conversations"][-1]["value"],
            }
            for row in self.annotations
        ]

    def test_prompt_local_choice_parser_accepts_compact_and_prefixed_answers(self) -> None:
        options = {"A": "Resistors", "B": "Capacitors"}
        self.assertEqual(
            multitask_metrics.parse_classification_prediction(
                "[A,B]", option_map=options
            ),
            ({"Resistors", "Capacitors"}, True),
        )
        self.assertEqual(
            multitask_metrics.parse_classification_prediction(
                "B. Capacitors", option_map=options
            ),
            ({"Capacitors"}, True),
        )

    def test_component_count_uses_instance_count_f1_and_strict_integer_parsing(self) -> None:
        annotation = {
            "id": "count-eval",
            "source_id": "count-eval",
            "target_id": "images/count.png",
            "dataset": "fixture-count",
            "task_type": "Component Count",
            "metric_family": "counting",
            "reference_cohort": "single_target",
            "prompt_format": "nvpaw.component_count.single.official_v1",
            "prompt_variant": "official_v1",
            "image_roles": ["target"],
            "images": ["images/count.png"],
            "option_map": {},
            "answer": {"kind": "count", "value": 4},
            "conversations": [
                {"from": "human", "value": "How many components are visible?"},
                {"from": "gpt", "value": "4"},
            ],
        }

        result = multitask_metrics.evaluate(
            [{"id": "count-eval", "response": "6"}],
            [annotation],
            required_tasks={"Component Count"},
        )

        sample = result["sample_metrics"][0]
        self.assertEqual((sample["tp"], sample["fp"], sample["fn"]), (4, 2, 0))
        self.assertEqual(sample["ground_truth_count"], 4)
        self.assertEqual(sample["prediction_count"], 6)
        self.assertAlmostEqual(sample["sample_score"], 0.8)
        metrics = result["task_metrics"]["Component Count"]
        self.assertEqual(metrics["primary_metric"], "count_micro_f1")
        self.assertAlmostEqual(metrics["count_micro_f1"], 0.8)
        self.assertEqual(metrics["exact_count_accuracy"], 0.0)
        self.assertEqual(metrics["mean_absolute_error"], 2.0)

        zero_annotation = copy.deepcopy(annotation)
        zero_annotation["id"] = "count-empty"
        zero_annotation["answer"]["value"] = 0
        perfect_zero = multitask_metrics.evaluate(
            [{"id": "count-empty", "response": "0"}],
            [zero_annotation],
            required_tasks={"Component Count"},
        )
        self.assertEqual(perfect_zero["sample_metrics"][0]["sample_score"], 1.0)
        self.assertEqual(
            perfect_zero["task_metrics"]["Component Count"]["count_micro_f1"],
            1.0,
        )

        malformed = multitask_metrics.evaluate(
            [{"id": "count-eval", "response": "about 4"}],
            [annotation],
            required_tasks={"Component Count"},
        )
        self.assertEqual(malformed["coverage"]["parse_failures"], 1)
        self.assertEqual(malformed["sample_metrics"][0]["gap_type"], "parse_failure")

    def test_perfect_multitask_predictions_have_balanced_score_one(self) -> None:
        result = multitask_metrics.evaluate(
            self.perfect_samples(), self.annotations, evaluation_role="benchmark"
        )
        self.assertEqual(result["metric_result"]["name"], "balanced_score")
        self.assertEqual(result["metric_result"]["value"], 1.0)
        self.assertTrue(result["summary"]["kpi"]["met"])
        self.assertEqual(set(result["task_metrics"]), EXPECTED_TASKS)
        self.assertEqual(
            set(result["reference_cohort_metrics"]),
            {"single_target", "golden_then_target"},
        )
        self.assertEqual(
            set(result["metric_family_metrics"]),
            {"classification", "counting", "detection"},
        )
        self.assertEqual(
            result["coverage"],
            {
                "expected_predictions": 9,
                "received_prediction_rows": 9,
                "missing_predictions": 0,
                "duplicate_prediction_ids": 0,
                "unknown_prediction_ids": 0,
                "parse_failures": 0,
            },
        )
        by_source = {row["source_id"]: row for row in result["sample_metrics"]}
        self.assertEqual(by_source["board-7#component-empty"]["sample_score"], 1.0)
        self.assertEqual(by_source["board-4#defect-detection"]["sample_score"], 1.0)

    def test_detection_is_label_aware_and_iou_is_strict(self) -> None:
        annotations = [
            row
            for row in self.annotations
            if row["source_id"] == "board-2#component-detection"
        ]
        record_id = annotations[0]["id"]
        wrong_label = multitask_metrics.evaluate(
            [
                {
                    "id": record_id,
                    "response": '[{"bbox_2d":[100,200,400,500],"label":"capacitor"}]',
                }
            ],
            annotations,
            required_tasks={"Component Detection"},
        )
        sample = wrong_label["sample_metrics"][0]
        self.assertEqual((sample["tp"], sample["fp"], sample["fn"]), (0, 1, 1))
        self.assertEqual(sample["sample_score"], 0.0)

        exact_half = copy.deepcopy(annotations)
        exact_half[0]["answer"] = {
            "kind": "detections",
            "objects": [{"label": "resistor", "bbox_2d": [0, 0, 100, 100]}],
        }
        boundary = multitask_metrics.evaluate(
            [
                {
                    "id": record_id,
                    "response": '[{"bbox_2d":[0,0,50,100],"label":"resistor"}]',
                }
            ],
            exact_half,
            iou_threshold=0.5,
            required_tasks={"Component Detection"},
        )
        self.assertEqual(boundary["sample_metrics"][0]["tp"], 0)

    def test_missing_task_group_fails_and_strong_tasks_cannot_hide_weakest(self) -> None:
        partial_annotations = [
            row
            for row in self.annotations
            if row["task_type"] != "Ref_based Defect Detection"
        ]
        partial_ids = {row["id"] for row in partial_annotations}
        with self.assertRaisesRegex(ValueError, "missing required task groups"):
            multitask_metrics.evaluate(
                [row for row in self.perfect_samples() if row["id"] in partial_ids],
                partial_annotations,
            )

        predictions = self.perfect_samples()
        weak_id = next(
            row["id"]
            for row in self.annotations
            if row["task_type"] == "Component Detection"
        )
        next(row for row in predictions if row["id"] == weak_id)["response"] = "[]"
        result = multitask_metrics.evaluate(predictions, self.annotations)
        self.assertEqual(result["metric_result"]["value"], 0.0)
        self.assertFalse(result["summary"]["kpi"]["met"])
        self.assertGreater(result["metric_result"]["tie_breakers"]["macro_attainment"], 0.0)

    def test_coverage_and_parse_failures_block_balanced_kpi(self) -> None:
        samples = self.perfect_samples()
        missing = samples.pop()
        samples.append(copy.deepcopy(samples[0]))
        samples.append({"id": "unknown-eval-id", "response": "A"})
        detection_id = next(
            row["id"]
            for row in self.annotations
            if row["source_id"] == "board-2#component-detection"
        )
        for sample in samples:
            if sample["id"] == detection_id:
                sample["response"] = "not-json"
        result = multitask_metrics.evaluate(samples, self.annotations)
        self.assertEqual(result["coverage"]["missing_predictions"], 1)
        self.assertEqual(result["coverage"]["duplicate_prediction_ids"], 1)
        self.assertEqual(result["coverage"]["unknown_prediction_ids"], 1)
        self.assertEqual(result["coverage"]["parse_failures"], 1)
        self.assertFalse(result["summary"]["kpi"]["met"])
        self.assertEqual(
            result["metric_result"]["constraints"]["missing_predictions"], 1
        )
        missing_row = next(row for row in result["sample_metrics"] if row["id"] == missing["id"])
        self.assertEqual(missing_row["sample_score"], 0.0)
        self.assertEqual(missing_row["gap_type"], "missing_prediction")

    def test_dataset_balanced_profile_requires_minimum_cell_support(self) -> None:
        insufficient = multitask_metrics.evaluate(
            self.perfect_samples(),
            self.annotations,
            kpi_profile="task_dataset_balanced_v1",
            min_group_support=2,
        )
        self.assertGreater(
            insufficient["metric_result"]["constraints"]["insufficient_support_groups"],
            0,
        )
        self.assertFalse(insufficient["summary"]["kpi"]["met"])
        supported = multitask_metrics.evaluate(
            self.perfect_samples(),
            self.annotations,
            kpi_profile="task_dataset_balanced_v1",
            min_group_support=1,
        )
        self.assertEqual(supported["metric_result"]["value"], 1.0)
        self.assertTrue(supported["summary"]["kpi"]["met"])

    def test_analyze_gaps_cli_emits_rich_proxy_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            annotations = root / "proxy.json"
            results = root / "results.json"
            output = root / "metrics"
            annotations.write_text(json.dumps(self.annotations) + "\n")
            results.write_text(json.dumps(self.perfect_samples()) + "\n")
            status = analyze_gaps.main(
                [
                    "--results-json",
                    str(results),
                    "--annotations",
                    str(annotations),
                    "--output-dir",
                    str(output),
                    "--annotation-profile",
                    "nvpaw_multitask_v1",
                    "--evaluation-role",
                    "proxy",
                ]
            )
            self.assertEqual(status, 0)
            for name in (
                "metrics_summary.json",
                "metric_result.json",
                "task_metrics.json",
                "sample_metrics.parquet",
                "prediction_coverage.json",
                "gap_candidates.parquet",
                "selected_gaps.parquet",
                "gap_analysis_summary.json",
                "gaps_summary.json",
            ):
                self.assertTrue((output / name).is_file(), name)
                self.assertGreater((output / name).stat().st_size, 0, name)


class NVPawStateTests(unittest.TestCase):
    def test_rich_state_uses_frozen_benchmark_observed_tasks_as_required_groups(self) -> None:
        source = nvpaw_annotations.load_source_records(
            FIXTURES / "nvpaw_multitask.jsonl"
        )
        _, annotations = nvpaw_annotations.materialize_records(source)
        benchmark = [
            row for row in annotations if row["task_type"] != "Component Count"
        ]
        expected_groups = sorted(EXPECTED_TASKS - {"Component Count"})

        with tempfile.TemporaryDirectory() as temporary:
            workspace = pathlib.Path(temporary) / "workspace"
            results = pathlib.Path(temporary) / "results"
            (workspace / "annotations").mkdir(parents=True)
            (workspace / "specs").mkdir()
            (workspace / "annotations" / "proxy_kpi.json").write_text(
                json.dumps(annotations) + "\n"
            )
            (workspace / "annotations" / "benchmark_kpi.json").write_text(
                json.dumps(benchmark) + "\n"
            )
            (workspace / "annotations" / "mining_pool.json").write_text(
                json.dumps(annotations) + "\n"
            )
            for name in (
                "train_spec.toml",
                "evaluate_spec_proxy.toml",
                "evaluate_spec_benchmark.toml",
            ):
                (workspace / "specs" / name).write_text(
                    "[policy]\nautoml_policy = \"off\"\n"
                )

            status = init_deft_state.main(
                [
                    "--results-dir",
                    str(results),
                    "--workspace",
                    str(workspace),
                    "--platform",
                    "docker",
                    "--max-iterations",
                    "1",
                    "--gpu-model",
                    "L40S 48GB",
                    "--cosmos-container",
                    "example/cosmos:1",
                    "--mining-container",
                    "example/mining:1",
                    "--anomalygen-container",
                    "example/anomalygen:1",
                    "--annotation-profile",
                    "nvpaw_multitask_v1",
                    "--kpi-profile",
                    "task_balanced_v1",
                ]
            )

            self.assertEqual(status, 0)
            state = json.loads((results / "deft_state.json").read_text())
            self.assertEqual(
                state["metric_contract"]["required_groups"], expected_groups
            )

    def test_rich_state_freezes_annotation_kpi_and_gap_profiles(self) -> None:
        source = nvpaw_annotations.load_source_records(
            FIXTURES / "nvpaw_multitask.jsonl"
        )
        _, annotations = nvpaw_annotations.materialize_records(source)
        with tempfile.TemporaryDirectory() as temporary:
            workspace = pathlib.Path(temporary) / "workspace"
            results = pathlib.Path(temporary) / "results"
            (workspace / "annotations").mkdir(parents=True)
            (workspace / "specs").mkdir()
            for name in ("proxy_kpi.json", "benchmark_kpi.json", "mining_pool.json"):
                (workspace / "annotations" / name).write_text(json.dumps(annotations) + "\n")
            for name in (
                "train_spec.toml",
                "evaluate_spec_proxy.toml",
                "evaluate_spec_benchmark.toml",
            ):
                (workspace / "specs" / name).write_text("[policy]\nautoml_policy = \"off\"\n")
            status = init_deft_state.main(
                [
                    "--results-dir",
                    str(results),
                    "--workspace",
                    str(workspace),
                    "--platform",
                    "docker",
                    "--max-iterations",
                    "1",
                    "--gpu-model",
                    "L40S 48GB",
                    "--cosmos-container",
                    "example/cosmos:1",
                    "--mining-container",
                    "example/mining:1",
                    "--anomalygen-container",
                    "example/anomalygen:1",
                    "--annotation-profile",
                    "nvpaw_multitask_v1",
                    "--kpi-profile",
                    "task_balanced_v1",
                    "--gap-analysis-profile",
                    "equal_task_round_robin",
                    "--gap-analysis-seed",
                    "23",
                    "--mining-router-mode",
                    "task_strict",
                ]
            )
            self.assertEqual(status, 0)
            state = json.loads((results / "deft_state.json").read_text())
            self.assertEqual(state["version"], 6)
            self.assertEqual(state["config"]["annotation_profile"], "nvpaw_multitask_v1")
            self.assertEqual(state["metric_contract"]["name"], "balanced_score")
            self.assertEqual(
                state["metric_contract"]["required_groups"], sorted(EXPECTED_TASKS)
            )
            self.assertRegex(state["metric_contract_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                {item["name"] for item in state["metric_contract"]["constraints"]},
                {
                    "missing_predictions",
                    "duplicate_prediction_ids",
                    "unknown_prediction_ids",
                    "parse_failures",
                },
            )
            self.assertEqual(state["config"]["gap_analysis"]["profile"], "equal_task_round_robin")
            self.assertEqual(state["config"]["gap_analysis"]["resolved"]["seed"], 23)
            self.assertRegex(state["config"]["gap_analysis"]["sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                state["config"]["mining"]["history_aware"]["identity"], "target_id"
            )
            self.assertEqual(
                state["config"]["mining"]["router_mode"], "task_strict"
            )
            self.assertEqual(
                state["config"]["mining"]["top_k_scope"], "target_task"
            )
            self.assertEqual(state["config"]["anomalygen"]["policy"], "auto")
            self.assertTrue((results / "resolved_gap_analysis.yaml").is_file())

            mutated = copy.deepcopy(state)
            mutated["metric_contract"]["group_metric_target"] = 0.25
            with self.assertRaisesRegex(ValueError, "changed after state initialization"):
                metric_contract.contract_from_state(mutated)

    def test_disabled_anomalygen_policy_skips_without_gap_evidence(self) -> None:
        source = nvpaw_annotations.load_source_records(
            FIXTURES / "nvpaw_multitask.jsonl"
        )
        _, annotations = nvpaw_annotations.materialize_records(source)
        with tempfile.TemporaryDirectory() as temporary:
            workspace = pathlib.Path(temporary) / "workspace"
            results = pathlib.Path(temporary) / "results"
            (workspace / "annotations").mkdir(parents=True)
            (workspace / "specs").mkdir()
            for name in ("proxy_kpi.json", "benchmark_kpi.json", "mining_pool.json"):
                (workspace / "annotations" / name).write_text(
                    json.dumps(annotations) + "\n"
                )
            for name in (
                "train_spec.toml",
                "evaluate_spec_proxy.toml",
                "evaluate_spec_benchmark.toml",
            ):
                (workspace / "specs" / name).write_text(
                    '[policy]\nautoml_policy = "off"\n'
                )

            self.assertEqual(
                init_deft_state.main(
                    [
                        "--results-dir",
                        str(results),
                        "--workspace",
                        str(workspace),
                        "--platform",
                        "slurm",
                        "--max-iterations",
                        "1",
                        "--gpu-model",
                        "L40S 48GB",
                        "--cosmos-container",
                        "example/cosmos:1",
                        "--mining-container",
                        "example/mining:1",
                        "--annotation-profile",
                        "nvpaw_multitask_v1",
                        "--anomalygen-policy",
                        "disabled",
                    ]
                ),
                0,
            )
            state = json.loads((results / "deft_state.json").read_text())
            self.assertEqual(
                state["config"]["anomalygen"], {"policy": "disabled"}
            )
            self.assertEqual(
                state["config"]["training"]["annotation_source"],
                "generated_from_mining",
            )

            # No routing summary or driving Proxy error files exist. The
            # immutable disabled policy alone must authorize the stage skip.
            self.assertEqual(
                commit_stage.main(
                    [
                        "--results-dir",
                        str(results),
                        "--iter-label",
                        "iter1",
                        "--stage",
                        "anomalygen",
                        "--skip",
                        "--duration-sec",
                        "1",
                        "--summary",
                        "AnomalyGen disabled by operator policy",
                    ]
                ),
                0,
            )
            state = json.loads((results / "deft_state.json").read_text())
            phase = state["iterations"]["iter1"]
            self.assertTrue(phase["anomalygen_skipped"])
            self.assertEqual(phase["anomalygen_skip_reason"], "policy_disabled")
            self.assertIn(
                "disabled by immutable run policy",
                (results / "DEFT_Loop_Report.html").read_text(),
            )

            run_args = commit_stage._parser().parse_args(
                [
                    "--results-dir",
                    str(results),
                    "--iter-label",
                    "iter2",
                    "--stage",
                    "anomalygen",
                    "--duration-sec",
                    "1",
                    "--summary",
                    "must not run",
                ]
            )
            with self.assertRaisesRegex(
                ValueError, "anomalygen policy is disabled; commit this stage with --skip"
            ):
                commit_stage.commit(run_args)

    def test_balanced_best_iteration_uses_declared_tie_breakers(self) -> None:
        contract = metric_contract.validate_contract(
            {
                "name": "balanced_score",
                "display_name": "Worst task attainment",
                "operator": ">=",
                "target": 1.0,
                "unit": "",
                "evaluator": {
                    "type": "artifact",
                    "producer": "scripts/analyze_gaps.py",
                    "path_template": "/tmp/{iter_label}/metric_result.json",
                },
                "constraints": [],
                "tie_breakers": [
                    {"name": "macro_attainment", "direction": "max"},
                    {"name": "attainment_spread", "direction": "min"},
                    {"name": "coverage_failures", "direction": "min"},
                ],
            }
        )
        candidates = [
            (
                "baseline",
                {},
                {
                    "name": "balanced_score",
                    "value": 0.8,
                    "unit": "",
                    "constraints": {},
                    "tie_breakers": {
                        "macro_attainment": 0.85,
                        "attainment_spread": 0.1,
                        "coverage_failures": 0,
                    },
                },
            ),
            (
                "iter1",
                {},
                {
                    "name": "balanced_score",
                    "value": 0.8,
                    "unit": "",
                    "constraints": {},
                    "tie_breakers": {
                        "macro_attainment": 0.9,
                        "attainment_spread": 0.2,
                        "coverage_failures": 0,
                    },
                },
            ),
        ]
        self.assertEqual(metric_contract.pick_best(candidates, contract)[0], "iter1")


class NVPawDataGrowthTests(unittest.TestCase):
    def setUp(self) -> None:
        source = nvpaw_annotations.load_source_records(
            FIXTURES / "nvpaw_multitask.jsonl"
        )
        _, self.records = nvpaw_annotations.materialize_records(source)

    def test_selected_records_collapse_to_one_target_query_then_fan_out(self) -> None:
        board_one = [
            {
                "id": row["id"],
                "target_id": row["target_id"],
                "target_path": row["images"][row["image_roles"].index("target")],
                "image_paths": row["images"],
                "evaluation_role": "proxy",
                "task_type": row["task_type"],
                "metric_family": row["metric_family"],
                "reference_cohort": row["reference_cohort"],
                "dataset": row["dataset"],
                "sample_score": 0.0,
                "weakness_score": 1.0,
                "gap_type": "error",
                "parse_ok": True,
            }
            for row in self.records
            if row["target_id"] == "images/board-1.png"
        ]
        targets, routing = route_selected_gaps.route(board_one)
        self.assertEqual(len(targets), 1)
        self.assertEqual(len(targets[0]["record_ids"]), 2)
        self.assertEqual(routing["selected_records"], 2)
        self.assertEqual(routing["unique_targets"], 1)

        emitted, summary = emit_mined_sharegpt.emit_records(
            ["images/board-1.png"],
            self.records,
            media_root=pathlib.Path("/dataset"),
            relative=True,
            annotation_profile="nvpaw_multitask_v1",
        )
        self.assertEqual(len(emitted), 2)
        self.assertEqual({row["task_type"] for row in emitted}, {"Component Classification", "Defect Classification"})
        self.assertEqual(summary["embedding_queries"], 1)
        self.assertEqual(summary["output_records"], 2)

    def test_mining_router_modes_separate_visual_similarity_from_task_eligibility(self) -> None:
        target = [
            {
                "filepath": "proxy/detection.png",
                "target_id": "proxy-detection",
                "task_types": ["Defect Detection"],
                "embedding": [1.0, 0.0],
            }
        ]
        source = [
            {"filepath": "images/board-1.png", "embedding": [1.0, 0.0]},
            {"filepath": "images/board-4.png", "embedding": [0.9, 0.1]},
        ]

        image_only, image_summary = task_mining_router.route_candidates(
            target,
            source,
            self.records,
            media_root=pathlib.Path("/dataset"),
            mode="image_only",
            top_k_per_target=1,
            min_similarity=0.0,
        )
        strict, strict_summary = task_mining_router.route_candidates(
            target,
            source,
            self.records,
            media_root=pathlib.Path("/dataset"),
            mode="task_strict",
            top_k_per_target=1,
            min_similarity=0.0,
        )
        fallback, fallback_summary = task_mining_router.route_candidates(
            target,
            source,
            self.records,
            media_root=pathlib.Path("/dataset"),
            mode="task_then_fallback",
            top_k_per_target=2,
            min_similarity=0.0,
        )

        self.assertEqual(image_only[0]["filepath"], "images/board-1.png")
        self.assertEqual(image_only[0]["route_tier"], "image_only")
        self.assertEqual(strict[0]["filepath"], "images/board-4.png")
        self.assertEqual(strict[0]["route_tier"], "strict")
        self.assertEqual(strict[0]["routed_task_types"], ["Defect Detection"])
        self.assertEqual(
            [(row["filepath"], row["route_tier"]) for row in fallback],
            [
                ("images/board-4.png", "strict"),
                ("images/board-1.png", "fallback"),
            ],
        )
        self.assertEqual(image_summary["route_tier_counts"], {"image_only": 1})
        self.assertEqual(strict_summary["route_tier_counts"], {"strict": 1})
        self.assertEqual(
            fallback_summary["route_tier_counts"], {"fallback": 1, "strict": 1}
        )

    def test_strict_router_allocates_top_k_for_each_task_on_a_shared_target(self) -> None:
        routed, summary = task_mining_router.route_candidates(
            [
                {
                    "filepath": "proxy/shared.png",
                    "target_id": "proxy-shared",
                    "task_types": ["Component Classification", "Component Detection"],
                    "embedding": [1.0, 0.0],
                }
            ],
            [
                {"filepath": "images/board-1.png", "embedding": [1.0, 0.0]},
                {"filepath": "images/board-2.png", "embedding": [0.9, 0.1]},
            ],
            self.records,
            media_root=pathlib.Path("/dataset"),
            mode="task_strict",
            top_k_per_target=1,
            min_similarity=0.0,
        )
        self.assertEqual(
            [(row["filepath"], row["routed_task_types"]) for row in routed],
            [
                ("images/board-1.png", ["Component Classification"]),
                ("images/board-2.png", ["Component Detection"]),
            ],
        )
        self.assertEqual(summary["raw_selections"], 2)
        self.assertEqual(
            summary["targets"][0]["task_routes"],
            {
                "Component Classification": {
                    "fallback_selected": 0,
                    "selected": 1,
                    "shortfall": 0,
                    "strict_eligible": 1,
                    "strict_selected": 1,
                },
                "Component Detection": {
                    "fallback_selected": 0,
                    "selected": 1,
                    "shortfall": 0,
                    "strict_eligible": 1,
                    "strict_selected": 1,
                },
            },
        )

    def test_task_strict_router_keeps_component_count_in_its_own_route(self) -> None:
        routed, summary = task_mining_router.route_candidates(
            [
                {
                    "filepath": "proxy/count.png",
                    "target_id": "proxy-count",
                    "task_types": ["Component Count"],
                    "embedding": [1.0, 0.0],
                }
            ],
            [
                {"filepath": "images/board-1.png", "embedding": [1.0, 0.0]},
                {"filepath": "images/board-8.png", "embedding": [0.9, 0.1]},
            ],
            self.records,
            media_root=pathlib.Path("/dataset"),
            mode="task_strict",
            top_k_per_target=1,
            min_similarity=0.0,
        )

        self.assertEqual(len(routed), 1)
        self.assertEqual(routed[0]["filepath"], "images/board-8.png")
        self.assertEqual(routed[0]["routed_task_types"], ["Component Count"])
        self.assertEqual(
            summary["targets"][0]["task_routes"]["Component Count"],
            {
                "fallback_selected": 0,
                "selected": 1,
                "shortfall": 0,
                "strict_eligible": 1,
                "strict_selected": 1,
            },
        )

    def test_strict_mining_fanout_emits_only_routed_tasks(self) -> None:
        emitted, summary = emit_mined_sharegpt.emit_records(
            [
                {
                    "filepath": "images/board-1.png",
                    "route_tier": "strict",
                    "routed_task_types": ["Defect Classification"],
                }
            ],
            self.records,
            media_root=pathlib.Path("/dataset"),
            relative=True,
            annotation_profile="nvpaw_multitask_v1",
        )
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0]["task_type"], "Defect Classification")
        self.assertEqual(summary["route_tiers"], {"strict": 1})
        self.assertEqual(summary["tasks"], {"Defect Classification": 1})

    def test_task_router_and_emit_clis_preserve_routing_provenance(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            target_path = root / "target_embeddings.parquet"
            source_path = root / "source_embeddings.parquet"
            annotations_path = root / "mining.json"
            routed_path = root / "routed.parquet"
            router_summary_path = root / "router_summary.json"
            emitted_path = root / "mined_sharegpt.json"
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "filepath": "proxy/detection.png",
                            "target_id": "proxy-detection",
                            "task_types": ["Defect Detection"],
                            "embedding": [1.0, 0.0],
                        }
                    ]
                ),
                target_path,
            )
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {"filepath": "images/board-1.png", "embedding": [1.0, 0.0]},
                        {"filepath": "images/board-4.png", "embedding": [0.9, 0.1]},
                    ]
                ),
                source_path,
            )
            annotations_path.write_text(json.dumps(self.records) + "\n")

            self.assertEqual(
                task_mining_router.main(
                    [
                        "--target-embeddings",
                        str(target_path),
                        "--source-embeddings",
                        str(source_path),
                        "--source-annotations",
                        str(annotations_path),
                        "--media-root",
                        "/dataset",
                        "--mode",
                        "task_strict",
                        "--top-k-per-target",
                        "1",
                        "--min-similarity",
                        "0",
                        "--output",
                        str(routed_path),
                        "--summary",
                        str(router_summary_path),
                    ]
                ),
                0,
            )
            routed = pq.read_table(routed_path).to_pylist()
            self.assertEqual(routed[0]["route_tier"], "strict")
            self.assertEqual(routed[0]["routed_task_types"], ["Defect Detection"])

            self.assertEqual(
                emit_mined_sharegpt.main(
                    [
                        "--mined-parquet",
                        str(routed_path),
                        "--source-annotations",
                        str(annotations_path),
                        "--media-root",
                        "/dataset",
                        "--output",
                        str(emitted_path),
                        "--emit-relative",
                        "--annotation-profile",
                        "nvpaw_multitask_v1",
                    ]
                ),
                0,
            )
            emitted = json.loads(emitted_path.read_text())
            self.assertEqual([row["task_type"] for row in emitted], ["Defect Detection"])

    def test_rich_assembly_deduplicates_records_not_shared_media(self) -> None:
        shared = [row for row in self.records if row["target_id"] == "images/board-1.png"]
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            new_path = root / "new.json"
            new_path.write_text(json.dumps([*shared, copy.deepcopy(shared[0])]) + "\n")
            merged, summary = assemble_training_json.assemble(
                None,
                [new_path],
                dedupe=True,
                validation_paths=[],
                annotation_profile="nvpaw_multitask_v1",
            )
            self.assertEqual(len(merged), 2)
            self.assertEqual(summary["duplicates_skipped"], 1)
            self.assertEqual(summary["dedupe_key"], "record_fingerprint")
            self.assertEqual(summary["unique_target_images"]["output_total"], 1)
            self.assertEqual(summary["tasks"], {"Component Classification": 1, "Defect Classification": 1})

    def test_sdg_rejects_detection_and_emits_known_defect_classification(self) -> None:
        detection = next(row for row in self.records if row["task_type"] == "Ref_based Defect Detection")
        classification = next(row for row in self.records if row["task_type"] == "Ref_based Defect Classification")
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            generated = root / "reconstructed_image" / "board+scratch_0.png"
            clean = root / "original_image" / "board+scratch_0.png"
            generated.parent.mkdir()
            clean.parent.mkdir()
            generated.write_bytes(b"png")
            clean.write_bytes(b"png")
            csv_path = root / "SDG_result.csv"
            csv_path.write_text(
                "reconstructed_image,original_image\n"
                "reconstructed_image/board+scratch_0.png,original_image/board+scratch_0.png\n"
            )
            with self.assertRaisesRegex(ValueError, "detection.*bbox"):
                emit_sdg_sharegpt.emit_records(
                    csv_path,
                    media_root=root,
                    prompt=classification["conversations"][0]["value"],
                    relative=True,
                    annotation_profile="nvpaw_multitask_v1",
                    template_record=detection,
                )
            records, summary = emit_sdg_sharegpt.emit_records(
                csv_path,
                media_root=root,
                prompt=classification["conversations"][0]["value"],
                relative=True,
                annotation_profile="nvpaw_multitask_v1",
                template_record=classification,
            )
            self.assertEqual(records[0]["image_roles"], ["golden", "target"])
            self.assertEqual(records[0]["answer"], {"kind": "choice_set", "labels": ["defect"]})
            self.assertEqual(summary["tasks"], {"Ref_based Defect Classification": 1})

            option_template = copy.deepcopy(classification)
            option_template["task_type"] = "Defect Classification"
            option_template["reference_cohort"] = "single_target"
            option_template["prompt_format"] = (
                "nvpaw.defect_classification.single.official_v1"
            )
            option_template["image_roles"] = ["target"]
            option_template["images"] = [classification["images"][-1]]
            option_template["option_map"] = {
                "A": "Yes, this image contains a defect.",
                "B": "No, this image does not contain a defect.",
            }
            option_template["conversations"][0]["value"] = (
                "Is there a defect?\nA. Yes, this image contains a defect.\n"
                "B. No, this image does not contain a defect."
            )
            option_records, _ = emit_sdg_sharegpt.emit_records(
                csv_path,
                media_root=root,
                prompt=option_template["conversations"][0]["value"],
                relative=True,
                annotation_profile="nvpaw_multitask_v1",
                template_record=option_template,
            )
            self.assertEqual(option_records[0]["conversations"][-1]["value"], "A")
            self.assertEqual(
                option_records[0]["answer"]["labels"],
                ["Yes, this image contains a defect."],
            )
            validate_sharegpt.validate_records(
                option_records,
                media_root=root,
                require_files=False,
                annotation_profile="nvpaw_multitask_v1",
            )


class NVPawNoDockerIntegrationTests(unittest.TestCase):
    def test_materialize_score_select_route_assemble_validate_and_commit(self) -> None:
        source = nvpaw_annotations.load_source_records(
            FIXTURES / "nvpaw_multitask.jsonl"
        )
        _manifest, records = nvpaw_annotations.materialize_records(source)
        malformed_predictions = [
            {"id": record["id"], "response": "not a valid task answer"}
            for record in records
        ]

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = root / "workspace"
            results = root / "results"
            annotations_dir = workspace / "annotations"
            specs_dir = workspace / "specs"
            annotations_dir.mkdir(parents=True)
            specs_dir.mkdir()
            for name in ("proxy_kpi.json", "benchmark_kpi.json", "mining_pool.json"):
                (annotations_dir / name).write_text(json.dumps(records) + "\n")
            for name in (
                "train_spec.toml",
                "evaluate_spec_proxy.toml",
                "evaluate_spec_benchmark.toml",
            ):
                (specs_dir / name).write_text('[policy]\nautoml_policy = "off"\n')

            self.assertEqual(
                init_deft_state.main(
                    [
                        "--results-dir",
                        str(results),
                        "--workspace",
                        str(workspace),
                        "--platform",
                        "docker",
                        "--max-iterations",
                        "1",
                        "--gpu-model",
                        "dry-run-only",
                        "--cosmos-container",
                        "example/cosmos:1",
                        "--mining-container",
                        "example/mining:1",
                        "--anomalygen-container",
                        "example/anomalygen:1",
                        "--annotation-profile",
                        "nvpaw_multitask_v1",
                        "--kpi-profile",
                        "task_balanced_v1",
                        "--gap-analysis-profile",
                        "equal_task_round_robin",
                    ]
                ),
                0,
            )

            def commit(label: str, stage: str, *artifacts: str) -> None:
                self.assertEqual(
                    commit_stage.main(
                        [
                            "--results-dir",
                            str(results),
                            "--iter-label",
                            label,
                            "--stage",
                            stage,
                            "--summary",
                            f"offline dry-run {stage}",
                            "--duration-sec",
                            "1",
                            *artifacts,
                        ]
                    ),
                    0,
                    stage,
                )

            benchmark_eval = results / "baseline" / "evaluate_benchmark"
            benchmark_eval.mkdir(parents=True)
            benchmark_results = benchmark_eval / "results.json"
            benchmark_results.write_text(json.dumps(malformed_predictions) + "\n")
            benchmark_metrics = results / "baseline" / "benchmark_metrics"
            self.assertEqual(
                analyze_gaps.main(
                    [
                        "--results-json",
                        str(benchmark_results),
                        "--annotations",
                        str(annotations_dir / "benchmark_kpi.json"),
                        "--output-dir",
                        str(benchmark_metrics),
                        "--annotation-profile",
                        "nvpaw_multitask_v1",
                        "--evaluation-role",
                        "benchmark",
                    ]
                ),
                0,
            )
            commit(
                "baseline",
                "evaluate_benchmark",
                "--benchmark-results",
                str(benchmark_results),
            )
            commit(
                "baseline",
                "benchmark_metrics",
                "--benchmark-metrics-summary",
                str(benchmark_metrics / "metrics_summary.json"),
                "--metric-result",
                str(benchmark_metrics / "metric_result.json"),
                "--task-metrics",
                str(benchmark_metrics / "task_metrics.json"),
                "--sample-metrics",
                str(benchmark_metrics / "sample_metrics.parquet"),
                "--prediction-coverage",
                str(benchmark_metrics / "prediction_coverage.json"),
            )

            proxy_eval = results / "baseline" / "evaluate_proxy"
            proxy_eval.mkdir()
            proxy_results = proxy_eval / "results.json"
            proxy_results.write_text(json.dumps(malformed_predictions) + "\n")
            proxy_rcca = results / "baseline" / "proxy_rcca"
            self.assertEqual(
                analyze_gaps.main(
                    [
                        "--results-json",
                        str(proxy_results),
                        "--annotations",
                        str(annotations_dir / "proxy_kpi.json"),
                        "--output-dir",
                        str(proxy_rcca),
                        "--annotation-profile",
                        "nvpaw_multitask_v1",
                        "--evaluation-role",
                        "proxy",
                        "--gap-analysis-profile",
                        "equal_task_round_robin",
                    ]
                ),
                0,
            )
            commit(
                "baseline",
                "evaluate_proxy",
                "--proxy-results",
                str(proxy_results),
            )
            commit(
                "baseline",
                "proxy_rcca",
                "--proxy-gaps-summary",
                str(proxy_rcca / "gaps_summary.json"),
                "--task-metrics",
                str(proxy_rcca / "task_metrics.json"),
                "--sample-metrics",
                str(proxy_rcca / "sample_metrics.parquet"),
                "--prediction-coverage",
                str(proxy_rcca / "prediction_coverage.json"),
                "--gap-candidates",
                str(proxy_rcca / "gap_candidates.parquet"),
                "--selected-gaps",
                str(proxy_rcca / "selected_gaps.parquet"),
                "--gap-analysis-summary",
                str(proxy_rcca / "gap_analysis_summary.json"),
            )

            routing = results / "iter1" / "routing"
            targets_json = routing / "mining_targets.json"
            targets_parquet = routing / "mining_targets.parquet"
            routing_summary = routing / "routing_summary.json"
            self.assertEqual(
                route_selected_gaps.main(
                    [
                        "--selected-gaps",
                        str(proxy_rcca / "selected_gaps.parquet"),
                        "--output-json",
                        str(targets_json),
                        "--output-parquet",
                        str(targets_parquet),
                        "--summary",
                        str(routing_summary),
                    ]
                ),
                0,
            )
            routing_evidence = json.loads(routing_summary.read_text())
            self.assertEqual(routing_evidence["selected_records"], len(records))
            self.assertEqual(routing_evidence["embedding_queries"], 8)
            commit(
                "iter1",
                "routing",
                "--mining-targets",
                str(targets_json),
                "--mining-targets-parquet",
                str(targets_parquet),
                "--routing-summary",
                str(routing_summary),
            )

            targets = json.loads(targets_json.read_text())
            mined_records, emit_summary = emit_mined_sharegpt.emit_records(
                [target["filepath"] for target in targets],
                records,
                media_root=workspace,
                relative=True,
                annotation_profile="nvpaw_multitask_v1",
            )
            assemble_dir = results / "iter1" / "assemble"
            assemble_dir.mkdir()
            mined_json = assemble_dir / "mined_sharegpt.json"
            mined_json.write_text(json.dumps(mined_records) + "\n")
            combined, assemble_summary = assemble_training_json.assemble(
                None,
                [mined_json],
                dedupe=True,
                validation_paths=[],
                annotation_profile="nvpaw_multitask_v1",
            )
            combined_json = assemble_dir / "train_iter_1.json"
            assemble_summary_path = assemble_dir / "assemble_summary.json"
            combined_json.write_text(json.dumps(combined) + "\n")
            assemble_summary_path.write_text(json.dumps(assemble_summary) + "\n")
            self.assertEqual(emit_summary["embedding_queries"], 8)
            self.assertEqual(len(combined), len(records))
            commit(
                "iter1",
                "assemble_data",
                "--mined-sharegpt",
                str(mined_json),
                "--combined-training",
                str(combined_json),
                "--assemble-summary",
                str(assemble_summary_path),
            )

            validation_dir = results / "iter1" / "validate"
            validation_dir.mkdir()
            validation_report = validation_dir / "validation_report.json"
            validation_summary = validate_sharegpt.validate_records(
                combined,
                media_root=workspace,
                require_files=False,
                annotation_profile="nvpaw_multitask_v1",
            )
            validation_report.write_text(json.dumps(validation_summary) + "\n")
            commit(
                "iter1",
                "validate_data",
                "--validation-report",
                str(validation_report),
            )

            state = json.loads((results / "deft_state.json").read_text())
            self.assertEqual(state["version"], 6)
            self.assertEqual(state["iterations"]["iter1"]["stage_completed"], "validate_data")
            self.assertEqual(
                [event["stage"] for event in state["events"]],
                [
                    "evaluate_benchmark",
                    "benchmark_metrics",
                    "evaluate_proxy",
                    "proxy_rcca",
                    "routing",
                    "assemble_data",
                    "validate_data",
                ],
            )
            report = (results / "DEFT_Loop_Report.html").read_text()
            self.assertIn(
                "NVPaw multi-task classification, counting, and detection", report
            )
            self.assertIn("Task KPI attainment", report)
            self.assertIn("Worst-group summary", report)
            self.assertIn("Component Detection", report)
            self.assertIn("Prediction coverage constraints", report)


if __name__ == "__main__":
    unittest.main()
