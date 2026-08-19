#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

# Other DEFT suites use the same top-level module names. Ensure this suite
# imports the Cosmos3-local implementations when pytest shares an interpreter.
for module_name in (
    "commit_stage",
    "deft_context",
    "init_deft_state",
    "metric_contract",
    "record_metric_result",
    "render_report",
):
    sys.modules.pop(module_name, None)

import analyze_gaps  # noqa: E402
import assemble_training_json  # noqa: E402
import check_annotations  # noqa: E402
import commit_stage  # noqa: E402
import deft_context  # noqa: E402
import emit_mined_sharegpt  # noqa: E402
import init_deft_state  # noqa: E402
import validate_sharegpt  # noqa: E402
import validate_split_contract  # noqa: E402


def record(image: str, label: str, *, record_id: str | None = None) -> dict:
    payload = {
        "images": [image],
        "conversations": [
            {"from": "human", "value": "Inspect this image. Return OK or NG."},
            {"from": "gpt", "value": label},
        ],
    }
    if record_id is not None:
        payload["id"] = record_id
    return payload


def write_json(path: pathlib.Path, payload: object) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


class BareAnnotationTests(unittest.TestCase):
    def test_single_image_and_exact_bare_labels(self) -> None:
        summary = validate_sharegpt.validate_records(
            [record("a.png", "OK")],
            media_root=pathlib.Path("/tmp"),
            require_files=False,
        )
        self.assertEqual(summary["mode"], "bare_okng")
        self.assertEqual(summary["labels"], {"OK": 1})

        invalid_label = [record("b.png", "Final answer: NG")]
        with self.assertRaisesRegex(ValueError, "exactly OK or NG"):
            validate_sharegpt.validate_records(
                invalid_label,
                media_root=pathlib.Path("/tmp"),
                require_files=False,
            )

        two_images = record("a.png", "OK")
        two_images["images"].append("reference.png")
        with self.assertRaisesRegex(ValueError, "exactly one image"):
            validate_sharegpt.validate_records(
                [two_images],
                media_root=pathlib.Path("/tmp"),
                require_files=False,
            )

    def test_evaluation_ids_are_unique_and_filesystem_safe(self) -> None:
        media_root = pathlib.Path("/tmp")
        validate_sharegpt.validate_records(
            [record("a.png", "OK", record_id="a-01")],
            media_root=media_root,
            require_files=False,
            require_id=True,
        )
        with self.assertRaisesRegex(ValueError, "id must be a non-empty string"):
            validate_sharegpt.validate_records(
                [record("a.png", "OK")],
                media_root=media_root,
                require_files=False,
                require_id=True,
            )
        with self.assertRaisesRegex(ValueError, "filesystem-safe"):
            validate_sharegpt.validate_records(
                [record("a.png", "OK", record_id="not/a/path")],
                media_root=media_root,
                require_files=False,
                require_id=True,
            )

    def test_annotation_contract_is_checked_per_role(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = pathlib.Path(temporary)
            for role, spec in check_annotations.ROLE_CONTRACT.items():
                identifier = role if role != "mining" else None
                write_json(
                    workspace / "annotations" / spec["filename"],
                    [record(f"images/{role}.png", "OK", record_id=identifier)],
                )
            report, failures = check_annotations.check(
                {
                    role: workspace / "annotations" / spec["filename"]
                    for role, spec in check_annotations.ROLE_CONTRACT.items()
                },
                media_root=workspace,
                require_files=False,
            )
            self.assertEqual(failures, [])
            self.assertEqual(report["mining"]["id_coverage"], "n/a")
            self.assertEqual(report["benchmark"]["id_coverage"], "1/1")

    def test_loader_requires_one_json_array(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            valid = write_json(root / "valid.json", [record("a.png", "OK")])
            self.assertEqual(len(validate_sharegpt.load_records(valid)), 1)
            jsonl = root / "invalid.jsonl"
            jsonl.write_text(
                json.dumps(record("a.png", "OK"))
                + "\n"
                + json.dumps(record("b.png", "NG"))
                + "\n"
            )
            with self.assertRaisesRegex(ValueError, "JSONL is not supported"):
                validate_sharegpt.load_records(jsonl)

    def test_mined_alignment_preserves_single_image_prompt_and_label(self) -> None:
        source = [record("pool/a.png", "NG")]
        output, summary = emit_mined_sharegpt.emit_records(
            ["pool/a.png"],
            source,
            media_root=pathlib.Path("/workspace"),
            relative=True,
        )
        self.assertEqual(summary["mode"], "bare_okng")
        self.assertEqual(output[0]["images"], ["pool/a.png"])
        self.assertEqual(output[0]["conversations"][-1]["value"], "NG")

    def test_training_assembly_is_monotonic_and_deduplicated_by_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            first_input = write_json(root / "new.json", [record("new.png", "NG")])
            first, summary = assemble_training_json.assemble(
                None, [first_input], dedupe=True, validation_paths=[]
            )
            self.assertEqual(len(first), 1)
            self.assertEqual(summary["dedupe_key"], "media")

            previous = write_json(root / "iter1.json", first)
            second_input = write_json(
                root / "next.json",
                [record("new.png", "NG"), record("next.png", "OK")],
            )
            merged, second_summary = assemble_training_json.assemble(
                previous, [second_input], dedupe=True, validation_paths=[]
            )
            self.assertEqual([item["images"] for item in merged], [["new.png"], ["next.png"]])
            self.assertEqual(second_summary["duplicates_skipped"], 1)


class IsolationAndMetricTests(unittest.TestCase):
    def test_split_contract_and_frozen_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            role_paths = {
                role: write_json(root / f"{role}.json", [record(f"{role}.png", "OK")])
                for role in ("proxy", "benchmark", "mining")
            }
            expected = hashlib.sha256(role_paths["benchmark"].read_bytes()).hexdigest()
            summary = validate_split_contract.validate(
                role_paths,
                media_root=root,
                expected_benchmark_sha256=expected,
            )
            self.assertTrue(summary["benchmark_hash_verified"])

            iter1 = write_json(root / "train_iter1.json", [record("mining.png", "OK")])
            iter1_summary = validate_split_contract.validate(
                {**role_paths, "train": iter1},
                media_root=root,
                expected_benchmark_sha256=expected,
            )
            self.assertEqual(iter1_summary["roles"]["train"], "generated_from_mining")

            mining_iter2 = write_json(
                root / "mining_iter2.json",
                [record("mining2.png", "NG")],
            )
            iter2_roles = {**role_paths, "mining": mining_iter2, "previous_train": iter1}
            iter2 = write_json(
                root / "train_iter2.json",
                [record("mining.png", "OK"), record("mining2.png", "NG")],
            )
            iter2_summary = validate_split_contract.validate(
                {**iter2_roles, "train": iter2}, media_root=root
            )
            self.assertEqual(iter2_summary["target_overlap"]["train:previous_train"], 1)

            dropped = write_json(root / "dropped.json", [record("mining2.png", "NG")])
            with self.assertRaisesRegex(ValueError, "retain every record"):
                validate_split_contract.validate(
                    {**iter2_roles, "train": dropped}, media_root=root
                )

            outside = write_json(root / "outside.json", [record("outside.png", "OK")])
            with self.assertRaisesRegex(ValueError, "must come from the Mining"):
                validate_split_contract.validate(
                    {**role_paths, "train": outside}, media_root=root
                )

            leaking_roles = {
                **role_paths,
                "mining": write_json(root / "leaking_mining.json", [record("proxy.png", "NG")]),
            }
            with self.assertRaisesRegex(ValueError, "target leakage"):
                validate_split_contract.validate(leaking_roles, media_root=root)

    def test_proxy_never_gates_and_benchmark_unknown_blocks(self) -> None:
        samples = [
            {"gt": "NG", "response": "NG"},
            {"gt": "OK", "response": "OK"},
        ]
        proxy, *_ = analyze_gaps.analyze(samples, evaluation_role="proxy")
        self.assertNotIn("metric_result", proxy)

        benchmark, *_ = analyze_gaps.analyze(
            [{"gt": "NG", "response": "unclear"}],
            evaluation_role="benchmark",
        )
        self.assertEqual(benchmark["unknown_samples"], 1)


class StateMachineTests(unittest.TestCase):
    def test_routing_transitions_directly_to_data_mining(self) -> None:
        state = {
            "status": "in_progress",
            "current_iteration": 1,
            "max_iterations": 2,
            "iterations": {"iter1": {"stage_completed": "routing"}},
        }
        self.assertEqual(deft_context._next_stage(state), ("iter1", "data_mining"))

    def test_state_initialization_has_mining_only_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = root / "workspace"
            results = root / "results"
            (workspace / "specs").mkdir(parents=True)
            for name in ("train_spec.toml", "evaluate_spec.toml"):
                (workspace / "specs" / name).write_text("value = 1\n")
            for filename, label in (
                ("proxy_kpi.json", "OK"),
                ("benchmark_kpi.json", "NG"),
                ("mining_pool.json", "OK"),
            ):
                write_json(
                    workspace / "annotations" / filename,
                    [record(f"images/{filename}.png", label)],
                )

            rc = init_deft_state.main(
                [
                    "--results-dir",
                    str(results),
                    "--workspace",
                    str(workspace),
                    "--platform",
                    "docker",
                    "--gpu-model",
                    "NVIDIA H100 80GB HBM3",
                    "--max-iterations",
                    "1",
                    "--cosmos-container",
                    "example/cosmos:1",
                    "--mining-container",
                    "example/mining:1",
                ]
            )
            self.assertEqual(rc, 0)
            state = json.loads((results / "deft_state.json").read_text())
            self.assertEqual(
                state["config"]["training"]["annotation_source"],
                "generated_from_mining",
            )
            self.assertNotIn("anomalygen", state["config"])
            self.assertNotIn("anomalygen", state["_completed_step_values"])

    def test_removed_stage_and_options_are_not_public(self) -> None:
        init_options = {
            option
            for action in init_deft_state._parser()._actions
            for option in action.option_strings
        }
        commit_parser = commit_stage._parser()
        commit_options = {
            option for action in commit_parser._actions for option in action.option_strings
        }
        stage_action = next(
            action for action in commit_parser._actions if action.dest == "stage"
        )
        self.assertFalse(any("anomaly" in option for option in init_options))
        self.assertFalse(any("anomaly" in option for option in commit_options))
        self.assertNotIn("anomalygen", stage_action.choices)

    def test_cosmos3_variant_aliases_are_explicit_and_canonical(self) -> None:
        self.assertEqual(
            init_deft_state.canonicalize_base_model("nano"),
            "nvidia/Cosmos3-Nano",
        )
        self.assertEqual(
            init_deft_state.canonicalize_base_model("edge"),
            "nvidia/Cosmos3-Edge",
        )
        self.assertEqual(
            init_deft_state.canonicalize_base_model("super"),
            "nvidia/Cosmos3-Super",
        )


if __name__ == "__main__":
    unittest.main()
