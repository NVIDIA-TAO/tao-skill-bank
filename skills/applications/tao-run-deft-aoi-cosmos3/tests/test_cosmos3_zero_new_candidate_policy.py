# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Zero-new-candidate policy (Feature C): skip individually exhausted maintenance tasks,
record the shortage, continue; fail closed when all are exhausted or nothing is added."""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import assemble_training_json  # noqa: E402
import defect_detection_ablation as ablation  # noqa: E402
import init_deft_state  # noqa: E402
import render_iteration_mining_runner as runner  # noqa: E402
from test_cosmos3_defect_detection_ablation_contract import _candidate, _row  # noqa: E402

ALL_MAINTENANCE = ablation.MAINTENANCE_TASK_TYPES
PRESENT = ("Component Detection", "Defect Classification", "Ref_based Defect Detection")
ABSENT = ("Component Classification", "Ref_based Defect Classification")
ONE_BOX = [{"bbox_2d": [10, 10, 100, 100], "label": "open"}]


def _fixture(maintenance_tasks=PRESENT, *, reference_mix: bool = False, near_duplicate_absent_task: bool = False):
    """12 Defect Detection rows (8 positive, 3 FP empties, 1 calibration empty) + 4 rows per maintenance task."""
    rows: list[dict] = []
    candidates: list[dict] = []
    for index in range(8):
        boxes = [{"bbox_2d": [10 + index, 20, 110 + index * 10, 120], "label": ("open", "short")[index % 2]}]
        row = _row(f"dd-pos-{index}", "Defect Detection", boxes=boxes, dataset=f"source-{index % 2}")
        rows.append(row)
        candidates.append(_candidate(row, evidence=["hard_positive_proxy_false_negative"], phash=f"{index + 1:016x}"))
    for index in range(4):
        row = _row(f"dd-empty-{index}", "Defect Detection", boxes=[])
        rows.append(row)
        candidates.append(_candidate(
            row, evidence=["hard_negative_proxy_false_positive"] if index < 3 else ["calibration_empty_ground_truth"],
            phash=f"{index + 20:016x}", route_tier="strict" if index < 3 else "calibration",
        ))
    counter = 0
    for task in maintenance_tasks:
        for index in range(4):
            boxes = ONE_BOX if (reference_mix and task == "Ref_based Defect Detection" and index % 2) else None
            row = _row(f"maint-{task.replace(' ', '_')}-{index}", task, boxes=boxes)
            rows.append(row)
            candidates.append(_candidate(row, phash=f"{counter + 40:016x}"))
            counter += 1
    if near_duplicate_absent_task:
        # an eligible Component Classification row that the perceptual filter drops (same hash as dd-pos-0)
        row = _row("maint-cc-near-dup", "Component Classification")
        rows.append(row)
        candidates.append(_candidate(row, phash=f"{1:016x}"))
    return rows, candidates


def _materialize(rows, candidates, **overrides):
    kwargs = dict(
        candidate_rows=candidates, source_records=rows, validation_records=[], media_root=pathlib.Path("/data"),
        max_rows=24, row_multiple=6, defect_detection_fraction=0.5, proxy_empty_rate=1 / 3,
        epochs=5, global_batch=6, near_duplicate_hamming_distance=0,
    )
    kwargs.update(overrides)
    return ablation.materialize(**kwargs)


def _write(path: pathlib.Path, rows: list[dict]) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _proxy_rows(root: pathlib.Path) -> list[dict]:
    # single-image empty rate 1/3, reference empty rate 1/2; distinct images from the fixture.
    # Reference pairs are validation records whose ordered image bytes are hashed, so they
    # point at small real files under the temp root (absolute paths bypass --media-root).
    rows = [
        _row("proxy-dd-empty", "Defect Detection", boxes=[]),
        _row("proxy-dd-pos-0", "Defect Detection", boxes=ONE_BOX),
        _row("proxy-dd-pos-1", "Defect Detection", boxes=ONE_BOX),
        _row("proxy-ref-empty", "Ref_based Defect Detection", boxes=[]),
        _row("proxy-ref-pos", "Ref_based Defect Detection", boxes=ONE_BOX),
    ]
    for row in rows[3:]:
        for item in row["messages"][0]["content"]:
            if item.get("type") == "image":
                target = root / "proxy-images" / pathlib.Path(item["image"]).name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(f"png:{target.name}".encode())
                item["image"] = str(target)
    return rows


class ZeroNewCandidatePolicyTests(unittest.TestCase):
    def test_default_policy_fails_closed_when_a_maintenance_task_is_absent(self) -> None:
        rows, candidates = _fixture()
        selected, manifest = _materialize(rows, candidates)
        self.assertEqual(len(selected), 24)
        self.assertFalse(manifest["verified"])
        self.assertEqual(manifest["zero_new_candidate_policy"], "fail_closed")
        self.assertFalse(manifest["verification"]["all_five_maintenance_tasks_present"])
        self.assertFalse(manifest["verification"]["maintenance_tasks_present_or_exhausted"])
        self.assertEqual(manifest["maintenance_tasks"]["missing"], list(ABSENT))
        self.assertEqual(sorted(manifest["exhausted_tasks"]), list(ABSENT))
        self.assertEqual(manifest["skipped_tasks"], [])
        self.assertEqual(manifest["zero_new_candidate_block_reason"], "policy_fail_closed")
        self.assertEqual(manifest["verification_policy_exclusions"], [])
        with self.assertRaisesRegex(ValueError, "not verified"):
            ablation.bind_cumulative_manifest(
                manifest, current_jsonl=pathlib.Path("/nonexistent"), training_jsonl=pathlib.Path("/nonexistent"),
                assembly_summary={}, epochs=5, global_batch=6,
            )

    def test_skip_exhausted_accepts_exhausted_tasks_and_records_them(self) -> None:
        rows, candidates = _fixture()
        selected, manifest = _materialize(rows, candidates, zero_new_candidate_policy="skip_exhausted")
        self.assertEqual(len(selected), 24)
        self.assertTrue(manifest["verified"], manifest["verification"])
        self.assertEqual(manifest["zero_new_candidate_policy"], "skip_exhausted")
        self.assertFalse(manifest["verification"]["all_five_maintenance_tasks_present"])  # the raw fact is kept
        self.assertTrue(manifest["verification"]["maintenance_tasks_present_or_exhausted"])
        self.assertEqual(manifest["verification_policy_exclusions"], ["all_five_maintenance_tasks_present"])
        self.assertEqual(manifest["skipped_tasks"], list(ABSENT))
        for task in ABSENT:
            self.assertEqual(manifest["exhausted_tasks"][task], {
                "routed_candidates": 0, "eligible_after_exclusion": 0, "selected": 0, "materialized": 0,
            })
        self.assertEqual(manifest["maintenance_tasks"]["present"], list(PRESENT))
        self.assertEqual(manifest["maintenance_tasks"]["eligible_after_exclusion"]["Component Detection"], 4)
        self.assertEqual(manifest["maintenance_tasks"]["routed_candidates"]["Component Detection"], 4)
        self.assertIsNone(manifest["zero_new_candidate_block_reason"])
        self.assertEqual(manifest["maintenance_marginal_quota"]["shortages"]["Component Classification"], 12 // 5 + 1)

    def test_skip_exhausted_still_fails_when_an_absent_task_has_eligible_candidates(self) -> None:
        rows, candidates = _fixture(near_duplicate_absent_task=True)
        selected, manifest = _materialize(rows, candidates, zero_new_candidate_policy="skip_exhausted")
        self.assertFalse(manifest["verified"])
        self.assertNotIn("Component Classification", {r["task_type"] for r in selected})
        self.assertEqual(manifest["maintenance_tasks"]["eligible_after_exclusion"]["Component Classification"], 1)
        self.assertEqual(manifest["maintenance_tasks"]["routed_candidates"]["Component Classification"], 1)
        self.assertNotIn("Component Classification", manifest["exhausted_tasks"])
        self.assertIn("Ref_based Defect Classification", manifest["exhausted_tasks"])
        self.assertEqual(manifest["skipped_tasks"], [])
        self.assertFalse(manifest["verification"]["maintenance_tasks_present_or_exhausted"])
        self.assertEqual(manifest["zero_new_candidate_block_reason"], "maintenance_task_absent_with_eligible_candidates")

    def test_skip_exhausted_fails_when_every_maintenance_task_is_exhausted(self) -> None:
        rows: list[dict] = []
        candidates: list[dict] = []
        for index in range(16):
            row = _row(f"dd-pos-{index}", "Defect Detection", boxes=[{"bbox_2d": [10 + index, 20, 110 + index * 10, 120], "label": "open"}])
            rows.append(row)
            candidates.append(_candidate(row, evidence=["hard_positive_proxy_false_negative"], phash=f"{index + 1:016x}"))
        for index in range(8):
            row = _row(f"dd-empty-{index}", "Defect Detection", boxes=[])
            rows.append(row)
            candidates.append(_candidate(row, evidence=["hard_negative_proxy_false_positive"], phash=f"{index + 40:016x}"))
        selected, manifest = _materialize(rows, candidates, zero_new_candidate_policy="skip_exhausted")
        self.assertEqual(len(selected), 24)
        self.assertEqual(sorted(manifest["exhausted_tasks"]), sorted(ALL_MAINTENANCE))
        self.assertFalse(manifest["verification"]["maintenance_tasks_present_or_exhausted"])
        self.assertFalse(manifest["verified"])
        self.assertEqual(manifest["skipped_tasks"], [])
        self.assertEqual(manifest["zero_new_candidate_block_reason"], "all_maintenance_tasks_exhausted")

    def test_zero_new_rows_fail_closed_under_both_policies(self) -> None:
        rows, candidates = _fixture(ALL_MAINTENANCE)
        for policy in ("fail_closed", "skip_exhausted"):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "zero new rows"):
                _materialize(rows, candidates, previous_records=rows, zero_new_candidate_policy=policy)
        with self.assertRaisesRegex(ValueError, "zero_new_candidate_policy"):
            _materialize(rows, candidates, zero_new_candidate_policy="bogus")

    def test_cli_exit_codes_and_the_cumulative_bind_follow_the_policy(self) -> None:
        rows, candidates = _fixture(reference_mix=True)
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = _write(root / "source.jsonl", rows)
            proxy = _write(root / "proxy.jsonl", _proxy_rows(root))
            argv = [
                "--candidate-parquet", str(root / "candidates.parquet"), "--source-annotations", str(source),
                "--proxy-annotations", str(proxy), "--media-root", "/data", "--max-rows", "24", "--row-multiple", "6",
                "--epochs", "5", "--global-batch", "6", "--near-duplicate-hamming-distance", "0",
                "--output", str(root / "off/mined.jsonl"), "--manifest", str(root / "off/quota.json"),
            ]
            with mock.patch.object(ablation, "_read_parquet", return_value=candidates):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
                    rc = ablation.main(argv)
                self.assertEqual(rc, 2)
                self.assertIn("not verified", stderr.getvalue())
                self.assertFalse(json.loads((root / "off/quota.json").read_text())["verified"])
                argv_on = argv[:-4] + ["--zero-new-candidate-policy", "skip_exhausted",
                                       "--output", str(root / "on/mined.jsonl"), "--manifest", str(root / "on/quota.json")]
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    rc = ablation.main(argv_on)
                self.assertEqual(rc, 0, stdout.getvalue())
                self.assertIn("skipped_tasks=", stdout.getvalue())
            manifest = json.loads((root / "on/quota.json").read_text())
            self.assertTrue(manifest["verified"])
            self.assertEqual(manifest["skipped_tasks"], list(ABSENT))
            self.assertEqual(manifest["training_jsonl"]["rows"], 24)
            mined = root / "on/mined.jsonl"
            train_rows, summary = assemble_training_json.assemble(None, mined, validation_paths=[], media_root=pathlib.Path("/data"), row_multiple=6)
            train = _write(root / "on/train.jsonl", train_rows)
            summary = assemble_training_json.bind_summary(summary, train)
            bound = ablation.bind_cumulative_manifest(
                manifest, current_jsonl=mined, training_jsonl=train, assembly_summary=summary, epochs=5, global_batch=6,
            )
            self.assertTrue(bound["verified"])
            self.assertTrue(bound["verification"]["cumulative_lineage_verified"])
            self.assertFalse(bound["verification"]["all_five_maintenance_tasks_present"])
            self.assertEqual(bound["current_selection"]["skipped_tasks"], list(ABSENT))
            self.assertEqual(sorted(bound["current_selection"]["exhausted_tasks"]), list(ABSENT))
            quota = _write(root / "on/final_quota.json", [])
            quota.write_text(json.dumps(bound), encoding="utf-8")
            payload = ablation.verify_bound_manifest(quota, training_jsonl=train, expected_rows=24, epochs=5, global_batch=6)
            self.assertEqual(payload["zero_new_candidate_policy"], "skip_exhausted")


class PolicyWiringTests(unittest.TestCase):
    def test_runner_passes_the_policy_to_the_materializer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            request = dict(
                selector_command=[sys.executable, "defect_detection_ablation.py", "--max-rows", "20000"],
                previous_jsonl=None, previous_sha256=None,
                mined_jsonl=root / "mined.jsonl", current_quota_manifest=root / "current-quota.json",
                train_jsonl=root / "train.jsonl", assemble_summary=root / "assembly.json",
                final_quota_manifest=root / "quota.json", media_root=root,
                max_rows=768, row_multiple=768, epochs=5, global_batch=768,
            )
            plan = runner.build_plan(**request, zero_new_candidate_policy="skip_exhausted")
            command = plan["selector"]["command"]
            self.assertEqual(command[command.index("--zero-new-candidate-policy") + 1], "skip_exhausted")
            self.assertEqual(plan["zero_new_candidate_policy"], "skip_exhausted")
            self.assertNotIn("--zero-new-candidate-policy", plan["assembler"]["command"])
            plain = runner.build_plan(**request)
            self.assertNotIn("--zero-new-candidate-policy", plain["selector"]["command"])
            self.assertIsNone(plain["zero_new_candidate_policy"])
            with self.assertRaisesRegex(ValueError, "zero_new_candidate_policy"):
                runner.build_plan(**request, zero_new_candidate_policy="bogus")
            with self.assertRaisesRegex(ValueError, "owned by this renderer"):
                runner.build_plan(**dict(request, selector_command=[*request["selector_command"], "--zero-new-candidate-policy", "fail_closed"]),
                                  zero_new_candidate_policy="skip_exhausted")
            # without the request field the caller may still pass the flag explicitly
            explicit = runner.build_plan(**dict(request, selector_command=[*request["selector_command"], "--zero-new-candidate-policy", "fail_closed"]))
            self.assertIn("--zero-new-candidate-policy", explicit["selector"]["command"])

    def test_init_records_the_policy(self) -> None:
        from test_cosmos3_init_state_contract import Cosmos3InitStateContractTests as Base
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = Base._workspace(root)
            rc = init_deft_state.main(Base._argv(root, workspace))
            self.assertEqual(rc, 0)
            mining = json.loads((root / "results/deft_state.json").read_text())["config"]["mining"]
            self.assertEqual(mining["zero_new_candidate_policy"], "fail_closed")
            rc = init_deft_state.main(Base._argv(root / "b", workspace, "--zero-new-candidate-policy", "skip_exhausted"))
            self.assertEqual(rc, 0)
            mining = json.loads((root / "b/results/deft_state.json").read_text())["config"]["mining"]
            self.assertEqual(mining["zero_new_candidate_policy"], "skip_exhausted")
            self.assertIn("exhausted", mining["zero_new_candidate_policy_rule"])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                init_deft_state.main(Base._argv(root / "c", workspace, "--zero-new-candidate-policy", "bogus"))


if __name__ == "__main__":
    unittest.main()
