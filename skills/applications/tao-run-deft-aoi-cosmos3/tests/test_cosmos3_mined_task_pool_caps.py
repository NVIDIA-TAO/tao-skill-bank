# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Feature P5-S: launch-recorded mined per-task pool caps, fill order and Defect Detection
fraction (Phase 5-S, 2026-09-17). Caps are a fraction of each task's Mining-pool rows,
cumulative over iterations and counted over mined rows only; the fill order gives the thin
single-image tasks their rows before the filler tasks; defaults reproduce HEAD byte-for-byte."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import anchor_rows  # noqa: E402
import coverage_rows  # noqa: E402
import defect_detection_ablation as ablation  # noqa: E402
import init_deft_state  # noqa: E402
import render_iteration_mining_runner as runner  # noqa: E402
from test_cosmos3_defect_detection_ablation_contract import _candidate, _row  # noqa: E402
from test_cosmos3_guard_aware_calibration import GOLDEN, _golden_materializer_rows, _rows_sha256  # noqa: E402
from test_cosmos3_zero_new_candidate_policy import _proxy_rows, _write  # noqa: E402

DD = ablation.DEFECT_DETECTION_TASK
DC, CD, CC = "Defect Classification", "Component Detection", "Component Classification"
REF_DD, REF_DC = "Ref_based Defect Detection", "Ref_based Defect Classification"
SINGLE_IMAGE_FIRST = (DC, CD, CC)
CAPS = {DC: 0.6, DD: 0.6, CD: 0.6, CC: 0.6, REF_DD: 0.4}
ONE_BOX = [{"bbox_2d": [10, 10, 100, 100], "label": "open"}]
# HEAD (8316fe31) default selection on ``_pool()`` (computed with that snapshot's materializer on this
# exact fixture): 30 rows, Defect Detection 15 + the first 3 rows of every maintenance task.
HEAD_DEFAULT_SHA256 = "956ec8c06c44e442755b835d973a3d0753926e2dde3e2c547ea27a9b39ea0820"
HEAD_DEFAULT_BY_TASK = {CC: 3, CD: 3, DC: 3, DD: 15, REF_DC: 3, REF_DD: 3}


def _pool(*, dc_rows: int = 10, maintenance_rows: int = 10, ref_dc_rows: int = 100) -> tuple[list[dict], list[dict]]:
    """Mining pool + routed candidates: Defect Detection 15 positive / 5 empty, Defect Classification
    ``dc_rows``, Component Detection / Component Classification / Ref DD ``maintenance_rows`` (Ref DD
    pairs alternate changed / no-change), Ref_based Defect Classification ``ref_dc_rows`` (10x)."""
    rows: list[dict] = []
    candidates: list[dict] = []
    counter = [1]

    def add(row: dict, **kwargs) -> None:
        rows.append(row)
        candidates.append(_candidate(row, phash=f"{counter[0]:016x}", **kwargs))
        counter[0] += 1

    for index in range(15):
        boxes = [{"bbox_2d": [10 + index, 20, 110 + index * 10, 120 + index * 5], "label": ("open", "short")[index % 2]}]
        add(_row(f"dd-pos-{index:02d}", DD, boxes=boxes, dataset=f"source-{index % 3}"), evidence=["hard_positive_proxy_false_negative"])
    for index in range(5):
        add(_row(f"dd-empty-{index:02d}", DD, boxes=[]), evidence=["hard_negative_proxy_false_positive"])
    for task, count in ((DC, dc_rows), (CD, maintenance_rows), (CC, maintenance_rows), (REF_DD, maintenance_rows), (REF_DC, ref_dc_rows)):
        for index in range(count):
            boxes = ONE_BOX if task == REF_DD and index % 2 else None
            add(_row(f"{task.replace(' ', '_').lower()}-{index:03d}", task, boxes=boxes))
    return rows, candidates


def _materialize(rows: list[dict], candidates: list[dict], **overrides) -> tuple[list[dict], dict]:
    kwargs = dict(
        candidate_rows=candidates, source_records=rows, validation_records=[], media_root=pathlib.Path("/data"),
        max_rows=60, minimum_rows=6, row_multiple=6, defect_detection_fraction=0.5, proxy_empty_rate=1 / 3,
        epochs=1, global_batch=6, near_duplicate_hamming_distance=0,
    )
    kwargs.update(overrides)
    return ablation.materialize(**kwargs)


def _by_task(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["task_type"]] = counts.get(row["task_type"], 0) + 1
    return counts


def _usage(manifest: dict, task: str, *, pool: int, fraction: float | None, cap: int | None, before: int, now: int) -> dict:
    remaining = None if cap is None else max(0, cap - before - now)
    return {
        "pool_rows": pool, "cap_fraction": fraction, "cap_rows": cap, "used_before": before, "selected_now": now,
        "remaining_after": remaining, "capped_this_iteration": cap is not None and remaining == 0,
    }


class MinedTaskPoolCapTests(unittest.TestCase):
    def test_init_and_runner_record_the_three_options_and_reject_bad_values(self) -> None:
        from test_cosmos3_init_state_contract import Cosmos3InitStateContractTests as Base
        fill_order = ",".join(SINGLE_IMAGE_FIRST)
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = Base._workspace(root)
            # defaults: today's behaviour, launch-recorded
            self.assertEqual(init_deft_state.main(Base._argv(root / "default", workspace)), 0)
            mining = json.loads((root / "default/results/deft_state.json").read_text())["config"]["mining"]
            self.assertEqual(mining["defect_detection_fraction"], 0.5)
            self.assertEqual(mining["mined_task_pool_caps"], {})
            self.assertEqual(mining["mined_task_fill_order"], [])
            self.assertIn("cumulative", mining["mined_task_pool_caps_rule"])
            self.assertIn("render_iteration_mining_runner.py", mining["mined_task_fill_order_rule"])
            self.assertIn("--defect-detection-fraction", mining["defect_detection_fraction_rule"])
            # the Phase 5-S launch
            rc = init_deft_state.main(Base._argv(
                root / "p5s", workspace, "--defect-detection-fraction", "0.15",
                "--mined-task-pool-cap", f"{DC}=0.6", "--mined-task-pool-cap", f"{REF_DD}=0.4",
                "--mined-task-fill-order", fill_order,
            ))
            self.assertEqual(rc, 0)
            mining = json.loads((root / "p5s/results/deft_state.json").read_text())["config"]["mining"]
            self.assertEqual(mining["defect_detection_fraction"], 0.15)
            self.assertEqual(mining["mined_task_pool_caps"], {DC: 0.6, REF_DD: 0.4})
            self.assertEqual(mining["mined_task_fill_order"], list(SINGLE_IMAGE_FIRST))
            # rejected values
            bad = [
                ("--defect-detection-fraction", "0"), ("--defect-detection-fraction", "1.5"),
                ("--mined-task-pool-cap", f"{DC}=0"), ("--mined-task-pool-cap", f"{DC}=1.2"),
                ("--mined-task-pool-cap", "Bogus Task=0.5"), ("--mined-task-pool-cap", DC),
                ("--mined-task-pool-cap", f"{DC}=0.5", "--mined-task-pool-cap", f"{DC}=0.6"),
                ("--mined-task-fill-order", "Bogus Task"), ("--mined-task-fill-order", f"{DC},{DC}"),
                ("--mined-task-fill-order", DD),
            ]
            for index, extra in enumerate(bad):
                with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()) as stderr:
                    rc = init_deft_state.main(Base._argv(root / f"bad{index}", workspace, *extra))
                    self.assertEqual(rc, 2)
                    self.assertFalse((root / f"bad{index}/results/deft_state.json").exists())
                    self.assertTrue(stderr.getvalue().strip())
        # runner: the launch-recorded values reach the materializer command and the plan
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            request = dict(
                selector_command=[sys.executable, "defect_detection_ablation.py", "--max-rows", "32256"],
                previous_jsonl=None, previous_sha256=None,
                mined_jsonl=root / "mined.jsonl", current_quota_manifest=root / "current-quota.json",
                train_jsonl=root / "train.jsonl", assemble_summary=root / "assembly.json",
                final_quota_manifest=root / "quota.json", media_root=root,
                max_rows=32256, row_multiple=768, epochs=1, global_batch=768,
            )
            options = dict(defect_detection_fraction=0.15, mined_task_pool_caps={DC: 0.6, REF_DD: 0.4},
                           mined_task_fill_order=list(SINGLE_IMAGE_FIRST))
            plan = runner.build_plan(**request, **options)
            command = plan["selector"]["command"]
            self.assertEqual(command[command.index("--defect-detection-fraction") + 1], "0.15")
            caps = [command[i + 1] for i, value in enumerate(command) if value == "--mined-task-pool-cap"]
            self.assertEqual(caps, [f"{DC}=0.6", f"{REF_DD}=0.4"])
            self.assertEqual(command[command.index("--mined-task-fill-order") + 1], fill_order)
            self.assertLess(command.index("--mined-task-fill-order"), command.index("--output"))
            for flag in ("--defect-detection-fraction", "--mined-task-pool-cap", "--mined-task-fill-order"):
                self.assertNotIn(flag, plan["assembler"]["command"])
            self.assertEqual(plan["defect_detection_fraction"], 0.15)
            self.assertEqual(plan["mined_task_pool_caps"], {DC: 0.6, REF_DD: 0.4})
            self.assertEqual(plan["mined_task_fill_order"], list(SINGLE_IMAGE_FIRST))
            plain = runner.build_plan(**request)
            for flag in ("--defect-detection-fraction", "--mined-task-pool-cap", "--mined-task-fill-order"):
                self.assertNotIn(flag, plain["selector"]["command"])
            self.assertIsNone(plain["defect_detection_fraction"])
            self.assertIsNone(plain["mined_task_pool_caps"])
            self.assertIsNone(plain["mined_task_fill_order"])
            # empty caps / order are recorded but add no flag (today's command)
            empty = runner.build_plan(**request, mined_task_pool_caps={}, mined_task_fill_order=[])
            self.assertEqual(empty["selector"]["command"], plain["selector"]["command"])
            self.assertEqual((empty["mined_task_pool_caps"], empty["mined_task_fill_order"]), ({}, []))
            for field, value in (("defect_detection_fraction", 0.0), ("defect_detection_fraction", 1.5),
                                 ("mined_task_pool_caps", {"Bogus": 0.5}), ("mined_task_pool_caps", {DC: 0.0}),
                                 ("mined_task_pool_caps", [DC]), ("mined_task_fill_order", [DD]),
                                 ("mined_task_fill_order", [DC, DC]), ("mined_task_fill_order", "Defect Classification")):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    runner.build_plan(**request, **{field: value})
            for flag, field, value in (("--defect-detection-fraction", "defect_detection_fraction", 0.15),
                                       ("--mined-task-pool-cap", "mined_task_pool_caps", {DC: 0.6}),
                                       ("--mined-task-fill-order", "mined_task_fill_order", [DC])):
                with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "owned by this renderer"):
                    runner.build_plan(**dict(request, selector_command=[*request["selector_command"], flag, "x"]), **{field: value})

    def test_caps_and_single_image_fill_order_shape_the_iteration_and_preserve_the_target(self) -> None:
        rows, candidates = _pool()
        selected, manifest = _materialize(rows, candidates, defect_detection_fraction=0.15,
                                          mined_task_pool_caps=CAPS, mined_task_fill_order=SINGLE_IMAGE_FIRST)
        self.assertTrue(manifest["verified"], manifest["verification"])
        self.assertEqual(len(selected), 60)
        # DD keeps its 0.15 lower bound (9 rows); the single-image tasks fill to their caps (6 / 6 / 6),
        # Ref DD stops at its 0.4 cap (4) and Ref DC takes the remainder (29): the target is preserved
        self.assertEqual(_by_task(selected), {DD: 9, DC: 6, CD: 6, CC: 6, REF_DD: 4, REF_DC: 29})
        self.assertEqual(manifest["row_counts"]["by_task"], _by_task(selected))
        self.assertEqual(manifest["mined_task_pool_caps"], CAPS)
        self.assertEqual(manifest["mined_task_fill_order"], list(SINGLE_IMAGE_FIRST))
        self.assertEqual(manifest["mined_task_fill_realized"], {DD: 9, DC: 6, CD: 6, CC: 6, REF_DD: 4, REF_DC: 29})
        self.assertEqual(manifest["mined_task_pool_usage"], {
            DD: _usage(manifest, DD, pool=20, fraction=0.6, cap=12, before=0, now=9),
            DC: _usage(manifest, DC, pool=10, fraction=0.6, cap=6, before=0, now=6),
            CD: _usage(manifest, CD, pool=10, fraction=0.6, cap=6, before=0, now=6),
            CC: _usage(manifest, CC, pool=10, fraction=0.6, cap=6, before=0, now=6),
            REF_DD: _usage(manifest, REF_DD, pool=10, fraction=0.4, cap=4, before=0, now=4),
            REF_DC: _usage(manifest, REF_DC, pool=100, fraction=None, cap=None, before=0, now=29),
        })
        self.assertEqual(sorted(manifest["capped_tasks"]), sorted([DC, CD, CC, REF_DD]))
        self.assertTrue(manifest["verification"]["mined_task_pool_caps_respected"])
        self.assertEqual(manifest["exhausted_tasks"], {})
        self.assertEqual(manifest["zero_new_candidate_block_reason"], None)
        # the priority rows come first in the emitted order (they survive the target trim)
        self.assertEqual([row["task_type"] for row in selected[9:12]], [DC, CD, CC])
        # contrast: without caps the round-robin gives Defect Classification all 10 of its rows
        plain, plain_manifest = _materialize(rows, candidates, defect_detection_fraction=0.15)
        self.assertEqual(_by_task(plain)[DC], 10)
        self.assertEqual(plain_manifest["mined_task_pool_caps"], {})
        self.assertEqual(plain_manifest["mined_task_pool_usage"][DC]["cap_rows"], None)
        self.assertEqual(plain_manifest["capped_tasks"], [])

    def test_fill_order_never_starves_a_later_task_that_has_candidates(self) -> None:
        # Defect Classification alone could fill every maintenance slot; every other task keeps one
        # row so the presence policies stay satisfiable, then the fill order takes the rest
        rows, candidates = _pool(dc_rows=100)
        selected, manifest = _materialize(rows, candidates, defect_detection_fraction=0.15, mined_task_fill_order=(DC,))
        self.assertTrue(manifest["verified"], manifest["verification"])
        self.assertEqual(_by_task(selected), {DD: 9, DC: 47, CD: 1, CC: 1, REF_DD: 1, REF_DC: 1})
        self.assertEqual(manifest["mined_task_fill_order"], [DC])
        self.assertEqual(manifest["maintenance_tasks"]["missing"], [])

    def test_caps_are_cumulative_over_iterations_and_count_mined_rows_only(self) -> None:
        rows, candidates = _pool(maintenance_rows=20, ref_dc_rows=40)
        options = dict(max_rows=30, defect_detection_fraction=0.15, mined_task_pool_caps={DC: 0.6}, mined_task_fill_order=(DC,))
        # iteration 1 routes only 4 of the 10 Defect Classification rows: cap 6, 4 used, 2 remain
        first_candidates = [c for c in candidates if not c["filepath"].startswith("images/defect_classification")
                            or c["filepath"] in {f"images/defect_classification-{i:03d}.png" for i in range(4)}]
        first, manifest1 = _materialize(rows, first_candidates, **options)
        self.assertTrue(manifest1["verified"], manifest1["verification"])
        self.assertEqual(len(first), 30)
        self.assertEqual(_by_task(first)[DC], 4)
        self.assertEqual(manifest1["mined_task_pool_usage"][DC], _usage(manifest1, DC, pool=10, fraction=0.6, cap=6, before=0, now=4))
        self.assertEqual(manifest1["capped_tasks"], [])
        # iteration 2: the cumulative corpus also holds anchor / coverage / calibration rows of the
        # task, which do not count; 6 eligible rows remain but only the 2-row remainder is selected
        prior = list(first)
        prior += [{**_row(f"anchor-dc-{i}", DC), anchor_rows.ANCHOR_MARK: True} for i in range(2)]
        prior += [{**_row("coverage-dc-0", DC), coverage_rows.COVERAGE_MARK: True}]
        prior += [{**_row("cal-dd-0", DD, boxes=[]), ablation.CALIBRATION_MARK: True, ablation.CALIBRATION_KIND_MARK: "detection"}]
        second, manifest2 = _materialize(rows, candidates, previous_records=prior, **options)
        self.assertTrue(manifest2["verified"], manifest2["verification"])
        self.assertEqual(len(second), 30)
        self.assertEqual(_by_task(second)[DC], 2)
        self.assertEqual(manifest2["maintenance_tasks"]["eligible_after_exclusion"][DC], 6)
        self.assertEqual(manifest2["mined_task_pool_usage"][DC], _usage(manifest2, DC, pool=10, fraction=0.6, cap=6, before=4, now=2))
        self.assertTrue(manifest2["mined_task_pool_usage"][DC]["capped_this_iteration"])
        self.assertEqual(manifest2["capped_tasks"], [DC])
        self.assertEqual(manifest2["mined_task_fill_realized"][DC], 2)
        self.assertTrue(manifest2["verification"]["mined_task_pool_caps_respected"])
        self.assertNotIn(DC, manifest2["exhausted_tasks"])
        # iteration 3: the cap is consumed; the absent task is reported as capped, not exhausted.
        # fail_closed fails as for any absent task; skip_exhausted accepts it (Feature P5-S follow-up)
        prior3 = [*first, *second]
        capped_record = {"routed_candidates": 10, "eligible_after_exclusion": 4, "cap_rows": 6, "used_before": 6,
                         "selected": 0, "materialized": 0}
        third, manifest3 = _materialize(rows, candidates, previous_records=prior3, zero_new_candidate_policy="fail_closed", **options)
        self.assertNotIn(DC, _by_task(third))
        self.assertEqual(manifest3["mined_task_pool_usage"][DC], _usage(manifest3, DC, pool=10, fraction=0.6, cap=6, before=6, now=0))
        self.assertEqual(manifest3["capped_tasks"], [DC])
        self.assertEqual(manifest3["capped_absent_tasks"], {DC: capped_record})
        self.assertEqual(manifest3["maintenance_tasks"]["missing"], [DC])
        self.assertEqual(manifest3["exhausted_tasks"], {})
        self.assertFalse(manifest3["verified"])
        self.assertFalse(manifest3["verification"]["maintenance_tasks_present_or_exhausted_or_capped"])
        self.assertEqual(manifest3["zero_new_candidate_block_reason"], "policy_fail_closed")
        accepted, manifest4 = _materialize(rows, candidates, previous_records=prior3, zero_new_candidate_policy="skip_exhausted", **options)
        self.assertTrue(manifest4["verified"], manifest4["verification"])
        # 24, not 30: only one Defect Detection empty is left for the 0.15 bound (2 needed at 30 rows),
        # so the batch shrinks one multiple; a DD shortage, not the caps (shortfall.accepted)
        self.assertEqual(len(accepted), 24)
        self.assertTrue(manifest4["shortfall"]["accepted"])
        self.assertNotIn(DC, _by_task(accepted))
        self.assertEqual(manifest4["capped_absent_tasks"], {DC: capped_record})
        self.assertEqual(manifest4["capped_tasks"], [DC])
        self.assertEqual((manifest4["exhausted_tasks"], manifest4["skipped_tasks"]), ({}, []))
        self.assertTrue(manifest4["verification"]["maintenance_tasks_present_or_exhausted_or_capped"])
        self.assertFalse(manifest4["verification"]["maintenance_tasks_present_or_exhausted"])  # raw fact kept
        self.assertFalse(manifest4["verification"]["all_five_maintenance_tasks_present"])
        self.assertEqual(manifest4["verification_policy_exclusions"],
                         ["all_five_maintenance_tasks_present", "maintenance_tasks_present_or_exhausted"])
        self.assertIsNone(manifest4["zero_new_candidate_block_reason"])
        self.assertTrue(manifest4["verification"]["mined_task_pool_caps_respected"])
        # the bound v2 manifest carries the capped record and stays verified under the policy
        import assemble_training_json
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined = _write(root / "mined3.jsonl", accepted)
            bound_input = ablation.bind_manifest(manifest4, mined)
            train_rows, summary = assemble_training_json.assemble(None, mined, validation_paths=[], media_root=pathlib.Path("/data"), row_multiple=6)
            train = _write(root / "train3.jsonl", train_rows)
            summary = assemble_training_json.bind_summary(summary, train)
            bound = ablation.bind_cumulative_manifest(bound_input, current_jsonl=mined, training_jsonl=train,
                                                      assembly_summary=summary, epochs=1, global_batch=6)
            self.assertTrue(bound["verified"])
            self.assertEqual(bound["current_selection"]["capped_absent_tasks"], {DC: capped_record})
            self.assertEqual(bound["current_selection"]["capped_tasks"], [DC])

    def test_every_maintenance_task_absent_still_fails_when_caps_are_consumed(self) -> None:
        # caps of one row per maintenance task, all consumed by the prior corpus: every task is absent
        # (capped, with eligible candidates) -> skip_exhausted still fails closed
        rows, candidates = _pool(dc_rows=4, maintenance_rows=4, ref_dc_rows=4)
        caps = {task: 0.25 for task in ablation.MAINTENANCE_TASK_TYPES}
        prior = [row for row in rows if row["task_type"] != DD and row["id"].endswith("-000")]
        self.assertEqual(len(prior), 5)
        selected, manifest = _materialize(rows, candidates, previous_records=prior, max_rows=12,
                                          mined_task_pool_caps=caps, zero_new_candidate_policy="skip_exhausted")
        self.assertEqual(set(_by_task(selected)), {DD})
        self.assertEqual(sorted(manifest["capped_absent_tasks"]), sorted(ablation.MAINTENANCE_TASK_TYPES))
        self.assertEqual(manifest["exhausted_tasks"], {})
        self.assertFalse(manifest["verified"])
        self.assertFalse(manifest["verification"]["maintenance_tasks_present_or_exhausted_or_capped"])
        self.assertEqual(manifest["zero_new_candidate_block_reason"], "all_maintenance_tasks_exhausted_or_capped")
        for task in ablation.MAINTENANCE_TASK_TYPES:
            usage = manifest["mined_task_pool_usage"][task]
            self.assertEqual((usage["cap_rows"], usage["used_before"], usage["selected_now"]), (1, 1, 0))

    def test_defect_detection_fraction_moves_the_lower_bound_and_is_recorded(self) -> None:
        rows, candidates = _pool()
        # 0.5 (today): 30 of 60 rows would have to be Defect Detection, the pool holds 20 -> the
        # accepted batch shrinks to 30 rows with 15 DD; 0.15 keeps the 60-row target with 9 DD
        default, default_manifest = _materialize(rows, candidates)
        self.assertEqual((len(default), _by_task(default)[DD]), (30, 15))
        self.assertEqual(default_manifest["defect_detection_fraction"], 0.5)
        self.assertEqual(default_manifest["row_counts"]["defect_detection_minimum_target"], 15)
        low, low_manifest = _materialize(rows, candidates, defect_detection_fraction=0.15)
        self.assertTrue(low_manifest["verified"], low_manifest["verification"])
        self.assertEqual((len(low), _by_task(low)[DD]), (60, 9))
        self.assertEqual(low_manifest["defect_detection_fraction"], 0.15)
        self.assertEqual(low_manifest["configuration"]["defect_detection_minimum_fraction"], 0.15)
        self.assertEqual(low_manifest["row_counts"]["defect_detection_minimum_target"], 9)
        self.assertEqual(low_manifest["row_counts"]["defect_detection_target"], 9)
        for bad in (0.0, -0.1, 1.5):
            with self.subTest(fraction=bad), self.assertRaisesRegex(ValueError, r"\(0, 1\]"):
                _materialize(rows, candidates, defect_detection_fraction=bad)
        # CLI: the three flags parse and land in the written manifest
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = _write(root / "source.jsonl", rows)
            proxy = _write(root / "proxy.jsonl", _proxy_rows(root))
            argv = [
                "--candidate-parquet", str(root / "candidates.parquet"), "--source-annotations", str(source),
                "--proxy-annotations", str(proxy), "--media-root", "/data", "--max-rows", "60", "--minimum-rows", "6",
                "--row-multiple", "6", "--epochs", "1", "--global-batch", "6", "--near-duplicate-hamming-distance", "0",
                "--defect-detection-fraction", "0.15", "--mined-task-pool-cap", f"{DC}=0.6",
                "--mined-task-fill-order", ",".join(SINGLE_IMAGE_FIRST),
                "--output", str(root / "mined.jsonl"), "--manifest", str(root / "quota.json"),
            ]
            with mock.patch.object(ablation, "_read_parquet", return_value=candidates):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    rc = ablation.main(argv)
                self.assertEqual(rc, 0, stdout.getvalue())
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
                    rc = ablation.main(argv[:-4] + ["--mined-task-pool-cap", "Bogus=0.5", *argv[-4:]])
                self.assertEqual(rc, 2)
                self.assertIn("Bogus", stderr.getvalue())
            manifest = json.loads((root / "quota.json").read_text())
            self.assertTrue(manifest["verified"])
            self.assertEqual(manifest["defect_detection_fraction"], 0.15)
            self.assertEqual(manifest["mined_task_pool_caps"], {DC: 0.6})
            self.assertEqual(manifest["mined_task_fill_order"], list(SINGLE_IMAGE_FIRST))
            self.assertEqual(manifest["mined_task_pool_usage"][DC]["cap_rows"], 6)
            self.assertEqual(manifest["training_jsonl"]["rows"], 60)

    def test_defaults_reproduce_head_byte_for_byte(self) -> None:
        # the existing guard-aware golden fixture (hybrid calibration slot, HEAD 8316fe31)
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        rows, manifest = _golden_materializer_rows()
        self.assertEqual([r["id"] for r in rows], golden["materializer"]["ids"])
        self.assertEqual(_rows_sha256(rows), golden["materializer"]["sha256"])
        self.assertEqual(manifest["verified"], golden["materializer"]["verified"])
        self.assertEqual(manifest["defect_detection_fraction"], 0.5)
        self.assertEqual(manifest["mined_task_pool_caps"], {})
        self.assertEqual(manifest["mined_task_fill_order"], [])
        self.assertEqual(manifest["capped_tasks"], [])
        self.assertEqual(set(manifest["mined_task_pool_usage"]), {DD, *ablation.MAINTENANCE_TASK_TYPES})
        for task, usage in manifest["mined_task_pool_usage"].items():
            self.assertEqual((usage["cap_fraction"], usage["cap_rows"], usage["remaining_after"], usage["capped_this_iteration"]),
                             (None, None, None, False), task)
            self.assertEqual(usage["used_before"], 0)
        mined = [r for r in rows if r.get(ablation.CALIBRATION_MARK) is not True]
        self.assertEqual(sum(manifest["mined_task_fill_realized"].values()), len(mined))
        self.assertEqual(manifest["mined_task_fill_realized"], {**{t: 0 for t in manifest["mined_task_fill_realized"]}, **_by_task(mined)})
        self.assertEqual(manifest["mined_task_pool_usage"][DD]["selected_now"], _by_task(mined).get(DD, 0))
        self.assertTrue(manifest["verification"]["mined_task_pool_caps_respected"])
        # the Phase 5-S synthetic pool under today's defaults (proxy-rate policy, round-robin fill)
        rows, candidates = _pool()
        selected, manifest = _materialize(rows, candidates)
        self.assertEqual(len(selected), 30)
        self.assertEqual(_by_task(selected), HEAD_DEFAULT_BY_TASK)
        self.assertEqual(_rows_sha256(selected), HEAD_DEFAULT_SHA256)
        self.assertTrue(manifest["verified"])
        self.assertEqual(manifest["mined_task_pool_usage"][REF_DC]["pool_rows"], 100)


if __name__ == "__main__":
    unittest.main()
