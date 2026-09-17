# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Feature P5-S.1: launch-recorded cross-task visual de-duplication switch
(``--cross-task-visual-dedup {on,off}``, 2026-09-17). In the NVPAW pool the single-image
Defect Classification (MCQ) and Defect Detection rows are asked on the same board images, so
the materializer's cross-task exclusion let Defect Detection consume the images first and
starved Defect Classification (run v12_p5s_pool10_r8 iteration 1: 858 routed, 19 available).
``off`` keeps every within-task and record-level exclusion but no longer drops a maintenance
row because its image was selected for another task; ``on`` (default) is HEAD byte-for-byte."""

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

import defect_detection_ablation as ablation  # noqa: E402
import init_deft_state  # noqa: E402
import render_iteration_mining_runner as runner  # noqa: E402
from test_cosmos3_defect_detection_ablation_contract import _candidate, _row  # noqa: E402
from test_cosmos3_guard_aware_calibration import GOLDEN, _golden_materializer_rows, _rows_sha256  # noqa: E402
from test_cosmos3_mined_task_pool_caps import (  # noqa: E402
    CC,
    CD,
    DC,
    DD,
    HEAD_DEFAULT_BY_TASK,
    HEAD_DEFAULT_SHA256,
    ONE_BOX,
    REF_DC,
    REF_DD,
    _by_task,
    _materialize,
    _pool,
)
from test_cosmos3_zero_new_candidate_policy import _proxy_rows, _write  # noqa: E402

POSITIVE = "hard_positive_proxy_false_negative"
NEGATIVE = "hard_negative_proxy_false_positive"
MAINTENANCE = ablation.MAINTENANCE_TASK_TYPES
ZERO_UNLOCKED = {task: 0 for task in MAINTENANCE}


def _phash(seed: str) -> str:
    """A 64-bit hash that is far (in Hamming distance) from every other seed's hash."""
    return hashlib.sha256(seed.encode()).hexdigest()[:16]


def _on_image(row: dict, image: str) -> dict:
    """Point a single-image row at ``image`` (the NVPAW pool asks several tasks on one board)."""
    for item in row["messages"][0]["content"]:
        if item.get("type") == "image":
            item["image"] = image
    return row


def _image_of(row: dict) -> str:
    return [item["image"] for item in row["messages"][0]["content"] if item.get("type") == "image"][-1]


def _shared_pool(*, boards: int = 12, dd_empty: int = 4, solo_dc: int = 1, filler: int = 3) -> tuple[list[dict], list[dict]]:
    """Mining pool + routed candidates where every board image carries one Defect Detection row
    and one Defect Classification row (routed to both tasks); ``solo_dc`` Defect Classification
    rows on their own images; ``filler`` rows of every other maintenance task on their own images."""
    rows: list[dict] = []
    candidates: list[dict] = []

    def add(row: dict, **kwargs) -> None:
        rows.append(row)
        candidates.append(_candidate(row, phash=_phash(row["id"]), **kwargs))

    for index in range(boards):
        image = f"images/board-{index:03d}.png"
        empty = index < dd_empty
        boxes = [] if empty else [{"bbox_2d": [10 + index, 20, 110 + index * 10, 120 + index * 5], "label": ("open", "short")[index % 2]}]
        detection = _on_image(_row(f"dd-{index:03d}", DD, boxes=boxes, dataset=f"source-{index % 3}"), image)
        rows.append(detection)
        rows.append(_on_image(_row(f"dc-{index:03d}", DC), image))
        candidate = _candidate(detection, phash=_phash(image), evidence=[NEGATIVE if empty else POSITIVE])
        candidate["routed_task_types"] = [DD, DC]
        candidates.append(candidate)
    for index in range(solo_dc):
        add(_row(f"dc-solo-{index:03d}", DC))
    for task in (CD, CC, REF_DD, REF_DC):
        for index in range(filler):
            add(_row(f"{task.replace(' ', '_').lower()}-{index:03d}", task, boxes=ONE_BOX if task == REF_DD and index % 2 else None))
    return rows, candidates


def _shared_materialize(rows: list[dict], candidates: list[dict], **overrides) -> tuple[list[dict], dict]:
    options = dict(max_rows=30, defect_detection_fraction=0.2)
    options.update(overrides)
    return _materialize(rows, candidates, **options)


class CrossTaskVisualDedupTests(unittest.TestCase):
    def test_off_unlocks_the_maintenance_rows_that_share_images_with_defect_detection(self) -> None:
        rows, candidates = _shared_pool()
        # today (on): the 12 Defect Detection rows take the board images first, the 12 Defect
        # Classification rows on them are dropped, one solo row is left and the batch shrinks
        selected_on, on = _shared_materialize(rows, candidates)
        self.assertTrue(on["verified"], on["verification"])
        self.assertEqual(on["cross_task_visual_dedup"], "on")
        self.assertIsNone(on["maintenance_rows_unlocked_by_cross_task"])
        self.assertEqual(on["maintenance_tasks"]["routed_candidates"][DC], 13)
        self.assertEqual(on["maintenance_tasks"]["eligible_after_exclusion"][DC], 13)
        self.assertEqual(on["maintenance_marginal_quota"]["available"][DC], 1)
        self.assertEqual(_by_task(selected_on), {DD: 11, DC: 1, CD: 3, CC: 3, REF_DD: 3, REF_DC: 3})
        self.assertTrue(on["shortfall"]["accepted"])
        self.assertEqual(on["uniqueness"]["unique_images"], len(selected_on))
        self.assertEqual(on["uniqueness"]["visual_identity_scope"], "all_tasks")
        # off: every routed Defect Classification image is available, the unlocked rows are
        # counted, the target is reached and the manifest verifies within each task type
        selected_off, off = _shared_materialize(rows, candidates, cross_task_visual_dedup="off")
        self.assertTrue(off["verified"], off["verification"])
        self.assertEqual(off["cross_task_visual_dedup"], "off")
        self.assertEqual(off["maintenance_marginal_quota"]["available"][DC], off["maintenance_tasks"]["routed_candidates"][DC])
        self.assertEqual(off["maintenance_marginal_quota"]["available"][DC], 13)
        self.assertEqual(off["maintenance_rows_unlocked_by_cross_task"], {**ZERO_UNLOCKED, DC: 12})
        self.assertEqual(len(selected_off), 30)
        self.assertFalse(off["shortfall"]["accepted"])
        self.assertEqual(_by_task(selected_off), {DD: 6, DC: 12, CD: 3, CC: 3, REF_DD: 3, REF_DC: 3})
        self.assertEqual(off["uniqueness"]["visual_identity_scope"], "within_task")
        for key in ("unique_target_images", "unique_image_content", "near_duplicate_free"):
            self.assertTrue(off["verification"][key], key)
        # the emitted rows really share images across the two tasks (the informational counts show it)
        detection_images = {_image_of(row) for row in selected_off if row["task_type"] == DD}
        shared = sum(_image_of(row) in detection_images for row in selected_off if row["task_type"] == DC)
        self.assertGreaterEqual(shared, 5)
        self.assertEqual(off["uniqueness"]["unique_images"], len(selected_off) - shared)
        self.assertEqual(off["uniqueness"]["unique_image_content"], len(selected_off) - shared)
        self.assertEqual(off["uniqueness"]["selected_near_duplicate_pairs"], 0)
        # within-task identity is untouched: no task holds two rows on one image
        for task in ablation.ALL_TASK_TYPES:
            images = [_image_of(row) for row in selected_off if row["task_type"] == task]
            self.assertEqual(len(images), len(set(images)), task)
        # the exclusion ledger only differs by the unlocked rows
        self.assertEqual(on["uniqueness"]["exact_duplicates_excluded"], 12)
        self.assertEqual(off["uniqueness"].get("exact_duplicates_excluded", 0), 0)
        # the assembler accepts the shared images and the bound v2 manifest carries the switch
        import assemble_training_json
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined = _write(root / "mined.jsonl", selected_off)
            bound_input = ablation.bind_manifest(off, mined)
            train_rows, summary = assemble_training_json.assemble(None, mined, validation_paths=[], media_root=pathlib.Path("/data"), row_multiple=6)
            self.assertEqual(len(train_rows), 30)
            train = _write(root / "train.jsonl", train_rows)
            summary = assemble_training_json.bind_summary(summary, train)
            bound = ablation.bind_cumulative_manifest(bound_input, current_jsonl=mined, training_jsonl=train,
                                                      assembly_summary=summary, epochs=1, global_batch=6)
            self.assertTrue(bound["verified"])
            self.assertEqual(bound["current_selection"]["cross_task_visual_dedup"], "off")
            self.assertEqual(bound["current_selection"]["maintenance_rows_unlocked_by_cross_task"], {**ZERO_UNLOCKED, DC: 12})

    def test_within_task_duplicates_are_still_removed_when_off(self) -> None:
        rows, candidates = _shared_pool()
        # a second Defect Classification record on board 0 (exact duplicate within the task)
        rows.append(_on_image(_row("dc-dup-000", DC), "images/board-000.png"))
        # two Component Classification rows whose perceptual hashes differ by one bit
        near_a, near_b = _row("cc-near-a", CC), _row("cc-near-b", CC)
        rows.extend([near_a, near_b])
        base = _phash("cc-near")
        candidates.append(_candidate(near_a, phash=base))
        candidates.append(_candidate(near_b, phash=f"{int(base, 16) ^ 1:016x}"))
        for mode in ("on", "off"):
            with self.subTest(mode=mode):
                selected, manifest = _shared_materialize(rows, candidates, cross_task_visual_dedup=mode, near_duplicate_hamming_distance=1)
                self.assertTrue(manifest["verified"], manifest["verification"])
                self.assertEqual(manifest["maintenance_tasks"]["eligible_after_exclusion"][DC], 14)
                self.assertEqual(manifest["maintenance_marginal_quota"]["available"][DC], 1 if mode == "on" else 13)
                self.assertEqual(manifest["maintenance_marginal_quota"]["available"][CC], 4)
                self.assertGreaterEqual(manifest["uniqueness"]["near_duplicates_excluded"], 1)
                dc_images = [_image_of(row) for row in selected if row["task_type"] == DC]
                self.assertEqual(len(dc_images), len(set(dc_images)))
                self.assertLessEqual(sum(row["id"] in {"cc-near-a", "cc-near-b"} for row in selected), 1)
        _, off = _shared_materialize(rows, candidates, cross_task_visual_dedup="off", near_duplicate_hamming_distance=1)
        # the duplicate record is not "unlocked": it would be dropped under both modes
        self.assertEqual(off["maintenance_rows_unlocked_by_cross_task"], {**ZERO_UNLOCKED, DC: 12})
        self.assertEqual(off["uniqueness"]["exact_duplicates_excluded"], 1)

    def test_previous_record_and_leakage_exclusions_are_unchanged_when_off(self) -> None:
        rows, candidates = _shared_pool()
        by_id = {row["id"]: row for row in rows}
        previous = [by_id["dd-000"], by_id["dc-001"]]
        leaked = _on_image(_row("proxy-shared", DD, boxes=[]), "images/board-002.png")
        results = {
            mode: _shared_materialize(rows, candidates, cross_task_visual_dedup=mode, previous_records=previous, validation_records=[leaked])
            for mode in ("on", "off")
        }
        for mode, (selected, manifest) in results.items():
            with self.subTest(mode=mode):
                self.assertTrue(manifest["verified"], manifest["verification"])
                self.assertEqual(manifest["previous_records_excluded"], 2)
                self.assertEqual(manifest["uniqueness"]["benchmark_or_proxy_leakage_excluded"], 1)
                self.assertEqual(manifest["maintenance_tasks"]["routed_candidates"][DC], 13)
                self.assertEqual(manifest["maintenance_tasks"]["eligible_after_exclusion"][DC], 11)
                ids = {row["id"] for row in selected}
                self.assertFalse(ids & {"dd-000", "dc-001", "dd-002", "dc-002"})
                self.assertNotIn("images/board-002.png", {_image_of(row) for row in selected})
        _, on = results["on"]
        _, off = results["off"]
        # on: the row of board 0 survives only because its Defect Detection row is history
        self.assertEqual(on["maintenance_marginal_quota"]["available"][DC], 2)
        self.assertEqual(off["maintenance_marginal_quota"]["available"][DC], 11)
        self.assertEqual(off["maintenance_rows_unlocked_by_cross_task"], {**ZERO_UNLOCKED, DC: 9})

    def test_off_keeps_the_guard_aware_calibration_few_box_rows_on_shared_images(self) -> None:
        """Guard-aware calibration (Feature B3) re-selects the few-box slot against the rows kept so
        far; under ``off`` that exclusion is also same-task, so a Defect Classification row on a
        few-box board no longer evicts the few-box calibration row."""
        source: list[dict] = []
        candidates: list[dict] = []

        def add(row: dict, **kwargs) -> dict:
            source.append(row)
            candidate = _candidate(row, phash=_phash(row["id"]), **kwargs)
            candidates.append(candidate)
            return candidate

        for index in range(2):
            add(_row(f"cal-e-{index}", DD, boxes=[], dataset="pool"), evidence=[ablation.CALIBRATION_EMPTY_EVIDENCE], route_tier="calibration")
        for index in range(4):
            image = f"images/fewbox-{index}.png"
            fewbox = _on_image(_row(f"cal-f-{index}", DD, boxes=[{"bbox_2d": [index, 0, index + 4, 6], "label": "open"}], dataset=f"pool{index % 2}"), image)
            source.append(fewbox)
            source.append(_on_image(_row(f"dc-f-{index}", DC), image))
            candidate = _candidate(fewbox, phash=_phash(image), evidence=[ablation.CALIBRATION_FEW_EVIDENCE], route_tier="calibration")
            candidate["routed_task_types"] = [DD, DC]
            candidate["route_tiers"] = ["calibration", "strict"]
            candidate["route_tier_by_task"] = {DD: "calibration", DC: "strict"}
            candidates.append(candidate)
        add(_row("ref-e", REF_DD, boxes=[], dataset="pool"), evidence=[ablation.CALIBRATION_EMPTY_EVIDENCE, ablation.REFERENCE_NO_CHANGE_EVIDENCE], route_tier="calibration")
        add(_row("ref-c", REF_DD, boxes=ONE_BOX, dataset="pool"), evidence=[ablation.CALIBRATION_FEW_EVIDENCE], route_tier="calibration")
        for index in range(2):
            add(_row(f"dd-s-{index}", DD, boxes=[{"bbox_2d": [index, index, index + 9, index + 9], "label": "short"}], dataset="mine"), evidence=[POSITIVE])
        add(_row("ref-m", REF_DD, boxes=ONE_BOX, dataset="mine"))
        for task in (CC, CD, REF_DC):
            add(_row(task.replace(" ", "_").lower(), task))
        previous = [_row(f"prev-dd-e-{index}", DD, boxes=[]) for index in range(20)]
        options = dict(
            candidate_rows=candidates, source_records=source, previous_records=previous, validation_records=[],
            media_root=pathlib.Path("/data"), max_rows=24, minimum_rows=4, row_multiple=4, defect_detection_fraction=0.5,
            proxy_empty_rate=0.25, epochs=1, global_batch=4, near_duplicate_hamming_distance=None,
            reference_proxy_empty_rate=0.5, single_image_calibration_max_empty=2, single_image_calibration_max_few=2,
            reference_calibration_total=2, empty_answer_guard=ablation.validate_empty_answer_guard_config(0.30, None, None, None),
            calibration_guard_aware=True, zero_new_candidate_policy="skip_exhausted",
        )
        # today: the Defect Classification rows on the two few-box boards the first pass did not pick
        # survive the cross-task exclusion, so the re-selection can only keep two few-box rows
        _, on = ablation.materialize(**options)
        self.assertTrue(on["verified"], on["verification"])
        self.assertEqual(on["calibration_empty_headroom"]["overall"], 0)
        self.assertEqual(on["single_image_calibration"]["selected_few_box"], 2)
        self.assertEqual(on["single_image_calibration"]["selected_empty"], 2)
        self.assertEqual(on["guard_aware_calibration"]["fewbox_shortfall"][DD], 2)
        self.assertEqual(on["calibration_headroom_overflow_rows"][DD], 2)
        # off: all four few-box rows fill the slot; the shared Defect Classification rows stay too
        selected, off = ablation.materialize(**options, cross_task_visual_dedup="off")
        self.assertTrue(off["verified"], off["verification"])
        self.assertEqual(off["single_image_calibration"]["selected_few_box"], 4)
        self.assertEqual(off["single_image_calibration"]["selected_empty"], 0)
        self.assertEqual(off["single_image_calibration"]["empty_substituted_by_few_box"], 2)
        self.assertEqual(off["guard_aware_calibration"]["fewbox_shortfall"][DD], 0)
        self.assertEqual(off["calibration_headroom_overflow_rows"][DD], 0)
        # the remaining overflow is the reference slot's (one changed pair for two slot rows), both modes
        for manifest in (on, off):
            self.assertEqual(manifest["calibration_headroom_overflow_rows"][REF_DD], 1)
            self.assertEqual(manifest["guard_aware_calibration"]["status"], "substituted_with_overflow")
        self.assertEqual(off["maintenance_rows_unlocked_by_cross_task"], {**ZERO_UNLOCKED, DC: 2})
        fewbox_images = {_image_of(row) for row in selected if row["task_type"] == DD and row.get(ablation.CALIBRATION_MARK)}
        self.assertEqual(len(fewbox_images), 4)
        self.assertTrue(any(_image_of(row) in fewbox_images for row in selected if row["task_type"] == DC))

    def test_default_on_reproduces_head_byte_for_byte_and_off_is_inert_without_shared_images(self) -> None:
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        rows, manifest = _golden_materializer_rows()
        self.assertEqual(_rows_sha256(rows), golden["materializer"]["sha256"])
        self.assertEqual(manifest["cross_task_visual_dedup"], "on")
        self.assertIsNone(manifest["maintenance_rows_unlocked_by_cross_task"])
        self.assertEqual(manifest["uniqueness"]["visual_identity_scope"], "all_tasks")
        pool_rows, candidates = _pool()
        for overrides in ({}, {"cross_task_visual_dedup": "on"}):
            with self.subTest(overrides=overrides):
                selected, manifest = _materialize(pool_rows, candidates, **overrides)
                self.assertEqual(_by_task(selected), HEAD_DEFAULT_BY_TASK)
                self.assertEqual(_rows_sha256(selected), HEAD_DEFAULT_SHA256)
                self.assertEqual(manifest["cross_task_visual_dedup"], "on")
                self.assertIsNone(manifest["maintenance_rows_unlocked_by_cross_task"])
        # no shared images in the Phase 5-S synthetic pool: off selects the same rows and unlocks nothing
        selected, manifest = _materialize(pool_rows, candidates, cross_task_visual_dedup="off")
        self.assertEqual(_rows_sha256(selected), HEAD_DEFAULT_SHA256)
        self.assertTrue(manifest["verified"], manifest["verification"])
        self.assertEqual(manifest["cross_task_visual_dedup"], "off")
        self.assertEqual(manifest["maintenance_rows_unlocked_by_cross_task"], ZERO_UNLOCKED)
        for bad in ("ON", "true", "", None, 1):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "cross-task visual de-duplication"):
                _materialize(pool_rows, candidates, cross_task_visual_dedup=bad)
        self.assertIn("cross_task_visual_dedup", ablation.CURRENT_SELECTION_COPIED_KEYS)
        self.assertIn("maintenance_rows_unlocked_by_cross_task", ablation.CURRENT_SELECTION_COPIED_KEYS)

    def test_init_runner_and_cli_record_the_switch(self) -> None:
        from test_cosmos3_init_state_contract import Cosmos3InitStateContractTests as Base
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = Base._workspace(root)
            self.assertEqual(init_deft_state.main(Base._argv(root / "default", workspace)), 0)
            mining = json.loads((root / "default/results/deft_state.json").read_text())["config"]["mining"]
            self.assertEqual(mining["cross_task_visual_dedup"], "on")
            for needle in ("--cross-task-visual-dedup", "render_iteration_mining_runner.py", "same task", "maintenance_rows_unlocked_by_cross_task"):
                self.assertIn(needle, mining["cross_task_visual_dedup_rule"])
            self.assertEqual(init_deft_state.main(Base._argv(root / "off", workspace, "--cross-task-visual-dedup", "off")), 0)
            mining = json.loads((root / "off/results/deft_state.json").read_text())["config"]["mining"]
            self.assertEqual(mining["cross_task_visual_dedup"], "off")
            with contextlib.redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit) as exit_info:
                init_deft_state.main(Base._argv(root / "bad", workspace, "--cross-task-visual-dedup", "maybe"))
            self.assertEqual(exit_info.exception.code, 2)
            self.assertIn("--cross-task-visual-dedup", stderr.getvalue())
            self.assertFalse((root / "bad/results/deft_state.json").exists())
        # runner: the request field is renderer-owned, forwarded to the materializer only, and recorded
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
            for mode in ("on", "off"):
                plan = runner.build_plan(**request, cross_task_visual_dedup=mode)
                command = plan["selector"]["command"]
                self.assertEqual(command[command.index("--cross-task-visual-dedup") + 1], mode)
                self.assertLess(command.index("--cross-task-visual-dedup"), command.index("--output"))
                self.assertNotIn("--cross-task-visual-dedup", plan["assembler"]["command"])
                self.assertEqual(plan["cross_task_visual_dedup"], mode)
            plain = runner.build_plan(**request)
            self.assertNotIn("--cross-task-visual-dedup", plain["selector"]["command"])
            self.assertIsNone(plain["cross_task_visual_dedup"])
            for bad in ("maybe", True, 0):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    runner.build_plan(**request, cross_task_visual_dedup=bad)
            with self.assertRaisesRegex(ValueError, "owned by this renderer"):
                runner.build_plan(**dict(request, selector_command=[*request["selector_command"], "--cross-task-visual-dedup", "on"]), cross_task_visual_dedup="off")
            # the caller keeps ownership when the request leaves the field unset
            owned = runner.build_plan(**dict(request, selector_command=[*request["selector_command"], "--cross-task-visual-dedup", "off"]))
            self.assertIn("off", owned["selector"]["command"])
        # materializer CLI: the flag parses, lands in the written manifest and rejects other values
        rows, candidates = _shared_pool()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = _write(root / "source.jsonl", rows)
            proxy = _write(root / "proxy.jsonl", _proxy_rows(root))
            argv = [
                "--candidate-parquet", str(root / "candidates.parquet"), "--source-annotations", str(source),
                "--proxy-annotations", str(proxy), "--media-root", "/data", "--max-rows", "30", "--minimum-rows", "6",
                "--row-multiple", "6", "--epochs", "1", "--global-batch", "6", "--near-duplicate-hamming-distance", "0",
                "--defect-detection-fraction", "0.2", "--cross-task-visual-dedup", "off",
                "--output", str(root / "mined.jsonl"), "--manifest", str(root / "quota.json"),
            ]
            with mock.patch.object(ablation, "_read_parquet", return_value=candidates):
                with contextlib.redirect_stdout(io.StringIO()) as stdout:
                    rc = ablation.main(argv)
                self.assertEqual(rc, 0, stdout.getvalue())
                with contextlib.redirect_stderr(io.StringIO()) as stderr, contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                    ablation.main([*argv[:-5], "maybe", *argv[-4:]])
                self.assertIn("--cross-task-visual-dedup", stderr.getvalue())
            manifest = json.loads((root / "quota.json").read_text())
            self.assertTrue(manifest["verified"])
            self.assertEqual(manifest["cross_task_visual_dedup"], "off")
            self.assertEqual(manifest["maintenance_rows_unlocked_by_cross_task"], {**ZERO_UNLOCKED, DC: 12})
            self.assertEqual(manifest["training_jsonl"]["rows"], 30)
            self.assertEqual(manifest["row_counts"]["by_task"][DC], 12)


if __name__ == "__main__":
    unittest.main()
