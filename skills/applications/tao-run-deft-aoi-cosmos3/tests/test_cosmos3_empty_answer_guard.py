# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Empty-answer guard (Phase 4 step 4c-B): answer profile, caps, trim order, pre-alignment trimming."""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
from collections import Counter


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import anchor_rows  # noqa: E402
import assemble_training_json  # noqa: E402
import init_deft_state  # noqa: E402
import render_iteration_mining_runner  # noqa: E402
from test_cosmos3_classification_calibration import BCQ_PROMPT, _det_row, _row as _mcq_row, _write  # noqa: E402

CAL = assemble_training_json.CALIBRATION_MARK
KIND = assemble_training_json.CALIBRATION_KIND_MARK


def _ref_det_row(record_id: str, boxes: list[dict] | None = None, dataset: str = "mine") -> dict:
    return {
        "id": record_id,
        "task_type": "Ref_based Defect Detection",
        "dataset": dataset,
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": f"images/{dataset}/{record_id}-golden.png", "min_pixels": 1, "max_pixels": 1},
                {"type": "image", "image": f"images/{dataset}/{record_id}.png", "min_pixels": 1, "max_pixels": 1},
                {"type": "text", "text": "Locate defects in Image 2. If no visible defect is present in Image 2, return []."},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps(boxes or [])}]},
        ],
    }


def _bcq_row(record_id: str, answer: str = "B. No, this image does not contain any defects.") -> dict:
    return _mcq_row(record_id, "Defect Classification", prompt=BCQ_PROMPT, answer=answer)


def _cal_empty(record_id: str) -> dict:
    """A detection calibration negative as the materializer emits it (both markers)."""
    return {**_det_row(record_id, dataset="cal"), CAL: True, KIND: "detection"}


BOX = [{"bbox_2d": [0, 0, 10, 10], "label": "x"}]


def _guard(**kwargs) -> dict:
    return assemble_training_json.validate_empty_answer_guard_config(
        kwargs.get("overall"), kwargs.get("per_task"), kwargs.get("classification"), kwargs.get("mode")
    )


class AnswerProfileSummaryTests(unittest.TestCase):
    def test_answer_profile_counts_task_format_images_and_new_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined_rows = [_det_row(f"dd-e-{i}") for i in range(4)] + [_det_row(f"dd-p-{i}", BOX) for i in range(2)]
            mined_rows += [_mcq_row(f"mcq-{i}", answer="A") for i in range(2)] + [_mcq_row("mcq-empty", answer="[]")]
            mined_rows += [_bcq_row(f"bcq-{i}") for i in range(2)]
            mined_rows += [_ref_det_row("ref-e"), _ref_det_row("ref-p", BOX)]
            mined = _write(root / "mined.jsonl", mined_rows)
            rows, summary = assemble_training_json.assemble(None, mined, validation_paths=[], media_root=root)
            profile = summary["answer_profile"]
            self.assertEqual((profile["rows"], profile["empty_rows"]), (13, 6))
            self.assertAlmostEqual(profile["empty_share"], 6 / 13)
            cells = profile["by_task_format_images"]
            self.assertEqual(cells["Defect Detection|DET|1"]["rows"], 6)
            self.assertEqual(cells["Defect Detection|DET|1"]["empty_rows"], 4)
            self.assertAlmostEqual(cells["Defect Detection|DET|1"]["empty_share"], 4 / 6)
            self.assertEqual(cells["Defect Classification|MCQ|1"], {"task_type": "Defect Classification", "format": "MCQ", "images": 1, "rows": 3, "empty_rows": 1, "empty_share": 1 / 3})
            self.assertEqual(cells["Defect Classification|BCQ|1"]["empty_rows"], 0)  # "No" is not empty
            self.assertEqual(cells["Ref_based Defect Detection|DET|2"], {"task_type": "Ref_based Defect Detection", "format": "DET", "images": 2, "rows": 2, "empty_rows": 1, "empty_share": 0.5})
            self.assertEqual(profile["by_task"]["Defect Classification"]["rows"], 5)
            self.assertEqual(profile["classification"]["rows"], 5)
            self.assertEqual(profile["classification"]["empty_rows"], 1)
            self.assertEqual(profile["classification"]["by_task"]["Defect Classification"]["empty_rows"], 1)
            self.assertEqual(summary["answer_profile_new_rows"], profile)
            guard = summary["empty_answer_guard"]
            self.assertFalse(guard["enabled"])
            self.assertEqual(guard["status"], "within_caps")
            self.assertEqual(guard["rows_trimmed_total"], 0)
            self.assertEqual(guard["passes"], 1)
            previous = _write(root / "train1.jsonl", rows)
            mined2 = _write(root / "mined2.jsonl", [_det_row(f"n-{i}", BOX) for i in range(3)] + [_det_row("n-empty")])
            _, summary2 = assemble_training_json.assemble(
                previous, mined2, previous_sha256=assemble_training_json.sha256_file(previous), validation_paths=[], media_root=root
            )
            self.assertEqual(summary2["answer_profile"]["rows"], 17)
            self.assertEqual(summary2["answer_profile"]["empty_rows"], 7)
            self.assertEqual((summary2["answer_profile_new_rows"]["rows"], summary2["answer_profile_new_rows"]["empty_rows"]), (4, 1))


class EmptyAnswerGuardTrimTests(unittest.TestCase):
    def _anchor_assets(self, root: pathlib.Path, *, empty_anchors: bool):
        if empty_anchors:
            anchors = [_det_row(f"a-{i}", dataset=f"pool{i % 2}") for i in range(200)]
        else:
            anchors = [_mcq_row(f"a-{i}", "Component Classification", answer="A", dataset=f"pool{i % 2}") for i in range(200)]
        source = _write(root / "anchor_candidates.jsonl", anchors)
        kpi = _write(root / "kpi.jsonl", [_det_row(f"k-{i}") for i in range(50)] if empty_anchors
                     else [_mcq_row(f"k-{i}", "Component Classification", answer="A") for i in range(50)])
        return anchor_rows.validate_anchor_config(0.10, source, kpi, None)

    def test_overall_cap_trims_calibration_negatives_before_mined_empties_and_never_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined_rows = [_det_row(f"pos-{i}", BOX) for i in range(20)]
            mined_rows += [_det_row(f"mined-empty-{i}") for i in range(10)]
            mined_rows += [_cal_empty(f"cal-empty-{i}") for i in range(10)]
            mined = _write(root / "mined.jsonl", mined_rows)
            cfg = self._anchor_assets(root, empty_anchors=True)
            for mode in ("enforce", None):
                rows, summary = assemble_training_json.assemble(
                    None, mined, validation_paths=[], media_root=root, anchor_config=cfg,
                    empty_answer_guard=_guard(overall=0.30, mode=mode),
                )
                guard = summary["empty_answer_guard"]
                self.assertTrue(guard["enabled"])
                self.assertEqual(guard["mode"], "enforce")
                self.assertEqual(guard["caps"]["overall"], 0.30)
                # 40 current rows + 4 anchors (all empty): 24 / 44 before; all 10 calibration negatives go
                # first, then 6 mined empties -> 8 / 28 <= 0.30; the 4 empty anchors are untouched.
                self.assertEqual((guard["before"]["rows"], guard["before"]["empty_rows"]), (44, 24))
                self.assertAlmostEqual(guard["before"]["overall_share"], 24 / 44)
                self.assertEqual(guard["exceeded_before"], ["overall"])
                self.assertEqual(guard["rows_trimmed_by_source"], {"detection_calibration_negative": 10, "mined_empty": 6})
                self.assertEqual(guard["rows_trimmed_by_task"], {"Defect Detection": 16})
                self.assertEqual(guard["rows_trimmed_total"], 16)
                self.assertEqual((guard["after"]["rows"], guard["after"]["empty_rows"]), (28, 8))
                self.assertAlmostEqual(guard["after"]["overall_share"], 8 / 28)
                self.assertEqual(guard["exceeded_after"], [])
                self.assertEqual(guard["status"], "trimmed_to_caps")
                self.assertEqual((guard["aligned_rows_before"], guard["aligned_rows_after"], guard["backfilled_rows"]), (44, 28, 0))
                self.assertEqual(guard["passes"], 2)
                self.assertEqual(len(rows), 28)
                self.assertEqual(summary["materialized_anchor_records"], 4)
                self.assertEqual(sum(r.get("deft_anchor") is True for r in rows), 4)
                self.assertEqual(summary["materialized_calibration_records"], 0)
                self.assertEqual(sum(r["id"].startswith("mined-empty") for r in rows), 4)
                self.assertEqual(summary["materialized_current_records"], 24)
                self.assertEqual(summary["records_truncated"], 16)
                self.assertEqual(summary["answer_profile"]["empty_rows"], 8)
                self.assertEqual(summary["answer_profile_new_rows"]["rows"], 28)

    def test_per_task_cap_trims_only_that_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined_rows = [_det_row(f"dd-e-{i}") for i in range(6)] + [_det_row(f"dd-p-{i}", BOX) for i in range(4)]
            mined_rows += [_ref_det_row(f"ref-e-{i}") for i in range(5)] + [_ref_det_row(f"ref-p-{i}", BOX) for i in range(5)]
            mined = _write(root / "mined.jsonl", mined_rows)
            rows, summary = assemble_training_json.assemble(
                None, mined, validation_paths=[], media_root=root,
                empty_answer_guard=_guard(per_task={"Defect Detection": 0.45}),
            )
            guard = summary["empty_answer_guard"]
            self.assertEqual(guard["caps"]["per_task"], {"Defect Detection": 0.45})
            self.assertIsNone(guard["caps"]["overall"])
            self.assertEqual(guard["exceeded_before"], ["task:Defect Detection"])
            self.assertEqual(guard["rows_trimmed_by_task"], {"Defect Detection": 3})
            self.assertEqual(guard["rows_trimmed_by_source"], {"mined_empty": 3})
            self.assertAlmostEqual(guard["after"]["per_task"]["Defect Detection"], 3 / 7)
            self.assertAlmostEqual(guard["after"]["per_task"]["Ref_based Defect Detection"], 0.5)
            self.assertAlmostEqual(guard["before"]["per_task"]["Defect Detection"], 0.6)
            self.assertEqual(len(rows), 17)
            self.assertEqual(sum(r["task_type"] == "Ref_based Defect Detection" for r in rows), 10)
            self.assertEqual(guard["status"], "trimmed_to_caps")

    def test_classification_cap_applies_to_the_union_and_to_each_classification_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined_rows = [_mcq_row(f"cc-{i}", "Component Classification", answer="A") for i in range(7)]
            mined_rows += [_mcq_row(f"cc-empty-{i}", "Component Classification", answer="[]") for i in range(3)]
            mined_rows += [_mcq_row(f"dc-{i}", answer="B") for i in range(10)]
            mined_rows += [_det_row(f"dd-e-{i}") for i in range(5)] + [_det_row(f"dd-p-{i}", BOX) for i in range(5)]
            mined = _write(root / "mined.jsonl", mined_rows)
            rows, summary = assemble_training_json.assemble(
                None, mined, validation_paths=[], media_root=root,
                empty_answer_guard=_guard(classification=0.10),
            )
            guard = summary["empty_answer_guard"]
            self.assertAlmostEqual(guard["before"]["classification_share"], 3 / 20)
            self.assertEqual(guard["exceeded_before"], ["classification", "classification_task:Component Classification"])
            # 2 removals satisfy the union (1/18) but Component Classification is still 1/8 -> the third goes too
            self.assertEqual(guard["rows_trimmed_total"], 3)
            self.assertEqual(guard["rows_trimmed_by_task"], {"Component Classification": 3})
            self.assertEqual(guard["after"]["classification_share"], 0.0)
            self.assertEqual(guard["after"]["classification_per_task"]["Component Classification"], 0.0)
            self.assertEqual(sum(r["id"].startswith("dd-e") for r in rows), 5)  # detection empties are not a classification matter
            self.assertEqual(len(rows), 27)

    def test_report_mode_never_trims_and_records_exceeded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined_rows = [_det_row(f"pos-{i}", BOX) for i in range(5)] + [_det_row(f"empty-{i}") for i in range(5)]
            mined = _write(root / "mined.jsonl", mined_rows)
            rows, summary = assemble_training_json.assemble(
                None, mined, validation_paths=[], media_root=root,
                empty_answer_guard=_guard(overall=0.30, mode="report"),
            )
            guard = summary["empty_answer_guard"]
            self.assertEqual(guard["mode"], "report")
            self.assertEqual(guard["status"], "exceeded")
            self.assertEqual(guard["rows_trimmed_total"], 0)
            self.assertEqual(guard["exceeded_after"], ["overall"])
            self.assertEqual(len(rows), 10)
            out = root / "train.jsonl"
            rc = assemble_training_json.main([
                "--mined-jsonl", str(mined), "--output", str(out), "--media-root", str(root),
                "--max-empty-answer-share", "0.30", "--empty-answer-guard-mode", "report",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue(out.is_file())
            written = json.loads((root / "assemble_summary.json").read_text())
            self.assertEqual(written["empty_answer_guard"]["status"], "exceeded")

    def test_exceeded_after_trimming_fails_closed_in_enforce_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            previous = _write(root / "train0.jsonl", [_det_row(f"old-empty-{i}") for i in range(30)])
            mined = _write(root / "mined.jsonl", [_det_row(f"new-{i}", BOX) for i in range(10)])
            rows, summary = assemble_training_json.assemble(
                previous, mined, previous_sha256=assemble_training_json.sha256_file(previous),
                validation_paths=[], media_root=root, empty_answer_guard=_guard(overall=0.30),
            )
            guard = summary["empty_answer_guard"]
            self.assertEqual(guard["status"], "exceeded")
            self.assertEqual(guard["rows_trimmed_total"], 0)  # previous rows are never trimmed
            self.assertEqual(len(rows), 40)
            out = root / "train.jsonl"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = assemble_training_json.main([
                    "--mined-jsonl", str(mined), "--previous-jsonl", str(previous),
                    "--previous-sha256", assemble_training_json.sha256_file(previous),
                    "--output", str(out), "--media-root", str(root), "--max-empty-answer-share", "0.30",
                ])
            self.assertEqual(rc, 2)
            self.assertIn("empty-answer guard", stderr.getvalue())
            self.assertFalse(out.exists())
            written = json.loads((root / "assemble_summary.json").read_text())
            self.assertEqual(written["empty_answer_guard"]["status"], "exceeded")
            self.assertEqual(written["empty_answer_guard"]["mode"], "enforce")

    def test_trimming_happens_before_alignment_and_backfills_from_spare_candidates(self) -> None:
        # 52 candidates (40 non-empty mined + 12 detection calibration negatives), aligned to 48:
        # the cap step drops 4 non-empty mined rows. Trimming 3 calibration negatives before the
        # alignment lets the standard step back-fill those 3 slots from the spare mined rows, so
        # the aligned corpus stays 48 rows and its empty share drops from 12/48 to 9/48.
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined_rows = [_det_row(f"pos-{i}", BOX) for i in range(40)] + [_cal_empty(f"cal-empty-{i}") for i in range(12)]
            mined = _write(root / "mined.jsonl", mined_rows)
            rows, summary = assemble_training_json.assemble(
                None, mined, validation_paths=[], media_root=root, max_rows=64, row_multiple=8,
                empty_answer_guard=_guard(overall=0.20),
            )
            guard = summary["empty_answer_guard"]
            self.assertEqual(len(rows), 48)
            self.assertEqual((guard["aligned_rows_before"], guard["aligned_rows_after"]), (48, 48))
            self.assertEqual(guard["aligned_rows_shrunk"], 0)
            self.assertEqual(guard["backfilled_rows"], 3)
            self.assertEqual(guard["rows_trimmed_total"], 3)
            self.assertEqual(guard["rows_trimmed_by_source"], {"detection_calibration_negative": 3})
            self.assertEqual(guard["rows_trimmed_by_task"], {"Defect Detection": 3})
            self.assertAlmostEqual(guard["before"]["overall_share"], 12 / 48)
            self.assertAlmostEqual(guard["after"]["overall_share"], 9 / 48)
            self.assertEqual(guard["status"], "trimmed_to_caps")
            self.assertEqual(guard["passes"], 2)
            self.assertEqual(summary["materialized_calibration_records"], 9)
            self.assertEqual(sum(r["id"].startswith("pos-") for r in rows), 39)
            self.assertEqual(len({r["id"] for r in rows}), 48)
            self.assertEqual(summary["records_truncated"], 4)  # 52 candidates -> 48 rows

    def test_anchor_reservation_is_resolved_on_the_trimmed_candidates(self) -> None:
        # 20 non-empty mined + 12 calibration negatives + 4 anchors -> 32 aligned (17 mined, 12 cal, 3 anchors).
        # Trimming 4 negatives frees slots that the remaining mined rows and a spare anchor back-fill.
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined = _write(root / "mined.jsonl", [_det_row(f"pos-{i}", BOX) for i in range(20)] + [_cal_empty(f"cal-empty-{i}") for i in range(12)])
            cfg = self._anchor_assets(root, empty_anchors=False)
            rows, summary = assemble_training_json.assemble(
                None, mined, validation_paths=[], media_root=root, anchor_config=cfg, max_rows=64, row_multiple=8,
                empty_answer_guard=_guard(overall=0.30),
            )
            guard = summary["empty_answer_guard"]
            self.assertEqual(len(rows), 32)
            self.assertEqual((guard["aligned_rows_before"], guard["aligned_rows_after"]), (32, 32))
            self.assertEqual(guard["rows_trimmed_by_source"], {"detection_calibration_negative": 4})
            self.assertEqual(guard["backfilled_rows"], 4)  # 3 mined rows + 1 spare anchor
            self.assertAlmostEqual(guard["after"]["overall_share"], 8 / 32)
            self.assertEqual(guard["status"], "trimmed_to_caps")
            self.assertEqual(summary["materialized_anchor_records"], 4)
            self.assertEqual(summary["materialized_calibration_records"], 8)
            self.assertEqual(sum(r["id"].startswith("pos-") for r in rows), 20)
            self.assertEqual(len({r["id"] for r in rows}), 32)

    def test_aligned_size_shrinks_only_when_candidates_run_out(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined = _write(root / "mined.jsonl", [_det_row(f"pos-{i}", BOX) for i in range(20)] + [_det_row(f"empty-{i}") for i in range(12)])
            rows, summary = assemble_training_json.assemble(
                None, mined, validation_paths=[], media_root=root, max_rows=64, row_multiple=8,
                empty_answer_guard=_guard(overall=0.30),
            )
            guard = summary["empty_answer_guard"]
            self.assertEqual(len(rows), 24)
            self.assertEqual((guard["aligned_rows_before"], guard["aligned_rows_after"], guard["aligned_rows_shrunk"]), (32, 24, 8))
            self.assertEqual(guard["backfilled_rows"], 0)
            self.assertEqual(guard["rows_trimmed_total"], 4)
            self.assertEqual(guard["rows_trimmed_by_source"], {"mined_empty": 4})
            self.assertLessEqual(guard["after"]["overall_share"], 0.30)
            self.assertEqual(guard["exceeded_after"], [])
            self.assertEqual(guard["status"], "trimmed_to_caps")
            self.assertEqual(len(rows) % 8, 0)

    def test_classification_calibration_rows_and_coverage_rows_are_never_trimmed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined = _write(root / "mined.jsonl", [_det_row(f"pos-{i}", BOX) for i in range(4)] + [_det_row(f"empty-{i}") for i in range(4)])
            # an (artificial) empty classification calibration row must survive; only the mined empties go
            classification = _write(root / "cc.jsonl", [{**_mcq_row("cc-empty", answer="[]"), CAL: True, KIND: "classification"}])
            rows, summary = assemble_training_json.assemble(
                None, mined, validation_paths=[], media_root=root, classification_calibration_path=classification,
                empty_answer_guard=_guard(overall=0.20),
            )
            self.assertIn("cc-empty", {r["id"] for r in rows})
            guard = summary["empty_answer_guard"]
            self.assertEqual(guard["rows_trimmed_by_source"], {"mined_empty": 4})
            self.assertEqual(guard["status"], "trimmed_to_caps")  # 1 / 5 == 0.20 is within the cap
            self.assertEqual(guard["never_trimmed"], ["previous_iteration", "anchor_correct", "coverage_blend", "classification_calibration"])

    def test_guard_config_validation_and_repetition_exclusion(self) -> None:
        cfg = _guard()
        self.assertFalse(cfg["enabled"])
        self.assertIsNone(cfg["mode"])
        self.assertEqual(_guard(overall=0.3)["mode"], "enforce")
        self.assertEqual(_guard(overall=0.3, mode="report")["mode"], "report")
        for bad in ({"overall": 0.0}, {"overall": 1.5}, {"classification": -0.1}, {"per_task": {"Defect Detection": 2.0}}, {"per_task": {"Bogus": 0.5}}):
            with self.assertRaises(ValueError):
                _guard(**bad)
        with self.assertRaisesRegex(ValueError, "mode requires at least one cap"):
            _guard(mode="enforce")
        self.assertEqual(assemble_training_json.parse_task_shares(["Defect Detection=0.45", "Ref_based Defect Detection=0.5"]),
                         {"Defect Detection": 0.45, "Ref_based Defect Detection": 0.5})
        with self.assertRaises(ValueError):
            assemble_training_json.parse_task_shares(["Defect Detection"])
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined = _write(root / "mined.jsonl", [_det_row(f"pos-{i}", BOX) for i in range(4)] + [_det_row(f"empty-{i}") for i in range(4)])
            repetition = {"enabled": True, "policy": "deficit_proportional", "rep_min": 0.5, "rep_max": 3.0,
                          "never_repeat_empty_gt": True, "explicit_multipliers": {}}
            with self.assertRaisesRegex(ValueError, "cannot be combined with the repetition blend"):
                assemble_training_json.assemble(
                    None, mined, validation_paths=[], media_root=root, max_rows=8, row_multiple=1,
                    repetition_config=repetition, deficit_weights={"Defect Detection": 1.0}, repetition_seed=17,
                    empty_answer_guard=_guard(overall=0.3),
                )
            _, summary = assemble_training_json.assemble(
                None, mined, validation_paths=[], media_root=root, max_rows=8, row_multiple=1,
                repetition_config=repetition, deficit_weights={"Defect Detection": 1.0}, repetition_seed=17,
                empty_answer_guard=_guard(overall=0.3, mode="report"),
            )
            self.assertEqual(summary["empty_answer_guard"]["mode"], "report")
            self.assertIn("answer_profile", summary)


class GuardBackfillTaskIsolationTests(unittest.TestCase):
    def test_backfill_never_consumes_rows_of_a_task_without_remaining_candidates(self) -> None:
        # Ref_based Defect Detection has 6 candidates, all already in the aligned corpus (zero spare);
        # Defect Detection has 4 empties at the front and 44 positives (42 fit). 4 / 48 exceeds the 5%
        # cap, 2 / 46 does not: trimming 2 Defect Detection empties frees 2 slots that only Defect
        # Detection's spare positives may back-fill.
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined_rows = [_det_row(f"empty-{i}") for i in range(4)] + [_det_row(f"pos-{i}", BOX) for i in range(44)]
            mined_rows += [_ref_det_row(f"ref-{i}", BOX) for i in range(6)]
            mined = _write(root / "mined.jsonl", mined_rows)
            before_rows, before = assemble_training_json.assemble(
                None, mined, validation_paths=[], media_root=root, max_rows=48, row_multiple=8,
            )
            rows, summary = assemble_training_json.assemble(
                None, mined, validation_paths=[], media_root=root, max_rows=48, row_multiple=8,
                empty_answer_guard=_guard(overall=0.05),
            )
            guard = summary["empty_answer_guard"]
            self.assertEqual((len(before_rows), len(rows)), (48, 48))
            self.assertEqual((guard["aligned_rows_before"], guard["aligned_rows_after"]), (48, 48))
            self.assertEqual(guard["rows_trimmed_by_source"], {"mined_empty": 2})
            self.assertEqual(guard["backfilled_rows"], 2)
            self.assertEqual(guard["passes"], 2)
            self.assertEqual(guard["status"], "trimmed_to_caps")
            self.assertAlmostEqual(guard["after"]["overall_share"], 2 / 48)
            counts_before = Counter(r["task_type"] for r in before_rows)
            counts_after = Counter(r["task_type"] for r in rows)
            self.assertEqual(counts_before["Ref_based Defect Detection"], 6)
            self.assertEqual(counts_after["Ref_based Defect Detection"], 6)  # untouched: no spare candidates, no trimming
            self.assertEqual(counts_after["Defect Detection"], 42)
            backfilled = {r["id"] for r in rows} - {r["id"] for r in before_rows}
            self.assertEqual(backfilled, {"pos-38", "pos-39"})
            self.assertEqual(sum(r["id"].startswith("empty-") for r in rows), 2)
            self.assertAlmostEqual(guard["after"]["per_task"]["Ref_based Defect Detection"], 0.0)


class GuardWiringTests(unittest.TestCase):
    def test_runner_passes_guard_options_to_the_assembler_only(self) -> None:
        selector, assembler = render_iteration_mining_runner._partition_materialization_arguments([
            "python", "select.py", "--top-k", "50",
            "--max-empty-answer-share", "0.30",
            "--max-empty-answer-share-task", "Defect Detection=0.45",
            "--max-empty-answer-share-task", "Ref_based Defect Detection=0.50",
            "--max-classification-empty-share", "0.10",
            "--empty-answer-guard-mode", "enforce",
        ])
        self.assertEqual(selector, ["python", "select.py", "--top-k", "50", "--no-repetition-blend"])
        self.assertEqual(assembler, [
            "--max-empty-answer-share", "0.30",
            "--max-empty-answer-share-task", "Defect Detection=0.45",
            "--max-empty-answer-share-task", "Ref_based Defect Detection=0.50",
            "--max-classification-empty-share", "0.10",
            "--empty-answer-guard-mode", "enforce",
        ])
        for bogus in ("--max-empty-answer-bogus", "--empty-answer-guard-bogus", "--max-classification-empty-bogus"):
            with self.assertRaisesRegex(ValueError, "unsupported materialization option"):
                render_iteration_mining_runner._partition_materialization_arguments(["x", bogus, "1"])
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            plan = render_iteration_mining_runner.build_plan(
                selector_command=[sys.executable, "selector.py", "--max-empty-answer-share", "0.3"],
                previous_jsonl=None, previous_sha256=None,
                mined_jsonl=root / "mined.jsonl", current_quota_manifest=root / "current-quota.json",
                train_jsonl=root / "train.jsonl", assemble_summary=root / "assembly.json",
                final_quota_manifest=root / "quota.json", media_root=root,
                max_rows=768, row_multiple=768, epochs=5, global_batch=768,
            )
            self.assertTrue(plan["invariants"]["empty_answer_guard_owned_by_assembler"])
            self.assertIn("--max-empty-answer-share", plan["assembler"]["command"])
            self.assertNotIn("--max-empty-answer-share", plan["selector"]["command"])

    def test_init_records_the_empty_answer_guard(self) -> None:
        from test_cosmos3_init_state_contract import Cosmos3InitStateContractTests as Base
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = Base._workspace(root)
            rc = init_deft_state.main(Base._argv(
                root, workspace, "--max-empty-answer-share", "0.30",
                "--max-empty-answer-share-task", "Defect Detection=0.45",
                "--max-empty-answer-share-task", "Ref_based Defect Detection=0.50",
                "--max-classification-empty-share", "0.10",
            ))
            self.assertEqual(rc, 0)
            guard = json.loads((root / "results/deft_state.json").read_text())["config"]["mining"]["empty_answer_guard"]
            self.assertTrue(guard["enabled"])
            self.assertEqual(guard["mode"], "enforce")
            self.assertEqual(guard["max_empty_answer_share"], 0.30)
            self.assertEqual(guard["max_empty_answer_share_task"], {"Defect Detection": 0.45, "Ref_based Defect Detection": 0.50})
            self.assertEqual(guard["max_classification_empty_share"], 0.10)
            self.assertEqual(guard["trim_order"], ["detection_calibration_negative", "mined_empty"])
            self.assertIn("assembler-only options", guard["owner"])
            rc = init_deft_state.main(Base._argv(root / "b", workspace, "--max-empty-answer-share", "0.30", "--empty-answer-guard-mode", "report"))
            self.assertEqual(rc, 0)
            guard = json.loads((root / "b/results/deft_state.json").read_text())["config"]["mining"]["empty_answer_guard"]
            self.assertEqual(guard["mode"], "report")
            rc = init_deft_state.main(Base._argv(root / "c", workspace))
            self.assertEqual(rc, 0)
            guard = json.loads((root / "c/results/deft_state.json").read_text())["config"]["mining"]["empty_answer_guard"]
            self.assertFalse(guard["enabled"])
            self.assertIsNone(guard["mode"])
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = init_deft_state.main(Base._argv(root / "d", workspace, "--max-empty-answer-share", "1.5"))
            self.assertNotEqual(rc, 0)
            self.assertIn("empty-answer", stderr.getvalue())

    def test_ablation_quota_manifest_records_the_empty_count_and_tags_detection_calibration(self) -> None:
        import defect_detection_ablation
        from test_cosmos3_defect_detection_ablation_contract import _candidate, _row as _ablation_row
        rows: list[dict] = []
        candidates: list[dict] = []
        for index in range(8):
            boxes = [{"bbox_2d": [10 + index, 20, 110 + index * 10, 120], "label": ("open", "short")[index % 2]}]
            row = _ablation_row(f"dd-pos-{index}", "Defect Detection", boxes=boxes, dataset=f"source-{index % 2}")
            rows.append(row)
            candidates.append(_candidate(row, evidence=["hard_positive_proxy_false_negative"], phash=f"{index + 1:016x}"))
        for index in range(4):
            row = _ablation_row(f"dd-empty-{index}", "Defect Detection", boxes=[])
            rows.append(row)
            candidates.append(_candidate(
                row, evidence=["hard_negative_proxy_false_positive"] if index < 3 else ["calibration_empty_ground_truth"],
                phash=f"{index + 20:016x}", route_tier="strict" if index < 3 else "calibration",
            ))
        maintenance = ("Component Classification", "Component Detection", "Defect Classification",
                       "Ref_based Defect Classification", "Ref_based Defect Detection")
        for index in range(12):
            row = _ablation_row(f"maint-{index}", maintenance[index % 5])  # fixture answers are "[]" -> empty
            rows.append(row)
            candidates.append(_candidate(row, phash=f"{index + 40:016x}"))
        selected, manifest = defect_detection_ablation.materialize(
            candidate_rows=candidates, source_records=rows, validation_records=[], media_root=pathlib.Path("/data"),
            max_rows=24, row_multiple=6, defect_detection_fraction=0.5, proxy_empty_rate=1 / 3,
            epochs=5, global_batch=6, near_duplicate_hamming_distance=0,
        )
        self.assertTrue(manifest["verified"])
        self.assertEqual(manifest["new_rows_empty"], sum(assemble_training_json.is_empty_ground_truth(r) for r in selected))
        self.assertEqual(manifest["new_rows_empty"], 16)
        self.assertEqual(manifest["new_rows_empty_by_task"]["Defect Detection"], 4)
        self.assertEqual(manifest["row_counts"]["total"], 24)  # no behavioural change
        calibration = [r for r in selected if r.get(CAL) is True]
        self.assertEqual([r["id"] for r in calibration], ["dd-empty-3"])
        self.assertEqual(calibration[0][KIND], "detection")
        self.assertTrue(all(KIND not in r for r in selected if r.get(CAL) is not True))


if __name__ == "__main__":
    unittest.main()
