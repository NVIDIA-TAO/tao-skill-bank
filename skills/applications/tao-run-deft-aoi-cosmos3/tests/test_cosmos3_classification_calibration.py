# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Classification calibration quota (Phase 4 step 4c-A): single-image MCQ rows with a
non-empty class answer, class shares from the KPI set, identity exclusion, markers."""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import string
import sys
import tempfile
import unittest
from collections import Counter


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import anchor_rows  # noqa: E402
import answer_profile  # noqa: E402
import assemble_training_json  # noqa: E402
import init_deft_state  # noqa: E402
import render_iteration_mining_runner  # noqa: E402
import select_classification_calibration as scc  # noqa: E402

DEFECT_OPTIONS = [
    "Missing Component: A part that should occupy a footprint is absent.",
    "Shift / Skew / Rotation: A component is offset or turned.",
    "Tombstoning: One end of a small component has risen.",
    "Other: A visible PCBA defect outside A-C.",
]
COMPONENT_OPTIONS = ["Resistors", "Capacitors", "Others"]
TAIL = (
    "Return only the option letter when one class applies. If multiple classes apply, "
    "return only a compact letter list such as [B,D]."
)


def _mcq_prompt(options: list[str], *, question: str = "Which defect classes are visible in this image?",
                empty_clause: str = "If no visible defect is present in this image, return [].") -> str:
    lines = [
        "This is an industrial visual inspection task for PCBA defect analysis. Use only the provided image.",
        f"Question: {question}",
        "current possible classes:.",
        *[f"{letter}. {text}" for letter, text in zip(string.ascii_uppercase, options)],
        empty_clause,
        TAIL,
    ]
    return "\n".join(lines)


BCQ_PROMPT = (
    "This is an industrial visual inspection task. Use only the provided AOI image.\n"
    "Question: Does this image contain any visible defect?\n"
    "A. Yes, this image contains a defect.\nB. No, this image does not contain any defects.\n"
    "Answer with the complete option text from the given choices."
)


def _row(record_id: str, task: str = "Defect Classification", *, answer: str = "A", dataset: str = "dsA",
         options: list[str] | None = None, prompt: str | None = None, two_images: bool = False) -> dict:
    if prompt is None:
        prompt = _mcq_prompt(options or (COMPONENT_OPTIONS if task == "Component Classification" else DEFECT_OPTIONS))
    images = [{"type": "image", "image": f"images/{dataset}/{record_id}.png", "min_pixels": 1, "max_pixels": 1}]
    if two_images:
        images.insert(0, {"type": "image", "image": f"images/{dataset}/{record_id}-golden.png", "min_pixels": 1, "max_pixels": 1})
    return {
        "id": record_id,
        "task_type": task,
        "dataset": dataset,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [*images, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ],
    }


def _det_row(record_id: str, boxes: list[dict] | None = None, task: str = "Defect Detection", dataset: str = "mine") -> dict:
    return {
        "id": record_id,
        "task_type": task,
        "dataset": dataset,
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": f"images/{dataset}/{record_id}.png", "min_pixels": 1, "max_pixels": 1},
                {"type": "text", "text": "Locate every visible defect. If no visible defect is present, return []."},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps(boxes or [])}]},
        ],
    }


def _write(path: pathlib.Path, rows: list[dict]) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _kpi(missing: int = 5, shift: int = 3, tomb: int = 2) -> list[dict]:
    rows = [_row(f"k-miss-{i}", answer="A") for i in range(missing)]
    rows += [_row(f"k-shift-{i}", answer="B") for i in range(shift)]
    rows += [_row(f"k-tomb-{i}", answer="C") for i in range(tomb)]
    return rows


class AnswerProfileHelpersTests(unittest.TestCase):
    def test_mcq_options_labels_and_letter_parsing(self) -> None:
        options = answer_profile.mcq_options(_mcq_prompt(DEFECT_OPTIONS))
        self.assertEqual(options, {"A": "Missing Component", "B": "Shift / Skew / Rotation", "C": "Tombstoning", "D": "Other"})
        self.assertIsNone(answer_profile.mcq_options(BCQ_PROMPT))
        self.assertIsNone(answer_profile.mcq_options("detect boxes"))
        self.assertEqual(answer_profile.mcq_answer_letters("F"), {"F"})
        self.assertEqual(answer_profile.mcq_answer_letters("[B,D]"), {"B", "D"})
        self.assertEqual(answer_profile.mcq_answer_letters('["B", "D"]'), {"B", "D"})
        self.assertEqual(answer_profile.mcq_answer_letters("[]"), set())
        self.assertEqual(answer_profile.mcq_answer_letters("```json\n[]\n```"), set())
        self.assertIsNone(answer_profile.mcq_answer_letters("no idea"))
        self.assertEqual(answer_profile.mcq_labels(_row("x", answer="[A,C]")), ("ok", ["Missing Component", "Tombstoning"]))
        self.assertEqual(answer_profile.mcq_labels(_row("x", answer="[]"))[0], "empty_ground_truth")
        self.assertEqual(answer_profile.mcq_labels(_row("x", answer="Z"))[0], "unknown_option_letter")
        self.assertEqual(answer_profile.mcq_labels(_row("x", prompt=BCQ_PROMPT, answer="B. No, this image does not contain any defects."))[0], "not_mcq_format")

    def test_empty_ground_truth_and_formats(self) -> None:
        self.assertTrue(answer_profile.is_empty_ground_truth(_det_row("d", [])))
        self.assertTrue(answer_profile.is_empty_ground_truth(_row("m", answer="[]")))
        self.assertFalse(answer_profile.is_empty_ground_truth(_det_row("d", [{"bbox_2d": [0, 0, 1, 1], "label": "x"}])))
        self.assertFalse(answer_profile.is_empty_ground_truth(_row("b", prompt=BCQ_PROMPT, answer="B. No, this image does not contain any defects.")))
        self.assertEqual(answer_profile.answer_format(_det_row("d")), "DET")
        self.assertEqual(answer_profile.answer_format(_row("m")), "MCQ")
        self.assertEqual(answer_profile.answer_format(_row("b", prompt=BCQ_PROMPT, answer="B. No, this image does not contain any defects.")), "BCQ")
        self.assertEqual(answer_profile.answer_format(_row("y", prompt="Does this image contain any visible defect?", answer="Yes, the target image contains a defect.")), "BCQ")
        self.assertEqual(answer_profile.image_count(_row("r", "Ref_based Defect Classification", two_images=True)), 2)


class SelectorEligibilityTests(unittest.TestCase):
    def _select(self, pool: list[dict], kpi: list[dict], totals: dict[str, int], **kwargs):
        return scc.select_classification_calibration(
            pool, kpi_records=kpi, task_totals=totals, seed=17, min_fill_fraction=0.9, **kwargs
        )

    def test_eligibility_filter_rejects_bcq_empty_two_image_and_detection_rows(self) -> None:
        pool = [_row(f"ok-{i}", answer="ABC"[i % 3]) for i in range(8)]  # 3 Missing / 3 Shift / 2 Tombstoning
        pool.append(_row("bcq", prompt=BCQ_PROMPT, answer="B. No, this image does not contain any defects."))
        pool.append(_row("empty", answer="[]"))
        pool.append(_row("pair", "Ref_based Defect Classification", two_images=True, answer="A"))
        pool.append(_det_row("det", [{"bbox_2d": [0, 0, 1, 1], "label": "x"}]))
        pool.append(_row("badletter", answer="Q"))
        pool.append(_row("garbage", answer="not an option"))
        rows, manifest = self._select(pool, _kpi(missing=3, shift=3, tomb=2), {"Defect Classification": 8})
        self.assertEqual(sorted(r["id"] for r in rows), [f"ok-{i}" for i in range(8)])
        task = manifest["tasks"]["Defect Classification"]
        self.assertEqual((task["requested"], task["eligible"], task["selected"]), (8, 8, 8))
        self.assertEqual(task["fill_fraction"], 1.0)
        rejected = manifest["rejected"]
        self.assertEqual(rejected["not_mcq_format"], 1)
        self.assertEqual(rejected["empty_ground_truth"], 1)
        self.assertEqual(rejected["task_not_eligible"], 2)  # the reference pair and the detection row
        self.assertEqual(rejected["unknown_option_letter"], 1)
        self.assertEqual(rejected["unparsed_answer"], 1)
        self.assertEqual(manifest["examined_records"], len(pool))
        for row in rows:
            self.assertIs(row["deft_calibration"], True)
            self.assertEqual(row["deft_calibration_kind"], "classification")
            original = next(p for p in pool if p["id"] == row["id"])
            self.assertEqual({k: v for k, v in row.items() if k not in ("deft_calibration", "deft_calibration_kind")}, original)

    def test_two_image_classification_rows_are_rejected_even_when_task_total_names_them(self) -> None:
        with self.assertRaisesRegex(ValueError, "single-image classification"):
            self._select([], _kpi(), {"Ref_based Defect Classification": 4})
        with self.assertRaisesRegex(ValueError, "single-image classification"):
            self._select([], _kpi(), {"Defect Detection": 4})

    def test_class_shares_follow_kpi_with_largest_remainder_and_redistribute_missing_classes(self) -> None:
        # KPI: Missing 50% / Shift 30% / Tombstoning 20%; the pool has no Tombstoning rows.
        pool = [_row(f"miss-{i}", answer="A", dataset=f"ds{i % 3}") for i in range(20)]
        pool += [_row(f"shift-{i}", answer="B", dataset=f"ds{i % 2}") for i in range(20)]
        pool += [_row(f"other-{i}", answer="D") for i in range(20)]  # present in the pool, absent from the KPI set -> share 0
        rows, manifest = self._select(pool, _kpi(), {"Defect Classification": 10})
        task = manifest["tasks"]["Defect Classification"]
        self.assertEqual(task["class_share_source"], "kpi")
        self.assertEqual(task["classes_missing_in_pool"], ["Tombstoning"])
        per_class = task["per_class"]
        # 50/30 renormalised over the present classes -> 6.25 / 3.75 -> largest remainder 6 / 4
        self.assertEqual(per_class["Missing Component"]["target"], 6)
        self.assertEqual(per_class["Shift / Skew / Rotation"]["target"], 4)
        self.assertEqual(per_class["Tombstoning"]["target"], 0)
        self.assertEqual(per_class["Tombstoning"]["eligible"], 0)
        self.assertEqual(per_class["Other"]["target"], 0)
        self.assertEqual(per_class["Missing Component"]["kpi_rows"], 5)
        self.assertAlmostEqual(per_class["Missing Component"]["target_share"], 0.625)
        selected = Counter(answer_profile.mcq_labels(r)[1][0] for r in rows)
        self.assertEqual(selected, {"Missing Component": 6, "Shift / Skew / Rotation": 4})
        self.assertEqual(per_class["Missing Component"]["selected"], 6)
        self.assertEqual(per_class["Shift / Skew / Rotation"]["selected"], 4)
        # spread across datasets, not file order
        self.assertEqual(task["datasets"], {"ds0": 4, "ds1": 4, "ds2": 2})
        # deterministic regardless of pool order
        again, _ = self._select(list(reversed(pool)), _kpi(), {"Defect Classification": 10})
        self.assertEqual([r["id"] for r in rows], [r["id"] for r in again])

    def test_uniform_fallback_when_kpi_has_no_rows_for_the_task(self) -> None:
        pool = [_row(f"res-{i}", "Component Classification", answer="A") for i in range(6)]
        pool += [_row(f"cap-{i}", "Component Classification", answer="B") for i in range(6)]
        rows, manifest = self._select(pool, _kpi(), {"Component Classification": 5})
        task = manifest["tasks"]["Component Classification"]
        self.assertEqual(task["class_share_source"], "uniform_pool_fallback")
        self.assertEqual({k: v["target"] for k, v in task["per_class"].items()}, {"Capacitors": 3, "Resistors": 2})
        self.assertEqual(len(rows), 5)

    def test_multi_label_rows_count_once_and_fill_from_one_class_bucket(self) -> None:
        pool = [_row(f"multi-{i}", answer="[A,B]") for i in range(4)] + [_row(f"shift-{i}", answer="B") for i in range(4)]
        rows, manifest = self._select(pool, _kpi(missing=5, shift=5, tomb=0), {"Defect Classification": 6})
        self.assertEqual(len(rows), 6)
        self.assertEqual(len({r["id"] for r in rows}), 6)
        per_class = manifest["tasks"]["Defect Classification"]["per_class"]
        self.assertEqual(per_class["Missing Component"]["selected"] + per_class["Shift / Skew / Rotation"]["selected"], 6)

    def test_exclusion_by_identity_and_marker_free_dedupe(self) -> None:
        pool = [_row(f"ok-{i}", answer="A") for i in range(6)]
        pool.append({**_row("ok-0", answer="A"), "deft_anchor": True})  # same content behind an inert marker
        pool.append({**_row("ok-0", answer="A"), "id": "ok-0-alias"})  # same image, another id
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            corpus = _write(root / "train.jsonl", [{**_row("ok-1", answer="A"), "deft_calibration": True}, _det_row("dd-1")])
            anchors = _write(root / "anchors.jsonl", [_row("ok-2", answer="A")])
            identities = root / "ids.txt"
            identity = assemble_training_json.record_identity(_row("ok-3", answer="A"), media_root=None)
            identities.write_text(identity + "\n", encoding="utf-8")
            excluded_identities, excluded_ids, inputs = scc.load_exclusions([corpus, anchors, identities], media_root=None)
            self.assertEqual(len(inputs), 3)
            self.assertIn(identity, excluded_identities)
            self.assertEqual(excluded_ids, {"ok-1", "dd-1", "ok-2"})
            rows, manifest = self._select(
                pool, _kpi(), {"Defect Classification": 3},
                excluded_identities=excluded_identities, excluded_ids=excluded_ids,
            )
        self.assertEqual(sorted(r["id"] for r in rows), ["ok-0", "ok-4", "ok-5"])
        self.assertEqual(manifest["excluded_by_identity"], 3)
        self.assertEqual(manifest["deduplicated_content"], 1)
        self.assertEqual(manifest["deduplicated_identity"], 1)
        self.assertEqual(manifest["tasks"]["Defect Classification"]["eligible"], 3)

    def test_shortfall_is_recorded_and_fails_closed_unless_allowed(self) -> None:
        pool = [_row(f"ok-{i}", answer="A") for i in range(4)]
        rows, manifest = self._select(pool, _kpi(), {"Defect Classification": 8})
        self.assertEqual(len(rows), 4)
        self.assertFalse(manifest["fill_ok"])
        self.assertEqual(manifest["shortfall_tasks"], ["Defect Classification"])
        self.assertEqual(manifest["tasks"]["Defect Classification"]["shortfall"], 4)
        self.assertAlmostEqual(manifest["tasks"]["Defect Classification"]["fill_fraction"], 0.5)
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            pool_path = _write(root / "pool.jsonl", pool)
            kpi_path = _write(root / "kpi.jsonl", _kpi())
            argv = ["--pool", str(pool_path), "--kpi", str(kpi_path), "--task-total", "Defect Classification=8",
                    "--seed", "17", "--output", str(root / "cc.jsonl"), "--manifest", str(root / "cc.json")]
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
                rc = scc.main(argv)
            self.assertEqual(rc, 2)
            self.assertIn("shortfall", stderr.getvalue())
            written = json.loads((root / "cc.json").read_text())
            self.assertFalse(written["accepted"])
            self.assertEqual(written["tasks"]["Defect Classification"]["shortfall"], 4)
            with contextlib.redirect_stdout(io.StringIO()):
                rc = scc.main([*argv, "--allow-shortfall"])
            self.assertEqual(rc, 0)
            written = json.loads((root / "cc.json").read_text())
            self.assertTrue(written["accepted"])
            self.assertTrue(written["allow_shortfall"])
            self.assertFalse(written["fill_ok"])
            self.assertEqual(len((root / "cc.jsonl").read_text().splitlines()), 4)
            with contextlib.redirect_stdout(io.StringIO()):
                rc = scc.main([*argv[:-4], "--min-fill-fraction", "0.5", "--output", str(root / "cc2.jsonl"), "--manifest", str(root / "cc2.json")])
            self.assertEqual(rc, 0)

    def test_cli_writes_marked_rows_and_manifest_with_input_hashes(self) -> None:
        pool = [_row(f"ok-{i}", answer="ABCD"[i % 4], dataset=f"ds{i % 2}") for i in range(12)]
        pool += [_row(f"cc-{i}", "Component Classification", answer="AB"[i % 2]) for i in range(6)]
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            pool_path = _write(root / "pool.jsonl", pool)
            kpi_path = _write(root / "kpi.jsonl", _kpi())
            corpus = _write(root / "train.jsonl", [_row("ok-0", answer="A", dataset="ds0")])
            with contextlib.redirect_stdout(io.StringIO()):
                rc = scc.main([
                    "--pool", str(pool_path), "--kpi", str(kpi_path), "--media-root", str(root),
                    "--task-total", "Defect Classification=4", "--task-total", "Component Classification=2",
                    "--exclude-identities-file", str(corpus), "--seed", "23",
                    "--output", str(root / "out/cc.jsonl"), "--manifest", str(root / "out/cc.json"),
                ])
            self.assertEqual(rc, 0)
            rows = [json.loads(line) for line in (root / "out/cc.jsonl").read_text().splitlines()]
            manifest = json.loads((root / "out/cc.json").read_text())
            self.assertEqual(len(rows), 6)
            self.assertEqual(Counter(r["task_type"] for r in rows), {"Defect Classification": 4, "Component Classification": 2})
            self.assertTrue(all(r["deft_calibration"] is True and r["deft_calibration_kind"] == "classification" for r in rows))
            self.assertNotIn("ok-0", {r["id"] for r in rows})
            self.assertEqual(manifest["schema_version"], "classification_calibration_v1")
            self.assertEqual(manifest["seed"], 23)
            self.assertEqual(manifest["min_fill_fraction"], 0.9)
            self.assertEqual(manifest["excluded_by_identity"], 1)
            self.assertEqual(manifest["selected_total"], 6)
            self.assertEqual(manifest["task_totals"], {"Defect Classification": 4, "Component Classification": 2})
            self.assertEqual(manifest["inputs"]["pool"]["sha256"], assemble_training_json.sha256_file(pool_path))
            self.assertEqual(manifest["inputs"]["kpi"]["sha256"], assemble_training_json.sha256_file(kpi_path))
            self.assertEqual(manifest["inputs"]["exclusions"][0]["sha256"], assemble_training_json.sha256_file(corpus))
            self.assertEqual(manifest["markers"], {"deft_calibration": True, "deft_calibration_kind": "classification"})
            for task in ("Defect Classification", "Component Classification"):
                payload = manifest["tasks"][task]
                for key in ("requested", "eligible", "selected", "fill_fraction", "shortfall", "per_class", "class_share_source"):
                    self.assertIn(key, payload)
                for label, entry in payload["per_class"].items():
                    self.assertEqual(set(entry), {"kpi_rows", "target_share", "target", "eligible", "selected"})
            self.assertEqual(manifest["output"]["rows"], 6)
            self.assertEqual(manifest["output"]["sha256"], assemble_training_json.sha256_file(root / "out/cc.jsonl"))


class AssemblerClassificationCalibrationTests(unittest.TestCase):
    def _corpus(self, root: pathlib.Path, classification_rows: int = 30):
        mined = _write(root / "mined.jsonl", [_det_row(f"m-{i}") for i in range(90)])
        classification = _write(
            root / "classification_calibration.jsonl",
            [{**_row(f"c-{i}", answer="A"), "deft_calibration": True, "deft_calibration_kind": "classification"}
             for i in range(classification_rows)],
        )
        anchors = [_row(f"a-{i}", "Component Classification", answer="A", dataset=f"ds{i % 3}") for i in range(200)]
        anchors += [_det_row(f"b-{i}", dataset="pool") for i in range(200)]
        source = _write(root / "anchor_candidates.jsonl", anchors)
        kpi = _write(root / "kpi.jsonl", [_row(f"k-{i}", "Component Classification", answer="A") for i in range(50)]
                     + [_det_row(f"kd-{i}") for i in range(50)])
        return mined, classification, source, kpi

    def test_classification_rows_join_as_current_rows_and_survive_the_cap_trim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined, classification, source, kpi = self._corpus(root)
            cfg = anchor_rows.validate_anchor_config(0.10, source, kpi, None)
            rows, summary = assemble_training_json.assemble(
                None, mined, validation_paths=[], media_root=root, anchor_config=cfg,
                classification_calibration_path=classification, max_rows=128, row_multiple=32,
            )
            self.assertEqual(len(rows), 128)
            kept = [r for r in rows if r.get("deft_calibration_kind") == "classification"]
            self.assertEqual(len(kept), 30)
            self.assertEqual(summary["materialized_classification_calibration_records"], 30)
            block = summary["classification_calibration"]
            self.assertTrue(block["enabled"])
            self.assertEqual(block["input_records"], 30)
            self.assertEqual(block["materialized_new"], 30)
            self.assertEqual(block["materialized_total"], 30)
            self.assertEqual(block["source_sha256"], assemble_training_json.sha256_file(classification))
            self.assertEqual(summary["materialized_anchor_records"], 13)  # round(128 * 0.1)
            self.assertEqual(summary["materialized_current_records"], 85)  # 128 - 30 - 13 mined rows survive
            self.assertEqual(summary["anchor"]["cap_reservation"]["calibration_rows_protected"], 30)
            kinds = Counter(p["source_kind"] for p in summary["provenance"])
            self.assertEqual(kinds["classification_calibration"], 30)
            self.assertEqual(summary["exposure_rows"]["classification_calibration_new"], 30)
            self.assertEqual(len({r["id"] for r in rows}), 128)

    def test_remined_classification_row_is_deduplicated_marker_free_and_previous_rows_are_counted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined, classification, source, kpi = self._corpus(root, classification_rows=10)
            rows, _ = assemble_training_json.assemble(
                None, mined, validation_paths=[], media_root=root, classification_calibration_path=classification,
            )
            previous = _write(root / "train1.jsonl", rows)
            remined = [{k: v for k, v in rows[-1].items() if k not in ("deft_calibration", "deft_calibration_kind")}]
            self.assertEqual(remined[0]["id"], "c-9")
            mined2 = _write(root / "mined2.jsonl", remined + [_det_row(f"n-{i}") for i in range(20)])
            classification2 = _write(
                root / "cc2.jsonl",
                [{**_row("c-0", answer="A"), "deft_calibration": True, "deft_calibration_kind": "classification"},
                 {**_row("c-new", answer="B"), "deft_calibration": True, "deft_calibration_kind": "classification"}],
            )
            rows2, summary2 = assemble_training_json.assemble(
                previous, mined2, previous_sha256=assemble_training_json.sha256_file(previous),
                validation_paths=[], media_root=root, classification_calibration_path=classification2,
            )
            ids = [r["id"] for r in rows2]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(summary2["duplicates_skipped"], 1)  # the re-mined marker-free copy of c-9
            self.assertEqual(summary2["classification_calibration"]["duplicates_skipped"], 1)  # c-0 already retained
            self.assertEqual(summary2["classification_calibration"]["materialized_new"], 1)
            self.assertEqual(summary2["classification_calibration"]["prior_rows"], 10)
            self.assertEqual(summary2["materialized_classification_calibration_records"], 11)
            self.assertEqual(summary2["exposure_rows"]["previous_classification_calibration"], 10)
            self.assertEqual(summary2["exposure_rows"]["previous_mined"], 90)
            kept = [r for r in rows2 if r["id"] == "c-9"]
            self.assertEqual(len(kept), 1)
            self.assertEqual(kept[0].get("deft_calibration_kind"), "classification")

    def test_unmarked_or_two_image_classification_input_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined, _, _, _ = self._corpus(root)
            bad = _write(root / "bad.jsonl", [_row("u", answer="A")])
            with self.assertRaisesRegex(ValueError, "classification calibration rows must carry"):
                assemble_training_json.assemble(None, mined, validation_paths=[], media_root=root, classification_calibration_path=bad)
            empty = root / "empty.jsonl"
            empty.write_text("", encoding="utf-8")
            rows, summary = assemble_training_json.assemble(None, mined, validation_paths=[], media_root=root, classification_calibration_path=empty)
            self.assertEqual(len(rows), 90)
            self.assertEqual(summary["classification_calibration"]["input_records"], 0)

    def test_cli_flag_and_disabled_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined, classification, _, _ = self._corpus(root, classification_rows=5)
            out = root / "train.jsonl"
            rc = assemble_training_json.main([
                "--mined-jsonl", str(mined), "--output", str(out), "--media-root", str(root),
                "--classification-calibration-jsonl", str(classification),
            ])
            self.assertEqual(rc, 0)
            summary = json.loads((root / "assemble_summary.json").read_text())
            self.assertEqual(summary["materialized_classification_calibration_records"], 5)
            self.assertEqual(summary["output_records"], 95)
            _, plain = assemble_training_json.assemble(None, mined, validation_paths=[], media_root=root)
            self.assertFalse(plain["classification_calibration"]["enabled"])
            self.assertEqual(plain["materialized_classification_calibration_records"], 0)


class InitAndRunnerWiringTests(unittest.TestCase):
    def test_init_records_classification_calibration_totals(self) -> None:
        from test_cosmos3_init_state_contract import Cosmos3InitStateContractTests as Base
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = Base._workspace(root)
            rc = init_deft_state.main(Base._argv(
                root, workspace, "--calibration-task-total", "Defect Classification=256",
                "--calibration-task-total", "Component Classification=128",
            ))
            self.assertEqual(rc, 0)
            mining = json.loads((root / "results/deft_state.json").read_text())["config"]["mining"]
            self.assertEqual(mining["classification_calibration"], {"Defect Classification": 256, "Component Classification": 128})
            contract = mining["classification_calibration_contract"]
            self.assertTrue(contract["enabled"])
            self.assertEqual(contract["min_fill_fraction"], 0.9)
            self.assertEqual(contract["seed"], 17)
            self.assertEqual(contract["class_shares_source"], str((workspace / "annotations/proxy_kpi.jsonl").resolve()))
            self.assertEqual(contract["pool"], str((workspace / "annotations/mining.jsonl").resolve()))
            self.assertEqual(contract["markers"], {"deft_calibration": True, "deft_calibration_kind": "classification"})
            # detection totals keep the profile policy; classification totals do not enter the box-count contract
            self.assertEqual(mining["calibration_quota_contract"]["policy"], "proxy_empty_rate_by_reference_cohort")
            rc = init_deft_state.main(Base._argv(
                root / "b", workspace, "--calibration-task-total", "Defect Detection=8",
                "--calibration-task-total", "Defect Classification=4", "--classification-calibration-seed", "5",
            ))
            self.assertEqual(rc, 0)
            mining = json.loads((root / "b/results/deft_state.json").read_text())["config"]["mining"]
            self.assertEqual(mining["classification_calibration"], {"Defect Classification": 4})
            self.assertEqual(mining["classification_calibration_contract"]["seed"], 5)
            self.assertEqual(mining["calibration_quota_contract"]["policy"], "kpi_profile_count_bins")
            self.assertEqual(mining["calibration_quota_contract"]["task_totals"], {"Defect Detection": 8})
            rc = init_deft_state.main(Base._argv(root / "c", workspace))
            self.assertEqual(rc, 0)
            mining = json.loads((root / "c/results/deft_state.json").read_text())["config"]["mining"]
            self.assertEqual(mining["classification_calibration"], {})
            self.assertFalse(mining["classification_calibration_contract"]["enabled"])

    def test_runner_renders_the_classification_stage_and_feeds_the_assembler(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            previous = _write(root / "train0.jsonl", [_det_row("old")])
            request = {
                "selector_command": [sys.executable, "selector.py", "--anchor-share", "0.1", "--anchor-source", str(root / "anchors.jsonl"),
                                     "--anchor-task-shares", str(root / "kpi.jsonl")],
                "classification_calibration_command": [sys.executable, "select_classification_calibration.py", "--pool", str(root / "pool.jsonl"),
                                                       "--kpi", str(root / "kpi.jsonl"), "--task-total", "Defect Classification=256"],
                "previous_jsonl": previous, "previous_sha256": assemble_training_json.sha256_file(previous),
                "mined_jsonl": root / "mined.jsonl", "current_quota_manifest": root / "current-quota.json",
                "train_jsonl": root / "train.jsonl", "assemble_summary": root / "assembly.json",
                "final_quota_manifest": root / "quota.json", "media_root": root,
                "max_rows": 768, "row_multiple": 768, "epochs": 5, "global_batch": 768,
            }
            plan = render_iteration_mining_runner.build_plan(**request)
            stage = plan["classification_calibration"]
            command = stage["command"]
            self.assertEqual(command[:2], request["classification_calibration_command"][:2])
            exclusions = [command[i + 1] for i, v in enumerate(command) if v == "--exclude-identities-file"]
            self.assertEqual(exclusions, [str(previous), str(root / "anchors.jsonl"), str(root / "mined.jsonl")])
            self.assertEqual(stage["exclusions"], exclusions)
            self.assertEqual(stage["output"], str(root / "classification_calibration.jsonl"))
            self.assertEqual(stage["manifest"], str(root / "classification_calibration_manifest.json"))
            self.assertEqual(command[-4:], ["--output", stage["output"], "--manifest", stage["manifest"]])
            assembler = plan["assembler"]["command"]
            self.assertIn("--classification-calibration-jsonl", assembler)
            self.assertEqual(assembler[assembler.index("--classification-calibration-jsonl") + 1], stage["output"])
            self.assertTrue(plan["invariants"]["classification_calibration_output_is_not_training_jsonl"])
            runner_text = render_iteration_mining_runner.render_runner(plan)
            names = [s["name"] for s in eval(runner_text.split("STAGES = ", 1)[1].split("\n\n", 1)[0])]
            self.assertEqual(names, ["selector", "classification_calibration", "assembler"])
            # iteration 1: no previous corpus, no anchors -> only the mined rows are excluded
            first = dict(request, previous_jsonl=None, previous_sha256=None, selector_command=[sys.executable, "selector.py"])
            plan1 = render_iteration_mining_runner.build_plan(**first)
            self.assertEqual(plan1["classification_calibration"]["exclusions"], [str(root / "mined.jsonl")])
            # renderer-owned options are rejected; absent command -> no stage and no assembler flag
            for owned in ("--output", "--manifest", "--exclude-identities-file"):
                with self.assertRaisesRegex(ValueError, "owned by this renderer"):
                    render_iteration_mining_runner.build_plan(**dict(request, classification_calibration_command=[
                        *request["classification_calibration_command"], owned, "x"]))
            with self.assertRaisesRegex(ValueError, "owned by this renderer"):
                render_iteration_mining_runner.build_plan(**dict(request, selector_command=[
                    *request["selector_command"], "--classification-calibration-jsonl", "x"]))
            plain = render_iteration_mining_runner.build_plan(**dict(request, classification_calibration_command=None))
            self.assertIsNone(plain["classification_calibration"])
            self.assertNotIn("--classification-calibration-jsonl", plain["assembler"]["command"])


if __name__ == "__main__":
    unittest.main()
