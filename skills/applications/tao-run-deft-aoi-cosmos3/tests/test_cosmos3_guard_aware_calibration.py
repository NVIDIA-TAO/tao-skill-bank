# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Feature B3: guard-aware detection calibration selection and no anchor over-fill under the
empty-answer guard (run v12_p4b_emptyguard_r4 iteration 4, snapshot 6fb5dcbc).

Feature B3.1: contract-driven calibration feed reserve and best-effort substitution with
recorded headroom overflow (run v12_p4b_emptyguard_r5 iteration 1, snapshot 3c4e042b).

Feature B4: detection calibration rows yield to the growth slot under the guard instead of
failing the iteration (run v12_p4b_emptyguard_r6 iteration 2, snapshot bab8537b).

Regenerate the golden fixture from a snapshot whose behaviour is the reference:
``python3 tests/test_cosmos3_guard_aware_calibration.py --write-golden``.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import pathlib
import sys
import tempfile
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import anchor_rows  # noqa: E402
import assemble_training_json as atj  # noqa: E402
import atomic_samples  # noqa: E402
import defect_detection_ablation as dda  # noqa: E402
import init_deft_state  # noqa: E402
import render_iteration_mining_runner as runner  # noqa: E402
import select_detection_calibration as sdc  # noqa: E402
from test_cosmos3_classification_calibration import _det_row, _row as _mcq_row, _write  # noqa: E402
from test_cosmos3_detection_calibration_contract import _row as _pool_row  # noqa: E402
from test_cosmos3_empty_answer_guard import _ref_det_row  # noqa: E402

DD = dda.DEFECT_DETECTION_TASK
REF = dda.REFERENCE_DEFECT_DETECTION_TASK
REF_DC = "Ref_based Defect Classification"
CC = "Component Classification"
BOX = [{"bbox_2d": [0, 0, 10, 10], "label": "x"}]
CAL, KIND = dda.CALIBRATION_MARK, dda.CALIBRATION_KIND_MARK
GOLDEN = SKILL_ROOT / "tests" / "fixtures" / "guard_aware_calibration_golden.json"
MEDIA = pathlib.Path("/data")


def _guard(overall: float | None = 0.30, per_task: dict | None = None, mode: str | None = None) -> dict:
    return atj.validate_empty_answer_guard_config(overall, per_task, None, mode)


def _cand(record: dict, *, evidence: list[str] | None = None, route_tier: str = "strict", contrast: float = 0.5) -> dict:
    """A routed candidate for ``defect_detection_ablation.materialize`` (synthetic paths, trusted identity)."""
    user = next(message for message in record["messages"] if message.get("role") == "user")
    image = [item["image"] for item in user["content"] if item.get("type") == "image"][-1]
    sample = atomic_samples.sample_from_record(record, media_root=MEDIA, context=str(record["id"]))
    sha = hashlib.sha256(f"fixture:{sample['atomic_sample_id']}".encode()).hexdigest()
    return {
        "filepath": image,
        "atomic_sample_id": sample["atomic_sample_id"],
        "sample_kind": sample["sample_kind"],
        "source_image_paths": sample["image_paths"],
        "content_sha256": sha,
        "perceptual_hash": sha[:16],
        "route_tier": route_tier,
        "routed_task_types": [record["task_type"]],
        "defect_detection_evidence": list(evidence or []),
        "local_contrast": contrast,
        "max_cosine_similarity": 0.95,
        "is_replay": False,
    }


def _rows_sha256(rows: list[dict]) -> str:
    return hashlib.sha256("".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows).encode()).hexdigest()


def _materialize(prior: list[dict], source: list[dict], candidates: list[dict], *,
                 reference_proxy_empty_rate: float = 0.542, **extra) -> tuple[list[dict], dict]:
    return dda.materialize(
        candidate_rows=candidates, source_records=source, previous_records=prior or None, validation_records=[],
        media_root=MEDIA, max_rows=1536, row_multiple=768, defect_detection_fraction=0.5, proxy_empty_rate=0.25,
        epochs=1, global_batch=768, near_duplicate_hamming_distance=None,
        reference_proxy_empty_rate=reference_proxy_empty_rate,
        single_image_calibration_max_empty=512, single_image_calibration_max_few=512, reference_calibration_total=500,
        zero_new_candidate_policy="skip_exhausted", **extra,
    )


def _iteration_candidates(*, fewbox: int, no_change: int, changed: int) -> tuple[list[dict], list[dict]]:
    """One calibration-dominated iteration: 512 empty boards, ``fewbox`` few-box boards, ``no_change``
    / ``changed`` reference pairs (all calibration tier), 9 mined pairs (5 empty) and 3 Ref DC rows."""
    source: list[dict] = []
    candidates: list[dict] = []

    def add(record: dict, **kwargs) -> None:
        source.append(record)
        candidates.append(_cand(record, **kwargs))

    for i in range(512):
        add(_det_row(f"c-dd-e-{i:04d}", dataset="pool"), evidence=[dda.CALIBRATION_EMPTY_EVIDENCE], route_tier="calibration")
    for i in range(fewbox):
        boxes = [{"bbox_2d": [i % 50, 0, i % 50 + 5 + i % 3, 7], "label": ("open", "short")[i % 2]}]
        add(_det_row(f"c-dd-f-{i:04d}", boxes, dataset=f"pool{i % 3}"), evidence=[dda.CALIBRATION_FEW_EVIDENCE],
            route_tier="calibration", contrast=(i % 97) / 100.0)
    for i in range(no_change):
        add(_ref_det_row(f"c-ref-e-{i:04d}", dataset="pool"),
            evidence=[dda.CALIBRATION_EMPTY_EVIDENCE, dda.REFERENCE_NO_CHANGE_EVIDENCE], route_tier="calibration")
    for i in range(changed):
        add(_ref_det_row(f"c-ref-c-{i:04d}", BOX, dataset="pool"), evidence=[dda.CALIBRATION_FEW_EVIDENCE], route_tier="calibration")
    for i in range(5):
        add(_ref_det_row(f"m-ref-e-{i:04d}", dataset="mine"))
    for i in range(4):
        add(_ref_det_row(f"m-ref-p-{i:04d}", BOX, dataset="mine"))
    for i in range(3):
        add(_mcq_row(f"m-rdc-{i:04d}", REF_DC, answer="A", dataset="mine", two_images=True))
    return source, candidates


# ---------------------------------------------------------------------------------------------
# r4 iteration-4 numbers (synthetic): prior corpus 5,376 rows at 0.294 empty share against the
# 0.30 cap; this iteration 1,536 rows that are almost all detection calibration (DD 512 empty +
# 512 few-box, Ref DD 500 pairs at the KPI empty rate 0.542 + 9 mined, Ref DC 3).
# ---------------------------------------------------------------------------------------------
def _r4_fixture() -> tuple[list[dict], list[dict], list[dict]]:
    prior = [_det_row(f"p-dd-e-{i:04d}", dataset="prev") for i in range(880)]
    prior += [_det_row(f"p-dd-p-{i:04d}", BOX, dataset="prev") for i in range(1120)]
    prior += [_ref_det_row(f"p-ref-e-{i:04d}", dataset="prev") for i in range(700)]
    prior += [_ref_det_row(f"p-ref-p-{i:04d}", BOX, dataset="prev") for i in range(800)]
    prior += [_mcq_row(f"p-cc-{i:04d}", CC, answer="A", dataset="prev") for i in range(1876)]
    assert len(prior) == 5376
    source, candidates = _iteration_candidates(fewbox=1024, no_change=400, changed=500)
    return prior, source, candidates


# ---------------------------------------------------------------------------------------------
# r5 iteration-1 numbers (synthetic, Feature B3.1): prior corpus 6,144 rows (8 global batches)
# at 0.280 empty share; the caps leave 426 single-image / 151 no-change empties against the KPI
# targets 512 / 278 (headroom overall floor(0.30 * 7,680 - 1,727) = 577, DD
# floor(0.45 * 3,024 - 880) = 480, Ref DD floor(0.50 * 2,035 - 847) = 170; the shared 577 splits
# 426 / 151 by largest remainder), so the guard-aware split asks for 86 few-box rows and 127
# changed pairs that today's feed (512 few-box, 222 changed) does not hold. ``reserve`` supplies
# the contract-driven feed instead (1,024 few-box, 500 changed).
# KPI reference empty rate 0.556 -> floor(500 * 0.556 + 0.5) = 278 no-change, 222 changed.
# ---------------------------------------------------------------------------------------------
R5_REFERENCE_RATE = 0.556


def _r5_fixture(*, reserve: bool) -> tuple[list[dict], list[dict], list[dict]]:
    prior = [_det_row(f"p-dd-e-{i:04d}", dataset="prev") for i in range(880)]
    prior += [_det_row(f"p-dd-p-{i:04d}", BOX, dataset="prev") for i in range(1120)]
    prior += [_ref_det_row(f"p-ref-e-{i:04d}", dataset="prev") for i in range(842)]
    prior += [_ref_det_row(f"p-ref-p-{i:04d}", BOX, dataset="prev") for i in range(684)]
    prior += [_mcq_row(f"p-cc-{i:04d}", CC, answer="A", dataset="prev") for i in range(2618)]
    assert len(prior) == 6144 == 8 * 768
    source, candidates = _iteration_candidates(
        fewbox=1024 if reserve else 512, no_change=278, changed=500 if reserve else 222
    )
    return prior, source, candidates


# ---------------------------------------------------------------------------------------------
# Golden scenarios (guard off): the materializer's hybrid calibration selection and the
# assembler's anchor reservation incl. the leftover -> extra anchors fill.
# ---------------------------------------------------------------------------------------------
def _golden_materializer_rows() -> tuple[list[dict], dict]:
    source: list[dict] = []
    candidates: list[dict] = []

    def add(record: dict, **kwargs) -> None:
        source.append(record)
        candidates.append(_cand(record, **kwargs))

    for i in range(6):
        add(_det_row(f"g-dd-e-{i}", dataset="pool"), evidence=[dda.CALIBRATION_EMPTY_EVIDENCE], route_tier="calibration")
    for i in range(6):
        add(_det_row(f"g-dd-f-{i}", [{"bbox_2d": [i, 0, i + 4, 6], "label": "open"}], dataset=f"pool{i % 2}"),
            evidence=[dda.CALIBRATION_FEW_EVIDENCE], route_tier="calibration", contrast=0.3 + i / 20)
    for i in range(4):
        add(_det_row(f"g-dd-s-{i}", [{"bbox_2d": [i, i, i + 9, i + 9], "label": "short"}], dataset="mine"),
            evidence=["hard_positive_proxy_false_negative"], contrast=0.6 + i / 20)
    for i in range(4):
        add(_ref_det_row(f"g-ref-e-{i}", dataset="pool"),
            evidence=[dda.CALIBRATION_EMPTY_EVIDENCE, dda.REFERENCE_NO_CHANGE_EVIDENCE], route_tier="calibration")
    for i in range(4):
        add(_ref_det_row(f"g-ref-c-{i}", BOX, dataset="pool"), evidence=[dda.CALIBRATION_FEW_EVIDENCE], route_tier="calibration")
    add(_ref_det_row("g-ref-m-e", dataset="mine"))
    add(_ref_det_row("g-ref-m-p", BOX, dataset="mine"))
    add(_mcq_row("g-cc", CC, answer="A", dataset="mine"))
    add(_det_row("g-cd", BOX, task="Component Detection", dataset="mine"))
    add(_mcq_row("g-dc", answer="A", dataset="mine"))
    add(_mcq_row("g-rdc", REF_DC, answer="A", dataset="mine", two_images=True))
    return dda.materialize(
        candidate_rows=candidates, source_records=source, validation_records=[], media_root=MEDIA,
        max_rows=32, minimum_rows=8, row_multiple=4, defect_detection_fraction=0.5, proxy_empty_rate=0.25, epochs=1,
        global_batch=4, near_duplicate_hamming_distance=None, reference_proxy_empty_rate=0.5,
        single_image_calibration_max_empty=4, single_image_calibration_max_few=4, reference_calibration_total=6,
    )


def _golden_assembler_rows(root: pathlib.Path) -> tuple[list[dict], dict]:
    previous = _write(root / "train0.jsonl", [_det_row(f"o-dd-{i}", BOX, dataset="old") for i in range(8)]
                      + [_mcq_row(f"o-cc-{i}", CC, answer="A", dataset="old") for i in range(8)])
    mined_rows = [_det_row(f"n-pos-{i:02d}", BOX) for i in range(20)] + [_det_row(f"n-empty-{i:02d}") for i in range(12)]
    mined_rows += [{**_det_row(f"n-cal-{i:02d}", dataset="cal"), CAL: True, KIND: "detection"} for i in range(6)]
    mined = _write(root / "mined.jsonl", mined_rows)
    anchors = [_mcq_row(f"a-cc-{i:03d}", CC, answer="A", dataset=f"pool{i % 2}") for i in range(100)]
    anchors += [_det_row(f"a-dd-{i:03d}", dataset=f"pool{i % 2}") for i in range(50)]
    source = _write(root / "anchor_candidates.jsonl", anchors)
    kpi = _write(root / "kpi.jsonl", [_det_row(f"k-{i}") for i in range(50)] + [_mcq_row(f"kc-{i}", CC, answer="A") for i in range(50)])
    cfg = anchor_rows.validate_anchor_config(0.10, source, kpi, None)
    return atj.assemble(
        previous, mined, previous_sha256=atj.sha256_file(previous), validation_paths=[], media_root=root,
        anchor_config=cfg, max_rows=64, row_multiple=8,
    )


def write_golden() -> dict:
    rows, manifest = _golden_materializer_rows()
    with tempfile.TemporaryDirectory() as temporary:
        arows, asummary = _golden_assembler_rows(pathlib.Path(temporary))
    golden = {
        "materializer": {"sha256": _rows_sha256(rows), "rows": len(rows), "ids": [r["id"] for r in rows],
                         "verified": manifest["verified"], "row_counts_total": manifest["row_counts"]["total"]},
        "assembler": {"sha256": _rows_sha256(arows), "rows": len(arows), "ids": [r["id"] for r in arows],
                      "materialized_anchor_records": asummary["materialized_anchor_records"]},
    }
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(golden, indent=2) + "\n", encoding="utf-8")
    return golden


class HeadroomAllocationTests(unittest.TestCase):
    def test_headroom_is_floor_of_cap_times_rows_minus_non_calibration_empties(self) -> None:
        self.assertEqual(dda.empty_headroom(cap=0.30, rows=6912, empty_rows=1585), 488)
        self.assertEqual(dda.empty_headroom(cap=0.45, rows=3024, empty_rows=880), 480)
        self.assertEqual(dda.empty_headroom(cap=0.50, rows=2009, empty_rows=705), 299)
        self.assertEqual(dda.empty_headroom(cap=0.30, rows=100, empty_rows=40), 0)  # already over the cap
        self.assertEqual(dda.empty_headroom(cap=0.30, rows=10, empty_rows=0), 3)  # 3 / 10 == 0.30 is within

    def test_shared_overall_headroom_is_split_by_largest_remainder_over_the_task_limits(self) -> None:
        self.assertEqual(dda.allocate_empty_headroom({DD: 480, REF: 271}, 488), {DD: 312, REF: 176})
        self.assertEqual(dda.allocate_empty_headroom({DD: 480, REF: 271}, 800), {DD: 480, REF: 271})  # not binding
        self.assertEqual(dda.allocate_empty_headroom({DD: 10, REF: 10}, 0), {DD: 0, REF: 0})
        self.assertEqual(dda.allocate_empty_headroom({DD: 0, REF: 5}, 3), {DD: 0, REF: 3})
        self.assertEqual(dda.allocate_empty_headroom({DD: 5, REF: 5}, 5), {DD: 3, REF: 2})  # tie -> task name order


class GuardAwareCalibrationTests(unittest.TestCase):
    def test_r4_iteration_4_numbers_assemble_without_failure_and_within_caps(self) -> None:
        prior, source, candidates = _r4_fixture()
        guard = _guard(0.30, {DD: 0.45, REF: 0.50})
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            previous = _write(root / "train3.jsonl", prior)
            previous_sha = atj.sha256_file(previous)
            # Guard-aware off: 512 + 271 empty calibration rows on a corpus at 0.294 -> the guard
            # trims 421 rows, the aligned size drops by one global batch and the 1,103 protected
            # calibration rows left no longer fit its 768 slots. Before Feature B4 the iteration
            # failed closed (r4); now 347 more Defect Detection negatives yield to the growth slot.
            rows0, manifest0 = _materialize(prior, source, candidates)
            self.assertTrue(manifest0["verified"])
            self.assertFalse(manifest0["calibration_guard_aware"])
            self.assertEqual(manifest0["calibration_empty_selected"], {DD: 512, REF: 271, "total": 783})
            self.assertEqual(manifest0["calibration_fewbox_substituted"], {DD: 0, REF: 0, "total": 0})
            self.assertEqual(set(manifest0["calibration_empty_headroom"].values()), {None})
            mined0 = _write(root / "mined0.jsonl", rows0)
            out0, summary0 = atj.assemble(previous, mined0, previous_sha256=previous_sha, validation_paths=[], media_root=root,
                                          row_multiple=768, empty_answer_guard=guard)
            self.assertEqual((len(out0), summary0["growth_rows"]), (6144, 768))
            report0 = summary0["empty_answer_guard"]
            self.assertEqual((report0["status"], report0["passes"]), ("trimmed_to_caps", 2))
            self.assertEqual(report0["rows_trimmed_by_source"], {"detection_calibration_negative": 421})
            self.assertEqual(summary0["calibration_rows_dropped_for_cap"],
                             {"by_task": {DD: 347}, "by_kind": {"empty": 347, "non_empty": 0}, "total": 347})
            self.assertEqual(summary0["calibration_rows_kept"], {DD: 527, REF: 229})
            self.assertEqual(summary0["operator_attention"], ["calibration_yielded"])
            self.assertLessEqual(report0["after"]["overall_share"], 0.30)
            # Guard-aware: headroom overall floor(0.30 * 6912 - 1585) = 488, DD floor(0.45 * 3024 - 880) = 480,
            # Ref DD floor(0.50 * 2009 - 705) = 299; the shared 488 splits 312 / 176 by largest remainder.
            rows, manifest = _materialize(prior, source, candidates, empty_answer_guard=guard, calibration_guard_aware=True)
            self.assertTrue(manifest["verified"], manifest["verification"])
            self.assertEqual(len(rows), 1536)
            self.assertTrue(manifest["calibration_guard_aware"])
            self.assertEqual(manifest["calibration_empty_headroom"],
                             {"overall": 488, f"task:{DD}": 480, f"task:{REF}": 299})
            self.assertEqual(manifest["calibration_empty_selected"], {DD: 312, REF: 176, "total": 488})
            self.assertEqual(manifest["calibration_fewbox_substituted"], {DD: 200, REF: 95, "total": 295})
            single = manifest["single_image_calibration"]
            self.assertEqual((single["selected_empty"], single["selected_few_box"], single["selected_total"]), (312, 712, 1024))
            reference = manifest["reference_calibration"]
            self.assertEqual((reference["selected_no_change"], reference["selected_changed"], reference["selected_total"]), (176, 324, 500))
            self.assertEqual((reference["target_no_change"], reference["kpi_target_no_change"]), (176, 271))
            self.assertEqual(manifest["row_counts"]["total"], 1536)
            self.assertEqual(manifest["new_rows_empty"], 488 + 5)  # calibration empties + the 5 mined reference empties
            detail = manifest["guard_aware_calibration"]
            self.assertEqual(detail["ledger"]["previous"], {"rows": 5376, "empty_rows": 1580})
            self.assertEqual(detail["ledger"]["current_non_calibration"], {"rows": 12, "empty_rows": 5})
            self.assertEqual(detail["kpi_empty_targets"], {DD: 512, REF: 271})
            self.assertEqual(detail["fewbox_shortfall"], {DD: 0, REF: 0})
            # every calibration row still carries the markers, the count contract is unchanged
            self.assertEqual(sum(r.get(CAL) is True for r in rows), 1524)
            mined = _write(root / "mined4.jsonl", rows)
            out, summary = atj.assemble(previous, mined, previous_sha256=previous_sha, validation_paths=[], media_root=root,
                                        row_multiple=768, empty_answer_guard=guard, calibration_guard_aware=True)
            self.assertEqual(len(out), 6912)
            self.assertEqual(summary["growth_rows"], 1536)
            report = summary["empty_answer_guard"]
            self.assertEqual(report["status"], "within_caps")
            self.assertEqual(report["rows_trimmed_total"], 0)
            self.assertTrue(report["calibration_guard_aware"])
            self.assertLessEqual(report["after"]["overall_share"], 0.30)
            self.assertLessEqual(report["after"]["per_task"][DD], 0.45)
            self.assertLessEqual(report["after"]["per_task"][REF], 0.50)
            self.assertEqual(summary["answer_profile"]["empty_rows"], 1580 + 488 + 5)

    def test_guard_aware_requires_caps_and_the_fixed_calibration_slot(self) -> None:
        _, source, candidates = _golden_materializer_inputs()
        with self.assertRaisesRegex(ValueError, "requires an enabled empty-answer guard"):
            dda.materialize(
                candidate_rows=candidates, source_records=source, validation_records=[], media_root=MEDIA,
                max_rows=32, row_multiple=4, defect_detection_fraction=0.5, proxy_empty_rate=0.25, epochs=1,
                global_batch=4, near_duplicate_hamming_distance=None, reference_proxy_empty_rate=0.5,
                single_image_calibration_max_empty=4, single_image_calibration_max_few=4, reference_calibration_total=6,
                calibration_guard_aware=True,
            )
        with self.assertRaisesRegex(ValueError, "fixed calibration"):
            dda.materialize(
                candidate_rows=candidates, source_records=source, validation_records=[], media_root=MEDIA,
                max_rows=32, row_multiple=4, defect_detection_fraction=0.5, proxy_empty_rate=0.25, epochs=1,
                global_batch=4, near_duplicate_hamming_distance=None,
                empty_answer_guard=_guard(0.30), calibration_guard_aware=True,
            )


class FeedReserveAndOverflowTests(unittest.TestCase):
    """Feature B3.1 (run v12_p4b_emptyguard_r5 iteration 1): the calibration feed carries a
    contract-driven reserve of non-empty rows, and when the reserve runs out the slot is filled
    with empty candidates beyond the headroom instead of failing the materializer."""

    R5_HEADROOM = {"overall": 577, f"task:{DD}": 480, f"task:{REF}": 170}

    def test_r5_todays_feed_fills_the_slot_with_overflow_empties_and_the_guard_trims_them(self) -> None:
        prior, source, candidates = _r5_fixture(reserve=False)
        guard = _guard(0.30, {DD: 0.45, REF: 0.50})
        rows, manifest = _materialize(prior, source, candidates, reference_proxy_empty_rate=R5_REFERENCE_RATE,
                                      empty_answer_guard=guard, calibration_guard_aware=True)
        # the slot is full (1,024 + 500), so the existing total-slot verification passes
        self.assertTrue(manifest["verified"], manifest["verification"])
        self.assertEqual(len(rows), 1536)
        self.assertEqual(manifest["calibration_empty_headroom"], self.R5_HEADROOM)
        detail = manifest["guard_aware_calibration"]
        self.assertEqual(detail["kpi_empty_targets"], {DD: 512, REF: 278})
        self.assertEqual(detail["headroom_empty_targets"], {DD: 426, REF: 151})
        # the reserve had 0 few-box / 0 changed rows beyond today's 512 / 222: nothing substituted,
        # the 86 / 127 remaining slot rows are empties beyond the headroom (recorded, not fatal)
        self.assertEqual(manifest["calibration_empty_selected"], {DD: 512, REF: 278, "total": 790})
        self.assertEqual(manifest["calibration_fewbox_substituted"], {DD: 0, REF: 0, "total": 0})
        self.assertEqual(manifest["calibration_headroom_overflow_rows"], {DD: 86, REF: 127, "total": 213})
        self.assertEqual(detail["fewbox_shortfall"], {DD: 86, REF: 127})
        self.assertEqual(detail["status"], "substituted_with_overflow")
        single = manifest["single_image_calibration"]
        self.assertEqual((single["selected_empty"], single["selected_few_box"], single["selected_total"]), (512, 512, 1024))
        self.assertEqual((single["max_few_box_effective"], single["empty_beyond_headroom"]), (512, 86))
        reference = manifest["reference_calibration"]
        self.assertEqual((reference["selected_no_change"], reference["selected_changed"], reference["selected_total"]), (278, 222, 500))
        self.assertEqual((reference["target_no_change"], reference["kpi_target_no_change"], reference["no_change_beyond_headroom"]), (278, 278, 127))
        self.assertEqual(manifest["new_rows_empty"], 790 + 5)
        self.assertEqual(sum(r.get(CAL) is True for r in rows), 1524)
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            previous = _write(root / "train0.jsonl", prior)
            previous_sha = atj.sha256_file(previous)
            mined = _write(root / "mined1.jsonl", rows)
            # the assembler's guard trims the overflow empties (calibration negatives first); with
            # no back-fill candidates every trimmed row also shrinks the denominator, so it trims
            # past the 213 overflow rows until the caps hold
            out, summary = atj.assemble(previous, mined, previous_sha256=previous_sha, validation_paths=[], media_root=root,
                                        row_multiple=8, empty_answer_guard=guard, calibration_guard_aware=True)
            report = summary["empty_answer_guard"]
            self.assertEqual(report["status"], "trimmed_to_caps")
            self.assertGreaterEqual(report["rows_trimmed_total"], 213)
            self.assertEqual(set(report["rows_trimmed_by_source"]), {"detection_calibration_negative"})
            self.assertLessEqual(report["after"]["overall_share"], 0.30)
            self.assertLessEqual(report["after"]["per_task"][DD], 0.45)
            self.assertLessEqual(report["after"]["per_task"][REF], 0.50)
            self.assertEqual(len(out), 6144 + 1536 - report["rows_trimmed_total"] - (1536 - report["rows_trimmed_total"]) % 8)
            # at the pinned 768-row global batch the trimmed iteration no longer fills a batch that
            # holds its 1,189 protected calibration rows; before Feature B4 the assembler failed
            # closed here, now 433 more Defect Detection negatives yield and the step grows by one
            # batch (the reserve feed below still keeps the full 1,536-row growth)
            out768, summary768 = atj.assemble(previous, mined, previous_sha256=previous_sha, validation_paths=[], media_root=root,
                                              row_multiple=768, empty_answer_guard=guard, calibration_guard_aware=True)
            self.assertEqual((len(out768), summary768["growth_rows"]), (6912, 768))
            self.assertEqual(summary768["empty_answer_guard"]["rows_trimmed_total"], 335)
            self.assertEqual(summary768["calibration_rows_dropped_for_cap"],
                             {"by_task": {DD: 433}, "by_kind": {"empty": 433, "non_empty": 0}, "total": 433})
            self.assertEqual(summary768["calibration_rows_kept"], {DD: 534, REF: 222})
            self.assertTrue(summary768["calibration_yielded"])
            self.assertLessEqual(summary768["empty_answer_guard"]["after"]["overall_share"], 0.30)

    def test_r5_reserve_feed_substitutes_without_overflow_and_assembles_at_the_global_batch(self) -> None:
        prior, source, candidates = _r5_fixture(reserve=True)
        guard = _guard(0.30, {DD: 0.45, REF: 0.50})
        rows, manifest = _materialize(prior, source, candidates, reference_proxy_empty_rate=R5_REFERENCE_RATE,
                                      empty_answer_guard=guard, calibration_guard_aware=True)
        self.assertTrue(manifest["verified"], manifest["verification"])
        self.assertEqual(len(rows), 1536)
        self.assertEqual(manifest["calibration_empty_headroom"], self.R5_HEADROOM)
        detail = manifest["guard_aware_calibration"]
        self.assertEqual(detail["headroom_empty_targets"], {DD: 426, REF: 151})
        self.assertEqual(manifest["calibration_empty_selected"], {DD: 426, REF: 151, "total": 577})
        self.assertEqual(manifest["calibration_fewbox_substituted"], {DD: 86, REF: 127, "total": 213})
        self.assertEqual(manifest["calibration_headroom_overflow_rows"], {DD: 0, REF: 0, "total": 0})
        self.assertEqual(detail["fewbox_shortfall"], {DD: 0, REF: 0})
        self.assertEqual(detail["status"], "substituted")
        single = manifest["single_image_calibration"]
        self.assertEqual((single["selected_empty"], single["selected_few_box"], single["selected_total"]), (426, 598, 1024))
        self.assertEqual((single["max_few_box_effective"], single["empty_beyond_headroom"]), (598, 0))
        reference = manifest["reference_calibration"]
        self.assertEqual((reference["selected_no_change"], reference["selected_changed"], reference["selected_total"]), (151, 349, 500))
        self.assertEqual((reference["target_no_change"], reference["kpi_target_no_change"], reference["no_change_beyond_headroom"]), (151, 278, 0))
        self.assertEqual(manifest["new_rows_empty"], 577 + 5)
        self.assertEqual(sum(r.get(CAL) is True for r in rows), 1524)
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            previous = _write(root / "train0.jsonl", prior)
            mined = _write(root / "mined1.jsonl", rows)
            out, summary = atj.assemble(previous, mined, previous_sha256=atj.sha256_file(previous), validation_paths=[],
                                        media_root=root, row_multiple=768, empty_answer_guard=guard, calibration_guard_aware=True)
            self.assertEqual(len(out), 6144 + 1536)
            self.assertEqual(summary["growth_rows"], 1536)
            report = summary["empty_answer_guard"]
            self.assertEqual(report["status"], "within_caps")
            self.assertEqual(report["rows_trimmed_total"], 0)
            self.assertLessEqual(report["after"]["overall_share"], 0.30)
            self.assertLessEqual(report["after"]["per_task"][DD], 0.45)
            self.assertLessEqual(report["after"]["per_task"][REF], 0.50)

    def test_init_records_the_contract_driven_feed_reserve_for_guard_aware_on_and_off(self) -> None:
        from test_cosmos3_init_state_contract import Cosmos3InitStateContractTests as Base
        slot = ("--single-image-calibration-max-empty", "512", "--single-image-calibration-max-few", "512",
                "--reference-calibration-total", "500")
        cap = ("--max-empty-answer-share", "0.30")
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = Base._workspace(root)  # KPI reference empty rate 0.5 -> 250 no-change pairs

            def contract(name: str, *extra: str) -> dict:
                self.assertEqual(init_deft_state.main(Base._argv(root / name, workspace, *slot, *extra)), 0)
                state = json.loads((root / name / "results/deft_state.json").read_text())
                self.assertEqual(state["config"]["mining"]["empty_answer_guard"]["calibration_guard_aware"],
                                 name.startswith("on"))
                return state["config"]["mining"]["calibration_quota_contract"]

            on = contract("on", *cap)
            self.assertEqual(on["feed_bucket_quotas"], {
                "non_reference_based": {"empty": 512, "few": 1024},
                "reference_based": {"empty": 250, "few": 500},
            })
            self.assertIn("feed_bucket_quotas", on["owner"])
            self.assertIn("reserve", on["feed_bucket_quotas_rule"])
            for name, extra in (("off-flag", (*cap, "--calibration-guard-aware", "off")), ("off-no-cap", ())):
                off = contract(name, *extra)
                self.assertEqual(off["feed_bucket_quotas"], {
                    "non_reference_based": {"empty": 512, "few": 512},
                    "reference_based": {"empty": 250, "few": 250},
                })
                self.assertEqual((off["single_image_max_empty"], off["single_image_max_few_box"]), (512, 512))
                self.assertEqual((off["reference_empty"], off["reference_few_box"]), (250, 250))

    def test_select_calibration_honours_the_feed_reserve_and_records_it(self) -> None:
        one_box = [{"bbox_2d": [0, 0, 1, 1], "label": "x"}]
        proxy = [_pool_row("proxy-single-empty", []), _pool_row("proxy-single-few", one_box),
                 _pool_row("proxy-ref-empty", [], task=REF), *[_pool_row(f"proxy-ref-few{i}", one_box, task=REF) for i in range(3)]]
        source = [*[_pool_row(f"single-empty{i}", []) for i in range(3)], *[_pool_row(f"single-few{i}", one_box) for i in range(6)],
                  *[_pool_row(f"ref-empty{i}", [], task=REF) for i in range(2)], *[_pool_row(f"ref-few{i}", one_box, task=REF) for i in range(5)]]
        contract = {"non_reference_based": {"empty": 2, "few": 3}, "reference_based": {"empty": 1, "few": 3}}
        with tempfile.TemporaryDirectory() as temporary:
            media_root = pathlib.Path(temporary)
            (media_root / "images").mkdir()
            from PIL import Image
            for record_index, record in enumerate(source):
                for image_index, item in enumerate(record["messages"][0]["content"]):
                    if item.get("type") == "image":
                        Image.new("RGB", (2, 2), color=(record_index, image_index, 0)).save(media_root / item["image"])
            rates = sdc.derive_proxy_empty_rates(proxy)
            common = dict(media_root=media_root, cohort_bucket_quotas=contract, cohort_rates=rates,
                          pair_assets_dir=media_root / "pair-assets", max_boxes=2)
            # default: today's behaviour, reserve 0
            selected, summary = sdc.select_calibration(source, **common)
            self.assertEqual(len(selected), 9)
            for cohort in ("non_reference_based", "reference_based"):
                self.assertEqual(summary["cohorts"][cohort]["feed_bucket_quotas"], contract[cohort])
                self.assertEqual(summary["cohorts"][cohort]["feed_reserve_rows"], {"empty": 0, "few": 0, "total": 0})
            # reserve: the feed carries the contract rows plus the reserve; the contract stays the fail-closed floor
            feed = {"non_reference_based": {"empty": 2, "few": 5}, "reference_based": {"empty": 1, "few": 4}}
            selected, summary = sdc.select_calibration(source, feed_bucket_quotas=feed, **common)
            self.assertEqual(len(selected), 12)
            self.assertEqual(summary["schema_version"], "detection_calibration_v3")
            single, reference = summary["cohorts"]["non_reference_based"], summary["cohorts"]["reference_based"]
            self.assertEqual((single["requested_empty"], single["requested_few_box"]), (2, 3))
            self.assertEqual((single["selected_empty"], single["selected_few_box"], single["selected_total"]), (2, 5, 7))
            self.assertEqual(single["feed_bucket_quotas"], feed["non_reference_based"])
            self.assertEqual(single["feed_reserve_rows"], {"empty": 0, "few": 2, "total": 2})
            self.assertEqual((reference["requested_empty"], reference["requested_few_box"]), (1, 3))
            self.assertEqual((reference["selected_empty"], reference["selected_few_box"], reference["selected_total"]), (1, 4, 5))
            self.assertEqual(reference["feed_reserve_rows"], {"empty": 0, "few": 1, "total": 1})
            self.assertEqual(summary["selected_total"], 12)
            self.assertEqual(sum(row["calibration_box_count"] == 0 for row in selected), 3)
            # a reserve the pool cannot fill is best-effort (no error); the contract rows still fail closed
            over = {"non_reference_based": {"empty": 2, "few": 9}, "reference_based": {"empty": 1, "few": 9}}
            selected, summary = sdc.select_calibration(source, feed_bucket_quotas=over, **common)
            self.assertEqual(len(selected), 14)
            self.assertEqual(summary["cohorts"]["non_reference_based"]["feed_reserve_rows"]["few"], 3)
            self.assertEqual(summary["cohorts"]["reference_based"]["feed_reserve_rows"]["few"], 2)
            with self.assertRaisesRegex(ValueError, "reference calibration content-unique shortfall"):
                sdc.select_calibration(source, feed_bucket_quotas=over, **{**common, "cohort_bucket_quotas": {
                    "non_reference_based": {"empty": 2, "few": 3}, "reference_based": {"empty": 1, "few": 6}}})
            # the feed may not undercut the contract, and it only exists for the fixed-slot contract
            with self.assertRaisesRegex(ValueError, "feed_bucket_quotas"):
                sdc.select_calibration(source, feed_bucket_quotas={"non_reference_based": {"empty": 1, "few": 3},
                                                                   "reference_based": {"empty": 1, "few": 3}}, **common)
            with self.assertRaisesRegex(ValueError, "feed_bucket_quotas"):
                sdc.select_calibration(source, feed_bucket_quotas=feed, media_root=media_root, cohort_rates=rates,
                                       cohort_quotas={"non_reference_based": 5, "reference_based": 4},
                                       pair_assets_dir=media_root / "pair-assets")
            with self.assertRaisesRegex(ValueError, "feed_bucket_quotas"):
                sdc.select_calibration(source, feed_bucket_quotas={"non_reference_based": {"empty": 2, "few": 5}}, **common)


def _golden_materializer_inputs() -> tuple[list[dict], list[dict], list[dict]]:
    """The golden materializer scenario's inputs (for tests that need a small hybrid corpus)."""
    source: list[dict] = []
    candidates: list[dict] = []
    for i in range(6):
        record = _det_row(f"g-dd-e-{i}", dataset="pool")
        source.append(record)
        candidates.append(_cand(record, evidence=[dda.CALIBRATION_EMPTY_EVIDENCE], route_tier="calibration"))
    for i in range(6):
        record = _det_row(f"g-dd-f-{i}", [{"bbox_2d": [i, 0, i + 4, 6], "label": "open"}], dataset=f"pool{i % 2}")
        source.append(record)
        candidates.append(_cand(record, evidence=[dda.CALIBRATION_FEW_EVIDENCE], route_tier="calibration"))
    for i in range(4):
        record = _ref_det_row(f"g-ref-e-{i}", dataset="pool")
        source.append(record)
        candidates.append(_cand(record, evidence=[dda.CALIBRATION_EMPTY_EVIDENCE, dda.REFERENCE_NO_CHANGE_EVIDENCE], route_tier="calibration"))
    for i in range(4):
        record = _ref_det_row(f"g-ref-c-{i}", BOX, dataset="pool")
        source.append(record)
        candidates.append(_cand(record, evidence=[dda.CALIBRATION_FEW_EVIDENCE], route_tier="calibration"))
    for record in (_mcq_row("g-cc", CC, answer="A"), _det_row("g-cd", BOX, task="Component Detection"),
                   _mcq_row("g-dc", answer="A"), _mcq_row("g-rdc", REF_DC, answer="A", two_images=True)):
        source.append(record)
        candidates.append(_cand(record))
    return [], source, candidates


class GoldenSelectionTests(unittest.TestCase):
    def test_guard_off_materializer_and_assembler_selection_match_the_head_golden(self) -> None:
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        rows, manifest = _golden_materializer_rows()
        self.assertEqual([r["id"] for r in rows], golden["materializer"]["ids"])
        self.assertEqual(_rows_sha256(rows), golden["materializer"]["sha256"])
        self.assertEqual(manifest["verified"], golden["materializer"]["verified"])
        self.assertEqual(manifest["row_counts"]["total"], golden["materializer"]["row_counts_total"])
        # Feature B3.1 fields are inert when the guard-aware selection is off
        self.assertEqual(manifest["calibration_headroom_overflow_rows"], {DD: 0, REF: 0, "total": 0})
        self.assertIsNone(manifest["guard_aware_calibration"]["status"])
        self.assertIsNone(manifest["guard_aware_calibration"]["headroom_empty_targets"])
        self.assertEqual(manifest["single_image_calibration"]["empty_beyond_headroom"], 0)
        self.assertEqual(manifest["reference_calibration"]["no_change_beyond_headroom"], 0)
        with tempfile.TemporaryDirectory() as temporary:
            arows, asummary = _golden_assembler_rows(pathlib.Path(temporary))
        self.assertEqual([r["id"] for r in arows], golden["assembler"]["ids"])
        self.assertEqual(_rows_sha256(arows), golden["assembler"]["sha256"])
        self.assertEqual(asummary["materialized_anchor_records"], golden["assembler"]["materialized_anchor_records"])


class NoAnchorOverfillTests(unittest.TestCase):
    def _anchor_cfg(self, root: pathlib.Path) -> dict:
        anchors = [_mcq_row(f"a-cc-{i:03d}", CC, answer="A", dataset=f"pool{i % 2}") for i in range(300)]
        source = _write(root / "anchor_candidates.jsonl", anchors)
        kpi = _write(root / "kpi.jsonl", [_mcq_row(f"k-{i}", CC, answer="A") for i in range(50)])
        return anchor_rows.validate_anchor_config(0.10, source, kpi, None)

    def _run_iterations(self, root: pathlib.Path, *, anchor_overfill: str | None, iterations: int = 3) -> list[dict]:
        cfg = self._anchor_cfg(root)
        summaries: list[dict] = []
        previous: pathlib.Path | None = None
        for iteration in range(1, iterations + 1):
            mined_rows = [_det_row(f"i{iteration}-pos-{i:02d}", BOX) for i in range(30)]
            mined_rows += [_det_row(f"i{iteration}-empty-{i:02d}") for i in range(60)]
            mined = _write(root / f"mined{iteration}.jsonl", mined_rows)
            rows, summary = atj.assemble(
                previous, mined, previous_sha256=atj.sha256_file(previous) if previous else None, validation_paths=[],
                media_root=root, anchor_config=cfg, row_multiple=8, empty_answer_guard=_guard(0.30),
                anchor_overfill=anchor_overfill,
            )
            self.assertEqual(len(rows) % 8, 0)
            self.assertLessEqual(summary["empty_answer_guard"]["after"]["overall_share"], 0.30)
            summaries.append(summary)
            previous = _write(root / f"train{iteration}.jsonl", rows)
        return summaries

    def test_anchors_never_exceed_share_plus_tolerance_when_mined_candidates_run_out(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            forbid = self._run_iterations(root / "forbid", anchor_overfill=None)  # default under the guard
            for summary in forbid:
                self.assertEqual(summary["empty_answer_guard"]["anchor_overfill"], "forbid")
                self.assertEqual(summary["anchor"]["cap_reservation"]["anchor_overfill"], "forbid")
                self.assertLessEqual(summary["anchor"]["cumulative_share_rows"], 0.10 + 0.02)
                self.assertEqual(summary["operator_attention"], [])
                self.assertEqual(summary["growth_rows"], summary["output_records"] - summary["previous_records"])
                self.assertGreater(summary["growth_rows"], 0)
            # iteration 1: 30 + 16 mined rows + 10 anchor candidates after the guard; 56 would need 4
            # extra anchors -> the aligned size shrinks to 48 (5 anchors, 43 mined) instead
            first = forbid[0]
            self.assertEqual(first["output_records"], 48)
            self.assertEqual(first["materialized_anchor_records"], 5)
            self.assertEqual(first["anchor"]["cap_reservation"]["aligned_rows_shrunk_for_share"], 8)
            self.assertEqual(first["empty_answer_guard"]["status"], "trimmed_to_caps")
            allow = self._run_iterations(root / "allow", anchor_overfill="allow")
            self.assertEqual(allow[0]["empty_answer_guard"]["anchor_overfill"], "allow")
            self.assertEqual(allow[0]["output_records"], 56)
            self.assertEqual(allow[0]["materialized_anchor_records"], 10)
            self.assertGreater(allow[0]["anchor"]["cumulative_share_rows"], 0.12)
            self.assertEqual(allow[0]["operator_attention"], ["anchor_share_exceeded"])
            self.assertTrue(any(s["operator_attention"] == ["anchor_share_exceeded"] for s in allow))

    def test_default_without_the_guard_allows_overfill_and_reports_the_share(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            cfg = self._anchor_cfg(root)
            mined = _write(root / "mined.jsonl", [_det_row(f"pos-{i:02d}", BOX) for i in range(30)])
            rows, summary = atj.assemble(None, mined, validation_paths=[], media_root=root, anchor_config=cfg, row_multiple=8)
            self.assertEqual(summary["empty_answer_guard"]["anchor_overfill"], "allow")
            self.assertIsNone(summary["empty_answer_guard"]["calibration_guard_aware"])
            self.assertEqual(summary["anchor"]["cap_reservation"]["anchor_overfill"], "allow")
            self.assertEqual(summary["anchor"]["cap_reservation"]["aligned_rows_shrunk_for_share"], 0)
            self.assertAlmostEqual(summary["anchor"]["cumulative_share_rows"], summary["materialized_anchor_records"] / len(rows))
            self.assertEqual(summary["operator_attention"], [])
            self.assertEqual(summary["growth_rows"], len(rows))
            with self.assertRaisesRegex(ValueError, "anchor_overfill"):
                atj.assemble(None, mined, validation_paths=[], media_root=root, anchor_config=cfg, anchor_overfill="maybe")

    def test_zero_growth_still_fails_closed_and_names_the_shortage(self) -> None:
        # prior 16 rows (5 empty); this iteration 2 non-empty + 10 empty rows and 3 anchor candidates.
        # The guard trims the empties down to 3 current rows; with the anchors bound to their share
        # nothing fills a global batch above the prior corpus -> growth would be zero.
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            cfg = self._anchor_cfg(root)
            previous = _write(root / "train0.jsonl", [_det_row(f"o-e-{i}") for i in range(5)]
                              + [_det_row(f"o-p-{i:02d}", BOX) for i in range(11)])
            mined = _write(root / "mined.jsonl", [_det_row(f"n-p-{i}", BOX) for i in range(2)] + [_det_row(f"n-e-{i:02d}") for i in range(10)])
            pattern = r"would add zero rows.*current_rows_after_guard=\d+.*anchor_slots=\d+.*row_multiple=8"
            with self.assertRaisesRegex(ValueError, pattern):
                atj.assemble(previous, mined, previous_sha256=atj.sha256_file(previous), validation_paths=[], media_root=root,
                             anchor_config=cfg, row_multiple=8, empty_answer_guard=_guard(0.30))
            out = root / "train1.jsonl"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = atj.main([
                    "--mined-jsonl", str(mined), "--previous-jsonl", str(previous), "--previous-sha256", atj.sha256_file(previous),
                    "--output", str(out), "--media-root", str(root), "--row-multiple", "8", "--max-empty-answer-share", "0.30",
                    "--anchor-share", "0.10", "--anchor-source", str(root / "anchor_candidates.jsonl"),
                    "--anchor-task-shares", str(root / "kpi.jsonl"),
                ])
            self.assertEqual(rc, 2)
            self.assertRegex(stderr.getvalue(), pattern)
            self.assertFalse(out.exists())


def _cal(record: dict) -> dict:
    """A detection calibration row as the materializer emits it (both markers)."""
    return {**record, CAL: True, KIND: dda.DETECTION_CALIBRATION_KIND}


# ---------------------------------------------------------------------------------------------
# r6 iteration-2 shape (synthetic, Feature B4): prior 2,304 rows (240 retained anchors, 450
# empty), this iteration 1,387 current rows = 221 mined non-empty rows (Component Detection 6,
# Defect Detection 120, Ref_based Defect Detection 95) + 1,166 detection calibration rows (500
# empty: 350 boards / 150 no-change pairs; 666 non-empty: 450 few-box / 216 changed), 200 anchor
# candidates. The uncapped anchor target adds 143 anchors, 3,834 candidates align down to 3,072,
# the 0.10 share reserves 67 new anchors and the growth slot holds 701 current rows, fewer than
# the 1,166 calibration rows. (The run had 230 retained anchors -> 77 slots / 691; 240 keeps the
# same inputs on the round-up path when the guard is off, which needs <= 148 anchor candidates in
# the corpus.)
# ---------------------------------------------------------------------------------------------
def _r6_fixture() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    prior = [{**_mcq_row(f"p-anc-{i:04d}", CC, answer="A", dataset="pool"), anchor_rows.ANCHOR_MARK: True} for i in range(240)]
    prior += [_det_row(f"p-dd-e-{i:04d}", dataset="prev") for i in range(300)]
    prior += [_det_row(f"p-dd-p-{i:04d}", BOX, dataset="prev") for i in range(900)]
    prior += [_ref_det_row(f"p-ref-e-{i:04d}", dataset="prev") for i in range(150)]
    prior += [_ref_det_row(f"p-ref-p-{i:04d}", BOX, dataset="prev") for i in range(714)]
    assert len(prior) == 2304 == 3 * 768
    current = [_det_row(f"m-cd-{i:04d}", BOX, task="Component Detection", dataset="mine") for i in range(6)]
    current += [_det_row(f"m-dd-{i:04d}", BOX, dataset="mine") for i in range(120)]
    current += [_ref_det_row(f"m-ref-{i:04d}", BOX, dataset="mine") for i in range(95)]
    current += [_cal(_det_row(f"c-dd-e-{i:04d}", dataset="pool")) for i in range(350)]
    current += [_cal(_det_row(f"c-dd-f-{i:04d}", BOX, dataset="pool")) for i in range(450)]
    current += [_cal(_ref_det_row(f"c-ref-e-{i:04d}", dataset="pool")) for i in range(150)]
    current += [_cal(_ref_det_row(f"c-ref-c-{i:04d}", BOX, dataset="pool")) for i in range(216)]
    assert len(current) == 1387 and sum(r.get(CAL) is True for r in current) == 1166
    anchors = [_mcq_row(f"a-cc-{i:03d}", CC, answer="A", dataset=f"pool{i % 2}") for i in range(200)]
    kpi = [_mcq_row(f"k-{i}", CC, answer="A") for i in range(50)]
    return prior, current, anchors, kpi


class CalibrationYieldTests(unittest.TestCase):
    """Feature B4 (run v12_p4b_emptyguard_r6 iteration 2): under the guard the calibration quota is
    an upper bound and the growth slot is the binding constraint; calibration rows yield instead of
    the iteration failing, empties first, proportionally over the two detection tasks."""

    GUARD = {DD: 0.45, REF: 0.50}

    def _write_r6(self, root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, dict]:
        prior, current, anchors, kpi = _r6_fixture()
        previous = _write(root / "train1.jsonl", prior)
        mined = _write(root / "mined2.jsonl", current)
        cfg = anchor_rows.validate_anchor_config(
            0.10, _write(root / "anchor_candidates.jsonl", anchors), _write(root / "kpi.jsonl", kpi), None
        )
        return previous, mined, cfg

    def test_r6_iteration_2_shape_yields_calibration_to_the_growth_slot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            previous, mined, cfg = self._write_r6(root)
            out, summary = atj.assemble(
                previous, mined, previous_sha256=atj.sha256_file(previous), validation_paths=[], media_root=root,
                anchor_config=cfg, row_multiple=768, empty_answer_guard=_guard(0.30, self.GUARD), calibration_guard_aware=True,
            )
            self.assertEqual(len(out), 3072)
            self.assertEqual(summary["growth_rows"], 768)
            ids = [r["id"] for r in out]
            # every mined row is kept, the calibration rows fill the 480 slots that are left
            self.assertEqual(sum(i.startswith("m-") for i in ids), 221)
            self.assertEqual(summary["materialized_calibration_records"], 480)
            # empties first (all 500 go), then the non-empty rows proportionally: 666 -> 480 splits
            # 324 / 156 by largest remainder; a prefix of each task's rows in materializer order
            self.assertEqual([i for i in ids if i.startswith("c-dd-e-") or i.startswith("c-ref-e-")], [])
            self.assertEqual([i for i in ids if i.startswith("c-dd-f-")], [f"c-dd-f-{i:04d}" for i in range(324)])
            self.assertEqual([i for i in ids if i.startswith("c-ref-c-")], [f"c-ref-c-{i:04d}" for i in range(156)])
            self.assertEqual(summary["calibration_rows_dropped_for_cap"], {
                "by_task": {DD: 476, REF: 210}, "by_kind": {"empty": 500, "non_empty": 186}, "total": 686,
            })
            self.assertEqual(summary["calibration_rows_kept"], {DD: 324, REF: 156})
            self.assertTrue(summary["calibration_yielded"])
            self.assertEqual(summary["operator_attention"], ["calibration_yielded"])
            # anchors exactly at the share: round(0.10 * 3,072) = 307 total, 240 retained -> 67 new
            self.assertEqual(summary["materialized_anchor_records"], 67)
            reservation = summary["anchor"]["cap_reservation"]
            self.assertEqual((reservation["new_anchor_slots"], reservation["alignment_fill_anchors"]), (67, 0))
            self.assertEqual(reservation["alignment_policy"], "round_down_calibration_yields_to_growth_slot")
            self.assertEqual(reservation["calibration_rows_protected"], 1166)
            self.assertEqual(reservation["aligned_rows_shrunk_for_share"], 0)
            self.assertAlmostEqual(summary["anchor"]["cumulative_share_rows"], 307 / 3072)
            self.assertEqual(ids[221:701], [i for i in ids if i.startswith("c-")])  # mined -> calibration -> anchors -> prior
            # the guard measures the final corpus (after the yield) and converges in one pass
            report = summary["empty_answer_guard"]
            self.assertEqual((report["status"], report["passes"], report["rows_trimmed_total"]), ("within_caps", 1, 0))
            self.assertEqual(report["after"]["rows"], 3072)
            self.assertEqual(report["after"]["empty_rows"], sum(atj.is_empty_ground_truth(r) for r in out))
            self.assertAlmostEqual(report["after"]["overall_share"], 450 / 3072)
            self.assertAlmostEqual(report["after"]["per_task"][DD], 300 / 1644)
            self.assertAlmostEqual(report["after"]["per_task"][REF], 150 / 1115)
            self.assertEqual(summary["answer_profile"]["empty_rows"], 450)
            self.assertEqual(summary["answer_profile_new_rows"]["empty_rows"], 0)
            self.assertEqual(len(set(ids)), 3072)

    def test_same_shape_with_the_guard_off_still_takes_the_round_up_path(self) -> None:
        # HEAD behaviour: the calibration rows are protected, the corpus rounds UP to 3,840 and the
        # 82-row gap is filled with anchors (76 spare from the uncapped target + 6 freshly selected)
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            previous, mined, cfg = self._write_r6(root)
            out, summary = atj.assemble(
                previous, mined, previous_sha256=atj.sha256_file(previous), validation_paths=[], media_root=root,
                anchor_config=cfg, row_multiple=768,
            )
            self.assertEqual(len(out), 3840)
            self.assertEqual(summary["growth_rows"], 1536)
            self.assertEqual(summary["materialized_calibration_records"], 1166)
            self.assertEqual(sum(r["id"].startswith("m-") for r in out), 221)
            self.assertEqual(summary["materialized_anchor_records"], 149)
            reservation = summary["anchor"]["cap_reservation"]
            self.assertEqual(reservation["alignment_policy"], "round_up_fill_with_anchors")
            self.assertEqual((reservation["alignment_fill_anchors"], reservation["anchor_overfill"]), (6, "allow"))
            self.assertEqual(sum("alignment_fill" in item.get("purpose_tags", []) for item in summary["provenance"]), 6)
            self.assertEqual(summary["calibration_rows_dropped_for_cap"],
                             {"by_task": {}, "by_kind": {"empty": 0, "non_empty": 0}, "total": 0})
            self.assertEqual(summary["calibration_rows_kept"], {DD: 800, REF: 366})
            self.assertFalse(summary["calibration_yielded"])
            self.assertEqual(summary["operator_attention"], [])
            self.assertEqual(summary["empty_answer_guard"]["anchor_overfill"], "allow")

    def test_zero_growth_still_fails_closed_after_the_yield(self) -> None:
        # prior 16 rows (4 empty); this iteration 1 mined row + 7 calibration negatives, 3 anchor
        # candidates at the 0.10 share. Pass 1 yields 2 negatives (7 > 6 slots) but 9 / 24 empties
        # exceed the cap, the guard trims 3 negatives, and the 5 rows left plus 2 anchors no longer
        # fill a global batch above the prior corpus: the existing zero-growth message, not a corpus.
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            anchors = [_mcq_row(f"a-cc-{i:03d}", CC, answer="A", dataset=f"pool{i % 2}") for i in range(300)]
            cfg = anchor_rows.validate_anchor_config(
                0.10, _write(root / "anchor_candidates.jsonl", anchors),
                _write(root / "kpi.jsonl", [_mcq_row(f"k-{i}", CC, answer="A") for i in range(50)]), None,
            )
            previous = _write(root / "train0.jsonl", [_det_row(f"o-e-{i}") for i in range(4)]
                              + [_det_row(f"o-p-{i:02d}", BOX) for i in range(12)])
            mined = _write(root / "mined.jsonl", [_det_row("n-p-0", BOX)] + [_cal(_det_row(f"n-cal-e-{i:02d}", dataset="pool")) for i in range(7)])
            pattern = (r"would add zero rows.*current_rows_after_guard=5.*anchor_slots=2.*row_multiple=8.*"
                       r"previous_rows=16.*aligned_rows=16")
            with self.assertRaisesRegex(ValueError, pattern):
                atj.assemble(previous, mined, previous_sha256=atj.sha256_file(previous), validation_paths=[], media_root=root,
                             anchor_config=cfg, row_multiple=8, empty_answer_guard=_guard(0.30))
            # without the guard's trimming the same shape yields and grows by one batch
            _, summary = atj.assemble(previous, mined, previous_sha256=atj.sha256_file(previous), validation_paths=[], media_root=root,
                                      anchor_config=cfg, row_multiple=8, empty_answer_guard=_guard(0.30, mode="report"))
            self.assertEqual((summary["output_records"], summary["growth_rows"]), (24, 8))
            self.assertEqual(summary["calibration_rows_dropped_for_cap"]["total"], 2)
            self.assertTrue(summary["calibration_yielded"])

    def test_mined_rows_alone_beyond_the_slot_keep_the_mined_first_rule_and_drop_all_calibration(self) -> None:
        # prior 16; 30 mined non-empty rows (15 / 15 over the two detection tasks) + 40 calibration
        # rows (10 empty + 10 non-empty per task) under a 48-row cap: 5 anchor slots, 27 current
        # slots, the mined rows alone exceed them -> task-balanced mined trim, zero calibration rows.
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            anchors = [_mcq_row(f"a-cc-{i:03d}", CC, answer="A", dataset=f"pool{i % 2}") for i in range(300)]
            anchor_source = _write(root / "anchor_candidates.jsonl", anchors)
            kpi = _write(root / "kpi.jsonl", [_mcq_row(f"k-{i}", CC, answer="A") for i in range(50)])
            cfg = anchor_rows.validate_anchor_config(0.10, anchor_source, kpi, None)
            previous = _write(root / "train0.jsonl", [_det_row(f"o-p-{i:02d}", BOX) for i in range(16)])
            current = [_det_row(f"m-dd-{i:02d}", BOX) for i in range(15)] + [_ref_det_row(f"m-ref-{i:02d}", BOX) for i in range(15)]
            current += [_cal(_det_row(f"c-dd-e-{i:02d}", dataset="pool")) for i in range(10)]
            current += [_cal(_det_row(f"c-dd-f-{i:02d}", BOX, dataset="pool")) for i in range(10)]
            current += [_cal(_ref_det_row(f"c-ref-e-{i:02d}", dataset="pool")) for i in range(10)]
            current += [_cal(_ref_det_row(f"c-ref-c-{i:02d}", BOX, dataset="pool")) for i in range(10)]
            mined = _write(root / "mined.jsonl", current)
            common = dict(previous_sha256=atj.sha256_file(previous), validation_paths=[], media_root=root,
                          anchor_config=cfg, max_rows=48, row_multiple=8)
            out, summary = atj.assemble(previous, mined, empty_answer_guard=_guard(0.30), **common)
            self.assertEqual((len(out), summary["growth_rows"]), (48, 32))
            ids = [r["id"] for r in out]
            balanced = [f"m-{task}-{i:02d}" for i in range(13) for task in ("dd", "ref")] + ["m-dd-13"]
            self.assertEqual(ids[:27], balanced)  # the existing task-balanced mined trim, unchanged
            self.assertEqual(sum(i.startswith("m-dd-") for i in ids), 14)
            self.assertEqual(sum(i.startswith("m-ref-") for i in ids), 13)
            self.assertEqual(summary["materialized_calibration_records"], 0)
            self.assertEqual(summary["materialized_anchor_records"], 5)
            self.assertEqual(summary["calibration_rows_dropped_for_cap"], {
                "by_task": {DD: 20, REF: 20}, "by_kind": {"empty": 20, "non_empty": 20}, "total": 40,
            })
            self.assertEqual(summary["calibration_rows_kept"], {})
            self.assertTrue(summary["calibration_yielded"])
            self.assertEqual(summary["operator_attention"], ["calibration_yielded"])
            self.assertEqual(summary["empty_answer_guard"]["status"], "within_caps")
            # the same shape without the guard is the HEAD round-up path, which the 48-row cap forbids
            with self.assertRaisesRegex(ValueError, "cannot retain the calibration rows under the configured cap"):
                atj.assemble(previous, mined, **common)
            # CLI: the informational attention line goes to stderr, the corpus is written
            out_path = root / "train1.jsonl"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = atj.main([
                    "--mined-jsonl", str(mined), "--previous-jsonl", str(previous), "--previous-sha256", atj.sha256_file(previous),
                    "--output", str(out_path), "--media-root", str(root), "--max-rows", "48", "--row-multiple", "8",
                    "--max-empty-answer-share", "0.30", "--anchor-share", "0.10", "--anchor-source", str(anchor_source),
                    "--anchor-task-shares", str(kpi),
                ])
            self.assertEqual(rc, 0, stderr.getvalue())
            self.assertIn("operator_attention: calibration_yielded", stderr.getvalue())
            self.assertIn("calibration_rows_dropped_for_cap=40", stderr.getvalue())
            self.assertEqual(len(out_path.read_text().splitlines()), 48)
            written = json.loads((root / "assemble_summary.json").read_text())
            self.assertEqual(written["calibration_rows_dropped_for_cap"]["total"], 40)
            self.assertTrue(written["calibration_yielded"])


class WiringTests(unittest.TestCase):
    def test_init_records_the_flags_and_their_guard_dependent_defaults(self) -> None:
        from test_cosmos3_init_state_contract import Cosmos3InitStateContractTests as Base
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = Base._workspace(root)
            guard_args = ("--max-empty-answer-share", "0.30", "--max-empty-answer-share-task", "Defect Detection=0.45")
            rc = init_deft_state.main(Base._argv(root / "a", workspace, *guard_args))
            self.assertEqual(rc, 0)
            guard = json.loads((root / "a/results/deft_state.json").read_text())["config"]["mining"]["empty_answer_guard"]
            self.assertTrue(guard["calibration_guard_aware"])  # default on under the guard
            self.assertEqual(guard["anchor_overfill"], "forbid")  # default forbid under the guard
            self.assertIn("--calibration-guard-aware", guard["owner"])
            self.assertIn("--anchor-overfill", guard["owner"])
            self.assertIn("headroom", guard["calibration_guard_aware_rule"])
            self.assertIn("share", guard["anchor_overfill_rule"])
            rc = init_deft_state.main(Base._argv(root / "b", workspace, *guard_args, "--calibration-guard-aware", "off",
                                                 "--anchor-overfill", "allow"))
            self.assertEqual(rc, 0)
            guard = json.loads((root / "b/results/deft_state.json").read_text())["config"]["mining"]["empty_answer_guard"]
            self.assertFalse(guard["calibration_guard_aware"])
            self.assertEqual(guard["anchor_overfill"], "allow")
            rc = init_deft_state.main(Base._argv(root / "c", workspace))
            self.assertEqual(rc, 0)
            guard = json.loads((root / "c/results/deft_state.json").read_text())["config"]["mining"]["empty_answer_guard"]
            self.assertFalse(guard["enabled"])
            self.assertFalse(guard["calibration_guard_aware"])
            self.assertEqual(guard["anchor_overfill"], "allow")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = init_deft_state.main(Base._argv(root / "d", workspace, "--calibration-guard-aware", "on"))
            self.assertNotEqual(rc, 0)
            self.assertIn("calibration-guard-aware", stderr.getvalue())

    def test_runner_mirrors_the_caps_to_the_materializer_only_when_guard_aware_is_on(self) -> None:
        command = [
            "python", "select.py", "--top-k", "50",
            "--max-empty-answer-share", "0.30", "--max-empty-answer-share-task", "Defect Detection=0.45",
            "--max-classification-empty-share", "0.10", "--empty-answer-guard-mode", "enforce",
            "--calibration-guard-aware", "on", "--anchor-overfill", "forbid",
        ]
        selector, assembler = runner._partition_materialization_arguments(command)
        self.assertEqual(selector, [
            "python", "select.py", "--top-k", "50", "--calibration-guard-aware", "on",
            "--max-empty-answer-share", "0.30", "--max-empty-answer-share-task", "Defect Detection=0.45",
            "--no-repetition-blend",
        ])
        self.assertEqual(assembler, [
            "--max-empty-answer-share", "0.30", "--max-empty-answer-share-task", "Defect Detection=0.45",
            "--max-classification-empty-share", "0.10", "--empty-answer-guard-mode", "enforce",
            "--calibration-guard-aware", "on", "--anchor-overfill", "forbid",
        ])
        selector, assembler = runner._partition_materialization_arguments(
            ["python", "select.py", "--max-empty-answer-share", "0.30", "--calibration-guard-aware=off"]
        )
        self.assertEqual(selector, ["python", "select.py", "--calibration-guard-aware=off", "--no-repetition-blend"])
        self.assertEqual(assembler, ["--max-empty-answer-share", "0.30", "--calibration-guard-aware=off"])
        with self.assertRaisesRegex(ValueError, "on or off"):
            runner._partition_materialization_arguments(["x", "--calibration-guard-aware", "yes"])
        with self.assertRaisesRegex(ValueError, "requires a value"):
            runner._partition_materialization_arguments(["x", "--calibration-guard-aware"])
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            plan = runner.build_plan(
                selector_command=[sys.executable, "selector.py", "--max-empty-answer-share", "0.3", "--calibration-guard-aware", "on"],
                previous_jsonl=None, previous_sha256=None,
                mined_jsonl=root / "mined.jsonl", current_quota_manifest=root / "current-quota.json",
                train_jsonl=root / "train.jsonl", assemble_summary=root / "assembly.json",
                final_quota_manifest=root / "quota.json", media_root=root,
                max_rows=768, row_multiple=768, epochs=5, global_batch=768,
            )
            self.assertTrue(plan["invariants"]["empty_answer_guard_owned_by_assembler"])
            self.assertTrue(plan["invariants"]["calibration_guard_caps_mirrored_to_materializer"])
            self.assertIn("--max-empty-answer-share", plan["selector"]["command"])
            self.assertIn("--calibration-guard-aware", plan["selector"]["command"])
            self.assertIn("--calibration-guard-aware", plan["assembler"]["command"])
            plan = runner.build_plan(
                selector_command=[sys.executable, "selector.py", "--max-empty-answer-share", "0.3"],
                previous_jsonl=None, previous_sha256=None,
                mined_jsonl=root / "mined.jsonl", current_quota_manifest=root / "current-quota.json",
                train_jsonl=root / "train.jsonl", assemble_summary=root / "assembly.json",
                final_quota_manifest=root / "quota.json", media_root=root,
                max_rows=768, row_multiple=768, epochs=5, global_batch=768,
            )
            self.assertTrue(plan["invariants"]["empty_answer_guard_owned_by_assembler"])
            self.assertFalse(plan["invariants"]["calibration_guard_caps_mirrored_to_materializer"])
            self.assertNotIn("--max-empty-answer-share", plan["selector"]["command"])

    def test_assembler_fails_closed_when_the_manifest_disagrees_on_guard_aware_calibration(self) -> None:
        atj.check_calibration_guard_aware_manifest({"calibration_guard_aware": True}, expected=True, source="m")
        atj.check_calibration_guard_aware_manifest({"calibration_guard_aware": False}, expected=False, source="m")
        atj.check_calibration_guard_aware_manifest({}, expected=False, source="m")
        with self.assertRaisesRegex(ValueError, "--calibration-guard-aware"):
            atj.check_calibration_guard_aware_manifest({"calibration_guard_aware": False}, expected=True, source="m")
        with self.assertRaisesRegex(ValueError, "--calibration-guard-aware"):
            atj.check_calibration_guard_aware_manifest({}, expected=True, source="m")
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined = _write(root / "mined.jsonl", [_det_row(f"pos-{i}", BOX) for i in range(8)])
            manifest = root / "current-quota.json"
            manifest.write_text(json.dumps({"schema_version": "defect_detection_quota_manifest_v1", "calibration_guard_aware": False}))
            common = ["--mined-jsonl", str(mined), "--output", str(root / "train.jsonl"), "--media-root", str(root),
                      "--current-quota-manifest", str(manifest), "--quota-manifest", str(root / "quota.json"),
                      "--epochs", "1", "--global-batch", "8", "--max-empty-answer-share", "0.30"]
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = atj.main(common)  # guard on -> guard-aware calibration expected by default
            self.assertEqual(rc, 2)
            self.assertIn("--calibration-guard-aware", stderr.getvalue())
            self.assertFalse((root / "train.jsonl").exists())
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = atj.main([*common, "--calibration-guard-aware", "off"])
            self.assertNotIn("--calibration-guard-aware", stderr.getvalue())  # cross-check passes; binding fails later
            self.assertEqual(rc, 2)

    def test_bound_manifest_carries_the_guard_aware_fields_under_current_selection(self) -> None:
        keys = ("calibration_guard_aware", "calibration_empty_headroom", "calibration_empty_selected", "calibration_fewbox_substituted",
                "calibration_headroom_overflow_rows")
        for key in keys:
            self.assertIn(key, dda.CURRENT_SELECTION_COPIED_KEYS)


if __name__ == "__main__":
    if "--write-golden" in sys.argv:
        print(json.dumps(write_golden(), indent=2))
    else:
        unittest.main()
