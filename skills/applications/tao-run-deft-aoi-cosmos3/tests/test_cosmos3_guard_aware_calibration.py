# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Feature B3: guard-aware detection calibration selection and no anchor over-fill under the
empty-answer guard (run v12_p4b_emptyguard_r4 iteration 4, snapshot 6fb5dcbc).

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
from test_cosmos3_classification_calibration import _det_row, _row as _mcq_row, _write  # noqa: E402
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


def _materialize(prior: list[dict], source: list[dict], candidates: list[dict], **extra) -> tuple[list[dict], dict]:
    return dda.materialize(
        candidate_rows=candidates, source_records=source, previous_records=prior or None, validation_records=[],
        media_root=MEDIA, max_rows=1536, row_multiple=768, defect_detection_fraction=0.5, proxy_empty_rate=0.25,
        epochs=1, global_batch=768, near_duplicate_hamming_distance=None, reference_proxy_empty_rate=0.542,
        single_image_calibration_max_empty=512, single_image_calibration_max_few=512, reference_calibration_total=500,
        zero_new_candidate_policy="skip_exhausted", **extra,
    )


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
    source: list[dict] = []
    candidates: list[dict] = []

    def add(record: dict, **kwargs) -> None:
        source.append(record)
        candidates.append(_cand(record, **kwargs))

    for i in range(512):
        add(_det_row(f"c-dd-e-{i:04d}", dataset="pool"), evidence=[dda.CALIBRATION_EMPTY_EVIDENCE], route_tier="calibration")
    for i in range(1024):
        boxes = [{"bbox_2d": [i % 50, 0, i % 50 + 5 + i % 3, 7], "label": ("open", "short")[i % 2]}]
        add(_det_row(f"c-dd-f-{i:04d}", boxes, dataset=f"pool{i % 3}"), evidence=[dda.CALIBRATION_FEW_EVIDENCE],
            route_tier="calibration", contrast=(i % 97) / 100.0)
    for i in range(400):
        add(_ref_det_row(f"c-ref-e-{i:04d}", dataset="pool"),
            evidence=[dda.CALIBRATION_EMPTY_EVIDENCE, dda.REFERENCE_NO_CHANGE_EVIDENCE], route_tier="calibration")
    for i in range(500):
        add(_ref_det_row(f"c-ref-c-{i:04d}", BOX, dataset="pool"), evidence=[dda.CALIBRATION_FEW_EVIDENCE], route_tier="calibration")
    for i in range(5):
        add(_ref_det_row(f"m-ref-e-{i:04d}", dataset="mine"))
    for i in range(4):
        add(_ref_det_row(f"m-ref-p-{i:04d}", BOX, dataset="mine"))
    for i in range(3):
        add(_mcq_row(f"m-rdc-{i:04d}", REF_DC, answer="A", dataset="mine", two_images=True))
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
            # HEAD behaviour (guard-aware off): 512 + 271 empty calibration rows on a corpus at
            # 0.294 -> the guard must trim 421 rows, the aligned size drops by one global batch
            # and the protected calibration rows no longer fit: the iteration fails closed.
            rows0, manifest0 = _materialize(prior, source, candidates)
            self.assertTrue(manifest0["verified"])
            self.assertFalse(manifest0["calibration_guard_aware"])
            self.assertEqual(manifest0["calibration_empty_selected"], {DD: 512, REF: 271, "total": 783})
            self.assertEqual(manifest0["calibration_fewbox_substituted"], {DD: 0, REF: 0, "total": 0})
            self.assertEqual(set(manifest0["calibration_empty_headroom"].values()), {None})
            mined0 = _write(root / "mined0.jsonl", rows0)
            with self.assertRaises(ValueError):
                atj.assemble(previous, mined0, previous_sha256=previous_sha, validation_paths=[], media_root=root,
                             row_multiple=768, empty_answer_guard=guard)
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
        keys = ("calibration_guard_aware", "calibration_empty_headroom", "calibration_empty_selected", "calibration_fewbox_substituted")
        for key in keys:
            self.assertIn(key, dda.CURRENT_SELECTION_COPIED_KEYS)


if __name__ == "__main__":
    if "--write-golden" in sys.argv:
        print(json.dumps(write_golden(), indent=2))
    else:
        unittest.main()
