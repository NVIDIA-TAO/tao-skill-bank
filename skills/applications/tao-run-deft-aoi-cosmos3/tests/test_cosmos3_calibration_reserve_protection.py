# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Feature P5-S.2: calibration reserve protection and guard-induced calibration shortfall
acceptance (run v12_p5s_pool10_r9 iteration 1, snapshot 9acf22f9, 2026-09-17).

r9 shape: the mined Ref_based Defect Detection selection is large (8,936 rows, 55.6 percent
empty, above the 0.50 task cap), so the guard-aware split lowers the no-change target to 0 and
asks for 500 changed pairs; the re-selection found none (``selected_total 0``,
``fewbox_shortfall 500``), no batch-aligned target was feasible and the materializer dumped
37,164 unverified rows (exit 2). The re-selection of the reserve was starved of its novel-image
budget by the pre-trim over-selection of the mined rows; with the reserve protected the 500
changed pairs are reserved. When the protected reserve is genuinely short at a zero empty
headroom the slot yields the missing rows to mined rows instead of failing closed
(``substituted_with_guard_shortfall``). Guard-off is byte-identical to HEAD (pinned below)."""

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

import assemble_training_json as atj  # noqa: E402
import defect_detection_ablation as dda  # noqa: E402
from test_cosmos3_classification_calibration import _det_row, _row as _mcq_row, _write  # noqa: E402
from test_cosmos3_empty_answer_guard import _ref_det_row  # noqa: E402
from test_cosmos3_guard_aware_calibration import BOX, GOLDEN, _cand, _golden_materializer_rows, _guard, _rows_sha256  # noqa: E402
from test_cosmos3_zero_new_candidate_policy import _proxy_rows  # noqa: E402

DD = dda.DEFECT_DETECTION_TASK
REF = dda.REFERENCE_DEFECT_DETECTION_TASK
CC, CD, DC, REF_DC = "Component Classification", "Component Detection", "Defect Classification", "Ref_based Defect Classification"
CAL = dda.CALIBRATION_MARK
MEDIA = pathlib.Path("/data")
FILL_ORDER = (DC, CD, CC)
# KPI reference empty rate 0.556 -> floor(50 * 0.556 + 0.5) = 28 no-change / 22 changed of a 50-pair slot
REFERENCE_RATE = 0.556
GUARD = {DD: 0.45, REF: 0.50}
# HEAD (9acf22f9) guard-off selections on the fixtures below (guard-off must stay byte-identical)
HEAD_GUARD_OFF = {
    "r9": ("043b0addcbd521b3682dee408c5603391ca210428f81726017092df110c34c26", 1024, True),
    "short_reserve": ("7f167fc208f7a9eb5203b644142f54785e61faa528ca854629121edc35562e3c", 1288, False),
    "single_image": ("13d87713dbd07648bcb4c85fd0bd26482eba904a130e6692fea198caf4518741", 1024, True),
}


def _by_task(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["task_type"]] = counts.get(row["task_type"], 0) + 1
    return counts


# ---------------------------------------------------------------------------------------------
# r9 iteration-1 shape, scaled 1:32 (target 1,024 = 16 batches of 64; slot 32 + 32 single-image
# rows and 50 reference pairs): a calibration feed of 28 no-change + ``changed`` pairs, 900 mined
# reference pairs at 55.6 percent empty (the guard-aware split lowers the no-change target to 0),
# 200 strict Defect Detection positives, 60 rows of every other maintenance task. ``overlap``
# adds a strict (mined) candidate for every feed pair BEFORE the calibration candidate, so the
# reserve records are also mined candidates. ``prior`` prepends a cumulative corpus.
# ---------------------------------------------------------------------------------------------
def _fixture(*, changed: int = 50, fewbox: int = 64, overlap: bool = False, mined_ref: int = 900,
             mined_ref_empty_share: float = 0.556, other: int = 60) -> tuple[list[dict], list[dict]]:
    source: list[dict] = []
    candidates: list[dict] = []

    def add(record: dict, **kwargs) -> None:
        source.append(record)
        candidates.append(_cand(record, **kwargs))

    for i in range(32):
        add(_det_row(f"c-dd-e-{i:04d}", dataset="pool"), evidence=[dda.CALIBRATION_EMPTY_EVIDENCE], route_tier="calibration")
    for i in range(fewbox):
        boxes = [{"bbox_2d": [i % 50, 0, i % 50 + 5 + i % 3, 7], "label": ("open", "short")[i % 2]}]
        add(_det_row(f"c-dd-f-{i:04d}", boxes, dataset=f"pool{i % 3}"), evidence=[dda.CALIBRATION_FEW_EVIDENCE],
            route_tier="calibration", contrast=(i % 97) / 100.0)
    feed = [(_ref_det_row(f"c-ref-e-{i:04d}", dataset="pool"), [dda.CALIBRATION_EMPTY_EVIDENCE, dda.REFERENCE_NO_CHANGE_EVIDENCE])
            for i in range(28)]
    feed += [(_ref_det_row(f"c-ref-c-{i:04d}", BOX, dataset="pool"), [dda.CALIBRATION_FEW_EVIDENCE]) for i in range(changed)]
    for record, evidence in feed:
        if overlap:
            # the router also routed the pair as a mined (strict) candidate, listed first
            source.append(record)
            candidates.append(_cand(record))
            candidates.append(_cand(record, evidence=evidence, route_tier="calibration"))
        else:
            add(record, evidence=evidence, route_tier="calibration")
    empties = int(mined_ref * mined_ref_empty_share)
    for i in range(empties):
        add(_ref_det_row(f"m-ref-e-{i:04d}", dataset="mine"))
    for i in range(mined_ref - empties):
        add(_ref_det_row(f"m-ref-p-{i:04d}", BOX, dataset="mine"))
    for i in range(200):
        add(_det_row(f"m-dd-p-{i:04d}", [{"bbox_2d": [i % 40, i % 30, i % 40 + 9, i % 30 + 9], "label": "short"}], dataset="mine"),
            evidence=["hard_positive_proxy_false_negative"], contrast=(i % 89) / 100.0)
    for i in range(other):
        add(_mcq_row(f"m-cc-{i:04d}", CC, answer="A", dataset="mine"))
        add(_det_row(f"m-cd-{i:04d}", BOX, task=CD, dataset="mine"))
        add(_mcq_row(f"m-dc-{i:04d}", DC, answer="A", dataset="mine"))
        add(_mcq_row(f"m-rdc-{i:04d}", REF_DC, answer="A", dataset="mine", two_images=True))
    return source, candidates


def _single_image_prior() -> list[dict]:
    """A cumulative corpus at 60 percent empty Defect Detection (over the 0.45 cap: DD headroom 0)
    and 20 percent empty reference pairs (well under the 0.50 cap: no reference substitution)."""
    prior = [_det_row(f"p-dd-e-{i:04d}", dataset="prev") for i in range(600)]
    prior += [_det_row(f"p-dd-p-{i:04d}", BOX, dataset="prev") for i in range(400)]
    prior += [_ref_det_row(f"p-ref-e-{i:04d}", dataset="prev") for i in range(200)]
    prior += [_ref_det_row(f"p-ref-p-{i:04d}", BOX, dataset="prev") for i in range(800)]
    return prior


def _materialize(source: list[dict], candidates: list[dict], *, guard_on: bool, prior: list[dict] | None = None,
                 cross_task_visual_dedup: str = "off", **extra) -> tuple[list[dict], dict]:
    return dda.materialize(
        candidate_rows=candidates, source_records=source, previous_records=prior, validation_records=[],
        media_root=MEDIA, max_rows=1024, minimum_rows=64, row_multiple=64, defect_detection_fraction=0.15,
        proxy_empty_rate=0.25, epochs=1, global_batch=64, near_duplicate_hamming_distance=None,
        reference_proxy_empty_rate=REFERENCE_RATE, single_image_calibration_max_empty=32,
        single_image_calibration_max_few=32, reference_calibration_total=50, zero_new_candidate_policy="skip_exhausted",
        empty_answer_guard=_guard(None, GUARD), calibration_guard_aware=guard_on,
        cross_task_visual_dedup=cross_task_visual_dedup, mined_task_fill_order=FILL_ORDER, **extra,
    )


class ReserveProtectionTests(unittest.TestCase):
    """(a) the r9 shape: the reserve is protected and the 50 changed pairs are reserved."""

    def _assert_r9_reserved(self, rows: list[dict], manifest: dict, *, protected: int) -> None:
        self.assertTrue(manifest["verified"], manifest["verification"])
        self.assertEqual(len(rows), 1024)
        self.assertFalse(manifest["shortfall"]["accepted"])
        # the mined reference rows sit above the 0.50 cap: no-change headroom 0, every no-change
        # pair of the KPI-rate reservation is substituted by a changed pair of the protected reserve
        self.assertEqual(manifest["calibration_empty_headroom"][f"task:{REF}"], 0)
        detail = manifest["guard_aware_calibration"]
        self.assertEqual(detail["kpi_empty_targets"], {DD: 32, REF: 28})
        self.assertEqual(detail["headroom_empty_targets"][REF], 0)
        reference = manifest["reference_calibration"]
        self.assertEqual((reference["selected_no_change"], reference["selected_changed"], reference["selected_total"]), (0, 50, 50))
        self.assertEqual((reference["target_no_change"], reference["kpi_target_no_change"], reference["no_change_substituted_by_changed"]), (0, 28, 28))
        self.assertEqual((reference["no_change_beyond_headroom"], reference["shortfall_accepted_under_guard"], reference["effective_total"]), (0, 0, 50))
        self.assertEqual(detail["fewbox_shortfall"], {DD: 0, REF: 0})
        self.assertEqual(detail["shortfall_accepted_under_guard"], {DD: 0, REF: 0})
        self.assertEqual(detail["status"], "substituted")
        self.assertEqual(manifest["calibration_reserve_rows_protected"], {DD: 0, REF: protected})
        self.assertEqual(manifest["calibration_shortfall_accepted_under_guard"], {DD: 0, REF: 0, "total": 0})
        self.assertEqual(manifest["reference_calibration_shortfall_accepted_under_guard"], 0)
        self.assertEqual(manifest["single_image_calibration_shortfall_accepted_under_guard"], 0)
        verification = manifest["verification"]
        self.assertTrue(verification["reference_calibration_contract_reached"])
        self.assertTrue(verification["reference_calibration_contract_reached_or_yielded"])
        self.assertTrue(verification["reference_empty_rate_matched"])
        self.assertTrue(verification["novel_image_limit_respected"])
        # every reserved pair is emitted once, as a calibration row; no mined row shares its record
        changed_ids = {f"c-ref-c-{i:04d}" for i in range(50)}
        emitted = [row for row in rows if row["id"] in changed_ids]
        self.assertEqual(len(emitted), 50)
        self.assertTrue(all(row.get(CAL) is True for row in emitted))
        self.assertEqual(sum(row["id"].startswith("c-ref-e-") for row in rows), 0)
        self.assertEqual(sum(row.get(CAL) is True for row in rows), 64 + 50)
        self.assertEqual(manifest["row_counts"]["by_task"][DD], 154)

    def test_r9_shape_reserves_the_changed_pairs_under_the_guard(self) -> None:
        source, candidates = _fixture()
        for mode in ("off", "on"):
            with self.subTest(cross_task_visual_dedup=mode):
                rows, manifest = _materialize(source, candidates, guard_on=True, cross_task_visual_dedup=mode)
                self._assert_r9_reserved(rows, manifest, protected=0)
        # the same feed pairs also routed as mined candidates (listed first): the calibration copy
        # wins and the strict copies are removed from the mined candidate set before selection
        source, candidates = _fixture(overlap=True)
        rows, manifest = _materialize(source, candidates, guard_on=True)
        self._assert_r9_reserved(rows, manifest, protected=78)
        self.assertEqual(manifest["uniqueness"].get("exact_record_duplicates_excluded", 0), 0)

    def test_guard_off_is_byte_identical_to_head(self) -> None:
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        rows, manifest = _golden_materializer_rows()
        self.assertEqual(_rows_sha256(rows), golden["materializer"]["sha256"])
        self.assertIsNone(manifest["calibration_reserve_rows_protected"])
        self.assertEqual(manifest["calibration_shortfall_accepted_under_guard"], {DD: 0, REF: 0, "total": 0})
        self.assertNotIn("reference_calibration_contract_reached_or_yielded", manifest["verification"])
        self.assertIn("reference_calibration_contract_reached", manifest["verification"])
        fixtures = {
            "r9": (_fixture(), None),
            "short_reserve": (_fixture(changed=15), None),
            "single_image": (_fixture(fewbox=20), _single_image_prior()),
        }
        for name, ((source, candidates), prior) in fixtures.items():
            with self.subTest(fixture=name):
                rows, manifest = _materialize(source, candidates, guard_on=False, prior=prior)
                sha, count, verified = HEAD_GUARD_OFF[name]
                self.assertEqual((len(rows), manifest["verified"]), (count, verified), manifest["verification"])
                self.assertEqual(_rows_sha256(rows), sha)
                self.assertIsNone(manifest["calibration_reserve_rows_protected"])
                self.assertEqual(manifest["calibration_shortfall_accepted_under_guard"]["total"], 0)
                self.assertIsNone(manifest["guard_aware_calibration"]["status"])
        # the overlap shape at guard-off: the strict copies win by candidate order, the reserve is
        # short and the materializer fails closed as at HEAD (no protection without the guard)
        source, candidates = _fixture(overlap=True)
        rows, manifest = _materialize(source, candidates, guard_on=False)
        self.assertFalse(manifest["verified"])
        self.assertFalse(manifest["verification"]["reference_calibration_contract_reached"])
        self.assertEqual(manifest["uniqueness"]["exact_record_duplicates_excluded"], 78)
        self.assertIsNone(manifest["calibration_reserve_rows_protected"])


class GuardShortfallAcceptanceTests(unittest.TestCase):
    """(b) / (c) / (d): a protected reserve that is still short at a zero empty headroom yields the
    missing slot rows to mined rows under the guard and fails closed without it."""

    def test_short_changed_reserve_at_zero_headroom_is_accepted_and_mined_rows_fill(self) -> None:
        source, candidates = _fixture(changed=15)
        rows, manifest = _materialize(source, candidates, guard_on=True)
        self.assertTrue(manifest["verified"], manifest["verification"])
        self.assertEqual(len(rows), 1024)
        self.assertEqual(manifest["calibration_empty_headroom"][f"task:{REF}"], 0)
        detail = manifest["guard_aware_calibration"]
        # the KPI-rate first pass over 28 + 15 pairs stops when the changed pairs run out (34 pairs,
        # 19 no-change); its no-change count is the KPI target the split substitutes
        self.assertEqual(detail["kpi_empty_targets"], {DD: 32, REF: 19})
        self.assertEqual(detail["headroom_empty_targets"][REF], 0)
        # 15 changed pairs are all the non-empty reserve holds: the slot keeps them, adds no
        # no-change pair beyond the zero headroom and yields the other 35 rows
        reference = manifest["reference_calibration"]
        self.assertEqual((reference["selected_no_change"], reference["selected_changed"], reference["selected_total"]), (0, 15, 15))
        self.assertEqual((reference["requested_total"], reference["effective_total"], reference["shortfall_accepted_under_guard"]), (50, 15, 35))
        self.assertEqual((reference["target_no_change"], reference["kpi_target_no_change"]), (0, 28))
        self.assertEqual((reference["no_change_substituted_by_changed"], reference["no_change_beyond_headroom"]), (19, 0))
        self.assertEqual(detail["fewbox_shortfall"], {DD: 0, REF: 35})
        self.assertEqual(detail["shortfall_accepted_under_guard"], {DD: 0, REF: 35})
        self.assertEqual(detail["status"], "substituted_with_guard_shortfall")
        self.assertEqual(manifest["calibration_headroom_overflow_rows"], {DD: 0, REF: 0, "total": 0})
        self.assertEqual(manifest["calibration_shortfall_accepted_under_guard"], {DD: 0, REF: 35, "total": 35})
        self.assertEqual(manifest["reference_calibration_shortfall_accepted_under_guard"], 35)
        self.assertEqual(manifest["calibration_reserve_rows_protected"], {DD: 0, REF: 0})
        verification = manifest["verification"]
        self.assertNotIn("reference_calibration_contract_reached", verification)
        self.assertTrue(verification["reference_calibration_contract_reached_or_yielded"])
        self.assertTrue(verification["reference_empty_rate_matched"])
        self.assertTrue(verification["reference_calibration_content_unique"])
        # the 35 yielded slot rows became mined rows of the same task: the maintenance slice keeps
        # its 870 rows (15 calibration + 855 mined against 50 + 820 with a full reserve) and the
        # extra mined rows are reference pairs (the other tasks' supply is exhausted)
        self.assertEqual(sum(row.get(CAL) is True for row in rows), 64 + 15)
        self.assertEqual(manifest["row_counts"]["by_task"][DD], 154)
        self.assertEqual(manifest["pre_repetition"]["maintenance"], 870)
        self.assertEqual(manifest["mined_task_fill_realized"][REF], manifest["row_counts"]["by_task"][REF] - 15)
        _, full = _materialize(*_fixture(), guard_on=True)
        self.assertEqual(full["pre_repetition"]["maintenance"], 870)
        self.assertEqual(manifest["mined_task_fill_realized"][REF], full["mined_task_fill_realized"][REF] + 35)
        for task in (DC, CD, CC, REF_DC):
            self.assertEqual(manifest["mined_task_fill_realized"][task], full["mined_task_fill_realized"][task])
        self.assertEqual(sum(row["id"].startswith("c-ref-e-") for row in rows), 0)
        # the bound v2 manifest carries the accepted shortfall and its content gate uses the
        # effective total, so the iteration binds and verifies
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined = _write(root / "mined.jsonl", rows)
            bound_input = dda.bind_manifest(manifest, mined)
            train_rows, summary = atj.assemble(None, mined, validation_paths=[], media_root=MEDIA, row_multiple=64)
            train = _write(root / "train.jsonl", train_rows)
            summary = atj.bind_summary(summary, train)
            bound = dda.bind_cumulative_manifest(bound_input, current_jsonl=mined, training_jsonl=train,
                                                 assembly_summary=summary, epochs=1, global_batch=64)
            self.assertTrue(bound["verified"])
            current = bound["current_selection"]
            self.assertEqual(current["calibration_shortfall_accepted_under_guard"], {DD: 0, REF: 35, "total": 35})
            self.assertEqual(current["reference_calibration_shortfall_accepted_under_guard"], 35)
            self.assertEqual(current["calibration_reserve_rows_protected"], {DD: 0, REF: 0})
            self.assertEqual(current["reference_calibration"]["effective_total"], 15)
            (root / "quota.json").write_text(json.dumps(bound))
            dda.verify_bound_manifest(root / "quota.json", training_jsonl=train, expected_rows=len(train_rows), epochs=1, global_batch=64)
        for key in ("calibration_reserve_rows_protected", "calibration_shortfall_accepted_under_guard",
                    "reference_calibration_shortfall_accepted_under_guard", "single_image_calibration_shortfall_accepted_under_guard"):
            self.assertIn(key, dda.CURRENT_SELECTION_COPIED_KEYS)
        self.assertIn("substituted_with_guard_shortfall", dda.GUARD_AWARE_CALIBRATION_STATUSES)

    def test_the_same_short_reserve_fails_closed_without_the_guard(self) -> None:
        source, candidates = _fixture(changed=15)
        rows, manifest = _materialize(source, candidates, guard_on=False)
        sha, count, verified = HEAD_GUARD_OFF["short_reserve"]
        self.assertFalse(verified)
        self.assertEqual((len(rows), manifest["verified"]), (count, False))
        self.assertEqual(_rows_sha256(rows), sha)
        self.assertFalse(manifest["verification"]["reference_calibration_contract_reached"])
        self.assertNotIn("reference_calibration_contract_reached_or_yielded", manifest["verification"])
        self.assertLess(manifest["reference_calibration"]["selected_total"], 50)
        self.assertEqual(manifest["reference_calibration"]["shortfall_accepted_under_guard"], 0)
        self.assertEqual(manifest["reference_calibration"]["effective_total"], 50)
        self.assertEqual(manifest["calibration_shortfall_accepted_under_guard"]["total"], 0)
        # a short reserve while headroom is available is not guard-induced: still fails closed
        # (the guard on, the reference cap not binding: the prior corpus is well under it)
        rows, manifest = _materialize(source, candidates, guard_on=True, prior=_single_image_prior())
        self.assertFalse(manifest["verified"])
        self.assertGreater(manifest["guard_aware_calibration"]["headroom_empty_targets"][REF], 0)
        self.assertEqual(manifest["calibration_shortfall_accepted_under_guard"]["total"], 0)
        self.assertIn("reference_calibration_contract_reached", manifest["verification"])
        self.assertFalse(manifest["verification"]["reference_calibration_contract_reached"])

    def test_single_image_short_fewbox_reserve_at_zero_headroom_is_accepted(self) -> None:
        source, candidates = _fixture(fewbox=20)
        prior = _single_image_prior()
        rows, manifest = _materialize(source, candidates, guard_on=True, prior=prior)
        self.assertTrue(manifest["verified"], manifest["verification"])
        self.assertEqual(len(rows), 1024)
        self.assertEqual(manifest["calibration_empty_headroom"][f"task:{DD}"], 0)
        detail = manifest["guard_aware_calibration"]
        self.assertEqual(detail["headroom_empty_targets"][DD], 0)
        self.assertEqual(detail["headroom_empty_targets"][REF], 28)  # the reference slot is untouched
        # 20 few-box rows are all the reserve holds: the slot keeps them, adds no empty beyond the
        # zero headroom and yields the other 44 rows (to strict Defect Detection rows)
        single = manifest["single_image_calibration"]
        self.assertEqual((single["selected_empty"], single["selected_few_box"], single["selected_total"]), (0, 20, 20))
        self.assertEqual((single["empty_substituted_by_few_box"], single["empty_beyond_headroom"], single["shortfall_accepted_under_guard"]), (32, 0, 44))
        self.assertEqual(single["max_few_box_effective"], 64)
        self.assertEqual(detail["fewbox_shortfall"], {DD: 44, REF: 0})
        self.assertEqual(detail["shortfall_accepted_under_guard"], {DD: 44, REF: 0})
        self.assertEqual(detail["status"], "substituted_with_guard_shortfall")
        self.assertEqual(manifest["calibration_headroom_overflow_rows"], {DD: 0, REF: 0, "total": 0})
        self.assertEqual(manifest["calibration_shortfall_accepted_under_guard"], {DD: 44, REF: 0, "total": 44})
        self.assertEqual(manifest["single_image_calibration_shortfall_accepted_under_guard"], 44)
        self.assertEqual(manifest["reference_calibration_shortfall_accepted_under_guard"], 0)
        reference = manifest["reference_calibration"]
        self.assertEqual((reference["selected_no_change"], reference["selected_changed"], reference["selected_total"]), (28, 22, 50))
        self.assertTrue(manifest["verification"]["reference_calibration_contract_reached"])
        self.assertTrue(manifest["verification"]["reference_calibration_contract_reached_or_yielded"])
        self.assertTrue(manifest["verification"]["single_image_calibration_caps_respected"])
        self.assertEqual(sum(row.get(CAL) is True for row in rows), 20 + 50)
        self.assertEqual(sum(row["id"].startswith("c-dd-e-") for row in rows), 0)
        self.assertEqual(manifest["row_counts"]["by_task"][DD], 154)
        self.assertEqual(manifest["row_counts"]["task_strict_defect_detection"], 134)
        # guard off on the same inputs: HEAD byte-for-byte (the slot is soft there and fills with empties)
        rows, off = _materialize(source, candidates, guard_on=False, prior=prior)
        sha, count, verified = HEAD_GUARD_OFF["single_image"]
        self.assertEqual((len(rows), off["verified"], _rows_sha256(rows)), (count, verified, sha))
        self.assertEqual(off["single_image_calibration"]["selected_empty"], 32)
        self.assertEqual(off["single_image_calibration"]["shortfall_accepted_under_guard"], 0)

    def test_cli_prints_the_accepted_shortfall_on_the_success_line(self) -> None:
        source, candidates = _fixture(changed=15)
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source_path = _write(root / "source.jsonl", source)
            proxy = _write(root / "proxy.jsonl", _proxy_rows(root))  # reference empty rate 0.5 -> 25 no-change
            argv = [
                "--candidate-parquet", str(root / "candidates.parquet"), "--source-annotations", str(source_path),
                "--proxy-annotations", str(proxy), "--media-root", str(MEDIA), "--max-rows", "1024", "--minimum-rows", "64",
                "--row-multiple", "64", "--epochs", "1", "--global-batch", "64", "--no-near-duplicate-filter",
                "--defect-detection-fraction", "0.15", "--mined-task-fill-order", ",".join(FILL_ORDER),
                "--cross-task-visual-dedup", "off", "--zero-new-candidate-policy", "skip_exhausted",
                "--single-image-calibration-max-empty", "32", "--single-image-calibration-max-few", "32",
                "--reference-calibration-total", "50", "--calibration-guard-aware", "on",
                "--max-empty-answer-share-task", f"{DD}=0.45", "--max-empty-answer-share-task", f"{REF}=0.40",
                "--output", str(root / "mined.jsonl"), "--manifest", str(root / "quota.json"),
            ]
            with mock.patch.object(dda, "_read_parquet", return_value=candidates):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    rc = dda.main(argv)
            self.assertEqual(rc, 0, stdout.getvalue())
            line = stdout.getvalue().strip()
            self.assertIn("verified=true", line)
            self.assertIn("calibration_shortfall_accepted_under_guard=35", line)
            self.assertIn(f"{REF}=35", line)
            manifest = json.loads((root / "quota.json").read_text())
            self.assertTrue(manifest["verified"])
            self.assertEqual(manifest["reference_calibration_shortfall_accepted_under_guard"], 35)
            self.assertEqual(manifest["guard_aware_calibration"]["status"], "substituted_with_guard_shortfall")
            self.assertEqual(manifest["training_jsonl"]["rows"], 1024)
            # guard off: the same feed fails closed and the CLI exits 2 with the manifest written
            stderr = io.StringIO()
            with mock.patch.object(dda, "_read_parquet", return_value=candidates):
                with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
                    rc = dda.main([*argv[:argv.index("--calibration-guard-aware")], *argv[argv.index("--output"):]])
            self.assertEqual(rc, 2)
            self.assertIn("not verified", stderr.getvalue())
            self.assertFalse(json.loads((root / "quota.json").read_text())["verified"])


if __name__ == "__main__":
    unittest.main()
