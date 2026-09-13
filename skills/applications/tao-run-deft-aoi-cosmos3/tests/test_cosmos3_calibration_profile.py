# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Profile-matched calibration: quotas follow the KPI set's box-count bins per detection task."""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import init_deft_state  # noqa: E402
import select_detection_calibration as sdc  # noqa: E402
from test_cosmos3_detection_calibration_contract import _row  # noqa: E402


def _boxes(n: int) -> list[dict]:
    return [{"bbox_2d": [i, i, i + 1, i + 1], "label": "x"} for i in range(n)]


def _materialize_images(media_root: pathlib.Path, rows: list[dict]) -> None:
    from PIL import Image

    (media_root / "images").mkdir(exist_ok=True)
    for record_index, record in enumerate(rows):
        for image_index, item in enumerate(record["messages"][0]["content"]):
            if item.get("type") == "image":
                Image.new("RGB", (2, 2), color=(record_index % 255, image_index, 1)).save(media_root / item["image"])


class ProfileDerivationTests(unittest.TestCase):
    def test_count_bins_and_profiles(self) -> None:
        self.assertEqual([sdc.count_bin(n) for n in (0, 1, 2, 3, 4, 9, 10, 40)], ["0", "1", "2-3", "2-3", "4-9", "4-9", "10+", "10+"])
        kpi = [_row("d0", []), _row("d1", _boxes(1)), _row("d2", _boxes(3)), _row("d3", _boxes(6)),
               _row("c1", _boxes(5), task="Component Detection"),
               _row("r0", [], task="Ref_based Defect Detection"), _row("r1", _boxes(4), task="Ref_based Defect Detection")]
        profiles = sdc.derive_task_count_profiles(kpi)
        self.assertEqual(profiles["Defect Detection"]["rows"], 4)
        self.assertEqual(profiles["Defect Detection"]["bins"], {"0": 1, "1": 1, "2-3": 1, "4-9": 1, "10+": 0})
        self.assertAlmostEqual(profiles["Defect Detection"]["empty_rate"], 0.25)
        self.assertEqual(profiles["Component Detection"]["bins"]["4-9"], 1)
        self.assertAlmostEqual(profiles["Ref_based Defect Detection"]["shares"]["0"], 0.5)

    def test_bin_quotas_follow_shares_with_largest_remainder(self) -> None:
        kpi = [_row(f"d{i}", _boxes(n)) for i, n in enumerate([0, 0, 0, 1, 1, 1, 1, 2, 2, 5])]  # 30% / 40% / 20% / 10% / 0%
        profiles = sdc.derive_task_count_profiles(kpi)
        quotas = sdc.profile_bin_quotas({"Defect Detection": 10}, profiles)
        self.assertEqual(quotas["Defect Detection"], {"0": 3, "1": 4, "2-3": 2, "4-9": 1, "10+": 0})
        quotas = sdc.profile_bin_quotas({"Defect Detection": 7}, profiles)
        self.assertEqual(sum(quotas["Defect Detection"].values()), 7)
        with self.assertRaisesRegex(ValueError, "no Component Detection rows"):
            sdc.profile_bin_quotas({"Component Detection": 4}, profiles)
        with self.assertRaisesRegex(ValueError, "non-detection task"):
            sdc.profile_bin_quotas({"Defect Classification": 4}, profiles)
        self.assertEqual(sdc.parse_task_totals(["Defect Detection=512", "Ref_based Defect Detection=500"]),
                         {"Defect Detection": 512, "Ref_based Defect Detection": 500})
        with self.assertRaises(ValueError):
            sdc.parse_task_totals(["Defect Detection"])


class ProfileSelectionTests(unittest.TestCase):
    def _source(self) -> list[dict]:
        rows = []
        rows += [_row(f"e{i}", []) for i in range(6)]
        rows += [_row(f"one{i}", _boxes(1)) for i in range(6)]
        rows += [_row(f"two{i}", _boxes(3)) for i in range(6)]
        rows += [_row(f"many{i}", _boxes(7)) for i in range(6)]
        rows += [_row(f"huge{i}", _boxes(12)) for i in range(2)]
        rows += [_row(f"rz{i}", [], task="Ref_based Defect Detection") for i in range(4)]
        rows += [_row(f"rm{i}", _boxes(5), task="Ref_based Defect Detection") for i in range(4)]
        rows += [_row(f"cd{i}", _boxes(8), task="Component Detection") for i in range(3)]  # not requested -> ignored
        return rows

    def test_profile_mode_fills_every_bin_including_many_box_rows_and_pairs(self) -> None:
        source = self._source()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary); _materialize_images(root, source)
            quotas = {"Defect Detection": {"0": 2, "1": 3, "2-3": 2, "4-9": 2, "10+": 1},
                      "Ref_based Defect Detection": {"0": 2, "1": 0, "2-3": 0, "4-9": 3, "10+": 0}}
            selected, summary = sdc.select_calibration(source, media_root=root, pair_assets_dir=root / "pair-assets", task_bin_quotas=quotas)
        self.assertEqual(summary["policy"], "kpi_profile_count_bins")
        self.assertEqual(len(selected), 15)
        bins = {}
        for row in selected:
            bins.setdefault(row["routed_task_types"][0], {}).setdefault(row["calibration_count_bin"], 0)
            bins[row["routed_task_types"][0]][row["calibration_count_bin"]] += 1
        self.assertEqual(bins["Defect Detection"], {"0": 2, "1": 3, "2-3": 2, "4-9": 2, "10+": 1})
        self.assertEqual(bins["Ref_based Defect Detection"], {"0": 2, "4-9": 3})
        self.assertNotIn("Component Detection", bins)
        many = [r for r in selected if r["calibration_count_bin"] == "10+"][0]
        self.assertEqual(many["calibration_box_count"], 12)
        self.assertIn("calibration_few_box_ground_truth", many["defect_detection_evidence"])  # legacy label kept
        empty_pair = [r for r in selected if r["routed_task_types"] == ["Ref_based Defect Detection"] and r["calibration_box_count"] == 0][0]
        self.assertIn("calibration_reference_no_change_ground_truth", empty_pair["defect_detection_evidence"])
        self.assertEqual(summary["tasks"]["Defect Detection"]["fill_fraction"], 1.0)
        self.assertEqual(summary["tasks"]["Ref_based Defect Detection"]["shortage"], {})

    def test_shortage_reported_and_fail_closed_below_min_fill(self) -> None:
        source = self._source()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary); _materialize_images(root, source)
            quotas = {"Defect Detection": {"0": 2, "1": 2, "2-3": 2, "4-9": 2, "10+": 4}}  # only 2 huge rows exist
            selected, summary = sdc.select_calibration(source, media_root=root, pair_assets_dir=root / "pair-assets",
                                                      task_bin_quotas=quotas, min_fill_fraction=0.8)
            self.assertEqual(len(selected), 10)
            self.assertEqual(summary["tasks"]["Defect Detection"]["shortage"], {"10+": 2})
            self.assertAlmostEqual(summary["tasks"]["Defect Detection"]["fill_fraction"], 10 / 12, places=4)
            with self.assertRaisesRegex(ValueError, "profile calibration quotas cannot be filled"):
                sdc.select_calibration(source, media_root=root, pair_assets_dir=root / "pair-assets",
                                       task_bin_quotas=quotas, min_fill_fraction=0.95)

    def test_profile_mode_excludes_cohort_and_legacy_quotas(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            sdc.select_calibration([], media_root=pathlib.Path("."), task_bin_quotas={"Defect Detection": {"0": 1, "1": 0, "2-3": 0, "4-9": 0, "10+": 0}}, max_empty=1, max_few=1)
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            sdc.select_calibration([], media_root=pathlib.Path("."), task_bin_quotas={"Defect Detection": {"0": 1}})

    def test_cli_profile_totals_write_summary_with_profiles(self) -> None:
        source = self._source()
        kpi = [_row(f"k{i}", _boxes(n)) for i, n in enumerate([0, 1, 1, 3, 7, 12])] + \
              [_row(f"kr{i}", _boxes(n), task="Ref_based Defect Detection") for i, n in enumerate([0, 0, 5])]
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary); _materialize_images(root, source + kpi)
            src = root / "source.jsonl"; src.write_text("".join(json.dumps(r) + "\n" for r in source))
            kp = root / "kpi.jsonl"; kp.write_text("".join(json.dumps(r) + "\n" for r in kpi))
            try:
                import pyarrow  # noqa: F401
            except ImportError:
                self.skipTest("pyarrow not installed")
            rc = sdc.main(["--source-annotations", str(src), "--media-root", str(root), "--proxy-annotations", str(kp),
                           "--pair-assets-dir", str(root / "pair-assets"), "--profile-task-total", "Defect Detection=6",
                           "--profile-task-total", "Ref_based Defect Detection=3",
                           "--output", str(root / "out.parquet"), "--summary", str(root / "summary.json")])
            self.assertEqual(rc, 0)
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["policy"], "kpi_profile_count_bins")
            self.assertEqual(summary["task_totals"], {"Defect Detection": 6, "Ref_based Defect Detection": 3})
            self.assertEqual(sum(summary["tasks"]["Defect Detection"]["bins"][b]["requested"] for b in summary["count_bins"]), 6)
            self.assertEqual(summary["task_profiles"]["Ref_based Defect Detection"]["bins"]["0"], 2)
            rc = sdc.main(["--source-annotations", str(src), "--media-root", str(root), "--proxy-annotations", str(kp),
                           "--pair-assets-dir", str(root / "pair-assets"), "--profile-task-total", "Defect Detection=6",
                           "--single-image-total", "4", "--reference-total", "2",
                           "--output", str(root / "out2.parquet"), "--summary", str(root / "summary2.json")])
            self.assertEqual(rc, 2)


class InitProfileContractTests(unittest.TestCase):
    def test_init_records_profile_quotas_from_the_kpi_set(self) -> None:
        from test_cosmos3_init_state_contract import Cosmos3InitStateContractTests as Base
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = Base._workspace(root)
            rc = init_deft_state.main(Base._argv(root, workspace, "--calibration-task-total", "Defect Detection=8",
                                                 "--calibration-task-total", "Ref_based Defect Detection=4"))
            self.assertEqual(rc, 0)
            contract = json.loads((root / "results/deft_state.json").read_text())["config"]["mining"]["calibration_quota_contract"]
            self.assertEqual(contract["policy"], "kpi_profile_count_bins")
            self.assertEqual(contract["task_totals"], {"Defect Detection": 8, "Ref_based Defect Detection": 4})
            self.assertEqual(sum(contract["task_bin_quotas"]["Defect Detection"].values()), 8)
            self.assertEqual(contract["count_bins"], ["0", "1", "2-3", "4-9", "10+"])
            self.assertEqual(contract["min_fill_fraction"], 0.9)
            self.assertIn("Defect Detection", contract["task_profiles"])
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = init_deft_state.main(Base._argv(root / "b", workspace, "--calibration-task-total", "Defect Detection=8",
                                                     "--single-image-calibration-max-empty", "1", "--single-image-calibration-max-few", "1",
                                                     "--reference-calibration-total", "1"))
            self.assertNotEqual(rc, 0)
            self.assertIn("cannot be combined", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
