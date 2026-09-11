# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

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
import build_coverage_candidates  # noqa: E402
import coverage_rows  # noqa: E402
import init_deft_state  # noqa: E402
import render_iteration_mining_runner  # noqa: E402


def _row(record_id: str, task_type: str = "Component Classification", dataset: str = "dsA", status: str | None = None) -> dict:
    row = {
        "id": record_id,
        "task_type": task_type,
        "dataset": dataset,
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": f"images/{dataset}/{record_id}.png", "min_pixels": 1, "max_pixels": 1},
                {"type": "text", "text": "classify"},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "A"}]},
        ],
    }
    if status is not None:
        row[coverage_rows.POOL_STATUS_KEY] = status
    return row


def _write(path: pathlib.Path, rows: list[dict]) -> pathlib.Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _coverage_candidates() -> list[dict]:
    rows = [_row(f"c0-{i}", "Component Classification", "ds0", "residual" if i < 10 else "correct") for i in range(20)]
    rows += [_row(f"c1-{i}", "Component Classification", "ds1", "residual") for i in range(20)]
    rows += [_row(f"dp-{i}", "Defect Detection", "pool", "correct") for i in range(20)]
    rows.append(_row("m-3", "Defect Detection", "mine", "residual"))  # duplicate of a mined row id -> skipped
    return rows


class CoverageRowsTests(unittest.TestCase):
    def test_config_validation(self) -> None:
        self.assertFalse(coverage_rows.validate_coverage_config(None, None, None, None)["enabled"])
        with self.assertRaises(ValueError):
            coverage_rows.validate_coverage_config(0.05, None, pathlib.Path("x"), None)
        with self.assertRaises(ValueError):
            coverage_rows.validate_coverage_config(0.05, "plain", None, None)
        with self.assertRaises(ValueError):
            coverage_rows.validate_coverage_config(0.05, "plain", pathlib.Path("x"), -1)
        cfg = coverage_rows.validate_coverage_config(0.05, "residual", pathlib.Path("x"), None)
        self.assertEqual((cfg["enabled"], cfg["mode"], cfg["min_rows_per_cell"]), (True, "residual", 8))

    def test_joint_targets_match_the_single_slice_formula_and_subtract_existing(self) -> None:
        only_anchor = coverage_rows.joint_targets(900, {"anchor": 0.10, "coverage": 0.0}, {"anchor": 0, "coverage": 0})
        self.assertEqual(only_anchor, {"anchor": (100, 100), "coverage": (0, 0)})
        self.assertEqual(only_anchor["anchor"], anchor_rows.anchor_target_rows(0.10, 900, 0))
        both = coverage_rows.joint_targets(900, {"anchor": 0.10, "coverage": 0.05}, {"anchor": 40, "coverage": 60})
        self.assertEqual(both, {"anchor": (106, 66), "coverage": (53, 0)})
        with self.assertRaises(ValueError):
            coverage_rows.joint_targets(10, {"a": 0.6, "b": 0.5}, {})

    def test_plain_allocation_spreads_evenly_and_fills_the_lowest_cells_first(self) -> None:
        picked, report = coverage_rows.select_coverage(
            _coverage_candidates(), new_rows=9, mode="plain", min_rows_per_cell=8, seed=17,
            is_excluded=lambda r: "already_in_corpus" if r["id"] == "m-3" else None,
        )
        self.assertEqual(len(picked), 9)
        self.assertEqual(report["per_cell_new"], {"Component Classification|ds0": 3, "Component Classification|ds1": 3, "Defect Detection|pool": 3})
        self.assertEqual(report["skipped"], {"already_in_corpus": 1})
        self.assertEqual(len(report["cells_below_floor_after_selection"]), 3)
        self.assertFalse(any(is_fallback for _, is_fallback in picked))
        exclude_mined = lambda r: "already_in_corpus" if r["id"] == "m-3" else None  # noqa: E731
        again, _ = coverage_rows.select_coverage(
            list(reversed(_coverage_candidates())), new_rows=9, mode="plain", min_rows_per_cell=8, seed=17,
            is_excluded=exclude_mined,
        )
        self.assertEqual([r["id"] for r, _ in picked], [r["id"] for r, _ in again])
        topped, report2 = coverage_rows.select_coverage(
            _coverage_candidates(), new_rows=6, mode="plain", min_rows_per_cell=8, seed=17,
            is_excluded=exclude_mined, prior_cell_counts={("Component Classification", "ds0"): 5},
        )
        self.assertEqual(report2["per_cell_new"], {"Component Classification|ds1": 3, "Defect Detection|pool": 3})
        self.assertEqual(report2["per_cell_cumulative"]["Component Classification|ds0"], 5)

    def test_residual_mode_prefers_residual_rows_and_falls_back_to_correct_rows_tagged(self) -> None:
        picked, report = coverage_rows.select_coverage(
            _coverage_candidates(), new_rows=36, mode="residual", min_rows_per_cell=0, seed=17,
            is_excluded=lambda r: "already_in_corpus" if r["id"] == "m-3" else None,
        )
        self.assertEqual(len(picked), 36)
        by_cell = Counter((coverage_rows.cell_of(r), is_fallback) for r, is_fallback in picked)
        self.assertEqual(by_cell[(("Component Classification", "ds0"), False)], 10)
        self.assertEqual(by_cell[(("Component Classification", "ds0"), True)], 2)
        self.assertEqual(by_cell[(("Component Classification", "ds1"), False)], 12)
        self.assertEqual(by_cell[(("Defect Detection", "pool"), True)], 12)
        self.assertEqual(report["fallback_correct_rows"], 14)
        self.assertEqual(report["selected_pool_status_counts"], {"correct": 14, "residual": 22})
        self.assertEqual(report["cells_below_floor_after_selection"], {})

    def test_floor_check_fails_closed_only_when_the_eventual_budget_is_too_small(self) -> None:
        with self.assertRaisesRegex(ValueError, "floor cannot be met"):
            coverage_rows.check_floor_budget(3, 8, 5)
        self.assertEqual(coverage_rows.check_floor_budget(3, 8, 24)["floor_rows_needed"], 24)
        self.assertIsNone(coverage_rows.check_floor_budget(3, 8, None)["budget_total"])


class AssemblerCoverageTests(unittest.TestCase):
    def _corpus(self, root: pathlib.Path):
        mined = _write(root / "mined.jsonl", [_row(f"m-{i}", "Defect Detection", "mine") for i in range(90)])
        coverage = _write(root / "coverage_candidates_plain.jsonl", _coverage_candidates())
        anchors = [_row(f"a-{i}", "Component Classification", f"ds{i % 3}") for i in range(200)]
        anchors += [_row(f"b-{i}", "Defect Detection", "pool2") for i in range(200)]
        anchor_source = _write(root / "anchor_candidates.jsonl", anchors)
        kpi = _write(root / "kpi.jsonl", [_row(f"k-{i}", "Component Classification") for i in range(50)]
                     + [_row(f"kd-{i}", "Defect Detection") for i in range(50)])
        return mined, coverage, anchor_source, kpi

    def test_off_by_default_leaves_the_corpus_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined, _, _, _ = self._corpus(root)
            rows, summary = assemble_training_json.assemble(None, mined, validation_paths=[], media_root=root)
            self.assertEqual(len(rows), 90)
            self.assertFalse(summary["coverage_blend"]["enabled"])
            self.assertEqual(summary["materialized_coverage_records"], 0)
            self.assertEqual(summary["exposure_rows"]["total"], 90)
            self.assertEqual(summary["exposure_rows"]["current_mining"], 90)
            self.assertFalse(any("deft_coverage" in r for r in rows))

    def test_plain_blend_reaches_the_share_strips_status_and_marks_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined, coverage, _, _ = self._corpus(root)
            cfg = coverage_rows.validate_coverage_config(0.10, "plain", coverage, 0)
            rows, summary = assemble_training_json.assemble(
                None, mined, validation_paths=[], media_root=root, coverage_config=cfg, coverage_seed=17
            )
            self.assertEqual(len(rows), 100)
            self.assertEqual(summary["materialized_coverage_records"], 10)
            self.assertAlmostEqual(summary["coverage_blend"]["realized_share_rows"], 0.10, places=6)
            kinds = [p["source_kind"] for p in summary["provenance"]]
            self.assertEqual(kinds[:90], ["current_mining"] * 90)
            self.assertEqual(kinds[90:], ["coverage_blend"] * 10)
            cov = rows[90:]
            self.assertTrue(all(r.get("deft_coverage") is True for r in cov))
            self.assertFalse(any(coverage_rows.POOL_STATUS_KEY in r for r in rows))
            self.assertEqual(summary["coverage_blend"]["selection"]["skipped"], {"already_in_corpus": 1})
            self.assertEqual(summary["coverage_blend"]["selection"]["per_cell_new"],
                             {"Component Classification|ds0": 4, "Component Classification|ds1": 3, "Defect Detection|pool": 3})
            self.assertTrue(all(p["pool_status"] in {"correct", "residual"} for p in summary["provenance"][90:]))

    def test_residual_blend_tags_fallback_rows_and_reports_the_status_mix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined, coverage, _, _ = self._corpus(root)
            cfg = coverage_rows.validate_coverage_config(0.25, "residual", coverage, 0)
            rows, summary = assemble_training_json.assemble(
                None, mined, validation_paths=[], media_root=root, coverage_config=cfg
            )
            self.assertEqual(len(rows), 120)  # 90 / 0.75
            self.assertEqual(summary["materialized_coverage_records"], 30)
            self.assertEqual(summary["coverage_blend"]["materialized_fallback_correct_records"], 10)
            fallback = [p for p in summary["provenance"] if "anchor" in p.get("purpose_tags", [])]
            self.assertEqual(len(fallback), 10)
            self.assertTrue(all(p["id"].startswith("dp-") for p in fallback))
            self.assertFalse(any(r.get("deft_anchor") for r in rows))
            self.assertEqual(summary["coverage_blend"]["selection"]["selected_pool_status_counts"], {"correct": 10, "residual": 20})

    def test_anchors_and_coverage_share_the_cap_and_current_rows_absorb_the_trim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined, coverage, anchor_source, kpi = self._corpus(root)
            anchor_cfg = anchor_rows.validate_anchor_config(0.10, anchor_source, kpi, None)
            cov_cfg = coverage_rows.validate_coverage_config(0.05, "plain", coverage, 0)
            rows, summary = assemble_training_json.assemble(
                None, mined, validation_paths=[], media_root=root, anchor_config=anchor_cfg,
                coverage_config=cov_cfg, max_rows=96, row_multiple=32,
            )
            self.assertEqual(len(rows), 96)
            self.assertEqual(summary["materialized_anchor_records"], 10)
            self.assertEqual(summary["materialized_coverage_records"], 5)
            self.assertEqual(summary["materialized_current_records"], 81)
            self.assertAlmostEqual(summary["anchor"]["realized_share_rows"], 10 / 96, places=6)
            self.assertAlmostEqual(summary["coverage_blend"]["realized_share_rows"], 5 / 96, places=6)
            # only 90 mined rows exist, so 9 of them (90 - 81) were displaced by the 15 slice slots
            displaced = (summary["anchor"]["cap_reservation"]["current_rows_displaced_by_anchors"]
                         + summary["coverage_blend"]["cap_reservation"]["current_rows_displaced_by_coverage"])
            self.assertEqual(displaced, 9)
            self.assertEqual(summary["coverage_blend"]["cap_reservation"]["new_coverage_slots"], 5)
            kinds = [p["source_kind"] for p in summary["provenance"]]
            self.assertEqual(kinds, ["current_mining"] * 81 + ["coverage_blend"] * 5 + ["anchor_correct"] * 10)
            ledger = summary["exposure_rows"]
            self.assertEqual((ledger["current_mining"], ledger["anchor_new"], ledger["coverage_new"], ledger["total"]), (81, 10, 5, 96))

    def test_second_iteration_tops_up_coverage_under_the_cap_and_keeps_prior_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined, coverage, _, _ = self._corpus(root)
            cfg = coverage_rows.validate_coverage_config(0.10, "plain", coverage, 0)
            rows, _ = assemble_training_json.assemble(None, mined, validation_paths=[], media_root=root, coverage_config=cfg)
            previous = _write(root / "train1.jsonl", rows)  # 100 rows, 10 coverage
            mined2 = _write(root / "mined2.jsonl", [_row(f"n-{i}", "Defect Detection", "mine") for i in range(90)])
            rows2, summary2 = assemble_training_json.assemble(
                previous, mined2, previous_sha256=assemble_training_json.sha256_file(previous),
                validation_paths=[], media_root=root, coverage_config=cfg, max_rows=160, row_multiple=32,
            )
            self.assertEqual(len(rows2), 160)
            self.assertEqual(summary2["retained_previous_records"], 100)
            self.assertEqual(summary2["coverage_blend"]["prior_coverage_rows"], 10)
            self.assertEqual(summary2["materialized_coverage_records"], 6)
            self.assertEqual(summary2["coverage_blend"]["materialized_coverage_records_total"], 16)
            self.assertAlmostEqual(summary2["coverage_blend"]["realized_share_rows"], 0.10, places=6)
            self.assertEqual(sum(r.get("deft_coverage") is True for r in rows2), 16)
            cumulative = summary2["coverage_blend"]["materialized_per_cell_cumulative"]
            self.assertEqual(sum(cumulative.values()), 16)
            # the pre-cap selection asked for 10 new rows; the cap kept 6 of them
            self.assertEqual(summary2["coverage_blend"]["selection"]["selected_rows"], 10)
            self.assertEqual(summary2["exposure_rows"]["previous_coverage"], 10)
            self.assertEqual(summary2["exposure_rows"]["previous_mined"], 90)

    def test_remined_coverage_row_is_deduplicated_so_ids_stay_unique(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined, coverage, _, _ = self._corpus(root)
            cfg = coverage_rows.validate_coverage_config(0.10, "plain", coverage, 0)
            rows, _ = assemble_training_json.assemble(None, mined, validation_paths=[], media_root=root, coverage_config=cfg)
            cov = [r for r in rows if r.get("deft_coverage") is True]
            previous = _write(root / "train1.jsonl", rows)
            remined = [{k: v for k, v in cov[0].items() if k != "deft_coverage"}]
            mined2 = _write(root / "mined2.jsonl", remined + [_row(f"n-{i}", "Defect Detection", "mine") for i in range(89)])
            rows2, summary2 = assemble_training_json.assemble(
                previous, mined2, previous_sha256=assemble_training_json.sha256_file(previous),
                validation_paths=[], media_root=root, coverage_config=cfg,
            )
            ids = [r["id"] for r in rows2]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(summary2["duplicates_skipped"], 1)
            self.assertEqual(summary2["selected_current_records"], 89)
            self.assertEqual(sum(1 for r in rows2 if r["id"] == cov[0]["id"] and r.get("deft_coverage")), 1)
            self.assertEqual(summary2["coverage_blend"]["prior_coverage_rows"], 10)

    def test_floor_fails_closed_against_the_eventual_budget_not_the_first_slice(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined, coverage, _, _ = self._corpus(root)
            cfg = coverage_rows.validate_coverage_config(0.05, "plain", coverage, 8)
            with self.assertRaisesRegex(ValueError, "floor cannot be met"):
                assemble_training_json.assemble(
                    None, mined, validation_paths=[], media_root=root, coverage_config=cfg, max_rows=96, row_multiple=32
                )
            rows, summary = assemble_training_json.assemble(
                None, mined, validation_paths=[], media_root=root, coverage_config=cfg, max_rows=960
            )
            # first slice (5 rows) is below 3 cells x 8 but allowed: the eventual budget round(960*0.05)=48 fits
            self.assertEqual(summary["materialized_coverage_records"], 5)
            self.assertEqual(summary["coverage_blend"]["floor_check"]["budget_total"], 48)
            self.assertEqual(len(summary["coverage_blend"]["selection"]["cells_below_floor_after_selection"]), 3)

    def test_cli_writes_the_coverage_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined, coverage, _, _ = self._corpus(root)
            out = root / "train.jsonl"
            rc = assemble_training_json.main([
                "--mined-jsonl", str(mined), "--output", str(out), "--media-root", str(root),
                "--coverage-blend-share", "0.1", "--coverage-blend-mode", "plain", "--coverage-blend-source", str(coverage),
                "--coverage-blend-min-rows-per-dataset", "0",
            ])
            self.assertEqual(rc, 0)
            manifest = json.loads((root / "coverage_blend_manifest.json").read_text())
            self.assertTrue(manifest["enabled"])
            self.assertEqual(manifest["mode"], "plain")
            self.assertEqual(manifest["training_jsonl"]["rows"], 100)
            self.assertEqual(manifest["exposure_rows"]["coverage_new"], 10)
            summary = json.loads((root / "assemble_summary.json").read_text())
            self.assertEqual(summary["materialized_coverage_records"], 10)

    def test_coverage_and_repetition_cannot_be_combined(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined, coverage, _, _ = self._corpus(root)
            cfg = coverage_rows.validate_coverage_config(0.10, "plain", coverage, 0)
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                assemble_training_json.assemble(
                    None, mined, validation_paths=[], media_root=root, coverage_config=cfg,
                    repetition_config={"enabled": True, "row_cap": 96}, max_rows=96,
                )


class BuildCoverageCandidatesTests(unittest.TestCase):
    def _pool(self, root: pathlib.Path):
        pool = [_row(f"p-{i}", "Defect Detection", f"ds{i % 2}") for i in range(40)]
        pool += [_row(f"u-{i}", "Defect Detection", "ds1") for i in range(4)]  # not in the scored file -> unscored
        pool += [_row(f"seg-{i}", "Segmentation", "x") for i in range(3)]
        mining = _write(root / "mining.jsonl", pool)
        scored = _write(root / "scored.jsonl", [
            {"id": f"p-{i}", "task_type": "Defect Detection", "dataset": f"ds{i % 2}",
             "row_score": 0.0 if i % 4 == 0 else 1.0, "is_residual": bool(i % 4 == 0)} for i in range(40)
        ])
        return mining, scored

    def test_plain_mode_samples_every_status_and_annotates_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mining, scored = self._pool(root)
            manifest = build_coverage_candidates.build(mining, scored, root / "out", mode="plain", per_cell=5, fallback_per_cell=3, seed=17)
            rows = [json.loads(l) for l in (root / "out/coverage_candidates_plain.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 10)
            self.assertTrue(all(r["task_type"] == "Defect Detection" for r in rows))
            self.assertTrue(all(r[coverage_rows.POOL_STATUS_KEY] in {"correct", "residual", "unscored"} for r in rows))
            self.assertEqual(manifest["pool_rows_seen_six_tasks"], 44)
            self.assertEqual(manifest["pool_cells_by_status"]["Defect Detection|ds0|residual"], 10)
            self.assertEqual(manifest["pool_cells_by_status"]["Defect Detection|ds1|unscored"], 4)
            self.assertEqual(manifest["rows_by_task_dataset_bucket"]["Defect Detection"]["ds0"], {"primary": 5})
            self.assertEqual(manifest["output"]["sha256"], anchor_rows.sha256_file(root / "out/coverage_candidates_plain.jsonl"))

    def test_residual_mode_keeps_residual_rows_plus_a_correct_fallback_reservoir(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mining, scored = self._pool(root)
            manifest = build_coverage_candidates.build(mining, scored, root / "out", mode="residual", per_cell=5, fallback_per_cell=3, seed=17)
            rows = [json.loads(l) for l in (root / "out/coverage_candidates_residual.jsonl").read_text().splitlines()]
            statuses = Counter((r["dataset"], r[coverage_rows.POOL_STATUS_KEY]) for r in rows)
            self.assertEqual(statuses, {("ds0", "residual"): 5, ("ds0", "correct"): 3, ("ds1", "correct"): 3})
            self.assertEqual(manifest["rows_by_task_dataset_bucket"]["Defect Detection"]["ds0"], {"fallback": 3, "primary": 5})
            self.assertEqual(manifest["rows_by_task_dataset_bucket"]["Defect Detection"]["ds1"], {"fallback": 3})


class RunnerAndInitCoverageWiringTests(unittest.TestCase):
    def test_runner_moves_coverage_options_to_the_assembler(self) -> None:
        selector, assembler = render_iteration_mining_runner._partition_materialization_arguments(
            ["python", "select.py", "--top-k", "50", "--coverage-blend-share", "0.05", "--coverage-blend-mode", "plain",
             "--coverage-blend-source", "/c.jsonl", "--coverage-blend-min-rows-per-dataset", "8", "--coverage-blend-seed", "17"]
        )
        self.assertEqual(selector, ["python", "select.py", "--top-k", "50", "--no-repetition-blend"])
        self.assertEqual(assembler, ["--coverage-blend-share", "0.05", "--coverage-blend-mode", "plain",
                                     "--coverage-blend-source", "/c.jsonl", "--coverage-blend-min-rows-per-dataset", "8",
                                     "--coverage-blend-seed", "17"])
        with self.assertRaisesRegex(ValueError, "unsupported materialization option"):
            render_iteration_mining_runner._partition_materialization_arguments(["x", "--coverage-blend-bogus", "1"])

    def test_init_records_the_coverage_config_and_requires_mode_and_source(self) -> None:
        from test_cosmos3_init_state_contract import Cosmos3InitStateContractTests as Base
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = Base._workspace(root)
            source = _write(workspace / "coverage_candidates_residual.jsonl", [_row("c-1", "Defect Detection", "pool", "residual")])
            rc = init_deft_state.main(Base._argv(
                root, workspace, "--coverage-blend-share", "0.05", "--coverage-blend-mode", "residual",
                "--coverage-blend-source", str(source),
            ))
            self.assertEqual(rc, 0)
            cfg = json.loads((root / "results/deft_state.json").read_text())["config"]["mining"]["coverage_blend"]
            self.assertTrue(cfg["enabled"])
            self.assertEqual((cfg["share"], cfg["mode"], cfg["min_rows_per_dataset"]), (0.05, "residual", 8))
            self.assertEqual(cfg["source_sha256"], anchor_rows.sha256_file(source))
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = init_deft_state.main(Base._argv(root / "b", workspace, "--coverage-blend-share", "0.05", "--coverage-blend-source", str(source)))
            self.assertNotEqual(rc, 0)
            self.assertIn("requires --coverage-blend-mode", stderr.getvalue())
            rc = init_deft_state.main(Base._argv(root / "c", workspace))
            self.assertEqual(rc, 0)
            self.assertFalse(json.loads((root / "c/results/deft_state.json").read_text())["config"]["mining"]["coverage_blend"]["enabled"])


if __name__ == "__main__":
    unittest.main()
