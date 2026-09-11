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
import build_anchor_candidates  # noqa: E402
import init_deft_state  # noqa: E402
import render_iteration_mining_runner  # noqa: E402


def _row(record_id: str, task_type: str = "Component Classification", dataset: str = "dsA") -> dict:
    return {
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


def _write(path: pathlib.Path, rows: list[dict]) -> pathlib.Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


class AnchorRowsTests(unittest.TestCase):
    def test_target_rows_reach_the_share_and_only_add_the_missing_part(self) -> None:
        total, new = anchor_rows.anchor_target_rows(0.10, 900, 0)
        self.assertEqual((total, new), (100, 100))
        total, new = anchor_rows.anchor_target_rows(0.10, 1800, 100)
        self.assertEqual((total, new), (200, 100))
        self.assertEqual(anchor_rows.anchor_target_rows(0.0, 900, 0), (0, 0))

    def test_config_validation(self) -> None:
        cfg = anchor_rows.validate_anchor_config(None, None, None, None)
        self.assertFalse(cfg["enabled"])
        with self.assertRaises(ValueError):
            anchor_rows.validate_anchor_config(0.1, None, pathlib.Path("x"), None)
        with self.assertRaises(ValueError):
            anchor_rows.validate_anchor_config(1.0, pathlib.Path("a"), pathlib.Path("x"), None)

    def test_selection_follows_task_shares_caps_datasets_and_skips_excluded(self) -> None:
        candidates = []
        for i in range(60):
            candidates.append(_row(f"cc-{i}", "Component Classification", f"ds{i % 2}"))
        for i in range(60):
            candidates.append(_row(f"dd-{i}", "Defect Detection", "onlyds"))
        excluded_ids = {"cc-0", "dd-0"}
        picked, report = anchor_rows.select_anchors(
            candidates, new_rows=30,
            task_shares={"Component Classification": 0.5, "Defect Detection": 0.5},
            source_cap=0.35, seed=17,
            is_excluded=lambda r: "already_in_corpus" if r["id"] in excluded_ids else None,
        )
        by_task = Counter(r["task_type"] for r in picked)
        self.assertEqual(by_task, {"Component Classification": 15, "Defect Detection": 15})
        self.assertEqual(report["skipped"], {"already_in_corpus": 2})
        # two datasets -> the cap relaxes to ceil(15/2)=8 so the quota can be filled
        self.assertEqual(sum(report["per_task_dataset"]["Component Classification"].values()), 15)
        self.assertTrue(all(v <= 8 for v in report["per_task_dataset"]["Component Classification"].values()))
        again, _ = anchor_rows.select_anchors(
            list(reversed(candidates)), new_rows=30,
            task_shares={"Component Classification": 0.5, "Defect Detection": 0.5},
            source_cap=0.35, seed=17, is_excluded=lambda r: None,
        )
        self.assertEqual([r["id"] for r in picked if r["id"] not in excluded_ids][:5],
                         [r["id"] for r in again if r["id"] not in excluded_ids][:5])


class AssemblerAnchorTests(unittest.TestCase):
    def _corpus(self, root: pathlib.Path):
        mined = _write(root / "mined.jsonl", [_row(f"m-{i}", "Defect Detection", "mine") for i in range(90)])
        anchors = [_row(f"a-{i}", "Component Classification", f"ds{i % 3}") for i in range(200)]
        anchors += [_row(f"b-{i}", "Defect Detection", "pool") for i in range(200)]
        anchors.append(_row("m-3", "Defect Detection", "mine"))  # duplicate of a mined row -> must be skipped
        source = _write(root / "anchor_candidates.jsonl", anchors)
        kpi = _write(root / "kpi.jsonl", [_row(f"k-{i}", "Component Classification") for i in range(50)]
                     + [_row(f"kd-{i}", "Defect Detection") for i in range(50)])
        return mined, source, kpi

    def test_off_by_default_is_unchanged_and_summary_says_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined, _, _ = self._corpus(root)
            rows, summary = assemble_training_json.assemble(None, mined, validation_paths=[], media_root=root)
            self.assertEqual(len(rows), 90)
            self.assertFalse(summary["anchor"]["enabled"])
            self.assertEqual(summary["materialized_anchor_records"], 0)

    def test_anchors_reach_the_share_dedup_and_follow_kpi_task_shares(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined, source, kpi = self._corpus(root)
            cfg = anchor_rows.validate_anchor_config(0.10, source, kpi, None)
            rows, summary = assemble_training_json.assemble(
                None, mined, validation_paths=[], media_root=root, anchor_config=cfg, anchor_seed=17
            )
            self.assertEqual(len(rows), 100)
            self.assertEqual(summary["materialized_anchor_records"], 10)
            self.assertAlmostEqual(summary["anchor"]["realized_share_rows"], 0.10, places=6)
            anchors = [r for r, p in zip(rows, summary["provenance"]) if p["source_kind"] == "anchor_correct"]
            self.assertEqual(Counter(r["task_type"] for r in anchors), {"Component Classification": 5, "Defect Detection": 5})
            self.assertNotIn("m-3", {r["id"] for r in anchors})
            self.assertEqual(summary["anchor"]["selection"]["skipped"].get("already_in_corpus"), 1)
            self.assertEqual(summary["provenance"][-1]["source_kind"], "current_mining" if False else summary["provenance"][-1]["source_kind"])

    def test_second_iteration_only_tops_up_anchors_and_keeps_previous_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined, source, kpi = self._corpus(root)
            cfg = anchor_rows.validate_anchor_config(0.10, source, kpi, None)
            rows, summary = assemble_training_json.assemble(None, mined, validation_paths=[], media_root=root, anchor_config=cfg)
            previous = _write(root / "train1.jsonl", rows)
            sha = assemble_training_json.sha256_file(previous)
            mined2 = _write(root / "mined2.jsonl", [_row(f"n-{i}", "Defect Detection", "mine") for i in range(90)])
            rows2, summary2 = assemble_training_json.assemble(
                previous, mined2, previous_sha256=sha, validation_paths=[], media_root=root, anchor_config=cfg
            )
            self.assertEqual(summary2["anchor"]["prior_anchor_rows"], 10)
            self.assertEqual(summary2["materialized_anchor_records"], 10)  # 180 non-anchor rows -> 20 total anchors
            self.assertEqual(len(rows2), 200)
            self.assertEqual(summary2["retained_previous_records"], 100)
            self.assertTrue(summary2["previous_fingerprints_subset"])

    def test_cap_trims_anchors_before_current_rows_and_keeps_batch_multiple(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined, source, kpi = self._corpus(root)
            cfg = anchor_rows.validate_anchor_config(0.10, source, kpi, None)
            rows, summary = assemble_training_json.assemble(
                None, mined, validation_paths=[], media_root=root, anchor_config=cfg, max_rows=96, row_multiple=32
            )
            self.assertEqual(len(rows), 96)
            self.assertEqual(summary["materialized_current_records"], 90)
            self.assertEqual(summary["materialized_anchor_records"], 6)

    def test_anchor_that_is_an_evaluation_target_is_skipped_not_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined, source, kpi = self._corpus(root)
            leak = _write(root / "eval.jsonl", [_row("a-1", "Component Classification", "ds1")])
            cfg = anchor_rows.validate_anchor_config(0.10, source, kpi, None)
            rows, summary = assemble_training_json.assemble(
                None, mined, validation_paths=[leak], media_root=root, anchor_config=cfg
            )
            self.assertNotIn("a-1", {r["id"] for r in rows})
            self.assertEqual(summary["anchor"]["selection"]["skipped"].get("evaluation_target"), 1)

    def test_cli_writes_anchor_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined, source, kpi = self._corpus(root)
            out = root / "train.jsonl"
            rc = assemble_training_json.main([
                "--mined-jsonl", str(mined), "--output", str(out), "--media-root", str(root),
                "--anchor-share", "0.1", "--anchor-source", str(source), "--anchor-task-shares", str(kpi),
            ])
            self.assertEqual(rc, 0)
            manifest = json.loads((root / "anchor_manifest.json").read_text())
            self.assertTrue(manifest["enabled"])
            self.assertEqual(manifest["training_jsonl"]["rows"], 100)
            summary = json.loads((root / "assemble_summary.json").read_text())
            self.assertEqual(summary["materialized_anchor_records"], 10)

    def test_anchor_and_repetition_cannot_be_combined(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined, source, kpi = self._corpus(root)
            cfg = anchor_rows.validate_anchor_config(0.10, source, kpi, None)
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                assemble_training_json.assemble(
                    None, mined, validation_paths=[], media_root=root, anchor_config=cfg,
                    repetition_config={"enabled": True, "row_cap": 96}, max_rows=96,
                )


class BuildAnchorCandidatesTests(unittest.TestCase):
    def test_streams_correct_rows_with_per_cell_cap_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            pool = [_row(f"p-{i}", "Defect Detection", f"ds{i % 2}") for i in range(40)]
            pool += [_row(f"seg-{i}", "Segmentation", "x") for i in range(3)]
            mining = _write(root / "mining.jsonl", pool)
            scored = _write(root / "scored.jsonl", [
                {"id": f"p-{i}", "task_type": "Defect Detection", "dataset": f"ds{i % 2}",
                 "row_score": 1.0 if i % 4 else 0.5, "is_residual": bool(i % 4 == 0)} for i in range(40)
            ])
            manifest = build_anchor_candidates.build(mining, scored, root / "out", per_cell=5, seed=17)
            rows = [json.loads(l) for l in (root / "out/anchor_candidates.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 10)
            self.assertTrue(all(r["task_type"] == "Defect Detection" for r in rows))
            self.assertEqual(manifest["correct_ids"], 30)
            self.assertEqual(manifest["rows_by_task_dataset"]["Defect Detection"], {"ds0": 5, "ds1": 5})
            self.assertEqual(manifest["output"]["sha256"], anchor_rows.sha256_file(root / "out/anchor_candidates.jsonl"))


class RunnerAndInitAnchorWiringTests(unittest.TestCase):
    def test_runner_moves_anchor_options_to_the_assembler(self) -> None:
        selector, assembler = render_iteration_mining_runner._partition_materialization_arguments(
            ["python", "select.py", "--top-k", "50", "--anchor-share", "0.1", "--anchor-source", "/a.jsonl",
             "--anchor-task-shares", "/k.jsonl", "--anchor-seed", "17"]
        )
        self.assertEqual(selector, ["python", "select.py", "--top-k", "50", "--no-repetition-blend"])
        self.assertEqual(assembler, ["--anchor-share", "0.1", "--anchor-source", "/a.jsonl",
                                     "--anchor-task-shares", "/k.jsonl", "--anchor-seed", "17"])
        with self.assertRaisesRegex(ValueError, "unsupported materialization option"):
            render_iteration_mining_runner._partition_materialization_arguments(["x", "--anchor-bogus", "1"])

    def test_init_records_anchor_config_and_defaults_task_shares_to_the_kpi_set(self) -> None:
        from test_cosmos3_init_state_contract import Cosmos3InitStateContractTests as Base
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = Base._workspace(root)
            source = _write(workspace / "anchor_candidates.jsonl", [_row("a-1", "Defect Detection", "pool")])
            rc = init_deft_state.main(Base._argv(root, workspace, "--anchor-share", "0.1", "--anchor-source", str(source)))
            self.assertEqual(rc, 0)
            anchor = json.loads((root / "results/deft_state.json").read_text())["config"]["mining"]["anchor"]
            self.assertTrue(anchor["enabled"])
            self.assertEqual(anchor["share"], 0.1)
            self.assertEqual(anchor["task_shares_source"], str((workspace / "annotations/proxy_kpi.jsonl").resolve()))
            self.assertEqual(anchor["source_sha256"], anchor_rows.sha256_file(source))
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = init_deft_state.main(Base._argv(root / "b", workspace, "--anchor-share", "0.1"))
            self.assertNotEqual(rc, 0)
            self.assertIn("requires --anchor-source", stderr.getvalue())
            rc = init_deft_state.main(Base._argv(root / "c", workspace))
            self.assertEqual(rc, 0)
            self.assertFalse(json.loads((root / "c/results/deft_state.json").read_text())["config"]["mining"]["anchor"]["enabled"])


if __name__ == "__main__":
    unittest.main()
