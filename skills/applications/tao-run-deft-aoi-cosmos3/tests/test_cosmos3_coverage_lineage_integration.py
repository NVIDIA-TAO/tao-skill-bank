# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from test_cosmos3_coverage_stratified_selector import _pool, _proxy
from test_cosmos3_defect_detection_ablation_contract import _row

import assemble_training_json
import atomic_samples
import commit_stage
import defect_detection_ablation
import render_iteration_mining_runner as runner
import task_mining_router


SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
SELECTOR = "coverage_stratified_hardness_v1"
DD = "Defect Detection"
TASKS = (DD, *defect_detection_ablation.MAINTENANCE_TASK_TYPES)
BOX = {"bbox_2d": [10, 10, 100, 100], "label": "bridge"}


def _write(path: pathlib.Path, rows: list[dict]) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _supply(root: pathlib.Path, iteration: int) -> tuple[list[dict], list[dict]]:
    inventory, records = [], []
    for task in TASKS:
        for original in _pool():
            key = f"round{iteration}-{task}-{original['source_group_id']}"
            record = _row(
                key, task, dataset=original["source_dataset"],
                boxes=[{**BOX, "label": original["canonical_phenotype"]}] * original["gt_count"],
            )
            if "Classification" in task:
                record["messages"][-1]["content"][0]["text"] = "A"
            sample = atomic_samples.sample_from_record(record, media_root=root, context=key)
            inventory.append({
                **original,
                "task_type": task,
                "canonical_phenotype": "Resistors" if "Classification" in task else original["canonical_phenotype"],
                "source_group_id": key,
                "visual_cluster": key if task == "Component Detection" else original["visual_cluster"],
                "parent_record_id": key,
                "atomic_sample_id": sample["atomic_sample_id"],
                "sample_kind": sample["sample_kind"],
                "filepath": sample["target_filepath"],
                "source_image_paths": sample["image_paths"],
                "source_record_ids": [key],
                "source_task_types": [task],
                # Metadata-only CPU fixture, like the DD contract suite: no
                # encoder or rendering is needed to test the assembly seam.
                "content_sha256": hashlib.sha256(key.encode()).hexdigest(),
                "local_contrast": 0.5,
            })
            records.append(record)
    return inventory, records


class Cosmos3CoverageLineageIntegrationTests(unittest.TestCase):
    def test_coverage_five_round_runner_retains_lineage_without_repetition(self) -> None:
        self._five_rounds(repetition=False)

    def test_coverage_five_round_runner_retains_lineage_with_repetition(self) -> None:
        self._five_rounds(repetition=True)

    def _five_rounds(self, *, repetition: bool) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            proxy = _write(root / "proxy.jsonl", [
                _row(f"proxy-{task}-{index}", task, boxes=[] if index < 2 else [BOX])
                for task in (DD, "Ref_based Defect Detection") for index in range(5)
            ])
            for record in defect_detection_ablation.load_records(proxy):
                for part in record["messages"][0]["content"]:
                    if part.get("type") == "image":
                        path = root / part["image"]
                        path.parent.mkdir(parents=True, exist_ok=True)
                        Image.new("RGB", (4, 4), color="black").save(path)
            proxy_errors = _proxy()
            for task in TASKS[1:]:
                if "Classification" in task:
                    proxy_errors.append({
                        "task_type": task, "canonical_phenotype": "Resistors",
                        "false_negative_count": 15 / 175,
                    })
                else:
                    proxy_errors.extend({
                        **row, "task_type": task,
                        "false_negative_count": row["false_negative_count"] / 25,
                        "false_positive_count": row["false_positive_count"] / 25,
                    } for row in _proxy())
            state = {"config": {
                "mining": {"candidate_selector": SELECTOR, "defect_detection_ablation": {"enabled": True}},
                "training": {"epochs_per_iteration": 5, "global_batch": 4},
            }, "iterations": {}}
            previous, occurrences, sizes = None, Counter(), []
            for iteration in range(1, 6):
                phase = root / f"iter{iteration}"
                phase.mkdir()
                inventory, records = _supply(root, iteration)
                selected, coverage = task_mining_router.select_coverage_stratified_candidates(
                    inventory, proxy_errors, budget=40, round_index=iteration,
                    epochs=5, iteration_budget=40,
                )
                task_mining_router.require_coverage_training_eligible(coverage)
                self.assertTrue(all(row["candidate_selector"] == SELECTOR for row in selected))
                candidate_path = phase / "candidates.parquet"
                pq.write_table(pa.Table.from_pylist(selected), candidate_path)
                source_path = _write(phase / "source.jsonl", records)
                selector_command = [
                    sys.executable, str(SCRIPTS / "defect_detection_ablation.py"),
                    "--candidate-parquet", str(candidate_path),
                    "--source-annotations", str(source_path),
                    "--proxy-annotations", str(proxy), "--media-root", str(root),
                    "--max-rows", "40", "--minimum-rows", "32",
                    "--row-multiple", "4", "--epochs", "5", "--global-batch", "4",
                    "--no-near-duplicate-filter",
                    "--repetition-blend" if repetition else "--no-repetition-blend",
                ]
                plan = runner.build_plan(
                    selector_command=selector_command, previous_jsonl=previous,
                    previous_sha256=assemble_training_json.sha256_file(previous) if previous else None,
                    mined_jsonl=phase / "data/mined.jsonl",
                    current_quota_manifest=phase / "data/quota.json",
                    train_jsonl=phase / "assemble_data/train.jsonl",
                    assemble_summary=phase / "assemble_data/summary.json",
                    final_quota_manifest=phase / "assemble_data/quota.json",
                    media_root=root, max_rows=400, row_multiple=4, epochs=5, global_batch=4,
                )
                self.assertNotEqual(plan["selector"]["output"], plan["assembler"]["output"])
                self.assertEqual(plan["selector"]["output"], plan["assembler"]["mined_input"])
                self.assertNotIn("--repetition-blend", plan["selector"]["command"])
                executable = phase / "runner.py"
                executable.write_text(runner.render_runner(plan))
                result = subprocess.run(
                    [sys.executable, str(executable)], capture_output=True,
                    text=True, check=False, timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{"event"')]
                self.assertEqual([event["stage"] for event in events if event["event"] == "stage_start"], ["selector", "assembler"])
                train = pathlib.Path(plan["assembler"]["output"])
                rows = defect_detection_ablation.load_records(train)
                current = Counter(json.dumps(row, sort_keys=True) for row in rows)
                self.assertTrue(occurrences <= current)
                self.assertGreater(len(rows), sum(occurrences.values()))
                self.assertEqual(len(rows) % 4, 0)
                args = argparse.Namespace(
                    iter_label=f"iter{iteration}", stage="assemble_data",
                    mined_jsonl=pathlib.Path(plan["selector"]["output"]),
                    combined_training_jsonl=train,
                    assemble_summary=pathlib.Path(plan["assembler"]["summary"]),
                    quota_manifest=pathlib.Path(plan["assembler"]["quota_manifest"]),
                )
                phase_state = state["iterations"].setdefault(args.iter_label, {})
                commit_stage._apply_success(phase_state, args, root, state["iterations"], state)
                quota = json.loads(args.quota_manifest.read_text())
                self.assertEqual(quota["schema_version"], "defect_detection_quota_manifest_v2")
                self.assertFalse(quota["current_selection"]["repetition_blend"]["enabled"])
                self.assertEqual(quota["repetition_blend"]["enabled"], repetition)
                self.assertIsNone(quota["reference_calibration"]["requested_total"])
                if repetition:
                    self.assertEqual(set(quota["repetition_blend"]["tasks"]), set(TASKS))
                    self.assertEqual(quota["repetition_blend"]["totals"]["rows_after"], len(rows))
                occurrences, previous = current, train
                sizes.append(len(rows))

            # A passing coverage manifest must not waive assembly lineage.
            summary = json.loads(args.assemble_summary.read_text())
            summary["retained_previous_records"] = 0
            args.assemble_summary.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "retain every previous record"):
                commit_stage._apply_success(phase_state, args, root, state["iterations"], state)
            with self.assertRaisesRegex(ValueError, "retain every previous record"):
                commit_stage._apply_success(
                    {}, argparse.Namespace(iter_label="iter6", stage="data_mining"),
                    root, state["iterations"], state,
                )
            self.assertEqual(sizes, sorted(set(sizes)))


if __name__ == "__main__":
    unittest.main()
