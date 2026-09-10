# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from collections import Counter
import json
import os
import pathlib
import tempfile
import tomllib
import unittest
import subprocess
from unittest.mock import patch
from types import SimpleNamespace

import pyarrow.parquet as pq

from test_target_profile import row, box, write_rows
from analyze_gaps import _load_evaluator
from cfw_jsonl_runtime import evaluation_row_sort_key
from cfw_predictions import normalize_prediction
import build_target_profile as profile
import score_pool_residual as residual

EVALUATOR = pathlib.Path(os.environ.get("DEFT_EXACT_EVALUATOR", str(
    pathlib.Path.home() / "projects/deft/workspace/eval/calculate_f1_metrics.py")))


class PoolResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Deliberately use the recorded evaluator, not a simplified fake parser.
        cls.evaluator = _load_evaluator(EVALUATOR)

    def test_detection_errors_and_fractional_credit(self):
        gt = box(100)
        far = {"bbox_2d": [500, 500, 600, 600], "label": "other"}
        cases = [
            ([], "[]", "correct", 1.0),
            ([gt], json.dumps([gt]), "correct", 1.0),
            ([gt], "[]", "empty_on_positive", 0.0),
            ([], json.dumps([gt]), "boxes_on_empty", 0.0),
            ([gt, far], json.dumps([gt]), "missed_boxes", 0.5),
            ([gt], json.dumps([gt, far]), "extra_boxes", 0.5),
            ([gt], json.dumps([box(100 * 0.5)]), "missed_boxes", 0.0),
            ([gt], '[{"bbox_2d": [0, 0, 50, 100]}]', "misaligned", 0.0),
            ([], "I cannot tell", "parse_failure", 0.0),
        ]
        for boxes, prediction, error, score in cases:
            with self.subTest(error=error, prediction=prediction):
                result = residual.score_row(self.evaluator, row("r", boxes=boxes), {"raw_prediction": prediction})
                self.assertEqual(result.get("error_type"), error)
                self.assertEqual(result["row_score"], score)

    def test_classification_uses_canonical_sets_and_direct_bcq(self):
        source = row("r", task="Defect Classification", answer='["B", "C"]')
        for raw, expected in (("C, B", "correct"), ("B", "wrong_class"), ("unsure", "parse_failure")):
            self.assertEqual(residual.score_row(self.evaluator, source, {"raw_prediction": raw}).get("error_type"), expected)
        source["messages"][0]["content"][-1]["text"] = "Does this image contain any visible defect?"
        source["messages"][-1]["content"] = self.evaluator.DIRECT_BCQ_NEGATIVE
        self.assertEqual(residual.score_row(self.evaluator, source, {"raw_prediction": "No."}).get("error_type"), "correct")

    def test_matching_reuses_authoritative_hungarian_counts_and_ignores_labels(self):
        boxes = [box(100), {"bbox_2d": [20, 0, 120, 100], "label": "x"}]
        predictions = [{"bbox_2d": [10, 0, 110, 100], "label": "unrelated"}, box(100)]
        native_gt, _ = self.evaluator.parse_boxes(json.dumps(boxes))
        native_pred, _ = self.evaluator.parse_boxes(json.dumps(predictions))
        expected = self.evaluator.one_to_one_detection_counts(native_gt, native_pred, 0.5)
        result = residual.score_row(self.evaluator, row("r", boxes=boxes), {"raw_prediction": json.dumps(predictions)})
        self.assertEqual(result.get("matched_boxes"), expected[0])
        self.assertEqual((result["false_positives"], result["false_negatives"]), expected[1:])

    def test_sampling_is_capped_by_task_dataset_seeded_and_message_exact(self):
        records = [row(f"{task}-{family}-{index}", task=task, answer="B" if "Classification" in task else "[]",
                       paths=[f"/data/NVPAW_pair/{family}/{index}.png"])
                   for task in ("Defect Detection", "Component Classification")
                   for family in ("a", "b") for index in range(12)]
        selected, manifest = residual.sample_pool(records, per_group=3, seed=17)
        self.assertEqual(len(selected), 12)
        again, _ = residual.sample_pool(reversed(records), per_group=3, seed=17)
        self.assertEqual(selected, again)
        self.assertEqual(set(Counter((item["task_type"], profile.dataset_family(item)) for item in selected).values()), {3})
        self.assertTrue(all(item in records for item in selected))
        changed, _ = residual.sample_pool(records, per_group=3, seed=18)
        self.assertNotEqual(selected, changed)

    def _prepare(self, root, records):
        source = write_rows(root / "mining.jsonl", records)
        output = root / "analysis"
        status = residual.main(["prepare", "--input", str(source), "--output-dir", str(output),
            "--model-path", str(root / "model"), "--media-root", str(root),
            "--image", "nvcr.io/nvidia/test@sha256:" + "1" * 64, "--per-group", "2000", "--seed", "17"])
        return status, output

    def test_prepare_reuses_existing_runtime_sorted_eight_gpu_plan_without_launching(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            status, output = self._prepare(root, [row(f"r{i}") for i in range(8)])
            self.assertEqual(status, 0)
            plan = json.loads((output / "pool_scoring_plan.json").read_text())
            self.assertIn("cfw_jsonl_runtime.py", " ".join(plan["command"]))
            self.assertIn("torch.distributed.run", plan["command"])
            self.assertEqual(plan["resources"]["gpus"], 8)
            self.assertEqual(plan["resources"]["partition"], "interactive")
            self.assertEqual(plan["resources"]["time"], "03:59:00")
            config = tomllib.loads((output / "evaluate_pool.toml").read_text())
            self.assertEqual(config["evaluation"]["row_order"], "task_length_sorted")
            self.assertEqual(config["generation"]["max_tokens"], 1024)
            self.assertIn("merge_cfw_prediction_shards.py", " ".join(plan["merge_command"]))
            self.assertFalse((output / "predictions.jsonl").exists())

    def test_scoring_cli_writes_typed_parquet_profile_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            status, output = self._prepare(root, [row("ok"), row("miss", boxes=[box(33)])])
            self.assertEqual(status, 0)
            predictions = write_rows(output / "predictions.jsonl", [{"id": "ok", "raw_prediction": "[]"}, {"id": "miss", "raw_prediction": "[]"}])
            self.assertEqual(residual.main(["score", "--sample", str(output / "pool_sample.jsonl"),
                "--predictions", str(predictions), "--checkpoint", str(root / "model"),
                "--evaluator", str(EVALUATOR), "--output-dir", str(output)]), 0)
            table = pq.read_table(output / "pool_residual.parquet")
            required = {"id", "task_type", "dataset", "row_score", "error_type", "gt_boxes", "size_bin", "count_bin", "pair_kind", "checkpoint", "scored_at"}
            self.assertTrue(required.issubset(table.column_names))
            self.assertEqual(str(table.schema.field("row_score").type), "double")
            self.assertEqual(str(table.schema.field("gt_boxes").type), "int64")
            payload = json.loads((output / "residual_profile.json").read_text())
            self.assertEqual(payload["rows"], 1)
            self.assertEqual(payload["scoring"]["rows_scored"], 2)
            self.assertIn("confidence", (output / "RESIDUAL_REPORT.md").read_text())

    def test_prediction_coverage_metadata_and_sample_seals_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            status, output = self._prepare(root, [row("a")])
            self.assertEqual(status, 0)
            sample = output / "pool_sample.jsonl"
            manifest = json.loads((output / "pool_sample_manifest.json").read_text())
            for predictions in ([], [{"id": "unknown", "raw_prediction": "[]"}],
                                [{"id": "a", "raw_prediction": "[]"}] * 2,
                                [{"id": "a", "task_type": "wrong", "raw_prediction": "[]"}]):
                with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                    residual.score_records(self.evaluator, [row("a")], predictions, checkpoint="model", scored_at="now")
            sample.write_text(sample.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "sample.*SHA-256"):
                residual.validate_sample_manifest(sample, manifest, str(root / "model"))

    def test_label_noise_flags_are_review_only_not_invented_confidence(self):
        source = row("r", task="Defect Classification", answer="B")
        plain = residual.score_row(self.evaluator, source, {"raw_prediction": "C"})
        self.assertEqual(plain.get("error_type"), "wrong_class")
        self.assertIsNone(plain["prediction_confidence"])
        self.assertFalse(plain["review_label_noise"])
        high = residual.score_row(self.evaluator, source, {"raw_prediction": "C", "confidence": 0.99})
        self.assertTrue(high["review_label_noise"])
        self.assertIn("review", high["review_reason"])

    def test_systematic_label_pattern_disagreement_is_a_review_hint_not_confidence(self):
        sources = [row(str(i), task="Defect Classification", answer="B") for i in range(20)]
        predictions = [{"id": source["id"], "raw_prediction": "C"} for source in sources]
        scored = residual.score_records(self.evaluator, sources, predictions, checkpoint="model", scored_at="now")
        self.assertEqual(len(scored), 20)
        self.assertTrue(all(item["review_label_noise"] for item in scored))
        self.assertTrue(all(item["prediction_confidence"] is None for item in scored))
        self.assertTrue(all("pattern" in item["review_reason"] for item in scored))

    def _full_request(self, root, records, chunks=16):
        source = write_rows(root / "mining.jsonl", records)
        sqsh = root / "cfw.sqsh"
        sqsh.write_bytes(b"hsqsCPU-fixture")
        output = root / "full"
        args = ["prepare", "--mode", "full", "--input", str(source), "--output-dir", str(output),
                "--model-path", str(root / "model"), "--media-root", str(root),
                "--image", "nvcr.io/nvidia/test@sha256:" + "1" * 64, "--sqsh", str(sqsh),
                "--container-mounts", f"{root}:{root}",
                "--chunks", str(chunks), "--partition", "test_partition"]
        return args, output

    def test_full_chunks_cover_every_id_except_segmentation_and_render_bounded_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            records = [row(str(i)) for i in range(48)] + [row("seg", task="Defect Segmentation")]
            args, output = self._full_request(root, records)
            self.assertEqual(residual.main(args), 0)
            manifest = json.loads((output / "pool_full_manifest.json").read_text())
            self.assertEqual(manifest["rows"], 48)
            self.assertEqual(manifest["skipped_segmentation_rows"], 1)
            self.assertEqual(len(manifest["chunks"]), 16)
            recovered = []
            for chunk in manifest["chunks"]:
                rows = list(profile.read_rows(pathlib.Path(chunk["source"])))
                self.assertEqual(rows, sorted(rows, key=evaluation_row_sort_key))
                recovered.extend(rows)
            self.assertEqual(Counter(item["id"] for item in recovered), Counter(str(i) for i in range(48)))
            self.assertTrue(all(item in records for item in recovered))
            script = (output / "pool_array.sbatch").read_text()
            for directive in ("--array=0-15%8", "--nodes=1", "--gres=gpu:8", "--constraint=h100", "--time=03:50:00", "--partition=test_partition"):
                self.assertIn(directive, script)
            self.assertIn("--no-container-mount-home", script)
            self.assertNotIn("--container-image=nvcr", script)
            subprocess.run(["bash", "-n", str(output / "pool_array.sbatch")], check=True)

    def _finish_chunk(self, output, chunk, *, wrong_id=None):
        sources = list(profile.read_rows(pathlib.Path(chunk["source"])))
        predictions = [normalize_prediction(source, {"raw_prediction": "[]"}) for source in sources]
        if wrong_id:
            predictions[0]["id"] = wrong_id
        write_rows(pathlib.Path(chunk["predictions"]), predictions)
        residual.seal_chunk(output, chunk["index"])

    def test_full_resume_skips_only_verified_complete_chunks_and_rejects_changed_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            args, output = self._full_request(root, [row(str(i)) for i in range(32)], chunks=2)
            self.assertEqual(residual.main(args), 0)
            manifest = json.loads((output / "pool_full_manifest.json").read_text())
            chunk = manifest["chunks"][0]
            self._finish_chunk(output, chunk)
            before = pathlib.Path(chunk["predictions"]).stat().st_mtime_ns
            self.assertEqual(residual.main(args), 0)
            self.assertIn("--array=1%8", (output / "pool_array.sbatch").read_text())
            self.assertEqual(pathlib.Path(chunk["predictions"]).stat().st_mtime_ns, before)
            pathlib.Path(chunk["predictions"]).write_text('{}\n')
            self.assertEqual(residual.main(args), 0)
            self.assertIn("--array=0-1%8", (output / "pool_array.sbatch").read_text())
            changed = [*args, "--seed", "999"]
            self.assertEqual(residual.main(changed), 2)

    def test_full_merge_requires_all_chunks_then_scores_one_id_keyed_parquet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            records = [row(str(i), boxes=[box(33)] if i % 2 else []) for i in range(24)]
            args, output = self._full_request(root, records, chunks=2)
            self.assertEqual(residual.main(args), 0)
            chunks = json.loads((output / "pool_full_manifest.json").read_text())["chunks"]
            merge = ["merge", "--output-dir", str(output), "--evaluator", str(EVALUATOR)]
            self.assertEqual(residual.main(merge), 2)
            self.assertFalse((output / "pool_residual.parquet").exists())
            with self.assertRaisesRegex(ValueError, "coverage"):
                self._finish_chunk(output, chunks[0], wrong_id="alien")
            for chunk in chunks:
                self._finish_chunk(output, chunk)
            self.assertEqual(residual.main(merge), 0)
            rows = pq.read_table(output / "pool_residual.parquet").to_pylist()
            self.assertEqual(Counter(item["id"] for item in rows), Counter(str(i) for i in range(24)))
            self.assertEqual(Counter(item["error_type"] for item in rows), {"correct": 12, "empty_on_positive": 12})
            self.assertEqual(len(list(profile.read_rows(output / "predictions.jsonl"))), 24)
            self.assertEqual(json.loads((output / "residual_profile.json").read_text())["rows"], 12)

    def test_chunk_failure_preserves_exit_and_never_marks_complete_or_runs_merge(self):
        from pool_residual_full import run_chunk
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            args, output = self._full_request(root, [row(str(i)) for i in range(8)], chunks=1)
            self.assertEqual(residual.main(args), 0)
            # Mock only the external GPU-process boundary: the real sealed plan,
            # exit handling and absence of completion evidence are under test.
            with patch("pool_residual_full.subprocess.run", return_value=SimpleNamespace(returncode=17)) as process:
                with self.assertRaises(SystemExit) as error:
                    run_chunk(SimpleNamespace(output_dir=output, chunk=0))
                self.assertEqual(error.exception.code, 17)
                self.assertEqual(process.call_count, 1)
            self.assertFalse((output / "chunks/0000/COMPLETE.json").exists())

    def test_full_merge_label_patterns_are_global_across_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            sources = [row(str(i), task="Defect Classification", answer="B") for i in range(20)]
            args, output = self._full_request(root, sources, chunks=2)
            self.assertEqual(residual.main(args), 0)
            chunks = json.loads((output / "pool_full_manifest.json").read_text())["chunks"]
            for chunk in chunks:
                rows = list(profile.read_rows(pathlib.Path(chunk["source"])))
                write_rows(pathlib.Path(chunk["predictions"]), [normalize_prediction(row, {"raw_prediction": "C"}) for row in rows])
                residual.seal_chunk(output, chunk["index"])
            self.assertEqual(residual.main(["merge", "--output-dir", str(output), "--evaluator", str(EVALUATOR)]), 0)
            scores = pq.read_table(output / "pool_residual.parquet").to_pylist()
            self.assertTrue(all(row["review_label_noise"] for row in scores))
            self.assertTrue(all(row["prediction_confidence"] is None for row in scores))

    def test_component_count_extension_uses_exact_nonnegative_integers(self):
        source = row("count", task="Component Count", answer="34")
        source["messages"][0]["content"][-1]["text"] = "How many components are visible? Answer with only the integer count."
        for raw, expected in (("34", "correct"), ("35", "wrong_class"), ("34.0", "parse_failure"), ("34 components", "parse_failure"), ("9" * 100, "parse_failure")):
            result = residual.score_row(self.evaluator, source, {"raw_prediction": raw})
            self.assertEqual(result["error_type"], expected)
            self.assertEqual(result["gt_count"], 34)
            self.assertEqual(result["gt_boxes"], 0)

    def test_full_count_rows_are_not_dropped_from_merge_or_failure_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            sources = [row("count", task="Component Count", answer="34"), row("empty")]
            args, output = self._full_request(root, sources, chunks=1)
            self.assertEqual(residual.main(args), 0)
            chunk = json.loads((output / "pool_full_manifest.json").read_text())["chunks"][0]
            rows = list(profile.read_rows(pathlib.Path(chunk["source"])))
            write_rows(pathlib.Path(chunk["predictions"]), [normalize_prediction(row, {"raw_prediction": "35" if row["id"] == "count" else "[]"}) for row in rows])
            residual.seal_chunk(output, 0)
            self.assertEqual(residual.main(["merge", "--output-dir", str(output), "--evaluator", str(EVALUATOR)]), 0)
            scores = pq.read_table(output / "pool_residual.parquet").to_pylist()
            self.assertEqual({row["id"] for row in scores}, {"count", "empty"})
            payload = json.loads((output / "residual_profile.json").read_text())
            self.assertEqual(payload["rows"], 1)
            self.assertEqual(payload["tasks"]["Component Count"]["rows"], 1)
            self.assertEqual(payload["scoring"]["residual_rows"], 1)


if __name__ == "__main__":
    unittest.main()
