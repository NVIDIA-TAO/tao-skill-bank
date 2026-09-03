from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest


APP_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "scripts"))

import record_training_capacity_refinement as refinement


class TrainingCapacityRefinementTest(unittest.TestCase):
    def fixture(self, *, gradient_accumulation: int = 1) -> tuple[pathlib.Path, pathlib.Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = pathlib.Path(temporary.name) / "results"
        phase = root / "iter1" / "capacity_refinement_mb5"
        phase.mkdir(parents=True)
        train = phase / "train.jsonl"
        train.write_text("".join(json.dumps({"id": i}) + "\n" for i in range(40)))
        assemble = phase / "assemble_summary.json"
        assemble.write_text(json.dumps({"output_records": 40, "row_multiple": 40}))
        validation = phase / "validation_report.json"
        validation.write_text(json.dumps({"state": "COMPLETE", "materialized_rows": 40, "row_multiple": 40}))
        failure_log = root / "iter1" / "failed.log"
        failure_log.write_text("torch.OutOfMemoryError: CUDA out of memory")
        failure_record = root / "failed_job.json"
        failure_record.write_text(json.dumps({"id": "train-failed", "backend_ref": "11", "terminal_state": "ERROR", "err_class": "ERR_PROGRAM"}))
        probe_status = root / "iter1" / "probe_status.json"
        probe_status.write_text(json.dumps({"state": "COMPLETE", "micro_batch_per_rank": 5, "gradient_accumulation": 1, "effective_global_batch": 40, "learning_rate": 1e-6, "optimizer_updates_observed": 10}))
        probe_record = root / "probe_job.json"
        probe_record.write_text(json.dumps({"id": "train-probe", "backend_ref": "12", "terminal_state": "COMPLETE"}))
        evidence = phase / "capacity_evidence.json"
        evidence.write_text(json.dumps({
            "schema": "training_capacity_refinement_v1",
            "previous_micro_batch_per_rank": 6,
            "refined_micro_batch_per_rank": 5,
            "failed_job_record": str(failure_record),
            "failure_log": str(failure_log),
            "probe_job_record": str(probe_record),
            "probe_status": str(probe_status),
        }))
        state = {
            "version": 7,
            "status": "in_progress",
            "current_iteration": 1,
            "max_iterations": 5,
            "config": {"training": {
                "num_gpus": 8,
                "micro_batch_per_rank": 6,
                "gradient_accumulation": gradient_accumulation,
                "global_batch": 48,
                "learning_rate_scaling": "fixed",
                "optimizer": {"learning_rate": 1e-6},
            }},
            "iterations": {"iter1": {"status": "in_progress", "stage_completed": "validate_data"}},
            "events": [{"seq": 1, "iter": "iter1", "stage": "validate_data", "status": "ok"}],
        }
        state_path = root / "deft_state.json"
        state_path.write_text(json.dumps(state))
        return state_path, evidence

    def test_records_probe_backed_refinement_without_changing_ga_or_lr(self) -> None:
        state_path, evidence = self.fixture()
        updated = refinement.apply(state_path, "iter1", evidence)
        training = updated["config"]["training"]
        self.assertEqual(training["micro_batch_per_rank"], 5)
        self.assertEqual(training["global_batch"], 40)
        self.assertEqual(training["gradient_accumulation"], 1)
        self.assertEqual(training["optimizer"]["learning_rate"], 1e-6)
        self.assertEqual(updated["events"][-1]["stage"], "training_capacity_refinement")
        self.assertTrue(updated["iterations"]["iter1"]["combined_training_jsonl"].endswith("train.jsonl"))

    def test_rejects_refinement_if_gradient_accumulation_is_not_one(self) -> None:
        state_path, evidence = self.fixture(gradient_accumulation=2)
        with self.assertRaisesRegex(ValueError, "gradient_accumulation=1"):
            refinement.apply(state_path, "iter1", evidence)


if __name__ == "__main__":
    unittest.main()
