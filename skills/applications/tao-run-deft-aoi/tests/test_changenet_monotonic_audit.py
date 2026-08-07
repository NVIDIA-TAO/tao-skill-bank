# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
import pathlib
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import audit_deft_run  # noqa: E402

DATA_SKILL_SCRIPTS = (
    SKILL_ROOT.parents[1]
    / "data"
    / "tao-mine-aoi-images"
    / "scripts"
)
sys.path.insert(0, str(DATA_SKILL_SCRIPTS))
import filter_mined_history  # noqa: E402


TRAIN_COLUMNS = [
    "input_path",
    "golden_path",
    "label",
    "object_name",
    "boardname",
]


def train_row(name: str, label: str = "PASS") -> dict[str, str]:
    return {
        "input_path": f"images/{name}",
        "golden_path": f"golden/{name}",
        "label": label,
        "object_name": name,
        "boardname": "board",
    }


def write_csv(
    path: pathlib.Path,
    rows: list[dict[str, str]],
    columns: list[str],
) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_provenance(path: pathlib.Path, sources: list[str]) -> pathlib.Path:
    return write_csv(path, [{"source": source} for source in sources], ["source"])


class ChangeNetMonotonicAuditTests(unittest.TestCase):
    def test_iter2_accepts_exact_previous_train_plus_new_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            previous_rows = [train_row("base"), train_row("synthetic_iter1", "ng")]
            current_rows = [*previous_rows, train_row("mined_iter2")]
            previous = write_csv(root / "iter1.csv", previous_rows, TRAIN_COLUMNS)
            current = write_csv(root / "iter2.csv", current_rows, TRAIN_COLUMNS)
            provenance = write_provenance(
                root / "iter2_provenance.csv",
                ["previous_iter_train", "previous_iter_train", "mining_pool"],
            )
            errors: list[str] = []

            audit_deft_run._training_merge_proof(
                "iter2",
                {
                    "combined_training_csv": str(current),
                    "provenance_csv": str(provenance),
                },
                {"iter1": {"combined_training_csv": str(previous)}},
                errors,
            )

            self.assertEqual(errors, [])

    def test_iter2_rejects_dropped_historical_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            previous_rows = [train_row("base"), train_row("synthetic_iter1", "ng")]
            current_rows = [train_row("base"), train_row("mined_iter2")]
            previous = write_csv(root / "iter1.csv", previous_rows, TRAIN_COLUMNS)
            current = write_csv(root / "iter2.csv", current_rows, TRAIN_COLUMNS)
            provenance = write_provenance(
                root / "iter2_provenance.csv",
                ["previous_iter_train", "mining_pool"],
            )
            errors: list[str] = []

            audit_deft_run._training_merge_proof(
                "iter2",
                {
                    "combined_training_csv": str(current),
                    "provenance_csv": str(provenance),
                },
                {"iter1": {"combined_training_csv": str(previous)}},
                errors,
            )

            self.assertTrue(
                any("must retain the exact iter1" in error for error in errors)
            )
            self.assertTrue(any("missing=1" in error for error in errors))

    def test_iter2_rejects_non_monotonic_provenance_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            previous_rows = [train_row("base")]
            current_rows = [*previous_rows, train_row("mined_iter2")]
            previous = write_csv(root / "iter1.csv", previous_rows, TRAIN_COLUMNS)
            current = write_csv(root / "iter2.csv", current_rows, TRAIN_COLUMNS)
            provenance = write_provenance(
                root / "iter2_provenance.csv",
                ["base_train", "mining_pool"],
            )
            errors: list[str] = []

            audit_deft_run._training_merge_proof(
                "iter2",
                {
                    "combined_training_csv": str(current),
                    "provenance_csv": str(provenance),
                },
                {"iter1": {"combined_training_csv": str(previous)}},
                errors,
            )

            self.assertTrue(
                any("invalid values for iter2" in error for error in errors)
            )
            self.assertTrue(
                any("must include previous_iter_train" in error for error in errors)
            )

    def test_provenance_must_align_one_to_one_with_train_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            current = write_csv(
                root / "iter1.csv",
                [train_row("base"), train_row("mined_iter1")],
                TRAIN_COLUMNS,
            )
            provenance = write_provenance(
                root / "iter1_provenance.csv", ["base_train"]
            )
            errors: list[str] = []

            audit_deft_run._training_merge_proof(
                "iter1",
                {
                    "combined_training_csv": str(current),
                    "provenance_csv": str(provenance),
                },
                {},
                errors,
            )

            self.assertTrue(
                any("one row per combined Train row" in error for error in errors)
            )

    def test_history_audit_accepts_disjoint_iteration_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            history = root / "mining_history.json"
            state = {
                "config": {
                    "mining_filter": {
                        "top_k_per_target": 25,
                        "history_aware": {
                            "enabled": True,
                            "history_file": str(history),
                        },
                    }
                }
            }
            phases = []
            for iteration, values in (
                (1, ["a.png", "b.png"]),
                (2, ["b.png", "c.png"]),
            ):
                directory = root / f"iter{iteration}"
                directory.mkdir()
                candidates = directory / "mined_candidates.parquet"
                output = directory / "mined.parquet"
                summary = directory / "mining_history_summary.json"
                pq.write_table(pa.table({"filepath": values}), candidates)
                result = filter_mined_history.select_novel_samples(
                    candidate_parquet=candidates,
                    output_parquet=output,
                    history_file=history,
                    summary_file=summary,
                    iteration=iteration,
                    topn=25,
                )
                phases.append(
                    {
                        "mining_history": str(history),
                        "mining_history_summary": str(summary),
                        "mining_candidate_parquet": str(candidates),
                        "mining_mined_parquet": str(output),
                        "mining_mined_count": result["selected_count"],
                    }
                )

            errors: list[str] = []
            audit_deft_run._mining_history_proof(
                "iter2", phases[1], state, 2, 1, errors
            )

            self.assertEqual(errors, [])

    def test_history_audit_rejects_cross_iteration_reselection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            history = root / "mining_history.json"
            candidates = root / "iter1/mined_candidates.parquet"
            output = root / "iter1/mined.parquet"
            summary = root / "iter1/mining_history_summary.json"
            candidates.parent.mkdir()
            pq.write_table(pa.table({"filepath": ["a.png"]}), candidates)
            result = filter_mined_history.select_novel_samples(
                candidate_parquet=candidates,
                output_parquet=output,
                history_file=history,
                summary_file=summary,
                iteration=1,
                topn=25,
            )
            ledger = json.loads(history.read_text())
            duplicate = dict(ledger["iterations"][0])
            duplicate["iteration"] = 2
            duplicate["selected_filepaths"] = ["a.png"]
            ledger["iterations"].append(duplicate)
            history.write_text(json.dumps(ledger))
            state = {
                "config": {
                    "mining_filter": {
                        "top_k_per_target": 25,
                        "history_aware": {
                            "enabled": True,
                            "history_file": str(history),
                        },
                    }
                }
            }
            errors: list[str] = []

            audit_deft_run._mining_history_proof(
                "iter1",
                {
                    "mining_history": str(history),
                    "mining_history_summary": str(summary),
                    "mining_candidate_parquet": str(candidates),
                    "mining_mined_parquet": str(output),
                    "mining_mined_count": result["selected_count"],
                },
                state,
                1,
                1,
                errors,
            )

            self.assertTrue(any("reselects 1 prior" in error for error in errors))

    def test_history_audit_rejects_ledger_output_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            history = root / "mining_history.json"
            candidates = root / "iter1/mined_candidates.parquet"
            output = root / "iter1/mined.parquet"
            summary = root / "iter1/mining_history_summary.json"
            candidates.parent.mkdir()
            pq.write_table(pa.table({"filepath": ["a.png"]}), candidates)
            result = filter_mined_history.select_novel_samples(
                candidate_parquet=candidates,
                output_parquet=output,
                history_file=history,
                summary_file=summary,
                iteration=1,
                topn=25,
            )
            ledger = json.loads(history.read_text())
            ledger["iterations"][0]["selected_filepaths"] = ["other.png"]
            history.write_text(json.dumps(ledger))
            state = {
                "config": {
                    "mining_filter": {
                        "top_k_per_target": 25,
                        "history_aware": {
                            "enabled": True,
                            "history_file": str(history),
                        },
                    }
                }
            }
            errors: list[str] = []

            audit_deft_run._mining_history_proof(
                "iter1",
                {
                    "mining_history": str(history),
                    "mining_history_summary": str(summary),
                    "mining_candidate_parquet": str(candidates),
                    "mining_mined_parquet": str(output),
                    "mining_mined_count": result["selected_count"],
                },
                state,
                1,
                1,
                errors,
            )

            self.assertTrue(
                any("disagree with the final mined parquet" in error for error in errors)
            )


if __name__ == "__main__":
    unittest.main()
