# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import filter_mined_history  # noqa: E402


def write_candidates(path: pathlib.Path, values: list[str]) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "filepath": values,
                "max_cosine_similarity": [0.99 - index * 0.01 for index in range(len(values))],
            }
        ),
        path,
    )
    return path


class HistoryAwareMiningTests(unittest.TestCase):
    def select(
        self,
        root: pathlib.Path,
        iteration: int,
        values: list[str],
        *,
        topn: int = 5,
        resume: bool = False,
        pool_size: int | None = None,
        max_cumulative_fraction: float | None = None,
    ) -> tuple[dict, pathlib.Path]:
        candidate = write_candidates(root / f"iter{iteration}/candidates.parquet", values)
        output = root / f"iter{iteration}/mined.parquet"
        summary = root / f"iter{iteration}/history_summary.json"
        payload = filter_mined_history.select_novel_samples(
            candidate_parquet=candidate,
            output_parquet=output,
            history_file=root / "mining_history.json",
            summary_file=summary,
            iteration=iteration,
            topn=topn,
            resume=resume,
            pool_size=pool_size,
            max_cumulative_fraction=max_cumulative_fraction,
        )
        return payload, output

    def test_filters_duplicates_across_iterations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            first, first_output = self.select(root, 1, ["a.png", "b.png"])
            second, second_output = self.select(
                root, 2, ["b.png", "c.png", "c.png"]
            )

            self.assertEqual(first["selected_count"], 2)
            self.assertEqual(
                pq.read_table(first_output)["filepath"].to_pylist(),
                ["a.png", "b.png"],
            )
            self.assertEqual(second["candidate_duplicate_count"], 1)
            self.assertEqual(second["already_mined_count"], 1)
            self.assertEqual(second["selected_count"], 1)
            self.assertEqual(pq.read_table(second_output)["filepath"].to_pylist(), ["c.png"])

            history = json.loads((root / "mining_history.json").read_text())
            self.assertEqual(history["cumulative_unique_count"], 3)
            self.assertEqual(
                [entry["selected_filepaths"] for entry in history["iterations"]],
                [["a.png", "b.png"], ["c.png"]],
            )

    def test_normalizes_path_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            self.select(root, 1, ["images\\board\\a.png"])
            summary, output = self.select(root, 2, ["images/board/./a.png", "new.png"])

            self.assertEqual(summary["already_mined_count"], 1)
            self.assertEqual(pq.read_table(output)["filepath"].to_pylist(), ["new.png"])

    def test_atomic_sample_id_prevents_pair_sides_from_becoming_history_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            candidate = root / "iter1/candidates.parquet"
            candidate.parent.mkdir(parents=True)
            pq.write_table(
                pa.table(
                    {
                        "filepath": ["pair-a.png", "pair-b.png"],
                        "atomic_sample_id": ["reference_pair:a", "reference_pair:b"],
                        "target_filepath": ["shared-test.png", "shared-test.png"],
                    }
                ),
                candidate,
            )

            payload = filter_mined_history.select_novel_samples(
                candidate_parquet=candidate,
                output_parquet=root / "iter1/mined.parquet",
                history_file=root / "mining_history.json",
                summary_file=root / "iter1/summary.json",
                iteration=1,
                topn=2,
            )

            history = json.loads((root / "mining_history.json").read_text())
            self.assertEqual(payload["identity"], "atomic_sample_id")
            self.assertEqual(payload["selected_count"], 2)
            self.assertEqual(
                history["iterations"][0]["selected_identities"],
                ["reference_pair:a", "reference_pair:b"],
            )

    def test_ledger_declares_mining_budget_scope_not_training_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            summary, _ = self.select(root, 1, ["a.png"])
            history = json.loads((root / "mining_history.json").read_text())

            self.assertEqual(summary["purpose"], "mining_budget_only")
            self.assertEqual(history["purpose"], "mining_budget_only")
            self.assertNotIn("retained_previous_records", history)

    def test_empty_novel_output_is_auditable_and_recommends_wider_topn(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            self.select(root, 1, ["a.png"])
            summary, output = self.select(root, 2, ["a.png"])

            self.assertEqual(summary["selected_count"], 0)
            self.assertEqual(pq.read_table(output).num_rows, 0)
            self.assertIn("increase topn", summary["recommendation"])

    def test_iteration_two_requires_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            with self.assertRaises(FileNotFoundError):
                self.select(root, 2, ["a.png"])

    def test_resume_verifies_committed_artifacts_and_topn(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            expected, output = self.select(root, 1, ["a.png"])
            resumed, resumed_output = self.select(
                root, 1, ["a.png"], resume=True
            )
            self.assertEqual(resumed, expected)
            self.assertEqual(resumed_output, output)

            with self.assertRaises(ValueError):
                self.select(root, 1, ["a.png"], topn=50, resume=True)

    def test_tampered_prior_output_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _, output = self.select(root, 1, ["a.png"])
            write_candidates(output, ["tampered.png"])

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                self.select(root, 2, ["b.png"])

    def test_resume_rejects_changed_candidate_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            self.select(root, 1, ["a.png"])

            with self.assertRaisesRegex(ValueError, "candidate_parquet hash mismatch"):
                self.select(root, 1, ["different.png"], resume=True)

    def test_cumulative_selection_is_hard_capped_by_pool_fraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            first, first_output = self.select(
                root,
                1,
                ["a.png", "b.png", "c.png", "d.png"],
                pool_size=6,
                max_cumulative_fraction=0.5,
            )
            second, second_output = self.select(
                root,
                2,
                ["e.png", "f.png"],
                pool_size=6,
                max_cumulative_fraction=0.5,
            )

            self.assertEqual(first["cumulative_budget_cap"], 3)
            self.assertEqual(first["selected_count"], 3)
            self.assertEqual(
                pq.read_table(first_output)["filepath"].to_pylist(),
                ["a.png", "b.png", "c.png"],
            )
            self.assertEqual(first["budget_excluded_count"], 1)
            self.assertEqual(second["selected_count"], 0)
            self.assertEqual(pq.read_table(second_output).num_rows, 0)


if __name__ == "__main__":
    unittest.main()
