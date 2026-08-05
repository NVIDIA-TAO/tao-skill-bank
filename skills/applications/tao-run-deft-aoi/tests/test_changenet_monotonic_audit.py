# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import pathlib
import sys
import tempfile
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import audit_deft_run  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
