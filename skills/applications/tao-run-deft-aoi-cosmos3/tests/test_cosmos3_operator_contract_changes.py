# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import hashlib
import pathlib
import sys
import tempfile
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import validate_split_contract  # noqa: E402
import apply_operator_contract_change  # noqa: E402


def _row(record_id: str, image: str) -> dict:
    return {
        "id": record_id,
        "task_type": "Component Classification",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image, "min_pixels": 1, "max_pixels": 1},
                    {"type": "text", "text": "classify"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "A"}]},
        ],
    }


def _write(path: pathlib.Path, rows: list[dict]) -> pathlib.Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


class AuthorizedSplitOverlapTests(unittest.TestCase):
    def test_exact_operator_authorized_proxy_benchmark_overlap_is_audited(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            roles = {
                "proxy": _write(root / "proxy.jsonl", [_row("proxy-1", "images/shared.png")]),
                "benchmark": _write(
                    root / "benchmark.jsonl", [_row("benchmark-1", "images/shared.png")]
                ),
                "mining": _write(root / "mining.jsonl", [_row("mining-1", "images/mining.png")]),
            }

            summary = validate_split_contract.validate(
                roles,
                media_root=root,
                allowed_target_overlaps={"proxy:benchmark": 1},
            )

            self.assertEqual(summary["target_overlap"]["proxy:benchmark"], 1)
            self.assertEqual(
                summary["authorized_target_overlap_exceptions"],
                {"proxy:benchmark": {"allowed": 1, "observed": 1}},
            )

    def test_authorized_overlap_is_exact_not_an_upper_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            roles = {
                "proxy": _write(root / "proxy.jsonl", [_row("proxy-1", "images/shared.png")]),
                "benchmark": _write(
                    root / "benchmark.jsonl", [_row("benchmark-1", "images/shared.png")]
                ),
                "mining": _write(root / "mining.jsonl", [_row("mining-1", "images/mining.png")]),
            }

            with self.assertRaisesRegex(ValueError, "authorized target overlap mismatch"):
                validate_split_contract.validate(
                    roles,
                    media_root=root,
                    allowed_target_overlaps={"proxy:benchmark": 2},
                )


class OperatorBenchmarkReplacementTests(unittest.TestCase):
    @staticmethod
    def _sha256(path: pathlib.Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _fixture(self, root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, list[dict]]:
        active = root / "benchmark.jsonl"
        retired = root / "benchmark_retired.jsonl"
        provenance = root / "benchmark_provenance.jsonl"
        proxy = root / "proxy.jsonl"
        active.write_text('{"id":"new-1"}\n{"id":"new-2"}\n', encoding="utf-8")
        provenance.write_bytes(active.read_bytes())
        retired.write_text('{"id":"old-1"}\n', encoding="utf-8")
        proxy.write_text('{"id":"proxy-1"}\n', encoding="utf-8")
        state = {
            "version": 7,
            "status": "in_progress",
            "config": {
                "annotations": {"benchmark": str(active), "proxy": str(proxy)},
                "annotation_sha256": {
                    "benchmark": self._sha256(retired),
                    "proxy": self._sha256(proxy),
                },
                "evaluation": {
                    "benchmark": {"annotations": str(active), "sha256": self._sha256(retired)}
                },
            },
            "events": [{"seq": 1, "stage": "validate_data"}],
        }
        state_path = root / "deft_state.json"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        exception = {
            "schema": "deft_operator_contract_exception_v1",
            "authorized_by": "Sean",
            "authorization_date": "2026-09-01",
            "scope": "test run only",
            "benchmark_replacement": {
                "active_path": str(active),
                "active_rows": 2,
                "active_sha256": self._sha256(active),
                "provenance_path": str(provenance),
                "retired_path": str(retired),
                "retired_rows": 1,
                "retired_sha256": self._sha256(retired),
                "effective_iteration": 2,
            },
            "proxy": {"path": str(proxy), "rows": 1, "sha256": self._sha256(proxy)},
            "authorized_overlap_exception": {
                "physical_target_overlap": 1,
                "benchmark_rows_on_overlapping_targets": 1,
                "proxy_rows_on_overlapping_targets": 1,
                "cohort_mutation_allowed": False,
            },
        }
        exception_path = root / "exception.json"
        exception_path.write_text(json.dumps(exception), encoding="utf-8")
        return state_path, exception_path, state["events"]

    def test_benchmark_replacement_updates_only_current_seals_and_adds_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_path, exception_path, original_events = self._fixture(pathlib.Path(temporary))

            updated = apply_operator_contract_change.apply(state_path, exception_path)

            replacement = json.loads(exception_path.read_text())["benchmark_replacement"]
            self.assertEqual(
                updated["config"]["annotation_sha256"]["benchmark"],
                replacement["active_sha256"],
            )
            self.assertEqual(
                updated["config"]["evaluation"]["benchmark"]["sha256"],
                replacement["active_sha256"],
            )
            self.assertEqual(updated["events"], original_events)
            self.assertEqual(len(updated["operator_contract_changes"]), 1)
            self.assertEqual(
                updated["operator_contract_changes"][0]["previous_benchmark_sha256"],
                replacement["retired_sha256"],
            )

            repeated = apply_operator_contract_change.apply(state_path, exception_path)
            self.assertEqual(len(repeated["operator_contract_changes"]), 1)

    def test_mismatched_active_row_count_fails_without_modifying_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_path, exception_path, _ = self._fixture(pathlib.Path(temporary))
            exception = json.loads(exception_path.read_text())
            exception["benchmark_replacement"]["active_rows"] = 3
            exception_path.write_text(json.dumps(exception), encoding="utf-8")
            before = state_path.read_bytes()

            with self.assertRaisesRegex(ValueError, "active Benchmark row count mismatch"):
                apply_operator_contract_change.apply(state_path, exception_path)

            self.assertEqual(state_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
