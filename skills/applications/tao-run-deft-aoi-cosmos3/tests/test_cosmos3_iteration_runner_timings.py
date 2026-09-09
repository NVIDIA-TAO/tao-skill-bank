# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import render_iteration_mining_runner as runner  # noqa: E402


def _request(root: pathlib.Path) -> dict:
    return {
        "selector_command": [sys.executable, "selector.py"],
        "previous_jsonl": None, "previous_sha256": None,
        "mined_jsonl": root / "mined.jsonl",
        "current_quota_manifest": root / "current-quota.json",
        "train_jsonl": root / "train.jsonl",
        "assemble_summary": root / "assembly.json",
        "final_quota_manifest": root / "quota.json",
        "media_root": root, "max_rows": 768, "row_multiple": 768,
        "epochs": 5, "global_batch": 768,
        "source_pair_assets_dir": root / "old-source-cache",
        "query_pair_assets_dir": root / "new-query-cache",
        "mining_commands": {
            name: [sys.executable, "-c", f"print('child:{name}', flush=True)"]
            for name in (
                "source_inputs", "source_embeddings", "query_inputs",
                "query_embeddings", "routing", "history", "emission",
            )
        },
    }


class Cosmos3IterationRunnerTimingTests(unittest.TestCase):
    def test_cli_generates_plan_and_runner_without_executing_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            request = _request(root)
            request_path = root / "request.json"
            request_path.write_text(json.dumps(request, default=str), encoding="utf-8")
            plan_path, runner_path = root / "plan.json", root / "runner.py"
            with contextlib.redirect_stdout(io.StringIO()):
                status = runner.main([
                    "--request", str(request_path), "--output", str(plan_path),
                    "--runner-output", str(runner_path),
                ])
            self.assertEqual(status, 0)
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(plan, runner.build_plan(**request))
            self.assertEqual(runner_path.read_text(encoding="utf-8"), runner.render_runner(plan))
            self.assertEqual(set(root.iterdir()), {request_path, plan_path, runner_path})

    def test_plan_wires_source_and_query_roots_to_their_own_consumers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            request = _request(root)
            plan = runner.build_plan(**request)
            self.assertEqual(plan["source_pair_assets_dir"], str(root / "old-source-cache"))
            self.assertEqual(plan["query_pair_assets_dir"], str(root / "new-query-cache"))
            stages = {stage["name"]: stage["command"] for stage in plan["mining_stages"]}
            self.assertEqual(list(stages), list(request["mining_commands"]))
            for name in ("source_inputs", "routing", "emission"):
                self.assertEqual(stages[name][-2:], ["--pair-assets-dir", plan["source_pair_assets_dir"]])
            self.assertEqual(stages["query_inputs"][-2:], ["--pair-assets-dir", plan["query_pair_assets_dir"]])
            for name in ("source_embeddings", "query_embeddings", "history"):
                self.assertNotIn("--pair-assets-dir", stages[name])
            self.assertNotEqual(plan["selector"]["output"], plan["assembler"]["output"])
            self.assertTrue(plan["invariants"]["single_assembly_boundary"])
            self.assertEqual(list(root.iterdir()), [])

    def test_missing_roots_and_caller_pair_directory_overrides_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            for missing in ("source_pair_assets_dir", "query_pair_assets_dir"):
                with self.subTest(missing=missing):
                    request = _request(root)
                    request[missing] = None
                    with self.assertRaisesRegex(ValueError, missing):
                        runner.build_plan(**request)
            for flag in (["--pair-assets-dir", "wrong-cache"], ["--pair-assets-dir=wrong-cache"]):
                with self.subTest(flag=flag):
                    request = _request(root)
                    request["mining_commands"]["routing"].extend(flag)
                    with self.assertRaisesRegex(ValueError, "pair-assets-dir.*owned"):
                        runner.build_plan(**request)

    def test_generated_runner_flushes_start_and_end_around_each_real_cpu_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            plan = runner.build_plan(**_request(root))
            for name in ("selector", "assembler"):
                plan[name]["command"] = [
                    sys.executable, "-c", f"print('child:{name}', flush=True)",
                ]
            executable = root / "iteration_mining_runner.py"
            executable.write_text(runner.render_runner(plan), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(executable)], capture_output=True, text=True,
                check=False, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = result.stdout.splitlines()
            names = list(_request(root)["mining_commands"]) + ["selector", "assembler"]
            self.assertEqual(len(lines), 3 * len(names))
            for index, name in enumerate(names):
                start, child, end = lines[3 * index:3 * index + 3]
                start, end = json.loads(start), json.loads(end)
                self.assertEqual((start["event"], start["stage"]), ("stage_start", name))
                self.assertEqual(child, f"child:{name}")
                self.assertEqual((end["event"], end["stage"]), ("stage_end", name))
                self.assertGreaterEqual(end["elapsed_seconds"], 0)
                self.assertGreaterEqual(end["timestamp"], start["timestamp"])
                self.assertEqual(end["returncode"], 0)

    def test_generated_runner_logs_failure_and_never_starts_later_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            request = _request(root)
            request["mining_commands"] = {"routing": [sys.executable, "-c", "raise SystemExit(7)"]}
            plan = runner.build_plan(**request)
            executable = root / "iteration_mining_runner.py"
            executable.write_text(runner.render_runner(plan), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(executable)], capture_output=True, text=True,
                check=False, timeout=30,
            )
            self.assertEqual(result.returncode, 7, result.stderr)
            events = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual([event["event"] for event in events], ["stage_start", "stage_end"])
            self.assertEqual({event["stage"] for event in events}, {"routing"})
            self.assertEqual(events[-1]["returncode"], 7)


if __name__ == "__main__":
    unittest.main()
