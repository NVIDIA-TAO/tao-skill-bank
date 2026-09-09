#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Render a timed mining runner with one immutable selector-to-assembler boundary."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

from assemble_training_json import sha256_file


_ASSEMBLER_ONLY_TOGGLES = {
    "--repetition-blend",
    "--no-repetition-blend",
    "--repetition-redistribute",
    "--no-repetition-redistribute",
    "--repetition-never-repeat-empty-gt",
    "--repetition-allow-empty-gt",
}
_ASSEMBLER_ONLY_VALUES = {
    "--gap-analysis-summary",
    "--repetition-config",
    "--repetition-policy",
    "--repetition-rep-min",
    "--repetition-rep-max",
    "--repetition-budget-multiplier",
    "--repetition-share-gap-tolerance",
    "--repetition-explicit-multiplier",
    "--repetition-seed",
}

# Optional upstream commands execute in dependency order. Only embedding-input
# preparation may render images; routing/emission use the source cache as keys.
_MINING_STAGE_ROOTS = {
    "source_inputs": "source_pair_assets_dir",
    "source_embeddings": None,
    "query_inputs": "query_pair_assets_dir",
    "query_embeddings": None,
    "routing": "source_pair_assets_dir",
    "history": None,
    "emission": "source_pair_assets_dir",
}


def _mining_stages(
    commands: dict[str, list[str]] | None, roots: dict[str, str | None]
) -> list[dict[str, Any]]:
    if commands is None:
        return []
    if not isinstance(commands, dict) or set(commands) - set(_MINING_STAGE_ROOTS):
        raise ValueError("mining_commands must use the documented mining stage names")
    stages = []
    for name, root_field in _MINING_STAGE_ROOTS.items():
        if name not in commands:
            continue
        command = commands[name]
        if not isinstance(command, list) or not command or not all(
            isinstance(value, str) and value for value in command
        ):
            raise ValueError(f"{name} command must be a non-empty string list")
        command = list(command)
        if root_field is not None:
            if any(value.partition("=")[0] == "--pair-assets-dir" for value in command):
                raise ValueError(f"{name} --pair-assets-dir is owned by this renderer")
            root = roots[root_field]
            if root is None:
                raise ValueError(f"{name} requires {root_field}")
            command.extend(["--pair-assets-dir", root])
        stages.append({"name": name, "command": command})
    return stages


def _partition_materialization_arguments(
    command: list[str],
) -> tuple[list[str], list[str]]:
    """Move final-corpus repetition controls off the current-row selector."""

    selector: list[str] = []
    assembler: list[str] = []
    index = 0
    while index < len(command):
        value = command[index]
        if value in _ASSEMBLER_ONLY_TOGGLES:
            assembler.append(value)
            index += 1
            continue
        option = value.partition("=")[0]
        if option in _ASSEMBLER_ONLY_VALUES:
            assembler.append(value)
            if "=" not in value:
                if index + 1 == len(command):
                    raise ValueError(f"{value} requires a value")
                assembler.append(command[index + 1])
                index += 2
            else:
                index += 1
            continue
        if value.startswith(("--repetition-", "--no-repetition-")):
            raise ValueError(
                f"unsupported repetition materialization option {value!r}"
            )
        selector.append(value)
        index += 1
    if assembler:
        selector.append("--no-repetition-blend")
    return selector, assembler


def build_plan(
    *,
    selector_command: list[str],
    previous_jsonl: pathlib.Path | None,
    previous_sha256: str | None,
    mined_jsonl: pathlib.Path,
    current_quota_manifest: pathlib.Path,
    train_jsonl: pathlib.Path,
    assemble_summary: pathlib.Path,
    final_quota_manifest: pathlib.Path,
    media_root: pathlib.Path,
    max_rows: int,
    row_multiple: int,
    epochs: int,
    global_batch: int,
    source_pair_assets_dir: pathlib.Path | None = None,
    query_pair_assets_dir: pathlib.Path | None = None,
    mining_commands: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    if not selector_command or not all(
        isinstance(value, str) and value for value in selector_command
    ):
        raise ValueError("selector_command must be a non-empty string list")
    if any(
        value == "--output"
        or value.startswith("--output=")
        or value == "--manifest"
        or value.startswith("--manifest=")
        or value == "--repetition-manifest"
        or value.startswith("--repetition-manifest=")
        for value in selector_command
    ):
        raise ValueError("selector_command output paths are owned by this renderer")
    if min(max_rows, row_multiple, epochs, global_batch) <= 0:
        raise ValueError("row, epoch, and global-batch values must be positive")
    if row_multiple != global_batch:
        raise ValueError("row_multiple must equal global_batch")

    mined = mined_jsonl.expanduser().resolve()
    current_quota = current_quota_manifest.expanduser().resolve()
    train = train_jsonl.expanduser().resolve()
    summary = assemble_summary.expanduser().resolve()
    final_quota = final_quota_manifest.expanduser().resolve()
    if len({mined, current_quota, train, summary, final_quota}) != 5:
        raise ValueError("selector, assembly, and manifest outputs must be distinct")
    if previous_jsonl is None:
        if previous_sha256 is not None:
            raise ValueError("previous_sha256 requires previous_jsonl")
        previous = None
    else:
        previous = previous_jsonl.expanduser().resolve(strict=True)
        if previous_sha256 is None:
            raise ValueError("previous_sha256 is required with previous_jsonl")
        if sha256_file(previous) != previous_sha256:
            raise ValueError("previous training JSONL SHA-256 changed")
        if previous in {mined, current_quota, train, summary, final_quota}:
            raise ValueError("previous training JSONL must not be an iteration output")

    selector_argv, assembler_materialization_argv = (
        _partition_materialization_arguments(selector_command)
    )
    if previous is not None:
        selector_argv.extend(["--previous-jsonl", str(previous)])
    selector_argv.extend(
        ["--output", str(mined), "--manifest", str(current_quota)]
    )
    assembler_argv = [
        sys.executable,
        str(pathlib.Path(__file__).with_name("assemble_training_json.py")),
        "--mined-jsonl",
        str(mined),
        "--output",
        str(train),
        "--summary",
        str(summary),
        "--media-root",
        str(media_root.expanduser().resolve()),
        "--max-rows",
        str(max_rows),
        "--row-multiple",
        str(row_multiple),
        "--current-quota-manifest",
        str(current_quota),
        "--quota-manifest",
        str(final_quota),
        "--epochs",
        str(epochs),
        "--global-batch",
        str(global_batch),
    ]
    if previous is not None:
        assembler_argv.extend(
            ["--previous-jsonl", str(previous), "--previous-sha256", previous_sha256]
        )
    assembler_argv.extend(assembler_materialization_argv)
    roots = {
        "source_pair_assets_dir": str(source_pair_assets_dir.expanduser().resolve())
        if source_pair_assets_dir is not None else None,
        "query_pair_assets_dir": str(query_pair_assets_dir.expanduser().resolve())
        if query_pair_assets_dir is not None else None,
    }
    return {
        "schema_version": "deft_iteration_materialization_plan_v1",
        **roots,
        "mining_stages": _mining_stages(mining_commands, roots),
        "previous_jsonl": str(previous) if previous is not None else None,
        "previous_sha256": previous_sha256,
        "selector": {
            "command": selector_argv,
            "output": str(mined),
            "quota_manifest": str(current_quota),
        },
        "assembler": {
            "command": assembler_argv,
            "previous_jsonl": str(previous) if previous is not None else None,
            "previous_sha256": previous_sha256,
            "mined_input": str(mined),
            "output": str(train),
            "summary": str(summary),
            "quota_manifest": str(final_quota),
        },
        "invariants": {
            "selector_output_is_not_training_jsonl": mined != train,
            "previous_hash_verified": previous is None
            or sha256_file(previous) == previous_sha256,
            "single_assembly_boundary": True,
            "repetition_controls_owned_by_assembler": not any(
                value.startswith("--repetition-")
                and value != "--no-repetition-blend"
                for value in selector_argv
            ),
        },
    }


def render_runner(plan: dict[str, Any]) -> str:
    """Generate, but never execute, a standalone runner for the approved plan."""

    stages = [
        {"name": stage["name"], "command": stage["command"]}
        for stage in plan.get("mining_stages", [])
    ] + [
        {"name": name, "command": plan[name]["command"]}
        for name in ("selector", "assembler")
    ]
    return '''#!/usr/bin/env python3
# Generated by render_iteration_mining_runner.py; no stages run at generation time.
import json
import subprocess
import sys
import time

STAGES = ''' + repr(stages) + '''


def main():
    for stage in STAGES:
        started = time.monotonic()
        print(json.dumps({"event": "stage_start", "stage": stage["name"],
                          "timestamp": time.time()}), flush=True)
        returncode = 1
        try:
            returncode = subprocess.run(stage["command"], check=False).returncode
        except OSError as exc:
            print(str(exc), file=sys.stderr, flush=True)
            returncode = 127
        finally:
            print(json.dumps({"event": "stage_end", "stage": stage["name"],
                              "timestamp": time.time(), "returncode": returncode,
                              "elapsed_seconds": time.monotonic() - started}), flush=True)
        if returncode:
            return returncode if returncode > 0 else 128 - returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument(
        "--runner-output", type=pathlib.Path,
        help="also write a standalone Python runner (does not execute it)",
    )
    args = parser.parse_args(argv)
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        for field in (
            "previous_jsonl",
            "mined_jsonl",
            "current_quota_manifest",
            "train_jsonl",
            "assemble_summary",
            "final_quota_manifest",
            "media_root",
            "source_pair_assets_dir",
            "query_pair_assets_dir",
        ):
            if request.get(field) is not None:
                request[field] = pathlib.Path(request[field])
        plan = build_plan(**request)
        if (
            args.runner_output is not None
            and args.runner_output.resolve() == args.output.resolve()
        ):
            raise ValueError("runner-output must differ from the plan output")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if args.runner_output is not None:
            args.runner_output.parent.mkdir(parents=True, exist_ok=True)
            args.runner_output.write_text(render_runner(plan), encoding="utf-8")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"render_iteration_mining_runner: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(plan, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
