#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Render the immutable selector-to-assembler boundary for one DEFT iteration."""

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
    return {
        "schema_version": "deft_iteration_materialization_plan_v1",
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
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
        ):
            if request.get(field) is not None:
                request[field] = pathlib.Path(request[field])
        plan = build_plan(**request)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"render_iteration_mining_runner: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(plan, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
