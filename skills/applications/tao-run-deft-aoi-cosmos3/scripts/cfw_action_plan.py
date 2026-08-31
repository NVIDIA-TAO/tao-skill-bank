#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Build platform-neutral Cosmos Framework action descriptors for DEFT AOI."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any


def _image(value: str) -> str:
    if not re.search(r"@sha256:[0-9a-f]{64}$", value):
        raise ValueError("action descriptor requires an immutable image digest")
    return value


def build_train_plan(
    *,
    image: str,
    config_path: str,
    results_dir: str,
    adapter_root: str,
    model_path: str,
    train_jsonl: str,
    media_root: str,
    index_path: str,
    hydra_overrides: list[str],
    num_gpus: int,
) -> dict[str, Any]:
    if num_gpus <= 0:
        raise ValueError("num_gpus must be positive")
    required_paths = (
        config_path,
        results_dir,
        adapter_root,
        model_path,
        train_jsonl,
        media_root,
        index_path,
    )
    if not all(required_paths):
        raise ValueError("train descriptor requires all model/data/config paths")
    command = [
        "/workspace/.venv/bin/python",
        "-m",
        "nvpaw_cfw.train",
        f"--sft-toml={config_path}",
        *hydra_overrides,
    ]
    return {
        "schema_version": 1,
        "application": "tao-run-deft-aoi-cosmos3",
        "workload": "deft-aoi",
        "backend": "cosmos-framework",
        "action": "train",
        "image": _image(image),
        "command": command,
        "environment": {"PYTHONPATH": str(pathlib.PurePosixPath(adapter_root).parent)},
        "inputs": {
            "config": config_path,
            "adapter_package": adapter_root,
            "model": model_path,
            "train_jsonl": train_jsonl,
            "media_root": media_root,
        },
        "outputs": {
            "results_dir": results_dir,
            "dataset_index": index_path,
            "checkpoint_format": "framework_dcp",
        },
        "working_directory": results_dir,
        "resources": {"gpus": num_gpus, "nodes": 1, "distributed": num_gpus > 1},
        "supporting_files": [adapter_root],
        "submission_owner": "selected_platform_four_verb_contract",
    }


def build_action_plan(
    *,
    action: str,
    image: str,
    config_path: str,
    results_dir: str,
    runtime_adapter: str,
    output_jsonl: str,
    source_jsonl: str,
    media_root: str,
    checkpoint_path: str,
    base_model_path: str,
    framework_config: str = "",
    action_model_dir: str = "",
    media_path: str = "",
    prompt: str = "",
    media_type: str = "image",
    max_new_tokens: int = 1024,
) -> dict[str, Any]:
    if action not in {"evaluate", "inference"}:
        raise ValueError("action must be evaluate or inference")
    if not runtime_adapter or not output_jsonl:
        raise ValueError("evaluate/inference require runtime_adapter and output_jsonl")
    handoff = [bool(framework_config), bool(action_model_dir)]
    if any(handoff) and not all(handoff):
        raise ValueError(
            "Framework DCP handoff requires framework_config and action_model_dir"
        )
    selected_model_path = action_model_dir if framework_config else checkpoint_path
    command = [
        "/workspace/.venv/bin/python",
        runtime_adapter,
        f"--action={action}",
        f"--model-path={selected_model_path}",
        f"--output-jsonl={output_jsonl}",
    ]
    if action == "evaluate":
        if not config_path:
            raise ValueError("evaluate requires config_path")
        if not source_jsonl or not media_root:
            raise ValueError("evaluate requires source_jsonl and media_root")
        command.append(f"--config={config_path}")
    else:
        if not media_path or not prompt:
            raise ValueError("inference requires media_path and prompt")
        if media_type not in {"image", "video"}:
            raise ValueError("media_type must be image or video")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        command.extend(
            [
                f"--media-type={media_type}",
                f"--media={media_path}",
                f"--prompt={prompt}",
                f"--max-new-tokens={max_new_tokens}",
            ]
        )
    result = {
        "schema_version": 1,
        "application": "tao-run-deft-aoi-cosmos3",
        "workload": "deft-aoi",
        "backend": "cosmos-framework",
        "action": action,
        "image": _image(image),
        "command": command,
        "inputs": {
            "checkpoint": checkpoint_path,
            "base_model": base_model_path,
            "runtime_adapter": runtime_adapter,
        },
        "outputs": {
            "results_dir": results_dir,
            "predictions_jsonl": output_jsonl,
            "format": "jsonl",
        },
        "working_directory": results_dir,
        "resources": {"gpus": 1, "nodes": 1, "distributed": False},
        "prediction_contract": {
            "producer": "cfw_jsonl_runtime.py",
            "output_schema": ["id", "task_type", "message", "GT", "raw_prediction"],
            "atomic": True,
            "complete_coverage_required": action == "evaluate",
        },
        "supporting_files": [runtime_adapter],
        "submission_owner": "selected_platform_four_verb_contract",
    }
    if config_path:
        result["inputs"]["config"] = config_path
    if source_jsonl:
        result["inputs"]["source_jsonl"] = source_jsonl
    if media_root:
        result["inputs"]["media_root"] = media_root
    if media_path:
        result["inputs"]["media"] = media_path
    if framework_config:
        result["inputs"].update(
            {
                "framework_config": framework_config,
                "action_model_dir": action_model_dir,
            }
        )
        result["checkpoint_pre_action"] = {
            "owner": "tao-finetune-cosmos-reason",
            "module": "framework_checkpoint_action.py",
            "verb": "prepare",
            "action": action,
            "checkpoint_path": checkpoint_path,
            "config_file": framework_config,
            "export_dir": action_model_dir,
            "base_model_path_or_uri": base_model_path,
            "completion_evidence": "verified exact-key export manifest",
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        if not isinstance(request, dict):
            raise ValueError("request must be a nested JSON object")
        action = request.get("action")
        if action == "train":
            result = build_train_plan(**request["parameters"])
        else:
            result = build_action_plan(action=action, **request["parameters"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"cfw_action_plan: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
