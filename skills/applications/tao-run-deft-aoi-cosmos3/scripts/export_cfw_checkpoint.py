#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Compose, execute, and verify the native Framework DCP-to-HF export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import pwd
import re
import shlex
import subprocess
import sys
from typing import Any

from submit_cfw_train import IMAGE_KEY, default_bank, resolve_image


EXPORTER = "cosmos_framework.scripts.export_vlm_dcp"
REQUIRED_RUNTIME_FILES = (
    "config.json",
    "checkpoint.json",
    "export_manifest.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
)
SHARD_RE = re.compile(r"^model-([0-9]{5})-of-([0-9]{5})\.safetensors$")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dcp_metadata(checkpoint: pathlib.Path) -> pathlib.Path:
    for candidate in (checkpoint / "model/.metadata", checkpoint / ".metadata"):
        if candidate.is_file():
            return candidate
    raise ValueError(f"no Framework model DCP metadata below {checkpoint}")


def verify_hf_artifact(output: pathlib.Path) -> dict[str, Any]:
    """Fail closed unless the merged HF directory is evaluator-complete."""
    if not output.is_dir():
        raise ValueError(f"export output directory is missing: {output}")
    missing_runtime = [
        name
        for name in REQUIRED_RUNTIME_FILES
        if not (output / name).is_file() or (output / name).stat().st_size == 0
    ]
    if missing_runtime:
        raise ValueError(
            "export is incomplete; missing or empty evaluator runtime file(s): "
            + ", ".join(missing_runtime)
        )

    index = output / "model.safetensors.index.json"
    payload = json.loads(index.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
    if not isinstance(weight_map, dict) or not weight_map or not all(
        isinstance(name, str) and name for name in weight_map.values()
    ):
        raise ValueError("model.safetensors.index.json has no valid weight_map")

    expected_names = sorted(set(weight_map.values()))
    actual_names = sorted(path.name for path in output.glob("*.safetensors"))
    if actual_names != expected_names:
        missing = sorted(set(expected_names) - set(actual_names))
        unexpected = sorted(set(actual_names) - set(expected_names))
        raise ValueError(
            "export shard set does not match model.safetensors.index.json; "
            f"expected={len(expected_names)} actual={len(actual_names)} "
            f"missing={missing} unexpected={unexpected}"
        )

    shard_parts = [SHARD_RE.fullmatch(name) for name in expected_names]
    if any(match is None for match in shard_parts):
        raise ValueError(
            "indexed export shards must use model-NNNNN-of-NNNNN.safetensors names"
        )
    ordinals = {int(match.group(1)) for match in shard_parts if match}
    totals = {int(match.group(2)) for match in shard_parts if match}
    if totals != {len(expected_names)} or ordinals != set(
        range(1, len(expected_names) + 1)
    ):
        raise ValueError(
            "export shard names do not declare one complete contiguous shard set; "
            f"expected_count={len(expected_names)} ordinals={sorted(ordinals)} "
            f"declared_totals={sorted(totals)}"
        )
    empty = [
        name for name in expected_names if (output / name).stat().st_size == 0
    ]
    if empty:
        raise ValueError(
            "export contains empty safetensors shard(s): " + ", ".join(empty)
        )
    return {
        "weight_files": expected_names,
        "expected_shard_count": len(expected_names),
        "required_runtime_files": list(REQUIRED_RUNTIME_FILES),
    }


def verify_export(checkpoint: pathlib.Path, output: pathlib.Path) -> dict[str, Any]:
    metadata = dcp_metadata(checkpoint)
    manifest_path = output / "export_manifest.json"
    artifact = verify_hf_artifact(output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "cosmos-framework-vlm-dcp":
        raise ValueError("export manifest has the wrong Framework format")
    if manifest.get("checkpoint_metadata_sha256") != sha256_file(metadata):
        raise ValueError("export manifest does not match the DCP metadata")
    if manifest.get("lora", {}).get("enabled") is not True or int(manifest.get("merged_adapters", 0)) < 1:
        raise ValueError("export must prove that native VLM LoRA adapters were merged")
    return {
        "schema_version": 1,
        "status": "VERIFIED",
        "train_backend": "cosmos-framework",
        "source_checkpoint": str(checkpoint),
        "action_model_path": str(output),
        "evaluation_backend": "cosmos-rl-vllm",
        "evaluation_model_contract": {"enable_lora": False, "model_name": str(output)},
        **artifact,
        "checkpoint_metadata_sha256": sha256_file(metadata),
        "manifest": manifest,
    }


def build_docker_argv(
    *,
    immutable_image: str,
    checkpoint: pathlib.Path,
    config: pathlib.Path,
    output: pathlib.Path,
    base_model: pathlib.Path,
    gpus: str,
) -> list[str]:
    username = pwd.getpwuid(os.getuid()).pw_name
    return [
        "docker", "run", "--rm", "--gpus", gpus, "--ipc=host",
        "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", f"USER={username}", "-e", f"LOGNAME={username}", "-e", "HOME=/tmp",
        "-v", "/etc/passwd:/etc/passwd:ro", "-v", "/etc/group:/etc/group:ro",
        "--mount", f"type=bind,src={checkpoint},dst=/tao/checkpoint,readonly",
        "--mount", f"type=bind,src={config},dst=/tao/config.yaml,readonly",
        "--mount", f"type=bind,src={base_model},dst=/tao/base,readonly",
        "--mount", f"type=bind,src={output.parent},dst=/tao/export",
        "-w", "/tao/export",
        "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
        "--entrypoint", "/workspace/.venv/bin/python", immutable_image,
        "-m", EXPORTER,
        "--checkpoint-path", "/tao/checkpoint",
        "--config-file", "/tao/config.yaml",
        "--output-dir", f"/tao/export/{output.name}",
        "--base-model-path-or-uri", "/tao/base",
        "--dtype", "bfloat16",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill-bank", type=pathlib.Path, default=default_bank())
    parser.add_argument("--checkpoint-path", required=True, type=pathlib.Path)
    parser.add_argument("--config-file", required=True, type=pathlib.Path, help="Resolved Framework config.yaml")
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--base-model-path", required=True, type=pathlib.Path)
    parser.add_argument("--gpus", default="all")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--manifest", type=pathlib.Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        checkpoint = args.checkpoint_path.expanduser().resolve(strict=True)
        config = args.config_file.expanduser().resolve(strict=True)
        base_model = args.base_model_path.expanduser().resolve(strict=True)
        output = args.output_dir.expanduser().resolve()
        if args.verify_only:
            result = verify_export(checkpoint, output.resolve(strict=True))
        else:
            tagged, immutable = resolve_image(args.skill_bank.expanduser().resolve(strict=True))
            command = build_docker_argv(
                immutable_image=immutable,
                checkpoint=checkpoint,
                config=config,
                output=output,
                base_model=base_model,
                gpus=args.gpus,
            )
            result = {
                "schema_version": 1,
                "train_backend": "cosmos-framework",
                "image_key": IMAGE_KEY,
                "resolved_image": tagged,
                "image_digest": immutable,
                "exporter": EXPORTER,
                "command_argv": command,
                "command": shlex.join(command),
                "executed": False,
            }
            if args.execute:
                if output.exists():
                    raise ValueError(f"refusing to overwrite export output: {output}")
                output.parent.mkdir(parents=True, exist_ok=True)
                try:
                    subprocess.run(command, check=True)
                except subprocess.CalledProcessError as exc:
                    raise ValueError(
                        f"export command failed with exit code {exc.returncode}; "
                        f"partial output at {output} is unverified and must not "
                        "be committed (remove it before retrying)"
                    ) from exc
                result.update(verify_export(checkpoint, output.resolve(strict=True)))
                result["executed"] = True
        if args.manifest:
            args.manifest.expanduser().resolve().write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"export_cfw_checkpoint: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
