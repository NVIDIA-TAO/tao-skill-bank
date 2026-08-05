#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare Cosmos3-Nano weights for Framework and Cosmos-RL loaders.

The conversion runs through a clean repository-derived Cosmos Framework image
and records the source and output checkpoint provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_uri(value: str) -> bool:
    return "://" in value or ("/" in value and not Path(value).expanduser().exists())


def identity(value: str) -> dict[str, Any]:
    path = Path(value).expanduser()
    return {
        "original": value,
        "resolved": str(path.resolve()) if path.exists() else None,
        "kind": "uri" if is_uri(value) else "local",
    }


def validate(path: Path) -> dict[str, Any]:
    config_file = path / "config.json"
    if not config_file.is_file():
        raise ValueError(f"prepared checkpoint is missing config.json: {path}")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_vl":
        raise ValueError(f"prepared model_type must be qwen3_vl, found {config.get('model_type')!r}")
    weights = sorted(path.glob("*.safetensors"))
    index = path / "model.safetensors.index.json"
    if not weights and not index.is_file():
        raise ValueError("prepared checkpoint has no safetensors weights/index")
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        missing = sorted({name for name in weight_map.values() if not (path / name).is_file()})
        if missing:
            raise ValueError(f"prepared checkpoint is missing indexed shards: {missing[:10]}")
    required_processor = ("tokenizer_config.json", "tokenizer.json")
    missing_processor = [name for name in required_processor if not (path / name).is_file()]
    if missing_processor:
        raise ValueError(f"prepared checkpoint is missing tokenizer files: {missing_processor}")
    files = []
    for file in sorted(path.iterdir()):
        if file.is_file() and (file.suffix in {".json", ".safetensors", ".jinja"}):
            files.append({"name": file.name, "size": file.stat().st_size, "sha256": sha256(file)})
    return {"model_type": "qwen3_vl", "files": files, "fingerprint": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def docker_mount(value: str, container_root: str) -> tuple[list[str], str]:
    path = Path(value).expanduser()
    if not path.exists():
        return [], value
    resolved = path.resolve()
    container = f"{container_root}/{resolved.name}"
    return ["-v", f"{resolved}:{container}:ro"], container


def command(args: argparse.Namespace, output: Path, cache: Path) -> list[str]:
    source_mount, source = docker_mount(args.base_model_path_or_uri, "/inputs/base")
    donor_mount, donor = docker_mount(args.vlm_architecture_model_path_or_uri, "/inputs/architecture")
    script = r'''
set -Eeuo pipefail
source_value="$BASE_MODEL"
architecture_value="$ARCHITECTURE_MODEL"
if [[ "$BASE_MODEL_KIND" == "uri" ]]; then
  source_value="$(python - <<'PY'
import os
from huggingface_hub import snapshot_download
print(snapshot_download(os.environ['BASE_MODEL'], revision=os.environ['BASE_MODEL_REVISION'], cache_dir='/cache/huggingface'))
PY
)"
fi
if [[ "$ARCHITECTURE_MODEL_KIND" == "uri" ]]; then
  architecture_value="$(python - <<'PY'
import os
from huggingface_hub import snapshot_download
print(snapshot_download(os.environ['ARCHITECTURE_MODEL'], revision=os.environ['ARCHITECTURE_MODEL_REVISION'], cache_dir='/cache/huggingface'))
PY
)"
fi
python -m cosmos_framework.scripts.convert_model_to_vlm_safetensors \
  --checkpoint-path "$source_value" --output-path /output/prepared \
  --vlm-model-name "$architecture_value"
'''
    result = [
        "docker", "run", "--rm", "--ipc=host", "--entrypoint", "bash",
        "-e", f"BASE_MODEL={source}", "-e", f"BASE_MODEL_KIND={'uri' if is_uri(args.base_model_path_or_uri) else 'local'}",
        "-e", f"BASE_MODEL_REVISION={args.base_model_revision}",
        "-e", f"ARCHITECTURE_MODEL={donor}", "-e", f"ARCHITECTURE_MODEL_KIND={'uri' if is_uri(args.vlm_architecture_model_path_or_uri) else 'local'}",
        "-e", f"ARCHITECTURE_MODEL_REVISION={args.vlm_architecture_model_revision}",
        "-e", "HF_HOME=/cache/huggingface", "-e", "PYTHONUNBUFFERED=1",
        "-v", f"{output.parent}:/output", "-v", f"{cache}:/cache",
        *source_mount, *donor_mount,
    ]
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(name):
            result.extend(["-e", name])
    result.extend([args.framework_image, "-lc", script])
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path-or-uri", required=True)
    parser.add_argument("--base-model-revision", default="")
    parser.add_argument("--vlm-architecture-model-path-or-uri", required=True)
    parser.add_argument("--vlm-architecture-model-revision", default="")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--framework-image", required=True)
    parser.add_argument("--framework-image-digest", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for value, revision, label in (
        (args.base_model_path_or_uri, args.base_model_revision, "base model"),
        (args.vlm_architecture_model_path_or_uri, args.vlm_architecture_model_revision, "architecture model"),
    ):
        if is_uri(value) and not revision:
            print(f"ERROR: immutable revision is required for {label} URI {value!r}", file=sys.stderr)
            return 2
        if not is_uri(value) and not Path(value).expanduser().is_dir():
            print(f"ERROR: local {label} path is inaccessible: {value}", file=sys.stderr)
            return 2
    output = Path(args.output_path).expanduser()
    cache = Path(args.cache_dir).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    if output.exists():
        try:
            existing = validate(output)
        except (OSError, ValueError, json.JSONDecodeError):
            if not args.force:
                print("ERROR: output exists but is incomplete; use --force to replace this exact target", file=sys.stderr)
                return 2
            shutil.rmtree(output)
        else:
            metadata = output / "tao_conversion_provenance.json"
            if metadata.is_file() and not args.force:
                print(json.dumps({"status": "reused_verified", **existing}, indent=2))
                return 0
            if not args.force:
                print("ERROR: prepared model lacks conversion provenance; use --force to reproduce it", file=sys.stderr)
                return 2
            shutil.rmtree(output)
    run = subprocess.run(command(args, output, cache), check=False)
    if run.returncode:
        return run.returncode
    prepared = validate(output)
    provenance = {
        "schema_version": 1,
        "base_model": identity(args.base_model_path_or_uri), "base_model_revision": args.base_model_revision or None,
        "architecture_model": identity(args.vlm_architecture_model_path_or_uri), "architecture_model_revision": args.vlm_architecture_model_revision or None,
        "framework_image": args.framework_image, "framework_image_digest": args.framework_image_digest,
        "output": identity(str(output)), "prepared": prepared,
    }
    (output / "tao_conversion_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps(provenance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
