#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare Cosmos3-Nano weights for Framework and Cosmos-RL loaders.

The helper owns the reproducible Nano conversion defaults. It checks out an
exact Cosmos Framework revision, runs its converter in a digest-pinned NVIDIA
PyTorch image, and records the complete input and output provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


SKILL_DIR = Path(__file__).resolve().parents[1]
DEFAULTS_PATH = SKILL_DIR / "references" / "cosmos3-conversion-defaults.json"
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def conversion_defaults() -> dict[str, Any]:
    """Load and minimally validate the checked-in conversion policy."""
    payload = json.loads(DEFAULTS_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported conversion-default schema: {DEFAULTS_PATH}")
    converter = payload.get("converter", {})
    required = {"repository", "revision", "image", "image_digest", "dependency_group"}
    missing = sorted(required - set(converter))
    if missing:
        raise ValueError(f"conversion defaults are incomplete; missing: {missing}")
    if not re.fullmatch(r"[0-9a-f]{40}", str(converter["revision"])):
        raise ValueError("converter revision must be an immutable 40-character commit")
    if not _DIGEST_RE.fullmatch(str(converter["image_digest"])):
        raise ValueError("converter image_digest must be an immutable sha256 digest")
    for model, model_defaults in payload.get("models", {}).items():
        if not re.fullmatch(r"[0-9a-f]{40}", str(model_defaults.get("revision", ""))):
            raise ValueError(f"{model} revision must be an immutable 40-character commit")
        architecture = model_defaults.get("architecture_model", {})
        if not architecture.get("path_or_uri") or not re.fullmatch(
            r"[0-9a-f]{40}", str(architecture.get("revision", ""))
        ):
            raise ValueError(f"{model} architecture model must include an immutable revision")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_uri(value: str) -> bool:
    if "://" in value:
        return True
    path = Path(value).expanduser()
    if path.is_absolute() or value.startswith((".", "~")):
        return False
    return "/" in value and not path.exists()


def identity(value: str) -> dict[str, Any]:
    path = Path(value).expanduser()
    return {
        "original": value,
        "resolved": None if is_uri(value) else str(path.resolve()),
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
        if file.is_file() and file.name != "tao_conversion_provenance.json" and (
            file.suffix in {".json", ".safetensors", ".jinja"}
        ):
            files.append({"name": file.name, "size": file.stat().st_size, "sha256": sha256(file)})
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"model_type": "qwen3_vl", "files": files, "fingerprint": fingerprint}


def docker_mount(value: str, container_root: str) -> tuple[list[str], str]:
    path = Path(value).expanduser()
    if not path.exists():
        return [], value
    resolved = path.resolve()
    if resolved.parent.name == "snapshots" and (resolved.parent.parent / "blobs").is_dir():
        repository = resolved.parent.parent.resolve()
        container = f"{container_root}/repository/snapshots/{resolved.name}"
        return ["-v", f"{repository}:{container_root}/repository:ro"], container
    container = f"{container_root}/{resolved.name}"
    return ["-v", f"{resolved}:{container}:ro"], container


def _run_checked(command: list[str], *, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=capture_output, check=True)


def ensure_framework_checkout(path: Path, repository: str, revision: str) -> Path:
    """Create or verify a clean, exact converter checkout."""
    converter = path / "cosmos_framework" / "scripts" / "convert_model_to_vlm_safetensors.py"
    if path.exists():
        if not path.is_dir() or not converter.is_file():
            raise ValueError(f"framework checkout is invalid: {path}")
        head = _run_checked(
            ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True
        ).stdout.strip()
        if head != revision:
            raise ValueError(f"framework checkout revision is {head}, expected {revision}: {path}")
        dirty = _run_checked(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=normal"],
            capture_output=True,
        ).stdout.strip()
        if dirty:
            raise ValueError(f"framework checkout has source changes and cannot be reused: {path}")
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    if partial.exists():
        shutil.rmtree(partial)
    try:
        _run_checked([
            "git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1",
            repository, str(partial),
        ])
        _run_checked(["git", "-C", str(partial), "fetch", "--depth", "1", "origin", revision])
        _run_checked(["git", "-C", str(partial), "checkout", "--detach", "FETCH_HEAD"])
        if not (partial / "cosmos_framework" / "scripts" / "convert_model_to_vlm_safetensors.py").is_file():
            raise ValueError(f"revision {revision} does not contain the Cosmos3 VLM converter")
        partial.replace(path)
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise
    return path


def resolve_args(args: argparse.Namespace) -> tuple[dict[str, Any], str, str]:
    """Apply packaged Nano defaults and return defaults, image ref, and digest."""
    defaults = conversion_defaults()
    nano = defaults["models"]["nvidia/Cosmos3-Nano"]
    converter = defaults["converter"]

    if args.base_model_path_or_uri == "nvidia/Cosmos3-Nano" and not args.base_model_revision:
        args.base_model_revision = nano["revision"]
    args.base_model_ignore_patterns = (
        list(nano.get("ignore_patterns", []))
        if args.base_model_path_or_uri == "nvidia/Cosmos3-Nano"
        else []
    )
    if not args.vlm_architecture_model_path_or_uri:
        args.vlm_architecture_model_path_or_uri = nano["architecture_model"]["path_or_uri"]
    if (
        args.vlm_architecture_model_path_or_uri == nano["architecture_model"]["path_or_uri"]
        and not args.vlm_architecture_model_revision
    ):
        args.vlm_architecture_model_revision = nano["architecture_model"]["revision"]
    if not args.cosmos_framework_repo:
        args.cosmos_framework_repo = converter["repository"]
    if args.cosmos_framework_repo == converter["repository"] and not args.cosmos_framework_revision:
        args.cosmos_framework_revision = converter["revision"]
    if not args.conversion_image:
        args.conversion_image = converter["image"]

    inline_digest = ""
    if "@" in args.conversion_image:
        image, inline_digest = args.conversion_image.rsplit("@", 1)
    else:
        image = args.conversion_image
    digest = args.framework_image_digest or inline_digest
    if image == converter["image"] and not digest:
        digest = converter["image_digest"]
    if not _DIGEST_RE.fullmatch(digest):
        raise ValueError("conversion image must include an immutable sha256 digest")
    if inline_digest and args.framework_image_digest and inline_digest != args.framework_image_digest:
        raise ValueError("conversion image digest conflicts with --framework-image-digest")
    if not args.cosmos_framework_revision:
        raise ValueError("an immutable --cosmos-framework-revision is required")
    if not re.fullmatch(r"[0-9a-f]{40}", args.cosmos_framework_revision):
        raise ValueError("cosmos-framework revision must be an immutable 40-character commit")
    return defaults, f"{image}@{digest}", digest


def request_provenance(
    args: argparse.Namespace, image_ref: str, image_digest: str
) -> dict[str, Any]:
    return {
        "base_model": identity(args.base_model_path_or_uri),
        "base_model_revision": args.base_model_revision or None,
        "base_model_ignore_patterns": list(args.base_model_ignore_patterns),
        "architecture_model": identity(args.vlm_architecture_model_path_or_uri),
        "architecture_model_revision": args.vlm_architecture_model_revision or None,
        "cosmos_framework_repository": args.cosmos_framework_repo,
        "cosmos_framework_revision": args.cosmos_framework_revision,
        "framework_image": image_ref,
        "framework_image_digest": image_digest,
    }


def provenance_matches(metadata: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(metadata.get(key) == value for key, value in expected.items())


def command(
    args: argparse.Namespace,
    output: Path,
    cache: Path,
    framework_checkout: Path,
    image_ref: str,
    dependency_group: str,
) -> list[str]:
    source_mount, source = docker_mount(args.base_model_path_or_uri, "/inputs/base")
    donor_mount, donor = docker_mount(args.vlm_architecture_model_path_or_uri, "/inputs/architecture")
    container_output = f"/output/{output.name}"
    script = r'''
set -Eeuo pipefail
cd /workspace/cosmos-framework
if ! .venv/bin/python -c 'import cosmos_framework, torch' >/dev/null 2>&1; then
  uv sync --frozen --no-default-groups --group "$DEPENDENCY_GROUP"
fi
# The pinned "$DEPENDENCY_GROUP" lockfile omits iopath, which
# cosmos_framework.scripts.convert_model_to_vlm_safetensors needs via its
# inference.common.config -> utils.lazy_config import chain. Installing it
# into the already-synced venv (not via `uv sync`) keeps the frozen lockfile
# and pinned cosmos-framework checkout untouched.
if ! .venv/bin/python -c 'import iopath' >/dev/null 2>&1; then
  uv pip install --python .venv/bin/python iopath
fi
source_value="$BASE_MODEL"
architecture_value="$ARCHITECTURE_MODEL"
if [[ "$BASE_MODEL_KIND" == "uri" ]]; then
  source_value="$(.venv/bin/python - <<'PY'
import json
import os
from huggingface_hub import snapshot_download
print(snapshot_download(
    os.environ['BASE_MODEL'], revision=os.environ['BASE_MODEL_REVISION'],
    cache_dir='/cache/huggingface',
    ignore_patterns=json.loads(os.environ['BASE_MODEL_IGNORE_PATTERNS']),
))
PY
)"
fi
if [[ "$ARCHITECTURE_MODEL_KIND" == "uri" ]]; then
  architecture_value="$(.venv/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download
print(snapshot_download(os.environ['ARCHITECTURE_MODEL'], revision=os.environ['ARCHITECTURE_MODEL_REVISION'], cache_dir='/cache/huggingface'))
PY
)"
fi
export LD_LIBRARY_PATH=
.venv/bin/python -m cosmos_framework.scripts.convert_model_to_vlm_safetensors \
  --checkpoint-path "$source_value" --output-path "$OUTPUT_PATH" \
  --vlm-model-name "$architecture_value"
'''
    result = [
        "docker", "run", "--rm", "--ipc=host", "--entrypoint", "bash",
        "--user", f"{os.getuid()}:{os.getgid()}",
    ]
    if args.secrets_env:
        secrets = Path(args.secrets_env).expanduser()
        if not secrets.is_file():
            raise ValueError(f"secrets env file is inaccessible: {args.secrets_env}")
        result.extend(["--env-file", str(secrets.resolve())])
    result.extend([
        "-e", f"BASE_MODEL={source}",
        "-e", f"BASE_MODEL_KIND={'uri' if is_uri(args.base_model_path_or_uri) else 'local'}",
        "-e", f"BASE_MODEL_REVISION={args.base_model_revision}",
        "-e", f"BASE_MODEL_IGNORE_PATTERNS={json.dumps(args.base_model_ignore_patterns)}",
        "-e", f"ARCHITECTURE_MODEL={donor}",
        "-e", f"ARCHITECTURE_MODEL_KIND={'uri' if is_uri(args.vlm_architecture_model_path_or_uri) else 'local'}",
        "-e", f"ARCHITECTURE_MODEL_REVISION={args.vlm_architecture_model_revision}",
        "-e", f"OUTPUT_PATH={container_output}",
        "-e", f"DEPENDENCY_GROUP={dependency_group}",
        "-e", "HOME=/cache/home", "-e", "XDG_CACHE_HOME=/cache/xdg",
        "-e", "UV_CACHE_DIR=/cache/uv", "-e", "HF_HOME=/cache/huggingface",
        "-e", "UV_LINK_MODE=copy", "-e", "PYTHONUNBUFFERED=1",
        "-v", f"{framework_checkout}:/workspace/cosmos-framework",
        "-v", f"{output.parent}:/output", "-v", f"{cache}:/cache",
        *source_mount, *donor_mount,
    ])
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(name):
            result.extend(["-e", name])
    result.extend([image_ref, "-lc", script])
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path-or-uri", "--checkpoint-path", dest="base_model_path_or_uri", required=True)
    parser.add_argument("--base-model-revision", default="")
    parser.add_argument("--vlm-architecture-model-path-or-uri", "--vlm-model-name", dest="vlm_architecture_model_path_or_uri", default="")
    parser.add_argument("--vlm-architecture-model-revision", default="")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--cache-dir", default="~/.cache/tao-cosmos3-conversion")
    parser.add_argument("--cosmos-framework-path", default="")
    parser.add_argument("--cosmos-framework-repo", default="")
    parser.add_argument("--cosmos-framework-revision", default="")
    parser.add_argument("--conversion-image", "--framework-image", dest="conversion_image", default="")
    parser.add_argument("--framework-image-digest", default="")
    parser.add_argument("--secrets-env", default="")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        defaults, image_ref, image_digest = resolve_args(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    requested_output = Path(args.output_path).expanduser()
    if requested_output.is_symlink():
        print(f"ERROR: output path must not be a symlink: {requested_output}", file=sys.stderr)
        return 2
    output = requested_output.resolve()
    cache = Path(args.cache_dir).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    expected = request_provenance(args, image_ref, image_digest)

    if output.exists():
        try:
            existing = validate(output)
            metadata = json.loads((output / "tao_conversion_provenance.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            if not args.force:
                print("ERROR: output exists but is incomplete; use --force to replace this exact target", file=sys.stderr)
                return 2
            shutil.rmtree(output)
        else:
            if provenance_matches(metadata, expected) and not args.force:
                print(json.dumps({"status": "reused_verified", **existing}, indent=2))
                return 0
            if not args.force:
                print("ERROR: prepared model provenance does not match this conversion request; use --force to reproduce it", file=sys.stderr)
                return 2
            shutil.rmtree(output)

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

    converter = defaults["converter"]
    checkout = (
        Path(args.cosmos_framework_path).expanduser().resolve()
        if args.cosmos_framework_path
        else cache / "cosmos-framework" / args.cosmos_framework_revision
    )
    try:
        ensure_framework_checkout(checkout, args.cosmos_framework_repo, args.cosmos_framework_revision)
        run = subprocess.run(
            command(args, output, cache, checkout, image_ref, converter["dependency_group"]),
            check=False,
        )
        if run.returncode:
            return run.returncode
        prepared = validate(output)
        provenance = {
            "schema_version": 2,
            **expected,
            "cosmos_framework_checkout": str(checkout),
            "output": identity(str(output)),
            "prepared": prepared,
        }
        (output / "tao_conversion_provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "converted", **prepared}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
