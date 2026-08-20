#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage a checksum-closed Cosmos action-helper bundle.

SLURM actions execute checked-in orchestration helpers from their job input
directory.  Stage the complete declared dependency set atomically so a helper
can never be submitted without its sibling imports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

BUNDLES = {
    "framework-checkpoint": (
        "framework_checkpoint_action.py",
        "cosmos_common.py",
    ),
    "cosmos-rl-checkpoint": (
        "cosmos_rl_checkpoint_action.py",
        "cosmos_common.py",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage(bundle: str, output: Path) -> dict[str, object]:
    source_root = Path(__file__).resolve().parent
    names = BUNDLES[bundle]
    missing = [name for name in names if not (source_root / name).is_file()]
    if missing:
        raise ValueError(f"action bundle source files are missing: {missing}")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent)
    )
    try:
        files = []
        for name in names:
            target = temporary / name
            shutil.copy2(source_root / name, target)
            files.append(
                {"path": name, "sha256": _sha256(target), "size": target.stat().st_size}
            )
        manifest = {
            "schema_version": 1,
            "bundle": bundle,
            "files": files,
        }
        manifest_path = temporary / "bundle_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if output.exists():
            existing_manifest = output / "bundle_manifest.json"
            if existing_manifest.is_file() and json.loads(
                existing_manifest.read_text(encoding="utf-8")
            ) == manifest:
                return {**manifest, "output": str(output), "state": "reused"}
            raise ValueError(f"refusing to overwrite a different staged action bundle: {output}")
        os.replace(temporary, output)
        return {**manifest, "output": str(output), "state": "staged"}
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", choices=sorted(BUNDLES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(stage(args.bundle, args.output), indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
