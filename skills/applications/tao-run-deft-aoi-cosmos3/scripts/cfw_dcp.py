#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Validate a complete Cosmos Framework DCP before DEFT commits or handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
from typing import Any


COMPONENTS = ("model", "optim", "scheduler", "trainer")
_SHARD_PATTERN = re.compile(r"__([0-9]+)_[0-9]+\.distcp")
_LOADER_PATTERN = re.compile(r"rank_([0-9]+)\.pkl")


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_for(checkpoint: pathlib.Path) -> pathlib.Path:
    for candidate in (
        checkpoint.parent.parent / "config.yaml",
        checkpoint.parent.parent / "config.json",
        checkpoint.parent / "config.yaml",
        checkpoint / "config.yaml",
    ):
        if candidate.is_file() and candidate.stat().st_size:
            return candidate.resolve()
    raise ValueError(f"Framework DCP has no adjacent non-empty config: {checkpoint}")


def _pointer_target(pointer: pathlib.Path) -> pathlib.Path:
    if pointer.is_symlink():
        return pointer.resolve(strict=True)
    value = pointer.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"latest pointer is empty: {pointer}")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        target = value
    else:
        if not isinstance(payload, dict) or not isinstance(payload.get("checkpoint"), str):
            raise ValueError(f"latest pointer JSON requires checkpoint: {pointer}")
        target = payload["checkpoint"]
    candidate = pathlib.Path(target).expanduser()
    return (candidate if candidate.is_absolute() else pointer.parent / candidate).resolve(strict=True)


def validate_checkpoint(
    checkpoint: pathlib.Path | str,
    *,
    latest_pointer: pathlib.Path | str | None = None,
) -> dict[str, Any]:
    root = pathlib.Path(checkpoint).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Framework DCP must be a directory: {root}")
    match = re.fullmatch(r"iter_([0-9]{9})", root.name)
    if not match:
        raise ValueError("Framework DCP directory must be named iter_#########")
    component_manifest: dict[str, Any] = {}
    expected_ranks: set[int] | None = None
    metadata: list[pathlib.Path] = []
    shards: list[pathlib.Path] = []
    for component in COMPONENTS:
        directory = root / component
        component_metadata = directory / ".metadata"
        if not component_metadata.is_file() or component_metadata.stat().st_size == 0:
            raise ValueError(
                f"Framework DCP is incomplete: {component}/.metadata is missing or empty"
            )
        component_shards = sorted(directory.glob("*.distcp"))
        if not component_shards:
            raise ValueError(
                f"Framework DCP is incomplete: {component} has no .distcp shards"
            )
        ranks: set[int] = set()
        for shard in component_shards:
            rank_match = _SHARD_PATTERN.fullmatch(shard.name)
            if rank_match is None:
                raise ValueError(
                    f"Framework DCP has an unrecognized {component} shard: {shard.name}"
                )
            ranks.add(int(rank_match.group(1)))
        if expected_ranks is None:
            expected_ranks = ranks
            if expected_ranks != set(range(len(expected_ranks))):
                raise ValueError(
                    f"Framework DCP model ranks must be contiguous from zero: {sorted(expected_ranks)}"
                )
        elif ranks != expected_ranks:
            raise ValueError(
                f"Framework DCP {component} ranks {sorted(ranks)} do not match model ranks "
                f"{sorted(expected_ranks)}"
            )
        # Framework writes zero-byte scheduler placeholders on non-primary
        # ranks. Every other state shard must carry bytes; scheduler must have
        # at least one materialized shard.
        nonempty = [path for path in component_shards if path.stat().st_size > 0]
        if component == "scheduler":
            if not nonempty:
                raise ValueError("Framework DCP scheduler has no materialized shard")
        elif len(nonempty) != len(component_shards):
            raise ValueError(f"Framework DCP {component} contains an empty shard")
        metadata.append(component_metadata)
        shards.extend(component_shards)
        component_manifest[component] = {
            "metadata_sha256": _sha256(component_metadata),
            "shard_files": len(component_shards),
            "shard_bytes": sum(path.stat().st_size for path in component_shards),
            "ranks": sorted(ranks),
        }
    assert expected_ranks is not None  # COMPONENTS is non-empty
    loader_dir = root / "dataloader"
    loader_files = sorted(loader_dir.glob("rank_*.pkl"))
    loader_ranks: set[int] = set()
    for loader in loader_files:
        loader_match = _LOADER_PATTERN.fullmatch(loader.name)
        if loader_match is None or loader.stat().st_size == 0:
            raise ValueError(f"Framework DCP has invalid dataloader rank state: {loader}")
        loader_ranks.add(int(loader_match.group(1)))
    if loader_ranks != expected_ranks:
        raise ValueError(
            f"Framework DCP dataloader rank state {sorted(loader_ranks)} does not match "
            f"model ranks {sorted(expected_ranks)}"
        )
    config = _config_for(root)
    if latest_pointer is None:
        for candidate in (
            root.parent / "latest_checkpoint.txt",
            root.parent / "latest",
        ):
            if candidate.exists():
                latest_pointer = candidate
                break
    if latest_pointer is None:
        raise ValueError("Framework DCP has no latest checkpoint pointer")
    pointer_path = pathlib.Path(latest_pointer).expanduser().absolute()
    if not pointer_path.exists():
        raise ValueError(f"latest checkpoint pointer does not exist: {pointer_path}")
    pointer = _pointer_target(pointer_path)
    if pointer != root:
        raise ValueError(f"latest pointer selects {pointer}, expected {root}")
    return {
        "schema_version": 1,
        "checkpoint_kind": "framework_dcp",
        "checkpoint": str(root),
        "iteration": int(match.group(1)),
        "rank_count": len(expected_ranks),
        "components": component_manifest,
        "metadata_files": len(metadata),
        "metadata_sha256": {str(path.relative_to(root)): _sha256(path) for path in metadata},
        "shard_files": len(shards),
        "shard_bytes": sum(path.stat().st_size for path in shards),
        "dataloader_state_files": len(loader_files),
        "dataloader_state_sha256": {
            str(path.relative_to(root)): _sha256(path) for path in loader_files
        },
        "config": str(config),
        "config_sha256": _sha256(config),
        "latest_pointer": str(pointer_path),
        "latest_pointer_target": str(pointer),
        "complete": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--latest-pointer", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        result = validate_checkpoint(args.checkpoint, latest_pointer=args.latest_pointer)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    except (OSError, ValueError) as exc:
        print(f"cfw_dcp: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
