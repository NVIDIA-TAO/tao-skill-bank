#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Require a new plugin cache key whenever shipped PAS skill content changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


PAS_PREFIX = "skills/applications/tao-run-deft-pas/"
MANIFESTS = (
    ".claude-plugin/marketplace.json",
    ".claude-plugin/plugin.json",
    ".codex-plugin/plugin.json",
)
SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:[-+][0-9A-Za-z.-]+)?$")


def parse_version(value: str) -> tuple[int, int, int]:
    match = SEMVER.fullmatch(value)
    if match is None:
        raise ValueError(f"plugin version is not semantic x.y.z: {value!r}")
    return tuple(int(part) for part in match.groups())


def _version(document: dict, path: str) -> str:
    value = (
        document.get("metadata", {}).get("version")
        if path.endswith("marketplace.json")
        else document.get("version")
    )
    if not isinstance(value, str):
        raise ValueError(f"{path}: missing string plugin version")
    parse_version(value)
    return value


def current_versions(root: Path) -> dict[str, str]:
    return {
        path: _version(json.loads((root / path).read_text(encoding="utf-8")), path)
        for path in MANIFESTS
    }


def base_versions(root: Path, base_ref: str) -> dict[str, str]:
    versions: dict[str, str] = {}
    for path in MANIFESTS:
        completed = subprocess.run(
            ["git", "show", f"{base_ref}:{path}"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(f"cannot read {path} at base ref {base_ref!r}")
        versions[path] = _version(json.loads(completed.stdout), path)
    return versions


def changed_pas_files(root: Path, base_ref: str) -> list[str]:
    commands = (
        ["git", "diff", "--name-only", f"{base_ref}...HEAD", "--", PAS_PREFIX],
        ["git", "diff", "--name-only", "HEAD", "--", PAS_PREFIX],
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            PAS_PREFIX,
        ],
    )
    changed: set[str] = set()
    for command in commands:
        completed = subprocess.run(
            command, cwd=root, check=True, capture_output=True, text=True
        )
        changed.update(line for line in completed.stdout.splitlines() if line)
    return sorted(changed)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve()
    try:
        changed = changed_pas_files(root, args.base_ref)
        current = current_versions(root)
        if len(set(current.values())) != 1:
            raise ValueError(f"current plugin manifest versions disagree: {current}")
        if not changed:
            print("PAS plugin version check: no shipped PAS skill changes")
            return 0
        base = base_versions(root, args.base_ref)
        if len(set(base.values())) != 1:
            raise ValueError(f"base plugin manifest versions disagree: {base}")
        current_value = next(iter(current.values()))
        base_value = next(iter(base.values()))
        if parse_version(current_value) <= parse_version(base_value):
            sample = ", ".join(changed[:5])
            raise ValueError(
                "shipped PAS skill content changed without increasing the synchronized "
                f"plugin version above {base_value}; changed: {sample}"
            )
    except (json.JSONDecodeError, OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"PAS plugin version check failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"PAS plugin version check: {base_value} -> {current_value} "
        f"for {len(changed)} changed PAS path(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
