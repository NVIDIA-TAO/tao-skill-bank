#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Migrate `container_image` literals in `references/skill_info.yaml` to dotted keys.

Walks every `references/skill_info.yaml`. For each `container_image` whose
value is an absolute path (e.g., `nvcr.io/nvidia/tao/tao-toolkit:6.26.3-pyt`):

  - Look up the string in `versions.yaml` images tree.
  - If found → replace the value with the matching key (e.g., `tao_toolkit.pyt`).
  - If NOT found → leave the value as-is. The author decides whether to
    promote it to a manifest entry; we don't auto-add to `versions.yaml`.

Idempotent — values already in dotted-key form are left alone.

Usage:
    python3 scripts/migrate-to-version-keys.py             # dry-run, prints actions
    python3 scripts/migrate-to-version-keys.py --apply     # writes changes
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


def load_image_lookup(versions_path: Path) -> dict[str, str]:
    """Flatten the manifest's `images` tree into a {full_uri: dotted_key} map."""
    with open(versions_path) as f:
        manifest = yaml.safe_load(f) or {}
    out: dict[str, str] = {}

    def walk(node: dict, prefix: list[str]) -> None:
        for k, v in node.items():
            path = prefix + [k]
            if isinstance(v, dict):
                walk(v, path)
            elif isinstance(v, str):
                out[v] = ".".join(path)

    images = manifest.get("images", {})
    if isinstance(images, dict):
        walk(images, [])
    return out


def looks_like_uri(value: str) -> bool:
    """True if value looks like an absolute container image URI."""
    return isinstance(value, str) and ("/" in value or ":" in value)


def migrate_one(skill_info: Path, lookup: dict[str, str], apply: bool) -> str | None:
    with open(skill_info) as f:
        info = yaml.safe_load(f) or {}
    if not isinstance(info, dict):
        return None
    img = info.get("container_image")
    if not isinstance(img, str):
        return None
    if not looks_like_uri(img):
        return None  # already a key reference
    key = lookup.get(img)
    if not key:
        return f"{skill_info}: kept absolute (not in versions.yaml): {img}"
    info["container_image"] = key
    if apply:
        with open(skill_info, "w") as f:
            yaml.safe_dump(info, f, sort_keys=False, default_flow_style=False, width=120, allow_unicode=True)
    return f"{skill_info}: {img} → {key}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Apply changes (otherwise dry-run)")
    args = ap.parse_args()

    repo_root = Path(__file__).parent.parent
    os.chdir(repo_root)

    versions_path = Path("versions.yaml")
    if not versions_path.is_file():
        print("ERROR: versions.yaml not found at repo root", file=sys.stderr)
        return 1

    lookup = load_image_lookup(versions_path)
    print(f"Loaded {len(lookup)} image entries from versions.yaml\n")

    skill_infos = sorted(Path(".").rglob("references/skill_info.yaml"))
    skill_infos = [p for p in skill_infos if "templates/" not in str(p)]
    if not skill_infos:
        print("No references/skill_info.yaml files found", file=sys.stderr)
        return 1

    migrated = []
    kept_absolute = []
    for skill_info in skill_infos:
        result = migrate_one(skill_info, lookup, apply=args.apply)
        if result is None:
            continue
        if "kept absolute" in result:
            kept_absolute.append(result)
        else:
            migrated.append(result)

    print(f"{'Applied' if args.apply else 'Would apply'} {len(migrated)} migrations:")
    for line in migrated:
        print(f"  {line}")

    if kept_absolute:
        print(f"\nKept absolute paths ({len(kept_absolute)} — author may promote to versions.yaml):")
        for line in kept_absolute:
            print(f"  {line}")

    if not args.apply:
        print("\nRe-run with --apply to write changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
