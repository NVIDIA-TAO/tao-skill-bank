#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Set values in a spec emitted by ``emit_default_spec.py``.

Completes the pair: ``emit_default_spec.py`` produces the canonical spec with
``???`` on mandatory fields, this fills them, and no spec is ever hand-edited.

    --set dataset.batch_size=8 --set inference.checkpoint=/abs/model.pth

Keys are dotted paths into nested mappings. Values are parsed as YAML scalars, so
``8`` is an int, ``true`` a bool, ``null`` None, ``[a, b]`` a list, and anything
else a string — matching how the same text would behave written into the file.

Creating a key that is not already present requires ``--allow-new``. Without it a
typo'd path is an error rather than a silently added field the loader ignores,
which is the failure mode this script exists to prevent.

``--require-no-mandatory`` exits non-zero if any ``???`` remains, so a stage can
refuse to launch a container against an incomplete spec.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

MANDATORY = "???"


def parse_value(raw: str):
    """Parse a scalar the way PyYAML would if it were written into the file."""
    return yaml.safe_load(raw)


def set_path(tree: dict, dotted: str, value, allow_new: bool,
             materialised: set[str]) -> str:
    """Set tree[a][b][c] = value. Returns a repr of what was there before.

    ``materialised`` accumulates the dotted paths of blocks this invocation turned
    from ``null`` into a mapping, and is shared across every ``--set``.

    A block the schema declares ``null`` -- ``dataset.infer_data_sources`` is one --
    has no declared members, so no member key can be checked against it. Requiring
    the leaf to pre-exist there would reject every key the caller came to add, and
    checking only the first would reject the second. Once a block is materialised
    it stays open for the rest of the call; blocks the schema really defines still
    reject unknown keys.
    """
    parts = dotted.split(".")
    node = tree
    for i, key in enumerate(parts[:-1]):
        path = ".".join(parts[: i + 1])
        if not isinstance(node, dict) or key not in node:
            if not allow_new:
                raise KeyError(f"{path!r} is not in the spec (pass --allow-new to create it)")
            node[key] = {}
            materialised.add(path)
        elif node[key] is None:
            node[key] = {}
            materialised.add(path)
        node = node[key]

    leaf = parts[-1]
    parent = ".".join(parts[:-1])
    if not isinstance(node, dict):
        raise KeyError(f"{parent!r} is not a mapping")
    if leaf not in node and not allow_new and parent not in materialised:
        raise KeyError(f"{dotted!r} is not in the spec (pass --allow-new to create it)")
    previous = node.get(leaf, "<absent>")
    node[leaf] = value
    return repr(previous)


def remaining_mandatory(tree, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(tree, dict):
        for k, v in tree.items():
            found += remaining_mandatory(v, f"{prefix}{k}.")
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            found += remaining_mandatory(v, f"{prefix}{i}.")
    elif tree == MANDATORY:
        found.append(prefix.rstrip("."))
    return found


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True, help="Spec to modify.")
    parser.add_argument("--out", default=None, help="Where to write. Defaults to --spec.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Dotted key and YAML-parsed value. Repeatable.")
    parser.add_argument("--allow-new", action="store_true",
                        help="Permit creating keys absent from the spec.")
    parser.add_argument("--require-no-mandatory", action="store_true",
                        help="Exit non-zero if any ??? remains after the overrides.")
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        if yaml is None:
            raise RuntimeError("PyYAML is required; run through scripts/deft_python.sh.")

        spec_path = Path(args.spec).expanduser().resolve()
        if not spec_path.is_file():
            raise FileNotFoundError(f"--spec does not exist: {spec_path}")
        tree = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
        if not isinstance(tree, dict):
            raise ValueError(f"{spec_path}: expected a mapping at the top level")

        # Collected rather than printed as we go: nothing is written until every
        # override succeeds, so reporting one as applied before then is a lie if a
        # later one raises.
        applied: list[str] = []
        materialised: set[str] = set()
        for item in args.set:
            if "=" not in item:
                raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
            dotted, raw = item.split("=", 1)
            try:
                previous = set_path(tree, dotted.strip(), parse_value(raw),
                                    args.allow_new, materialised)
            except KeyError as exc:
                raise KeyError(
                    f"{exc.args[0]} — no changes were written; the spec is unmodified"
                ) from exc
            applied.append(f"  {dotted.strip()}: {previous} -> {raw}")

        out = Path(args.out).expanduser().resolve() if args.out else spec_path
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(tree, fh, sort_keys=False, default_flow_style=False)
        for line in applied:
            print(line)
        print(f"spec -> {out}")

        left = remaining_mandatory(tree)
        if left:
            print(f"  {len(left)} field(s) still ???: {', '.join(left[:8])}",
                  file=sys.stderr)
            if args.require_no_mandatory:
                return 1
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
