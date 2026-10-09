#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report PAS credential presence without ever reading values into output."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass


NAME_PATTERN = re.compile(r"[A-Z][A-Z0-9_]*")


@dataclass(frozen=True)
class CredentialPresence:
    """One value-free environment presence result."""

    name: str
    requirement: str
    status: str


def check(required: list[str], optional: list[str]) -> tuple[dict, bool]:
    """Build a deterministic presence-only report and its success verdict."""
    names = [*required, *optional]
    invalid = [name for name in names if NAME_PATTERN.fullmatch(name) is None]
    if invalid:
        raise ValueError("credential names must use uppercase environment syntax")
    if len(names) != len(set(names)):
        raise ValueError("credential names must be unique across required and optional")
    entries = [
        CredentialPresence(
            name=name,
            requirement="required" if name in required else "optional",
            status="set" if bool(os.environ.get(name)) else "missing",
        )
        for name in names
    ]
    missing = [
        entry.name
        for entry in entries
        if entry.requirement == "required" and entry.status == "missing"
    ]
    return (
        {
            "schema_version": "1",
            "credentials": [asdict(entry) for entry in entries],
            "missing_required": missing,
        },
        not missing,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--required", action="append", default=[], metavar="ENV_NAME")
    parser.add_argument("--optional", action="append", default=[], metavar="ENV_NAME")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if not args.required and not args.optional:
            raise ValueError("at least one --required or --optional name is required")
        report, ok = check(args.required, args.optional)
    except ValueError as exc:
        print(f"check_pas_credentials: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
