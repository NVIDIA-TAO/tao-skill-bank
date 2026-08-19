#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize NVPaw JSONL authoring records for Cosmos-RL and evaluation."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from nvpaw_annotations import load_source_records, materialize_records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=pathlib.Path)
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--prompt-variant", default="official_v1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest, sharegpt = materialize_records(
            load_source_records(args.source), prompt_variant=args.prompt_variant
        )
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest),
            encoding="utf-8",
        )
        args.output.write_text(
            json.dumps(sharegpt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"materialize_nvpaw_annotations: {exc}", file=sys.stderr)
        return 2
    print(
        f"materialize_nvpaw_annotations: wrote records={len(sharegpt)} "
        f"manifest={args.manifest} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
