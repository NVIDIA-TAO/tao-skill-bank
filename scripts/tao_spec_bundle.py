#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate a spec-bundle before handing it to a platform skill's submit verb.

The spec-bundle is the bank's producer -> consumer interface: "everything needed
to run one action, nothing platform-specific" (tao-artifacts). A producer
authors it; a platform skill runs it. It is the reason a workflow does not need
to know whether it is landing on docker, sbatch or kubectl.

`tao-launch-workflow` already tells submit to lint the assembled command with
`redact_secrets.py lint`. This is the same shape of check one level earlier:
lint the bundle before anything renders it, so a malformed bundle fails on the
laptop rather than three minutes into a GPU allocation.

Stdlib only, matching the rest of scripts/. When `jsonschema` happens to be
importable the full schema runs too, but the checks below stand alone -- and
they carry better messages for the mistakes that actually happen.

    tao_spec_bundle.py validate <bundle.json> [--schema PATH]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any

DEFAULT_SCHEMA = (
    pathlib.Path(__file__).resolve().parents[1]
    / "skills/core/tao-artifacts/references/spec_bundle.schema.json"
)

REQUIRED = (
    "network_arch", "action", "image", "mode", "command",
    "declared_inputs", "declared_outputs", "compute_shape",
)
IMAGE_URI = re.compile(r"^\S+/\S+$")
# `tao_toolkit.pyt` and friends: a versions.yaml key, not a resolved URI.
VERSIONS_KEY = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)+$")


def _dotted_keys(node: Any, trail: str = "spec") -> list[str]:
    """Every dotted key in the nested spec, deepest paths included.

    The schema calls this "the #1 mistake": a spec is nested dicts, so
    {"train": {"num_epochs": 12}} is right and {"train.num_epochs": 12} is not.
    A dotted key silently becomes a literal key the container never reads.
    """
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, str) and "." in key:
                found.append(f"{trail}.{key}")
            found += _dotted_keys(value, f"{trail}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found += _dotted_keys(value, f"{trail}[{index}]")
    return found


def _check_mode(bundle: dict[str, Any]) -> list[str]:
    mode = bundle.get("mode")
    errors: list[str] = []
    if mode == "config":
        for field in ("spec", "config_format"):
            if field not in bundle:
                errors.append(f"mode=config requires `{field}`")
        if "args" in bundle:
            errors.append("mode=config cannot carry `args`")
        command = bundle.get("command")
        if isinstance(command, str) and "{config_path}" not in command:
            errors.append(
                "mode=config requires `{config_path}` in `command` — the "
                "consumer substitutes the compute-frame path of the spec file "
                "it writes, so a command without it never reads the spec"
            )
        spec = bundle.get("spec")
        if spec is not None and not isinstance(spec, dict):
            errors.append("`spec` must be a nested dict")
        elif isinstance(spec, dict):
            for dotted in _dotted_keys(spec):
                errors.append(
                    f"dotted spec key `{dotted}` — specs are NESTED dicts; write "
                    '{"train": {"num_epochs": 12}}, not {"train.num_epochs": 12}'
                )
    elif mode == "args":
        if "args" not in bundle:
            errors.append("mode=args requires `args`")
        for field in ("spec", "config_format"):
            if field in bundle:
                errors.append(f"mode=args cannot carry `{field}`")
    elif mode is not None:
        errors.append(f"`mode` must be config or args, got {mode!r}")
    return errors


def _check_io(bundle: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for index, item in enumerate(bundle.get("declared_inputs") or []):
        where = f"declared_inputs[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{where} must be an object")
            continue
        for field in ("spec_key", "type", "uri"):
            if not item.get(field):
                errors.append(f"{where} is missing `{field}`")
        if item.get("type") not in (None, "file", "folder"):
            errors.append(f"{where}.type must be file or folder")
        if "storage_tier" in item:
            errors.append(
                f"{where} carries `storage_tier` — the tier is chosen by "
                "tao-data-io at stage time and stamped into the job-record, "
                "never declared by the producer"
            )
    outputs = bundle.get("declared_outputs")
    if isinstance(outputs, list) and not outputs:
        errors.append("`declared_outputs` needs at least one entry")
    for index, item in enumerate(outputs or []):
        where = f"declared_outputs[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{where} must be an object")
            continue
        for field in ("spec_key", "type"):
            if not item.get(field):
                errors.append(f"{where} is missing `{field}`")
    return errors


def _check_compute(bundle: dict[str, Any]) -> list[str]:
    shape = bundle.get("compute_shape")
    if not isinstance(shape, dict):
        return ["`compute_shape` must be an object with gpus and nodes"]
    errors: list[str] = []
    gpus, nodes = shape.get("gpus"), shape.get("nodes")
    if not isinstance(gpus, int) or isinstance(gpus, bool) or gpus < 0:
        errors.append("compute_shape.gpus must be an integer >= 0 (GPUs per node)")
    if not isinstance(nodes, int) or isinstance(nodes, bool) or nodes < 1:
        errors.append("compute_shape.nodes must be an integer >= 1")
    return errors


def validate(bundle: dict[str, Any], schema_path: pathlib.Path | None = None) -> list[str]:
    """Return a list of human-readable problems; empty means valid."""
    if not isinstance(bundle, dict):
        return ["bundle must be a JSON object"]

    errors = [f"missing required field `{f}`" for f in REQUIRED if f not in bundle]

    image = bundle.get("image")
    if isinstance(image, str):
        if VERSIONS_KEY.match(image):
            errors.append(
                f"`image` is the versions.yaml key {image!r}, not a resolved URI — "
                "resolve it first with scripts/resolve_tao_image.py"
            )
        elif not IMAGE_URI.match(image):
            errors.append(f"`image` must be a resolved container URI, got {image!r}")

    errors += _check_mode(bundle)
    errors += _check_io(bundle)
    errors += _check_compute(bundle)

    schema_file = schema_path or DEFAULT_SCHEMA
    try:
        import jsonschema
    except ImportError:
        return errors
    if schema_file.is_file():
        validator = jsonschema.Draft202012Validator(
            json.loads(schema_file.read_text(encoding="utf-8"))
        )
        for problem in validator.iter_errors(bundle):
            location = "/".join(str(p) for p in problem.absolute_path) or "<root>"
            rendered = f"schema: {location}: {problem.message}"
            if rendered not in errors:
                errors.append(rendered)
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("validate", help="validate a spec-bundle JSON file")
    check.add_argument("bundle", type=pathlib.Path)
    check.add_argument("--schema", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    try:
        payload = json.loads(args.bundle.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"tao_spec_bundle: cannot read {args.bundle}: {exc}", file=sys.stderr)
        return 2

    problems = validate(payload, args.schema)
    if problems:
        print(f"tao_spec_bundle: {args.bundle} is not a valid spec-bundle", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"OK: {args.bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
