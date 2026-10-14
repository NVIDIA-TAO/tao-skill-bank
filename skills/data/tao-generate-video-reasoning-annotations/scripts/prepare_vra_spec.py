#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a video_reasoning_annotation spec by editing the real bundled template.

The video_reasoning_annotation pipeline ships its experiment specs inside the
data-services container, not in this repo. This helper pulls the named spec
straight off the image with `docker create` + `docker cp` (no entrypoint runs,
so no startup-banner text leaks into the file), applies a small set of
dotted-path overrides in place, and prints a diff against the untouched
original so every deviation from the bundled template is visible before
launch. It never rewrites the file with a YAML dump, so the bundled file's
comments and rationale survive unchanged in every field you didn't override.

Requires the target image to already be present locally; this script never
pulls an image.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SPECS_DIR_IN_IMAGE = (
    "import nvidia_tao_ds, os; "
    "print(os.path.join(os.path.dirname(nvidia_tao_ds.__file__), "
    "'auto_label/experiment_specs'))"
)

KEY_LINE_RE = re.compile(r"^( *)([A-Za-z0-9_]+):(.*)$")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", required=True, help="Local image tag/id (never pulled by this script)."
    )
    parser.add_argument(
        "--spec-name",
        required=True,
        help="Bundled spec filename, e.g. video_reasoning_annotation_public_safety_openai.yaml",
    )
    parser.add_argument("--output", required=True, type=Path, help="Path to write the edited spec.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="DOTTED.KEY=JSON_VALUE",
        help=(
            "Override one field, addressed by its dotted YAML path from the file root "
            "(e.g. video_reasoning_annotation.workflow.use_folder_floor=false). "
            "The value is parsed as JSON, so strings need quotes: model_name='\"foo\"'. "
            "Repeatable."
        ),
    )
    parser.add_argument(
        "--overrides-file",
        type=Path,
        help="JSON file mapping dotted-path -> value, merged with --set (--set wins on conflicts).",
    )
    parser.add_argument(
        "--pristine-out",
        type=Path,
        help="Optional path to also save the untouched bundled spec, for audit trail.",
    )
    parser.add_argument(
        "--no-diff",
        action="store_true",
        help="Suppress the diff printed against the untouched bundled spec.",
    )
    return parser.parse_args()


def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    """Run a subprocess, raising with captured output on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr}"
        )
    return result


def require_local_image(image: str) -> None:
    """Fail fast rather than let `docker create` silently pull a missing image."""
    result = subprocess.run(
        ["docker", "image", "inspect", image], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"image {image!r} is not present locally; this script never pulls. "
            f"Pull it yourself first if that's intended."
        )


def resolve_specs_dir(image: str) -> str:
    """Ask the image where its bundled experiment_specs directory lives."""
    result = run(["docker", "run", "--rm", "--entrypoint", "python3", image, "-c", SPECS_DIR_IN_IMAGE])
    return result.stdout.strip()


def extract_spec(image: str, spec_path_in_image: str) -> str:
    """Copy one file out of the image via `docker create` + `docker cp` (no entrypoint run)."""
    container_id = run(["docker", "create", image]).stdout.strip()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "spec.yaml"
            run(["docker", "cp", f"{container_id}:{spec_path_in_image}", str(dest)])
            return dest.read_text()
    finally:
        subprocess.run(["docker", "rm", container_id], capture_output=True, text=True)


def serialize_value(value: Any) -> str:
    """Render a Python value as the scalar/flow-list text used in these bundled specs."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(serialize_value(v) for v in value) + "]"
    return str(value)


def load_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Merge --overrides-file with --set (--set takes precedence)."""
    overrides: dict[str, Any] = {}
    if args.overrides_file:
        overrides.update(json.loads(args.overrides_file.read_text()))
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"--set must be DOTTED.KEY=JSON_VALUE, got: {item!r}")
        key, raw_value = item.split("=", 1)
        try:
            overrides[key] = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"--set {key}: value must be valid JSON (quote strings), got {raw_value!r}"
            ) from exc
    return overrides


def strip_comment(value: str) -> str:
    """Drop a trailing `# comment` that sits outside any quoted string."""
    quote = None
    for index, char in enumerate(value):
        if quote:
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char == "#" and (index == 0 or value[index - 1].isspace()):
            return value[:index]
    return value


def bracket_depth(value: str) -> int:
    """Net count of open `[`/`{` outside quoted strings."""
    depth = 0
    quote = None
    for char in strip_comment(value):
        if quote:
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
    return depth


def value_extent(lines: list[str], start: int, indent: int, rest: str) -> tuple[int, str]:
    """Return (index past the key's value, kind) for the key on lines[start].

    kind is "scalar", "flow" (a flow list/mapping spanning several lines),
    "sequence" (a block list of `- item` lines), or "mapping" (a nested block).
    """
    value = strip_comment(rest).strip()
    end = start + 1
    if value:
        if value[0] in "[{" and bracket_depth(value) > 0:
            depth = bracket_depth(value)
            while end < len(lines) and depth > 0:
                depth += bracket_depth(lines[end])
                end += 1
            return end, "flow"
        return end, "scalar"
    first_child = None
    last_child = start
    while end < len(lines):
        line = lines[end]
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            if len(line) - len(line.lstrip(" ")) <= indent and not stripped.startswith("- "):
                break
            if first_child is None:
                first_child = stripped
            last_child = end
        end += 1
    kind = "sequence" if first_child and first_child.startswith("- ") else "mapping"
    # Stop after the last child so trailing comments of the next field are kept.
    return last_child + 1, kind


def apply_overrides(text: str, overrides: dict[str, Any]) -> tuple[str, set[str]]:
    """Rewrite only the value of each matched dotted-path key; leave everything else byte-identical.

    Tracks nesting purely by indentation, matching this file family's simple
    2-space block-mapping style. A value spanning several lines (a wrapped flow
    list or a block `- item` list) is replaced in full. Overriding a nested
    mapping is refused, since it would silently delete every key beneath it.
    Anchors (&name), merge keys (<<:) and aliases (*name) are left untouched.
    """
    lines = text.splitlines()
    stack: list[tuple[int, str]] = []
    applied: set[str] = set()
    out_lines: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        match = KEY_LINE_RE.match(line)
        if match:
            indent = len(match.group(1))
            key = match.group(2)
            rest = match.group(3)
            while stack and stack[-1][0] >= indent:
                stack.pop()
            dotted = ".".join([k for _, k in stack] + [key])
            stack.append((indent, key))
            if dotted in overrides:
                end, kind = value_extent(lines, index, indent, rest)
                if kind == "mapping":
                    raise ValueError(
                        f"{dotted} is a nested mapping; override its individual fields instead"
                    )
                new_value = serialize_value(overrides[dotted])
                old_value = strip_comment(rest).strip() if kind == "scalar" else f"<multi-line {kind}>"
                out_lines.append(
                    f"{' ' * indent}{key}: {new_value}  # OVERRIDE (bundled default: {old_value})"
                )
                applied.add(dotted)
                index = end
                continue
        out_lines.append(line)
        index += 1
    return "\n".join(out_lines) + "\n", applied


def verify_edited(edited: str, overrides: dict[str, Any]) -> None:
    """Parse the edited spec and check every override resolved to the requested value."""
    try:
        import yaml
    except ImportError:
        print("WARNING: PyYAML not installed; edited spec was not parse-checked.", file=sys.stderr)
        return
    try:
        document = yaml.safe_load(edited)
    except yaml.YAMLError as exc:
        raise ValueError(f"edited spec is not valid YAML: {exc}") from exc
    for dotted, expected in overrides.items():
        node: Any = document
        try:
            for part in dotted.split("."):
                node = node[part]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{dotted} not found in the edited spec") from exc
        if node != expected:
            raise ValueError(f"{dotted} parsed as {node!r}, expected {expected!r}")


def main() -> int:
    args = parse_args()
    overrides = load_overrides(args)

    require_local_image(args.image)
    specs_dir = resolve_specs_dir(args.image)
    spec_path_in_image = f"{specs_dir}/{args.spec_name}"
    pristine = extract_spec(args.image, spec_path_in_image)

    try:
        edited, applied = apply_overrides(pristine, overrides)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    missing = set(overrides) - applied
    if missing:
        print(
            "ERROR: these override keys never matched a line in the bundled spec "
            f"(typo, or the key doesn't exist in this spec): {sorted(missing)}",
            file=sys.stderr,
        )
        return 1

    try:
        verify_edited(edited, overrides)
    except ValueError as exc:
        print(f"ERROR: {exc}; nothing written.", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(edited)
    print(f"Wrote {args.output} ({len(applied)} field(s) overridden from {args.spec_name})")

    if args.pristine_out:
        args.pristine_out.parent.mkdir(parents=True, exist_ok=True)
        args.pristine_out.write_text(pristine)
        print(f"Saved untouched bundled spec to {args.pristine_out}")

    if not args.no_diff:
        diff = difflib.unified_diff(
            pristine.splitlines(keepends=True),
            edited.splitlines(keepends=True),
            fromfile=f"bundled:{args.spec_name}",
            tofile=str(args.output),
        )
        sys.stdout.writelines(diff)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
