#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Validate canonical NVPAW multi-task JSONL for Cosmos Framework.

The filename is retained for CLI compatibility, but JSON arrays and converted
ShareGPT records are deliberately rejected. The runtime boundary is one JSON
object per line with native ``messages`` and sealed image pixel bounds.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter
from typing import Any

from nvpaw_annotations import TASK_SPECS


def load_records(path: pathlib.Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            records.append(value)
    if not records:
        raise ValueError(f"{path}: expected non-empty JSONL")
    return records


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        item["text"]
        for item in content
        if isinstance(item, dict) and isinstance(item.get("text"), str)
    )


def prompt_and_response(record: dict[str, Any], *, context: str) -> tuple[str, str]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{context}: messages must be a list")
    prompts: list[str] = []
    answers: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(f"{context}: every message must be an object")
        text = _content_text(message.get("content")).strip()
        if message.get("role") == "user" and text:
            prompts.append(text)
        elif message.get("role") == "assistant" and text:
            answers.append(text)
    if not prompts:
        raise ValueError(f"{context}: no non-empty user prompt")
    if len(answers) != 1:
        raise ValueError(f"{context}: exactly one non-empty assistant answer is required")
    return "\n".join(prompts), answers[0]


def image_items(record: dict[str, Any], *, context: str) -> list[dict[str, Any]]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{context}: messages must be a list")
    result: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") == "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                result.append(item)
    return result


def image_paths(record: dict[str, Any], *, context: str) -> list[str]:
    result: list[str] = []
    for item in image_items(record, context=context):
        path = item.get("image")
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"{context}: image path must be a non-empty string")
        result.append(path)
    return result


def target_path(record: dict[str, Any], *, context: str) -> str:
    task_type = record.get("task_type")
    if task_type not in TASK_SPECS:
        raise ValueError(f"{context}: unsupported task_type {task_type!r}")
    paths = image_paths(record, context=context)
    roles = TASK_SPECS[task_type]["image_roles"]
    if len(paths) != len(roles):
        raise ValueError(f"{context}: {task_type} requires image roles {list(roles)}")
    return paths[list(roles).index("target")]


def resolve_image(path_text: str, media_root: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path_text).expanduser()
    return (path if path.is_absolute() else media_root / path).resolve()


def validate_records(
    records: list[dict[str, Any]],
    *,
    media_root: pathlib.Path,
    require_files: bool,
    require_id: bool = True,
    annotation_profile: str = "nvpaw_multitask_v1",
    skip_unsupported_tasks: bool = False,
) -> dict[str, Any]:
    if annotation_profile != "nvpaw_multitask_v1":
        raise ValueError("Cosmos Framework accepts only nvpaw_multitask_v1 JSONL")
    ids: set[str] = set()
    tasks: Counter[str] = Counter()
    targets: set[str] = set()
    image_count = 0
    max_images = 0
    unsupported_tasks: Counter[str] = Counter()
    supported_records = 0
    verified_files: set[pathlib.Path] = set()
    for index, record in enumerate(records):
        context = f"record[{index}]"
        task_type = record.get("task_type")
        if task_type not in TASK_SPECS:
            if not skip_unsupported_tasks:
                raise ValueError(f"{context}: unsupported task_type {task_type!r}")
            unsupported_tasks[str(task_type)] += 1
            continue
        record_id = record.get("id")
        if require_id or record_id is not None:
            if (
                not isinstance(record_id, str)
                or not record_id
                or record_id != record_id.strip()
                or any(ord(character) < 32 or ord(character) == 127 for character in record_id)
            ):
                raise ValueError(
                    f"{context}: id must be non-empty, trimmed, and contain no control characters"
                )
            if record_id in ids:
                raise ValueError(f"{context}: duplicate id {record_id!r}")
            ids.add(record_id)
        prompt_and_response(record, context=context)
        items = image_items(record, context=context)
        roles = TASK_SPECS[task_type]["image_roles"]
        if len(items) != len(roles):
            raise ValueError(f"{context}: {task_type} requires image roles {list(roles)}")
        for item in items:
            low = item.get("min_pixels")
            high = item.get("max_pixels")
            if type(low) is not int or type(high) is not int or low <= 0 or low > high:
                raise ValueError(
                    f"{context}: every image requires integer 0 < min_pixels <= max_pixels"
                )
            path = item.get("image")
            if not isinstance(path, str) or not path:
                raise ValueError(f"{context}: image path must be non-empty")
            resolved = resolve_image(path, media_root)
            if require_files and resolved not in verified_files:
                if not resolved.is_file():
                    raise ValueError(f"{context}: missing image file: {resolved}")
                verified_files.add(resolved)
        target = resolve_image(target_path(record, context=context), media_root)
        targets.add(str(target))
        tasks[str(task_type)] += 1
        image_count += len(items)
        max_images = max(max_images, len(items))
        supported_records += 1
    if not supported_records:
        raise ValueError("annotations contain no supported six-task DEFT records")
    return {
        "mode": annotation_profile,
        "format": "jsonl",
        "records": supported_records,
        "records_total": len(records),
        "unsupported_tasks": dict(sorted(unsupported_tasks.items())),
        "tasks": dict(sorted(tasks.items())),
        "labels": {},
        "unique_ids": len(ids),
        "unique_target_images": len(targets),
        "image_items": image_count,
        "max_images_per_record": max_images,
        "require_files": require_files,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=pathlib.Path)
    parser.add_argument("--media-root", required=True, type=pathlib.Path)
    parser.add_argument("--require-files", action="store_true")
    parser.add_argument("--require-id", action="store_true")
    parser.add_argument("--summary", type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        summary = validate_records(
            load_records(args.annotations),
            media_root=args.media_root.expanduser().resolve(),
            require_files=args.require_files,
            require_id=True,
        )
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"validate_sharegpt: {exc}", file=sys.stderr)
        return 2
    print(
        f"validate_sharegpt: OK format=jsonl records={summary['records']} "
        f"image_items={summary['image_items']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
