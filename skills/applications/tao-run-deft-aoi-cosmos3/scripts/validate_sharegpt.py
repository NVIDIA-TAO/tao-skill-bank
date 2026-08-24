#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Validate legacy or NVPaw multi-task ShareGPT annotations for Cosmos3 AOI."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter
from typing import Any

from nvpaw_annotations import TASK_SPECS, validate_bbox


VALID_LABELS = {"OK", "NG"}
HUMAN_ROLES = {"human", "user"}
ASSISTANT_ROLES = {"gpt", "assistant"}
_SAFE_ID = re.compile(r"[A-Za-z0-9._@-]+")


def load_records(path: pathlib.Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path}: expected one JSON array; JSONL is not supported: {exc.msg}"
        ) from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{path}: expected a non-empty JSON array")
    if not all(isinstance(record, dict) for record in payload):
        raise ValueError(f"{path}: every record must be an object")
    return payload


def _turn_role(turn: dict[str, Any]) -> Any:
    return turn.get("from", turn.get("role"))


def _turn_value(turn: dict[str, Any]) -> Any:
    return turn.get("value", turn.get("content"))


def prompt_and_label(record: dict[str, Any], *, context: str) -> tuple[str, str]:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError(f"{context}: conversations must be a list")
    prompt: str | None = None
    label: str | None = None
    for turn in conversations:
        if not isinstance(turn, dict):
            raise ValueError(f"{context}: every conversation turn must be an object")
        role = _turn_role(turn)
        value = _turn_value(turn)
        if role in HUMAN_ROLES and prompt is None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{context}: human prompt must be non-empty")
            prompt = value.strip()
        if role in ASSISTANT_ROLES:
            if not isinstance(value, str):
                raise ValueError(f"{context}: assistant response must be a string")
            label = value.strip().upper()
    if prompt is None:
        raise ValueError(f"{context}: missing human prompt")
    if label not in VALID_LABELS:
        raise ValueError(
            f"{context}: bare mode requires the final assistant response to be "
            f"exactly OK or NG, got {label!r}"
        )
    return prompt, label


def prompt_and_response(record: dict[str, Any], *, context: str) -> tuple[str, str]:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError(f"{context}: conversations must be a list")
    prompt: str | None = None
    response: str | None = None
    for turn in conversations:
        if not isinstance(turn, dict):
            raise ValueError(f"{context}: every conversation turn must be an object")
        role = _turn_role(turn)
        value = _turn_value(turn)
        if role in HUMAN_ROLES and prompt is None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{context}: human prompt must be non-empty")
            prompt = value.strip()
        elif role in ASSISTANT_ROLES:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{context}: assistant response must be non-empty")
            response = value.strip()
    if prompt is None:
        raise ValueError(f"{context}: missing human prompt")
    if response is None:
        raise ValueError(f"{context}: missing assistant response")
    return prompt, response


def resolve_image(path_text: str, media_root: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path_text).expanduser()
    if not path.is_absolute():
        path = media_root / path
    return path.resolve()


def validate_records(
    records: list[dict[str, Any]],
    *,
    media_root: pathlib.Path,
    require_files: bool,
    require_id: bool = False,
    annotation_profile: str = "bare_okng",
) -> dict[str, Any]:
    if annotation_profile not in {"bare_okng", "nvpaw_multitask_v1"}:
        raise ValueError(f"unsupported annotation profile {annotation_profile!r}")
    if annotation_profile == "nvpaw_multitask_v1":
        return _validate_nvpaw_records(
            records,
            media_root=media_root,
            require_files=require_files,
            require_id=require_id,
        )
    labels: Counter[str] = Counter()
    targets: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    duplicate_targets: list[str] = []
    duplicate_pairs: list[list[str]] = []
    ids: set[str] = set()

    for index, record in enumerate(records):
        context = f"record[{index}]"
        images = record.get("images")
        if not isinstance(images, list) or len(images) != 2:
            raise ValueError(
                f"{context}: images must contain exactly [AOI, golden_reference]"
            )
        if not all(isinstance(image, str) and image.strip() for image in images):
            raise ValueError(f"{context}: image paths must be non-empty strings")
        resolved = tuple(
            str(resolve_image(image, media_root)) for image in images
        )
        if require_files:
            missing = [path for path in resolved if not pathlib.Path(path).is_file()]
            if missing:
                raise ValueError(f"{context}: missing image file(s): {missing}")
        if resolved[0] in targets:
            duplicate_targets.append(resolved[0])
        targets.add(resolved[0])
        if resolved in pairs:
            duplicate_pairs.append(list(resolved))
        pairs.add(resolved)
        # cosmos-rl-evaluate hard-indexes item["id"] and reuses it as the
        # per-sample output filename, so evaluation splits need a unique,
        # filesystem-safe id. Training never reads it.
        record_id = record.get("id")
        if require_id or record_id is not None:
            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError(
                    f"{context}: id must be a non-empty string for evaluation "
                    "annotations"
                )
            if not _SAFE_ID.fullmatch(record_id):
                raise ValueError(
                    f"{context}: id must be filesystem-safe (letters, digits, "
                    f"dot, dash, underscore, @); got {record_id!r}"
                )
            if record_id in ids:
                raise ValueError(f"{context}: duplicate id {record_id!r}")
            ids.add(record_id)
        _, label = prompt_and_label(record, context=context)
        labels[label] += 1
        video_fps = record.get("video_fps")
        if video_fps is not None and (
            not isinstance(video_fps, (int, float))
            or isinstance(video_fps, bool)
            or video_fps <= 0
        ):
            raise ValueError(f"{context}: video_fps must be positive when present")

    if duplicate_targets:
        raise ValueError(
            "target AOI images must be unique; duplicates="
            f"{sorted(set(duplicate_targets))[:5]}"
        )
    if duplicate_pairs:
        raise ValueError(
            f"image pairs must be unique; duplicates={duplicate_pairs[:5]}"
        )
    return {
        "mode": "bare_okng",
        "records": len(records),
        "labels": dict(sorted(labels.items())),
        "unique_target_images": len(targets),
        "require_files": require_files,
        "unique_ids": len(ids),
        "max_images_per_record": 2,
    }


def _validate_nvpaw_records(
    records: list[dict[str, Any]],
    *,
    media_root: pathlib.Path,
    require_files: bool,
    require_id: bool,
) -> dict[str, Any]:
    tasks: Counter[str] = Counter()
    answer_kinds: Counter[str] = Counter()
    targets: set[str] = set()
    ids: set[str] = set()
    source_ids: set[str] = set()
    target_paths_by_id: dict[str, str] = {}
    target_ids_by_path: dict[str, str] = {}
    max_images = 0
    for index, record in enumerate(records):
        context = f"record[{index}]"
        if record.get("schema_version") not in (None, "nvpaw_multitask_v1"):
            raise ValueError(f"{context}: unsupported schema_version")
        task_type = record.get("task_type")
        if task_type not in TASK_SPECS:
            raise ValueError(f"{context}: unsupported task_type {task_type!r}")
        spec = TASK_SPECS[task_type]
        images = record.get("images")
        roles = record.get("image_roles")
        expected_roles = list(spec["image_roles"])
        if roles != expected_roles:
            raise ValueError(
                f"{context}: image_roles must be {expected_roles}, got {roles!r}"
            )
        if not isinstance(images, list) or len(images) != len(expected_roles):
            raise ValueError(
                f"{context}: images must match image_roles {expected_roles}"
            )
        if not all(isinstance(image, str) and image.strip() for image in images):
            raise ValueError(f"{context}: image paths must be non-empty strings")
        resolved = [str(resolve_image(image, media_root)) for image in images]
        if require_files:
            missing = [path for path in resolved if not pathlib.Path(path).is_file()]
            if missing:
                raise ValueError(f"{context}: missing image file(s): {missing}")
        target_index = expected_roles.index("target")
        targets.add(resolved[target_index])
        max_images = max(max_images, len(images))

        record_id = record.get("id")
        if require_id or record_id is not None:
            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError(f"{context}: id must be a non-empty string")
            if not _SAFE_ID.fullmatch(record_id):
                raise ValueError(
                    f"{context}: id must be filesystem-safe (letters, digits, "
                    f"dot, dash, underscore, @); got {record_id!r}"
                )
            if record_id in ids:
                raise ValueError(f"{context}: duplicate id {record_id!r}")
            ids.add(record_id)
        source_id = record.get("source_id")
        if source_id is not None:
            if not isinstance(source_id, str) or not source_id.strip():
                raise ValueError(f"{context}: source_id must be non-empty")
            if source_id in source_ids:
                raise ValueError(f"{context}: duplicate source_id {source_id!r}")
            source_ids.add(source_id)

        target_id = record.get("target_id")
        if not isinstance(target_id, str) or not target_id.strip():
            raise ValueError(f"{context}: target_id must be a non-empty string")
        target_path = resolved[target_index]
        previous_path = target_paths_by_id.setdefault(target_id, target_path)
        if previous_path != target_path:
            raise ValueError(
                f"{context}: target_id {target_id!r} maps to multiple target paths"
            )
        previous_id = target_ids_by_path.setdefault(target_path, target_id)
        if previous_id != target_id:
            raise ValueError(
                f"{context}: target path maps to multiple target_id values"
            )
        if record.get("metric_family") != spec["metric_family"]:
            raise ValueError(
                f"{context}: metric_family must be {spec['metric_family']!r}"
            )
        if record.get("reference_cohort") != spec["reference_cohort"]:
            raise ValueError(
                f"{context}: reference_cohort must be {spec['reference_cohort']!r}"
            )
        if record.get("prompt_format") != spec["prompt_format"]:
            raise ValueError(
                f"{context}: prompt_format must be {spec['prompt_format']!r}"
            )
        if record.get("prompt_variant") != "official_v1":
            raise ValueError(f"{context}: prompt_variant must be 'official_v1'")
        prompt_and_response(record, context=context)
        answer = record.get("answer")
        if not isinstance(answer, dict):
            raise ValueError(f"{context}: canonical answer must be an object")
        kind = answer.get("kind")
        expected_kind = {
            "classification": "choice_set",
            "counting": "count",
            "detection": "detections",
        }[spec["metric_family"]]
        if kind != expected_kind:
            raise ValueError(f"{context}: answer.kind must be {expected_kind!r}")
        if kind == "choice_set":
            labels = answer.get("labels")
            if not isinstance(labels, list) or not all(
                isinstance(label, str) and label.strip() for label in labels
            ):
                raise ValueError(f"{context}: answer.labels must be a string list")
            if len(labels) != len(set(labels)):
                raise ValueError(f"{context}: answer.labels must be unique")
            option_map = record.get("option_map", {})
            if not isinstance(option_map, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in option_map.items()
            ):
                raise ValueError(f"{context}: option_map must be a string map")
            if option_map and not set(labels).issubset(set(option_map.values())):
                raise ValueError(
                    f"{context}: answer.labels must resolve through option_map"
                )
        elif kind == "count":
            value = answer.get("value")
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"{context}: answer.value must be a non-negative integer"
                )
            if record.get("option_map", {}) != {}:
                raise ValueError(f"{context}: count records must have an empty option_map")
        else:
            objects = answer.get("objects")
            if not isinstance(objects, list):
                raise ValueError(f"{context}: answer.objects must be a list")
            for object_index, item in enumerate(objects):
                if not isinstance(item, dict):
                    raise ValueError(
                        f"{context}: detection object[{object_index}] must be an object"
                    )
                label = item.get("label")
                if not isinstance(label, str) or not label.strip():
                    raise ValueError(
                        f"{context}: detection object[{object_index}] label must be non-empty"
                    )
                validate_bbox(item.get("bbox_2d"), record_id=context)
        tasks[task_type] += 1
        answer_kinds[kind] += 1

    return {
        "mode": "nvpaw_multitask_v1",
        "records": len(records),
        "labels": dict(sorted(answer_kinds.items())),
        "tasks": dict(sorted(tasks.items())),
        "unique_target_images": len(targets),
        "require_files": require_files,
        "unique_ids": len(ids),
        "unique_source_ids": len(source_ids),
        "max_images_per_record": max_images,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=pathlib.Path)
    parser.add_argument("--media-root", required=True, type=pathlib.Path)
    parser.add_argument("--require-files", action="store_true")
    parser.add_argument(
        "--require-id",
        action="store_true",
        help="Require a unique, filesystem-safe id per record. Use for the "
        "Proxy and Benchmark splits: cosmos-rl-evaluate hard-indexes it.",
    )
    parser.add_argument("--summary", type=pathlib.Path)
    parser.add_argument(
        "--annotation-profile",
        choices=("bare_okng", "nvpaw_multitask_v1"),
        default="bare_okng",
    )
    args = parser.parse_args(argv)
    try:
        summary = validate_records(
            load_records(args.annotations),
            media_root=args.media_root.expanduser().resolve(),
            require_files=args.require_files,
            require_id=args.require_id,
            annotation_profile=args.annotation_profile,
        )
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"validate_sharegpt: {exc}", file=sys.stderr)
        return 2
    print(
        "validate_sharegpt: OK "
        f"mode={summary['mode']} records={summary['records']} labels={summary['labels']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
