#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize NVPaw multi-task annotations into typed and ShareGPT records."""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import re
from typing import Any


TASK_SPECS: dict[str, dict[str, Any]] = {
    "Component Classification": {
        "metric_family": "classification",
        "reference_cohort": "single_target",
        "image_roles": ("target",),
        "prompt_format": "nvpaw.component_classification.single.official_v1",
        "anomalygen": False,
        "mining": True,
    },
    "Component Count": {
        "metric_family": "counting",
        "reference_cohort": "single_target",
        "image_roles": ("target",),
        "prompt_format": "nvpaw.component_count.single.official_v1",
        "anomalygen": False,
        "mining": True,
    },
    "Component Detection": {
        "metric_family": "detection",
        "reference_cohort": "single_target",
        "image_roles": ("target",),
        "prompt_format": "nvpaw.component_detection.single.official_v1",
        "anomalygen": False,
        "mining": True,
    },
    "Defect Classification": {
        "metric_family": "classification",
        "reference_cohort": "single_target",
        "image_roles": ("target",),
        "prompt_format": "nvpaw.defect_classification.single.official_v1",
        "anomalygen": True,
        "mining": True,
    },
    "Defect Detection": {
        "metric_family": "detection",
        "reference_cohort": "single_target",
        "image_roles": ("target",),
        "prompt_format": "nvpaw.defect_detection.single.official_v1",
        "anomalygen": False,
        "mining": True,
    },
    "Ref_based Defect Classification": {
        "metric_family": "classification",
        "reference_cohort": "golden_then_target",
        "image_roles": ("golden", "target"),
        "prompt_format": "nvpaw.defect_classification.reference.official_v1",
        "anomalygen": True,
        "mining": True,
    },
    "Ref_based Defect Detection": {
        "metric_family": "detection",
        "reference_cohort": "golden_then_target",
        "image_roles": ("golden", "target"),
        "prompt_format": "nvpaw.defect_detection.reference.official_v1",
        "anomalygen": False,
        "mining": True,
    },
}

DIRECT_CLASS_LABELS = {
    "No, the target image does not contain any defects.": "no_defect",
    "Yes, the target image contains a defect.": "defect",
}
_SAFE_EVAL_ID = re.compile(r"[A-Za-z0-9._@-]+")


def filesystem_safe_id(source_id: str) -> str:
    """Return a stable ID safe for cosmos-rl-evaluate's output filename."""
    if _SAFE_EVAL_ID.fullmatch(source_id) and len(source_id) <= 180:
        return source_id
    readable = re.sub(r"[^A-Za-z0-9._@-]+", "_", source_id).strip("._-")
    readable = (readable or "record")[:120]
    suffix = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:12]
    return f"{readable}_{suffix}"


def _object_lines(path: pathlib.Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
        records.append(value)
    return records


def load_source_records(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"annotation source does not exist: {path}")
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        raise ValueError(f"annotation source is empty: {path}")
    if raw.lstrip().startswith("["):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid JSON: {exc}") from exc
        if not isinstance(payload, list) or not all(
            isinstance(row, dict) for row in payload
        ):
            raise ValueError(f"{path}: JSON input must be an array of objects")
        records = payload
    else:
        records = _object_lines(path)
    if not records:
        raise ValueError(f"annotation source has no records: {path}")
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


def _content_images(content: Any) -> list[str]:
    if not isinstance(content, list):
        return []
    images: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        value: Any = None
        if str(item.get("type", "")).casefold() in {
            "image",
            "input_image",
            "image_url",
        }:
            value = item.get("image", item.get("image_url"))
        elif "image" in item:
            value = item["image"]
        if isinstance(value, dict):
            value = value.get("url")
        if isinstance(value, str) and value:
            images.append(value)
    return images


def _messages(record: dict[str, Any], record_id: str) -> tuple[str, str, str, list[str]]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{record_id}: messages must be a list")
    system: list[str] = []
    prompts: list[str] = []
    answers: list[str] = []
    images: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(f"{record_id}: every message must be an object")
        role = message.get("role")
        text = _content_text(message.get("content"))
        if role == "system" and text:
            system.append(text)
        elif role == "user":
            if text:
                prompts.append(text)
            images.extend(_content_images(message.get("content")))
        elif role == "assistant" and text:
            answers.append(text)
    if not prompts:
        raise ValueError(f"{record_id}: no non-empty user prompt")
    if len(answers) != 1:
        raise ValueError(f"{record_id}: exactly one non-empty assistant answer is required")
    return "\n".join(system), "\n".join(prompts), answers[0].strip(), images


def _strip_fence(text: str) -> str:
    cleaned = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.I)
    return match.group(1).strip() if match else cleaned


def _structured(text: str, record_id: str) -> Any:
    cleaned = _strip_fence(text)
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(cleaned)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    raise ValueError(f"{record_id}: answer is not valid JSON-compatible data")


def _option_map(prompt: str) -> dict[str, str]:
    return {
        letter.upper(): value.strip()
        for letter, value in re.findall(
            r"(?m)^\s*([A-Z])\s*[.)]\s*([^\n]+?)\s*$", prompt
        )
    }


def _choice_tokens(value: Any) -> list[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        bracketed = re.fullmatch(
            r"\[\s*([A-Z](?:\s*,\s*[A-Z])*)\s*\]", cleaned, re.I
        )
        if bracketed:
            return [part.strip().upper() for part in bracketed.group(1).split(",")]
        if re.fullmatch(r"[A-Z]", cleaned, re.I):
            return [cleaned.upper()]
        if re.fullmatch(r"[A-Z](?:\s*,\s*[A-Z])+", cleaned, re.I):
            return [part.strip().upper() for part in cleaned.split(",")]
        return [cleaned]
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            result.extend(_choice_tokens(item))
        return result
    if isinstance(value, dict):
        for key in ("labels", "choices", "answer", "label", "choice"):
            if key in value:
                return _choice_tokens(value[key])
    return []


def _classification_answer(
    answer_text: str, prompt: str, record_id: str
) -> tuple[dict[str, Any], dict[str, str]]:
    options = _option_map(prompt)
    direct = DIRECT_CLASS_LABELS.get(answer_text)
    if direct is not None:
        return {"kind": "choice_set", "labels": [direct]}, {}

    cleaned = _strip_fence(answer_text)
    structured = False
    try:
        value = _structured(cleaned, record_id)
        structured = True
    except ValueError:
        value = cleaned
    if structured and isinstance(value, (list, tuple, set)) and not value:
        return {"kind": "choice_set", "labels": []}, options
    tokens = _choice_tokens(value)
    labels: list[str] = []
    normalized_options = {text.casefold(): text for text in options.values()}
    for token in tokens:
        if token in options:
            semantic = options[token]
        elif (
            marked := re.fullmatch(
                r"(?:answer|option|choice)?\s*[:=-]?\s*([A-Z])(?:\s*[.)].*)?",
                token,
                re.I,
            )
        ) and marked.group(1).upper() in options:
            semantic = options[marked.group(1).upper()]
        elif token.casefold() in normalized_options:
            semantic = normalized_options[token.casefold()]
        else:
            raise ValueError(
                f"{record_id}: classification answer {token!r} is not in the prompt options"
            )
        if semantic not in labels:
            labels.append(semantic)
    if not options and not labels:
        raise ValueError(f"{record_id}: classification prompt has no recoverable options")
    return {"kind": "choice_set", "labels": labels}, options


def validate_bbox(values: Any, *, record_id: str) -> list[int]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError(f"{record_id}: bbox_2d must contain four integers")
    if any(type(value) is not int for value in values):
        raise ValueError(f"{record_id}: bbox_2d coordinates must be integers")
    box = list(values)
    if any(value < 0 or value > 1000 for value in box):
        raise ValueError(f"{record_id}: bbox_2d coordinates must be in [0, 1000]")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"{record_id}: bbox_2d must have positive area")
    return box


def _detection_objects(value: Any, record_id: str) -> list[dict[str, Any]]:
    if isinstance(value, list):
        objects: list[dict[str, Any]] = []
        for item in value:
            objects.extend(_detection_objects(item, record_id))
        return objects
    if not isinstance(value, dict):
        raise ValueError(f"{record_id}: detection answer must be a JSON list/object")
    for collection_key in ("objects", "detections", "answer"):
        if collection_key in value:
            return _detection_objects(value[collection_key], record_id)
    box_value = None
    for box_key in ("bbox_2d", "box_2d", "bbox", "box"):
        if box_key in value:
            box_value = value[box_key]
            break
    if box_value is None:
        raise ValueError(f"{record_id}: detection object is missing bbox_2d")
    label = value.get("label", value.get("class", value.get("category")))
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"{record_id}: detection object label must be non-empty")
    return [
        {
            "label": label.strip(),
            "bbox_2d": validate_bbox(box_value, record_id=record_id),
        }
    ]


def parse_detection_answer(answer_text: str, *, record_id: str) -> dict[str, Any]:
    value = _structured(answer_text, record_id)
    return {
        "kind": "detections",
        "objects": _detection_objects(value, record_id),
    }


def parse_count_answer(answer_text: str, *, record_id: str) -> dict[str, Any]:
    cleaned = _strip_fence(answer_text)
    if re.fullmatch(r"[0-9]+", cleaned) is None:
        raise ValueError(
            f"{record_id}: count answer must be a non-negative integer"
        )
    return {"kind": "count", "value": int(cleaned)}


def materialize_records(
    records: list[dict[str, Any]], *, prompt_variant: str = "official_v1"
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if prompt_variant != "official_v1":
        raise ValueError(f"unsupported prompt variant {prompt_variant!r}")
    if not isinstance(records, list) or not records:
        raise ValueError("source records must be a non-empty list")

    manifest: list[dict[str, Any]] = []
    sharegpt: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, source in enumerate(records):
        if not isinstance(source, dict):
            raise ValueError(f"record[{index}] must be an object")
        source_id = source.get("id")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(f"record[{index}]: id must be a non-empty string")
        source_id = source_id.strip()
        if source_id in seen_ids:
            raise ValueError(f"duplicate id {source_id!r}")
        seen_ids.add(source_id)
        record_id = filesystem_safe_id(source_id)

        task_type = source.get("task_type")
        if task_type not in TASK_SPECS:
            raise ValueError(f"{source_id}: unsupported task_type {task_type!r}")
        spec = TASK_SPECS[task_type]
        system, prompt, answer_text, image_paths = _messages(source, source_id)
        roles = list(spec["image_roles"])
        if len(image_paths) != len(roles):
            raise ValueError(
                f"{source_id}: {task_type} requires image roles {roles}, got {len(image_paths)} image(s)"
            )
        images = [
            {"role": role, "path": path}
            for role, path in zip(roles, image_paths, strict=True)
        ]
        target_path = next(image["path"] for image in images if image["role"] == "target")
        target_id = source.get("target_id", target_path)
        if not isinstance(target_id, str) or not target_id.strip():
            raise ValueError(f"{source_id}: target_id must be a non-empty string")

        if spec["metric_family"] == "classification":
            canonical_answer, options = _classification_answer(
                answer_text, prompt, source_id
            )
        elif spec["metric_family"] == "detection":
            canonical_answer = parse_detection_answer(
                answer_text, record_id=source_id
            )
            options = {}
        elif spec["metric_family"] == "counting":
            canonical_answer = parse_count_answer(
                answer_text, record_id=source_id
            )
            options = {}
        else:
            raise ValueError(
                f"{source_id}: unsupported metric family {spec['metric_family']!r}"
            )

        normalized: dict[str, Any] = {
            "schema_version": "nvpaw_multitask_v1",
            "id": record_id,
            "source_id": source_id,
            "target_id": target_id.strip(),
            "dataset": str(source.get("dataset", "unknown")),
            "subset": source.get("subset"),
            "split": str(source.get("split", "unknown")),
            "category": source.get("category"),
            "task_type": task_type,
            "metric_family": spec["metric_family"],
            "reference_cohort": spec["reference_cohort"],
            "images": images,
            "prompt_format": spec["prompt_format"],
            "prompt_variant": prompt_variant,
            "system_prompt": system,
            "prompt": prompt,
            "option_map": options,
            "answer": canonical_answer,
            "rendered_answer": answer_text,
            "capabilities": {
                "mining": bool(spec["mining"]),
                "anomalygen": bool(spec["anomalygen"]),
            },
        }
        manifest.append(normalized)
        sharegpt.append(
            {
                "id": record_id,
                "source_id": source_id,
                "target_id": normalized["target_id"],
                "dataset": normalized["dataset"],
                "subset": normalized["subset"],
                "split": normalized["split"],
                "category": normalized["category"],
                "task_type": task_type,
                "metric_family": normalized["metric_family"],
                "reference_cohort": normalized["reference_cohort"],
                "prompt_format": normalized["prompt_format"],
                "prompt_variant": prompt_variant,
                "image_roles": roles,
                "images": image_paths,
                "option_map": options,
                "answer": canonical_answer,
                "capabilities": normalized["capabilities"],
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": answer_text},
                ],
            }
        )
    return manifest, sharegpt
