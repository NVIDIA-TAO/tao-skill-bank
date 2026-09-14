#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Ground-truth answer shape helpers shared by the selectors and the assembler.

The training rows keep their native NVPAW ``messages``; nothing here rewrites a
row. These helpers only *read* the prompt and the assistant answer to classify
a row by answer format (BCQ / MCQ / DET / COUNT), image count and whether the
ground truth is an empty answer (``[]`` / ``{}`` / blank). MCQ option parsing
maps the prompt's lettered ``current possible classes`` block to semantic
labels so class shares can be compared across rows whose letters differ.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Iterable

from nvpaw_annotations import TASK_SPECS
from validate_sharegpt import image_items, prompt_and_response

# The yes/no classification form ends with this sentence and never offers an
# empty answer; every other classification prompt is a lettered MCQ.
BCQ_PHRASE = "Answer with the complete option text"
MCQ_OPTIONS_MARKER = "current possible classes"
EMPTY_DEFINITION = "parsed assistant answer is [] / {} / blank after stripping code fences; BCQ 'No' is not empty"
FORMATS = ("BCQ", "MCQ", "DET", "COUNT")

_BCQ_ANSWER = re.compile(r"^\s*(?:[AB]\.\s*)?(?:Yes|No)\b", re.IGNORECASE)
_OPTION_LINE = re.compile(r"^\s*([A-Z])\.\s*(\S.*?)\s*$")
_LEADING_LETTER = re.compile(r"^\s*([A-Z])(?=$|[\s\.\):,])")


def strip_code_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    return value


def is_empty_answer_text(text: str) -> bool:
    value = strip_code_fence(text)
    if value in ("", "[]", "{}"):
        return True
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, (list, dict)) and len(parsed) == 0


def is_empty_ground_truth(record: dict[str, Any], *, context: str = "record") -> bool:
    _, answer = prompt_and_response(record, context=context)
    return is_empty_answer_text(answer)


def image_count(record: dict[str, Any], *, context: str = "record") -> int:
    return len(image_items(record, context=context))


def answer_format(record: dict[str, Any], *, context: str = "record") -> str:
    """BCQ / MCQ for classification tasks, DET for detection, COUNT for counting."""

    task = str(record.get("task_type"))
    spec = TASK_SPECS.get(task)
    if spec is None:
        return "OTHER"
    family = spec["metric_family"]
    if family == "detection":
        return "DET"
    if family == "counting":
        return "COUNT"
    prompt, answer = prompt_and_response(record, context=context)
    if BCQ_PHRASE in prompt or _BCQ_ANSWER.match(strip_code_fence(answer)):
        return "BCQ"
    return "MCQ"


def mcq_options(prompt: str) -> dict[str, str] | None:
    """Letter -> semantic label from the ``current possible classes`` block.

    Returns None when the prompt is not the lettered MCQ form (no options
    marker, the yes/no form, or fewer than two lettered options). The label is
    the option text before its first ``:`` (defect options carry a description
    after the colon; component options have none).
    """

    if BCQ_PHRASE in prompt:
        return None
    lowered = prompt.lower()
    position = lowered.find(MCQ_OPTIONS_MARKER)
    if position < 0:
        return None
    options: dict[str, str] = {}
    for line in prompt[position:].splitlines()[1:]:
        match = _OPTION_LINE.match(line)
        if match is None:
            continue
        letter, text = match.group(1), match.group(2)
        if letter in options:
            return None
        label = text.split(":", 1)[0].strip()
        if not label:
            return None
        options[letter] = label
    if len(options) < 2:
        return None
    return options


def mcq_answer_letters(answer: str) -> set[str] | None:
    """Parse ``F`` / ``[B,D]`` / ``["B","D"]`` / ``[]`` into a letter set; None if unparsable."""

    value = strip_code_fence(answer)
    if value in ("", "[]", "{}"):
        return set()
    items: list[str]
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        if not all(isinstance(item, str) for item in parsed):
            return None
        items = parsed
    elif parsed is not None:
        return None
    elif value.startswith("[") and value.endswith("]"):
        items = [item for item in value[1:-1].split(",") if item.strip()]
    else:
        items = [value]
    letters: set[str] = set()
    for item in items:
        match = _LEADING_LETTER.match(item)
        if match is None:
            return None
        letters.add(match.group(1))
    return letters


def mcq_labels(record: dict[str, Any], *, context: str = "record") -> tuple[str, list[str] | None]:
    """Return ``(status, labels)``; status is ``ok`` or the rejection reason.

    Rejections: ``not_mcq_format`` (no options block / yes-no form),
    ``unparsed_answer``, ``empty_ground_truth``, ``unknown_option_letter``.
    """

    prompt, answer = prompt_and_response(record, context=context)
    options = mcq_options(prompt)
    if options is None:
        return "not_mcq_format", None
    letters = mcq_answer_letters(answer)
    if letters is None:
        return "unparsed_answer", None
    if not letters:
        return "empty_ground_truth", None
    if not letters.issubset(options):
        return "unknown_option_letter", None
    return "ok", sorted({options[letter] for letter in letters})


def row_profile(record: dict[str, Any], *, context: str = "record") -> dict[str, Any]:
    return {
        "task_type": str(record.get("task_type")),
        "format": answer_format(record, context=context),
        "images": image_count(record, context=context),
        "empty": is_empty_ground_truth(record, context=context),
    }


def _share(empty: int, rows: int) -> float:
    return empty / rows if rows else 0.0


def profile_rows(profiles: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate ``row_profile`` outputs: overall, task x format x images, task, classification."""

    rows = 0
    empty = 0
    cells: dict[tuple[str, str, int], list[int]] = {}
    by_task: dict[str, list[int]] = {}
    classification_rows = 0
    classification_empty = 0
    classification_by_task: dict[str, list[int]] = {}
    for item in profiles:
        rows += 1
        empty += bool(item["empty"])
        key = (item["task_type"], item["format"], int(item["images"]))
        cells.setdefault(key, [0, 0])
        cells[key][0] += 1
        cells[key][1] += bool(item["empty"])
        by_task.setdefault(item["task_type"], [0, 0])
        by_task[item["task_type"]][0] += 1
        by_task[item["task_type"]][1] += bool(item["empty"])
        if item["format"] in ("BCQ", "MCQ"):
            classification_rows += 1
            classification_empty += bool(item["empty"])
            classification_by_task.setdefault(item["task_type"], [0, 0])
            classification_by_task[item["task_type"]][0] += 1
            classification_by_task[item["task_type"]][1] += bool(item["empty"])

    def payload(counts: list[int]) -> dict[str, Any]:
        return {"rows": counts[0], "empty_rows": counts[1], "empty_share": _share(counts[1], counts[0])}

    return {
        "unit": "rows",
        "empty_definition": EMPTY_DEFINITION,
        "formats": list(FORMATS),
        "rows": rows,
        "empty_rows": empty,
        "empty_share": _share(empty, rows),
        "by_task_format_images": {
            f"{task}|{fmt}|{images}": {"task_type": task, "format": fmt, "images": images, **payload(counts)}
            for (task, fmt, images), counts in sorted(cells.items())
        },
        "by_task": {task: payload(counts) for task, counts in sorted(by_task.items())},
        "classification": {
            "formats": ["BCQ", "MCQ"],
            **payload([classification_rows, classification_empty]),
            "by_task": {task: payload(counts) for task, counts in sorted(classification_by_task.items())},
        },
    }


def label_counts(records: Iterable[dict[str, Any]], *, context: str = "kpi") -> Counter[str]:
    """Semantic MCQ labels of non-empty rows (a multi-label row counts once per label)."""

    counts: Counter[str] = Counter()
    for index, record in enumerate(records):
        status, labels = mcq_labels(record, context=f"{context}[{index}]")
        if status == "ok" and labels:
            counts.update(labels)
    return counts


__all__ = [
    "BCQ_PHRASE",
    "EMPTY_DEFINITION",
    "FORMATS",
    "MCQ_OPTIONS_MARKER",
    "answer_format",
    "image_count",
    "is_empty_answer_text",
    "is_empty_ground_truth",
    "label_counts",
    "mcq_answer_letters",
    "mcq_labels",
    "mcq_options",
    "profile_rows",
    "row_profile",
    "strip_code_fence",
]
