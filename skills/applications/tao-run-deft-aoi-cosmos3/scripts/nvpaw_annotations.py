#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Canonical NVPAW task and image-role contract."""

from __future__ import annotations

from typing import Any


TASK_SPECS: dict[str, dict[str, Any]] = {
    "Component Classification": {
        "metric_family": "classification",
        "reference_cohort": "non_reference_based",
        "image_roles": ("target",),
        "prompt_format": "nvpaw.component_classification.single.official_v1",
        "mining": True,
    },
    "Component Detection": {
        "metric_family": "detection",
        "reference_cohort": "non_reference_based",
        "image_roles": ("target",),
        "prompt_format": "nvpaw.component_detection.single.official_v1",
        "mining": True,
    },
    "Defect Classification": {
        "metric_family": "classification",
        "reference_cohort": "non_reference_based",
        "image_roles": ("target",),
        "prompt_format": "nvpaw.defect_classification.single.official_v1",
        "mining": True,
    },
    "Defect Detection": {
        "metric_family": "detection",
        "reference_cohort": "non_reference_based",
        "image_roles": ("target",),
        "prompt_format": "nvpaw.defect_detection.single.official_v1",
        "mining": True,
    },
    "Ref_based Defect Classification": {
        "metric_family": "classification",
        "reference_cohort": "reference_based",
        "image_roles": ("golden", "target"),
        "prompt_format": "nvpaw.defect_classification.reference.official_v1",
        "mining": True,
    },
    "Ref_based Defect Detection": {
        "metric_family": "detection",
        "reference_cohort": "reference_based",
        "image_roles": ("golden", "target"),
        "prompt_format": "nvpaw.defect_detection.reference.official_v1",
        "mining": True,
    },
}


def validate_bbox(values: Any, *, record_id: str) -> list[int]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError(f"{record_id}: bbox_2d must contain four integers")
    if any(type(value) is not int for value in values):
        raise ValueError(f"{record_id}: bbox_2d coordinates must be integers")
    result = list(values)
    if any(value < 0 or value > 1000 for value in result):
        raise ValueError(f"{record_id}: bbox_2d coordinates must be in [0, 1000]")
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError(f"{record_id}: bbox_2d must have positive area")
    return result
