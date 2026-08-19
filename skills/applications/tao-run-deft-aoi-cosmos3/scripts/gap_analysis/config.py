"""Gap-analysis profile loading and validation."""

from __future__ import annotations

import pathlib
from typing import Any

import yaml


PACKAGED_PROFILES = (
    "legacy_bare_okng",
    "global_topk",
    "equal_task_round_robin",
    "deficit_weighted_round_robin",
    "task_dataset_round_robin",
    "random_control",
    "hardness_diversity",
)


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("gap-analysis config must be an object")
    allowed = {
        "schema_version",
        "candidate_builder",
        "scorer",
        "allocator",
        "selector",
        "budget",
        "fraction_per_group",
        "min_per_group",
        "max_per_group",
        "max_per_dataset",
        "expected_groups",
        "missing_group_policy",
        "seed",
    }
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"unknown gap-analysis keys: {unknown}")
    if config.get("schema_version") != "gap_analysis_v1":
        raise ValueError("schema_version must be gap_analysis_v1")
    if config.get("candidate_builder") not in {"legacy_bare_okng", "multitask_v1"}:
        raise ValueError("candidate_builder must be legacy_bare_okng or multitask_v1")

    component_contracts = {
        "scorer": (
            {"name", "risk_weights", "weights"},
            {
                "binary_error",
                "sample_metric_deficit",
                "risk_weighted_deficit",
                "uncertainty",
                "hybrid",
            },
        ),
        "allocator": (
            {"name", "group_by", "subgroup_by"},
            {
                "global_topk",
                "equal_task_round_robin",
                "support_proportional",
                "deficit_weighted_round_robin",
                "worst_group_first",
                "task_dataset_round_robin",
            },
        ),
        "selector": (
            {"name", "hardness_weight", "embedding_field"},
            {"hardest", "stratified_random", "diverse_topk", "hardness_diversity"},
        ),
    }
    normalized = dict(config)
    for component, (component_allowed, names) in component_contracts.items():
        value = config.get(component)
        if not isinstance(value, dict):
            raise ValueError(f"{component} must be an object")
        component_unknown = sorted(set(value) - component_allowed)
        if component_unknown:
            raise ValueError(f"unknown {component} keys: {component_unknown}")
        if value.get("name") not in names:
            raise ValueError(f"unsupported {component} name {value.get('name')!r}")
        normalized[component] = dict(value)

    allocator = normalized["allocator"]
    group_by = allocator.get("group_by", [])
    if not isinstance(group_by, list) or not all(
        isinstance(field, str) and field for field in group_by
    ):
        raise ValueError("allocator.group_by must be a string list")
    subgroup = allocator.get("subgroup_by")
    if subgroup is not None and not (
        isinstance(subgroup, str)
        or (isinstance(subgroup, list) and all(isinstance(field, str) for field in subgroup))
    ):
        raise ValueError("allocator.subgroup_by must be a string or string list")
    if allocator["name"] == "global_topk" and (group_by or subgroup):
        raise ValueError("global_topk requires empty group_by and no subgroup_by")
    if allocator["name"] == "task_dataset_round_robin" and not subgroup:
        raise ValueError("task_dataset_round_robin requires subgroup_by")
    if (
        normalized["candidate_builder"] == "legacy_bare_okng"
        and normalized["scorer"]["name"] != "binary_error"
    ):
        raise ValueError("legacy_bare_okng requires the binary_error scorer")

    budget = config.get("budget")
    if type(budget) is not int or budget <= 0:
        raise ValueError("budget must be a positive integer")
    fraction = config.get("fraction_per_group", 1.0)
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < float(fraction) <= 1:
        raise ValueError("fraction_per_group must be in (0, 1]")
    normalized["fraction_per_group"] = float(fraction)
    for field, default in (("min_per_group", 0), ("max_per_group", None), ("max_per_dataset", None)):
        value = config.get(field, default)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"{field} must be a non-negative integer or null")
        normalized[field] = value
    seed = config.get("seed", 0)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    normalized["seed"] = seed
    if config.get("missing_group_policy", "error") not in {"error", "warn"}:
        raise ValueError("missing_group_policy must be error or warn")
    normalized["missing_group_policy"] = config.get("missing_group_policy", "error")
    expected = config.get("expected_groups", {})
    if not isinstance(expected, dict):
        raise ValueError("expected_groups must be an object")
    expected_unknown = sorted(set(expected) - {"source", "groups"})
    if expected_unknown:
        raise ValueError(f"unknown expected_groups keys: {expected_unknown}")
    if "source" in expected and expected["source"] != "annotation_manifest":
        raise ValueError("expected_groups.source must be annotation_manifest")
    if "groups" in expected and (
        not isinstance(expected["groups"], list)
        or not expected["groups"]
        or not all(isinstance(group, str) and group for group in expected["groups"])
    ):
        raise ValueError("expected_groups.groups must be a non-empty string list")
    normalized["expected_groups"] = dict(expected)
    return normalized


def load_profile(
    name: str, *, assets_root: pathlib.Path | None = None
) -> dict[str, Any]:
    if name not in PACKAGED_PROFILES:
        raise ValueError(
            f"unknown packaged gap-analysis profile {name!r}; "
            f"choose one of {list(PACKAGED_PROFILES)}"
        )
    root = assets_root or pathlib.Path(__file__).resolve().parents[2] / "assets/gap-analysis"
    path = root / f"{name}.yaml"
    if not path.is_file():
        raise ValueError(f"packaged gap-analysis profile is missing: {path}")
    try:
        payload = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid gap-analysis profile {path}: {exc}") from exc
    return validate_config(payload)
