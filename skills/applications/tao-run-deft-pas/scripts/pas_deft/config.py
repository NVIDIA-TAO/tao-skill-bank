# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Bundled and adapted from the Apache-2.0 NVIDIA TAO Tutorials IAA DEFT utilities so the
# customer workflow does not depend on an external source checkout.

<<<<<<< HEAD:skills/applications/tao-run-deft-iaa/scripts/iaa_deft/config.py
"""Parsed configuration for an IAA CLIP DEFT experiment."""
=======
"""Typed configuration for the bundled PAS CLIP DEFT runtime.
>>>>>>> 0ea1223 ([TAO-6655434][Bugfix] Rename DEFT workflow from IAA to PAS (#194)):skills/applications/tao-run-deft-pas/scripts/pas_deft/config.py

import json
import os
import yaml

<<<<<<< HEAD:skills/applications/tao-run-deft-iaa/scripts/iaa_deft/config.py
=======
from pas_deft.config_fields import (
    BOOL_FIELD,
    DATACLASS_FIELD,
    FLOAT_FIELD,
    INT_FIELD,
    STR_FIELD,
)
>>>>>>> 0ea1223 ([TAO-6655434][Bugfix] Rename DEFT workflow from IAA to PAS (#194)):skills/applications/tao-run-deft-pas/scripts/pas_deft/config.py

def _bool_str(value) -> str:
    """Convert a config value to the 'true'/'false' string expected by pipeline functions."""
    return "true" if str(value).strip().lower() in ("true", "1", "yes", "y") else "false"


def _abs_data_path(value: str) -> str:
    """Absolutize an IAA data path, leaving blanks and absolute paths untouched.

    These values are handed to both host-side code (which reads the files) and
    the TAO container specs (which read them again through a bind mount), so a
    single string has to be valid on both sides. The workflow mounts the data
    root at its own host path to make that true, which only holds if the path
    is absolute — a relative ``data/...`` resolves against the container's
    working directory (``/opt/nvidia``) and silently finds nothing.
    """
    value = str(value or "")
    return os.path.abspath(value) if value else value


class IaaDeftConfig:
    """All parsed parameters for an IAA CLIP DEFT experiment.

    Loads a YAML spec file once and exposes every pipeline parameter as an
    attribute, so callers never touch raw dicts directly.

    Usage::

<<<<<<< HEAD:skills/applications/tao-run-deft-iaa/scripts/iaa_deft/config.py
        cfg = IaaDeftConfig("configs/clip_config.yaml")
    """
=======
    enabled: bool = BOOL_FIELD(
        value=False,
        description="Create contact sheets for weak and mined PAS samples.",
    )
    embeddings: bool = BOOL_FIELD(
        value=False,
        description="Create t-SNE plots for weak, mined, and prior training samples.",
    )
    max_samples_per_group: int = INT_FIELD(
        value=12,
        valid_min=1,
        description="Maximum contact-sheet samples per dataset/query-type group.",
    )
    max_total_samples: int = INT_FIELD(
        value=96,
        valid_min=1,
        description="Maximum total samples included in contact sheets.",
    )
    tile_size: int = INT_FIELD(
        value=192,
        valid_min=1,
        description="Contact-sheet tile edge length in pixels.",
    )


@dataclasses.dataclass
class IterationSection:
    """``iteration``: inclusive DEFT loop bounds."""

    start: int = INT_FIELD(
        value=MISSING,
        valid_min=1,
        description="First DEFT iteration to run, using one-based numbering.",
    )
    end: int = INT_FIELD(
        value=MISSING,
        valid_min=1,
        description="Last DEFT iteration to run, inclusive.",
    )


@dataclasses.dataclass
class TrainingSection:
    """``training``: checkpoint and dataset carry-over policy."""

    init_checkpoint: str = STR_FIELD(
        value="",
        description="Checkpoint used to initialize the first/base-model training iteration.",
    )
    continual_model: bool = BOOL_FIELD(
        value=False,
        description="Carry each trained model into the next iteration when true.",
    )
    continual_dataset: bool = BOOL_FIELD(
        value=True,
        description="Accumulate mined datasets across iterations when true.",
    )
    num_nodes: int = INT_FIELD(
        value=1,
        valid_min=1,
        description="Number of training nodes represented by the approved configuration.",
    )


@dataclasses.dataclass
class HistoryAwareSection:
    """``mining.history_aware``: cross-iteration selection ledger."""

    enabled: bool = BOOL_FIELD(
        value=False,
        description="Prevent already-selected source pairs from being selected as new pairs.",
    )
    replay_fraction: float = FLOAT_FIELD(
        value=0.20,
        valid_min=0.0,
        valid_max=1.0,
        description="Target-budget fraction used to replay prior pairs for non-continual datasets.",
    )


@dataclasses.dataclass
class CaptionExpansionSection:
    """``mining.recovery.caption_expansion``: mined-anchor caption expansion."""

    enabled: bool = BOOL_FIELD(
        value=False,
        description="Expand each mined image anchor to additional captions from the source pairs.",
    )
    mode: str = STR_FIELD(
        value="nearest",
        valid_options="nearest,all",
        description="Choose nearest captions by similarity or all captions in source order.",
    )
    max_pairs_per_image_path: int = INT_FIELD(
        value=2,
        valid_min=0,
        description="Maximum selected pairs per image path; zero means unlimited.",
    )
    max_expanded_pair_fraction: float = FLOAT_FIELD(
        value=0.25,
        valid_min=0.0,
        valid_max=1.0,
        description="Maximum target-budget fraction occupied by non-anchor expanded pairs.",
    )
    dedupe_normalized_caption: bool = BOOL_FIELD(
        value=True,
        description="Deduplicate normalized captions for each image path.",
    )
    count_expanded_pairs_toward_target: str = STR_FIELD(
        value="auto",
        valid_options="auto,true,false",
        description="Whether expanded pairs count toward the target budget; auto depends on mode.",
    )


@dataclasses.dataclass
class MiningSection:
    """``mining``: nearest-neighbor and selection settings."""

    knn_batch_size: int = INT_FIELD(
        value=32,
        valid_min=1,
        description="Batch size used by nearest-neighbor mining.",
    )
    topn: int = INT_FIELD(
        value=MISSING,
        valid_min=1,
        description="Nearest neighbors retained per target query; loaded from mining_spec.yaml.",
    )
    knn_metric: str = STR_FIELD(
        value=MISSING,
        valid_options="cosine,euclidean",
        description="Distance metric used by nearest-neighbor mining; loaded from mining_spec.yaml.",
    )
    history_aware: HistoryAwareSection = DATACLASS_FIELD(
        HistoryAwareSection(),
        description="Cross-iteration selection-ledger settings.",
    )
    caption_expansion: CaptionExpansionSection = DATACLASS_FIELD(
        CaptionExpansionSection(),
        description="Caption expansion settings under mining.recovery.",
    )


@dataclasses.dataclass
class PasSection:
    """PAS split materialization and source paths."""

    seed_exclude_datasets: str = STR_FIELD(
        value="CUHK_PEDES,ICFG_PEDES",
        description="Comma-separated real datasets excluded from the seed training split.",
    )
    augmented_suffix: str = STR_FIELD(
        value="_Aug",
        description="Dataset-name suffix that identifies augmented PAS rows.",
    )
    query_types: str = STR_FIELD(
        value=",".join(QUERY_TYPES),
        description=(
            "Comma-separated PAS query types retained during split materialization; "
            "each value must be easy, medium, hard, natural_caption, or "
            "original_captions."
        ),
    )
    max_seed_rows: int = INT_FIELD(
        value=0,
        valid_min=0,
        description="Maximum seed-training rows; zero means no limit.",
    )
    max_aug_pool_rows: int = INT_FIELD(
        value=0,
        valid_min=0,
        description="Maximum mining-pool rows; zero means no limit.",
    )
    mining_pool_mode: str = STR_FIELD(
        value="real_and_augmented",
        valid_options="real,augmented,real_and_augmented",
        description="Select real rows, augmented rows, or both for the mining pool.",
    )
    val_sample_size: int = INT_FIELD(
        value=512,
        valid_min=0,
        description="Number of evaluation images sampled into the TAO validation list.",
    )
    train_pairs_source_file: str = STR_FIELD(
        value="",
        description="Optional source train_pairs.json for seed training.",
    )
    pool_pairs_source_file: str = STR_FIELD(
        value="",
        description="Source pairs JSON for mining; falls back to train pairs when blank.",
    )
    eval_pairs_source_file: str = STR_FIELD(
        value=MISSING,
        description="Approved val_pairs.json or test_pairs.json used for evaluation.",
    )
    train_image_dir: str = STR_FIELD(
        value=MISSING,
        description="Image root for the seed training split.",
    )
    train_caption_dir: str = STR_FIELD(
        value=MISSING,
        description="Caption root for the seed training split.",
    )
    source_image_dir: str = STR_FIELD(
        value=MISSING,
        description="Image root for the PAS mining pool and mined outputs.",
    )
    source_caption_dir: str = STR_FIELD(
        value=MISSING,
        description="Caption root for the PAS mining pool and mined outputs.",
    )
    eval_image_dir: str = STR_FIELD(
        value=MISSING,
        description="Image root for the PAS evaluation split.",
    )
    eval_caption_dir: str = STR_FIELD(
        value=MISSING,
        description="Caption root for the PAS evaluation split.",
    )


@dataclasses.dataclass
class CaptionDiversitySection:
    """``gap_analysis.caption_diversity``: coverage-aware caption rotation."""

    enabled: bool = BOOL_FIELD(
        value=False,
        description="Rotate weak-query captions to maximize cross-iteration coverage.",
    )
    history_file: str = STR_FIELD(
        value="caption_selection_history.json",
        valid_options="caption_selection_history.json",
        description="Canonical result-relative file recording captions selected by prior iterations.",
    )
    history_policy: str = STR_FIELD(
        value="auto",
        valid_options="auto,prefer_unseen,novelty_with_replay",
        description="Caption-history policy; auto derives the policy from continual_dataset.",
    )
    coverage_target: float = FLOAT_FIELD(
        value=1.0,
        valid_min=0.0,
        valid_max=1.0,
        description="Fraction of each weak group's unseen captions planned across remaining iterations.",
    )
    min_unique_texts_per_attribute: int = INT_FIELD(
        value=0,
        valid_min=0,
        description="Minimum unique captions selected per weak attribute when available.",
    )
    max_unique_texts_per_attribute: int = INT_FIELD(
        value=0,
        valid_min=0,
        description="Maximum unique captions per weak attribute; zero means no limit.",
    )
    max_rows_per_unique_text: int = INT_FIELD(
        value=1,
        valid_min=1,
        description="Maximum selected rows for one normalized caption.",
    )
    max_rows_per_image_path: int = INT_FIELD(
        value=1,
        valid_min=0,
        description="Maximum selected rows for one image path; zero means no limit.",
    )
    recent_exclude_iters: int = INT_FIELD(
        value=0,
        valid_min=0,
        description="Number of recent iterations whose captions cannot be selected again.",
    )
    replay_fraction_when_noncontinual: float = FLOAT_FIELD(
        value=0.25,
        valid_min=0.0,
        valid_max=1.0,
        description="Per-group replay fraction for novelty_with_replay on non-continual data.",
    )


@dataclasses.dataclass
class GapAnalysisSection:
    """``gap_analysis``: weak-attribute selection for the next mining target."""

    metric_name: str = STR_FIELD(
        value="Rank-1",
        valid_options="mAP,Rank-1,Rank-5,Separability,Match@5,Zero@5",
        description="Metric used to rank weak PAS attributes.",
    )
    queries_per_slice: int = INT_FIELD(
        value=256,
        valid_min=0,
        description="Maximum weak-query captions sampled per attribute; zero means no limit.",
    )
    min_num_queries: int = INT_FIELD(
        value=1,
        valid_min=0,
        description="Ignore metric rows backed by fewer queries than this threshold.",
    )
    query_types: str = STR_FIELD(
        value="easy,medium",
        description=(
            "Comma-separated query types considered during weak-attribute selection; "
            "each value must be easy, medium, hard, natural_caption, or "
            "original_captions."
        ),
    )
    weak_attribute_topk: int = INT_FIELD(
        value=8,
        valid_min=0,
        description="Number of weakest attributes selected for each iteration; zero means all.",
    )
    target_query_count: int = INT_FIELD(
        value=100000,
        valid_min=0,
        description="Final mined-pair budget for an iteration; zero means unlimited.",
    )
    total_queries_map: int = INT_FIELD(
        value=768,
        valid_min=0,
        description="Query budget used when mAP-based analysis is enabled.",
    )
    analyze_by_map: bool = BOOL_FIELD(
        value=False,
        description="Rank weak attributes by mAP instead of the per-query metric breakdown.",
    )
    caption_diversity: CaptionDiversitySection = DATACLASS_FIELD(
        CaptionDiversitySection(),
        description="Coverage-aware caption rotation settings.",
    )


@dataclasses.dataclass
class DeftExperimentConfig:
    """Top-level typed PAS DEFT configuration."""

    experiment: ExperimentSection = DATACLASS_FIELD(
        ExperimentSection(), description="Experiment identity and TAO spec locations."
    )
    visualization: VisualizationSection = DATACLASS_FIELD(
        VisualizationSection(), description="Contact-sheet and embedding visualization controls."
    )
    iteration: IterationSection = DATACLASS_FIELD(
        IterationSection(), description="Inclusive DEFT loop iteration range."
    )
    training: TrainingSection = DATACLASS_FIELD(
        TrainingSection(), description="Training checkpoint and dataset carry-over policy."
    )
    mining: MiningSection = DATACLASS_FIELD(
        MiningSection(), description="Nearest-neighbor mining and selection controls."
    )
    pas: PasSection = DATACLASS_FIELD(
        PasSection(), description="PAS data split and source-path controls."
    )
    gap_analysis: GapAnalysisSection = DATACLASS_FIELD(
        GapAnalysisSection(), description="Weak-attribute selection controls."
    )


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return dict(value)


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"{path} contains unknown key(s): {', '.join(unknown)}")


def _validate_raw_keys(raw: Mapping[str, Any]) -> None:
    """Reject keys that would otherwise be lost while normalizing legacy layout."""
    _reject_unknown(
        raw,
        {"experiment", "visualization", "iteration", "training", "mining", "pas", "gap_analysis"},
        "deft_config",
    )

    section_types = {
        "experiment": ExperimentSection,
        "visualization": VisualizationSection,
        "iteration": IterationSection,
        "training": TrainingSection,
        "pas": PasSection,
        "gap_analysis": GapAnalysisSection,
    }
    for section_name, section_type in section_types.items():
        if section_name not in raw:
            continue
        section = _mapping(raw.get(section_name), section_name)
        allowed = {field.name for field in dataclasses.fields(section_type)}
        if section_name == "gap_analysis":
            allowed.update({"total_queries_mAP", "analyze_by_mAP"})
        _reject_unknown(section, allowed, section_name)

    mining = _mapping(raw.get("mining"), "mining")
    _reject_unknown(
        mining,
        {"knn_batch_size", "topn", "knn_metric", "history_aware", "recovery"},
        "mining",
    )
    history = _mapping(mining.get("history_aware"), "mining.history_aware")
    _reject_unknown(
        history,
        {field.name for field in dataclasses.fields(HistoryAwareSection)},
        "mining.history_aware",
    )
    recovery = _mapping(mining.get("recovery"), "mining.recovery")
    _reject_unknown(recovery, {"caption_expansion"}, "mining.recovery")
    expansion = _mapping(
        recovery.get("caption_expansion"),
        "mining.recovery.caption_expansion",
    )
    _reject_unknown(
        expansion,
        {field.name for field in dataclasses.fields(CaptionExpansionSection)},
        "mining.recovery.caption_expansion",
    )
    gap = _mapping(raw.get("gap_analysis"), "gap_analysis")
    diversity = _mapping(
        gap.get("caption_diversity"),
        "gap_analysis.caption_diversity",
    )
    _reject_unknown(
        diversity,
        {field.name for field in dataclasses.fields(CaptionDiversitySection)},
        "gap_analysis.caption_diversity",
    )


def _validate_field_constraints(instance: Any, path: str = "") -> None:
    for field in dataclasses.fields(instance):
        value = getattr(instance, field.name)
        field_path = f"{path}.{field.name}" if path else field.name
        if dataclasses.is_dataclass(value):
            _validate_field_constraints(value, field_path)
            continue
        metadata = field.metadata
        valid_min = metadata.get("valid_min", "")
        valid_max = metadata.get("valid_max", "")
        valid_options = metadata.get("valid_options", "")
        if valid_min != "" and value < valid_min:
            raise ValueError(
                f"{field_path}={value!r} is below the minimum allowed value {valid_min}"
            )
        if valid_max != "" and value > valid_max:
            raise ValueError(
                f"{field_path}={value!r} is above the maximum allowed value {valid_max}"
            )
        if valid_options:
            options = {option.strip() for option in valid_options.split(",")}
            if str(value) not in options:
                raise ValueError(f"{field_path}={value!r} must be one of: {valid_options}")


def _coerce_scalar(value: Any, expected_type: type, path: str) -> Any:
    """Apply the notebook config's safe scalar conversions without OmegaConf."""
    if value == MISSING:
        raise ValueError(f"{path} is missing a mandatory value")
    if value is None:
        raise ValueError(f"{path} cannot be null")

    if expected_type is bool:
        if type(value) is bool:
            return value
        if type(value) is int:
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "y", "on", "1"}:
                return True
            if normalized in {"false", "no", "n", "off", "0"}:
                return False
        raise ValueError(f"{path}={value!r} cannot be converted to bool")

    if expected_type is int:
        if type(value) is int:
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                pass
        raise ValueError(f"{path}={value!r} cannot be converted to int")

    if expected_type is float:
        if type(value) in {int, float}:
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
        raise ValueError(f"{path}={value!r} cannot be converted to float")

    if expected_type is str:
        if isinstance(value, (Mapping, list, tuple, set)):
            raise ValueError(f"{path}={value!r} cannot be converted to str")
        return str(value)

    if isinstance(value, expected_type):
        return value
    raise ValueError(
        f"{path}={value!r} has type {type(value).__name__}; "
        f"expected {expected_type.__name__}"
    )


def _materialize_dataclass(
    config_type: type,
    source: Mapping[str, Any],
    path: str = "",
) -> Any:
    """Recursively construct a typed config from a normalized source mapping."""
    if not isinstance(source, Mapping):
        location = path or config_type.__name__
        raise ValueError(f"{location} must be an object")

    fields = {field.name: field for field in dataclasses.fields(config_type)}
    unknown = sorted(set(source) - set(fields))
    if unknown:
        location = path or config_type.__name__
        raise ValueError(f"{location} contains unknown key(s): {', '.join(unknown)}")

    type_hints = get_type_hints(config_type)
    values: dict[str, Any] = {}
    for name, field in fields.items():
        field_path = f"{path}.{name}" if path else name
        if name in source:
            value = source[name]
        elif field.default is not dataclasses.MISSING:
            value = field.default
        elif field.default_factory is not dataclasses.MISSING:
            value = field.default_factory()
        else:
            raise ValueError(f"{field_path} is missing a mandatory value")

        expected_type = type_hints[name]
        if dataclasses.is_dataclass(expected_type):
            if dataclasses.is_dataclass(value):
                values[name] = value
            else:
                values[name] = _materialize_dataclass(
                    expected_type,
                    value,
                    field_path,
                )
        else:
            values[name] = _coerce_scalar(value, expected_type, field_path)
    return config_type(**values)


def _abs_or_missing(value: Any) -> Any:
    return value if value == MISSING else _abs_data_path(value)


def _build_source_dict(raw: dict[str, Any], mining_spec: dict[str, Any]) -> dict[str, Any]:
    _validate_raw_keys(raw)
    experiment = _mapping(raw.get("experiment"), "experiment")
    train_config = experiment.get("train_config", MISSING)
    eval_config = experiment.get("eval_config", MISSING)
    tao_pytorch_root = experiment.get("tao_pytorch_root", "")
    if not tao_pytorch_root:
        marker = "/nvidia_tao_pytorch/"
        for path in (train_config, eval_config):
            if path != MISSING and marker in str(path):
                tao_pytorch_root = str(path).split(marker, 1)[0]
                break

    visualization_raw = _mapping(raw.get("visualization"), "visualization")
    visualization = {
        "enabled": visualization_raw.get(
            "enabled", experiment.get("visualize", False)
        ),
        "embeddings": visualization_raw.get(
            "embeddings",
            experiment.get("visualize_embeddings", False),
        ),
    }
    for key in ("max_samples_per_group", "max_total_samples", "tile_size"):
        if key in visualization_raw:
            visualization[key] = visualization_raw[key]

    mining = _mapping(raw.get("mining"), "mining")
    if not isinstance(mining_spec, Mapping):
        raise ValueError("mining_spec.yaml root must be an object")
    for key in ("topn", "knn_metric"):
        if key not in mining_spec:
            raise ValueError(f"mining_spec.yaml is missing required key {key!r}")
        if key in mining and mining[key] != mining_spec[key]:
            raise ValueError(
                f"deft_config.mining.{key}={mining[key]!r} conflicts with "
                f"mining_spec.yaml {key}={mining_spec[key]!r}"
            )
    recovery = _mapping(mining.get("recovery"), "mining.recovery")
    expansion = _mapping(
        recovery.get("caption_expansion"),
        "mining.recovery.caption_expansion",
    )
    if "count_expanded_pairs_toward_target" in expansion:
        expansion["count_expanded_pairs_toward_target"] = str(
            expansion["count_expanded_pairs_toward_target"]
        ).lower()
    mining_out = {
        key: value
        for key, value in mining.items()
        if key not in {"history_aware", "recovery", "topn", "knn_metric"}
    }
    mining_out["history_aware"] = _mapping(
        mining.get("history_aware"),
        "mining.history_aware",
    )
    mining_out["caption_expansion"] = expansion
    mining_out["topn"] = mining_spec["topn"]
    mining_out["knn_metric"] = mining_spec["knn_metric"]

    pas = _mapping(raw.get("pas"), "pas")
    train_pairs = _abs_data_path(pas.get("train_pairs_source_file", ""))
    pool_pairs = _abs_data_path(pas.get("pool_pairs_source_file", "")) or train_pairs
    path_keys = {
        "train_pairs_source_file",
        "pool_pairs_source_file",
        "eval_pairs_source_file",
        "train_image_dir",
        "train_caption_dir",
        "source_image_dir",
        "source_caption_dir",
        "eval_image_dir",
        "eval_caption_dir",
    }
    pas_out = {key: value for key, value in pas.items() if key not in path_keys}
    pas_out.update(
        {
            "train_pairs_source_file": train_pairs,
            "pool_pairs_source_file": pool_pairs,
            "eval_pairs_source_file": _abs_or_missing(
                pas.get("eval_pairs_source_file", MISSING)
            ),
            "train_image_dir": _abs_or_missing(pas.get("train_image_dir", MISSING)),
            "train_caption_dir": _abs_or_missing(
                pas.get("train_caption_dir", MISSING)
            ),
            "source_image_dir": _abs_or_missing(pas.get("source_image_dir", MISSING)),
            "source_caption_dir": _abs_or_missing(
                pas.get("source_caption_dir", MISSING)
            ),
            "eval_image_dir": _abs_or_missing(pas.get("eval_image_dir", MISSING)),
            "eval_caption_dir": _abs_or_missing(pas.get("eval_caption_dir", MISSING)),
        }
    )

    gap = _mapping(raw.get("gap_analysis"), "gap_analysis")
    for canonical, legacy in (
        ("total_queries_map", "total_queries_mAP"),
        ("analyze_by_map", "analyze_by_mAP"),
    ):
        if canonical in gap and legacy in gap:
            canonical_value = gap[canonical]
            legacy_value = gap[legacy]
            if (
                type(canonical_value) is not type(legacy_value)
                or canonical_value != legacy_value
            ):
                raise ValueError(
                    f"gap_analysis.{canonical}={canonical_value!r} conflicts with "
                    f"legacy gap_analysis.{legacy}={legacy_value!r}"
                )
    gap_out = {
        key: value
        for key, value in gap.items()
        if key not in {"caption_diversity", "total_queries_mAP", "analyze_by_mAP"}
    }
    gap_out["caption_diversity"] = _mapping(
        gap.get("caption_diversity"),
        "gap_analysis.caption_diversity",
    )
    if "total_queries_mAP" in gap:
        gap_out["total_queries_map"] = gap["total_queries_mAP"]
    if "analyze_by_mAP" in gap:
        gap_out["analyze_by_map"] = gap["analyze_by_mAP"]

    return {
        "experiment": {
            "name": experiment.get("name", MISSING),
            "results_path": experiment.get("results_path", MISSING),
            "train_config": train_config,
            "eval_config": eval_config,
            "visualize": experiment.get("visualize", False),
            "visualize_embeddings": experiment.get("visualize_embeddings", False),
            "tao_pytorch_root": tao_pytorch_root,
        },
        "visualization": visualization,
        "iteration": _mapping(raw.get("iteration"), "iteration"),
        "training": _mapping(raw.get("training"), "training"),
        "mining": mining_out,
        "pas": pas_out,
        "gap_analysis": gap_out,
    }


def _validate_query_types(value: str, path: str) -> None:
    values = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(values) - set(QUERY_TYPES))
    if not values or invalid or len(values) != len(set(values)):
        detail = f"; unsupported={invalid}" if invalid else ""
        raise ValueError(
            f"{path} must be a non-empty, duplicate-free comma-separated subset "
            f"of {list(QUERY_TYPES)}{detail}"
        )


def config_field_metadata() -> dict[str, dict[str, Any]]:
    """Return field metadata keyed by dotted schema path for introspection."""
    result: dict[str, dict[str, Any]] = {}

    def visit(instance: Any, path: str = "") -> None:
        for field in dataclasses.fields(instance):
            value = getattr(instance, field.name)
            field_path = f"{path}.{field.name}" if path else field.name
            result[field_path] = dict(field.metadata)
            if dataclasses.is_dataclass(value):
                visit(value, field_path)

    visit(DeftExperimentConfig())
    return result


class PasDeftConfig:
    """Load and validate one immutable PAS DEFT configuration bundle."""
>>>>>>> 0ea1223 ([TAO-6655434][Bugfix] Rename DEFT workflow from IAA to PAS (#194)):skills/applications/tao-run-deft-pas/scripts/pas_deft/config.py

    CLIP_CKPT_RELPATH = "best/clip_best_val_t2i_mAP.pth"
    # Model-only copy of the above with the LightningModule "model." prefix
    # stripped, written by normalize_clip_pretrained_checkpoint. This is what
    # carries into the next iteration's train.pretrained_model_path; the raw
    # checkpoint above is what eval consumes.
    CLIP_PRETRAINED_RELPATH = "pretrained/model_state.pth"

    def __init__(
        self,
        config_path: str,
    ):
        self.config_path = config_path

        with open(config_path) as f:
            _cfg = yaml.safe_load(f)

        self.sweep_args_str: str = json.dumps({
            "config": config_path,
        })

        # ── Experiment ─────────────────────────────────────────────────────
        _exp = _cfg["experiment"]
        self.experiment_name: str = _exp["name"]
        self.base_experiment_path: str = _exp["results_path"]
        self.train_config: str = _exp["train_config"]
        self.eval_config: str = _exp["eval_config"]

        self.tao_pytorch_root: str = _exp.get("tao_pytorch_root", "")
        if not self.tao_pytorch_root:
            for _p in (self.train_config, self.eval_config):
                _marker = "/nvidia_tao_pytorch/"
                if _marker in _p:
                    self.tao_pytorch_root = _p.split(_marker, 1)[0]
                    break

        self.visualize: bool = bool(_exp.get("visualize", False))
        self.visualize_embeddings: bool = bool(_exp.get("visualize_embeddings", False))

        # ── Training ───────────────────────────────────────────────────────
        _train = _cfg["training"]
        self.init_checkpoint: str = _train["init_checkpoint"]
        self.continual_model: bool = bool(_train["continual_model"])
        self.continual_dataset: bool = bool(_train["continual_dataset"])

        # ── Mining ─────────────────────────────────────────────────────────
        _mining = _cfg["mining"]
        self.mining_topn: int = int(_mining.get("topn", 5) or 5)
        self.knn_metric: str = _mining.get("knn_metric", "cosine")

        _history_aware = _mining.get("history_aware", {}) or {}
        self.history_aware_enabled: str = _bool_str(_history_aware.get("enabled", False))
        self.history_aware_history_file: str = (
            f"{self.base_experiment_path}/mining_selection_history.json"
        )
<<<<<<< HEAD:skills/applications/tao-run-deft-iaa/scripts/iaa_deft/config.py
        self.history_aware_replay_fraction: float = float(
            _history_aware.get("replay_fraction", 0.20) or 0.0
=======
        if not os.path.isabs(self.experiment.results_path):
            raise ValueError("experiment.results_path must be an absolute skill result path")
        for field_name in (
            "eval_pairs_source_file",
            "train_image_dir",
            "train_caption_dir",
            "source_image_dir",
            "source_caption_dir",
            "eval_image_dir",
            "eval_caption_dir",
        ):
            value = getattr(self.pas, field_name)
            if not value or not os.path.isabs(value):
                raise ValueError(f"pas.{field_name} must be a non-empty absolute path")
        history_file = self.gap_analysis.caption_diversity.history_file
        if not history_file or pathlib.Path(history_file).is_absolute():
            raise ValueError(
                "gap_analysis.caption_diversity.history_file must be a non-empty "
                "result-relative path"
            )
        results_root = pathlib.Path(self.base_experiment_path).resolve()
        history_path = (results_root / history_file).resolve()
        if history_path == results_root or results_root not in history_path.parents:
            raise ValueError(
                "gap_analysis.caption_diversity.history_file must remain under "
                "experiment.results_path"
            )

    @property
    def base_experiment_path(self) -> str:
        return self.experiment.results_path

    @property
    def pas_splits_dir(self) -> str:
        return os.path.join(self.base_experiment_path, "pas_splits")

    @property
    def history_aware_history_file(self) -> str:
        return os.path.join(self.base_experiment_path, "mining_selection_history.json")

    @property
    def caption_history_file(self) -> str:
        return str(
            (
                pathlib.Path(self.base_experiment_path)
                / self.gap_analysis.caption_diversity.history_file
            ).resolve()
>>>>>>> 0ea1223 ([TAO-6655434][Bugfix] Rename DEFT workflow from IAA to PAS (#194)):skills/applications/tao-run-deft-pas/scripts/pas_deft/config.py
        )

        _cap_exp = (_mining.get("recovery") or {}).get("caption_expansion") or {}
        self.caption_expansion_enabled: str = _bool_str(_cap_exp.get("enabled", False))
        self.caption_expansion_mode: str = _cap_exp.get("mode", "nearest")
        self.caption_expansion_max_pairs_per_image_path: int = int(
            _cap_exp.get("max_pairs_per_image_path", 2) or 0
        )
        self.caption_expansion_max_expanded_pair_fraction: float = float(
            _cap_exp.get("max_expanded_pair_fraction", 0.25) or 0.0
        )
        self.caption_expansion_dedupe_normalized_caption: str = _bool_str(
            _cap_exp.get("dedupe_normalized_caption", True)
        )
        self.caption_expansion_count_expanded_pairs_toward_target: str = str(
            _cap_exp.get("count_expanded_pairs_toward_target", "auto")
        ).lower()

        # ── Visualization (contact sheets / t-SNE) ─────────────────────────
        _viz = _cfg.get("lepton_e2e", {}) or {}
        self.viz_max_samples_per_group: int = int(_viz.get("viz_max_samples_per_group", 12) or 12)
        self.viz_max_total_samples: int = int(_viz.get("viz_max_total_samples", 96) or 96)
        self.viz_tile_size: int = int(_viz.get("viz_tile_size", 192) or 192)

        # ── IAA ────────────────────────────────────────────────────────────
        _iaa = _cfg["iaa"]
        self.iaa_splits_dir: str = f"{self.base_experiment_path}/iaa_splits"
        self.iaa_seed_exclude_datasets: str = _iaa.get(
            "seed_exclude_datasets", "CUHK_PEDES,ICFG_PEDES"
        )
        self.iaa_augmented_suffix: str = _iaa.get("augmented_suffix", "_Aug")
        self.iaa_query_types: str = _iaa.get(
            "query_types", "easy,medium,hard,natural_caption,original_captions"
        )
        self.iaa_max_seed_rows: int = int(_iaa.get("max_seed_rows", 0) or 0)
        self.iaa_max_aug_pool_rows: int = int(_iaa.get("max_aug_pool_rows", 0) or 0)
        self.iaa_mining_pool_mode: str = _iaa.get("mining_pool_mode", "real_and_augmented")
        self.iaa_val_sample_size: int = int(_iaa.get("val_sample_size", 512) or 512)
        self.iaa_train_pairs_source_file: str = _abs_data_path(
            _iaa.get("train_pairs_source_file", "")
        )
        self.iaa_pool_pairs_source_file: str = (
            _abs_data_path(_iaa.get("pool_pairs_source_file", ""))
            or self.iaa_train_pairs_source_file
        )
        self.iaa_eval_pairs_source_file: str = _abs_data_path(
            _iaa["eval_pairs_source_file"]
        )
        self.iaa_train_image_dir: str = _abs_data_path(_iaa["train_image_dir"])
        self.iaa_train_caption_dir: str = _abs_data_path(_iaa["train_caption_dir"])
        self.iaa_source_image_dir: str = _abs_data_path(_iaa["source_image_dir"])
        self.iaa_source_caption_dir: str = _abs_data_path(_iaa["source_caption_dir"])
        self.iaa_eval_image_dir: str = _abs_data_path(_iaa["eval_image_dir"])
        self.iaa_eval_caption_dir: str = _abs_data_path(_iaa["eval_caption_dir"])

        # ── Gap analysis ───────────────────────────────────────────────────
        _gap = _cfg.get("gap_analysis", {}) or {}
        self.gap_metric_name: str = _gap.get("metric_name", "Rank-1")
        self.queries_per_slice: int = int(_gap.get("queries_per_slice", 256) or 0)
        self.min_gap_num_queries: int = int(_gap.get("min_num_queries", 1) or 0)
        self.gap_query_types: str = _gap.get("query_types", "easy,medium")
        self.weak_attribute_topk: int = int(_gap.get("weak_attribute_topk", 8) or 0)
        self.target_query_count: int = int(_gap.get("target_query_count", 100000) or 0)
        self.gap_total_queries_map: int = int(_gap.get("total_queries_mAP", 768) or 768)
        self.analyze_by_map: bool = bool(_gap.get("analyze_by_mAP", False))

        _cap_div = _gap.get("caption_diversity", {}) or {}
        self.caption_diversity_enabled: str = _bool_str(_cap_div.get("enabled", False))
        self.caption_history_file: str = (
            f"{self.base_experiment_path}/"
            f"{_cap_div.get('history_file', 'caption_selection_history.json')}"
        )
        self.caption_history_policy: str = _cap_div.get("history_policy", "auto")
        self.caption_coverage_target: float = float(
            _cap_div.get("coverage_target", 1.0) or 0.0
        )
        self.min_unique_texts_per_attribute: int = int(
            _cap_div.get("min_unique_texts_per_attribute", 0) or 0
        )
        self.max_unique_texts_per_attribute: int = int(
            _cap_div.get("max_unique_texts_per_attribute", 0) or 0
        )
        self.max_rows_per_unique_text: int = int(
            _cap_div.get("max_rows_per_unique_text", 1) or 1
        )
        self.max_rows_per_image_path: int = int(
            _cap_div.get("max_rows_per_image_path", 1) or 1
        )
        self.recent_exclude_iters: int = int(
            _cap_div.get("recent_exclude_iters", 0) or 0
        )
        self.replay_fraction_when_noncontinual: float = float(
            _cap_div.get("replay_fraction_when_noncontinual", 0.25) or 0.0
        )

        # ── Misc ───────────────────────────────────────────────────────────
        self.kratos_namespace: str = _cfg.get("kratos_namespace", "")
        self.iter_start: int = _cfg["iteration"]["start"]
        self.iter_end: int = _cfg["iteration"]["end"]

    # ──────────────────────────────────────────────────────────────────────

    def training_checkpoint_for_iter(self, iter_num: int) -> str:
        """Resolve the training checkpoint to use at the start of a DEFT iteration.

        Returns a *container* path (leading ``/``), since the value is written
        into the TAO spec as ``train.pretrained_model_path``. The normalized model-only
        checkpoint is used rather than the raw Lightning one.
        """
        if not self.continual_model:
            return self.init_checkpoint
        if iter_num == 1:
            host_ckpt = (
                f"{self.base_experiment_path}/sft/{self.CLIP_PRETRAINED_RELPATH}"
            )
            return (
                f"/{host_ckpt}"
                if (bool(self.iaa_train_pairs_source_file)
                    and os.path.exists(host_ckpt))
                else self.init_checkpoint
            )
        return (
            f"/{self.base_experiment_path}/iter_{iter_num - 1}"
            f"/{self.CLIP_PRETRAINED_RELPATH}"
        )

    def __repr__(self) -> str:
<<<<<<< HEAD:skills/applications/tao-run-deft-iaa/scripts/iaa_deft/config.py
        return (
            f"IaaDeftConfig(experiment={self.experiment_name!r}, "
            f"path={self.config_path!r})"
        )
=======
        return f"PasDeftConfig(experiment={self.experiment.name!r}, path={self.config_path!r})"
>>>>>>> 0ea1223 ([TAO-6655434][Bugfix] Rename DEFT workflow from IAA to PAS (#194)):skills/applications/tao-run-deft-pas/scripts/pas_deft/config.py
