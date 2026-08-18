# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Bundled and adapted from the Apache-2.0 NVIDIA TAO Tutorials IAA DEFT utilities so the
# customer workflow does not depend on an external source checkout.

"""Parsed configuration for an IAA CLIP DEFT experiment."""

import json
import math
import os
import yaml


_MISSING = object()


def _error(config_path: str, key: str, message: str) -> None:
    raise ValueError(f"{config_path}: {key} {message}")


def _mapping(root, key: str, source: str, config_path: str, *, required=True):
    value = root.get(key, _MISSING)
    qualified = f"{source}.{key}" if source else key
    if value is _MISSING:
        if required:
            _error(config_path, qualified, "is required")
        return {}
    if not isinstance(value, dict):
        _error(config_path, qualified, "must be an object")
    return value


def _known(mapping, allowed, source: str, config_path: str) -> None:
    unknown = sorted(set(mapping) - set(allowed), key=repr)
    if unknown:
        _error(
            config_path,
            f"{source}.{unknown[0]}" if source else unknown[0],
            "is not a recognized setting",
        )


def _value(mapping, key: str, source: str, config_path: str, *, required=False):
    value = mapping.get(key, _MISSING)
    qualified = f"{source}.{key}" if source else key
    if value is _MISSING and required:
        _error(config_path, qualified, "is required")
    return value, qualified


def _string(
    mapping,
    key: str,
    source: str,
    config_path: str,
    *,
    required=False,
    allow_empty=False,
    choices=None,
) -> None:
    value, qualified = _value(
        mapping, key, source, config_path, required=required
    )
    if value is _MISSING:
        return
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        suffix = "string" if allow_empty else "non-empty string"
        _error(config_path, qualified, f"must be a {suffix}; got {value!r}")
    if choices is not None and value not in choices:
        _error(
            config_path,
            qualified,
            f"must be one of {', '.join(sorted(choices))}; got {value!r}",
        )


def _boolean(mapping, key: str, source: str, config_path: str, *, required=False):
    value, qualified = _value(
        mapping, key, source, config_path, required=required
    )
    if value is _MISSING:
        return
    if not isinstance(value, bool):
        _error(config_path, qualified, f"must be a boolean; got {value!r}")


def _integer(
    mapping,
    key: str,
    source: str,
    config_path: str,
    *,
    required=False,
    minimum=None,
) -> None:
    value, qualified = _value(
        mapping, key, source, config_path, required=required
    )
    if value is _MISSING:
        return
    if not isinstance(value, int) or isinstance(value, bool):
        _error(config_path, qualified, f"must be an integer; got {value!r}")
    if minimum is not None and value < minimum:
        _error(config_path, qualified, f"must be >= {minimum}; got {value!r}")


def _number(
    mapping,
    key: str,
    source: str,
    config_path: str,
    *,
    minimum=None,
    maximum=None,
) -> None:
    value, qualified = _value(mapping, key, source, config_path)
    if value is _MISSING:
        return
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        _error(config_path, qualified, f"must be a finite number; got {value!r}")
    if minimum is not None and value < minimum:
        _error(config_path, qualified, f"must be >= {minimum}; got {value!r}")
    if maximum is not None and value > maximum:
        _error(config_path, qualified, f"must be <= {maximum}; got {value!r}")


def _validate_config(root, config_path: str) -> None:
    if not isinstance(root, dict):
        _error(config_path, "<root>", "must be an object")
    _known(
        root,
        {
            "experiment",
            "iteration",
            "training",
            "mining",
            "gap_analysis",
            "iaa",
            "visualization",
            "kratos_namespace",
        },
        "",
        config_path,
    )

    experiment = _mapping(root, "experiment", "", config_path)
    _known(
        experiment,
        {
            "name",
            "results_path",
            "train_config",
            "eval_config",
            "tao_pytorch_root",
            "visualize",
            "visualize_embeddings",
        },
        "experiment",
        config_path,
    )
    for key in ("name", "results_path", "train_config", "eval_config"):
        _string(experiment, key, "experiment", config_path, required=True)
    _string(
        experiment, "tao_pytorch_root", "experiment", config_path, allow_empty=True
    )
    for key in ("visualize", "visualize_embeddings"):
        _boolean(experiment, key, "experiment", config_path)

    iteration = _mapping(root, "iteration", "", config_path)
    _known(iteration, {"start", "end"}, "iteration", config_path)
    _integer(iteration, "start", "iteration", config_path, required=True, minimum=1)
    _integer(iteration, "end", "iteration", config_path, required=True, minimum=1)
    if iteration["end"] < iteration["start"]:
        _error(config_path, "iteration.end", "must be >= iteration.start")

    training = _mapping(root, "training", "", config_path)
    _known(
        training,
        {"init_checkpoint", "continual_model", "continual_dataset", "num_nodes"},
        "training",
        config_path,
    )
    _string(
        training,
        "init_checkpoint",
        "training",
        config_path,
        required=True,
        allow_empty=True,
    )
    _boolean(training, "continual_model", "training", config_path, required=True)
    _boolean(training, "continual_dataset", "training", config_path, required=True)
    _integer(training, "num_nodes", "training", config_path, minimum=1)

    mining = _mapping(root, "mining", "", config_path)
    _known(
        mining,
        {"topn", "knn_metric", "knn_batch_size", "history_aware", "recovery"},
        "mining",
        config_path,
    )
    _integer(mining, "topn", "mining", config_path, minimum=1)
    _string(
        mining,
        "knn_metric",
        "mining",
        config_path,
        choices={"cosine", "euclidean"},
    )
    _integer(mining, "knn_batch_size", "mining", config_path, minimum=1)
    history = _mapping(mining, "history_aware", "mining", config_path, required=False)
    _known(history, {"enabled", "replay_fraction"}, "mining.history_aware", config_path)
    _boolean(history, "enabled", "mining.history_aware", config_path)
    _number(
        history,
        "replay_fraction",
        "mining.history_aware",
        config_path,
        minimum=0.0,
        maximum=1.0,
    )
    recovery = _mapping(mining, "recovery", "mining", config_path, required=False)
    _known(recovery, {"caption_expansion"}, "mining.recovery", config_path)
    expansion = _mapping(
        recovery,
        "caption_expansion",
        "mining.recovery",
        config_path,
        required=False,
    )
    _known(
        expansion,
        {
            "enabled",
            "mode",
            "max_pairs_per_image_path",
            "max_expanded_pair_fraction",
            "dedupe_normalized_caption",
            "count_expanded_pairs_toward_target",
        },
        "mining.recovery.caption_expansion",
        config_path,
    )
    expansion_path = "mining.recovery.caption_expansion"
    _boolean(expansion, "enabled", expansion_path, config_path)
    _string(
        expansion,
        "mode",
        expansion_path,
        config_path,
        choices={"nearest", "all"},
    )
    _integer(
        expansion, "max_pairs_per_image_path", expansion_path, config_path, minimum=1
    )
    _number(
        expansion,
        "max_expanded_pair_fraction",
        expansion_path,
        config_path,
        minimum=0.0,
        maximum=1.0,
    )
    _boolean(expansion, "dedupe_normalized_caption", expansion_path, config_path)
    count_value, count_key = _value(
        expansion, "count_expanded_pairs_toward_target", expansion_path, config_path
    )
    if count_value is not _MISSING and not (
        isinstance(count_value, bool)
        or (
            isinstance(count_value, str)
            and count_value in {"auto", "true", "false"}
        )
    ):
        _error(config_path, count_key, "must be auto, true, false, or a boolean")

    visualization = _mapping(root, "visualization", "", config_path, required=False)
    viz_keys = {
        "viz_max_samples_per_group",
        "viz_max_total_samples",
        "viz_tile_size",
    }
    _known(visualization, viz_keys, "visualization", config_path)
    for key in viz_keys:
        _integer(visualization, key, "visualization", config_path, minimum=1)

    iaa = _mapping(root, "iaa", "", config_path)
    iaa_keys = {
        "train_pairs_source_file",
        "pool_pairs_source_file",
        "eval_pairs_source_file",
        "train_image_dir",
        "train_caption_dir",
        "source_image_dir",
        "source_caption_dir",
        "eval_image_dir",
        "eval_caption_dir",
        "seed_exclude_datasets",
        "augmented_suffix",
        "query_types",
        "mining_pool_mode",
        "max_seed_rows",
        "max_aug_pool_rows",
        "val_sample_size",
    }
    _known(iaa, iaa_keys, "iaa", config_path)
    _string(
        iaa, "train_pairs_source_file", "iaa", config_path, allow_empty=True
    )
    _string(iaa, "pool_pairs_source_file", "iaa", config_path)
    for key in (
        "eval_pairs_source_file",
        "train_image_dir",
        "train_caption_dir",
        "source_image_dir",
        "source_caption_dir",
        "eval_image_dir",
        "eval_caption_dir",
    ):
        _string(iaa, key, "iaa", config_path, required=True)
    for key in ("seed_exclude_datasets", "augmented_suffix", "query_types"):
        _string(iaa, key, "iaa", config_path)
    _string(
        iaa,
        "mining_pool_mode",
        "iaa",
        config_path,
        choices={"real", "augmented", "real_and_augmented"},
    )
    _integer(iaa, "max_seed_rows", "iaa", config_path, minimum=0)
    _integer(iaa, "max_aug_pool_rows", "iaa", config_path, minimum=0)
    _integer(iaa, "val_sample_size", "iaa", config_path, minimum=1)
    if not str(iaa.get("pool_pairs_source_file") or "").strip() and not str(
        iaa.get("train_pairs_source_file") or ""
    ).strip():
        _error(
            config_path,
            "iaa.pool_pairs_source_file",
            "or iaa.train_pairs_source_file must be a non-empty string",
        )

    gap = _mapping(root, "gap_analysis", "", config_path, required=False)
    gap_keys = {
        "metric_name",
        "queries_per_slice",
        "min_num_queries",
        "query_types",
        "weak_attribute_topk",
        "target_query_count",
        "total_queries_mAP",
        "analyze_by_mAP",
        "caption_diversity",
    }
    _known(gap, gap_keys, "gap_analysis", config_path)
    for key in ("metric_name", "query_types"):
        _string(gap, key, "gap_analysis", config_path)
    _integer(gap, "queries_per_slice", "gap_analysis", config_path, minimum=1)
    _integer(gap, "min_num_queries", "gap_analysis", config_path, minimum=0)
    _integer(gap, "weak_attribute_topk", "gap_analysis", config_path, minimum=1)
    _integer(gap, "target_query_count", "gap_analysis", config_path, minimum=1)
    _integer(gap, "total_queries_mAP", "gap_analysis", config_path, minimum=1)
    _boolean(gap, "analyze_by_mAP", "gap_analysis", config_path)
    diversity = _mapping(
        gap, "caption_diversity", "gap_analysis", config_path, required=False
    )
    diversity_path = "gap_analysis.caption_diversity"
    diversity_keys = {
        "enabled",
        "history_file",
        "history_policy",
        "coverage_target",
        "min_unique_texts_per_attribute",
        "max_unique_texts_per_attribute",
        "max_rows_per_unique_text",
        "max_rows_per_image_path",
        "recent_exclude_iters",
        "replay_fraction_when_noncontinual",
    }
    _known(diversity, diversity_keys, diversity_path, config_path)
    _boolean(diversity, "enabled", diversity_path, config_path)
    _string(diversity, "history_file", diversity_path, config_path)
    _string(
        diversity,
        "history_policy",
        diversity_path,
        config_path,
        choices={"auto", "prefer_unseen", "novelty_with_replay"},
    )
    _number(
        diversity,
        "coverage_target",
        diversity_path,
        config_path,
        minimum=0.0,
        maximum=1.0,
    )
    for key in (
        "min_unique_texts_per_attribute",
        "max_unique_texts_per_attribute",
        "recent_exclude_iters",
    ):
        _integer(diversity, key, diversity_path, config_path, minimum=0)
    for key in ("max_rows_per_unique_text", "max_rows_per_image_path"):
        _integer(diversity, key, diversity_path, config_path, minimum=1)
    _number(
        diversity,
        "replay_fraction_when_noncontinual",
        diversity_path,
        config_path,
        minimum=0.0,
        maximum=1.0,
    )
    _string(root, "kratos_namespace", "", config_path, allow_empty=True)


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

        cfg = IaaDeftConfig("configs/clip_config.yaml")
    """

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
        _validate_config(_cfg, config_path)

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
        self.history_aware_replay_fraction: float = float(
            _history_aware.get("replay_fraction", 0.20) or 0.0
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
        _viz = _cfg.get("visualization", {}) or {}
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
        valid_pool_modes = {"real", "augmented", "real_and_augmented"}
        if self.iaa_mining_pool_mode not in valid_pool_modes:
            choices = ", ".join(sorted(valid_pool_modes))
            raise ValueError(
                f"{self.config_path}: iaa.mining_pool_mode must be one of "
                f"{choices}; got {self.iaa_mining_pool_mode!r}"
            )
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
        return (
            f"IaaDeftConfig(experiment={self.experiment_name!r}, "
            f"path={self.config_path!r})"
        )
