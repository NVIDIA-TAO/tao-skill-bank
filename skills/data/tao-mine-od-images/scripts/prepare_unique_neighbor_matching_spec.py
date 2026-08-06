#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create and validate a TAO Data Services TMM unique-neighbor-matching spec."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

VALID_METRICS = {"euclidean", "cosine", "manhattan"}
VALID_POLICIES = {"global", "class_stratified"}
VALID_FORMATS = {"coco", "kitti"}


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required: install with `python3 -m pip install pyyaml`.")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required: install with `python3 -m pip install pyyaml`.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def absolute_path(raw: str, *, must_exist: bool, kind: str) -> str:
    path = Path(raw).expanduser().resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{kind} does not exist: {path}")
    return str(path)


def validate_config(config: dict[str, Any]) -> None:
    required = ("source_path", "target_path", "output_dir", "desired_unique_count")
    missing = [k for k in required if not config.get(k)]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    for key in ("source_path", "target_path"):
        p = Path(str(config[key])).expanduser()
        if not p.is_absolute():
            raise ValueError(f"{key} must be absolute: {p}")
        if not p.exists():
            raise FileNotFoundError(f"{key} does not exist: {p}")

    out = Path(str(config["output_dir"])).expanduser()
    if not out.is_absolute():
        raise ValueError(f"output_dir must be absolute: {out}")
    out.mkdir(parents=True, exist_ok=True)

    count = int(config["desired_unique_count"])
    if count < 1:
        raise ValueError("desired_unique_count must be at least 1.")
    config["desired_unique_count"] = count

    policy = str(config.get("allocation_policy", "global"))
    if policy not in VALID_POLICIES:
        raise ValueError(f"allocation_policy must be one of {sorted(VALID_POLICIES)}.")
    config["allocation_policy"] = policy

    metric = str(config.get("distance_metric", "euclidean"))
    if metric not in VALID_METRICS:
        raise ValueError(f"distance_metric must be one of {sorted(VALID_METRICS)}.")
    config["distance_metric"] = metric

    det_src = config.get("source_detection_file")
    det_tgt = config.get("target_detection_file")
    det_fmt = config.get("detection_format")
    if (det_src or det_tgt) and not det_fmt:
        raise ValueError("detection_format (coco or kitti) is required when a detection file is set.")
    if det_fmt and det_fmt not in VALID_FORMATS:
        raise ValueError(f"detection_format must be one of {sorted(VALID_FORMATS)}.")

    if policy == "class_stratified":
        if not config.get("rare_class_list"):
            raise ValueError("rare_class_list is required when allocation_policy is class_stratified.")
        if not det_src or not det_tgt:
            raise ValueError("source_detection_file and target_detection_file are required for class_stratified.")
        if not det_fmt:
            raise ValueError("detection_format is required for class_stratified.")


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    skill_dir = Path(__file__).resolve().parents[1]
    default_template = skill_dir / "assets" / "default_unique_neighbor_matching.yaml"
    template = Path(args.template).expanduser().resolve() if args.template else default_template
    config = load_yaml(template)
    config.update({
        "source_path": absolute_path(args.source_path, must_exist=True, kind="source_path"),
        "target_path": absolute_path(args.target_path, must_exist=True, kind="target_path"),
        "output_dir": absolute_path(args.output_dir, must_exist=False, kind="output_dir"),
        "desired_unique_count": args.desired_unique_count,
        "allocation_policy": args.allocation_policy,
        "distance_metric": args.distance_metric,
        "candidate_expansion_factor": args.candidate_expansion_factor,
    })
    if args.source_detection_file:
        config["source_detection_file"] = absolute_path(args.source_detection_file, must_exist=True, kind="source_detection_file")
    if args.target_detection_file:
        config["target_detection_file"] = absolute_path(args.target_detection_file, must_exist=True, kind="target_detection_file")
    if args.detection_format:
        config["detection_format"] = args.detection_format
    if args.rare_class_list:
        config["rare_class_list"] = args.rare_class_list
    validate_config(config)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", required=True, help="Absolute path to source embeddings parquet or directory.")
    parser.add_argument("--target-path", required=True, help="Absolute path to target embeddings parquet or directory.")
    parser.add_argument("--output-dir", required=True, help="Absolute path for the output directory.")
    parser.add_argument("--desired-unique-count", required=True, type=int, help="Total unique source files to retrieve.")
    parser.add_argument("--output-spec", required=True, help="Path where the generated YAML should be written.")
    parser.add_argument("--template", help="Optional YAML template. Defaults to assets/default_unique_neighbor_matching.yaml.")
    parser.add_argument("--allocation-policy", default="global", choices=sorted(VALID_POLICIES))
    parser.add_argument("--distance-metric", default="euclidean", choices=sorted(VALID_METRICS))
    parser.add_argument("--candidate-expansion-factor", type=int, default=5)
    parser.add_argument("--source-detection-file", default=None, help="COCO .json or KITTI label directory for source.")
    parser.add_argument("--target-detection-file", default=None, help="COCO .json or KITTI label directory for target.")
    parser.add_argument("--detection-format", default=None, choices=sorted(VALID_FORMATS), help="coco or kitti.")
    parser.add_argument("--rare-class-list", default="", help="Comma-separated rare class names, e.g. 'person,bicycle'.")
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        config = build_config(args)
        output_spec = Path(args.output_spec).expanduser().resolve()
        write_yaml(output_spec, config)
        print(f"Wrote unique-neighbor-matching spec: {output_spec}")
        print(f"Output directory: {config['output_dir']}")
        print(f"Expected artifacts: final_unique_files.parquet, summary.json")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
