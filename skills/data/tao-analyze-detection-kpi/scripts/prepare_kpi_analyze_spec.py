#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create and validate a TAO Data Services detection KPI-analyze spec."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

VALID_FORMATS = {"KITTI", "COCO"}
VALID_PLATFORMS = {"local", "wandb"}
SOURCE_KEYS = ("image_dir", "ground_truth_ann_path", "inference_ann_path")


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
    data = config.get("data") or {}

    fmt = str(data.get("input_format") or "")
    if fmt not in VALID_FORMATS:
        raise ValueError(f"data.input_format must be one of {sorted(VALID_FORMATS)} (uppercase).")

    sources = data.get("kpi_sources") or []
    if not sources:
        raise ValueError("data.kpi_sources must contain at least one source.")
    for idx, source in enumerate(sources):
        for key in SOURCE_KEYS:
            value = source.get(key)
            if not value:
                raise ValueError(f"data.kpi_sources[{idx}].{key} is required.")
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                raise ValueError(f"data.kpi_sources[{idx}].{key} must be absolute: {path}")
            if not path.exists():
                raise FileNotFoundError(f"data.kpi_sources[{idx}].{key} does not exist: {path}")
        # `Sequence Name` is image_dir.split('/')[-2]; a trailing slash shifts the pick.
        if str(source["image_dir"]).endswith("/"):
            raise ValueError(
                f"data.kpi_sources[{idx}].image_dir must not end with '/': the reported "
                "Sequence Name is derived from the second-to-last path component."
            )

    mapping = data.get("mapping")
    if not mapping:
        raise ValueError("data.mapping is required.")
    mapping_path = Path(str(mapping)).expanduser()
    if not mapping_path.is_absolute():
        raise ValueError(f"data.mapping must be absolute: {mapping_path}")
    if not mapping_path.is_file():
        raise FileNotFoundError(f"data.mapping does not exist or is not a file: {mapping_path}")

    results_dir = config.get("results_dir")
    if not results_dir:
        raise ValueError("results_dir is required.")
    out = Path(str(results_dir)).expanduser()
    if not out.is_absolute():
        raise ValueError(f"results_dir must be absolute: {out}")
    out.mkdir(parents=True, exist_ok=True)

    platform = str((config.get("visualize") or {}).get("platform", "local"))
    if platform not in VALID_PLATFORMS:
        raise ValueError(f"visualize.platform must be one of {sorted(VALID_PLATFORMS)}.")

    kpi = config.get("kpi") or {}
    for key in ("iou_threshold", "conf_threshold"):
        value = float(kpi.get(key, 0.5))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"kpi.{key} must be between 0.0 and 1.0.")
        kpi[key] = value
    if int(kpi.get("num_recall_points", 11)) < 1:
        raise ValueError("kpi.num_recall_points must be at least 1.")


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    skill_dir = Path(__file__).resolve().parents[1]
    default_template = skill_dir / "assets" / "default_kpi_analyze.yaml"
    template = Path(args.template).expanduser().resolve() if args.template else default_template
    config = load_yaml(template)

    if not (len(args.image_dir) == len(args.ground_truth_ann_path) == len(args.inference_ann_path)):
        raise ValueError(
            "--image-dir, --ground-truth-ann-path and --inference-ann-path must be repeated "
            "the same number of times; they are zipped in order into data.kpi_sources."
        )

    sources = []
    for image_dir, gt_path, pred_path in zip(
        args.image_dir, args.ground_truth_ann_path, args.inference_ann_path
    ):
        sources.append({
            "image_dir": absolute_path(image_dir, must_exist=True, kind="image_dir"),
            "ground_truth_ann_path": absolute_path(gt_path, must_exist=True, kind="ground_truth_ann_path"),
            "inference_ann_path": absolute_path(pred_path, must_exist=True, kind="inference_ann_path"),
        })

    config.setdefault("data", {})
    config["data"]["input_format"] = args.input_format
    config["data"]["kpi_sources"] = sources
    config["data"]["mapping"] = absolute_path(args.mapping, must_exist=True, kind="mapping")
    config["results_dir"] = absolute_path(args.results_dir, must_exist=False, kind="results_dir")

    config.setdefault("visualize", {})
    config["visualize"]["platform"] = args.platform
    if args.tag:
        config["visualize"]["tag"] = args.tag

    config.setdefault("kpi", {})
    config["kpi"]["iou_threshold"] = args.iou_threshold
    config["kpi"]["conf_threshold"] = args.conf_threshold
    config["kpi"]["num_recall_points"] = args.num_recall_points
    config["kpi"]["ignore_sqwidth"] = args.ignore_sqwidth
    config["kpi"]["is_internal"] = args.is_internal

    validate_config(config)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True, action="append",
                        help="KPI source image directory. Repeat once per source.")
    parser.add_argument("--ground-truth-ann-path", required=True, action="append",
                        help="Ground-truth annotations. Repeat once per source, in the same order.")
    parser.add_argument("--inference-ann-path", required=True, action="append",
                        help="Inference annotations. Repeat once per source, in the same order.")
    parser.add_argument("--mapping", required=True, help="Class-mapping YAML.")
    parser.add_argument("--results-dir", required=True, help="Output directory for kpi_calc.csv.")
    parser.add_argument("--output-spec", required=True, help="Path where the generated YAML is written.")
    parser.add_argument("--template", help="Optional YAML template. Defaults to assets/default_kpi_analyze.yaml.")
    parser.add_argument("--input-format", default="KITTI", choices=sorted(VALID_FORMATS))
    parser.add_argument("--platform", default="local", choices=sorted(VALID_PLATFORMS))
    parser.add_argument("--tag", default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--conf-threshold", type=float, default=0.3)
    parser.add_argument("--num-recall-points", type=int, default=11)
    parser.add_argument("--ignore-sqwidth", type=int, default=40)
    parser.add_argument("--is-internal", action="store_true",
                        help="Drop every class except person and append a Summary row.")
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        config = build_config(args)
        output_spec = Path(args.output_spec).expanduser().resolve()
        write_yaml(output_spec, config)
        print(f"Wrote kpi-analyze spec: {output_spec}")
        print(f"Sources: {len(config['data']['kpi_sources'])} | format: {config['data']['input_format']}")
        print(f"Expected output: {config['results_dir']}/kpi_calc.csv")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
