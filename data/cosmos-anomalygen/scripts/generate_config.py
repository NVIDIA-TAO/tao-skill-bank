#!/usr/bin/env python3
"""Render the AnomalyGen training YAML from a reference template.

Reads the `references/ag_config.yaml` template and substitutes
the placeholders in angle brackets with user values. Anomaly types are read
from --defect-spec (one JSONL entry per type with "defect_type" key).
"""
import argparse
import json
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
TEMPLATES = HERE.parent / "references"


def _types_from_description(path):
    types = []
    for line in pathlib.Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        full = json.loads(line)["defect_type"]
        texture, anomaly = full.split("+", 1)
        types.append([texture, anomaly])
    return types


def _render(template_path, subs):
    text = template_path.read_text()
    for key, val in subs.items():
        text = text.replace(f"<{key}>", val)
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--dataset-dir", required=True, type=pathlib.Path)
    p.add_argument("--defect-spec", required=True, type=pathlib.Path,
                   help="JSONL with a 'defect_type' key per line (TEXTURE+ANOMALY).")
    p.add_argument("--validation-jsonl", required=True)
    p.add_argument("--model-size", default="2b", choices=["2b", "14b"])
    p.add_argument("--max-iter", type=int, default=75000)
    p.add_argument("--save-iter", type=int, default=5000)
    p.add_argument("--validation-iter", type=int, default=5000)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--aug-type", default="random_ratio_crop",
                   help="Crop strategy passed to the dataset. Defaults to "
                        "'random_ratio_crop'. Pass 'null' (or empty) to disable.")
    p.add_argument("--ratio-range", nargs=2, type=float, metavar=("MIN", "MAX"),
                   default=[1.5, 8.0],
                   help="Min/max crop ratios; only used when --aug-type='random_ratio_crop'. "
                        "Defaults to [1.5, 8.0].")
    # Early-stop knobs. Off by default; opt in via --early-stop. The metric
    # (nn), min_delta (0), and min_delta_mode (rel) are fixed in the YAML
    # template; scope and cumulative_delta are left to upstream defaults
    # (Average / False). See cosmos_predict2/configs/anomaly_gen/early_stop.py.
    p.add_argument("--early-stop", action="store_true",
                   help="Enable early stopping on validation nn_score (key KPI).")
    p.add_argument("--es-patience", type=int, default=5,
                   help="Consecutive validations without improvement before stopping. Default 5.")
    p.add_argument("--output", required=True, type=pathlib.Path,
                   help="Where to write the rendered YAML (e.g., ag_configs/<name>.yaml).")
    args = p.parse_args()

    anomaly_types = _types_from_description(args.defect_spec)
    if not anomaly_types:
        print(f"error: no defect_type entries in {args.defect_spec}", file=sys.stderr)
        sys.exit(1)

    subs = {
        "NAME": args.name,
        "DATASET_DIR": str(args.dataset_dir),
        "VALIDATION_JSONL": args.validation_jsonl,
        "MAX_ITER": str(args.max_iter),
        "SAVE_ITER": str(args.save_iter),
        "VALIDATION_ITER": str(args.validation_iter),
        "IMAGE_SIZE": str(args.image_size),
        "LR": str(args.lr),
        "BATCH_SIZE": str(args.batch_size),
        "MODEL_SIZE": args.model_size.upper(),
        "ANOMALY_TYPES": yaml.safe_dump(anomaly_types, default_flow_style=True).rstrip(),
        "AUG_TYPE": "null" if args.aug_type is None else args.aug_type,
        "RATIO_RANGE": (
            "null" if args.ratio_range is None
            else f"[{args.ratio_range[0]}, {args.ratio_range[1]}]"
        ),
        "EARLY_STOP_ENABLED":  "true" if args.early_stop else "false",
        "EARLY_STOP_PATIENCE": str(args.es_patience),
    }

    rendered = _render(TEMPLATES / "ag_config.yaml", subs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(f"wrote {args.output} (types={len(anomaly_types)})")


if __name__ == "__main__":
    main()
