#!/usr/bin/env python3
"""Validate an SDG inference JSONL against a trained checkpoint.

Checks:
  - Every anomaly_type appears in the checkpoint's supported list
  - image_filename and mask_filename paths exist on disk

Fails fast (non-zero exit) when any anomaly_type is unsupported. Missing-file
issues are reported as warnings only so the user can fix paths and retry.

Usage: scripts/validate_jsonl.py <checkpoint_dir> <input_jsonl>
"""
import argparse
import json
import os
import sys

try:
    import yaml
except ImportError:
    print("error: PyYAML is required (pip install pyyaml)", file=sys.stderr)
    sys.exit(2)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("checkpoint_dir")
    p.add_argument("input_jsonl")
    p.add_argument("--max-warnings", type=int, default=20,
                   help="Cap the number of missing-file warnings printed.")
    args = p.parse_args()

    cfg_path = os.path.join(args.checkpoint_dir, "ag_config.yaml")
    if not os.path.isfile(cfg_path):
        print(f"error: {cfg_path} not found", file=sys.stderr)
        return 1
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    supported = {f"{t[0]}+{t[1]}"
                 for t in cfg["dataloader_train"]["dataset"]["anomaly_types"]}

    jsonl_types: set = set()
    entry_count = 0
    missing_files = 0
    warnings_printed = 0
    with open(args.input_jsonl) as f:
        for line_i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"error: line {line_i} is not valid JSON: {e}",
                      file=sys.stderr)
                return 1
            entry_count += 1
            jsonl_types.add(entry["anomaly_type"])
            for key in ("image_filename", "mask_filename"):
                path = entry[key]
                if not os.path.exists(path):
                    missing_files += 1
                    if warnings_printed < args.max_warnings:
                        print(f"warning: missing {key}={path} (line {line_i})",
                              file=sys.stderr)
                        warnings_printed += 1

    unsupported = jsonl_types - supported
    if unsupported:
        print(f"error: JSONL contains {len(unsupported)} anomaly type(s) "
              f"not in the checkpoint: {sorted(unsupported)}",
              file=sys.stderr)
        print(f"supported: {sorted(supported)}", file=sys.stderr)
        return 1

    if missing_files > args.max_warnings:
        print(f"warning: total missing files: {missing_files} "
              f"(printed first {args.max_warnings})", file=sys.stderr)

    print(f"OK: {entry_count} entries, "
          f"{len(jsonl_types)} anomaly type(s), "
          f"{missing_files} missing files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
