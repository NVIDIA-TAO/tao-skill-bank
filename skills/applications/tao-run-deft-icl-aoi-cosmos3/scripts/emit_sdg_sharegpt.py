#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Emit paired bare NG records from verified AnomalyGen 1.2 results."""
from __future__ import annotations
import argparse
import json
import pathlib
import sys
from collections import Counter
from validate_sharegpt import load_records, prompt_and_label
_shared_candidates = (pathlib.Path(__file__).resolve().parents[2] / "tao-generate-anomalies/scripts",
                      pathlib.Path(__file__).resolve().parents[3] / "data/tao-generate-anomalies/scripts")
sys.path.insert(0, str(next((p for p in _shared_candidates if p.is_dir()), _shared_candidates[-1])))
from anomalygen_consumer import load_generation, write_emission_receipt


def _inherited_prompt(path):
    prompts = {prompt_and_label(row, context=str(path))[0] for row in load_records(path)}
    if len(prompts) != 1:
        raise ValueError("Mining pool must contain one prompt; pass --prompt explicitly")
    return prompts.pop()


def emit_native(manifest, *, media_root, prompt, relative, exclude_stems=None):
    report, rows = load_generation(manifest, exclude_stems)
    records, defects = [], Counter()
    def formatted(value):
        path = pathlib.Path(value)
        return str(path.relative_to(media_root)) if relative else str(path)
    for row in rows:
        for key in ("paired_original", "paired_mask", "generated_image"):
            path = pathlib.Path(row[key])
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f"native paired artifact missing or empty: {path}")
        records.append({
            "images": [formatted(row["generated_image"]), formatted(row["paired_original"])],
            "conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": "NG"}],
            "anomalygen_lineage": row["lineage"],
        })
        defects[row["anomaly_type"]] += 1
    return records, {**report, "source": "anomalygen_sdg", "mode": "bare_okng", "generated_rows": len(rows), "output_records": len(records), "duplicates_skipped": 0, "labels": {"NG": len(records)}, "defect_types": dict(defects)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--media-root", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--summary", type=pathlib.Path)
    parser.add_argument("--exclude-stems", type=pathlib.Path)
    parser.add_argument("--emit-relative", action="store_true")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--prompt-from", type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        prompt = args.prompt if args.prompt is not None else _inherited_prompt(args.prompt_from)
        if not prompt.strip():
            raise ValueError("prompt must be non-empty")
        records, report = emit_native(args.manifest, media_root=args.media_root.resolve(), prompt=prompt, relative=args.emit_relative, exclude_stems=args.exclude_stems)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2) + "\n")
        (args.summary or args.output.with_name("emit_sdg_summary.json")).write_text(json.dumps(report, indent=2) + "\n")
        write_emission_receipt(args.output, args.manifest, report)
    except (OSError, ValueError, KeyError) as error:
        print(f"emit_sdg_sharegpt: {error}", file=sys.stderr)
        return 2
    print(f"emit_sdg_sharegpt: wrote {len(records)} synthetic NG records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
