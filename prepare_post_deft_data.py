#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build deterministic, duplicate-free post-DEFT mixture CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


FIELDS = ["input_path", "golden_path", "label", "object_name"]


def _normalize_path(value: str) -> str:
    normalized = value.strip().rstrip("/")
    if normalized.startswith("kpi/images/"):
        normalized = normalized[len("kpi/images/") :]
    return normalized


def _identity(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        _normalize_path(row["input_path"]),
        _normalize_path(row["golden_path"]),
        row["object_name"].strip(),
    )


def stable_deduplicate(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    seen: dict[tuple[str, str, str], str] = {}
    unique: list[dict[str, str]] = []
    rejected = 0
    for row in rows:
        identity = _identity(row)
        label = row["label"].strip()
        prior_label = seen.get(identity)
        if prior_label is None:
            seen[identity] = label
            unique.append(row)
        elif prior_label != label:
            raise ValueError(
                f"conflicting labels for {identity}: {prior_label!r} vs {label!r}"
            )
        else:
            rejected += 1
    return unique, rejected


def _label_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(sorted(Counter(row["label"] for row in rows).items()))


def build_mixtures(
    source: Path,
    output_dir: Path,
    ratios: tuple[float, ...],
    seed: int,
) -> dict:
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError(
                f"unexpected combined-CSV schema: {reader.fieldnames}; expected {FIELDS}"
            )
        input_rows = list(reader)

    rows, duplicate_rejected = stable_deduplicate(input_rows)
    base = [row for row in rows if row["input_path"].startswith("kpi/images/")]
    extra = [row for row in rows if not row["input_path"].startswith("kpi/images/")]
    if not base or not extra:
        raise ValueError(f"expected base and DEFT rows; got base={len(base)} extra={len(extra)}")

    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in extra:
        by_label[row["label"]].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": str(source.resolve()),
        "seed": seed,
        "input_rows": len(input_rows),
        "unique_rows": len(rows),
        "duplicate_rows_rejected": duplicate_rejected,
        "base_rows": len(base),
        "base_labels": _label_counts(base),
        "unique_deft_rows": len(extra),
        "unique_deft_labels": _label_counts(extra),
        "mixtures": {},
    }

    for ratio in ratios:
        if not 0 < ratio <= 1:
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")
        selected: list[dict[str, str]] = []
        for label, candidates in sorted(by_label.items()):
            shuffled = list(candidates)
            random.Random(f"{seed}:{ratio}:{label}").shuffle(shuffled)
            selected.extend(shuffled[: math.ceil(len(shuffled) * ratio)])
        random.Random(f"{seed}:{ratio}:merge").shuffle(selected)
        output_rows = base + selected
        suffix = round(ratio * 100)
        output = output_dir / f"train_unique_mix_{suffix:03d}.csv"
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(output_rows)
        manifest["mixtures"][f"mix_{suffix:03d}"] = {
            "path": str(output.resolve()),
            "ratio": ratio,
            "rows": len(output_rows),
            "labels": _label_counts(output_rows),
        }

    manifest_path = output_dir / "post_deft_data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ratios", default="0.25,0.50,0.75,1.00")
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args(argv)
    ratios = tuple(float(value) for value in args.ratios.split(","))
    manifest = build_mixtures(args.source, args.output_dir, ratios, args.seed)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
