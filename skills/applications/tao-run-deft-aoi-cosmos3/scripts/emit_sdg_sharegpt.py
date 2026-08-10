#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Turn AnomalyGen SDG output into bare OK/NG ShareGPT records.

AnomalyGen writes one synthetic defect per row of ``SDG_result.csv``: a
``reconstructed_image/<T>+<A>_<idx>.png`` carrying the generated defect, and
the ``original_image/<T>+<A>_<idx>.png`` clean board it was painted onto. That
is exactly the Cosmos3 AOI pair shape ``[AOI, golden_reference]``, so every
generated sample becomes one ``NG`` record.

The prompt is never invented. It is inherited from the recorded Mining pool so
that synthetic and mined records ask the model the same question, or supplied
verbatim with ``--prompt``.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from collections import Counter
from typing import Any

from validate_sharegpt import load_records, prompt_and_label

# AnomalyGen names every output "<texture>+<anomaly>_<index>.<ext>". The paired
# clean board reuses the stem verbatim under original_image/.
# Older paidf-anomalygen releases call the generated-defect column "output_filename".
RECONSTRUCTED_COLUMNS = (
    "reconstructed_image",
    "reconstructed",
    "image",
    "sdg_image",
    "output_filename",
)
# Deliberately excludes 1.0.0's "image_filename": that points at the dataset
# clean_image, not the paired copy under original_image/. Adding it would
# silently change which image lands in images[1]. The stem fallback below is
# the correct source for that container.
ORIGINAL_COLUMNS = ("original_image", "original", "clean_image", "golden_image")


def _column(row: dict[str, str], candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _defect_type(stem: str) -> str:
    head, sep, tail = stem.partition("+")
    if not sep:
        return "unknown"
    return tail.rsplit("_", 1)[0] or "unknown"


def _inherited_prompt(path: pathlib.Path) -> str:
    """Take the single prompt shared by every record of the Mining pool."""
    records = load_records(path)
    prompts = {
        prompt_and_label(record, context=f"{path}[{index}]")[0]
        for index, record in enumerate(records)
    }
    if len(prompts) != 1:
        raise ValueError(
            f"{path}: cannot inherit one inspection prompt from {len(prompts)} "
            "distinct prompts; pass --prompt explicitly"
        )
    return prompts.pop()


def _resolve_pair(
    row: dict[str, str],
    *,
    sdg_dir: pathlib.Path,
    row_number: int,
) -> tuple[pathlib.Path, pathlib.Path]:
    reconstructed = _column(row, RECONSTRUCTED_COLUMNS)
    if reconstructed is None:
        raise ValueError(
            f"SDG_result.csv row {row_number}: no reconstructed-image column "
            f"among {list(RECONSTRUCTED_COLUMNS)}"
        )
    generated = pathlib.Path(reconstructed)
    if not generated.is_absolute():
        generated = sdg_dir / generated
    generated = generated.resolve()

    original = _column(row, ORIGINAL_COLUMNS)
    if original is not None:
        clean = pathlib.Path(original)
        if not clean.is_absolute():
            clean = sdg_dir / clean
    else:
        # No explicit column: AnomalyGen mirrors the stem into original_image/.
        clean = sdg_dir / "original_image" / generated.name
    clean = clean.resolve()

    for role, path in (("reconstructed", generated), ("original", clean)):
        if not path.is_file():
            raise ValueError(
                f"SDG_result.csv row {row_number}: {role} image does not exist: {path}"
            )
        if path.stat().st_size == 0:
            raise ValueError(
                f"SDG_result.csv row {row_number}: {role} image is empty: {path}"
            )
    return generated, clean


def _format(path: pathlib.Path, *, media_root: pathlib.Path, relative: bool) -> str:
    if not relative:
        return str(path)
    try:
        return path.relative_to(media_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"cannot emit relative path outside media root {media_root}: {path}"
        ) from exc


def emit_records(
    sdg_csv: pathlib.Path,
    *,
    media_root: pathlib.Path,
    prompt: str,
    relative: bool,
    label: str = "NG",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sdg_dir = sdg_csv.parent.resolve()
    with sdg_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{sdg_csv}: no generated samples")

    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates = 0
    defects: Counter[str] = Counter()
    for offset, row in enumerate(rows, start=2):
        generated, clean = _resolve_pair(row, sdg_dir=sdg_dir, row_number=offset)
        key = str(generated)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        defects[_defect_type(generated.stem)] += 1
        output.append(
            {
                "images": [
                    _format(generated, media_root=media_root, relative=relative),
                    _format(clean, media_root=media_root, relative=relative),
                ],
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": label},
                ],
            }
        )
    if not output:
        raise ValueError(f"{sdg_csv}: no unique generated samples were emitted")
    return output, {
        "mode": "bare_okng",
        "source": "anomalygen_sdg",
        "sdg_result_csv": str(sdg_csv),
        "generated_rows": len(rows),
        "output_records": len(output),
        "duplicates_skipped": duplicates,
        "labels": {label: len(output)},
        "defect_types": dict(sorted(defects.items())),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sdg-csv",
        required=True,
        type=pathlib.Path,
        help="AnomalyGen SDG_result.csv under iterN/anomalygen/sdg/.",
    )
    parser.add_argument("--media-root", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--summary", default=None, type=pathlib.Path)
    parser.add_argument("--emit-relative", action="store_true")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--prompt",
        help="Exact inspection prompt for every generated record.",
    )
    source.add_argument(
        "--prompt-from",
        type=pathlib.Path,
        help="Annotation JSON (normally the Mining pool) to inherit the prompt from.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        prompt = (
            args.prompt.strip()
            if args.prompt is not None
            else _inherited_prompt(args.prompt_from.expanduser().resolve())
        )
        if not prompt:
            raise ValueError("--prompt must be non-empty")
        records, summary = emit_records(
            args.sdg_csv.expanduser().resolve(),
            media_root=args.media_root.expanduser().resolve(),
            prompt=prompt,
            relative=args.emit_relative,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2) + "\n")
        summary_path = args.summary or args.output.with_name("emit_sdg_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"emit_sdg_sharegpt: {exc}", file=sys.stderr)
        return 2
    print(
        f"emit_sdg_sharegpt: wrote {len(records)} synthetic NG records to "
        f"{args.output} ({summary['defect_types']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
