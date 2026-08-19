#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Turn AnomalyGen SDG output into profile-aware ShareGPT records.

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
import re
import sys
import copy
from collections import Counter
from typing import Any

from nvpaw_annotations import filesystem_safe_id
from validate_sharegpt import load_records, prompt_and_label, prompt_and_response

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
    media_root = media_root.resolve()
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
    annotation_profile: str = "bare_okng",
    template_record: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if annotation_profile not in {"bare_okng", "nvpaw_multitask_v1"}:
        raise ValueError(f"unsupported annotation profile {annotation_profile!r}")
    if annotation_profile == "nvpaw_multitask_v1":
        if not isinstance(template_record, dict):
            raise ValueError("rich SDG emission requires a classification template record")
        task_type = template_record.get("task_type")
        if "Detection" in str(task_type) or template_record.get("metric_family") == "detection":
            raise ValueError(
                "AnomalyGen cannot emit detection records without a validated bbox artifact"
            )
        if task_type not in {
            "Defect Classification",
            "Ref_based Defect Classification",
        }:
            raise ValueError(
                f"AnomalyGen rich mode does not support task {task_type!r}"
            )
        prompt_and_response(template_record, context="SDG template")
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
        if annotation_profile == "bare_okng":
            output.append({
                "images": [
                    _format(generated, media_root=media_root, relative=relative),
                    _format(clean, media_root=media_root, relative=relative),
                ],
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": label},
                ],
            })
        else:
            task_type = str(template_record["task_type"])
            generated_text = _format(
                generated, media_root=media_root, relative=relative
            )
            clean_text = _format(clean, media_root=media_root, relative=relative)
            if task_type == "Ref_based Defect Classification":
                images = [clean_text, generated_text]
                image_roles = ["golden", "target"]
            else:
                images = [generated_text]
                image_roles = ["target"]
            option_map = template_record.get("option_map", {})
            response = "Yes, the target image contains a defect."
            canonical_label = "defect"
            if option_map:
                matching = [
                    letter
                    for letter, semantic in option_map.items()
                    if str(semantic).strip().casefold() in {"defect", "visible defect"}
                    or (
                        "defect" in str(semantic).casefold()
                        and not re.search(
                            r"\b(?:no|not|none|without|doesn't|does\s+not)\b",
                            str(semantic).casefold(),
                        )
                    )
                ]
                if len(matching) != 1:
                    raise ValueError(
                        "SDG classification template must identify exactly one defect option"
                    )
                response = matching[0]
                canonical_label = str(option_map[matching[0]])
            source_id = f"sdg:{generated_text}#{task_type}"
            record = copy.deepcopy(template_record)
            record.update(
                {
                    "id": filesystem_safe_id(source_id),
                    "source_id": source_id,
                    "target_id": generated_text,
                    "images": images,
                    "image_roles": image_roles,
                    "reference_cohort": (
                        "golden_then_target"
                        if len(images) == 2
                        else "single_target"
                    ),
                    "answer": {
                        "kind": "choice_set",
                        "labels": [canonical_label],
                    },
                    "conversations": [
                        {"from": "human", "value": prompt},
                        {"from": "gpt", "value": response},
                    ],
                }
            )
            output.append(record)
    if not output:
        raise ValueError(f"{sdg_csv}: no unique generated samples were emitted")
    return output, {
        "mode": annotation_profile,
        "source": "anomalygen_sdg",
        "sdg_result_csv": str(sdg_csv),
        "generated_rows": len(rows),
        "output_records": len(output),
        "duplicates_skipped": duplicates,
        "labels": {
            ("defect" if annotation_profile == "nvpaw_multitask_v1" else label):
            len(output)
        },
        "defect_types": dict(sorted(defects.items())),
        "tasks": (
            {str(template_record["task_type"]): len(output)}
            if annotation_profile == "nvpaw_multitask_v1"
            else {}
        ),
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
    parser.add_argument(
        "--annotation-profile",
        choices=("bare_okng", "nvpaw_multitask_v1"),
        default="bare_okng",
    )
    parser.add_argument(
        "--template-id",
        help="Rich-mode id or source_id selecting one classification template from --prompt-from.",
    )
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
        template_record = None
        if args.annotation_profile == "nvpaw_multitask_v1":
            if args.prompt_from is None:
                raise ValueError(
                    "nvpaw_multitask_v1 requires --prompt-from with a materialized classification template"
                )
            template_records = load_records(args.prompt_from.expanduser().resolve())
            compatible = [
                record
                for record in template_records
                if record.get("task_type")
                in {"Defect Classification", "Ref_based Defect Classification"}
                and (
                    args.template_id is None
                    or args.template_id in {record.get("id"), record.get("source_id")}
                )
            ]
            if len(compatible) != 1:
                raise ValueError(
                    "rich --prompt-from must resolve exactly one compatible classification template; "
                    "use --template-id"
                )
            template_record = compatible[0]
            prompt = prompt_and_response(template_record, context="SDG template")[0]
        else:
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
            annotation_profile=args.annotation_profile,
            template_record=template_record,
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
