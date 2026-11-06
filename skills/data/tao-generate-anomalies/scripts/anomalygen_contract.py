#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataset-neutral exclusion and native AnomalyGen 1.2 accounting boundaries."""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    rows = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("testcase must contain nonempty JSON objects")
    return rows


def load_exclusions(path):
    if path is None:
        return set()
    entries = set()
    for line in Path(path).read_text().splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        if "/" in value or "\\" in value or value in {".", ".."}:
            raise ValueError("exclude-stems accepts exact filename stems, not paths")
        entries.add(value)
    return entries


def _excluded(path, excluded):
    stem = Path(path).stem
    return stem in excluded or (stem.endswith("_mask") and stem[:-5] in excluded)


def validate_testcases(rows, excluded=()):
    identities = [row.get("index", i) for i, row in enumerate(rows)]
    if any(type(i) is not int or i < 0 for i in identities) or len(
        set(identities)
    ) != len(identities):
        raise ValueError("testcase indices must be unique nonnegative integers")
    for index, row in enumerate(rows):
        for key in ("image_filename", "mask_filename", "anomaly_type"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"testcase {index} lacks {key}")
        if row.get("num_generated_images", 1) != 1:
            raise ValueError("DEFT accounting requires exactly one output per testcase")
        for key in ("image_filename", "mask_filename"):
            if not Path(row[key]).is_file():
                raise ValueError(f"testcase source file missing: {row[key]}")
        if any(
            _excluded(row[key], excluded) for key in ("image_filename", "mask_filename")
        ):
            raise ValueError(f'testcase uses excluded source: {row["image_filename"]}')


def filter_testcases(source, destination, exclude_stems=None, *, base_dir=None):
    source, destination = Path(source), Path(destination)
    rows = read_jsonl(source)
    excluded = load_exclusions(exclude_stems)
    base = Path(base_dir or Path.cwd()).resolve()
    for row in rows:
        for key in ("image_filename", "mask_filename"):
            if isinstance(row.get(key), str) and row[key]:
                path = Path(row[key])
                row[key] = str((path if path.is_absolute() else base / path).resolve())
    kept = [
        (i, row)
        for i, row in enumerate(rows)
        if not any(
            _excluded(row.get(key, ""), excluded)
            for key in ("image_filename", "mask_filename")
        )
    ]
    if not kept:
        raise ValueError("exclusions leave no testcase rows")
    validate_testcases([row for _, row in kept], excluded)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        stream.write("".join(json.dumps(row, sort_keys=True) + "\n" for _, row in kept))
    return {
        "source": str(source),
        "source_sha256": sha256(source),
        "testcase": str(destination),
        "testcase_sha256": sha256(destination),
        "retained": len(kept),
        "excluded": len(rows) - len(kept),
        "retained_source_indices": [i for i, _ in kept],
        "exclude_stems_sha256": sha256(exclude_stems) if exclude_stems else None,
    }


def prepare_dataset_view(source, destination, exclude_stems=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_dir():
        raise ValueError(f"dataset directory missing: {source}")
    if destination == source or source in destination.parents:
        raise ValueError("filtered dataset view must be outside its source")
    if destination.exists():
        raise FileExistsError(destination)
    excluded = load_exclusions(exclude_stems)
    removed = []
    copied = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if source not in path.resolve().parents:
            raise ValueError(f"dataset symlink escapes source: {path}")
        relative = path.relative_to(source)
        if path.suffix.lower() in IMAGE_EXTENSIONS and _excluded(path, excluded):
            removed.append(str(relative))
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(str(relative))
    if not any("/clean_image/" in "/" + name for name in copied):
        raise ValueError("filtered dataset has no clean images")
    return {
        "source": str(source),
        "dataset_view": str(destination),
        "copied_files": len(copied),
        "excluded_files": len(removed),
        "excluded_paths": removed,
        "exclude_stems_sha256": sha256(exclude_stems) if exclude_stems else None,
    }


def _csv(path, required, optional=False):
    if optional and not path.exists():
        return []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(required) <= set(reader.fieldnames or ()):
            raise ValueError(f"native manifest columns missing: {path}")
        return list(reader)


def verify_generation(testcase, output, exclude_stems=None):
    rows = read_jsonl(testcase)
    validate_testcases(rows, load_exclusions(exclude_stems))
    output = Path(output).resolve()
    image_root = output / "reconstructed_image"
    generated = _csv(
        output / "texture_ft_generation_result.csv",
        ("index", "output_filename", "image_filename", "mask_filename", "anomaly_type"),
    )
    blocked = _csv(
        output / "guardrail_blocked.csv",
        (
            "index",
            "output_idx",
            "image_filename",
            "mask_filename",
            "anomaly_type",
            "guardrail",
        ),
        True,
    )
    identities = {row.get("index", i): row for i, row in enumerate(rows)}
    observed = set()
    images = set()
    for kind, records in [("generated", generated), ("blocked", blocked)]:
        for row in records:
            try:
                index = int(row["index"])
            except (ValueError, TypeError) as error:
                raise ValueError("invalid testcase index") from error
            if index not in identities or index in observed:
                raise ValueError(
                    "duplicate, overlapping or out-of-range testcase identity"
                )
            observed.add(index)
            source = identities[index]
            for key in ("image_filename", "mask_filename", "anomaly_type"):
                if row[key] != source[key]:
                    raise ValueError(
                        f"native manifest source identity mismatch: {index}/{key}"
                    )
            if kind == "blocked":
                if int(row["output_idx"]) not in (-1, 0):
                    raise ValueError("blocked identity requires output_idx=-1 or 0")
            else:
                name = Path(row["output_filename"])
                if name.name != str(name) or not str(name):
                    raise ValueError("output image path escapes native image directory")
                image = (image_root / name).resolve()
                if image.parent != image_root.resolve() or not image.is_file():
                    raise ValueError(f"generated image missing or escaped: {image}")
                if name.name in images:
                    raise ValueError("duplicate generated image identity")
                images.add(name.name)
    if observed != set(identities):
        raise ValueError(
            "generated + guardrail-blocked identities do not cover testcases"
        )
    disk = (
        {p.name for p in image_root.iterdir() if p.is_file()}
        if image_root.is_dir()
        else set()
    )
    if disk != images:
        raise ValueError("reconstructed image files and native manifest differ")
    return {
        "schema": "anomalygen-output/1",
        "status": "COMPLETE",
        "testcases": len(rows),
        "generated": len(generated),
        "guardrail_blocked": len(blocked),
        "training_ready": bool(generated),
        "testcase_sha256": sha256(testcase),
        "output_dir": str(output),
        "generation_csv_sha256": sha256(output / "texture_ft_generation_result.csv"),
        "blocked_csv_sha256": (
            sha256(output / "guardrail_blocked.csv")
            if (output / "guardrail_blocked.csv").exists()
            else None
        ),
        "guardrail_semantics": "Blocked rows have no reconstructed image. Native safety configuration and timing remain in the generator run artifacts.",
    }


def prepare_refinement(testcase, original, destination):
    """Refine only generated identities; initial safety blocks are final for this run."""
    report = verify_generation(testcase, original)
    generated = _csv(Path(original) / "texture_ft_generation_result.csv", ("index",))
    eligible = {int(row["index"]) for row in generated}
    rows = [
        {**row, "index": row.get("index", i)}
        for i, row in enumerate(read_jsonl(testcase))
        if row.get("index", i) in eligible
    ]
    with Path(destination).open("x") as stream:
        stream.write("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return {
        "eligible": len(rows),
        "blocked_not_retried": report["guardrail_blocked"],
        "testcase": str(destination),
        "testcase_sha256": sha256(destination),
    }


def finalize_refinement(testcase, original, selected, exclude_stems=None):
    """Join native winners with original safety receipts without fabricating images."""
    verify_generation(testcase, original, exclude_stems)
    original, selected = Path(original), Path(selected)
    expected = {
        int(row["index"])
        for row in _csv(original / "texture_ft_generation_result.csv", ("index",))
    }
    actual = {
        int(row["index"])
        for row in _csv(selected / "texture_ft_generation_result.csv", ("index",))
    }
    if actual != expected:
        raise ValueError("refinement changed initial generated/blocked identities")
    source = original / "guardrail_blocked.csv"
    target = selected / "guardrail_blocked.csv"
    if target.exists() and (
        not source.exists() or target.read_bytes() != source.read_bytes()
    ):
        raise ValueError("refinement has conflicting blocked receipts")
    if source.exists() and not target.exists():
        shutil.copy2(source, target)
    return verify_generation(testcase, selected, exclude_stems)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--testcase", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--exclude-stems", type=Path)
    a = p.parse_args()
    print(
        json.dumps(
            verify_generation(a.testcase, a.output_dir, a.exclude_stems), indent=2
        )
    )


if __name__ == "__main__":
    main()
