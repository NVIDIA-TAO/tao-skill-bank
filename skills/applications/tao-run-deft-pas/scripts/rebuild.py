#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the canonical TAO PAS image/caption trees from exported metadata.

The dataset export owns images and pair metadata.  This skill-owned adapter
owns the deterministic transformation into the ``images/`` and ``captions/``
trees consumed by TAO, so the skill revision—not executable content embedded
in an archive—defines how a PAS dataset is materialized.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any


SPLITS = ("train", "val", "test")
REQUIRED_ROW_KEYS = ("unique_name", "source_split", "image_path", "caption")


def _safe_relative_path(value: Any, *, field: str, basename_only: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    if "\\" in value:
        raise ValueError(f"{field} must use POSIX path separators")
    path = PurePosixPath(value)
    if (
        not path.parts
        or str(path) != value
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ValueError(f"{field} must be a normalized relative path")
    if basename_only and len(path.parts) != 1:
        raise ValueError(f"{field} must be a basename, not a nested path")
    return value


def _project_row(raw: Any, *, split: str, index: int) -> tuple[str, str, str, str]:
    location = f"{split}_pairs.json row {index}"
    if not isinstance(raw, dict):
        raise ValueError(f"{location} must be an object")
    missing = [key for key in REQUIRED_ROW_KEYS if key not in raw]
    if missing:
        raise ValueError(f"{location} is missing key(s): {', '.join(missing)}")

    unique_name = _safe_relative_path(
        raw["unique_name"], field=f"{location}.unique_name", basename_only=True
    )
    source_split = _safe_relative_path(
        raw["source_split"], field=f"{location}.source_split", basename_only=True
    )
    image_path = _safe_relative_path(
        raw["image_path"], field=f"{location}.image_path"
    )
    caption = raw["caption"]
    if not isinstance(caption, str):
        raise ValueError(f"{location}.caption must be a string")
    return unique_name, source_split, image_path, caption


def load_rows(dataset_root: Path, split: str) -> list[tuple[str, str, str, str]]:
    """Load one exported split and project it to the builder contract."""

    pairs_path = dataset_root / f"{split}_pairs.json"
    if not pairs_path.is_file():
        raise ValueError(f"required PAS metadata is missing: {pairs_path}")
    with pairs_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{pairs_path} must contain a non-empty JSON array")
    rows = [_project_row(row, split=split, index=index) for index, row in enumerate(payload)]
    del payload
    return rows


def _shard(
    rows: list[tuple[str, str, str, str]], dataset_root: str
) -> tuple[int, int, int]:
    root = Path(dataset_root)
    images_dir = root / "images"
    captions_dir = root / "captions"
    made_links = made_captions = existing_links = 0

    for unique_name, source_split, image_path, caption in rows:
        link = images_dir / unique_name
        target = f"../images_raw/{source_split}/{image_path}"
        try:
            os.symlink(target, link)
            made_links += 1
        except FileExistsError:
            if not link.is_symlink() or os.readlink(link) != target:
                raise ValueError(
                    f"existing image entry does not match PAS metadata: {link}"
                )
            existing_links += 1

        caption_path = captions_dir / (Path(unique_name).stem + ".txt")
        if caption_path.is_symlink() or (
            caption_path.exists() and not caption_path.is_file()
        ):
            raise ValueError(f"unsafe existing caption output: {caption_path}")
        with caption_path.open("w", encoding="utf-8") as handle:
            handle.write(caption + "\n")
        made_captions += 1

    return made_links, made_captions, existing_links


def _prepare_output_dir(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"refusing a symlinked PAS output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise ValueError(f"PAS output path is not a directory: {path}")


def rebuild(dataset_root: Path, workers: int) -> int:
    images_dir = dataset_root / "images"
    captions_dir = dataset_root / "captions"

    grand_total = 0
    for split in SPLITS:
        started = time.monotonic()
        rows = load_rows(dataset_root, split)
        _prepare_output_dir(images_dir)
        _prepare_output_dir(captions_dir)
        worker_count = min(workers, len(rows))
        shard_size = (len(rows) + worker_count - 1) // worker_count
        shards = [rows[offset : offset + shard_size] for offset in range(0, len(rows), shard_size)]
        with mp.Pool(worker_count) as pool:
            results = pool.starmap(_shard, ((shard, str(dataset_root)) for shard in shards))

        links = sum(result[0] for result in results)
        captions = sum(result[1] for result in results)
        existing = sum(result[2] for result in results)
        elapsed = time.monotonic() - started
        grand_total += len(rows)
        print(
            f"{split:5s} rows={len(rows):9,d}  links={links:9,d}  "
            f"captions={captions:9,d}  existing={existing:9,d}  "
            f"{elapsed:6.1f}s  ({len(rows) / elapsed:,.0f} rows/s)",
            flush=True,
        )
        del rows
    return grand_total


def verify(dataset_root: Path, sample_size: int = 2000) -> bool:
    expected_names: list[str] = []
    for split in SPLITS:
        list_path = dataset_root / f"{split}_list.txt"
        if not list_path.is_file():
            raise ValueError(f"required PAS metadata is missing: {list_path}")
        names = list_path.read_text(encoding="utf-8").split()
        if not names:
            raise ValueError(f"{list_path} must contain at least one image name")
        expected_names.extend(
            _safe_relative_path(
                name,
                field=f"{list_path.name} image name",
                basename_only=True,
            )
            for name in names
        )

    images_dir = dataset_root / "images"
    captions_dir = dataset_root / "captions"
    image_count = sum(1 for _ in os.scandir(images_dir))
    caption_count = sum(1 for _ in os.scandir(captions_dir))
    expected_count = len(expected_names)
    print(
        f"expected {expected_count:,}   images/ {image_count:,}   "
        f"captions/ {caption_count:,}"
    )

    random.seed(11)
    bad: list[tuple[str, str]] = []
    for name in random.sample(expected_names, min(sample_size, expected_count)):
        image = images_dir / name
        if not image.exists():
            bad.append((name, "dangling or missing image link"))
            continue
        caption = captions_dir / (Path(name).stem + ".txt")
        if not caption.is_file():
            bad.append((name, "missing caption"))

    sampled = min(sample_size, expected_count)
    print(f"sampled {sampled:,} entries, problems {len(bad)}")
    for problem in bad[:10]:
        print("  BAD", problem)

    ok = image_count == expected_count and caption_count == expected_count and not bad
    print("VERIFY:", "PASS" if ok else "FAIL")
    return ok


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="extracted PAS root containing images_raw and split metadata",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    images_raw = dataset_root / "images_raw"
    if not images_raw.is_dir() or images_raw.is_symlink():
        raise ValueError(f"required PAS image directory is missing or unsafe: {images_raw}")

    started = time.monotonic()
    if not args.verify_only:
        total = rebuild(dataset_root, args.workers)
        print(f"\nrebuilt {total:,} pairs in {time.monotonic() - started:.1f}s\n")
    return 0 if verify(dataset_root) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"rebuild.py: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
