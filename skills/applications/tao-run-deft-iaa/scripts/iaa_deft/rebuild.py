# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rebuild the TAO-facing image and caption trees from an IAA export."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path


SPLITS = ("train", "val", "test")


def _shard(rows, out_root: Path):
    """Create one relative image symlink and caption for each pair row."""
    made_links = made_caps = skipped = 0
    for unique_name, source_split, image_path, caption in rows:
        unique_path = _relative_path(unique_name, "unique_name")
        source_path = (
            out_root
            / "images_raw"
            / _relative_path(source_split, "source_split")
            / _relative_path(image_path, "image_path")
        )
        link = out_root / "images" / unique_path
        link.parent.mkdir(parents=True, exist_ok=True)
        target = Path(os.path.relpath(source_path, start=link.parent))
        try:
            link.symlink_to(target)
            made_links += 1
        except FileExistsError:
            skipped += 1

        cap = out_root / "captions" / unique_path.with_suffix(".txt")
        cap.parent.mkdir(parents=True, exist_ok=True)
        cap.write_text(caption + "\n", encoding="utf-8")
        made_caps += 1
    return made_links, made_caps, skipped


def _relative_path(value: str, field: str) -> Path:
    """Return a non-empty relative path that cannot escape its owned root."""
    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or path == Path(".")
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{field} must be a non-empty relative path: {value!r}")
    return path


def load_rows(split: str, metadata_root: Path):
    """Project a pair JSON document down to the fields the rebuild needs."""
    with (metadata_root / f"{split}_pairs.json").open(encoding="utf-8") as handle:
        data = json.load(handle)
    rows = []
    for row in data:
        unique_name = str(_relative_path(row["unique_name"], "unique_name"))
        source_split = str(_relative_path(row["source_split"], "source_split"))
        image_path = str(_relative_path(row["image_path"], "image_path"))
        caption = row["caption"]
        if not isinstance(caption, str):
            raise ValueError("caption must be a string")
        rows.append((unique_name, source_split, image_path, caption))
    del data
    return rows


def rebuild(splits, workers: int, metadata_root: Path, out_root: Path) -> int:
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "captions").mkdir(parents=True, exist_ok=True)

    grand = 0
    for split in splits:
        started = time.time()
        rows = load_rows(split, metadata_root)
        shards = [rows[index::workers] for index in range(workers)]
        with mp.Pool(workers) as pool:
            results = pool.starmap(_shard, [(shard, out_root) for shard in shards])
        links = sum(item[0] for item in results)
        captions = sum(item[1] for item in results)
        skipped = sum(item[2] for item in results)
        elapsed = time.time() - started
        grand += len(rows)
        print(
            f"{split:5s} rows={len(rows):9,d} links={links:9,d} "
            f"captions={captions:9,d} existing={skipped:9,d} "
            f"{elapsed:6.1f}s ({len(rows) / max(elapsed, 1e-9):,.0f} rows/s)",
            flush=True,
        )
        del rows
    return grand


def verify(splits, metadata_root: Path, out_root: Path, sample: int = 2000) -> bool:
    """Check requested metadata coverage, then sample link provenance."""
    rows_by_name = {}
    listed_names = []
    for split in splits:
        names_for_split = [
            str(_relative_path(name.strip(), f"{split}_list.txt entry"))
            for name in (metadata_root / f"{split}_list.txt").read_text().splitlines()
            if name.strip()
        ]
        listed_names.extend(names_for_split)
        for row in load_rows(split, metadata_root):
            name = row[0]
            if name in rows_by_name:
                raise ValueError(f"duplicate unique_name across requested splits: {name}")
            rows_by_name[name] = row

    expected_images = set(listed_names)
    if len(expected_images) != len(listed_names):
        raise ValueError("split list metadata contains duplicate unique_name values")
    if expected_images != set(rows_by_name):
        missing_from_pairs = sorted(expected_images - set(rows_by_name))
        missing_from_lists = sorted(set(rows_by_name) - expected_images)
        raise ValueError(
            "pair/list metadata disagree; "
            f"missing from pairs={missing_from_pairs[:5]}, "
            f"missing from lists={missing_from_lists[:5]}"
        )
    expected_captions = {
        str(Path(name).with_suffix(".txt")) for name in expected_images
    }
    if len(expected_captions) != len(expected_images):
        raise ValueError("unique_name values collide after conversion to caption paths")

    image_root = out_root / "images"
    caption_root = out_root / "captions"
    actual_images = {
        str(path.relative_to(image_root))
        for path in image_root.rglob("*")
        if path.is_symlink() and path.exists()
    }
    actual_captions = {
        str(path.relative_to(caption_root))
        for path in caption_root.rglob("*")
        if path.is_file()
    }
    complete_run = set(splits) == set(SPLITS)
    counts_ok = (
        actual_images == expected_images and actual_captions == expected_captions
        if complete_run
        else expected_images.issubset(actual_images)
        and expected_captions.issubset(actual_captions)
    )
    print(
        f"expected {len(expected_images):,} requested entries; "
        f"images/ {len(actual_images):,} total captions/ {len(actual_captions):,} total"
    )

    random.seed(11)
    bad = []
    sampled_names = random.sample(
        sorted(expected_images), min(sample, len(expected_images))
    )
    for name in sampled_names:
        link = out_root / "images" / name
        if not link.is_symlink() or not link.exists():
            bad.append((name, "dangling or missing"))
            continue
        _, source_split, image_path, _ = rows_by_name[name]
        expected_source = (
            out_root / "images_raw" / source_split / image_path
        ).resolve()
        if link.resolve() != expected_source:
            bad.append((name, "points at the wrong source image"))
        cap = out_root / "captions" / Path(name).with_suffix(".txt")
        if not cap.is_file():
            bad.append((name, "missing caption"))
    print(f"sampled {len(sampled_names):,} entries, problems {len(bad)}")
    for problem in bad[:10]:
        print("  BAD", problem)

    ok = counts_ok and not bad
    print("VERIFY:", "PASS" if ok else "FAIL")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)

    metadata_root = args.metadata_root.expanduser().resolve()
    out_root = args.out.expanduser().resolve()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if not (out_root / "images_raw").is_dir():
        parser.error(f"{out_root}/images_raw not found; extract images_raw.tar first")
    for split in args.splits:
        for suffix in ("pairs.json", "list.txt"):
            path = metadata_root / f"{split}_{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                parser.error(f"missing or empty metadata input: {path}")

    started = time.time()
    if not args.verify_only:
        total = rebuild(args.splits, args.workers, metadata_root, out_root)
        print(f"\nrebuilt {total:,} pairs in {time.time() - started:.1f}s\n")
    return 0 if verify(args.splits, metadata_root, out_root) else 1


if __name__ == "__main__":
    sys.exit(main())
