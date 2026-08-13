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
        link = out_root / "images" / unique_name
        target = Path("..") / "images_raw" / source_split / image_path
        try:
            link.symlink_to(target)
            made_links += 1
        except FileExistsError:
            skipped += 1

        cap = out_root / "captions" / (Path(unique_name).stem + ".txt")
        cap.write_text(caption + "\n", encoding="utf-8")
        made_caps += 1
    return made_links, made_caps, skipped


def load_rows(split: str, metadata_root: Path):
    """Project a pair JSON document down to the fields the rebuild needs."""
    with (metadata_root / f"{split}_pairs.json").open(encoding="utf-8") as handle:
        data = json.load(handle)
    rows = [
        (row["unique_name"], row["source_split"], row["image_path"], row["caption"])
        for row in data
    ]
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
    """Check exact counts, then sample-resolve rebuilt links and captions."""
    expected = 0
    names = []
    for split in splits:
        names_for_split = (metadata_root / f"{split}_list.txt").read_text().split()
        expected += len(names_for_split)
        names.extend(names_for_split)

    image_count = len(list((out_root / "images").iterdir()))
    caption_count = len(list((out_root / "captions").iterdir()))
    print(f"expected {expected:,} images/ {image_count:,} captions/ {caption_count:,}")

    random.seed(11)
    bad = []
    for name in random.sample(names, min(sample, len(names))):
        link = out_root / "images" / name
        if not link.exists():
            bad.append((name, "dangling or missing"))
            continue
        cap = out_root / "captions" / (Path(name).stem + ".txt")
        if not cap.is_file():
            bad.append((name, "missing caption"))
    print(f"sampled {min(sample, len(names)):,} entries, problems {len(bad)}")
    for problem in bad[:10]:
        print("  BAD", problem)

    ok = image_count == expected and caption_count == expected and not bad
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
