#!/usr/bin/env python3
"""Overwrite guidance/crop_ratio per sample in a JSONL using a draws file.

Draws JSON shape: {"<sample_index>": {"guidance": <f>, "crop_ratio": <f>}, ...}
Sample index is the 0-based line number in --base-jsonl. Only entries listed
in draws are emitted to --output; the rest are dropped (this round only
covers the samples Claude chose to retry). Each emitted entry also carries
"index": sample_index so SDG_result.csv preserves base identity downstream.
"""
import argparse
import json
import pathlib
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-jsonl", required=True, type=pathlib.Path)
    p.add_argument("--draws", required=True, type=pathlib.Path)
    p.add_argument("--output", required=True, type=pathlib.Path)
    args = p.parse_args()

    base = [json.loads(l) for l in args.base_jsonl.read_text().splitlines() if l.strip()]
    draws = json.loads(args.draws.read_text())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w") as fp:
        for idx_str in sorted(draws, key=int):
            idx = int(idx_str)
            if idx >= len(base):
                print(f"warn: draw index {idx} out of range (base has {len(base)})", file=sys.stderr)
                continue
            entry = dict(base[idx])
            d = draws[idx_str]
            entry["guidance"] = d["guidance"]
            entry["crop_ratio"] = d["crop_ratio"]
            entry["index"] = idx
            fp.write(json.dumps(entry) + "\n")
            written += 1
    print(f"wrote {written} entries to {args.output}")


if __name__ == "__main__":
    main()
