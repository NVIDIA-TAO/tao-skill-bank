#!/usr/bin/env python3
"""Split captions and gap data into N chunks for parallel video generation.

Supports both local paths and S3 URIs (s3://) via fsspec.
"""
import argparse
import json
import math
import os

import fsspec
import pandas as pd


def _is_remote(path):
    return "://" in path


def _join(base, name):
    """Join path components — works for both local and S3."""
    return f"{base.rstrip('/')}/{name}"


def split_jsonl_and_parquet(
    input_jsonl: str,
    input_parquet: str,
    output_dir: str,
    num_splits: int,
    hf_tokens: list,
) -> list:
    data = []
    with fsspec.open(input_jsonl, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    df = pd.read_parquet(input_parquet)
    chunk_size = math.ceil(len(data) / num_splits)

    split_info = []
    for i in range(num_splits):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(data))
        if start_idx >= len(data):
            break

        jsonl_chunk = data[start_idx:end_idx]
        jsonl_output = _join(output_dir, f"captions_split_{i}.jsonl")
        with fsspec.open(jsonl_output, "w") as f:
            for item in jsonl_chunk:
                f.write(json.dumps(item) + "\n")

        video_ids = [item.get("video_id", "") for item in jsonl_chunk]
        df_chunk = df[df["video_id"].isin(video_ids)]
        parquet_output = _join(output_dir, f"weak_data_split_{i}.parquet")
        df_chunk.to_parquet(parquet_output, index=False)

        token_idx = 0 if i < num_splits // 2 else 1
        assigned_token = hf_tokens[token_idx] if token_idx < len(hf_tokens) else hf_tokens[-1]
        split_info.append({
            "split_id": i,
            "jsonl_path": jsonl_output,
            "parquet_path": parquet_output,
            "output_subdir": f"split_{i}",
            "hf_token": assigned_token,
        })
        print(f"Created split {i}: {len(jsonl_chunk)} items, using token {token_idx}")

    split_info_path = _join(output_dir, "split_info.json")
    with fsspec.open(split_info_path, "w") as f:
        json.dump(split_info, f, indent=2)

    return split_info


def main():
    parser = argparse.ArgumentParser(
        description="Split captions JSONL and gaps parquet into N chunks"
    )
    parser.add_argument("--captions-jsonl", required=True)
    parser.add_argument("--gaps-parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-splits", type=int, default=8)
    parser.add_argument("--hf-tokens", default=None,
                        help="Comma-separated HuggingFace tokens (falls back to HF_TOKEN env var)")
    args = parser.parse_args()

    # Resolve tokens: CLI arg > env var
    if args.hf_tokens:
        hf_tokens = [t.strip() for t in args.hf_tokens.split(",")]
    else:
        env_token = os.environ.get("HF_TOKEN", "")
        if not env_token:
            print("Error: --hf-tokens not provided and HF_TOKEN env var not set")
            exit(1)
        hf_tokens = [t.strip() for t in env_token.split(",")]

    split_jsonl_and_parquet(
        input_jsonl=args.captions_jsonl,
        input_parquet=args.gaps_parquet,
        output_dir=args.output_dir,
        num_splits=args.num_splits,
        hf_tokens=hf_tokens,
    )


if __name__ == "__main__":
    main()
