#!/usr/bin/env python3
"""Merge per-split video generation outputs into a single parquet and video directory.

Collects the per-split parquet files (updated with generated_video_path)
and per-split video directories into a unified output. Supports both
local and S3 paths via fsspec.
"""
import argparse
import os

import fsspec
import pandas as pd


def merge_cosmos_outputs(output_dir: str, num_splits: int, splits_dir: str) -> str:
    merged_videos_dir = os.path.join(output_dir, "videos")
    os.makedirs(merged_videos_dir, exist_ok=True)

    all_dfs = []
    for i in range(num_splits):
        parquet_path = os.path.join(splits_dir, f"weak_data_split_{i}.parquet")
        if os.path.exists(parquet_path):
            df = pd.read_parquet(parquet_path)
            all_dfs.append(df)
            print(f"Loaded parquet split {i}: {len(df)} rows")
        else:
            print(f"Warning: Parquet not found at {parquet_path}")

        split_videos_dir = os.path.join(output_dir, f"split_{i}", "videos")
        if os.path.exists(split_videos_dir):
            import shutil
            video_count = 0
            for video_file in os.listdir(split_videos_dir):
                src = os.path.join(split_videos_dir, video_file)
                dst = os.path.join(merged_videos_dir, video_file)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                    video_count += 1
            print(f"Copied {video_count} videos from split {i}")
        else:
            print(f"Warning: Videos directory not found at {split_videos_dir}")

    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        output_parquet = os.path.join(output_dir, "all_kpi_gaps_with_videos.parquet")
        merged_df.to_parquet(output_parquet, index=False)
        print(f"\nMerged {len(all_dfs)} splits into {output_parquet}")
        print(f"Total rows: {len(merged_df)}")
        print(f"Videos in {merged_videos_dir}: {len(os.listdir(merged_videos_dir))}")
        return output_parquet
    else:
        print("Error: No parquet files found")
        return ""


def main():
    parser = argparse.ArgumentParser(description="Merge split video gen outputs")
    parser.add_argument("--output-dir", required=True,
                        help="Base generated dir (contains split_0/, split_1/, ...)")
    parser.add_argument("--splits-dir", required=True,
                        help="Dir containing weak_data_split_N.parquet files")
    parser.add_argument("--num-splits", type=int, default=8)
    args = parser.parse_args()

    merge_cosmos_outputs(args.output_dir, args.num_splits, args.splits_dir)


if __name__ == "__main__":
    main()
