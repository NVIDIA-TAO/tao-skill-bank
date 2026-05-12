#!/usr/bin/env bash
# Sanity-check an SDG output directory against its input JSONL.
#
# Checks:
#   - SDG_result.csv exists and has the expected row count
#   - reconstructed_image/ count matches JSONL entry count
#
# Usage: scripts/verify_output.sh <input_jsonl> <output_dir>
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <input_jsonl> <output_dir>" >&2
    exit 2
fi

input_jsonl="$1"
output_dir="$2"

expected=$(grep -c '[^[:space:]]' "$input_jsonl")
csv="$output_dir/SDG_result.csv"

if [[ ! -f "$csv" ]]; then
    echo "error: $csv not found — SDG did not finish?" >&2
    exit 1
fi
csv_rows=$(($(wc -l < "$csv") - 1))   # minus header

recon_dir="$output_dir/reconstructed_image"
if [[ ! -d "$recon_dir" ]]; then
    echo "error: $recon_dir not found" >&2
    exit 1
fi
image_count=$(find "$recon_dir" -type f \( -name '*.png' -o -name '*.jpg' \) | wc -l)

printf "expected: %d\ncsv rows: %d\nimages  : %d\n" \
    "$expected" "$csv_rows" "$image_count"

if (( csv_rows != expected )) || (( image_count != expected )); then
    echo "error: count mismatch. SDG may have been interrupted;" >&2
    echo "       nn_score eval on this output will be unreliable." >&2
    exit 1
fi

echo "OK: counts match"
