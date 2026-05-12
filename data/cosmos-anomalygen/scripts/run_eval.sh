#!/usr/bin/env bash
# Wrap scripts.anomaly_gen.evaluate. Per-sample KPI CSV is always emitted
# (default <generated-path>/per_sample.csv; pass --per-sample-csv to override).
#
# Usage:
#   run_eval.sh \
#       --real-path <dir> \
#       --generated-path <dir> \
#       --anomaly-types <T+A> [<T+B> ...] \
#       [--per-sample-csv <path>] \
#       [--backbone cradio_v3_base]
set -euo pipefail

backbone="cradio_v3_base"
per_sample_csv=""
anomaly_types=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --real-path)         real_path="$2";       shift 2;;
        --generated-path)    generated_path="$2";  shift 2;;
        --per-sample-csv)    per_sample_csv="$2";  shift 2;;
        --backbone)          backbone="$2";        shift 2;;
        --anomaly-types)     shift
            while [[ $# -gt 0 && "$1" != --* ]]; do anomaly_types+=("$1"); shift; done
            ;;
        -h|--help)           sed -n '2,12p' "$0"; exit 0;;
        *) echo "error: unknown arg $1" >&2; exit 2;;
    esac
done
: "${real_path:?--real-path required}"
: "${generated_path:?--generated-path required}"
[[ ${#anomaly_types[@]} -gt 0 ]] || { echo "error: --anomaly-types required" >&2; exit 2; }
: "${per_sample_csv:=${generated_path}/per_sample.csv}"

args=(-m scripts.anomaly_gen.evaluate
      --real_path "${real_path}"
      --generated_path "${generated_path}"
      --backbone "${backbone}"
      --anomaly_types "${anomaly_types[@]}"
      --per_sample_csv "${per_sample_csv}")

exec python3 "${args[@]}"
