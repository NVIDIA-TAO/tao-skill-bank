#!/usr/bin/env bash
# Launch AnomalyGen SDG inference. Single-process by default; pass --num_gpus N
# for DDP-style rank-sharded inference (each rank holds a full model copy; FSDP
# is force-disabled by upstream when world_size > 1).
#
# Usage:
#   scripts/run_sdg.sh \
#       --checkpoint_dir <path> \
#       --step <int> \
#       --input_jsonl <path> \
#       --output_dir <path> \
#       [--model_size 2b|14b] (default: 2b)
#       [--seed N]            (default: 0)
#       [--num_gpus N]        (default: 1; uses torchrun --nproc_per_node=N)
#
# Must be run from the repo root (cosmos-anomalygen-predict2/).
set -euo pipefail

model_size=2b
seed=0
num_gpus=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint_dir) checkpoint_dir="$2"; shift 2;;
        --step)           step="$2"; shift 2;;
        --input_jsonl)    input_jsonl="$2"; shift 2;;
        --output_dir)     output_dir="$2"; shift 2;;
        --model_size)     model_size="$2"; shift 2;;
        --seed)           seed="$2"; shift 2;;
        --num_gpus)       num_gpus="$2"; shift 2;;
        -h|--help)        sed -n '2,17p' "$0"; exit 0;;
        *)  echo "error: unknown arg: $1" >&2; exit 2;;
    esac
done

: "${checkpoint_dir:?--checkpoint_dir is required}"
: "${step:?--step is required}"
: "${input_jsonl:?--input_jsonl is required}"
: "${output_dir:?--output_dir is required}"

case "$model_size" in
    2b|14b) ;;
    *) echo "error: --model_size must be 2b or 14b (got $model_size)" >&2; exit 2;;
esac

if ! [[ "$num_gpus" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --num_gpus must be a positive integer (got $num_gpus)" >&2
    exit 2
fi

export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-./results}"

# 2B: DDP (no FSDP needed). 14B: FSDP for the single-GPU case so it fits;
# at world_size>1 upstream auto-disables FSDP, so each rank holds the full
# 14B model — multi-GPU 14B requires per-rank VRAM ~= single-GPU-no-FSDP 14B.
case "$model_size" in
    2b)  exp="predict2_anomaly_gen_ddp_2b"  ;;
    14b) exp="predict2_anomaly_gen_fsdp_14b" ;;
esac

# `torchrun --nproc_per_node=N` works for any N>=1; world_size=1 falls back
# to the single-process path inside the runtime.
exec torchrun \
    --nproc_per_node="${num_gpus}" \
    --master_port="${MASTER_PORT:-12341}" \
    -m scripts.anomaly_gen.synthetic_dataset_generation \
    --config=cosmos_predict2/configs/base/ag_config.py \
    --ag_checkpoint_dir "${checkpoint_dir}" \
    --step "${step}" \
    --input_data_path "${input_jsonl}" \
    --output_image_path "${output_dir}" \
    --seed "${seed}" \
    -- "experiment=${exp}"
