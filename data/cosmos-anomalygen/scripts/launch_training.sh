#!/usr/bin/env bash
# Launch AnomalyGen training via torchrun. DDP only (FSDP is not a supported
# option in this skill; use the experiment config directly if you need it).
#
# Usage:
#   launch_training.sh \
#       --ag-config ag_configs/<name>.yaml \
#       [--num-gpus 1] \
#       [--model-size 2b|14b]   (default: 2b) \
#       [--master-port 12341]   (bump if port in use)
#
# IMAGINAIRE_OUTPUT_ROOT defaults to ./results if unset.
set -euo pipefail

num_gpus=1
model_size=2b
master_port=12341

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ag-config)       ag_config="$2";       shift 2;;
        --num-gpus)        num_gpus="$2";        shift 2;;
        --model-size)      model_size="$2";      shift 2;;
        --master-port)     master_port="$2";     shift 2;;
        -h|--help)         sed -n '2,13p' "$0"; exit 0;;
        *) echo "error: unknown arg $1" >&2; exit 2;;
    esac
done
: "${ag_config:?--ag-config required}"
case "${model_size}" in 2b|14b) ;; *) echo "error: --model-size must be 2b|14b" >&2; exit 2;; esac

export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-./results}"
exp="predict2_anomaly_gen_ddp_${model_size}"

exec torchrun --nproc_per_node="${num_gpus}" --master_port="${master_port}" \
    -m scripts.anomaly_gen.ag_train \
    --config=cosmos_predict2/configs/base/ag_config.py \
    --ag_config="${ag_config}" \
    -- "experiment=${exp}"
