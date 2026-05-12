#!/usr/bin/env bash
# Download every pretrained checkpoint AnomalyGen needs end-to-end (FT, AMP,
# SDG, eval). Delegates to the in-repo `scripts.download_checkpoints` module,
# which handles Cosmos-Predict2, T5, NVDINOV2, C-RADIO-V3, dinov2-large,
# SAM2, and Qwen3-VL.
#
# Usage:
#   download_checkpoints.sh [--checkpoint-dir checkpoints]
#
# Assumes the `cosmos-predict2` conda env is already active. Requires
# HF_TOKEN exported. Idempotent — the upstream module is invoked only when
# one of its artifacts is missing.
set -euo pipefail

ckpt_dir="checkpoints"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-dir) ckpt_dir="$2"; shift 2;;
        -h|--help)        sed -n '2,12p' "$0"; exit 0;;
        *) echo "error: unknown arg $1" >&2; exit 2;;
    esac
done

: "${HF_TOKEN:?HF_TOKEN must be exported (Hugging Face access token)}"
command -v huggingface-cli >/dev/null 2>&1 \
    || { echo "error: huggingface-cli not found in active env" >&2; exit 2; }
command -v python >/dev/null 2>&1 \
    || { echo "error: python not found in active env" >&2; exit 2; }

mkdir -p "${ckpt_dir}"

# Skip the upstream module call only if every artifact it would produce is
# already on disk — avoids the unconditional NVDINOV2 wget overwrite.
upstream_present=1
for path in \
    "${ckpt_dir}/nvidia/Cosmos-Predict2-2B-Text2Image/model.pt" \
    "${ckpt_dir}/nvidia/Cosmos-Predict2-14B-Text2Image/model.pt" \
    "${ckpt_dir}/google-t5/t5-large/config.json" \
    "${ckpt_dir}/google-t5/t5-11b/config.json" \
    "${ckpt_dir}/NVDINOV2/nv_dinov2_classification_model.ckpt" \
    "${ckpt_dir}/nvidia/C-RADIO-V3/model.safetensors" \
    "${ckpt_dir}/facebook/dinov2-large/config.json" \
    "${ckpt_dir}/sam2/sam2.1_hiera_large.pt" \
    "${ckpt_dir}/Qwen/Qwen3-VL-4B-Instruct/model-00001-of-00002.safetensors" \
    "${ckpt_dir}/Qwen/Qwen3-VL-4B-Instruct/model-00002-of-00002.safetensors"
do
    [[ -f "${path}" ]] || { echo "[need] ${path}"; upstream_present=0; }
done

if [[ "${upstream_present}" == "0" ]]; then
    echo "[setup] HF login"
    huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential >/dev/null
    echo "[fetch] python -m scripts.download_checkpoints --model_types text2image --model_sizes 2B 14B"
    CUDA_HOME="${CONDA_PREFIX:-}" python -m scripts.download_checkpoints \
        --model_types text2image \
        --model_sizes 2B 14B \
        --checkpoint_dir "${ckpt_dir}"
else
    echo "[skip] all upstream-handled artifacts present"
fi

echo "[done] run .claude/skills/cosmos-anomalygen/scripts/check.sh to verify"
