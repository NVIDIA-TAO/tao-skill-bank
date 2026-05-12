#!/usr/bin/env bash
# Verify every pretrained checkpoint AnomalyGen needs end-to-end. Exits 0
# when all are present, 1 otherwise.
#
# Usage:
#   check.sh [--checkpoint-dir checkpoints]
set -euo pipefail

ckpt_dir="checkpoints"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-dir) ckpt_dir="$2"; shift 2;;
        -h|--help)        sed -n '2,7p' "$0"; exit 0;;
        *) echo "error: unknown arg $1" >&2; exit 2;;
    esac
done

missing=0
ok()   { printf "  [ok]      %s\n" "$1"; }
miss() { printf "  [missing] %s -- %s\n" "$1" "$2"; missing=$((missing+1)); }

check_file() {  # path, remediation
    if [[ -f "$1" ]]; then ok "$1"; else miss "$1" "$2"; fi
}
check_nonempty_dir() {  # path, remediation
    if [[ -d "$1" ]] && [[ -n "$(ls -A "$1" 2>/dev/null)" ]]; then ok "$1"; else miss "$1" "$2"; fi
}

dl="bash .claude/skills/cosmos-anomalygen/scripts/download_checkpoints.sh"

check_file        "${ckpt_dir}/nvidia/Cosmos-Predict2-2B-Text2Image/model.pt"   "$dl"
check_file        "${ckpt_dir}/nvidia/Cosmos-Predict2-14B-Text2Image/model.pt"  "$dl"
# Either T5 variant satisfies training (configurable via ag_config.t5_model_name).
t5_large_present=0
t5_11b_present=0
[[ -d "${ckpt_dir}/google-t5/t5-large" ]] && [[ -n "$(ls -A "${ckpt_dir}/google-t5/t5-large" 2>/dev/null)" ]] && t5_large_present=1
[[ -d "${ckpt_dir}/google-t5/t5-11b" ]] && [[ -n "$(ls -A "${ckpt_dir}/google-t5/t5-11b" 2>/dev/null)" ]] && t5_11b_present=1
if [[ "${t5_large_present}" == 1 || "${t5_11b_present}" == 1 ]]; then
    [[ "${t5_large_present}" == 1 ]] && ok "${ckpt_dir}/google-t5/t5-large"
    [[ "${t5_11b_present}"   == 1 ]] && ok "${ckpt_dir}/google-t5/t5-11b"
else
    miss "${ckpt_dir}/google-t5/{t5-large,t5-11b}" "$dl  (one variant suffices)"
fi
check_file        "${ckpt_dir}/NVDINOV2/nv_dinov2_classification_model.ckpt"    "$dl"
check_file        "${ckpt_dir}/nvidia/C-RADIO-V3/model.safetensors"             "$dl"
check_nonempty_dir "${ckpt_dir}/facebook/dinov2-large"                           "$dl"
check_file        "${ckpt_dir}/sam2/sam2.1_hiera_large.pt"                      "$dl"
check_nonempty_dir "${ckpt_dir}/Qwen/Qwen3-VL-4B-Instruct"                       "$dl"

if [[ "${missing}" -gt 0 ]]; then
    echo
    echo "${missing} artifact(s) missing."
    exit 1
fi
echo
echo "all required artifacts present."
