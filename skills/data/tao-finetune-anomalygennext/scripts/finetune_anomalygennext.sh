#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

usage() {
  echo "Usage: $0 --recipe PATH --results-dir PATH [--hf-cache PATH] [--num-gpus N] [--min-nn-improvement X] [--parallelism ddp|fsdp] [--compile true|false] [--job-id ID] [--resume]"
}

recipe= results_dir= hf_cache= job_id=
num_gpus=1 min_improvement=0.0 parallelism=ddp compile=true resume=false
repo=/workspace/paidf-anomalygen
while [[ $# -gt 0 ]]; do
  case "$1" in
    --recipe) recipe=$2; shift 2 ;;
    --results-dir) results_dir=$2; shift 2 ;;
    --hf-cache) hf_cache=$2; shift 2 ;;
    --num-gpus) num_gpus=$2; shift 2 ;;
    --min-nn-improvement) min_improvement=$2; shift 2 ;;
    --parallelism) parallelism=$2; shift 2 ;;
    --compile) compile=$2; shift 2 ;;
    --job-id) job_id=$2; shift 2 ;;
    --resume) resume=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
[[ -f "$recipe" && -n "$results_dir" ]] || { usage >&2; exit 2; }
[[ -d "$repo" ]] || { echo "Pinned image source is missing: $repo" >&2; exit 2; }
[[ "$num_gpus" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid GPU count" >&2; exit 2; }
[[ "$parallelism" == ddp || "$parallelism" == fsdp ]] || { echo "Invalid parallelism" >&2; exit 2; }
[[ "$compile" == true || "$compile" == false ]] || { echo "Invalid compile value" >&2; exit 2; }
dinov2=$repo/checkpoints/facebook/dinov2-large
[[ -f "$dinov2/config.json" ]] || { echo "DINOv2 config missing: $dinov2" >&2; exit 2; }
[[ -f "$dinov2/model.safetensors" || -f "$dinov2/pytorch_model.bin" ]] || {
  echo "DINOv2 weights missing: $dinov2" >&2; exit 2;
}
[[ -z "$hf_cache" || -d "$hf_cache" ]] || { echo "HF cache missing: $hf_cache" >&2; exit 2; }
[[ -z "$hf_cache" ]] || export HF_HOME=$hf_cache
script_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
dataset=$(python3 -c 'import sys,yaml; d=yaml.safe_load(open(sys.argv[1])); assert d.get("run_validation_on_start") is True; print(d["dataset_name"])' "$recipe")
job=$(python3 -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["job_name"])' "$recipe")
for output in best_model_checkpoint.pt canonical_recipe.yaml training_handoff.json nn_improvement_summary.json status.json; do
  [[ ! -e "$results_dir/$output" ]] || { echo "Output exists: $results_dir/$output" >&2; exit 2; }
done
runtime=${TAO_RUNTIME_ROOT:-$results_dir/runtime}
[[ ! -e "$runtime" || "$resume" == true ]] || { echo "Runtime exists; pass --resume" >&2; exit 2; }
mkdir -p "$runtime"
export IMAGINAIRE_OUTPUT_ROOT=$runtime ANOMALYGEN_PARALLELISM=$parallelism
[[ "$compile" == true ]] && export ANOMALYGEN_COMPILE=1 || export ANOMALYGEN_COMPILE=0
cd "$repo"
torchrun --nproc_per_node="$num_gpus" anomalygen/scripts/texture/train.py \
  --config=cosmos_framework/configs/base/config.py --recipe="$recipe" -- experiment=anomalygen_texture_ft
python3 "$script_root/validate_nn_improvement.py" \
  --run-dir "$runtime/anomalygen/$dataset/$job" --recipe "$recipe" \
  --results-dir "$results_dir" --min-improvement "$min_improvement" --job-id "$job_id"
