#!/usr/bin/env bash
# End-to-end prep-testcase: validate → allocate → build AMP sample list →
# run_auto_roi_amp.py (n_seeds=1) → build JSONL → verify.
#
# 1:1 invariant: num_sdg → allocation → N AMP records → N AMP masks → N
# JSONL rows. For validation, callers set num_sdg = total training mask
# count so every training mask appears once; for inference, num_sdg is the
# SDG target count.
#
# spatial_dependency routing (per defect_spec entry):
#   free → whole-image ROI (run_auto_roi_amp.py does it internally)
#   text  → text2roi (Qwen VL text2box + SAM2), needs roi_prompt_defect_location
#   cad  → cad2roi, needs <dataset>/<TEXTURE>/cad_mask/<stem>.png and
#          <dataset>/semantic_segmentation_labels.json
#
# Usage:
#   prep_testcase.sh \
#       --name <exp> \
#       --num-sdg N \
#       --dataset-dir <dir> \
#       --amp-output-dir <dir> \
#       --output-jsonl <path> \
#       --defect-spec <jsonl> \
#       [--clean-dir <dir>]  (default: --dataset-dir)
#       [--guidance <F>] [--crop-ratio <F>]
#       [--seed <N>]   (default: 42, base seed for run_auto_roi_amp.py)
#
# Defect types are derived from --defect-spec.
set -euo pipefail

guidance=7.0
crop_ratio=2.0
base_seed=42
defect_spec=""
clean_dir=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)                  name="$2";                  shift 2;;
        --num-sdg)               num_sdg="$2";               shift 2;;
        --dataset-dir)           dataset_dir="$2";           shift 2;;
        --clean-dir)             clean_dir="$2";             shift 2;;
        --amp-output-dir)        amp_output="$2";            shift 2;;
        --output-jsonl)          output_jsonl="$2";          shift 2;;
        --defect-spec)           defect_spec="$2";           shift 2;;
        --guidance)              guidance="$2";              shift 2;;
        --crop-ratio)            crop_ratio="$2";            shift 2;;
        --seed)                  base_seed="$2";             shift 2;;
        -h|--help)               sed -n '2,30p' "$0"; exit 0;;
        *) echo "error: unknown arg $1" >&2; exit 2;;
    esac
done

: "${name:?--name required}"
: "${num_sdg:?--num-sdg required}"
: "${dataset_dir:?--dataset-dir required}"
: "${amp_output:?--amp-output-dir required}"
: "${output_jsonl:?--output-jsonl required}"
: "${defect_spec:?--defect-spec required (see .claude/skills/cosmos-anomalygen/assets/defect_spec_template.jsonl)}"
[[ -f "${defect_spec}" ]] || { echo "error: defect_spec not found: ${defect_spec}" >&2; exit 1; }
clean_dir="${clean_dir:-${dataset_dir}}"

readarray -t defect_types < <(python3 -c "
import json
for line in open('${defect_spec}'):
    line = line.strip()
    if line:
        print(json.loads(line)['defect_type'])
")
[[ ${#defect_types[@]} -gt 0 ]] || { echo "error: no defect_type entries in ${defect_spec}" >&2; exit 1; }

HERE="$(cd "$(dirname "$0")" && pwd)"

# 1. Fail-fast validation of the AMP input triple.
python3 "${HERE}/validate_amp_inputs.py" \
    --dataset-dir "${dataset_dir}" \
    --clean-dir "${clean_dir}" \
    --defect-spec "${defect_spec}"

# 2. Proportional allocation across defect types.
alloc_json="$(dirname "${output_jsonl}")/allocation.json"
mkdir -p "$(dirname "${alloc_json}")"
python3 "${HERE}/allocate_samples.py" \
    --num-sdg "${num_sdg}" --defect-types "${defect_types[@]}" \
    --mask-path "${dataset_dir}" --output "${alloc_json}"

# 3. Build the per-sample JSON for run_auto_roi_amp.py — exactly
# allocation[defect] records per defect (submask-first round-robin).
amp_samples="$(dirname "${output_jsonl}")/amp_samples.json"
python3 "${HERE}/build_amp_samples.py" \
    --dataset-dir "${dataset_dir}" \
    --clean-dir "${clean_dir}" \
    --defect-spec "${defect_spec}" \
    --allocation "${alloc_json}" \
    --output "${amp_samples}"

# 4. Run AMP. n_seeds is auto-computed from allocation (max over defects of
# ceil(alloc[d] / num_submasks[d])) so each submask can produce enough
# placements to cover its allocation when num_SDG >> total_training_masks.
n_seeds="$(cat "${amp_samples}.n_seeds")"
mkdir -p "${amp_output}"
python3 scripts/run_auto_roi_amp.py \
    --input "${amp_samples}" \
    --defect-desc "${defect_spec}" \
    --output "${amp_output}" \
    --n_seeds "${n_seeds}" \
    --seed "${base_seed}" \
    --model-id checkpoints/Qwen/Qwen3-VL-4B-Instruct

# 5. Build SDG JSONL from AMP output — one row per AMP mask.
python3 "${HERE}/build_jsonl.py" \
    --amp-output-dir "${amp_output}" \
    --clean-dir "${clean_dir}" \
    --allocation "${alloc_json}" \
    --defect-types "${defect_types[@]}" \
    --guidance "${guidance}" \
    --crop-ratio "${crop_ratio}" \
    --output "${output_jsonl}"

# 6. Verify + resize masks where needed.
python3 "${HERE}/verify_jsonl.py" \
    --jsonl "${output_jsonl}" \
    --cache-dir "$(dirname "${output_jsonl}")/resized_masks"

echo "=== prep-testcase done: ${output_jsonl} ==="
