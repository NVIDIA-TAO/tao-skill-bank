#!/usr/bin/env bash
# Run one search round: apply Claude-chosen (guidance, crop_ratio) draws to
# the base JSONL, then SDG + per-sample eval.
#
# Input:
#   draws.json — {"<sample_index>": {"guidance": <f>, "crop_ratio": <f>}, ...}
#                Only sample indices listed here are included in the round.
#                Claude decides the values; this script does no sampling.
#
# Optional — re-roll AMP augmentation on the same (clean, submask) pairs:
#   --reamp-seed <N> and --defect-spec <jsonl>. When both are set, AMP is
#   re-invoked with --seed <N> --n_seeds 1 into <output-dir>/amp/, and the
#   base JSONL is rewritten to point at the fresh masks before apply_draws.
#   Requires amp_samples.json to exist next to --base-jsonl (written by
#   prep-testcase) — that's how we find the (clean, submask) records.
#
# Outputs under <output-dir>:
#   testcase.jsonl, sdg/…, per_sample.csv
#   [amp/, base.jsonl]     (only when --reamp-seed is set)
#
# Usage:
#   run_round.sh \
#       --base-jsonl <path> \
#       --draws <draws.json> \
#       --output-dir <dir> \
#       --real-path <dataset_dir> \
#       --anomaly-types <T+A> [<T+B> ...] \
#       --checkpoint-dir <dir> --step <int> \
#       [--model-size 2b|14b] [--seed 0] \
#       [--reamp-seed <N> --defect-spec <jsonl>]
set -euo pipefail

model_size=2b
seed=0
reamp_seed=""
defect_spec=""
anomaly_types=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-jsonl)      base_jsonl="$2"; shift 2;;
        --draws)           draws="$2"; shift 2;;
        --output-dir)      out="$2"; shift 2;;
        --real-path)       real_path="$2"; shift 2;;
        --anomaly-types)   shift
            while [[ $# -gt 0 && "$1" != --* ]]; do anomaly_types+=("$1"); shift; done;;
        --checkpoint-dir)  ckpt="$2"; shift 2;;
        --step)            step="$2"; shift 2;;
        --model-size)      model_size="$2"; shift 2;;
        --seed)            seed="$2"; shift 2;;
        --reamp-seed)      reamp_seed="$2"; shift 2;;
        --defect-spec)     defect_spec="$2"; shift 2;;
        -h|--help)         sed -n '2,31p' "$0"; exit 0;;
        *) echo "error: unknown arg $1" >&2; exit 2;;
    esac
done
: "${base_jsonl:?--base-jsonl required}"
: "${draws:?--draws required}"
: "${out:?--output-dir required}"
: "${real_path:?--real-path required}"
: "${ckpt:?--checkpoint-dir required}"
: "${step:?--step required}"
[[ ${#anomaly_types[@]} -gt 0 ]] || { echo "error: --anomaly-types required" >&2; exit 2; }

HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "${out}"

# Optional: re-roll AMP augmentation with a fresh base seed.
effective_base="${base_jsonl}"
if [[ -n "${reamp_seed}" ]]; then
    : "${defect_spec:?--defect-spec required when --reamp-seed is set}"
    amp_samples="$(dirname "${base_jsonl}")/amp_samples.json"
    [[ -f "${amp_samples}" ]] || {
        echo "error: ${amp_samples} not found — prep-testcase must have written it next to the base JSONL." >&2
        exit 1
    }
    reamp_dir="${out}/amp"
    mkdir -p "${reamp_dir}"

    # Derive original AMP dir from the first row's mask_filename and pre-copy
    # roi_mask.png into the round's amp dir so run_auto_roi_amp.py's cache
    # check skips Qwen VL + SAM2 (only the per-seed placement step runs).
    orig_amp_dir="$(python3 -c "
import json, pathlib
row = next(json.loads(l) for l in open('${base_jsonl}') if l.strip())
# mask_filename = <amp>/<name>/<full_type>/seed*.png → strip 3 levels
print(pathlib.Path(row['mask_filename']).parent.parent.parent)
")"
    if [[ -d "${orig_amp_dir}" ]]; then
        echo "=== seeding ROI cache from ${orig_amp_dir} ==="
        find "${orig_amp_dir}" -mindepth 3 -maxdepth 3 -type d -name assets | while read -r src; do
            rel="${src#${orig_amp_dir}/}"          # name/full_type/assets
            dst="${reamp_dir}/${rel}"
            mkdir -p "$(dirname "${dst}")"
            cp -r "${src}" "${dst}"
        done
    else
        echo "warn: original AMP dir ${orig_amp_dir} not found — re-AMP will recompute ROIs" >&2
    fi

    # Re-AMP must use the SAME n_seeds prep-testcase used. Otherwise rows that
    # originally pointed at seed1.png (etc.) collapse onto the new seed0.png and
    # within-pair diversity halves. ${amp_samples}.n_seeds is the sidecar prep
    # writes for exactly this purpose.
    n_seeds_file="${amp_samples}.n_seeds"
    if [[ -f "${n_seeds_file}" ]]; then
        n_seeds="$(cat "${n_seeds_file}")"
    else
        echo "warn: ${n_seeds_file} not found — falling back to --n_seeds 1 (within-pair diversity may drop)" >&2
        n_seeds=1
    fi

    echo "=== re-AMP (seed=${reamp_seed}, n_seeds=${n_seeds}) → ${reamp_dir} ==="
    python3 scripts/run_auto_roi_amp.py \
        --input "${amp_samples}" \
        --defect-desc "${defect_spec}" \
        --output "${reamp_dir}" \
        --n_seeds "${n_seeds}" \
        --seed "${reamp_seed}" \
        --model-id checkpoints/Qwen/Qwen3-VL-4B-Instruct
    python3 "${HERE}/reamp_swap_masks.py" \
        --base-jsonl "${base_jsonl}" \
        --new-amp-dir "${reamp_dir}" \
        --output "${out}/base.jsonl"
    effective_base="${out}/base.jsonl"
fi

# Apply draws → testcase.jsonl
python3 "${HERE}/apply_draws.py" \
    --base-jsonl "${effective_base}" \
    --draws "${draws}" \
    --output "${out}/testcase.jsonl"

# SDG (intentionally single-GPU: per-round sample count is small; torchrun
# init overhead would dominate. Bulk generation happens in Phase 3, which
# is parallelized via the orchestrator's num_gpus.)
"${HERE}/run_sdg.sh" \
    --checkpoint_dir "${ckpt}" --step "${step}" \
    --input_jsonl "${out}/testcase.jsonl" --output_dir "${out}/sdg" \
    --model_size "${model_size}" --seed "${seed}" \
    --num_gpus 1

# Eval emits both set-wise KPI (FID + aggregate nn/mnn) and per-sample CSV in
# one pass. compute_kpi is robust to few-sample anomaly types (FID is skipped
# with a warning, correspondence falls back to NaN). sdg-refine consumes
# per_sample.csv; the set-wise table is informational.
"${HERE}/run_eval.sh" \
    --real-path "${real_path}" --generated-path "${out}/sdg" \
    --anomaly-types "${anomaly_types[@]}" \
    --per-sample-csv "${out}/per_sample.csv"

echo "=== round done: ${out}/per_sample.csv ==="
