# PAS V3.1.1 partial-negative alignment reproduction

Use this reference only for the fixed PAS V3.1.1 SigLIP2-256 + standard LoRA
+ partial-negative alignment (PA) run. It is not an AutoML recipe and does not
include SigLIP2-384, RandLoRA, reranking, AQE, or score fusion.

PA here means only the partial-negative component from Song et al., “Dual
alignment: Partial negative and soft-label alignment for text-to-image person
retrieval,” Information Fusion 127 (2026), DOI
`10.1016/j.inffus.2025.103644`. The soft-label alignment component is not
implemented. Do not call this the paper's full method.

## Pinned implementation

- TAO PyTorch draft PR: https://github.com/NVIDIA-TAO/tao-pytorch/pull/118
- Source branch: `feature/clip-pa-loss-repro-20260831`
- Source commit: `aed14d5c09ecfa2ad5a79d95930784f4a06658c0`
- PR base: `feature/clip-lora-dev-for-deft`
- Tested image: `nvcr.io/nvidia/tao/tao-toolkit:6.25.11-pyt`
- Registry digest: `sha256:0db93f4e531c12d01d833eb654255fdfff36ee0f46bee1c6be45be502e36c8e0`

The current general CLIP image in `skill_info.yaml` is not evidence that these
draft fields are installed. Before launch, import `CLIPTrainConfig` inside the
selected runtime and verify that all four fields exist:

```bash
python - <<'PY'
from nvidia_tao_pytorch.config.clip.default_config import CLIPTrainConfig
fields = CLIPTrainConfig.__dataclass_fields__
required = {
    "pa_loss_weight",
    "pa_margin",
    "pa_inverse_temperature",
    "pa_top_ratio",
}
missing = sorted(required - fields.keys())
if missing:
    raise SystemExit(f"runtime is missing PA fields: {missing}")
print("PA runtime fields verified")
PY
```

Until PR 118 is included in a published TAO image, mount that exact checkout
read-only at `/workspace/tao-pytorch` and prepend it to `PYTHONPATH`. For local
Docker this means adding:

```bash
-v /absolute/path/to/tao-pytorch:/workspace/tao-pytorch:ro \
-e PYTHONPATH=/workspace/tao-pytorch
```

For SLURM/Pyxis, add the equivalent read-only container mount and export
`PYTHONPATH=/workspace/tao-pytorch` in the job. The source checkout is the
implementation; the historical `.sqsh` alone does not contain PA.

## Data contract

Stage one balanced PAS V3.1.1 derived dataset root with:

- `images/` and `captions/`;
- `train_list.txt` and `train_pairs.json`;
- `val_list.txt` and `val_pairs.json`;
- `attribute_vocab.json` and `accessory_vocab.json` beside the pair files.

Keep the original train/validation/test lists unchanged. Training requires
`dataset.train.include_attribute_metadata: true`. Validation and held-out
evaluation both use `scalar_plus_accessories`. Select checkpoints using only
`val/pas/overall_mAP`; never use the test split for selection.

## Fixed profile

Copy `pas_pa_reproduction.yaml`, replace only the data and results roots, and
keep `automl_policy: off`. The launch spec is the nested YAML beneath
`spec_overrides`; `automl_policy` is workflow routing metadata and must not be
passed to `clip train`.

The fixed controls are:

| Control | Value |
|---|---:|
| Model | `siglip2-so400m-patch16-256` |
| LoRA | standard, both towers, last 24 blocks |
| Vision rank / alpha | 64 / 128 |
| Text rank / alpha | 64 / 128 |
| Epochs / GPUs | 20 / 8 |
| Batch size | 128 per GPU |
| PA weight | 0.01232533396889774 |
| PA margin | 0.1976525764912367 |
| PA inverse temperature | 1.9850472486080717 |
| PA top ratio | 0.07580481977201999 |
| Selection | maximum `val/pas/overall_mAP` |

Launch with the normal model action after resolving the profile:

```bash
clip train -e /absolute/path/to/resolved_pas_pa.yaml
```

For evaluation, set `evaluate.checkpoint` to the selected validation
checkpoint and use a new results directory:

```bash
clip evaluate -e /absolute/path/to/resolved_pas_pa_evaluate.yaml
```

## Historical reference

The recorded run selected `model_best_002.pth`:

| Split | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| Validation | 85.8242% | 83.9294% | 98.1736% | 99.3848% |
| Test | 83.7930% | 83.0727% | 97.7381% | 99.0349% |

These numbers are a historical comparison, not a pass/fail tolerance. A fair
reproduction must preserve the dataset version, split lists, metadata
vocabularies, metric implementation, seeds, and checkpoint-selection rule.
