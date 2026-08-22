# LAM SegFormer: AutoML, DEFT, fusion, and checkpoint soups

This directory preserves the executable experiment bundle used for the LAM
four-class SegFormer campaign started on 2026-08-20. It is an experiment
artifact, not a production skill or a supported TAO workflow.

## Dataset and label contract

The original campaign used:

```text
/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/lam_research
```

Expected layout:

```text
images/train  masks/train   # 316 labeled samples
images/val    masks/val     # 1,262 labeled samples
images/test               # image-only; never used for metric selection
```

The label contract is four grayscale classes with values `0`, `85`, `170`,
and `255`; `label_transform` is `null` and `num_classes` is `4`.

## Runtime contract

- Container: `nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt`
- Platform: one-node SLURM jobs
- GPUs: exactly 8 per job
- Partitions: `polar,polar3,polar4,grizzly`
- Credentials/platform settings: source `~/.tao/config.env` in the same shell
  invocation as every launcher. Do not commit that file.
- Full promotions train for 2,000 epochs.
- Validation interval is 10 epochs; requeue-safe durable checkpoint interval
  is 20 epochs for new specs.

The scripts preserve the original absolute campaign paths. For a new machine
or user, change `LOCAL_ROOT`, `REMOTE_ROOT`, dataset paths, and patch roots
before preflight. Run `scripts/run_lam_track.py --validate-only` before any
submission.

## Experiment sequence

The principal orchestration sequence is:

1. `launch_campaign.py` starts the original-data AutoML and control searches.
2. `launch_full2000_all_brains.py` promotes all 12 AutoML brains to full 2,000
   epochs.
3. `launch_deft_oof.py` runs four 20-epoch OOF folds per backbone.
4. `continue_deft_after_oof.py` scores all 316 training samples, extracts
   embeddings, constructs one model-driven mix25 snapshot per backbone, and
   launches the downstream DEFT stages.
5. `launch_deft_full2000.py` trains the three standalone DEFT snapshots.
6. `launch_deft_automl_campaign.py`,
   `continue_deft_automl_promotions.py`, and
   `launch_deft_automl_full2000.py` run and promote the 12 AutoML-on-DEFT
   brains. Promotion learning rates are capped at `6e-5`.
7. `launch_downstream_fusion_soups.py` launches six independent 8-GPU jobs:
   probability/class-rank prediction fusion and same-backbone checkpoint
   interpolation for FAN-base, FAN-large, and MiT-B5.

The current DEFT implementation performs one outer DEFT round on demand. It
does not expose `num_deft_iterations=5|10|20`. The four folds, 20 OOF epochs,
three neighbors, and 25% duplication fraction are inner-stage parameters, not
outer iteration counts.

## DEFT snapshot definition

For each backbone:

- Four OOF models score held-out samples so no sample is scored by a model that
  trained on it.
- Difficulty is `0.45 * mIoU-error-rank + 0.35 * rare-class-recall-error-rank
  + 0.20 * boundary-F1-error-rank`.
- The top 20% (64 samples) are hard anchors.
- Each anchor votes to its three cosine-nearest neighbors in backbone
  embedding space.
- Selection priority is `0.70 * difficulty-rank + 0.30 * neighbor-vote-rank`.
- Seventy-nine rows are duplicated, producing a 395-row mix25 snapshot from
  the original 316-row training set.
- Validation data never participates in difficulty scoring or snapshot
  construction.

## Fusion and soup contract

Every fusion/soup vector is ordered as:

1. original-data AutoML BFBO;
2. standalone DEFT mix25;
3. AutoML BFBO trained on the fixed DEFT mix25 snapshot.

The fusion scorer evaluates pairwise tenths and a three-model quarter simplex
for both probability fusion and per-pixel class-rank fusion. Exact rank ties
use an infinitesimal probability term only as a deterministic tie-break.

The soup scorer evaluates all single checkpoints, pairwise quarter-step
interpolations, and a near-uniform three-checkpoint average. It rejects
cross-backbone averaging and validates identical state-dict keys/shapes.

Candidate weights are selected only on the labeled validation split. Result
acceptance requires `sample_count == 1262`; checkpoint-soup completion also
requires a valid checkpoint larger than 1 MiB.

See [DOWNSTREAM_FUSION_SOUP_RESULTS.md](DOWNSTREAM_FUSION_SOUP_RESULTS.md) and
the exact JSON artifacts under `results/`.

## Required code fixes

The runtime patch files under `patches/` correspond to TAO PyTorch commits:

- `e1d5bbe` — globally reduce additive confusion-matrix counts before mIoU.
- `791fe7c` — make distributed visualization-directory creation atomic.

The fixed-range AutoML behavior used commit `5a3e73e` from the
`fix/fixed-automl-ranges` branch of `tao-automl`.

The first downstream launch exposed a direct-loader issue: serialized training
specs omit `evaluate` and `inference` because normal TAO entrypoints receive
those sections from Hydra defaults. The corrected scorers supply the minimal
runtime sections and the launcher validates artifacts before accepting backend
`COMPLETE`. Full job provenance is in
`results/downstream_fusion_soup_runtime_correction.json`.
