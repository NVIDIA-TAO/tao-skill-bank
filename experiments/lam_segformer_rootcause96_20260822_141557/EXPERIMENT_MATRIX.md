# LAM SegFormer root-cause-to-96 campaign

Target: exceed `0.9600` global validation mIoU without training on validation
images. Every SLURM allocation uses one 8-GPU node and a 2,000-epoch budget for
full runs. Native input resolution remains 1024 because every source image and
mask is already 1024x1024; larger settings only interpolate the same pixels.

## Fixed defects under test

1. Pass `train.segment.weights` to cross entropy and reject malformed vectors.
2. Add a stable weighted-CE + minimax-IoU composite loss.
3. Honor `random_color.color_probability` (previously a searched no-op).
4. Honor configured AdamW beta1 and add warmup-cosine scheduling.
5. Pass activation checkpointing through the SegFormer model builder.
6. Register C-RADIOv4-H and C-RADIOv4-SO400M SegFormer adapters.
7. Add a dense, dynamic-resolution ViT-5-L/16 adapter using the released
   ImageNet-1K checkpoint, memory-efficient SDPA, and checkpoint-compatible
   `[CLS, patches, registers]` ordering.

## Causal FAN-large runs (14)

| ID | Augmentation | Loss / pixel-normalized weights | Schedule |
|---|---|---|---|
| C01 | historical always-on crop + scale-crop + blur + color | unweighted CE | linear |
| C02 | D4 geometry only; no crop, blur, or color | unweighted CE | linear |
| C03 | D4 only | weighted CE `[0.815594,3.042000,2.551302,0.864234]` | linear |
| C04 | D4 only | weighted CE `[0.661819,5.058669,3.952071,0.713392]` | linear |
| C05 | D4 only | weighted CE `[0.516171,7.180647,5.050901,0.579573]` | linear |
| C06 | D4 only | C03 CE + 0.25 minimax-IoU | linear |
| C07 | D4 only | C03 CE + 0.50 minimax-IoU | linear |
| C08 | D4 only | pure minimax-IoU | linear |
| C09 | D4 only | C03 weighted CE | cosine |
| C10 | D4 + grayscale brightness/contrast, p=0.5 | C03 weighted CE | cosine |
| C11 | no augmentation | C03 weighted CE | cosine |
| C12 | flips only; no rotations/crop/blur/color | C03 weighted CE | cosine |
| C13 | D4 only | C03 CE + 0.25 Lovasz-Softmax | cosine |
| C14 | D4 only | C03 CE + 0.25 boundary F-score loss | cosine |

`D4` means horizontal/vertical flips and right-angle rotations, transformations
that preserve the geometry and label map without interpolation blur.

## DEFT-data correction runs (2)

| ID | Backbone/data | Fixed recipe |
|---|---|---|
| D01 | FAN base / DEFT mix100 | D4, C03 weighted CE, cosine |
| D02 | FAN large / DEFT mix50 | D4, C03 weighted CE, cosine |

## New-backbone runs (12)

Each backbone receives both adapter/decoder-only (`freeze_backbone=true`) and
full fine-tuning (`freeze_backbone=false`). Both use activation checkpointing,
D4, C03 weighted CE, cosine, and their own pretrained checkpoint.

| IDs | SegFormer backbone | Support path |
|---|---|---|
| B01/B02 | DINOv3 ViT-L/16 | native TAO 7.1 adapter; staged timm weights |
| B03/B04 | DINOv3 ViT-H+/16 | native TAO 7.1 adapter; staged timm weights |
| B05/B06 | C-RADIOv3-L | existing TAO adapter; newly exposed in schema |
| B07/B08 | C-RADIOv4-H | new SegFormer adapter and schema support |
| B09/B10 | C-RADIOv4-SO400M | new 27-block SegFormer adapter and schema support |
| B11/B12 | ViT-5-L/16 (2026) | new TAO dense adapter; official Apache-2.0 224 checkpoint; interpolated absolute positions + dynamic 2D RoPE |

Before the twelve full runs, six 1-epoch probes each execute frozen and unfrozen
training in the real TAO image. Full runs are submitted with `afterok`
dependencies so a bad checkpoint mapping or OOM cannot burn a 2,000-epoch
allocation. Activation checkpointing is enabled for every large ViT/RADIO run.
ViT-5-XL is not included because no official XL checkpoint is currently
released; the released ViT-5-L/16-224 is the largest reproducible option.

## Dependent fusion and post-processing jobs (7)

1. Cache validation logits for every successful corrected model.
2. Score uniform probability, raw-logit, geometric, vote, and per-pixel rank fusion.
3. Fit non-negative global model weights with sequence-group cross-validation.
4. Fit class-specific model weights, emphasizing the class-2 bottleneck, with the same cross-validation.
5. Score D4 test-time augmentation and its probability fusion.
6. Score within-run temporal checkpoint soups/SWA around each run's validation optimum.
7. Score original-data + DEFT-data pair fusions and the combined all-model ensemble.

No fusion weight, threshold, or post-processing parameter is selected on a
separate claimed test set: the present `test/images` are exact duplicates of
`val/images`, and no independent test masks exist. Validation mIoU therefore
remains the only honest labeled target for this campaign.

## Launch DAG and maximum allocation

- 1 patched-runtime probe.
- 6 new-backbone probes.
- 28 full 2,000-epoch training jobs.
- 7 dependent fusion/post-processing jobs.
- 42 SLURM job records total, each requesting 8 GPUs on
  `polar,polar3,polar4,grizzly` with a four-hour allocation and checkpointed
  requeue/resume for runs that exceed one allocation.
- Dependency-aware peak for this new campaign: 28 nodes / 224 GPUs during the
  full-training wave. If the 12 active AutoML allocations overlap, the combined
  request is 40 nodes / 320 GPUs. One four-hour allocation for all 42 new jobs
  totals 1,344 GPU-hours; usage may be higher when large backbones requeue.
