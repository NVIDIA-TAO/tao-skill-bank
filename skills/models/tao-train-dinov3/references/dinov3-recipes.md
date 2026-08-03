# DINOv3 tuning and evaluation

Use these as starting points and adjust them for the dataset, backbone, downstream task, and available compute.

## Start a run

- Begin with the train template and a checkpoint matching the configured backbone.
- Choose batch size and image resolution based on available memory.
- Scale learning-rate warmup, temperature warmup, and checkpoint intervals to the planned number of optimizer steps.
- Save multiple teacher checkpoints when the run is long enough to benefit from checkpoint comparison.

## Evaluate checkpoints

Training loss measures optimization progress but may not track downstream representation quality. Evaluate candidate teacher checkpoints using a held-out dataset and metrics relevant to the intended task.

Examples:

- Classification or retrieval: k-nearest-neighbor or linear evaluation on global features.
- Detection, segmentation, or depth: a lightweight head or decoder on patch features.
- Domain adaptation: compare in-domain performance and any generalization metrics that matter to deployment.

Keep evaluation samples separate from the unlabeled training corpus when possible.

## High-resolution training

`spec_template_train_highres.yaml` is an optional starting point for larger image sizes. High resolution uses more memory, so reduce batch size or use FSDP as needed. Gram anchoring can be enabled independently of resolution.

Compare the high-resolution result with a relevant base-resolution checkpoint before selecting a model.

## Checkpoint files

| File | Typical use |
|---|---|
| `model_epoch_*.pth` | Resume training, or export its EMA teacher |
| `teacher_epoch_*.pth` | Evaluation, conversion, export, inference, or another training seed |
| `student_epoch_*.pth` | Diagnostics |

## Resource sizing

Memory use grows with backbone size, image resolution, batch size, and attention implementation. If a configuration runs out of memory, reduce batch size or resolution, disable custom attention if it is incompatible with the environment, or select an appropriate distributed strategy.
