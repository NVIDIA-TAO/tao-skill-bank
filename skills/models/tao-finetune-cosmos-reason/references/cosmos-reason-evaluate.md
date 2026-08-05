# Cosmos Evaluate and Datasets

Load this only when `SKILL.md` points here. If this conflicts with `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the current/compact source wins.

## Evaluate

The `actions.evaluate` block in `references/skill_info.yaml` declares the action's inputs (annotation file + media folder + model) and outputs (results directory).

### Framework DCP pre-action

When `backend=cosmos-framework`, never pass a native Framework DCP directly to
the generic action or ask the user to export it. First run the checked-in
`scripts/framework_checkpoint_action.py plan` helper with the selected
checkpoint, saved Framework config when it cannot be inferred, base-model
identity/revision, and optional export directory. Then run `prepare` in the
clean repository-derived Framework TAO action image. It invokes the
Framework-owned exact-key exporter only when its manifest/config/DCP/base-model
checks do not prove that a complete export already exists.

Capture the helper JSON in the job metadata, run `verify` from the target
compute frame, put its `action_model_path` in `model.model_name`, and launch
`cosmos-framework-evaluate --config <config>`. The same pre-action contract
applies to `cosmos-framework-inference` and
`cosmos-framework-inference-microservice`, using `model_path`. Export child
failure or provenance mismatch blocks the downstream action. Framework PEFT
checkpoints are reconstructed from the saved config and merged by the native
exporter; do not set `model.enable_lora=true` after that merge.

### Config format

The evaluator reads a **flat TOML** config with top-level keys: `dataset`,
`model`, `task`, `evaluation`, `vision`, `generation`, `metrics`, `results`,
`num_gpus`, and `results_dir`. The defaults template
(`references/spec_template_evaluate.yaml`) matches this flat structure. Use
dotted overrides such as `dataset.annotation_path`, `model.model_name`, and
`evaluation.batch_size`.

### Task type

- Empty string (`""`) — General Evaluator. Auto-detects binary classification (yes/no) from ground truth and computes TP/FP/TN/FN/accuracy/precision/recall/F1.
- `"its_directionality"` — ITS-specific evaluator for left/right/straight classification. Do NOT use for collision detection.

### Cosmos-RL LoRA evaluation

To evaluate a fine-tuned LoRA model, pass the checkpoint path via spec_overrides:

```python
spec_overrides={
    'model.model_name': 's3://bucket/results/{train_job_id}/safetensors/epoch_2',
    'model.enable_lora': True,
    'model.base_model_path': '<BASE_MODEL_PATH_OR_URI>',
    'evaluation.batch_size': 10,
}
```

The LoRA adapter is downloaded from S3/Lustre before the evaluator runs; the evaluator merges it with the base model and runs inference on the merged weights.

### Selective download

When the input declaration carries a `selective` block (`{annotation, format, keys}`), only the files referenced in `dataset.annotation_path` (under the `video` key) are pulled — not the full media folder. For a 112-sample collision dataset, this downloads ~500MB instead of the full 4.8GB folder.

### Results

- `results.json` — per-sample predictions with `video_id`, `response`, `question`, `gt`
- Binary metrics: accuracy, balanced accuracy, precision, recall, F1
- Text metrics: BLEU, ROUGE, BERTScore
- When Lustre is available, results write to Lustre for cross-job persistence (e.g., gap analysis reads directly), then upload to S3.

## Datasets

The `data_sources` config in config.json maps dataset URIs to spec paths. It
appends `annotations.json` to the dataset directory URI by convention. If your
annotations and media do not share a root, or if the annotation file has a
different name, use direct spec overrides instead of forcing a root:

```python
spec_overrides={
    'custom.train_dataset': {
        'annotation_path': 's3://bucket/train/my_annotations.json',
        'media_path': 's3://bucket/media/videos_train.tar.gz',
    },
    'custom.val_dataset': {
        'annotation_path': 's3://bucket/eval/my_annotations.json',
        'media_path': 's3://bucket/eval/videos/',
    },
}
```

**Eval dataset** is optional for plain training only when `train.train_policy.dataset.test_size` is used to auto-split training data. For AutoML or any workflow optimizing a validation metric such as `val/avg_loss`, require either an explicit `custom.val_dataset` or a deliberate auto-split setting before launch preflight passes. If a validation dataset is provided, validation metrics are computed at the frequency set by `validation.freq_in_epoch`.

Do not infer dataset paths from prior validation runs. Ask the user for the
train and validation roots or direct spec paths unless a selected workflow
profile explicitly supplies them. Missing optional annotation fields are not a
launch blocker for current Cosmos-RL SFT training.
