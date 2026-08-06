# Spec Param / Parent Model Inference

Model-specific inference mappings belong in this MD file, not in `config.json`. Generated runners must read the parent job record's fixed `results_dir`, select the exact artifact using the filename rules below, write the resulting path into the nested spec field, and submit through the selected platform skill. This is the skill-owned replacement for the old microservices `infer_params.py` flow.

DepthNet Mono training writes checkpoint files under `<results_dir>/train/`
using `model_epoch_<epoch>_step_<step>.pth` and a `dn_model_latest.pth`
symlink. For `evaluate`, `inference`, `export`, `quantize`, and resume/retrain,
read the parent job record's `results_dir` and match the requested best, epoch,
or step to that exact file. Use `dn_model_latest.pth` only when the user
explicitly asks for latest.

Parent PyT `gen_trt_engine` is intentionally not used because the current `depth_net` PyT entrypoint rejects it. The `gen_trt_engine` metadata selects the TAO Deploy container, and the deploy workflow owns the TensorRT-only action details.

Inference mappings from TAO Core `depth_net_mono.config.json`:

| Action | Spec Field | Inference Function | Meaning |
|---|---|---|---|
| evaluate | `dataset.dataset_name` | `MonoDataset` | MonoDataset |
| evaluate | `evaluate.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| evaluate | `evaluate.trt_engine` | `parent_model` | model file inferred from the parent job results folder |
| evaluate | `results_dir` | `output_dir` | current job results directory |
| export | `dataset.dataset_name` | `MonoDataset` | MonoDataset |
| export | `export.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| export | `export.onnx_file` | `create_onnx_file` | output ONNX path |
| export | `results_dir` | `output_dir` | current job results directory |
| inference | `dataset.dataset_name` | `MonoDataset` | MonoDataset |
| inference | `inference.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| inference | `inference.trt_engine` | `parent_model` | model file inferred from the parent job results folder |
| inference | `results_dir` | `output_dir` | current job results directory |
| quantize | `dataset.dataset_name` | `MonoDataset` | MonoDataset |
| quantize | `quantize.model_path` | `parent_model` | model file inferred from the parent job results folder |
| quantize | `results_dir` | `output_dir` | current job results directory |
| train | `dataset.dataset_name` | `MonoDataset` | MonoDataset |
| train | `model.mono_backbone.pretrained_path` | `{'link': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth', 'destination_path': '/ptm/depth_net/mono_backbone/dinov2_vitl14_pretrain.pth'}` | {'link': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth', 'destination_path': '/ptm/depth_net/mono_backbone/dinov2_vitl14_pretrain.pth'} |
| train | `results_dir` | `output_dir` | current job results directory |
| train | `train.pretrained_model_path` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `train.resume_training_checkpoint_path` | `resume_model` | model file inferred from the current job results folder |

For `parent_model` or `parent_model_folder`, retain the upstream train/export/AutoML child job id as `parent_job_id`. Read that job's immutable record to get `results_dir`, apply the model-specific artifact rules to select the exact file or folder, and write the concrete path into the nested spec. Do not add these mappings back to `config.json` or make an unvalidated path guess.
