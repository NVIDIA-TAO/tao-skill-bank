# Cosmos Framework train, evaluate, and inference

Resolve `nvidia/Cosmos3-Nano` with action plus `workload=deft-aoi`. The selected
backend is `cosmos-framework`. Generic model workflows retain their own
backend policy; an explicit supported backend wins.

The default model input is a complete local Qwen3-VL Hugging Face snapshot at
`WORKSPACE/models/Cosmos3-Nano-VLM`, including `config.json`, processor files,
and safetensors. If the source is Omni instead, use the model skill's native
Framework converter and record its provenance.

Render training with `render_cfw_sft.py` and supply the canonical workspace as
its media root. The full profile is the reviewed
8-GPU full-parameter BF16 recipe in `cosmos_framework_sft_full.toml`; smoke is
explicit. The packaged `nvpaw_cfw` experiment consumes an indexed, sealed
JSONL dataset and writes synchronous Framework DCP.

Render evaluation with `render_cfw_evaluate.py`, including the same media root,
and plan both evaluation and inference with `cfw_action_plan.py`. The descriptor
runs `cfw_jsonl_runtime.py` inside the Framework image. That adapter streams
canonical JSONL and uses the Framework Transformers shim while retaining full
native messages, image order, and per-image pixel bounds. For a DCP, invoke the
model skill's `framework_checkpoint_action.py prepare` first and pass its
verified action-model directory to the runtime. The DCP is never loaded as an
HF directory. The adapter writes atomic normalized prediction JSONL;
`cfw_predictions.py` validates any external/sharded result before
`exact_f1_adapter.py`. Application planners never submit directly; the selected
platform owns the four verbs and job-record.
