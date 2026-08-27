# Bundled scripts and ownership

Run Python commands with `PYTHON=$(scripts/deft_python.sh)`.

| Script | Responsibility |
|---|---|
| `init_deft_state.py` | Validate immutable run inputs and create version-6 state once. |
| `deft_context.py` | Print the deterministic next stage from durable state. |
| `deft_exec.py` | Enforce recorded network policy for host commands. |
| `commit_stage.py` | Validate stage evidence and atomically commit snapshot plus event. |
| `finalize_run.py` | Verify terminal Benchmark evidence and commit `loop_stop`. |
| `validate_sharegpt.py` | Enforce one-image, exact OK/NG JSON-array records. |
| `validate_split_contract.py` | Enforce split isolation, frozen Benchmark, and Mining-only Train lineage. |
| `analyze_gaps.py` | Produce Proxy RCCA or Benchmark aggregate metrics. |
| `record_metric_result.py` | Bind Benchmark metrics to the frozen KPI contract. |
| `render_cfw_sft.py` | Render native Framework SFT TOML with optional previous DCP. |
| `submit_cfw_train.py` | Compose/execute recorded Docker Train submission. |
| `render_cfw_evaluate.py` | Render one-H200 bare-label evaluation TOML. |
| `submit_cfw_evaluate.py` | Compose/execute recorded Docker Evaluate submission. |
| `submit_cfw_inference.py` | Compose/execute recorded Docker Inference submission. |
| `emit_mined_sharegpt.py` | Align selected real paths to Mining prompt and label. |
| `assemble_training_json.py` | Append current Mining records to the prior Train JSON. |
| `filter_mined_by_cosine.py` | Apply the recorded similarity floor. |
| `render_report.py` | Render the evidence-backed HTML report. |

The application agent owns ordering, confirmation, state commits, job-records,
and finalization. The model skill owns native action semantics. The platform
skill owns `submit/status/logs/cancel` and live state mapping. The mining skill
owns embedding and `filter_by_label` selection. Do not delegate state writes or
hand-author report HTML.

The same Framework image and public console scripts are used for all model GPU
actions. The app never invokes an internal module path and never inserts a
separate checkpoint-processing stage.
