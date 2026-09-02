# Script ownership

| Script | Contract |
| --- | --- |
| `init_deft_state.py` | Freeze version-7 paths, hashes, backend, recipe, metric, and mining configuration once. |
| `deft_context.py` | Read state and identify the only valid next stage. |
| `commit_stage.py` | Validate evidence and atomically append a stage event. |
| `render_cfw_sft.py` | Render full or explicit smoke CFW SFT TOML and a sealed descriptor. |
| `nvpaw_cfw/*` | Indexed JSONL dataset, deterministic distributor/processor, experiment registration, and trainer entrypoint. |
| `cfw_action_plan.py` | Emit platform-neutral train/evaluate/inference descriptors. |
| `render_cfw_evaluate.py` | Render multi-task native-message evaluation config. |
| `cfw_jsonl_runtime.py` | Stream canonical JSONL through the Framework Transformers shim and atomically emit normalized multi-image predictions. |
| `cfw_dcp.py` | Validate synchronous Framework DCP completeness and manifest. |
| `cfw_predictions.py` | Strictly normalize/validate external or sharded results with exact source coverage and preserved message/image order. |
| `exact_f1_adapter.py` | Invoke the recorded workspace evaluator and build the five-component gate without recalculation. |
| `analyze_gaps.py` | Reuse the recorded evaluator's parsers for Proxy record-level weakness and task-aware RCCA selection. |
| `route_selected_gaps.py` | Collapse Proxy gap rows into real-image Mining queries. |
| `task_mining_router.py` | Apply deterministic task-aware neighbor routing and history policy. |
| `select_detection_calibration.py` | Prepend auditable empty/few-box Mining calibration candidates before history filtering. |
| `emit_mined_sharegpt.py` | Emit selected canonical Mining rows as JSONL; the legacy filename is retained for CLI compatibility. |
| `assemble_training_json.py` | Assemble monotonic real-only training JSONL. |
| `validate_sharegpt.py` | Validate canonical NVPAW JSONL; the legacy filename is retained for CLI compatibility. |
| `validate_split_contract.py` | Prove evaluation isolation, real-mining eligibility, prior-row retention, and Benchmark hash. |
| `render_report.py` | Render the durable state-backed HTML report. |

Application scripts do not pull images or submit jobs. The selected platform
owns all native execution through `submit/status/logs/cancel` and job-records.
The required Proxy RCCA artifact names, state fields, formats, and report
headings are mirrored in `references/rcca-artifact-manifest.json` for packaging
and agent handoff.
