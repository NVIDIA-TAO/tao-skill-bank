# Script ownership

| Script | Contract |
| --- | --- |
| `build_target_profile.py` | Analysis-only annotation geometry/dataset profile and target-versus-supply cell comparison; see [offline analysis tools](analysis-tools.md). |
| `build_validation_panel.py` | Analysis-only, prediction-free panel matching benchmark strata, excluding Benchmark/Proxy ids and either image, with a realized-family cap and explicit shortages. |
| `score_pool_residual.py` / `pool_residual_full.py` | Opt-in analysis sample/full-pool evaluation plans and CPU residual scoring using the authoritative evaluator; full mode renders a resumable, bounded array for operator submission, never submits or changes loop state. |
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
| `benchmark_cadence.py` | Validate exact Proxy KPI evidence and rank checkpoints for conditional Benchmark scoring. |
| `analyze_gaps.py` | Reuse the recorded evaluator's parsers for Proxy record-level weakness and task-aware RCCA selection. |
| `atomic_samples.py` | Define ordered single/pair identities and deterministic two-image embedding assets. |
| `build_mining_source_pool.py` | Build the source pool at atomic single-image/reference-pair granularity, with bounded parallel pair-asset materialization. |
| `route_selected_gaps.py` | Collapse Proxy gap rows into atomic Mining queries and pair embedding assets. |
| `coverage_stratified_selector.py` | Build/reuse the hash-bound parent inventory, solve shrunk Proxy-error quotas with source caps/floors, k-center cells, write the round manifest, and enforce the share gate. |
| `task_mining_router.py` | Apply the launch-recorded nearest-neighbor or coverage-stratified candidate policy before history filtering. |
| `select_detection_calibration.py` | Derive Proxy-bound per-cohort empty/few-box quotas, preserve reference pairs atomically, and optionally select Component Count replay before history filtering. |
| `defect_detection_ablation.py` | Materialize the task-strict Defect Detection corpus, enforce both detection-cohort empty rates, and bind its compatible quota manifest. |
| `repetition_blend.py` | Apply seeded deficit-proportional or explicit repetition after exclusion and bind its per-task manifest. |
| `emit_mined_sharegpt.py` | Emit selected atomic canonical Mining rows with both reference-pair sides intact; the legacy filename is retained for CLI compatibility. |
| `render_iteration_mining_runner.py` | Generate the immutable selector-to-assembler plan; selector output is always `data/mined.jsonl`, never cumulative Train. |
| `assemble_training_json.py` | Solely assemble monotonic cumulative training JSONL, bind previous path/hash/fingerprint lineage, bind the final quota manifest, and optionally apply the recorded repetition blend or add launch-recorded correct-row anchors (`--anchor-*`, see `anchor-and-coverage.md`). |
| `anchor_rows.py` / `build_anchor_candidates.py` | Correct-row anchor selection (task quotas from the KPI set, per-dataset cap, seed/id hash, corpus/evaluation exclusion) and the offline candidate builder that streams the canonical Mining JSONL against a scored-pool id list. Off unless `--anchor-share` is launch-recorded. |
| `coverage_rows.py` / `build_coverage_candidates.py` | Cross-dataset coverage blend (`plain` uniform pool rows or `residual` scored-wrong rows with tagged correct-row fallback): floor-first even round-robin over (task, dataset) cells, joint share solving with anchors, fail-closed floor check against the eventual budget, and the offline per-mode candidate builder. Off unless `--coverage-blend-share` is launch-recorded. |
| `validate_sharegpt.py` | Validate canonical NVPAW JSONL; the legacy filename is retained for CLI compatibility. |
| `validate_split_contract.py` | Prove evaluation isolation, real-mining eligibility, prior-row retention, and Benchmark hash. |
| `render_report.py` | Render the durable state-backed HTML report. |

Application scripts do not pull images or submit jobs. The selected platform
owns all native execution through `submit/status/logs/cancel` and job-records.
The required Proxy RCCA artifact names, state fields, formats, and report
headings are mirrored in `references/rcca-artifact-manifest.json` for packaging
and agent handoff.
