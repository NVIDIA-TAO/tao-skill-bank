# Cosmos3 DEFT Report Rendering

Use `scripts/render_report.py` as the only renderer for the source template at
`references/DEFT_Loop_Report.html`. It writes one self-contained
`${RESULTS_DIR}/DEFT_Loop_Report.html` from canonical disk evidence through a
temporary sibling and atomic replacement. `init_deft_state.py` invokes it at
loop start and `commit_stage.py` invokes it after every valid commit, so report
freshness never depends on an agent remembering a final task.

The Cosmos3 report intentionally shares the ChangeNet report's visual contract:
the same 1280 px single-column hero and cards, NVIDIA dark palette, typography,
32 px vertical rhythm, section headers, table treatment, KPI banner, and status
panels. Model-specific content and evidence differ, but the report must not
introduce a separate grid, card geometry, or visual hierarchy. The renderer
test compares these shared CSS properties across both source templates.

Required sections:

1. **Run Configuration & Outcome** — platform, model, bare mode, KPI contract,
   GPU/node shape, timestamps, current/terminal status.
2. **Dataset Isolation** — Proxy/Benchmark/Mining input paths, per-iteration
   AnomalyGen output, generated Train artifacts by iteration, OK/NG counts,
   Benchmark SHA-256, and role ownership. Attribute every Train record to its
   producer: mined real pairs or synthetic AnomalyGen pairs.
3. **Prompt Examples** — up to three distinct first-user prompts read from the
   recorded Proxy, Benchmark, and Mining annotations. Group identical prompts
   across roles, show their record count and exact `OK`/`NG` response contract,
   escape file-derived text, and truncate only the displayed preview at 600
   characters. Do not embed the rest of an annotation record.
4. **Iteration Metrics** — for baseline and every completed iteration:
   accuracy, NG recall, NG precision, NG F1, false accepts, false rejects,
   unknowns, KPI pass/fail. Label every figure with its source split; only
   Benchmark figures carry the KPI verdict.
5. **Pipeline Execution** — ordered committed stage events and durations from
   `loop_log.jsonl`.
6. **Augmentation Volume** — per iteration and per producer. Mining: raw
   candidates, cosine-kept paths. AnomalyGen: requested `num_SDG`,
   AMP-allocated, generated, and the per-defect-type breakdown, or the
   documented skip and its reason. Then newly added unique target images and
   cumulative training records by iteration.
7. **Artifacts** — baseline base-model reference, iteration checkpoints,
   Proxy/Benchmark results, RCCA, mining, AnomalyGen `SDG_result.csv` and
   generated ShareGPT, assembled JSON, and validation-report links/paths.
8. **Hard Stops / Warnings** — canonical audit warnings and error event, if
   present.

Cosmos3 bare mode is a discrete OK/NG classifier.

## Terminal iterations have no Proxy artifacts

The loop evaluates the frozen Benchmark gate first and runs Proxy evaluate and
RCCA only when it continues to another iteration. The iteration that ends a run
— gate passed, or `N = max_iterations` — therefore has Benchmark results and
KPI evidence but no `proxy_results_json`, `proxy_gaps_summary`,
`false_accepts_json`, or `false_rejects_json`. Its `stage_completed` is
`benchmark_metrics` rather than `proxy_rcca`.

This is the designed shape, not a gap. Render those cells as
`not run (terminal iteration)`, not as `not available`, and never present the
absence as a missing artifact, an incomplete iteration, or a failure. Do not
carry a previous iteration's Proxy numbers forward to fill them.

Escape all file-derived text before inserting it into HTML. Do not embed
credentials, environment values, full annotation records, or full model logs.
Mark missing optional artifacts as `not available`; never fabricate a metric or
stage.
