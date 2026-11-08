# Aggregate-feedback weight-edit experiment tools

These experimental host tools keep the editor in a development-only container, bound aggregate feedback queries, and evaluate the sealed split only after the editor stops. The algebraic tensor replacement primitive is independent of task/model names. THT-specific partitioning and membership checks belong to this application.

Set `WSP05_RUN_ROOT` to an existing private experiment root, `WSP05_BANK` to the immutable code bank, and `PYTHONPATH` to the parent application scripts directory. The partitioner additionally takes `WSP05_PREVIOUS_REQUEST_CONFIG`, which binds original request/image views and the frozen evaluator. Never expose this root's `private` subtree to an editor or print row-bearing files.

Use `partition_private.py` to fix seed42 joint groups before development feedback. `oracle.py serve` admits at most20 aggregate queries; each query binds a prior hypothesis, plan, source and exact tensor files. The editor protocol has no subset or row-level API. Failed admitted queries count; no inference is automatically retried. Public responses contain only approved aggregates and costs. Match unedited parent and reference profile for every selected configuration.

After the scientist writes its finite freeze and stops, run `audit_development.py`, then `final_evaluate.py`. The final gate rejects a running editor, active oracle, changed plan, unmatched baseline, failed independent audit, excess query budget or more than four unique finalists. Each frozen configuration is evaluated once on the sealed split; cached feedback predictions are joined privately for the explicitly Benchmark-informed full-cohort score. No training or optimizer is used.

The run root must have its pinned config, model bindings, context, fixture/isolation receipts and private partition before serving. `native_launcher.py` extends the existing light launcher with explicit read-only pristine-model mounts. These are protocol tools, not a replacement for scientific review or an automatically approved experiment.

Synthetic checks: run `test_oracle.py --output <new-path>` in the pinned CPU runtime (torch and safetensors required), and `test_final_gate.py --output <new-path>` on the host. Both use synthetic temporary fixtures only. The original experiment runtime source hashes remain separate from packaged-source hashes; no data, model weights, run configurations or result files are distributed here.
