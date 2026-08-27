# Pipeline and durable state

The gate-first transition graph is:

```text
baseline/evaluate_benchmark -> benchmark_metrics
  pass -> loop_stop
  fail -> evaluate_proxy -> proxy_rcca
       -> iter1/routing -> data_mining -> assemble_data -> validate_data
       -> train -> evaluate_benchmark -> benchmark_metrics
          pass or iter5 -> loop_stop
          otherwise -> evaluate_proxy -> proxy_rcca -> next iteration routing
```

`deft_state.json` is the single resume source. Initialization writes version 6
exactly once and freezes the Benchmark path/hash, local base path, platform,
image digest, metric contract, compute, training parameters, mining policy,
and iteration budget. Every `commit_stage.py` call appends a monotonically
increasing event and advances one iteration snapshot atomically.

## Stage artifacts

- `evaluate_benchmark`: native Framework result JSON.
- `benchmark_metrics`: metrics summary and contract-bound metric result.
- `evaluate_proxy`: native Framework result JSON.
- `proxy_rcca`: gaps summary, false accepts, false rejects, and validated RCCA
  Markdown.
- `routing`: JSON state artifact and equivalent parquet rows derived only from
  Proxy errors.
- `data_mining`: mined parquet, cosine-filter summary, durable filepath history,
  and kept-row count. OK and NG are queried separately with
  `filter_by_label`.
- `assemble_data`: aligned Mining ShareGPT JSON, monotonic Train JSON, and
  assembly summary.
- `validate_data`: exact-label/file/split/lineage report and frozen Benchmark
  verification.
- `train`: native DCP directory and the exact Framework SFT TOML that produced
  it.
- `loop_stop`: explicit reason and final report.

Iteration 1 Train uses the local HF base. Iteration N evaluation consumes its
DCP with its saved TOML and original HF vision checkpoint. Iteration N+1 sets
that DCP as `checkpoint.load_path`. No intermediate app stage sits between
Train and either consumer.

Executed stages require a positive measured duration. A backend error is
committed as `status=error` before stopping. Use `deft_context.py` before every
stage; never infer a next stage from filesystem presence alone.
