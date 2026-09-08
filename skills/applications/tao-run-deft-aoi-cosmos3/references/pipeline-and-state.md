# Pipeline and state

Schema version 7 is a fail-closed Cosmos Framework state:

```text
baseline evaluate_benchmark -> benchmark_metrics -> evaluate_proxy -> proxy_rcca
  -> routing -> data_mining -> assemble_data -> validate_data -> train
  -> evaluate_proxy -> proxy_rcca
       cadence=every or Proxy-best or final -> evaluate_benchmark -> benchmark_metrics
       otherwise -> next routing
  Benchmark metric met -> loop_stop
```

State initialization freezes `benchmark_cadence` (`every` or
`final_and_best`), absolute annotation/evaluator paths and hashes,
the five-component F1 contract, model path, Framework/data-services images,
recipe profile, platform, mining policy, and split contract. The event list is
append-only. `deft_context.py` derives the next stage; `commit_stage.py`
validates artifacts, rejects any label/stage other than the derived next stage,
and atomically advances state. A state with another schema version must not
resume.

Every `proxy_rcca` commit includes exact five-component Proxy KPI evidence and
records whether its deterministic KPI/tie-breaker ranking is best so far.
Proxy remains the sole gap-analysis source. Under `final_and_best`, the frozen
Benchmark runs for the validated reusable zero-shot baseline, every Proxy-best
checkpoint, and the final checkpoint; under `every`, it runs for all of them.

Training commits require a complete `iter_#########` DCP with non-empty
metadata and shard files plus adjacent config. Evaluate/inference first consume
the verified action model produced by the DCP pre-action. Benchmark commits
require exact prediction coverage plus a raw evaluator report whose absolute
path and SHA-256 match the frozen metric result. Only `loop_stop` makes the
overall run terminal complete.
