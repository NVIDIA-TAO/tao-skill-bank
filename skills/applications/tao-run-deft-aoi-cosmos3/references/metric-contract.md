# Frozen Benchmark Metric Contract

`init_deft_state.py` writes one approved metric contract. Default:

```json
{
  "name": "recall_ng",
  "display_name": "NG recall",
  "operator": ">=",
  "target": 1.0,
  "unit": "",
  "evaluator": {
    "type": "artifact",
    "producer": "scripts/analyze_gaps.py",
    "path_template": "<results>/{iter_label}/benchmark_metrics/metric_result.json"
  },
  "constraints": [
    {
      "name": "unknown_predictions",
      "operator": "<=",
      "target": 0,
      "unit": ""
    }
  ]
}
```

The contract is the source of truth for stop decisions and best-iteration
selection. Proxy metrics are diagnostic and never committed as the iteration
metric result.

Supported primary metrics in this migration are `recall_ng`,
`precision_ng`, `f1_ng`, and `accuracy`, all compared with `>=`. Values and
targets are fractions in `[0, 1]`.

`benchmark_metrics/metric_result.json` must match the configured name and unit
exactly, and must provide the value and constraints. `record_metric_result.py`
adds its absolute path as
evidence; `commit_stage.py` validates it before recording the stage.

## Rich task-balanced contracts

`task_balanced_v1` changes the primary metric to `balanced_score`, the minimum
attainment across the six required task groups. Each group uses its own
classification macro F1 or detection box-micro F1 against the frozen
`group_metric_target`. Ties prefer larger macro attainment, smaller attainment
spread, fewer coverage failures, then the earlier iteration.

The rich contract freezes its KPI profile, required groups, target, minimum
support, constraints, and SHA-256 in `deft_state.json`. Any later contract
mutation fails before a metric is recorded. Missing, duplicate, unknown, and
parse-failed predictions must all be zero for a pass.

`task_dataset_balanced_v1` instead gates every observed task×dataset cell and
adds `insufficient_support_groups <= 0`. It is experimental and requires an
explicit `--min-group-support`; the default task profile reports those cells
diagnostically without letting sparse cells become hard gates.
