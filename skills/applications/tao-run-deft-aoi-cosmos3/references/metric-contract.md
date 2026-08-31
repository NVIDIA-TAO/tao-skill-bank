# Exact F1 metric contract

The only KPI authority is the recorded absolute workspace
`eval/calculate_f1_metrics.py`. `exact_f1_adapter.py` invokes it into a fresh
temporary output, atomically preserves the resulting raw JSON, and records the
raw report's absolute path and SHA-256. `record_metric_result.py` verifies that
binding before commit, so a stale or modified report cannot satisfy the gate.
The `f1_cohort_balanced_v1` contract freezes five paths:

```text
non_reference_based.tasks.BCQ.macro_f1
non_reference_based.tasks.MCQ.macro_f1
non_reference_based.tasks.DET.f1
reference_based.tasks.BCQ.macro_f1
reference_based.tasks.DET.f1
```

The evaluator report physically nests these below
`tasks_by_reference_cohort`; the adapter records the stable flattened labels.
Each component must exist, be finite in `[0,1]`, and meet the frozen component
threshold. Missing evaluated predictions and unknown prediction IDs must both
be zero. The primary scalar is the minimum capped threshold attainment; raw
component values and deterministic tie breakers remain in the result. No app
module independently implements F1.
