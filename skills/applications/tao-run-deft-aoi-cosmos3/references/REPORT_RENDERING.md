# DEFT AOI report rendering

`scripts/render_report.py` is the only report producer. It reads committed
state and recorded artifacts, escapes untrusted text, fills
`DEFT_Loop_Report.html`, and writes atomically. Do not hand-edit the rendered
report.

Render once after initialization, after each successful stage commit, and
after finalization. A terminal claim requires:

```bash
scripts/render_report.py --results-dir "$RESULTS_DIR" --require-terminal
```

The report contains:

1. run configuration and outcome, including model lineage, direct local base
   path, platform, one-H200 shape, iteration/epoch budget, and KPI contract;
2. frozen Benchmark KPI trend from committed `metric_result.json` artifacts;
3. split isolation and exact OK/NG label counts;
4. bounded prompt examples from recorded Proxy/Benchmark/Mining annotations;
5. iteration metric and confusion tables;
6. pipeline event rows with measured durations;
7. Mining Volume with raw candidates, cosine-kept candidates, new unique
   targets, and cumulative Train rows;
8. recorded checkpoints, Framework TOML files, evaluation outputs, RCCA,
   routing, mining, assembled Train JSON, and validation evidence;
9. committed hard stops and warnings.

All numbers must come from state or files referenced by state. Missing files
render as unavailable or recorded, never as fabricated zeros. Proxy is shown
as not run for a terminal iteration when the frozen Benchmark gate already
ended the loop.
