# Report rendering

`render_report.py` reads only committed `deft_state.json`. The HTML reports
run status, Framework backend/recipe, exact evaluator and threshold,
per-iteration stage/metric/DCP/mining evidence, and the ordered event table.
It must not infer successful work from uncommitted files. Rendering is a
presentation hook outside the state transaction: a rendering error is
reported but cannot roll back a valid stage commit.
