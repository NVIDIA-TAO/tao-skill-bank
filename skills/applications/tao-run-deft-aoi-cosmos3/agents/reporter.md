# Cosmos3 DEFT Reporter Agent

Render `${RESULTS_DIR}/DEFT_Loop_Report.html` for
`tao-run-deft-aoi-cosmos3`.

Inputs:

- `results_dir`: absolute run directory;
- `skill_root`: absolute skill directory;
- `trigger`: `iteration-complete` or `loop-end`.

Procedure:

1. Run `scripts/deft_python.sh scripts/audit_deft_run.py --json` against
   `results_dir`. Stop on `INVALID`.
2. Read `references/REPORT_RENDERING.md`.
3. Read canonical `deft_state.json`, `loop_log.jsonl`, and only the artifact
   files referenced by state.
4. Render the required tables. Treat Proxy as RCCA-only and Benchmark as
   stop-gate-only. Report annotation mode as `bare_okng`. Attribute training
   records to their producer — mined real pairs or synthetic AnomalyGen pairs —
   and render a skipped `anomalygen` stage as the documented skip it is, with
   its reason, not as a missing artifact. The run's terminal
   iteration has Benchmark evidence but no Proxy artifacts by design; render
   those cells as `not run (terminal iteration)` and never as a missing
   artifact or an incomplete iteration.
5. Write a temporary sibling HTML file, verify it is non-empty and contains all
   required section headings, then atomically replace the report.

Never edit state/log, inspect credential files, use prior assistant prose as
evidence, or invent missing values.
