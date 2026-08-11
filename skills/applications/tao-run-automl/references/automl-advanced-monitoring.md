# AutoML Monitoring And Recovery

## Two sources of truth

`automl_step.py status --full` is authoritative for optimizer state. The bound
platform job is authoritative for runtime state and logs. Never infer a live
backend state from the job record alone.

For each active recommendation report its id/config id, bracket/rung/generation,
budget, parent/resume record, platform id/state, current observations, selected
parameters, elapsed time, and current best at the largest comparable budget.

## Recovery sequence

1. Lock/read the experiment with `status --full`.
2. Poll every READY/RUNNING recommendation's platform job.
3. If a RUNNING job is terminal, materialize and validate its metric record,
   then call `report`.
4. If the backend lost a job for infrastructure reasons, report `ERR_INFRA`,
   submit a linked retry, and bind the new id.
5. Call `recommend`; idempotency prevents duplicate work.

## Intermediate metrics

Use `observe` only for metrics emitted by the bound running job. A
Hyperband-ES `should_cancel` decision is advisory until the platform cancel
verb succeeds; then report CANCELED. Keep exact observation steps and metric
names so recovery cannot double-count or rewrite a curve.

## LLM diagnostics

Inspect `metadata.llm` for provider/model, purpose, attempt count, latency,
usage, proposal reason, and `fallback`. Never log or persist credentials. A
requested LLM/hybrid/autoresearch run is not validated as language-guided when
fallback is true, even though deterministic fallback keeps state recoverable.

## Completion audit

Before finalization, require no READY/RUNNING recommendations, complete
promotion/generation lineage, a successful largest-budget winner, declared
artifact existence, a valid `best_rec.json`, and a final evaluation record on
the baseline dataset/metric. Summarize failed trials and shared root causes.
