# Skill-Owned AutoML Step Engine

Use `scripts/automl_step.py` for every supported action and algorithm. The
engine owns optimizer state only. Platform skills exclusively own
`submit`/`status`/`logs`/`cancel`, so the same optimizer state works with
Docker, Kubernetes, SLURM, or Brev.

## Durable loop

```text
init -> recommend -> platform submit -> bind-job
     -> platform status/logs -> [observe] -> report -> recommend ... -> finalize
```

State is schema-validated, atomically replaced, protected by a file lock, and
safe to resume after process loss. An existing READY recommendation is returned
unchanged; a RUNNING recommendation remains bound to its job record.

## Initialize

```bash
SB="${TAO_SKILL_BANK_PATH:?}"
ENGINE="$SB/skills/applications/tao-run-automl/scripts/automl_step.py"

python "$ENGINE" init \
  --state "$WORKSPACE/automl_experiment.json" \
  --schema "$MODEL_SKILL/schemas/train.schema.json" \
  --base-spec "$WORKSPACE/base_spec.yaml" \
  --experiment-id "$EXPERIMENT_ID" \
  --network-arch "$NETWORK_ARCH" \
  --action train \
  --algorithm bayesian \
  --metric val_loss \
  --direction minimize \
  --max-recommendations 8 \
  --max-concurrent 1 \
  --candidate-count 1024 \
  --seed 17
```

Use repeated `--parameter dotted.path` to override the schema's
`automl_default_parameters`, or `--search-space ranges.yaml` for bounded range
overrides. The emitted recommendation includes both dotted `parameters` for
audit and a nested executable `spec`.

For multiple objectives, repeat:

```bash
--objective accuracy:maximize:2:1 \
--objective latency_ms:minimize:1:100
```

The fields are `name:direction[:weight[:scale]]`. Scalarized scores are always
maximized internally; raw objective values remain in state.

## Algorithms

- `bayesian`: Gaussian-process expected improvement.
- `bfbo`: batch GP-UCB with local penalization around active points.
- `hyperband`: all synchronous Hyperband brackets and rung promotion.
- `bohb`: Hyperband plus TPE-style good/bad density-ratio proposals.
- `asha`: asynchronous capacity refill and quota-based promotion.
- `pbt`: population generations with exploit, bounded perturbation, and resume.
- `dehb`: Hyperband budgets plus differential mutation/crossover once a
  population exists.
- `hyperband_es`: Hyperband plus learning-curve cancellation decisions.
- `llm`: OpenAI-compatible proposals with bounded vectors and explicit fallback
  metadata.
- `hybrid`: language-model phase planning over BFBO/Bayesian sub-searches.
- `autoresearch`: reflective proposals using results, failures, feedback, and
  an optional research program.

Budgeted algorithms use `--max-epochs`, `--reduction-factor`, and
`--epoch-multiplier`. PBT also uses `--population-size`, `--max-generations`,
`--eval-interval`, and `--perturbation-factor`. ASHA accepts `--max-trials` and
`--min-top-configs`. DEHB accepts `--mutation-factor` and
`--crossover-probability`.

Set ASHA `--max-trials` when a hard upper bound is required. Without it, ASHA
keeps filling capacity until `--min-top-configs` successful recommendations
reach the final rung, so repeated failed trials do not silently end the search.

## Recommend and submit

```bash
python "$ENGINE" recommend --state "$WORKSPACE/automl_experiment.json" \
  > "$WORKSPACE/recommend.json"
```

Submit every entry in `recommendations` through the selected platform skill.
Use the recommendation's nested `spec`, bind the resulting record id, and set
the experiment association by binding that id in AutoML state. Do not put an
experiment id in a job record's `parent_job`; that field is reserved for an
actual upstream job id. Promotions and PBT generations include `parent_rec_id`,
`resume_from_job_id`, and the target `budget`; use `resume_from_job_id` as the
child record's parent and resolve the exact checkpoint declared by that job.

```bash
python "$ENGINE" bind-job \
  --state "$WORKSPACE/automl_experiment.json" \
  --rec-id rec-0000 \
  --job-id "$JOB_ID"
```

## Intermediate observations

Record learning-curve data while a job is RUNNING:

```bash
python "$ENGINE" observe \
  --state "$WORKSPACE/automl_experiment.json" \
  --rec-id rec-0000 --job-id "$JOB_ID" --step 3 \
  --metric-value val_loss=0.42
```

If Hyperband-ES returns `should_cancel: true`, call the platform cancel verb,
then report the terminal cancellation. Replaying the same step and values is
idempotent; conflicting values are rejected.

## Terminal metric record

Prefer a canonical `metric_record.json`:

```json
{
  "schema_version": 1,
  "experiment_id": "automl-dino-01",
  "rec_id": "rec-0000",
  "job_id": "dino-train-000001",
  "status": "COMPLETE",
  "primary_metric": "val_loss",
  "direction": "minimize",
  "metrics": {"val_loss": 0.31, "latency_ms": 18.2},
  "artifacts": {"checkpoint_uri": "/results/rec-0000/model.pth"},
  "failure": null,
  "measured_at": "2026-07-21T12:00:00Z"
}
```

Training success requires `checkpoint_uri`; other actions require
`primary_uri` or a non-empty `output_uris`. Report it with:

```bash
python "$ENGINE" report \
  --state "$WORKSPACE/automl_experiment.json" \
  --rec-id rec-0000 --metric-record "$RESULTS/metric_record.json"
```

For a manually reconciled result, pass `--outcome`, `--job-id`, repeated
`--metric-value name=value`, and `--checkpoint-uri` or `--artifact-uri`.
Infrastructure errors return the exact recommendation to READY without
spending optimization budget. Program errors consume that trial.

## LLM algorithms

Pass `--llm-endpoint` and `--llm-model`, or export
`AUTOML_LLM_ENDPOINT`/`AUTOML_LLM_MODEL`. Credentials come only from
`AUTOML_LLM_API_KEY` or `NVIDIA_API_KEY` and are never persisted. Inspect
`recommendation.metadata.llm`: a requested language-model run is valid only
when `fallback` is false. Use `--evolvable-text-parameter dotted.path` only for
string-valued categorical schema fields. `--research-program` accepts a
JSON/YAML object for autoresearch context.

## GEPA

For prompt optimization, import the skill-owned `gepa_step.py` helpers. Wrap a
callback that executes one TAO action batch in `TAOActionBatchRunner`, score
aligned outputs with `TAOGEPAAdapter`, and call `GEPAutoPrompter.optimize`.
Official set-level validation reranks all candidates before the frozen winner
is evaluated once on the test set. Reflection records remove labels and private
media identity.

## Status and finalize

```bash
python "$ENGINE" status --state "$WORKSPACE/automl_experiment.json" --full

python "$ENGINE" finalize \
  --state "$WORKSPACE/automl_experiment.json" \
  --out "$WORKSPACE/best_rec.json"
```

For multi-fidelity methods, best selection is restricted to successful trials
at the largest observed budget. `best_rec.json` strips budget keys from winning
spec overrides and records them separately under `observed_budget`, preventing
a rung epoch count from overwriting a downstream full-training budget.
Multi-objective `status` output also includes the non-dominated successful
recommendations under `pareto_front`.
