# AutoML Engine Configuration

The executable interface is `scripts/automl_step.py`; compute always goes
through a platform skill. `init --help` is the authoritative CLI surface.

## Common fields

| Field | Purpose |
|---|---|
| `--schema`, `--base-spec`, `--search-space` | Packaged action contract, complete defaults, and optional bounded overrides |
| `--algorithm` | One of the eleven engine algorithms |
| `--metric`, `--direction`, `--objective` | Primary and optional weighted objectives |
| `--max-recommendations`, `--max-concurrent` | Trial and active-job limits |
| `--candidate-count`, `--seed` | Deterministic proposal pool |
| `--max-epochs`, `--reduction-factor`, `--epoch-multiplier` | Multi-fidelity budget |
| `--population-size`, `--max-generations`, `--eval-interval` | PBT budget |
| `--llm-endpoint`, `--llm-model` | OpenAI-compatible language-model routing |

Every emitted `spec` is nested. Dotted paths exist only in search-space and
audit maps.

## WandB

Enable with `--wandb --wandb-mode online|offline --wandb-project NAME`.
Online mode reads `WANDB_API_KEY` from the session environment. Each terminal
recommendation logs raw objectives, scalarized score, parameters, resource
budget, rung/generation, and experiment grouping. Tracking failure is recorded
in recommendation metadata without corrupting optimizer state.

## Multi-objective

Repeat `--objective NAME:DIRECTION:WEIGHT:SCALE`. A complete metric record must
contain every objective. Raw values and the scalarized score are stored
separately, replay validation rejects changes to either, and `status` returns
the successful non-dominated recommendations in `pareto_front`.

## Non-training actions

Set `--action` to the packaged action name and use its schema/template. A
successful result needs an artifact URI but not a checkpoint. Budgeted methods
are appropriate only when the action exposes a meaningful resource/fidelity
field; otherwise use Bayesian, BFBO, LLM, hybrid, or autoresearch.

## Failure semantics

- `ERR_INFRA`: return the same recommendation READY; submit a retry job linked
  to the prior record.
- `ERR_PROGRAM`: mark FAILED and spend the trial budget.
- `CANCELED`: terminal cancellation, including a platform-honored early-stop
  decision.
- Repeating an identical terminal report is safe; conflicting facts fail.
