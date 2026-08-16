# Cosmos evaluation

Load this only when `SKILL.md` points here. The repository evaluator is shared
by Cosmos-RL checkpoints and Framework checkpoints after the mandatory native
Framework export.

## Resolution contract

Never submit `references/spec_template_evaluate.yaml` directly. It is a
dataset-neutral shape template whose zero and empty semantic values are
deliberately unresolved. Run `scripts/evaluation_workflow.py` first. The
helper verifies a sealed fine-tuning plan, records field-level provenance,
returns all missing user inputs in one bounded list, and emits runtime TOML
only when the request is complete.

Resolve fields in this order:

1. Use an explicit current evaluation override when the user is deliberately
   changing the validation corpus or evaluation semantics.
2. Otherwise inherit exact values from the selected fine-tuning plan.
3. Run deterministic checkpoint pre-actions owned by the backend.
4. Ask the user only for fields that remain absent or ambiguous.

Do not use a template value, nearby directory, historical run, checkpoint
mtime, or filename convention as a fifth source.

### Inherit from fine-tuning

For the original validation split, inherit these without asking again:

- annotation manifest, media root, and dataset fingerprint;
- system prompt, including an explicitly empty prompt;
- frame count, pixel budget, precision, seed, and validation batch size;
- training sequence limit and evaluator tensor-parallel degree (one model replica per rank);
- task/answer/metric semantics when validation inspection proved them;
- backend, training mode, base-model identity/fingerprint, and GPU count;
- dense versus PEFT behavior and the prepared base model required to merge a
  Cosmos-RL adapter.

The fine-tuning planner stores these in `evaluation_contract`. Dataset
inspection stores `evaluation_profile`, including task semantics, declared
metrics, answer type, normalization version, and any fields that could not be
inferred safely.

### Ask only when unresolved

Prompt once for the remaining fields reported in
`required_user_inputs`. Common cases are:

- the new user-owned evaluation results directory;
- exact checkpoint or checkpoint epoch when more than one checkpoint event is
  present and the training plan did not record a selection;
- generation maximum tokens, because it is not a fine-tuning parameter;
- maximum video pixels when a non-Nano training plan did not record a usable
  budget. For Nano, a sealed native default remains an omitted evaluator
  override so preprocessing does not silently change;
- task type or metric names when annotation targets/metadata were ambiguous;
- exact annotation and media paths when evaluating a different corpus.

An empty system prompt is valid. The flow must distinguish “recorded empty” or
“user supplied empty” from “missing”. For a different evaluation corpus,
fingerprint and inspect that exact manifest/media pair on the selected compute
frame before launch; do not inherit the old dataset's prompt or scoring
semantics automatically.

### Resolver usage

First run with the selected training artifacts and whatever evaluation inputs
are already known:

```bash
python scripts/evaluation_workflow.py \
  --training-plan <sealed-training-plan.json> \
  --training-status <structured-training-status.json-or-jsonl> \
  --results-dir <new-evaluation-results-dir> \
  --plan-output <evaluation-plan.json> \
  --config-output <evaluation.toml>
```

Exit code `3` means the JSON plan was written but contains unresolved intake or
an automated pre-action. Ask only for entries in `required_user_inputs`; do
not ask for entries in `automated_actions`. Multiple recorded validation
manifests/media roots are an automated deterministic materialization step, not
a reason to ask the user to select a subset; preserve the sealed fingerprint
and full record coverage. Rerun with user inputs such as
`--checkpoint-epoch`, `--checkpoint`, `--generation-max-tokens`,
`--max-video-pixels`,
`--task-type`, `--answer-type`, `--metric`, `--validation-annotation`, or
`--validation-media-root`.

The plan and TOML contain SHA256 values. Persist both in the evaluation job
record. Validate all inherited fingerprints and paths from the target compute
frame before submit.

## Exact checkpoint selection

Use a checkpoint recorded by the selected training job. A single structured
checkpoint event is unambiguous. With multiple events, require an exact path,
an exact epoch, or a checkpoint selection already sealed in the training plan.
Never choose “latest” by directory order or mtime. Require terminal successful
training status before consuming a checkpoint.

For Cosmos-RL dense training, the selected checkpoint becomes
`model.model_name`. For Cosmos-RL PEFT, the selected adapter becomes
`model.model_name`, `model.enable_lora=true`, and
`model.base_model_path` is inherited from the fine-tuning model-preparation
record. Do not ask the user to repeat that base-model path.

## Framework DCP pre-action

When `backend=cosmos-framework`, never pass native DCP directly to the shared
evaluator and never ask the user to export it. `evaluation_workflow.py` emits a
`framework_checkpoint_pre_action` entry. Run
`scripts/framework_checkpoint_action.py plan`, then `prepare` in the clean
repository-derived Framework action image. The helper validates the saved
Framework config, DCP metadata, base-model identity/revision, exact exported
keys, indexed weights, and export manifest. Run `verify` from the target
compute frame and pass its `action_model_path` back to
`evaluation_workflow.py --action-model-path`.

Framework PEFT is reconstructed and merged by the native exporter, so the
shared evaluation config keeps `model.enable_lora=false`. Export failure or
provenance mismatch blocks evaluation; it never selects another checkpoint or
export.

## Task and metric semantics

The generic evaluator supports structurally detected conversation and
task-aware records. Validation inspection classifies complete `yes`/`no`
targets as binary and complete `A`-through-`D` targets as multiple choice.
Explicit task metadata such as `bcq`, `binary`, `binary_choice`, `mcq`, and
`multiple_choice` is canonicalized to the repository evaluator semantics.
Ambiguous `A`/`B` targets require the user to choose binary or multiple choice.

Accuracy-defined tasks use the shared task-aware scorer and `metrics.names=[]`.
Generative tasks use only metrics declared by the annotation metadata or
explicitly selected by the user. Do not turn text metrics on for a
classification task, and do not relabel generation NLL or validation loss as
answer accuracy.

Preserve the resolved prompt, frame sampling, pixel budget, generation,
parsing, normalization, coverage, and evaluator version in metadata. Aggregate
accuracy as correct/covered examples over tasks that define accuracy; report
excluded tasks and reasons.

## Decoder and execution

The packaged shared evaluator currently requires its strict
`pynvvideocodec` path. The template records that repository-owned runtime
contract, not a dataset choice. If fine-tuning recorded a validated video
override artifact, inherit its exact path/fingerprint. Otherwise validate GPU
random-access decoding for every evaluation media encoding on the allocated
GPU before launch. Never invent FPS metadata, rewrite annotations, or silently
fall back to CPU decoding.

Use `torchrun` data parallelism according to the resolved GPU count. Keep one
model replica per rank unless the selected backend contract explicitly
requires another topology. Full evaluation uses `limit=-1`, exact record
coverage, rank-aware result files, and global deduplication before scoring.

## Completion and results

Treat scheduler completion as provisional. Require child exit zero, terminal
TAO `SUCCESS`, a complete prediction set with no duplicate IDs, and evaluator
metrics whose numerator/denominator recompute to the reported accuracy.
Persist the selected checkpoint, Framework export when applicable, resolved
config and SHA256, evaluation plan and provenance, stdout/stderr, status,
results, per-task coverage, normalization/evaluator version, and duration in
the job record.
