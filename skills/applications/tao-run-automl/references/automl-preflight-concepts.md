# AutoML Preflight And Support

## Required evidence

Before initialization, prove:

1. The chosen platform skill passes preflight and its four verbs work.
2. The model has `automl_enabled: true` and packages a valid schema for the
   selected action.
3. The base spec is complete, nested, and uses paths visible to the platform.
4. The model image resolves and can be pulled or is already cached.
5. The objective metric has an explicit name, direction, and extraction path.
6. Training actions declare a checkpoint artifact; other actions declare a
   primary result artifact.
7. Dataset counts and effective-batch constraints are safe for every initial
   recommendation.

Verify the engine helpers:

```bash
python -c "import yaml, jsonschema, numpy, scipy, sklearn; print('OK')"
```

GEPA and WandB are optional and should be installed only when selected. LLM
search uses the Python standard library HTTP client and requires an
OpenAI-compatible endpoint/model plus a credential in the session environment
when the provider requires one.

## Model/action support

The source of truth is the model skill's action schema. Do not derive a search
space from historical workspaces or source checkouts. Non-core models may also
need `references/spec_template_<action>.yaml` for complete defaults. Reject an
action with no packaged schema instead of inventing ranges.

## Baseline and launch gate

After platform/data/image preflight, run the model's evaluate action once on the
selected validation set. Present its record id, metric, and artifact path in the
launch review alongside algorithm, budget, concurrency, full initial configs,
GPU shape, image, data counts, and runtime estimate. Submit tuning jobs only
after user confirmation. If evaluation is unavailable, continue on a proxy only
with explicit user acceptance.

## Resume

Initialize a fresh state for a plain “run AutoML” request. Reuse an existing
state only when the user explicitly asks to resume/recover/continue. Validate
the stored schema and reconcile each RUNNING recommendation against its bound
platform job before generating more work.
