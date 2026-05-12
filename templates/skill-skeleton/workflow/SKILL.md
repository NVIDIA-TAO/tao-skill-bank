---
name: REPLACE-WITH-WORKFLOW-NAME
description: >-
  Top-level workflow orchestrator for <use case>. Runs an end-to-end pipeline
  of <N> stages: <stage 1> → <stage 2> → … → <stage N>.
  Use when the user asks to "REPLACE-WITH-INTENT-1", "REPLACE-WITH-INTENT-2",
  or mentions REPLACE-WITH-DOMAIN-TERMS. Include literal trigger phrases.
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit. Sub-skills declare additional requirements.
metadata:
  author: REPLACE-WITH-AUTHOR-NAME
  version: "0.1"
allowed-tools: Read Bash
---

# Skill Name

High-level description of what this workflow accomplishes end-to-end.

## External dependencies

| Dependency | Purpose | Install |
|---|---|---|
| docker | Run sub-skill containers | https://docs.docker.com/engine/install/ |

Plus everything the sub-skills require — see "Related skills" below.

## Related skills

This workflow invokes:

- `tao-skill-bank:<skill-1>` — <one-line purpose>
- `tao-skill-bank:<skill-2>` — <one-line purpose>
- `tao-skill-bank:<platform>` — platform execution

## Quick start

### Default run

```
> Run the <workflow-name> workflow on <user-data-description>.
```

The orchestrator collects inputs, validates them, then dispatches the stages below.

### With custom KPI / parameters

```
> Run the <workflow-name> workflow with KPI <metric < target>, max_iterations=3.
```

### Resume after crash

```
> Resume the in-progress <workflow-name> run at <storage_root>.
```

State persists in `<storage_root>/<state_file>.json` — orchestrator reads on startup.

## Inputs

| Input | Required | Description |
|---|---|---|
| `<input-1>` | Yes | ... |
| `<input-2>` | No | ... |

## Stages

### Stage 1: <name>

- **Skill**: `tao-skill-bank:<skill>`
- **Input**: <from user or prior stage>
- **Output**: <what produced>
- **Condition**: when this stage runs (always / conditional)

### Stage 2: <name>

(similar structure)

## Storage layout

```
<storage_root>/
├── stage_1_output/
└── stage_2_output/
```

## Iteration control (if applicable)

- **Termination condition**: KPI / metric / turn count.
- **State management**: how iteration N's output flows into iteration N+1.
- **Checkpointing**: how to resume after a crash.

## Instructions

When the user asks to run this workflow, follow these steps in order:

### Step 1 — Gather inputs

Confirm with the user:
1. **`<input-1>`**: Path / value.
2. **`<input-2>`**: Path / value (optional, default `<value>`).

Verify inputs exist:
```bash
ls <path>
```

### Step 2 — Run the pipeline

Dispatch each stage in order. Read each sub-skill's SKILL.md for its specific invocation.

### Step 3 — Report results

After completion, report metrics + output path to the user.

## Known pitfalls

| Stage | Symptom | Cause | Fix |
|---|---|---|---|
| <stage> | ... | ... | ... |
