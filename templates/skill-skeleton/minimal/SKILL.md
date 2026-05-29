---
name: REPLACE-WITH-SKILL-NAME
description: >-
  One-to-three-sentence description of what the skill does and when to use it.
  Use when the user asks to "REPLACE-WITH-INTENT-1", "REPLACE-WITH-INTENT-2",
  or mentions REPLACE-WITH-DOMAIN-TERMS. Include literal trigger phrases the
  user is likely to say.
license: Apache-2.0
# OPTIONAL — fill in when authoring (validator warns if missing):
# compatibility: Describe runtime requirements only — what the user must have installed (docker, CLI, Python packages, env vars). Do NOT mention agent harness; the skill bank is harness-agnostic.
# metadata:
#   author: Your Name
#   version: "0.1"
# allowed-tools: Read Bash    # Pre-approve tools the skill needs frequently to reduce user prompts.
---

# Skill Name

## Quick start

```bash
docker run --gpus all --rm \
  -e <REQUIRED_ENV_VARS> \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  <container_image> \
  <command>
```

See `tao-skill-bank:tao-run-on-docker` for `docker run` conventions (NGC auth, --gpus, mount patterns).

## Inputs

What the user must provide.

## Outputs

What the skill produces and where to find it.

## Credentials

Env vars the container reads at runtime.

- **`<VAR_NAME>`** — what it's for.

## Known pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| ... | ... | ... |
