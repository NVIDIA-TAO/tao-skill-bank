<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Rendering a spec-bundle: the platform-owned contract

A spec-bundle says **what** one action needs — image, command, inputs by URI,
outputs, compute shape. It is deliberately platform-agnostic. Turning it into a
native launch is the **how**, and the how belongs to the platform skill, next to
the SKILL.md prose that documents it.

This is not a new idea in the bank: `tao-run-on-virtualenv` already ships
`references/virtualenv_runner.py`, "four-verb process lifecycle for
virtualenv-native TAO jobs". Rendering is the same shape of platform-owned code.

**Why not a table in the consumer.** A workflow that carried a
`render_docker`/`render_slurm`/… table would have to be edited before any new
platform could run — which is exactly the registry the four-verb contract avoids.
`--platform` is an open validated slug (`tao_job_record.py`), and an external
platform skill joins by *documenting* the verbs. Rendering follows the same rule:
**a new platform conforms by shipping the module below; no consumer changes.**

## The module

Each execution platform ships `references/render.py` exposing:

```python
PLATFORM: str                       # matches the --platform slug

def render(bundle: dict, ctx: dict) -> dict: ...
def status(backend_ref: str, ctx: dict) -> tuple[str, int]: ...
```

### `render(bundle, ctx) -> dict`

`ctx` carries what the bundle deliberately does not: `job_id` (already minted —
record-then-launch), `results_dir`, and platform settings (`login_host`,
`namespace`, `partition`, …).

Returns:

| key | meaning |
|---|---|
| `files` | `{absolute_path: content}` written before launch (sbatch script, K8s manifest). Empty for platforms that need none. |
| `argv` | the native submit command |
| `backend_ref` | the handle, when it is known without running anything (K8s `namespace/job`). `None` means take it from `argv`'s stdout (docker container id, `sbatch --parsable` job id). |

Rules every implementation follows:

- **Name the backend object after `ctx["job_id"]`.** The later verbs locate the
  job by that name; an object named anything else is unreachable.
- **Refuse a `declared_input` URI you cannot mount.** Staging is `tao-data-io`'s
  job. Silently dropping one yields a container that starts, finds no data, and
  fails deep inside the workload instead of at submit.
- **`compute_shape.gpus == 0` is a first-class case** — CPU-only glue and
  data-preparation stages are ordinary bundles and must render without a GPU
  request.
- **Never put a credential value in `argv` or in `files`.** Names only; the
  platform's documented credential path (sidecar, Secret, `-e VAR`) carries the
  value.

### `status(backend_ref, ctx) -> (state, exit_code)`

`state` is the fixed vocabulary — `PENDING RUNNING COMPLETE ERROR CANCELED
UNKNOWN`. The native sub-state belongs in the record's transition `message`,
never in the returned state.

## Conformance

`scripts/tests/test_platform_render_contract.py` asserts every execution
platform ships the module, exports `PLATFORM` matching its directory, renders
a GPU and a glue bundle, names the object after `job_id`, refuses unstaged
URIs, and maps every native state into the fixed vocabulary.
