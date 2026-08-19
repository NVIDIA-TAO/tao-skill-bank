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

def prepare(bundle: dict, ctx: dict) -> dict: ...
def render(bundle: dict, ctx: dict) -> dict: ...
def status(backend_ref: str, ctx: dict) -> tuple[str, int]: ...
```

### `prepare(bundle, ctx) -> {"image": str, "notes": [str]}`

Put the image into the platform's native form, **off** the metered resource,
and return the reference `render` should use. An implicit fetch inside the run
is billed as idle GPU time and hides an auth failure inside a training log.

**Idempotent by design — the common path does no work:**

| platform | behaviour |
|---|---|
| docker / brev | `image inspect` → pull only when missing (remotely, for brev) |
| slurm | reuse the cached Lustre `.sqsh`; convert only when missing or corrupt |
| kubernetes | none possible — the kubelet pulls on the node; optionally verify the reference resolves |
| virtualenv | no image; verify the interpreter exists |

Existence is not enough where a fetch can be interrupted. A conversion killed by
a wall-time cap leaves a **truncated** file that `test -e` accepts, so SLURM
reads the 4-byte `hsqs` squashfs magic and reconverts when it does not match. A
conversion that still fails verification is fatal: never fall back to the
registry reference, because that puts the pull back inside the GPU allocation
precisely when something is already wrong.

In air-gap mode `prepare` must refuse rather than fetch.

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

## Scheduling identity

A platform whose scheduler requires per-allocation identity (account, QOS,
reservation) must receive it from `ctx`, never from a default baked into the
renderer — those values are per cluster and per user.

The trap is that a renderer may issue **more than one allocation**. SLURM's
`prepare()` converts the image in its own job before the workload's job runs,
so identity supplied only to the workload leaves the conversion to be rejected
by the scheduler. That surfaces as "image conversion failed", which sends you
looking at enroot rather than at the missing setting.

So: derive every allocation's identity from the same `ctx` keys, and when a
platform-level command fails with none supplied, say so in the error rather
than reporting only the tool's own message.

## Where rendered files go

`render()` returns `files` as a path→content map. **Whether those paths are
local depends on where the argv runs**, and only the renderer knows:

| platform | argv runs | files read by | placement |
|---|---|---|---|
| docker, virtualenv | locally | — (no files) | n/a |
| kubernetes | locally (`kubectl`) | local kubectl | local write |
| slurm | remotely (`ssh … sbatch`) | the cluster | **remote write** |

A renderer whose launcher crosses a machine boundary must define
`place_files(files, ctx)` and place them itself. A generic caller writing them
locally produces a stray tree on the launch host and a launcher that cannot
find its input — or worse, where a same-named writable parent exists, a silent
submission of whatever file is already on the far side.

Two constraints on a remote implementation:

- **Content on stdin, never argv.** A rendered job script can carry credential
  material, and argv is visible to any user running `ps`.
- **Create it non-world-readable** (`umask 077`), matching the credential
  sidecar handling in the job template.

This is the same rule as scheduling identity and image references: what the
platform needs is derived by the platform, not assumed by the caller.

## How a workload finds its paths

A bundle is authored before the job exists, and the paths the workload sees are
chosen by the platform — identity mounts on SLURM, `-v` targets on docker, a
PVC subPath on kubernetes. A command that names a path directly is guessing at
a layout it cannot see.

Two environment variables close that, in both directions:

| variable | meaning |
|---|---|
| `TAO_RESULTS_ROOT` | where to write; the bundle never names its output path |
| `TAO_INPUT_<SPEC_KEY>` | where each `declared_inputs` entry landed |

`<SPEC_KEY>` is the declared `spec_key` upper-cased with non-alphanumerics
collapsed to `_`, so `train.pair-list` becomes `TAO_INPUT_TRAIN_PAIR_LIST`.

**Why this matters more than it looks.** A wrong input path does not fail
loudly. The directory is simply absent, and a command that does not check reads
nothing, writes empty output, and exits 0 — so the job records COMPLETE having
processed no data. That is worse than an error, because every downstream stage
treats it as real. An end-to-end run in this repo went green exactly that way
before these variables existed.

Bundle commands should fail closed on a missing input rather than trusting the
mount:

```sh
: "${TAO_INPUT_INPUT_DIR:?input env not exported by the platform}"
n=$(ls "$TAO_INPUT_INPUT_DIR" | wc -l)
[ "$n" -gt 0 ] || { echo "read 0 inputs" >&2; exit 3; }
```

Currently exported by the docker and SLURM renderers. Kubernetes and virtualenv
set the results path by their own mechanisms and do not yet export
`TAO_INPUT_*`; a bundle relying on it is not yet portable to them.

