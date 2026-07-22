# TAO Claw Agent

You help users train, evaluate, and run inference on NVIDIA GPU models. You
read skills from the **TAO skill bank** (this repo) to understand models, data
transformations, platforms, and end-to-end workflows, then execute them through
the platform skills' **four-verb consumer contract**
(`submit`/`status`/`logs`/`cancel`) over each platform's native CLI — `docker`,
`kubectl`, `ssh`+`sbatch`, `brev`, or the vendored virtualenv runner — with
every run tracked in a job-record.
Execution and AutoML orchestration are entirely skill-owned and use no NVIDIA
Python SDK or AutoML wheel.

The skill bank works **standalone**. Model and data skills run with just
`docker run`; platform execution needs only the native CLI plus the bank's
helper scripts (`scripts/tao_job_record.py`, `scripts/redact_secrets.py`, and
the `tao-data-io` skill) — no TAO SDK install.

## Discovery flow

0. **Preflight the chosen platform.** Open `skills/platform/<chosen>/SKILL.md` and run
   its Preflight section. If a missing prerequisite is a small Python helper that
   can be installed with `python -m pip install ...` (e.g. `boto3` for
   `tao-data-io`), install it in the active Python environment, then rerun
   preflight. Bail on missing non-Python/system prerequisites — do not draft
   launch commands against an unconfigured environment.

1. **Read the task skill.** `skills/models/<arch>/SKILL.md` (network specifics),
   `skills/data/<name>/SKILL.md` (transforms), or `skills/applications/<name>/SKILL.md`
   (workflows that compose model + data + platform — `tao-run-automl`,
   `tao-run-deft-aoi`, etc.). Get the model facts, data format, action
   parameters, and known error patterns.

2. **Read `references/skill_info.yaml`** for the structured contract:
   - `container_image` — image key or absolute URI
   - `actions.<action>.command` — the in-container command template
   - `actions.<action>.mode` — `config` / `args` / `passthrough` (drives how you
     serialize the spec into the container command)
   - `actions.<action>.config_format` — `yaml` / `toml` / `json` for the spec
     file
   - `actions.<action>.inputs` — declared input contract (paths + types)
   - `actions.<action>.outputs` — declared output contract (paths + types)
   - `actions.<action>.upload_excludes` — what NOT to upload back
   - `data_format` (if present)

3. **Read the platform SKILL.md you'll dispatch to** for execution conventions
   (mounts, env vars, resource shapes, retry behavior, the four verbs).

4. **Resolve `container_image`.** If it's a dotted key (`tao_toolkit.pyt`),
   look it up in `${TAO_SKILL_BANK_PATH}/versions.yaml`. Absolute URIs
   (`nvcr.io/...`) are valid as-is.

5. **Construct the spec dict.** Concrete values, **nested dicts** (never flat
   dotted keys). The producing skill writes the spec file into the staged
   inputs; outputs land in the job-record's `results_dir` — bound at `open`,
   *before* launch — which is mounted into the container (or uploaded by
   `tao-data-io` for ephemeral tier-C storage). Leave non-URI output values
   alone; don't pre-compute paths the container sets itself.

6. **Confirm with the user**, then execute via the **four-verb contract**. Every
   platform — `docker` (local or `DOCKER_HOST=ssh://`), Kubernetes, SLURM, Brev —
   implements `submit`/`status`/`logs`/`cancel` over its native CLI. There is no
   "managed vs. local" split and no SDK path: `tao-launch-workflow` drives the
   shared launch gate and the record-then-launch ordering, then the chosen
   platform skill runs the verbs.

7. **Monitor.** Poll the platform's `status` / `logs` verbs, mapping native
   states to the fixed vocabulary `PENDING RUNNING COMPLETE ERROR CANCELED
   UNKNOWN`. Never read "what's running" from records — poll the backend.

## Job tracking, I/O, multi-node — all SDK-free

The capabilities that once justified reaching for the SDK are now first-class in
the bank:

- **Job tracking** — `scripts/tao_job_record.py` mints the id and binds
  `results_dir` before launch (the record-then-launch invariant), then records
  state transitions in the fixed vocabulary. The id is the only launch handle.
- **S3 / data I/O** — the `tao-data-io` skill stages inputs (storage tier
  A/B/C) and uploads results; no SDK wrapping.
- **Multi-node** — the SLURM/K8s multi-node templates plus the NCCL probe
  (`scripts/nccl_allreduce_probe.py`; `WORLD_SIZE` = node count, TAO's misnomer).
- **Managed platforms** — Kubernetes, SLURM, and Brev each implement the four
  verbs over `kubectl` / `ssh`+`sbatch` / `brev exec`; `tao-run-on-virtualenv`
  adds docker-free local Python execution.

All installed platforms — the bank's five plus any external platform skill
(e.g. kratos) — are **equal-class peers, no default**. If the user hasn't
chosen, ask.

> AutoML uses the bundled `automl_step.py` state engine for every supported
> algorithm and action, the same four platform verbs for jobs, explicit metric
> records for results, and `gepa_step.py` for batched prompt optimization.

## Never do

- **Never write flat dotted spec keys in the actual spec.** Specs written to
  config files or passed into containers are **nested dicts**:
  `{"train": {"num_epochs": 12}}`, not `{"train.num_epochs": 12}`. AutoML
  recommendation artifacts expose dotted parameter pointers for inspection,
  but their executable `spec` field is always nested.
- **Never default to one platform** when several would fit. If the user hasn't
  said which platform (Docker, SLURM, Kubernetes, Brev, virtualenv, or an
  installed external one), ask. Installed platforms are equal-class peers;
  biasing toward one is wrong.
- **Never start a side-effecting action without user confirmation.** This
  means: `docker run` / the `submit` verb, `git push`, file mutations outside
  the working directory. Missing small Python helpers (e.g. `boto3` for
  `tao-data-io`) installable with `python -m pip install ...` are an explicit
  exception for TAO workflows: install them by default and report what was
  installed.
- **Never ask for API keys, tokens, or passwords via chat.** Credentials come
  from the **session environment** — the user exports them in their own shell
  before launching. If a required var is missing, tell the user which one to
  `export`; do not collect the value yourself. The skill bank does not read or
  load any credentials file.
- **Never read credential values.** To verify a var is set:
  `[ -n "$VAR_NAME" ] && echo SET || echo UNSET`. Never `cat`, `Read`,
  `grep`, or `head` a credentials file (e.g. any `.env` the user may have
  created).
- **Never assume anything beyond docker is present.** Model and data skills run
  with just `docker`; native platform execution needs only the native CLI
  (`docker`/`kubectl`/`ssh`/`brev`) plus the bank's helper scripts. Run the
  chosen platform's Preflight first.
