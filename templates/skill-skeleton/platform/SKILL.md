---
name: REPLACE-WITH-PLATFORM-NAME
description: >-
  Where and how GPU jobs run on this platform. One-to-three-sentence summary.
  Use when the user asks to "deploy on REPLACE-PLATFORM", "run on
  REPLACE-PLATFORM", or mentions the platform's distinctive concepts (e.g.,
  resource shape, instance, node group).
license: Apache-2.0
compatibility: REPLACE — examples — `Requires the brev CLI and an active brev login.` or `Requires kubectl configured against a GPU cluster and <PLATFORM>_API_TOKEN.`
metadata:
  author: REPLACE-WITH-AUTHOR-NAME
  version: "0.1.0"
allowed-tools: Read Bash
---

# Platform Name

> **Standalone install?** If this session was not initialized by the TAO skill bank plugin, run the `tao-setup` skill first (host preflight, credentials, cross-skill discovery).

Short summary of the platform's execution model. Instance-based or job-based? API-first or docker-first? Single-node or multi-node? Managed or self-hosted?

For generic docker conventions, defer to `tao-skill-bank:tao-run-on-docker`. This skill covers the platform-specific layering on top.

## External dependencies

| Dependency | Purpose | Install |
|---|---|---|
| `<cli-name>` | Submit jobs / manage resources | `<install-command>` |

## Authentication

| Method | When to use |
|---|---|
| API token (recommended) | Scripted / automated workflows |
| Browser login | Interactive development |

Env vars:

| Env var | Required | Purpose |
|---|---|---|
| `<PLATFORM>_API_TOKEN` | Yes (or manual login) | API auth |

## Preflight

```bash
# 1. CLI installed
which <cli-name> || echo "MISSING: install from <url>"

# 2. Logged in
<cli-name> ls >/dev/null 2>&1 || echo "NOT LOGGED IN"

# 3. Platform-specific checks
```

## Quick start

### Docker-native workflow

```bash
<platform-cli> exec <target> -- docker run --gpus all --rm \
  -e <env-vars> \
  -v <host-path>:<container-path> \
  nvcr.io/... \
  <command>
```

### Execution — the four verbs

A platform skill is a **consumer**: it runs a model/data skill's spec-bundle by
implementing `submit`/`status`/`logs`/`cancel` over the native CLI, mutating only
the job-record. No SDK. See `tao-skill-bank:tao-launch-workflow` for the shared
contract and `tao-skill-bank:tao-run-on-docker` for a worked example.

- **submit** — stage inputs via `tao-data-io`, lint the command with
  `redact_secrets.py`, then `tao_job_record.py open` (mints the id + binds
  `results_dir` *before* launch), launch naming the backend object after the id,
  and `mark ... --state RUNNING`.
- **status / logs** — poll the native backend; map states to the fixed vocab
  `PENDING RUNNING COMPLETE ERROR CANCELED UNKNOWN`.
- **cancel** — native cancel + teardown, then `mark ... --state CANCELED`.

## Platform-specific notes

- Storage model (shared NFS/Lustre? S3 only? instance-local?)
- Pricing / lifecycle considerations
- Known limitations

## Known pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `<error>` | ... | ... |
