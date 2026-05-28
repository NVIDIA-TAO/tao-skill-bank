---
name: REPLACE-WITH-PLATFORM-NAME
description: >-
  Where and how GPU jobs run on this platform. One-to-three-sentence summary.
  Use when the user asks to "deploy on REPLACE-PLATFORM", "run on
  REPLACE-PLATFORM", or mentions the platform's distinctive concepts (e.g.,
  resource shape, instance, node group).
license: Apache-2.0
compatibility: REPLACE — examples — `Requires the brev CLI and an active brev login.` or `Requires the tao-sdk Python package (pip install 'tao-sdk[<platform>]') plus <PLATFORM>_API_TOKEN.`
metadata:
  author: REPLACE-WITH-AUTHOR-NAME
  version: "0.1"
allowed-tools: Read Bash
---

# Platform Name

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

### SDK workflow (optional, for tracking)

```python
from tao_sdk.<platform>_sdk import <Platform>SDK
sdk = <Platform>SDK()
job = sdk.create_job(...)
```

See `tao-skill-bank:tao-run-platform` for SDK semantics.

## Platform-specific notes

- Storage model (shared NFS/Lustre? S3 only? instance-local?)
- Pricing / lifecycle considerations
- Known limitations
- Mixed-platform patterns

## Known pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `<error>` | ... | ... |
