---
name: REPLACE-WITH-SKILL-NAME
description: >-
  One-to-three-sentence description of what this data transformation does.
  Use when the user asks to "REPLACE-WITH-INTENT", or mentions
  REPLACE-WITH-DOMAIN-TERMS. Include literal trigger phrases the user is
  likely to say.
license: Apache-2.0
compatibility: REPLACE — typical examples — `Requires docker + nvidia-container-toolkit.` (containerized) or `Requires Python 3.8+ and Pillow.` (local script)
metadata:
  author: REPLACE-WITH-AUTHOR-NAME
  version: "0.1"
allowed-tools: Read Bash
---

# Skill Name

Two-line summary of the transformation. Inputs → outputs.

## External dependencies

| Dependency | Purpose | Install |
|---|---|---|
| docker | Run the container | https://docs.docker.com/engine/install/ |

(or for local scripts)

| Dependency | Purpose | Install |
|---|---|---|
| Python 3.8+ | Runtime | System / conda |
| Pillow | Image I/O | `pip install pillow` |

## Quick start

### Containerized

```bash
docker run --gpus all --rm \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  nvcr.io/nvidian/iva/<image>:<tag> \
  <command> --input /input --output /output
```

### Local Python script

```bash
python scripts/<script>.py --input-dir <path> --output <path>
```

### Multi-defect / multi-source loop (advanced)

```bash
for label in bridge tombstone shift; do
  python scripts/<script>.py --input-dir data/${label}/ --output ${label}.csv --label ${label}
done
```

See `tao-skill-bank:tao-run-on-docker` for docker conventions (when containerized).

## Inputs

| Field | Type | Description |
|---|---|---|
| `<input-dir>` | folder | What's in it, expected format |
| `<input-parquet>` | file | Schema, columns |

## Outputs

| Field | Type | Description |
|---|---|---|
| `<output-csv>` | file | Schema produced |

## CLI Reference

| Argument | Required | Default | Description |
|---|---|---|---|
| `--input-dir` | Yes | — | Input directory |
| `--output` | No | `output.csv` | Output path |
| `--label` | No | auto-detect | Force single label for all rows |

## How it works

Step-by-step description of the transformation:

1. Step 1
2. Step 2

Optional pipeline diagram:

```
Input A ─┐
Input B ─┼─→ [processing] ─→ Output
Input C ─┘
```

## Caveats

- Document non-obvious behaviors, edge cases, permission quirks.

## Known pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: ...` | Wrong column name in input parquet | Use `--column-name` to override |
