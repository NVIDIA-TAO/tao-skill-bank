---
name: multi-eval-smoke
description: CI smoke skill that exercises skill-eval's multi-case reporting path. Each eval case writes and reads back a small text file, so CI can confirm every case is reported separately. Use only for skill-eval CI validation.
license: Apache-2.0
compatibility: No external dependencies; runs anywhere bash is available.
metadata:
  version: "0.1.0"
  author: NVIDIA Corporation
allowed-tools: Read Bash Write
tags:
- ci
- smoke
- testing
---

# multi-eval-smoke

A trivial skill used only to validate skill-eval's **multi-case** CI reporting. It has no
model, no container, and no external dependency — each eval case just writes a small text
file and reads it back.

## Quick Start

Create a text file and confirm its contents:

```bash
echo 'case ok' > ./out.txt
cat ./out.txt
```

That is the entire skill. The real work lives in `eval.config`, which defines three
independent cases so CI can confirm all three are evaluated and reported as separate rows.
