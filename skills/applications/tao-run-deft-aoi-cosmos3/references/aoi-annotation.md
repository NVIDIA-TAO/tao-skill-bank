# Canonical NVPAW annotation contract

The three runtime inputs are newline-delimited JSON objects:

- `annotations/mining.jsonl`
- `annotations/benchmark.jsonl`
- `annotations/proxy_kpi.jsonl`

Each eligible row requires a unique, trimmed, non-empty `id` without control
characters, one of the six supported classification/detection `task_type`
values, and native `messages`. Non-reference tasks contain exactly one image;
reference tasks contain golden then target. Every image item contains `image`,
positive integer `min_pixels`, and integer `max_pixels >= min_pixels`. One
non-empty assistant answer is required.

The canonical Mining file also contains count/segmentation families that are
outside this application's approved six-task scope. Mining readers skip those
rows directly from the canonical JSONL and report their counts; they do not
materialize a transformed source file. Proxy, Benchmark, assembled Train, and
all other inputs reject unsupported tasks. The state still seals the complete
canonical Mining file by path and SHA-256.

Run `$PYTHON scripts/check_annotations.py --workspace WORKSPACE --require-files`.
Training assembly writes JSONL directly. `scripts/assemble_training_json.py`
accepts one current `--mined-jsonl`, an optional preceding
`--previous-jsonl`, and both evaluation inputs as repeated
`--validation-jsonl`. It rejects evaluation leakage, retains prior rows, and
requires a current real Mining contribution. `validate_split_contract.py`
proves the same lineage independently and verifies the frozen Benchmark hash.
Its Mining record count is the eligible six-task count; preflight also reports
the raw count and ignored task-family counts.

The application never converts the runtime input into a JSON array.
