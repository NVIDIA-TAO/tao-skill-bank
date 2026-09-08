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

For embedding, retrieval, de-duplication, history, leakage exclusion, and
materialization, a non-reference row is one `single_image` atomic sample and a
reference row is one ordered `(golden, target)` `reference_pair`. Reference
pairs are represented to the image encoder by a deterministic side-by-side
`nvpaw_reference_pair_embedding_v1` asset and one `atomic_sample_id`; neither
side is ever inserted into a retrieval pool on its own. Canonical training
JSONL always retains the original two ordered image items.

Source, Proxy, and Benchmark annotations require unique IDs. A materialized
Train JSONL may contain byte-equivalent repetitions of an accepted source row;
training validation allows the repeated ID only when the complete row content
is identical and still rejects one ID associated with different content.

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
`--validation-jsonl`; `--media-root` canonicalizes every atomic identity. A
reviewed materialization cap uses `--max-rows` and
`--row-multiple`; current Mining rows receive first claim on capped slots and
the output is truncated to an exact effective-global-batch multiple for native
epoch scheduling. It rejects atomic-sample evaluation leakage and requires a
current real Mining contribution. When the launch-recorded repetition blend is enabled,
the assembler consumes the current `gaps_summary.json`, retains every prior
row and at least one current row, and writes `repetition_blend_manifest.json`.
`validate_split_contract.py`
proves the same lineage independently and verifies the frozen Benchmark hash.
Its Mining record count is the eligible six-task count; preflight also reports
the raw count and ignored task-family counts.

The application never converts the runtime input into a JSON array.
