# Cosmos3 AOI Real-Pair Mining

Read `skills/data/tao-mine-aoi-images/SKILL.md` before launch. Reuse its fixed
three-step flow: embed Proxy targets, embed the Mining source pool with the
same encoder, then run nearest-neighbor mining. Dispatch each GPU invocation
through the selected platform's four verbs and give each invocation its own
job-record.

## Inputs

- targets: Proxy false accepts/rejects only;
- source pool: recorded Mining annotations/media;
- model: the mining skill's configured SigLIP embedding model;
- top-K and metric: recorded DEFT config;
- output root: `${RESULTS_DIR}/iterN/mining`.

Never use Benchmark errors as targets. The candidate/source side contains only
the recorded Mining pool; Proxy errors are query targets, not source samples.

## Container user

The mining skill's setup notes tell you to drop `--user` because it raises a
`getpwuid()` `KeyError` during the `transformers` import. Do not drop it here.
That error is only the mapped uid having no entry inside the image, and this
workflow already carries the fix — pass the account databases along with the
mapping:

```bash
--user $(id -u):$(id -g) -e USER="$(id -un)" -e HOME=/tmp \
-v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro
```

Verified 2026-07-30 against the pinned data-services image: `pwd.getpwuid`
resolves and `transformers` imports cleanly. This matters because mining writes
into `${RESULTS_DIR}/iterN/mining`, a tree the operator owns; run as root and
the parquet files come back root-owned and cannot be cleaned up afterwards.

If a future image genuinely rejects the mapping, repair ownership through a
container rather than assuming sudo:

```bash
docker run --rm -v "$WORKSPACE:/ws" busybox:latest \
  chown -R "$(id -u):$(id -g)" /ws/<relative/path/to/mining>
```

## Cosine floor

The native nearest-neighbor output is not sufficient proof of the configured
floor. Preserve raw outputs, then run:

```bash
"$PYTHON" "$SKILL_ROOT/scripts/filter_mined_by_cosine.py" \
  --mined-parquet "$MINING_DIR/mined_raw.parquet" \
  --source-embeddings "$MINING_DIR/source_embeddings.parquet" \
  --target-embeddings "$MINING_DIR/target_embeddings.parquet" \
  --min-similarity "$MIN_SIMILARITY" \
  --output "$MINING_DIR/mined_filtered.parquet" \
  --summary "$MINING_DIR/cosine_filter_summary.json"
```

The output must differ from the raw parquet. A missing embedding, dimension
mismatch, zero-norm vector, non-finite value, missing path, or zero kept rows
is a hard stop.

## Handoff

Commit `data_mining` with the filtered parquet, summary, both embedding
parquets, and exact positive row count. The next stage uses
`emit_mined_sharegpt.py` to recover the Mining source prompt, golden image, and
bare label.
