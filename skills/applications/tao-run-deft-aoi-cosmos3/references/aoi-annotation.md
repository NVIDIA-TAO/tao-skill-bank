# AOI annotation and mining-only Train assembly

All annotations are one JSON array of ShareGPT objects:

```json
[
  {
    "id": "benchmark-000001",
    "images": ["images/board/000001.png"],
    "conversations": [
      {"from": "human", "value": "Inspect the component. Return exactly OK or NG."},
      {"from": "gpt", "value": "NG"}
    ]
  }
]
```

There is exactly one image. The final assistant value is exactly `OK` or `NG`.
Evaluation records require unique filesystem-safe IDs. Resolve relative image
paths from the workspace root when paths begin with `images/`.

Validate source splits:

```bash
PYTHON=$(scripts/deft_python.sh)
for ROLE in proxy benchmark mining; do
  "$PYTHON" scripts/validate_sharegpt.py \
    --annotations "$WORKSPACE/annotations/${ROLE}_kpi.json" \
    --media-root "$WORKSPACE" --require-files
done
```

Use the actual Mining filename (`mining_pool.json`) where appropriate and pass
`--require-id` for Proxy and Benchmark.

After `filter_by_label` mining has selected OK and NG paths separately, align
each path back to its Mining record so the prompt and exact label are
preserved:

```bash
"$PYTHON" scripts/emit_mined_sharegpt.py \
  --mined-parquet "$RESULTS_DIR/$LABEL/data_mining/mined_filtered.parquet" \
  --source-json "$WORKSPACE/annotations/mining_pool.json" \
  --output "$RESULTS_DIR/$LABEL/assemble_data/mined_sharegpt.json"
```

Assemble iteration 1 solely from that aligned file:

```bash
"$PYTHON" scripts/assemble_training_json.py \
  --mined-json "$RESULTS_DIR/iter1/assemble_data/mined_sharegpt.json" \
  --validation-json "$WORKSPACE/annotations/proxy_kpi.json" \
  --validation-json "$WORKSPACE/annotations/benchmark_kpi.json" \
  --dedupe --output "$RESULTS_DIR/iter1/assemble_data/train_iter_1.json"
```

For iteration N greater than 1, add
`--previous-json .../train_iter_<N-1>.json`. The validator must prove exact
labels, existing unique targets, membership in Mining or the immediate prior
Train set, monotonic retention, no Proxy/Benchmark overlap, and an unchanged
Benchmark hash.
