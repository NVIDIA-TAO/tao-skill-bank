# Workspace and results layout

The validation workspace is mounted at the same absolute path on host and in
Docker:

```text
workspace/
├── annotations/
│   ├── proxy_kpi.json
│   ├── benchmark_kpi.json
│   └── mining_pool.json
├── images/
├── models/
│   └── cosmos3-vlm/             # complete local HF-format VLM snapshot
└── specs/
    ├── train_spec.toml
    ├── evaluate_spec_proxy.toml
    └── evaluate_spec_benchmark.toml
```

The workspace root is the media root when annotation paths begin with
`images/`. Proxy, Benchmark, and Mining targets must be pairwise disjoint.
Benchmark is immutable after initialization. Proxy and Benchmark may be used
for evaluation only. Mining is the sole source category eligible for new
Train records; the preceding Train file is only a monotonic seed.

The run results use this shape:

```text
results/run-id/
├── deft_state.json
├── DEFT_Loop_Report.html
├── baseline/
│   ├── evaluate_benchmark/
│   ├── benchmark_metrics/
│   ├── evaluate_proxy/
│   └── proxy_rcca/
└── iterN/
    ├── routing/
    ├── data_mining/
    ├── assemble_data/
    ├── validate_data/
    ├── train/
    │   ├── train.toml
    │   ├── checkpoints/
    │   └── action_model/
    ├── evaluate_benchmark/
    ├── benchmark_metrics/
    ├── evaluate_proxy/
    └── proxy_rcca/
```

All recorded paths are absolute compute-frame paths. Only stage result and
action-model directories are writable in model containers.
