# Workspace and results layout

```text
workspace/
├── annotations/
│   ├── mining.jsonl
│   ├── benchmark.jsonl
│   └── proxy_kpi.jsonl
├── eval/calculate_f1_metrics.py
├── models/Cosmos3-Nano-VLM/
├── specs/
│   ├── train_spec.toml
│   ├── evaluate_spec_proxy.toml
│   └── evaluate_spec_benchmark.toml
└── results/<run>/
    ├── deft_state.json
    ├── mining_history.json
    ├── DEFT_Loop_Report.html
    ├── baseline/
    └── iterN/
        ├── evaluate_benchmark/
        ├── benchmark_metrics/
        ├── evaluate_proxy/
        ├── proxy_rcca/
        ├── routing/
        ├── mining/
        ├── assemble/
        ├── validate/
        └── train/checkpoints/iter_#########/
```

Proxy and Benchmark are read-only evaluation inputs. Mining is the sole source
of newly selected training rows. Generated Train targets must be a subset of
Mining plus the previous committed Train, and must remain disjoint from both
evaluation splits. Only the six classification/detection task families are
eligible; Mining-only count/segmentation rows in the canonical source are
ignored and counted, while unsupported rows in every other role fail closed.
