# Cosmos3 DEFT AOI

Run a resumable five-iteration AOI improvement loop for Cosmos Reason 3 using
exact `OK`/`NG` labels and labeled real-image mining.

The application evaluates a frozen Benchmark first, uses Proxy errors only for
RCCA and routing, mines OK and NG targets separately with `filter_by_label`,
assembles monotonic Train JSON arrays, trains native Framework VLM LoRA, and
evaluates every native DCP checkpoint. It keeps the four platform verbs and a
job-record for every submitted GPU action.

Inputs:

- Proxy, frozen Benchmark, and Mining ShareGPT JSON arrays;
- exactly one image and an exact `OK` or `NG` assistant label per record;
- a complete local HF-format VLM snapshot under `workspace/models`;
- selected platform and compute profile.

Validation profile: Docker, one NVIDIA H200, 5 iterations, 10 epochs each,
top-K 15 for OK and NG separately, Benchmark every iteration, and target
accuracy at least 99% with zero unknown predictions.

Native container commands:

- `cosmos-framework-train --sft-toml=...`
- `cosmos-framework-evaluate --config ...`
- `cosmos-framework-inference --model_path ...`

The local HF base is used directly. Each trained DCP is evaluated directly and
is set as the next iteration's `checkpoint.load_path`; there is no explicit
weight conversion or export stage.
