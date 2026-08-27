# Air-gapped execution

Select this path only when the user or environment explicitly declares an air
gap. Before approval, verify locally without pulling:

- the Framework and data-services images resolved from `versions.yaml`, with
  repository digests;
- the complete HF-format VLM base under the workspace `models/` directory;
- all Proxy, Benchmark, Mining, and image files;
- the selected platform CLI and the dependency-complete skill Python.

After approval, set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` in every
model container, prohibit registry login/pull and package or asset fetches, and
use `scripts/deft_exec.py` for host commands. A missing image, model file, or
Python dependency is a hard stop; do not retry through network access.

DCP evaluation may materialize an action-local model inside its recorded
writable directory. That is local checkpoint handling, not a network fetch.
Keep the workspace mounted read-only except for explicit results and
action-model directories.
