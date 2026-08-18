# Host Prerequisites

Read this reference before workflow preflight and whenever `deft_python.sh`
cannot select an interpreter.

The host-side helpers require Python 3.11 with `pandas`, `pyarrow`, `PyYAML`,
and `huggingface_hub`. Resolve `DEFT_SKILL_ROOT` to the installed
`tao-run-deft-cr-its-mining` skill directory, then export the workspace and
select one dependency-complete interpreter:

```bash
export WORKSPACE_DIR="$WORKSPACE"
DEFT_PYTHON="$("$DEFT_SKILL_ROOT/scripts/deft_python.sh")"
```

The selector prefers `<deft_workspace>/.venv`, checks every required import,
and never installs packages. If it fails, show its exact commands and obtain
user approval before provisioning the workspace-local environment:

```bash
python3 -m venv "<deft_workspace>/.venv"
"<deft_workspace>/.venv/bin/python" -m pip install pandas pyarrow pyyaml huggingface_hub
```

Rerun the selector after installation. Use `DEFT_PYTHON` for every bundled
Python helper. Do not install these packages into an arbitrary system Python.
The selected platform must also provide its native CLI, GPU access, status and
log operations, and writable shared storage.
