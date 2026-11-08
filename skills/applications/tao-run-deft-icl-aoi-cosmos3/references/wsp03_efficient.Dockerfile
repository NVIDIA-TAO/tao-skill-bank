# Two separate interpreters/dependency trees. The original HF environment is
# inherited byte-for-byte and remains the default native-editor interpreter.
FROM vllm/vllm-openai@sha256:6766ce0c459e24b76f3e9ba14ffc0442131ef4248c904efdcbf0d89e38be01fe AS fast
FROM nvcr.io/nvidia/tao/tao-toolkit@sha256:897c7cb2088e761bb48003029f1f5ed4fac32997388efe4d158e93dd8d36ac8d
COPY --from=fast /usr/bin/python3.12 /opt/wsp-vllm/bin/python3.12
COPY --from=fast /usr/lib/python3.12 /opt/wsp-vllm/lib/python3.12
COPY --from=fast /usr/local/lib/python3.12/dist-packages /opt/wsp-vllm/lib/python3.12/site-packages
RUN ln -s python3.12 /opt/wsp-vllm/bin/python && \
    /opt/wsp-vllm/bin/python -c "import sys, torch, vllm; from vllm import ModelRegistry; assert 'Qwen3VLForConditionalGeneration' in ModelRegistry.get_supported_archs(); print(sys.version, torch.__version__, vllm.__version__)" && \
    /workspace/.venv/bin/python -c "import torch, transformers; assert torch.__version__ == '2.10.0+cu130'; assert transformers.__version__ == '4.57.6'"

COPY --from=fast /usr/include/python3.12 /opt/wsp-vllm/include/python3.12
# Debian's generic pyconfig.h redirects through a system multiarch path.
# Keep the matching official x86_64 configuration directly in this prefix.
COPY --from=fast /usr/include/x86_64-linux-gnu/python3.12/pyconfig.h /opt/wsp-vllm/include/python3.12/pyconfig.h

# Triton compiles Python extensions at engine startup. Verify that the relocated
# interpreter's actual include path supports compile+import without GPU access.
RUN /opt/wsp-vllm/bin/python - <<'PY'
import importlib.util
from pathlib import Path
import subprocess
import sysconfig
import tempfile
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    source = root / "probe.c"
    source.write_text("""
#include <Python.h>
static struct PyModuleDef probe = {PyModuleDef_HEAD_INIT, "_wsp_header_probe", NULL, -1, NULL};
PyMODINIT_FUNC PyInit__wsp_header_probe(void) {
    PyObject *module = PyModule_Create(&probe);
    if (module == NULL) return NULL;
    if (PyModule_AddIntConstant(module, "python_minor", PY_MINOR_VERSION) < 0) {
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
""")
    shared = root / ("_wsp_header_probe" + sysconfig.get_config_var("EXT_SUFFIX"))
    include = sysconfig.get_path("include")
    subprocess.run(["cc", "-shared", "-fPIC", "-I" + include, str(source), "-o", str(shared)], check=True)
    spec = importlib.util.spec_from_file_location("_wsp_header_probe", shared)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.python_minor == 12
    print("PASS: CPU extension compiled and imported using " + include)
PY
