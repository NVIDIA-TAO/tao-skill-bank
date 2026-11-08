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
