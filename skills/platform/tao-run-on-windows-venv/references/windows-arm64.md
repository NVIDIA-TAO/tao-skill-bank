# Native Windows ARM64 host gate

Use this reference when a model skill explicitly supports a native
Windows-on-ARM64 route. It is a platform gate only: it verifies interpreter
architecture, CUDA visibility, and an explicit allocation. It does not prove a
particular model, Torch ABI, compiled extension, decoder, or workload fits.

## Verify the interpreter and GPU

Use the target venv's Python, never a system Python selected by `PATH`:

```powershell
$PY = "<arm64-venv>\Scripts\python.exe"
& $PY -c "import platform, torch; print(platform.machine()); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
& $PY <bank>\skills\platform\tao-run-on-windows-venv\scripts\preflight_windows_arm64.py `
  --allocation-gib <model-required-gib>
```

The first command must report `ARM64` and a CUDA-visible GPU. The second
allocates a `uint8` CUDA buffer of the requested size, writes it, synchronizes,
and frees it before exit. It intentionally tests allocation rather than only
`torch.cuda.is_available()`.

Run the allocation probe only after reserving or otherwise coordinating access
to the GPU: Windows has no per-user GPU quota or memory partitioning. The
reported CUDA total may reflect unified or pageable capacity differently from a
WDDM `nvidia-smi` framebuffer counter. Use the model skill's measured capacity
requirement; this platform reference does not choose one.

If the architecture, CUDA import, or allocation fails, stop before downloading
weights or launching a job. Resolve the model-owned driver/Torch/CUDA ABI and
compiled dependency contract first.
