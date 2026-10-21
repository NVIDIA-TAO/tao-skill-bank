# Build vLLM on native Windows ARM64

Use this optional platform route when a model skill supports vLLM on native
Windows ARM64 but no published wheel matches its required vLLM revision,
Python ABI, Torch/CUDA ABI, or CUDA architecture. This route builds the generic
vLLM serving engine; it is not specific to Cosmos3 or any other model.

## Ownership boundary

The selected vLLM Windows source tree owns the compiler flags, dependency
patches, native extensions, and wheel build. Use its checked-in
`tools/build-win-arm64.ps1` and the adjacent Windows ARM64 section in its
`README.md` as the canonical procedure. Do not reproduce its CMake command or
patch CUDA headers manually in this skill.

This platform reference owns source selection, host/toolchain checks, artifact
provenance, and wheel acceptance. The consuming model skill still owns the
required vLLM revision, Torch/CUDA ABI, optional media packages, server
arguments, supported endpoints and modalities, model correctness, memory, and
performance validation.

## Required inputs

Collect these before building:

- an explicit vLLM Windows repository path and immutable commit;
- native ARM64 CPython venv and its intended wheel ABI;
- the model-owned Torch/CUDA ABI and CUDA Toolkit path;
- Visual Studio ARM64 toolset and Windows SDK versions;
- portable or exact-GPU CUDA architecture profile;
- user-owned build, wheel-output, and cache directories; and
- Rust, Perl, and Protobuf tool paths required by the checked-in script.

Do not infer a source commit from an installed wheel or silently build the tip
of a mutable branch. Keep source builds separate from an existing runtime venv.

## Build contract

From the selected clean source tree, first run configuration-only validation:

```powershell
.\tools\build-win-arm64.ps1 `
  -VenvDir "<native-arm64-build-venv>" `
  -CudaPath "<cuda-toolkit-root>" `
  -PerlPath "<perl.exe>" `
  -ProtocPath "<protoc.exe>" `
  -ProtocIncludePath "<protobuf-include-root>" `
  -SkipBuild
```

Then rerun without `-SkipBuild`. The script must select an ARM64 compiler,
resolve ARM64 CUDA libraries, validate native ARM64 Python and the exact CUDA
version reported by Torch, configure every native dependency, isolate the
architecture profile's build directory, emit a `win_arm64` wheel, and print its
SHA-256 digest.

The validated script defaults to a portable `12.0+PTX;10.3a` profile. An exact
SM 12.1 build uses `-CudaArchList "12.1a"` with
`-CMakeCudaArchitectures "121-real"` and a distinct wheel directory. Treat an
exact profile as a compatibility restriction, not a generally faster wheel;
benchmark the consuming workload before preferring it.

Do not reuse objects between profiles, lower `MAX_JOBS` only to manage host
memory pressure, and do not edit the installed CUDA Toolkit. The script's CUDA
header workaround is applied to a build-local overlay.

## Artifact acceptance

Record the source remote and commit, build-script SHA-256, Python ABI,
Torch/CUDA versions, MSVC and SDK versions, architecture profile, wheel name,
size, and SHA-256. Install into a fresh native ARM64 runtime venv and require:

1. the wheel tag ends in `win_arm64` and the interpreter reports ARM64;
2. `pip check` succeeds against the model-owned dependency set;
3. `import torch, vllm` succeeds and a CUDA tensor operation completes;
4. required vLLM native extensions import from the installed wheel; and
5. the model skill's own load, request, modality, shutdown, and performance
   checks pass.

A wheel build or import alone is not model validation. Do not publish a wheel
until the intended compatibility scope and the validation evidence are named.
