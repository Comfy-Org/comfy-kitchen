# AGENTS.md

Comfy Kitchen is a fast kernel library for diffusion inference (`comfy-kitchen` on PyPI, package `comfy_kitchen`). It exposes quantization, attention, RoPE, normalization and GEMM kernels behind a single dispatch registry that picks the best available backend per call — `hip` → `cuda` → `triton` → `eager` — validating each backend's device/dtype/shape/compute-capability constraints up front instead of using try/except fallbacks. The `cuda` and `hip` backends are native extensions compiled from C/C++/CUDA/HIP sources at build time; `eager` (pure PyTorch) and `triton` need no compiler.

## Build / install

Requires Python ≥3.10 (repo targets 3.12), PyTorch ≥2.7.0. Native builds need CMake ≥3.26, Ninja, nanobind ≥2.0.0, and the relevant toolchain (CUDA Toolkit ≥12.8 with `CUDA_HOME` set, and/or ROCm for HIP).

```bash
git submodule update --init --recursive   # third_party/cutlass + flash-attention are required for source builds
pip install .                             # source build; CUDA auto-detected, HIP auto-selected when nvcc is absent
pip install -e ".[dev]"                   # editable + dev deps (pytest, pytest-benchmark, ruff)
pip install -e . --no-build-isolation -v  # faster iterative rebuilds
```

- `COMFY_KITCHEN_BUILD_HIP=1` adds the HIP extension to a CUDA build; `COMFY_KITCHEN_BUILD_NO_HIP=1` / `--no-hip` suppresses it; `--no-cuda` builds a CPU-only (pure-Python) wheel.
- `COMFY_HIP_ARCHS` / `COMFY_CUDA_ARCHS` (or `--hip-archs=`/`--cuda-archs=` via `setup.py build_ext`) pin the GPU targets; under build isolation no GPU is visible, so the build falls back to the full manifest of targets.
- `COMFY_KITCHEN_DISABLE_HIP=1` removes the HIP backend from dispatch at runtime.
- Build options such as `--no-cuda`, `--hip`, `--debug-build`, `--lineinfo` require invoking `setup.py` directly (they are not `pip install` flags). See `README.md` for the full option table.

## Test / lint

CI (`.github/workflows/build-wheels.yml`, jobs `test_*`) installs the CPU-only wheel plus CPU PyTorch, then runs exactly:

```bash
ruff check .                       # lint (also the format checker; config in pyproject.toml)
python -m pytest tests/ -v --tb=short   # or just `pytest` — config in pytest.ini
```

- Tests requiring a GPU are marked (`@pytest.mark.cuda`, `slow`, `cupy`) and `tests/conftest.py` skips them when the backend/hardware is unavailable, so the suite runs green on CPU-only machines.
- `pytest.ini` sets `--strict-markers`: register any new marker there or collection fails.
- Run a single file/test with e.g. `pytest tests/test_rope.py` or `pytest tests/test_backends.py::TestBackendSystem::test_list_backends`.

## Layout

```
comfy_kitchen/           # the package
  registry.py            # backend registry + automatic dispatch (hip→cuda→triton→eager)
  constraints.py         # device/dtype/shape/compute-capability constraints checked before dispatch
  exceptions.py          # e.g. NoCapableBackendError
  tensor/                # QuantizedTensor (torch.Tensor subclass) + FP8/NVFP4/MXFP8 layouts
  backends/
    eager/               # pure-PyTorch reference implementations (CPU + GPU, always present)
    triton/              # Triton JIT kernels
    cuda/                # native CUDA extension sources (*.cu/*.cuh/*.h) → cuda/_C*.so
    hip/                 # native HIP extension sources + CMakeLists.txt + architectures.json → hip/_C*.so
  flash_attention.py, sage_attention.py, gated_delta.py, scaled_mm_v2.py, allocation.py, _rope_utils.py
tests/                   # pytest suite (test_*.py) + conftest.py
samples/                 # standalone usage examples (nvfp4_linear.py, mxfp8_model_patcher.py)
third_party/             # git submodules: cutlass, flash-attention (do not edit)
setup.py                 # all native-build logic (CUDA + HIP CMake orchestration, arch detection)
pyproject.toml           # metadata, ruff config, build-system requires
```

## Conventions & gotchas

- **Ruff is the single source for style** (`pyproject.toml` `[tool.ruff]`): line length 100, target py310, double-quote strings, respect magic trailing commas. Selected rule sets include `E/W/F/I/N/UP/B/C4/SIM/RUF`; `E501` is delegated to the formatter. Run `ruff check .` (and `ruff format` for formatting) before pushing — CI fails on lint errors.
- **DCO sign-off is required** on every commit (`CONTRIBUTING.md`): commit with `git commit -s`. Contributions are Apache-2.0.
- **CLA**: first-time PR authors must sign the Comfy CLA — the `CLA Assistant` workflow comments the signing instructions on the PR. Only the PR author signs (bots and co-committers are allowlisted).
- **Submodules are mandatory for native builds.** `third_party/cutlass` and `third_party/flash-attention` must be checked out (`--recursive`); CI checks out with `submodules: true`. Treat `third_party/` as vendored — do not edit it.
- **Native sources are packaged via `MANIFEST.in`** (HIP CMake + `*.cpp/*.h/*.hip/*.in/*.json`) and `pyproject.toml` `package-data` (CUDA `*.cuh/*.h`, HIP `architectures.json`). Adding a new native source file that must ship in the sdist/wheel means updating those lists too.
- **abi3 / Stable ABI:** on Python 3.12+ both extensions build against the limited API and the wheel keeps its `abi3` tag (covers 3.13+). CI asserts the cp312 wheel stays `abi3` and links `python3.dll`/the stable module — don't break the limited-API build.
- **HIP arch overrides fail closed:** a compiler-recognized gfx target absent from `comfy_kitchen/backends/hip/architectures.json` is rejected until its device + WMMA policies are added. RDNA2 lacks matrix cores, so WMMA GEMMs are not advertised there and fall back to triton/eager.
- **Dispatch is constraint-gated, not exception-gated:** a backend runs only when its declared constraints pass; if none qualify, a `NoCapableBackendError` is raised (see `registry.py`/`constraints.py`). Add new kernels by registering their constraints, not by wrapping calls in try/except.
- **Keep the backend capability matrix in `README.md` current** when adding or removing a function/backend pairing.

## Deeper docs

- `README.md` — capability matrix, HIP backend design (WMMA/RDNA generations), `QuantizedTensor` usage, full build-option table, requirements.
- `CONTRIBUTING.md` — license, DCO text.
- `.github/workflows/build-wheels.yml` — authoritative build/test/publish pipeline (Linux x86_64/arm64, Windows x64/arm, CPU-only wheel; PyPI publish on `v*` tags).
- `.github/workflows/cla.yml` — CLA enforcement.
