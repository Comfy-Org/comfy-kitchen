# Agent instructions

PyTorch kernel library for diffusion inference. Four backends (`hip`, `cuda`, `triton`, `eager`); the registry in `comfy_kitchen/registry.py` dispatches by constraints in priority order **hip → cuda → triton → eager** — no try/except fallback (unsupported inputs raise or fall through by constraint). Python ≥3.10, repo venv at `venv/`.

## Build (this host: Windows + ROCm/HIP, gfx1201)

- HIP is auto-selected when `nvcc` is absent: plain `pip install .` builds the HIP extension.
- PEP 517 build isolation cannot see the GPU, so it silently builds for the full architecture manifest (slow). Build for the local GPU instead:
  - `COMFY_HIP_ARCHS=gfx1201 pip install .` (PowerShell: `$env:COMFY_HIP_ARCHS = "gfx1201"; pip install .`)
  - or `pip install -e . --no-build-isolation -v` for dev rebuilds.
- Requirements: CMake ≥3.26, Ninja, MSVC v143 toolset + Windows SDK (clang locates them itself).

## Verify

```bash
ruff check .                      # lint gate (CI runs this first)
pytest                            # full suite; conftest skips by backend availability
pytest tests/test_rope.py         # one file
pytest tests/test_backends.py::TestBackendSystem::test_list_backends   # one test
```

## Testing gotchas

- On ROCm, `torch.cuda.is_available()` is **True even without a CUDA extension**. Gate CUDA-backend tests with the `requires_cuda_backend` fixture from `tests/conftest.py`, not `torch.cuda`.
- The autouse `restore_backend_selection` fixture resets process-global registry priority/disabled state after every test. Don't remove it or work around it; any new test that rewrites dispatch order relies on it to avoid silently re-routing later tests.
- Markers: `cuda`, `slow`, `cupy`, `performance`.

## Dispatch behavior

- Backend selection is process-global; override per-call with `backend="..."` or `with ck.use_backend("triton"):`. `COMFY_KITCHEN_DISABLE_HIP=1` removes the HIP backend at runtime.
- Debug dispatch decisions: `logging.getLogger("comfy_kitchen.dispatch").setLevel(logging.DEBUG)`.

## Rules

- **Build sequence, always in this order:** open MSVC x64 Native Tools Command Prompt → `cd` into repo → `venv\scripts\activate` → set `COMFY_HIP_ARCHS`/`COMFY_KITCHEN_BUILD_HIP` → then `pip install -e . --no-build-isolation -v`.
- **`git push` always to my fork, never to `upstream`/`Comfy-Org/comfy-kitchen`.** Check `git remote -v` first if unsure which remote is which.
- **Never commit/push directly to `main`.** Always a branch + PR.
- **Never edit `backends/hip/architectures.json` just to force an unvalidated arch through.** Fail-closed is intentional — a rejected target means "not reviewed," not "add it to make the build pass."
- **After any kernel/backend change, rebuild before claiming it's fixed.** A code edit without a rebuild is untested by definition.
- **Run `ruff check .` on changed files before considering a task done**, not just `pytest`.
