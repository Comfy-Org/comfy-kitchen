# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib

import pytest
import torch
from torch.nn import functional

import comfy_kitchen as ck
from tests.conftest import get_capable_backends

FP8 = torch.float8_e4m3fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ref(x, w, b=None):
    """What the eager path (and ComfyUI's manual cast) computes."""
    return functional.linear(x, w.to(x.dtype), b.to(x.dtype) if b is not None else None)


def _ref32(x, w, b=None):
    """fp32 reference on the same operands (bias rounded to x's dtype, as the reference does)."""
    out = x.float() @ w.float().t()
    if b is not None:
        out = out + b.to(x.dtype).float()
    return out


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30)).item()


def _bias(kind, n, device):
    if kind == "none":
        return None
    b = torch.randn(n, device=device) * 0.02
    dtypes = {"fp8": FP8, "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    return b.to(dtypes[kind])


def _inputs(m, k, n, dtype, bias="fp8", device="cuda"):
    """x ~ N(0, 1); the weight in the range of DiT checkpoints, stored as FP8."""
    x = torch.randn(m, k, device=device).to(dtype)
    w = (torch.randn(n, k, device=device) * 0.02).to(FP8)
    return x, w, _bias(bias, n, device)


def _require_triton():
    """Skip unless the triton backend can run the kernel on this GPU: on NVIDIA that
    needs the SM version of the triton constraint (8.9, FP8 e4m3 conversion)."""
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    if "triton" not in get_capable_backends("linear_w8a16", "cuda"):
        pytest.skip("triton backend not capable for linear_w8a16")
    need = ck.registry.get_constraints("triton", "linear_w8a16").min_compute_capability
    have = torch.cuda.get_device_capability()
    if need is not None and have < need:
        pytest.skip(f"triton linear_w8a16 needs SM {need[0]}.{need[1]}+ (FP8 e4m3), have {have}")


@contextlib.contextmanager
def _rocm_gate_lifted():
    """Lift the ROCm performance gate (row limit, minimum weight size) of the served
    archs, so the kernel's correctness is checked at any size; the gate itself has its
    own test. Archs that are not served stay refused."""
    from comfy_kitchen.backends.triton import w8a16 as mod

    rows, weight = dict(mod.HIP_MAX_ROWS), dict(mod.HIP_MIN_WEIGHT)
    mod.HIP_MAX_ROWS.update(dict.fromkeys(rows, 1 << 62))
    mod.HIP_MIN_WEIGHT.update(dict.fromkeys(weight, 0))
    try:
        yield
    finally:
        mod.HIP_MAX_ROWS.update(rows)
        mod.HIP_MIN_WEIGHT.update(weight)


def _triton(x, w, b=None):
    kwargs = {"x": x, "weight": w, "bias": b}
    with _rocm_gate_lifted():
        valid = ck.registry.validate_backend_for_call("triton", "linear_w8a16", kwargs)
        if not valid.success:
            why = f"{valid.failed_param}: {valid.failure_reason}"
            pytest.skip(f"triton refuses this call here: {why}")
        with ck.use_backend("triton"):
            return ck.linear_w8a16(x, w, b)


def _check(out, x, w, b):
    """Within bf16/fp16 rounding of the reference: the operands are identical, only the
    fp32 summation order differs."""
    ref = _ref(x, w, b)
    assert out.shape == ref.shape
    assert out.dtype == x.dtype
    assert _rel(out, ref) <= 2e-2
    assert _rel(out, _ref32(x, w, b)) <= 1e-2


# (M, K, N): Qwen-Image / Flux.1 linears (M 1 modulation, M 5 / 46 / 262 text,
# K 64 img_in, N 64 proj_out) at reduced M, plus M and N that are not tile multiples.
_SHAPES = [
    (1, 256, 3072),
    (1, 3072, 6144),
    (3, 512, 1000),
    (5, 3584, 3072),
    (8, 1024, 4100),
    (9, 512, 3000),
    (46, 3072, 3072),
    (262, 1024, 12288),
    (300, 12288, 320),
    (513, 3072, 64),
    (1000, 64, 3072),
]


# ---------------------------------------------------------------------------
# Triton
# ---------------------------------------------------------------------------


class TestLinearW8A16Triton:
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("shape", _SHAPES)
    def test_matches_reference(self, shape, dtype, seed):
        _require_triton()
        m, k, n = shape
        x, w, b = _inputs(m, k, n, dtype, "fp8")
        _check(_triton(x, w, b), x, w, b)

    @pytest.mark.parametrize("bias", ["none", "fp8", "bf16", "fp16", "fp32"])
    @pytest.mark.parametrize("m", [4, 200])
    def test_bias_dtypes(self, bias, m, seed):
        """Any float bias is rounded to x's dtype first, as the reference does."""
        _require_triton()
        x, w, b = _inputs(m, 512, 384, torch.bfloat16, bias)
        _check(_triton(x, w, b), x, w, b)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("m", [1, 8, 128])
    def test_every_fp8_value_exact(self, dtype, m):
        """x = rows of the identity: out[i, j] = weight[j, i], exactly. Covers every
        finite e4m3fn value (incl. subnormals and +-0) on the GEMV and GEMM paths."""
        _require_triton()
        codes = torch.arange(256, dtype=torch.int32, device="cuda")
        codes = codes[(codes & 0x7F) != 0x7F]  # 254 finite values; 0x7f / 0xff are NaN
        k, n = 128, codes.numel()
        idx = (torch.arange(n, device="cuda")[:, None] + torch.arange(k, device="cuda")) % n
        w = codes[idx].to(torch.uint8).view(FP8)
        x = torch.eye(k, device="cuda", dtype=dtype)[:m]
        out = _triton(x, w)
        assert torch.equal(out, w.t()[:m].to(dtype))

    def test_leading_dims_and_strided_rows(self, seed):
        """(B, S, K) input whose rows are a column slice of a wider buffer, and a weight
        whose rows are a slice of a wider buffer: both read in place."""
        _require_triton()
        buf = torch.randn(2, 37, 1024, device="cuda", dtype=torch.bfloat16)
        x = buf[..., :512]
        wbuf = (torch.randn(640, 1024, device="cuda") * 0.02).to(FP8)
        w = wbuf[:, 256:768]
        b = _bias("fp8", 640, "cuda")
        out = _triton(x, w, b)
        assert out.shape == (2, 37, 640)
        _check(out, x, w, b)
        x1 = buf[0, :3, :512]
        _check(_triton(x1, w, b), x1, w, b)

    def test_zero_rows(self):
        _require_triton()
        x = torch.empty(3, 0, 256, device="cuda", dtype=torch.bfloat16)
        w = torch.zeros(96, 256, device="cuda").to(FP8)
        out = _triton(x, w)
        assert out.shape == (3, 0, 96)
        assert out.dtype == torch.bfloat16

    @pytest.mark.parametrize("path", ["small_m", "large_m"])
    def test_every_config_covers_output(self, path, seed):
        """Each autotune config that fits this device (the configs the autotuner keeps
        after pruning) writes every output element correctly on its own, at a shape no
        tile divides (M and N masked)."""
        _require_triton()
        from comfy_kitchen.backends.triton import w8a16 as mod

        tuner = mod._small_m_kernel if path == "small_m" else mod._large_m_kernel
        m = 5 if path == "small_m" else 77
        x, w, b = _inputs(m, 512, 200, torch.bfloat16, "fp8")
        configs = list(tuner.configs)
        valid = mod._prune(configs, {"k": 512, "x_ptr": x})
        assert valid
        try:
            for cfg in valid:
                tuner.configs = [cfg]
                tuner.cache.clear()
                out = _triton(x, w, b)
                assert _rel(out, _ref(x, w, b)) <= 2e-2, str(cfg)
        finally:
            tuner.configs = configs
            tuner.cache.clear()

    def test_prune_shared_memory(self):
        """Pruning keeps only configs whose pipelined tiles fit the device's shared
        memory, never returns an empty list, and drops the K-untiled ones."""
        _require_triton()
        from comfy_kitchen.backends.triton import w8a16 as mod

        x = torch.zeros(4, 256, device="cuda", dtype=torch.bfloat16)
        limit = mod._max_shared_mem(x.device.index)
        huge = mod._mm(256, 256, 256, 8, 8, 4)  # 3 x 192 KB
        for configs in (mod._SMALL_M_CONFIGS, mod._LARGE_M_CONFIGS):
            kept = mod._prune([*configs, huge], {"k": 256, "x_ptr": x})
            assert kept
            assert huge not in kept
            assert all(
                mod._smem_bytes(c) <= limit and 256 % c.kwargs["block_k"] == 0 for c in kept
            )
        assert mod._prune([huge], {"k": 256, "x_ptr": x}) == [huge]  # last resort, never empty

    def test_read_flush_bench(self, seed):
        """The ROCm autotune timer returns one time (ms) per requested quantile, in order,
        and is the timer both tuners use under ROCm when Triton takes do_bench (3.2+);
        otherwise the tuners keep Triton's default timer."""
        _require_triton()
        import inspect

        import triton

        from comfy_kitchen.backends.triton import w8a16 as mod

        x, w, b = _inputs(64, 512, 256, torch.bfloat16)
        q50, q20, q80 = mod._bench_read_flush(
            lambda: mod.linear_w8a16(x, w, b), quantiles=(0.5, 0.2, 0.8), reps=5
        )
        assert 0 < q20 <= q50 <= q80
        takes_do_bench = "do_bench" in inspect.signature(triton.autotune).parameters
        hip = bool(getattr(torch.version, "hip", None))
        expected = {"do_bench": mod._bench_read_flush} if hip and takes_do_bench else {}
        assert expected == mod._AUTOTUNE_TIMER
        if mod._AUTOTUNE_TIMER:
            for tuner in (mod._small_m_kernel, mod._large_m_kernel):
                assert tuner.do_bench is mod._bench_read_flush

    def test_rocm_gate(self, seed):
        """Under ROCm the kernel is served only on gfx1151 (exact arch, feature suffixes
        stripped), only up to HIP_MAX_ROWS["gfx1151"] rows and only for weights of at
        least HIP_MIN_WEIGHT["gfx1151"] elements; every other ROCm arch, more rows and
        smaller weights fall back to eager. The arch is mocked, so this runs on any GPU;
        on a real ROCm device a refused call returns the eager reference."""
        _require_triton()
        import comfy_kitchen.backends.triton as triton_backend
        from comfy_kitchen.backends.triton import w8a16 as mod

        assert triton_backend._arch_name("gfx1151") == "gfx1151"
        assert triton_backend._arch_name("gfx942:sramecc+:xnack-") == "gfx942"
        limit = mod.HIP_MAX_ROWS["gfx1151"]
        min_weight = mod.HIP_MIN_WEIGHT["gfx1151"]
        big = (torch.randn(min_weight // 1024, 1024, device="cuda") * 0.02).to(FP8)
        small = big[: min_weight // 1024 - 1]

        def served(rows, w):
            x = torch.randn(rows, 1024, device="cuda", dtype=torch.bfloat16)
            kwargs = {"x": x, "weight": w, "bias": None}
            valid = ck.registry.validate_backend_for_call("triton", "linear_w8a16", kwargs)
            return x, valid.success

        real_hip = getattr(torch.version, "hip", None)
        saved_arch = triton_backend._hip_arch
        cases = [("gfx1151", 1, big, True), ("gfx1151", limit, big, True),
                 ("gfx1151", limit + 1, big, False), ("gfx1151", 1, small, False)]
        others = ("gfx1201", "gfx1200", "gfx1150", "gfx1100", "gfx942", "gfx950")
        cases += [(a, 1, big, False) for a in others]
        try:
            torch.version.hip = real_hip or "mocked"
            for arch, rows, w, ok in cases:
                triton_backend._hip_arch = lambda index, arch=arch: arch
                assert served(rows, w)[1] == ok, (arch, rows, w.shape)
        finally:
            triton_backend._hip_arch = saved_arch
            torch.version.hip = real_hip
        if real_hip:
            for rows, w in ((limit + 1, big), (1, small)):
                x, ok = served(rows, w)
                assert not ok
                assert torch.equal(ck.linear_w8a16(x, w), _ref(x, w))

    def test_dispatch_follows_gate(self, seed):
        """Dispatch picks triton where its gate serves the call (NVIDIA SM 8.9+, the
        served ROCm archs) and eager elsewhere (e.g. gfx1201). 16 rows and a 2^21
        element weight are within every served arch's limits."""
        _require_triton()
        import comfy_kitchen.backends.triton as triton_backend
        from comfy_kitchen.backends.triton import w8a16 as mod

        x, w, b = _inputs(16, 1024, 2048, torch.bfloat16)
        served = True
        if getattr(torch.version, "hip", None):
            served = triton_backend._hip_arch(x.device.index) in mod.HIP_MAX_ROWS
        kwargs = {"x": x, "weight": w, "bias": b}
        expected = "triton" if served else "eager"
        assert ck.registry.get_capable_backend("linear_w8a16", kwargs) == expected


# ---------------------------------------------------------------------------
# Eager reference and fallback
# ---------------------------------------------------------------------------


class TestLinearW8A16Eager:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("bias", ["none", "fp8", "fp32"])
    def test_eager_is_reference(self, dtype, bias, device, seed):
        if device == "cpu" and dtype == torch.float16:
            pytest.skip("fp16 linear on CPU not checked")
        x, w, b = _inputs(6, 64, 48, dtype, bias, device)
        x = x.reshape(2, 3, 64)
        with ck.use_backend("eager"):
            out = ck.linear_w8a16(x, w, b)
        assert out.shape == (2, 3, 48)
        assert torch.equal(out, _ref(x, w, b))

    @pytest.mark.parametrize(
        "case", ["fp32_x", "e5m2_weight", "k_not_multiple", "weight_col_stride", "cpu"]
    )
    def test_fallback(self, case, seed):
        """Calls the triton kernel does not serve are refused there and run the eager
        reference instead."""
        if not torch.cuda.is_available() or "triton" not in get_capable_backends(
            "linear_w8a16", "cuda"
        ):
            pytest.skip("triton backend required")
        device = "cpu" if case == "cpu" else "cuda"
        dtype = torch.float32 if case in ("fp32_x", "cpu") else torch.bfloat16
        k = 100 if case == "k_not_multiple" else 256
        x, w, b = _inputs(10, k, 64, dtype, "fp8", device)
        if case == "e5m2_weight":
            w = w.to(torch.bfloat16).to(torch.float8_e5m2)
        elif case == "weight_col_stride":
            w = w.t().contiguous().t()
        kwargs = {"x": x, "weight": w, "bias": b}
        assert not ck.registry.validate_backend_for_call("triton", "linear_w8a16", kwargs).success
        out = ck.linear_w8a16(x, w, b)
        assert torch.equal(out, _ref(x, w, b))

    def test_fake_shape(self):
        x = torch.empty(2, 7, 96, device="meta", dtype=torch.bfloat16)
        w = torch.empty(40, 96, device="meta", dtype=FP8)
        out = torch.ops.comfy_kitchen.linear_w8a16(x, w, None)
        assert out.shape == (2, 7, 40)
        assert out.dtype == torch.bfloat16
