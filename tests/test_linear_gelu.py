# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.nn import functional

import comfy_kitchen as ck
from tests.conftest import assert_values_close, get_capable_backends

# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def _ref_linear_gelu(x, weight, bias, approximate="tanh"):
    """fp32 reference, rounded to x's dtype at the end."""
    b = None if bias is None else bias.float()
    out = functional.gelu(functional.linear(x.float(), weight.float(), b), approximate=approximate)
    return out.to(x.dtype)


def _inputs(lead, k, n, dtype, device, bias=True):
    x = torch.randn(*lead, k, dtype=dtype, device=device)
    # Scale the weight so the pre-activation is O(1) and GELU's curved region is hit.
    weight = (torch.randn(n, k, dtype=torch.float32, device=device) * k ** -0.5).to(dtype)
    b = torch.randn(n, dtype=dtype, device=device) * 0.5 if bias else None
    return x, weight, b


def _tol(dtype):
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    if dtype == torch.float16:
        return 1e-2, 1e-2
    return 1e-4, 1e-4


# Shapes served by the triton kernel: weight (N, K) with N % 128 == 0,
# K % 64 == 0, and at least 256 rows of x. M not a multiple of any tile size
# exercises the row mask.
_SERVED = [
    ((300,), 256, 512),
    ((2, 520), 1536, 1024),
    ((1, 1170), 1536, 8960),  # Wan2.1-1.3B FFN up, short sequence
]


def _kwargs(x, weight, bias, approximate="tanh"):
    return {"x": x, "weight": weight, "bias": bias, "approximate": approximate}


def _skip_unless_served(backend, kwargs):
    # get_capable_backends does not check constraints such as
    # min_compute_capability, so ask the registry the same question dispatch does.
    valid = ck.registry.validate_backend_for_call(backend, "linear_gelu", kwargs)
    if not valid.success:
        why = f"{valid.failed_param}: {valid.failure_reason}"
        pytest.skip(f"{backend} refuses this call here: {why}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLinearGelu:
    @pytest.mark.parametrize("backend", ["triton", "eager"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("shape", _SERVED)
    @pytest.mark.parametrize("has_bias", [True, False])
    def test_forward(self, backend, dtype, shape, has_bias, seed, cuda_available):
        device = "cuda" if cuda_available else "cpu"
        if device == "cpu":
            pytest.skip("fp16/bf16 GEMM reference is checked on GPU")
        capable = get_capable_backends("linear_gelu", device)
        if backend not in capable:
            pytest.skip(f"backend '{backend}' not capable for linear_gelu on {device}")

        lead, k, n = shape
        x, weight, bias = _inputs(lead, k, n, dtype, device, bias=has_bias)
        if backend != "eager":
            _skip_unless_served(backend, _kwargs(x, weight, bias))

        with ck.use_backend(backend):
            out = ck.linear_gelu(x, weight, bias)
        ref = _ref_linear_gelu(x, weight, bias)

        assert out.shape == (*lead, n)
        assert out.dtype == dtype
        assert out.is_contiguous()
        rtol, atol = _tol(dtype)
        assert_values_close(out.float(), ref.float(), rtol=rtol, atol=atol,
                            name=f"linear_gelu[{backend}/{dtype}/{shape}]")

    @pytest.mark.parametrize("backend", ["triton"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_vs_eager(self, backend, dtype, seed, cuda_available):
        """The fused kernel must match the unfused eager ops it replaces."""
        device = "cuda" if cuda_available else "cpu"
        capable = get_capable_backends("linear_gelu", device)
        if backend not in capable:
            pytest.skip(f"backend '{backend}' not capable for linear_gelu on {device}")
        if "eager" not in capable:
            pytest.skip("eager backend not capable (reference unavailable)")

        x, weight, bias = _inputs((2, 384), 1536, 1024, dtype, device)
        _skip_unless_served(backend, _kwargs(x, weight, bias))
        with ck.use_backend("eager"):
            ref = ck.linear_gelu(x, weight, bias)
        with ck.use_backend(backend):
            out = ck.linear_gelu(x, weight, bias)

        rtol, atol = _tol(dtype)
        assert_values_close(out.float(), ref.float(), rtol=rtol, atol=atol,
                            name=f"linear_gelu[{backend} vs eager/{dtype}]")

    def test_non_contiguous_input(self, seed, cuda_available):
        """A transposed x still computes the right result on whichever backend runs."""
        device = "cuda" if cuda_available else "cpu"
        if not get_capable_backends("linear_gelu", device):
            pytest.skip("no capable backend available")
        dtype = torch.float16 if device == "cuda" else torch.float32

        x, weight, bias = _inputs((512, 2), 256, 512, dtype, device)
        x = x.transpose(0, 1)  # (2, 512, 256), non-contiguous
        out = ck.linear_gelu(x, weight, bias)
        ref = _ref_linear_gelu(x, weight, bias)

        rtol, atol = _tol(dtype)
        assert out.shape == (2, 512, 512)
        assert_values_close(out.float(), ref.float(), rtol=rtol, atol=atol,
                            name="linear_gelu[non-contiguous x]")

    @pytest.mark.parametrize("n", [384, 640])
    def test_partial_n_tile_vs_eager(self, n, seed, cuda_available):
        """A served N (multiple of 128) that the BN 256 config does not divide: the
        kernel must still write every output column. The config is forced so the
        result does not depend on what the autotuner would pick."""
        device = "cuda" if cuda_available else "cpu"
        capable = get_capable_backends("linear_gelu", device)
        if "triton" not in capable or "eager" not in capable:
            pytest.skip("triton and eager backends needed")
        import triton

        from comfy_kitchen.backends.triton import linear_act

        dtype = torch.float16
        x, weight, bias = _inputs((300,), 256, n, dtype, device)
        _skip_unless_served("triton", _kwargs(x, weight, bias))
        with ck.use_backend("eager"):
            eager = ck.linear_gelu(x, weight, bias)

        tuner = linear_act._linear_gelu_tanh_kernel
        saved = tuner.configs, tuner.cache
        tuner.configs = [triton.Config({"block_m": 128, "block_n": 256, "block_k": 32, "group_size_m": 8},
                                       num_warps=8, num_stages=1)]
        tuner.cache = {}
        try:
            # Hand the output allocation a NaN-filled block, so columns the
            # kernel never writes cannot pass by matching stale memory.
            torch.full((300, n), float("nan"), dtype=dtype, device=device)
            with ck.use_backend("triton"):
                out = ck.linear_gelu(x, weight, bias)
            torch.cuda.synchronize()
        finally:
            tuner.configs, tuner.cache = saved

        rtol, atol = _tol(dtype)
        assert_values_close(out.float(), eager.float(), rtol=rtol, atol=atol,
                            name=f"linear_gelu[BN 256 / N {n} vs eager]")

    @pytest.mark.parametrize(
        "case",
        ["n_not_multiple", "k_not_multiple", "few_rows", "float32", "erf",
         "weight_transposed"],
    )
    def test_unserved_calls_fall_back(self, case, seed, cuda_available):
        """Calls the triton kernel does not serve are rejected by its constraints and
        dispatch to eager, with the same result as the unfused ops."""
        device = "cuda" if cuda_available else "cpu"
        capable = get_capable_backends("linear_gelu", device)
        if "triton" not in capable:
            pytest.skip("triton backend not capable for linear_gelu")

        dtype = torch.float16
        lead, k, n = (300,), 256, 512
        approximate = "tanh"
        if case == "n_not_multiple":
            n = 520
        elif case == "k_not_multiple":
            k = 200
        elif case == "few_rows":
            lead = (16,)
        elif case == "float32":
            dtype = torch.float32
        elif case == "erf":
            approximate = "none"
        x, weight, bias = _inputs(lead, k, n, dtype, device)
        if case == "weight_transposed":
            weight = weight.t().contiguous().t()  # same values, stride(1) != 1

        kwargs = _kwargs(x, weight, bias, approximate)
        assert not ck.registry.validate_backend_for_call("triton", "linear_gelu", kwargs).success

        with ck.use_backend("triton"):
            out = ck.linear_gelu(x, weight, bias, approximate)
        ref = functional.gelu(functional.linear(x, weight, bias), approximate=approximate)

        rtol, atol = _tol(dtype)
        assert_values_close(out.float(), ref.float(), rtol=rtol, atol=atol,
                            name=f"linear_gelu[fallback/{case}]")

    @pytest.mark.parametrize("approximate", ["tanh", "none"])
    def test_eager_matches_torch(self, approximate, seed, cuda_available):
        """The eager backend is exactly the unfused PyTorch ops (fp32, any device)."""
        device = "cuda" if cuda_available else "cpu"
        if "eager" not in get_capable_backends("linear_gelu", device):
            pytest.skip("eager backend not capable")

        x, weight, bias = _inputs((3, 17), 40, 24, torch.float32, device)
        with ck.use_backend("eager"):
            out = ck.linear_gelu(x, weight, bias, approximate)
        ref = functional.gelu(functional.linear(x, weight, bias), approximate=approximate)
        assert torch.equal(out, ref)

    def test_public_api(self):
        assert "linear_gelu" in ck.__all__


# ---------------------------------------------------------------------------
# Autotune: per-arch config list and the autotune timer
# ---------------------------------------------------------------------------


def _linear_act():
    pytest.importorskip("triton")
    from comfy_kitchen.backends.triton import linear_act

    return linear_act


def _cfg(c):
    k = c.kwargs
    return (k["block_m"], k["block_n"], k["block_k"], k["group_size_m"], c.num_warps, c.num_stages)


_BK64 = (128, 128, 64, 8, 4, 2)
_BK32 = (128, 128, 32, 8, 4, 2)
_BM64 = (64, 128, 64, 8, 4, 2)
_BN64 = (128, 64, 32, 8, 4, 2)
_BK32_W8 = (128, 128, 32, 8, 8, 2)
_BN256 = (128, 256, 32, 8, 8, 1)
_ALL = [_BK64, _BK32, _BM64, _BN64, _BK32_W8, _BN256]

# arch -> configs timed. Untagged archs get the full list.
_ARCH_LISTS = {
    "sm_120": [_BK64, _BK32],
    "gfx1151": [_BK64, _BM64],
    "gfx1201": [_BN256],
    "sm_90": _ALL,
    "sm_89": _ALL,
    "gfx942": _ALL,
    "gfx1200": _ALL,
    None: _ALL,
}


class TestAutotuneArch:
    def test_full_list(self):
        la = _linear_act()
        assert [_cfg(c) for c in la._CONFIGS] == _ALL

    @pytest.mark.parametrize("arch", list(_ARCH_LISTS))
    def test_configs_for_arch(self, arch):
        la = _linear_act()
        got = la._configs_for_arch(la._CONFIGS, la._TAGGED, arch)
        assert [_cfg(c) for c in got] == _ARCH_LISTS[arch]

    def test_configs_not_in_tagged_list_are_kept(self):
        """A config list the tags do not cover (a test forcing its own config)
        is returned unchanged."""
        la = _linear_act()
        import triton

        own = [triton.Config({"block_m": 64, "block_n": 64, "block_k": 32, "group_size_m": 8},
                             num_warps=4, num_stages=1)]
        for arch in ("sm_120", "gfx1151", "gfx1201", None):
            assert la._configs_for_arch(own, la._TAGGED, arch) == own

    @pytest.mark.parametrize(
        ("backend", "arch", "name"),
        [("cuda", 120, "sm_120"), ("cuda", 89, "sm_89"), ("hip", "gfx1151", "gfx1151"),
         ("hip", "gfx942:sramecc+:xnack-", "gfx942"), ("hip", "gfx1201", "gfx1201"),
         ("xpu", "pvc", None)],
    )
    def test_arch_name(self, backend, arch, name):
        from collections import namedtuple

        la = _linear_act()
        target = namedtuple("GPUTarget", ["backend", "arch", "warp_size"])(backend, arch, 32)
        assert la._arch_name(target) == name

    @pytest.mark.parametrize("arch", ["sm_120", "gfx1151", "gfx1201", "sm_90", None])
    def test_autotuner_prunes_by_current_arch(self, arch, monkeypatch):
        """The autotuner prunes with the arch read when it tunes a new key."""
        la = _linear_act()
        monkeypatch.setattr(la, "_current_arch", lambda: arch)
        tuner = la._linear_gelu_tanh_kernel
        got = tuner.early_config_prune(tuner.configs, {})
        assert [_cfg(c) for c in got] == _ARCH_LISTS[arch]

    def test_current_arch_matches_device(self, cuda_available):
        if not cuda_available:
            pytest.skip("needs a GPU")
        la = _linear_act()
        arch = la._current_arch()
        if getattr(torch.version, "hip", None):
            name = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
            assert arch == name
        else:
            major, minor = torch.cuda.get_device_capability(0)
            assert arch == f"sm_{major}{minor}"


class TestAutotuneTimer:
    def test_autotuner_uses_warm_timer(self):
        """The autotuner times with _bench_warm where Triton accepts a do_bench."""
        la = _linear_act()
        import inspect

        import triton

        if "do_bench" not in inspect.signature(triton.autotune).parameters:
            pytest.skip("triton.autotune has no do_bench (Triton < 3.2)")
        assert la._linear_gelu_tanh_kernel.do_bench is la._bench_warm

    def test_warm_timer_quantiles(self, seed, cuda_available):
        """The timer returns one time (ms) per requested quantile, in order."""
        if not cuda_available:
            pytest.skip("needs a GPU")
        la = _linear_act()
        a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
        q50, q20, q80 = la._bench_warm(lambda: a @ a, quantiles=(0.5, 0.2, 0.8), warmup=1, rep=5)
        assert 0 < q20 <= q50 <= q80
