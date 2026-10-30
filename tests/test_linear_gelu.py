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
            assert ck.registry.validate_backend_for_call(
                backend, "linear_gelu", _kwargs(x, weight, bias)
            ).success

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
