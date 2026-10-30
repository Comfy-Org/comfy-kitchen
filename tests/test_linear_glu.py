# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.nn import functional

import comfy_kitchen as ck
from tests.conftest import assert_values_close, get_capable_backends

# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def _ref_geglu(x, weight, bias, approximate="none"):
    """fp32 reference, rounded to x's dtype at the end."""
    b = None if bias is None else bias.float()
    h, gate = functional.linear(x.float(), weight.float(), b).chunk(2, dim=-1)
    return (h * functional.gelu(gate, approximate=approximate)).to(x.dtype)


def _ref_swiglu(x, w1, w3, b1=None, b3=None):
    """fp32 reference, rounded to x's dtype at the end."""
    xf = x.float()
    gate = functional.linear(xf, w1.float(), None if b1 is None else b1.float())
    up = functional.linear(xf, w3.float(), None if b3 is None else b3.float())
    return (functional.silu(gate) * up).to(x.dtype)


def _weight(rows, k, dtype, device):
    # Scaled so the pre-activations are O(1) and the activations' curved region is hit.
    return (torch.randn(rows, k, dtype=torch.float32, device=device) * k ** -0.5).to(dtype)


def _bias(rows, dtype, device):
    return torch.randn(rows, dtype=dtype, device=device) * 0.5


def _geglu_inputs(lead, k, n, dtype, device, bias=True):
    x = torch.randn(*lead, k, dtype=dtype, device=device)
    weight = _weight(2 * n, k, dtype, device)  # packed [h | gate]
    return x, weight, _bias(2 * n, dtype, device) if bias else None


def _swiglu_inputs(lead, k, n, dtype, device, bias=False):
    x = torch.randn(*lead, k, dtype=dtype, device=device)
    w1, w3 = _weight(n, k, dtype, device), _weight(n, k, dtype, device)
    b1, b3 = (_bias(n, dtype, device), _bias(n, dtype, device)) if bias else (None, None)
    return x, w1, w3, b1, b3


def _tol(dtype):
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    if dtype == torch.float16:
        return 1e-2, 1e-2
    return 1e-4, 1e-4


def _geglu_kwargs(x, weight, bias, approximate="none"):
    return {"x": x, "weight": weight, "bias": bias, "approximate": approximate}


def _swiglu_kwargs(x, w1, w3, b1=None, b3=None):
    return {"x": x, "w1": w1, "w3": w3, "b1": b1, "b3": b3}


def _require(func, backend, device):
    capable = get_capable_backends(func, device)
    if backend not in capable:
        pytest.skip(f"backend '{backend}' not capable for {func} on {device}")


# Shapes served by the triton kernel: (lead, K, N) with N (one half / the output
# width) % 64 == 0, K % 64 == 0, and at least 256 rows of x. M not a multiple of
# any tile size exercises the row mask.
_SERVED = [
    ((300,), 256, 256),
    ((2, 520), 640, 1280),
]

# The model shapes the kernel was written for (batch 2 = cond + uncond for SDXL).
_SDXL_GEGLU = [
    ((2, 1024), 1280, 5120),  # SDXL UNet GEGLU, 1024x1024 image, 32x32 tokens
    ((2, 4096), 640, 2560),  # SDXL UNet GEGLU, 1024x1024 image, 64x64 tokens
]
_ZIMAGE_SWIGLU = ((1, 4160), 3840, 10240)  # Z-Image FFN w1 / w3, 1024x1024 image


# ---------------------------------------------------------------------------
# linear_geglu
# ---------------------------------------------------------------------------


class TestLinearGeglu:
    @pytest.mark.parametrize("backend", ["triton", "eager"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("shape", _SERVED)
    @pytest.mark.parametrize("has_bias", [True, False])
    @pytest.mark.parametrize("approximate", ["none", "tanh"])
    def test_forward(self, backend, dtype, shape, has_bias, approximate, seed, cuda_available):
        device = "cuda" if cuda_available else "cpu"
        if device == "cpu":
            pytest.skip("fp16/bf16 GEMM reference is checked on GPU")
        _require("linear_geglu", backend, device)

        lead, k, n = shape
        x, weight, bias = _geglu_inputs(lead, k, n, dtype, device, bias=has_bias)
        kwargs = _geglu_kwargs(x, weight, bias, approximate)
        if backend != "eager":
            assert ck.registry.validate_backend_for_call(backend, "linear_geglu", kwargs).success

        with ck.use_backend(backend):
            out = ck.linear_geglu(x, weight, bias, approximate)
        ref = _ref_geglu(x, weight, bias, approximate)

        assert out.shape == (*lead, n)
        assert out.dtype == dtype
        assert out.is_contiguous()
        rtol, atol = _tol(dtype)
        assert_values_close(out.float(), ref.float(), rtol=rtol, atol=atol,
                            name=f"linear_geglu[{backend}/{dtype}/{shape}/{approximate}]")

    @pytest.mark.parametrize("shape", _SDXL_GEGLU)
    def test_sdxl_shapes(self, shape, seed, cuda_available):
        """SDXL's GEGLU (fp16, packed weight with bias, erf GELU): triton vs eager vs fp32."""
        device = "cuda" if cuda_available else "cpu"
        _require("linear_geglu", "triton", device)
        _require("linear_geglu", "eager", device)

        lead, k, n = shape
        x, weight, bias = _geglu_inputs(lead, k, n, torch.float16, device)
        assert ck.registry.validate_backend_for_call(
            "triton", "linear_geglu", _geglu_kwargs(x, weight, bias)
        ).success
        with ck.use_backend("eager"):
            eager = ck.linear_geglu(x, weight, bias)
        with ck.use_backend("triton"):
            out = ck.linear_geglu(x, weight, bias)
        ref = _ref_geglu(x, weight, bias)

        rtol, atol = _tol(torch.float16)
        assert_values_close(out.float(), eager.float(), rtol=rtol, atol=atol,
                            name=f"linear_geglu[triton vs eager/{shape}]")
        assert_values_close(out.float(), ref.float(), rtol=rtol, atol=atol,
                            name=f"linear_geglu[triton vs fp32/{shape}]")

    def test_strided_weight_rows(self, seed, cuda_available):
        """Weight rows with a stride larger than in_features (a column slice) are read in place."""
        device = "cuda" if cuda_available else "cpu"
        _require("linear_geglu", "triton", device)

        x, weight, bias = _geglu_inputs((300,), 256, 256, torch.float16, device)
        wide = torch.zeros(weight.shape[0], 320, dtype=weight.dtype, device=device)
        wide[:, :256] = weight
        view = wide[:, :256]  # stride (320, 1)
        assert ck.registry.validate_backend_for_call(
            "triton", "linear_geglu", _geglu_kwargs(x, view, bias)
        ).success
        with ck.use_backend("triton"):
            out = ck.linear_geglu(x, view, bias)
        rtol, atol = _tol(torch.float16)
        assert_values_close(out.float(), _ref_geglu(x, weight, bias).float(), rtol=rtol,
                            atol=atol, name="linear_geglu[strided weight rows]")

    def test_non_contiguous_input(self, seed, cuda_available):
        """A transposed x still computes the right result on whichever backend runs."""
        device = "cuda" if cuda_available else "cpu"
        if not get_capable_backends("linear_geglu", device):
            pytest.skip("no capable backend available")
        dtype = torch.float16 if device == "cuda" else torch.float32

        x, weight, bias = _geglu_inputs((512, 2), 256, 256, dtype, device)
        x = x.transpose(0, 1)  # (2, 512, 256), non-contiguous
        out = ck.linear_geglu(x, weight, bias)
        ref = _ref_geglu(x, weight, bias)

        rtol, atol = _tol(dtype)
        assert out.shape == (2, 512, 256)
        assert_values_close(out.float(), ref.float(), rtol=rtol, atol=atol,
                            name="linear_geglu[non-contiguous x]")

    @pytest.mark.parametrize(
        "case", ["n_not_multiple", "k_not_multiple", "few_rows", "float32", "weight_transposed"]
    )
    def test_unserved_calls_fall_back(self, case, seed, cuda_available):
        """Calls the triton kernel does not serve are rejected by its constraints and
        dispatch to eager, with the same result as the unfused ops."""
        device = "cuda" if cuda_available else "cpu"
        _require("linear_geglu", "triton", device)

        dtype = torch.float16
        lead, k, n = (300,), 256, 256
        if case == "n_not_multiple":
            n = 260  # packed rows 520, not a multiple of 128
        elif case == "k_not_multiple":
            k = 200
        elif case == "few_rows":
            lead = (16,)
        elif case == "float32":
            dtype = torch.float32
        x, weight, bias = _geglu_inputs(lead, k, n, dtype, device)
        if case == "weight_transposed":
            weight = weight.t().contiguous().t()  # same values, stride(1) != 1

        kwargs = _geglu_kwargs(x, weight, bias)
        assert not ck.registry.validate_backend_for_call("triton", "linear_geglu", kwargs).success

        with ck.use_backend("triton"):
            out = ck.linear_geglu(x, weight, bias)
        h, gate = functional.linear(x, weight, bias).chunk(2, dim=-1)
        ref = h * functional.gelu(gate)

        rtol, atol = _tol(dtype)
        assert_values_close(out.float(), ref.float(), rtol=rtol, atol=atol,
                            name=f"linear_geglu[fallback/{case}]")

    @pytest.mark.parametrize("approximate", ["none", "tanh"])
    def test_eager_matches_torch(self, approximate, seed, cuda_available):
        """The eager backend is exactly the unfused PyTorch ops (fp32, any device)."""
        device = "cuda" if cuda_available else "cpu"
        _require("linear_geglu", "eager", device)

        x, weight, bias = _geglu_inputs((3, 17), 40, 12, torch.float32, device)
        with ck.use_backend("eager"):
            out = ck.linear_geglu(x, weight, bias, approximate)
        h, gate = functional.linear(x, weight, bias).chunk(2, dim=-1)
        assert torch.equal(out, h * functional.gelu(gate, approximate=approximate))

    def test_eager_cpu_matches_torch(self, seed):
        """CPU dispatch (no GPU backend involved) is the unfused PyTorch ops."""
        x, weight, bias = _geglu_inputs((3, 17), 40, 12, torch.float32, "cpu")
        out = ck.linear_geglu(x, weight, bias)
        h, gate = functional.linear(x, weight, bias).chunk(2, dim=-1)
        assert torch.equal(out, h * functional.gelu(gate))

    def test_public_api(self):
        assert "linear_geglu" in ck.__all__


# ---------------------------------------------------------------------------
# linear_swiglu
# ---------------------------------------------------------------------------


class TestLinearSwiglu:
    @pytest.mark.parametrize("backend", ["triton", "eager"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("shape", _SERVED)
    @pytest.mark.parametrize("has_bias", [True, False])
    def test_forward(self, backend, dtype, shape, has_bias, seed, cuda_available):
        device = "cuda" if cuda_available else "cpu"
        if device == "cpu":
            pytest.skip("fp16/bf16 GEMM reference is checked on GPU")
        _require("linear_swiglu", backend, device)

        lead, k, n = shape
        x, w1, w3, b1, b3 = _swiglu_inputs(lead, k, n, dtype, device, bias=has_bias)
        kwargs = _swiglu_kwargs(x, w1, w3, b1, b3)
        if backend != "eager":
            assert ck.registry.validate_backend_for_call(backend, "linear_swiglu", kwargs).success

        with ck.use_backend(backend):
            out = ck.linear_swiglu(x, w1, w3, b1, b3)
        ref = _ref_swiglu(x, w1, w3, b1, b3)

        assert out.shape == (*lead, n)
        assert out.dtype == dtype
        assert out.is_contiguous()
        rtol, atol = _tol(dtype)
        assert_values_close(out.float(), ref.float(), rtol=rtol, atol=atol,
                            name=f"linear_swiglu[{backend}/{dtype}/{shape}/bias={has_bias}]")

    def test_zimage_shape(self, seed, cuda_available):
        """Z-Image's FFN (bf16, separate w1 / w3, no bias): triton vs eager vs fp32."""
        device = "cuda" if cuda_available else "cpu"
        _require("linear_swiglu", "triton", device)
        _require("linear_swiglu", "eager", device)

        lead, k, n = _ZIMAGE_SWIGLU
        x, w1, w3, _, _ = _swiglu_inputs(lead, k, n, torch.bfloat16, device)
        assert ck.registry.validate_backend_for_call(
            "triton", "linear_swiglu", _swiglu_kwargs(x, w1, w3)
        ).success
        with ck.use_backend("eager"):
            eager = ck.linear_swiglu(x, w1, w3)
        with ck.use_backend("triton"):
            out = ck.linear_swiglu(x, w1, w3)
        ref = _ref_swiglu(x, w1, w3)

        rtol, atol = _tol(torch.bfloat16)
        assert_values_close(out.float(), eager.float(), rtol=rtol, atol=atol,
                            name="linear_swiglu[triton vs eager/Z-Image]")
        assert_values_close(out.float(), ref.float(), rtol=rtol, atol=atol,
                            name="linear_swiglu[triton vs fp32/Z-Image]")

    def test_one_bias(self, seed, cuda_available):
        """b1 and b3 are independent: only the up bias given."""
        device = "cuda" if cuda_available else "cpu"
        _require("linear_swiglu", "triton", device)

        x, w1, w3, _, b3 = _swiglu_inputs((300,), 256, 256, torch.float16, device, bias=True)
        assert ck.registry.validate_backend_for_call(
            "triton", "linear_swiglu", _swiglu_kwargs(x, w1, w3, None, b3)
        ).success
        with ck.use_backend("triton"):
            out = ck.linear_swiglu(x, w1, w3, None, b3)
        rtol, atol = _tol(torch.float16)
        assert_values_close(out.float(), _ref_swiglu(x, w1, w3, None, b3).float(), rtol=rtol,
                            atol=atol, name="linear_swiglu[b3 only]")

    def test_packed_w13_views(self, seed, cuda_available):
        """w1 / w3 may be row slices of one packed [w1; w3] tensor (no copies needed)."""
        device = "cuda" if cuda_available else "cpu"
        _require("linear_swiglu", "triton", device)

        x, w1, w3, _, _ = _swiglu_inputs((300,), 256, 256, torch.bfloat16, device)
        w13 = torch.cat([w1, w3], dim=0)
        v1, v3 = w13[:256], w13[256:]
        assert ck.registry.validate_backend_for_call(
            "triton", "linear_swiglu", _swiglu_kwargs(x, v1, v3)
        ).success
        with ck.use_backend("triton"):
            out = ck.linear_swiglu(x, v1, v3)
        rtol, atol = _tol(torch.bfloat16)
        assert_values_close(out.float(), _ref_swiglu(x, w1, w3).float(), rtol=rtol, atol=atol,
                            name="linear_swiglu[packed w13 views]")

    def test_non_contiguous_input(self, seed, cuda_available):
        """A transposed x still computes the right result on whichever backend runs."""
        device = "cuda" if cuda_available else "cpu"
        if not get_capable_backends("linear_swiglu", device):
            pytest.skip("no capable backend available")
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        x, w1, w3, _, _ = _swiglu_inputs((512, 2), 256, 256, dtype, device)
        x = x.transpose(0, 1)  # (2, 512, 256), non-contiguous
        out = ck.linear_swiglu(x, w1, w3)
        ref = _ref_swiglu(x, w1, w3)

        rtol, atol = _tol(dtype)
        assert out.shape == (2, 512, 256)
        assert_values_close(out.float(), ref.float(), rtol=rtol, atol=atol,
                            name="linear_swiglu[non-contiguous x]")

    @pytest.mark.parametrize(
        "case", ["n_not_multiple", "k_not_multiple", "few_rows", "float32", "w3_transposed"]
    )
    def test_unserved_calls_fall_back(self, case, seed, cuda_available):
        """Calls the triton kernel does not serve are rejected by its constraints and
        dispatch to eager, with the same result as the unfused ops."""
        device = "cuda" if cuda_available else "cpu"
        _require("linear_swiglu", "triton", device)

        dtype = torch.bfloat16
        lead, k, n = (300,), 256, 256
        if case == "n_not_multiple":
            n = 260
        elif case == "k_not_multiple":
            k = 200
        elif case == "few_rows":
            lead = (16,)
        elif case == "float32":
            dtype = torch.float32
        x, w1, w3, _, _ = _swiglu_inputs(lead, k, n, dtype, device)
        if case == "w3_transposed":
            w3 = w3.t().contiguous().t()  # same values, stride(1) != 1

        kwargs = _swiglu_kwargs(x, w1, w3)
        assert not ck.registry.validate_backend_for_call("triton", "linear_swiglu", kwargs).success

        with ck.use_backend("triton"):
            out = ck.linear_swiglu(x, w1, w3)
        ref = functional.silu(functional.linear(x, w1)) * functional.linear(x, w3)

        rtol, atol = _tol(dtype)
        assert_values_close(out.float(), ref.float(), rtol=rtol, atol=atol,
                            name=f"linear_swiglu[fallback/{case}]")

    def test_eager_matches_torch(self, seed, cuda_available):
        """The eager backend is exactly the unfused PyTorch ops (fp32, any device)."""
        device = "cuda" if cuda_available else "cpu"
        _require("linear_swiglu", "eager", device)

        x, w1, w3, b1, b3 = _swiglu_inputs((3, 17), 40, 24, torch.float32, device, bias=True)
        with ck.use_backend("eager"):
            out = ck.linear_swiglu(x, w1, w3, b1, b3)
        ref = functional.silu(functional.linear(x, w1, b1)) * functional.linear(x, w3, b3)
        assert torch.equal(out, ref)

    def test_eager_cpu_matches_torch(self, seed):
        """CPU dispatch (no GPU backend involved) is the unfused PyTorch ops."""
        x, w1, w3, _, _ = _swiglu_inputs((3, 17), 40, 24, torch.float32, "cpu")
        out = ck.linear_swiglu(x, w1, w3)
        ref = functional.silu(functional.linear(x, w1)) * functional.linear(x, w3)
        assert torch.equal(out, ref)

    def test_public_api(self):
        assert "linear_swiglu" in ck.__all__
