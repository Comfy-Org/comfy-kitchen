# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""int8_linear(input_act=...) — folding an MLP's activation into the quantizer."""

import pytest
import torch
from torch.nn import functional

import comfy_kitchen as ck
from comfy_kitchen.tensor.int8_utils import _build_hadamard
from tests.conftest import cuda_backend_available, get_capable_backends

_GROUP = 256


def _gelu(x):
    return functional.gelu(x, approximate="tanh")


def _exact_fp64(h, group=_GROUP):
    """gelu -> ConvRot -> row-wise int8, with no intermediate rounding."""
    hd = h.double()
    g = 0.5 * hd * (1 + torch.tanh(0.7978845608028654 * (hd + 0.044715 * hd**3)))
    k = h.shape[-1]
    mat = _build_hadamard(group, device=h.device, dtype=torch.float64)
    rot = (g.reshape(-1, k // group, group) @ mat).reshape(-1, k)
    scale = (rot.abs().amax(-1, keepdim=True) / 127.0).clamp(min=1e-30)
    return (rot / scale).round().clamp(-128, 127), scale


class TestInputActQuantizer:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("shape", [(512, 256), (1024, 4096), (256, 16384), (37, 2048)])
    def test_at_least_as_accurate_as_eager_chain(self, dtype, shape, seed, cuda_available):
        """Folding gelu in must not be worse than gelu-then-quantize.

        It is normally better: the eager chain rounds gelu's output to the input
        dtype before quantizing, while the fused path keeps float32 all the way
        to the int8 conversion.
        """
        if not cuda_backend_available():
            pytest.skip("compiled CUDA backend required")
        from comfy_kitchen.backends import cuda as cuda_backend

        h = torch.randn(shape, dtype=dtype, device="cuda") * 2.0
        exact_q, _ = _exact_fp64(h)

        chain_q, _ = cuda_backend.quantize_int8_rowwise_convrot64(
            _gelu(h).contiguous(), _GROUP
        )
        fused_q, fused_s = cuda_backend.quantize_int8_rowwise_convrot64(
            h, _GROUP, input_act="gelu_tanh"
        )

        err_chain = (chain_q.double() - exact_q).abs().mean().item()
        err_fused = (fused_q.double() - exact_q).abs().mean().item()
        assert err_fused <= max(err_chain * 1.05, 1e-4), (
            f"fused ({err_fused:.6f}) less accurate than chain ({err_chain:.6f})"
        )
        assert fused_q.shape == h.shape
        assert fused_q.dtype == torch.int8
        assert fused_s.shape == (h.shape[0], 1)

    @pytest.mark.parametrize(
        "kwargs", [{"convrot": True, "convrot_groupsize": _GROUP}, {"convrot": False}]
    )
    def test_rejects_unknown_activation(self, kwargs, cuda_available):
        """Every route must reject the same way - the fused path used to raise a
        bare KeyError while the fallbacks raised a descriptive ValueError."""
        if not cuda_available:
            pytest.skip("CUDA required")

        h = torch.randn(1024, 4096, dtype=torch.bfloat16, device="cuda")
        weight = torch.randint(-127, 127, (256, 4096), dtype=torch.int8, device="cuda")
        wscale = torch.tensor(0.01, dtype=torch.float32, device="cuda")
        with pytest.raises(ValueError, match="unsupported input_act"):
            ck.int8_linear(
                h, weight, wscale, None, torch.bfloat16, input_act="silu", **kwargs
            )


class TestInt8LinearInputAct:
    @pytest.mark.parametrize("backend", ["cuda", "triton", "eager"])
    def test_matches_eager_activation(self, backend, seed, cuda_available):
        """int8_linear(x, input_act=a) == int8_linear(a(x)) on every backend."""
        device = "cuda" if cuda_available else "cpu"
        if backend not in get_capable_backends("int8_linear", device):
            pytest.skip(f"backend '{backend}' not capable")

        m, k, n = 1024, 4096, 512
        h = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        weight = torch.randint(-127, 127, (n, k), dtype=torch.int8, device=device)
        wscale = torch.tensor(0.01, dtype=torch.float32, device=device)

        with ck.use_backend(backend):
            ref = ck.int8_linear(
                _gelu(h), weight, wscale, None, torch.bfloat16,
                convrot=True, convrot_groupsize=_GROUP,
            )
            got = ck.int8_linear(
                h, weight, wscale, None, torch.bfloat16,
                convrot=True, convrot_groupsize=_GROUP, input_act="gelu_tanh",
            )

        denom = ref.float().abs().max()
        rel = ((got.float() - ref.float()).abs().max() / denom).item()
        # Both quantize to int8; they differ only by the intermediate rounding
        # the fused path avoids.
        assert rel < 0.05, f"{backend}: rel={rel:.3e}"

    @pytest.mark.parametrize(
        "tag,shape,kwargs",
        [
            ("no convrot", (1024, 4096), {"convrot": False}),
            ("K not %256", (1024, 300), {"convrot": False}),
            ("K over smem cap", (64, 256 * 72), {"convrot": True, "convrot_groupsize": 256}),
            ("m == 1", (1, 4096), {"convrot": True, "convrot_groupsize": 256}),
        ],
    )
    def test_fallback_paths_agree(self, tag, shape, kwargs, seed, cuda_available):
        """Paths the fused kernel cannot serve must still apply the activation."""
        if not cuda_available:
            pytest.skip("CUDA required")

        m, k = shape
        h = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        weight = torch.randint(-127, 127, (256, k), dtype=torch.int8, device="cuda")
        wscale = torch.tensor(0.01, dtype=torch.float32, device="cuda")

        ref = ck.int8_linear(_gelu(h), weight, wscale, None, torch.bfloat16, **kwargs)
        got = ck.int8_linear(
            h, weight, wscale, None, torch.bfloat16, input_act="gelu_tanh", **kwargs
        )
        denom = ref.float().abs().max().clamp(min=1e-9)
        rel = ((got.float() - ref.float()).abs().max() / denom).item()
        assert rel < 0.05, f"{tag}: rel={rel:.3e}"

    def test_none_is_identity(self, seed, cuda_available):
        """input_act=None and omitting it must be bit-identical."""
        if not cuda_available:
            pytest.skip("CUDA required")

        h = torch.randn(512, 4096, dtype=torch.bfloat16, device="cuda")
        weight = torch.randint(-127, 127, (256, 4096), dtype=torch.int8, device="cuda")
        wscale = torch.tensor(0.01, dtype=torch.float32, device="cuda")

        a = ck.int8_linear(h, weight, wscale, None, torch.bfloat16,
                           convrot=True, convrot_groupsize=_GROUP)
        b = ck.int8_linear(h, weight, wscale, None, torch.bfloat16,
                           convrot=True, convrot_groupsize=_GROUP, input_act=None)
        c = ck.int8_linear(h, weight, wscale, None, torch.bfloat16,
                           convrot=True, convrot_groupsize=_GROUP, input_act="none")
        assert torch.equal(a, b)
        assert torch.equal(a, c)

    def test_3d_input(self, seed, cuda_available):
        """Batched (B, T, K) input keeps its shape and applies the activation."""
        if not cuda_available:
            pytest.skip("CUDA required")

        h = torch.randn(2, 512, 4096, dtype=torch.bfloat16, device="cuda")
        weight = torch.randint(-127, 127, (256, 4096), dtype=torch.int8, device="cuda")
        wscale = torch.tensor(0.01, dtype=torch.float32, device="cuda")

        ref = ck.int8_linear(_gelu(h), weight, wscale, None, torch.bfloat16,
                             convrot=True, convrot_groupsize=_GROUP)
        got = ck.int8_linear(h, weight, wscale, None, torch.bfloat16,
                             convrot=True, convrot_groupsize=_GROUP,
                             input_act="gelu_tanh")
        assert got.shape == (2, 512, 256)
        # Same metric as the 2D cases: an elementwise tolerance is meaningless
        # for int8 GEMM outputs, where a single-LSB difference in the quantized
        # activation shifts the whole accumulated sum for that output element.
        denom = ref.float().abs().max()
        rel = ((got.float() - ref.float()).abs().max() / denom).item()
        assert rel < 0.05, f"3d: rel={rel:.3e}"


def _swiglu(x):
    gate, up = x.chunk(2, dim=-1)
    return functional.silu(gate) * up


class TestSwiGLUInputAct:
    """swiglu is the gated pair: input rows are [gate | up], the activated row
    silu(gate) * up is half as wide, and the weight matches the halved width."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("shape", [(512, 256), (1024, 4096), (37, 14336)])
    def test_at_least_as_accurate_as_eager_chain(self, dtype, shape, seed, cuda_available):
        if not cuda_backend_available():
            pytest.skip("compiled CUDA backend required")
        from comfy_kitchen.backends import cuda as cuda_backend

        m, k = shape
        h = torch.randn((m, 2 * k), dtype=dtype, device="cuda") * 2.0

        hd = h.double()
        g = functional.silu(hd[:, :k]) * hd[:, k:]
        mat = _build_hadamard(_GROUP, device=h.device, dtype=torch.float64)
        rot = (g.reshape(-1, k // _GROUP, _GROUP) @ mat).reshape(-1, k)
        scale = (rot.abs().amax(-1, keepdim=True) / 127.0).clamp(min=1e-30)
        exact_q = (rot / scale).round().clamp(-128, 127)

        chain_q, _ = cuda_backend.quantize_int8_rowwise_convrot64(
            _swiglu(h).contiguous(), _GROUP
        )
        fused_q, fused_s = cuda_backend.quantize_int8_rowwise_convrot64(
            h, _GROUP, input_act="swiglu"
        )

        err_chain = (chain_q.double() - exact_q).abs().mean().item()
        err_fused = (fused_q.double() - exact_q).abs().mean().item()
        assert err_fused <= max(err_chain * 1.05, 1e-4), (
            f"fused ({err_fused:.6f}) less accurate than chain ({err_chain:.6f})"
        )
        assert fused_q.shape == (m, k)
        assert fused_q.dtype == torch.int8
        assert fused_s.shape == (m, 1)

    @pytest.mark.parametrize("backend", ["cuda", "triton", "eager"])
    def test_matches_eager_activation(self, backend, seed, cuda_available):
        """int8_linear(x, input_act="swiglu") == int8_linear(swiglu(x)) everywhere."""
        device = "cuda" if cuda_available else "cpu"
        if backend not in get_capable_backends("int8_linear", device):
            pytest.skip(f"backend '{backend}' not capable")

        m, k, n = 1024, 4096, 512
        h = torch.randn(m, 2 * k, dtype=torch.bfloat16, device=device)
        weight = torch.randint(-127, 127, (n, k), dtype=torch.int8, device=device)
        wscale = torch.tensor(0.01, dtype=torch.float32, device=device)

        with ck.use_backend(backend):
            ref = ck.int8_linear(
                _swiglu(h), weight, wscale, None, torch.bfloat16,
                convrot=True, convrot_groupsize=_GROUP,
            )
            got = ck.int8_linear(
                h, weight, wscale, None, torch.bfloat16,
                convrot=True, convrot_groupsize=_GROUP, input_act="swiglu",
            )

        assert got.shape == (m, n)
        denom = ref.float().abs().max()
        rel = ((got.float() - ref.float()).abs().max() / denom).item()
        assert rel < 0.05, f"{backend}: rel={rel:.3e}"

    @pytest.mark.parametrize(
        "tag,shape,kwargs",
        [
            ("no convrot", (1024, 2 * 4096), {"convrot": False}),
            ("K over smem cap", (64, 2 * 256 * 72), {"convrot": True, "convrot_groupsize": 256}),
            ("m == 1", (1, 2 * 4096), {"convrot": True, "convrot_groupsize": 256}),
        ],
    )
    def test_fallback_paths_agree(self, tag, shape, kwargs, seed, cuda_available):
        """Paths the fused kernel cannot serve must still apply the activation."""
        if not cuda_available:
            pytest.skip("CUDA required")

        m, k2 = shape
        k = k2 // 2
        h = torch.randn(m, k2, dtype=torch.bfloat16, device="cuda")
        weight = torch.randint(-127, 127, (256, k), dtype=torch.int8, device="cuda")
        wscale = torch.tensor(0.01, dtype=torch.float32, device="cuda")

        ref = ck.int8_linear(_swiglu(h), weight, wscale, None, torch.bfloat16, **kwargs)
        got = ck.int8_linear(
            h, weight, wscale, None, torch.bfloat16, input_act="swiglu", **kwargs
        )
        denom = ref.float().abs().max().clamp(min=1e-9)
        rel = ((got.float() - ref.float()).abs().max() / denom).item()
        assert rel < 0.05, f"{tag}: rel={rel:.3e}"
