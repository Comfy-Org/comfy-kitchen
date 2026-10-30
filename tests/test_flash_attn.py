# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""flash_attn -- dense non-causal attention on [B, S, H, head_dim] (Wan self-attention).

The eager backend is SDPA on the head-major transpose. The Triton kernel runs an online
base-2 softmax with the scale applied to the fp32 scores, so it is compared to eager with
a tolerance, and to an fp32 reference to show the gap is the half-precision P @ V, not the
softmax.
"""

import math

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends import eager as eager_backend
from comfy_kitchen.registry import registry


def _naive(q, k, v, scale=None):
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * scale
    return torch.einsum("bhqk,bkhd->bqhd", s.softmax(-1), v.float())


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


class TestFlashAttnEager:
    @pytest.mark.parametrize("scale", [None, 0.05])
    def test_matches_naive(self, scale):
        q = torch.randn(2, 37, 3, 64)
        k = torch.randn(2, 21, 3, 64)
        v = torch.randn(2, 21, 3, 64)
        with ck.use_backend("eager"):
            out = ck.flash_attn(q, k, v, scale)
        assert out.shape == q.shape and out.is_contiguous()
        torch.testing.assert_close(out, _naive(q, k, v, scale), rtol=1e-5, atol=1e-5)

    def test_rejects_mismatched_heads(self):
        q = torch.randn(1, 8, 4, 64)
        k = torch.randn(1, 8, 2, 64)
        with pytest.raises(ValueError):
            ck.flash_attn(q, k, k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestFlashAttnTriton:
    @pytest.fixture(autouse=True)
    def _triton(self):
        pytest.importorskip("triton")
        if not registry.is_available("triton"):
            pytest.skip("triton backend not available")

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("b,sq,sk,h,d", [
        (1, 4680, 4680, 12, 128),   # Wan 1.3B self-attention, 9 x 20 x 26 latent tokens
        (2, 333, 333, 4, 64),       # S not a multiple of either block
        (1, 200, 77, 4, 128),       # S_q != S_k
    ])
    def test_matches_eager(self, dtype, b, sq, sk, h, d):
        q = torch.randn(b, sq, h, d, dtype=dtype, device="cuda")
        k = torch.randn(b, sk, h, d, dtype=dtype, device="cuda")
        v = torch.randn(b, sk, h, d, dtype=dtype, device="cuda")
        kwargs = {"q": q, "k": k, "v": v, "scale": None}
        assert registry.validate_backend_for_call("triton", "flash_attn", kwargs).success
        with ck.use_backend("triton"):
            got = ck.flash_attn(q, k, v)
        ref = eager_backend.flash_attn(q, k, v)
        ref32 = _naive(q, k, v)
        assert got.shape == ref.shape and got.dtype == dtype
        tol = 1e-2 if dtype == torch.bfloat16 else 2e-3
        assert _rel(got, ref) < tol
        assert _rel(got, ref32) < tol

    def test_strided_views_of_packed_qkv(self):
        # q, k, v as views of one [B, S, 3, H, D] buffer: per-tensor strides, unit stride on D
        qkv = torch.randn(1, 300, 3, 8, 128, dtype=torch.bfloat16, device="cuda")
        q, k, v = qkv.unbind(2)
        kwargs = {"q": q, "k": k, "v": v, "scale": None}
        assert registry.validate_backend_for_call("triton", "flash_attn", kwargs).success
        with ck.use_backend("triton"):
            got = ck.flash_attn(q, k, v, 1.0 / math.sqrt(128))
        assert _rel(got, _naive(q, k, v)) < 1e-2

    @pytest.mark.parametrize("dtype,d", [(torch.float32, 128), (torch.bfloat16, 96)])
    def test_unsupported_inputs_go_to_eager(self, dtype, d):
        q = torch.randn(1, 16, 2, d, dtype=dtype, device="cuda")
        kwargs = {"q": q, "k": q, "v": q, "scale": None}
        assert not registry.validate_backend_for_call("triton", "flash_attn", kwargs).success
        assert registry.get_capable_backend("flash_attn", kwargs) == "eager"
