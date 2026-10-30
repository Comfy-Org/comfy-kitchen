# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""linear_qk_rms_norm -- QKV projection with per-head q/k RMSNorm (MMDiT pre-attention).

The eager backend is F.linear -> split -> F.rms_norm, the sequence ComfyUI's mmdit
SelfAttention.pre_attention runs today. The Triton kernel accumulates the GEMM in fp32
with its own K order, so it is compared to eager with a tolerance, and to an fp32 reference
to show the gap is GEMM rounding, not the norm.
"""

import pytest
import torch
from torch.nn import functional

import comfy_kitchen as ck
from comfy_kitchen.backends import eager as eager_backend


def _make(b, s, k, heads, head_dim, dtype, bias, device="cuda"):
    dim = heads * head_dim
    x = torch.randn(b, s, k, dtype=dtype, device=device)
    w = torch.randn(3 * dim, k, dtype=dtype, device=device) * k ** -0.5
    bb = torch.randn(3 * dim, dtype=dtype, device=device) * 0.1 if bias else None
    qs = torch.rand(head_dim, dtype=dtype, device=device) + 0.5
    ks = torch.rand(head_dim, dtype=dtype, device=device) + 0.5
    return x, w, bb, qs, ks


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


class TestLinearQkRmsNormEager:
    @pytest.mark.parametrize("bias", [True, False])
    def test_matches_split_and_rms_norm(self, bias):
        x, w, bb, qs, ks = _make(2, 37, 128, 4, 64, torch.float32, bias, device="cpu")
        q, k, v = ck.linear_qk_rms_norm(x, w, bb, qs, ks, 64, 1e-6)
        y = functional.linear(x, w, bb).view(2, 37, 3, 4, 64)
        assert torch.equal(q, functional.rms_norm(y[:, :, 0], (64,), qs, 1e-6))
        assert torch.equal(k, functional.rms_norm(y[:, :, 1], (64,), ks, 1e-6))
        assert torch.equal(v, y[:, :, 2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestLinearQkRmsNormTriton:
    @pytest.fixture(autouse=True)
    def _triton(self):
        pytest.importorskip("triton")

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("b,s,k,heads,head_dim", [
        (2, 4096, 2432, 38, 64),   # SD3.5 Large image stream, 1024x1024
        (2, 154, 2432, 38, 64),    # SD3.5 Large context stream
        (1, 37, 256, 4, 64),       # M not a multiple of any block_m
        (1, 100, 384, 2, 128),
    ])
    def test_matches_eager(self, dtype, bias, b, s, k, heads, head_dim):
        x, w, bb, qs, ks = _make(b, s, k, heads, head_dim, dtype, bias)
        with ck.use_backend("triton"):
            got = ck.linear_qk_rms_norm(x, w, bb, qs, ks, head_dim)
        ref = eager_backend.linear_qk_rms_norm(x, w, bb, qs, ks, head_dim, 1e-6)
        ref32 = eager_backend.linear_qk_rms_norm(x.float(), w.float(), None if bb is None else bb.float(),
                                                 qs.float(), ks.float(), head_dim, 1e-6)
        tol = 1e-2 if dtype == torch.bfloat16 else 2e-3
        for g, r, r32 in zip(got, ref, ref32, strict=True):
            assert g.shape == r.shape and g.dtype == r.dtype
            assert _rel(g, r) < tol
            assert _rel(g, r32) < tol
