# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""rms_rope(norm_over_heads=True) -- RMSNorm over the full H * head_dim row, then RoPE.

Wan normalizes q and k with an RMSNorm(dim) before splitting heads
(comfy/ldm/wan/model.py WanSelfAttention), unlike the per-head norm of rms_rope. The eager
backend is that sequence; the per-head native backends (CUDA, HIP, Ascend) must decline
the flag; the Triton kernel is compared to eager and to an fp32 reference.
"""

import pytest
import torch
from torch.nn import functional

import comfy_kitchen as ck
from comfy_kitchen.backends import eager as eager_backend
from comfy_kitchen.backends.eager.rope import apply_rope1
from comfy_kitchen.constraints import rms_rope_norm_over_heads_call_rule
from comfy_kitchen.registry import registry


def _freqs(seq, head_dim, device="cpu", batch=1):
    pos = torch.arange(seq, dtype=torch.float32, device=device)[:, None]
    omega = 1.0 / (10000 ** (torch.arange(head_dim // 2, dtype=torch.float32, device=device) * 2 / head_dim))
    ang = pos * omega
    f = torch.stack([ang.cos(), -ang.sin(), ang.sin(), ang.cos()], -1).reshape(1, seq, 1, head_dim // 2, 2, 2)
    return f.expand(batch, -1, -1, -1, -1, -1).contiguous()


def _make(b, s, h, d, dtype, device="cpu"):
    q = torch.randn(b, s, h * d, dtype=dtype, device=device)
    k = torch.randn(b, s, h * d, dtype=dtype, device=device)
    qs = (torch.rand(h * d, device=device) + 0.5).to(dtype)
    ks = (torch.rand(h * d, device=device) + 0.5).to(dtype)
    return q, k, qs, ks


def _wan_reference(x, scale, freqs, h, d, eps=1e-6):
    # what WanSelfAttention does today: RMSNorm(dim) on [B, S, dim], view, RoPE
    b, s, _ = x.shape
    return apply_rope1(functional.rms_norm(x, (h * d,), scale, eps).view(b, s, h, d), freqs)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


class TestNormOverHeadsEager:
    def test_matches_wan_sequence(self):
        b, s, h, d = 2, 19, 4, 32
        q, k, qs, ks = _make(b, s, h, d, torch.float32)
        f = _freqs(s, d)
        with ck.use_backend("eager"):
            qo, ko = ck.rms_rope(q.view(b, s, h, d), k.view(b, s, h, d), f, qs, ks, 1e-6,
                                 norm_over_heads=True)
        torch.testing.assert_close(qo, _wan_reference(q, qs, f, h, d), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(ko, _wan_reference(k, ks, f, h, d), rtol=1e-5, atol=1e-6)

    def test_differs_from_per_head(self):
        b, s, h, d = 1, 8, 4, 32
        q, k, qs, _ = _make(b, s, h, d, torch.float32)
        f = _freqs(s, d)
        q4, k4 = q.view(b, s, h, d), k.view(b, s, h, d)
        with ck.use_backend("eager"):
            full, _ = ck.rms_rope(q4, k4, f, qs, None, 1e-6, norm_over_heads=True)
            head, _ = ck.rms_rope(q4, k4, f, qs[:d], None, 1e-6)
        assert not torch.allclose(full, head)

    def test_rejects_per_head_scale(self):
        q = torch.randn(1, 8, 4, 32)
        with pytest.raises(ValueError):
            ck.rms_rope(q, q, _freqs(8, 32), torch.ones(32), None, 1e-6, norm_over_heads=True)

    def test_per_head_backends_decline(self):
        q = torch.randn(1, 8, 4, 32, dtype=torch.bfloat16)
        kwargs = {"q": q, "k": q, "freqs_cis": _freqs(8, 32), "q_scale": torch.ones(128),
                  "k_scale": None, "epsilon": 1e-6, "norm_over_heads": True}
        assert not rms_rope_norm_over_heads_call_rule(kwargs).success
        assert rms_rope_norm_over_heads_call_rule({**kwargs, "norm_over_heads": False}).success
        for name in ("ascend", "cuda", "hip"):
            rules = registry.get_constraints(name, "rms_rope")
            if rules is not None:
                assert rms_rope_norm_over_heads_call_rule in rules.call_rules
                assert not registry.validate_backend_for_call(name, "rms_rope", kwargs).success

    def test_native_constraint_tables_carry_the_rule(self):
        # checked on the tables themselves, so it holds where the extensions are not built
        from comfy_kitchen.backends import cuda as cuda_backend
        tables = [cuda_backend._build_constraints()]
        if getattr(torch.version, "hip", None):
            from comfy_kitchen.backends import hip as hip_backend
            tables.append(hip_backend._build_constraints())
        for table in tables:
            for name in ("rms_rope", "rms_rope_"):
                assert rms_rope_norm_over_heads_call_rule in table[name].call_rules


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestNormOverHeadsTriton:
    @pytest.fixture(autouse=True)
    def _triton(self):
        pytest.importorskip("triton")
        if not registry.is_available("triton"):
            pytest.skip("triton backend not available")

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("b,s,h,d", [
        (1, 1760, 24, 128),   # Wan2.2 5B, 2 x 22 x 40 latent tokens
        (1, 515, 40, 128),    # Wan 14B width (5120)
        (2, 333, 12, 128),    # batch 2 sharing one freqs
        (1, 100, 4, 64),
    ])
    def test_matches_eager(self, dtype, b, s, h, d):
        q, k, qs, ks = _make(b, s, h, d, dtype, "cuda")
        f = _freqs(s, d, "cuda")
        q4, k4 = q.view(b, s, h, d), k.view(b, s, h, d)
        kwargs = {"q": q4, "k": k4, "freqs_cis": f, "q_scale": qs, "k_scale": ks,
                  "epsilon": 1e-6, "norm_over_heads": True}
        assert registry.validate_backend_for_call("triton", "rms_rope", kwargs).success
        with ck.use_backend("triton"):
            got = ck.rms_rope(q4, k4, f, qs, ks, 1e-6, norm_over_heads=True)
        ref = eager_backend.rms_rope(q4, k4, f, qs, ks, 1e-6, norm_over_heads=True)
        ref32 = (_wan_reference(q.float(), qs.float(), f, h, d), _wan_reference(k.float(), ks.float(), f, h, d))
        tol = 1e-2 if dtype == torch.bfloat16 else 2e-3
        for g, r, r32 in zip(got, ref, ref32, strict=True):
            assert g.shape == r.shape and g.dtype == dtype
            assert _rel(g, r) < tol
            assert _rel(g, r32) < tol

    def test_freqs_broadcast_over_seq(self):
        b, s, h, d = 1, 64, 4, 128
        q, k, qs, ks = _make(b, s, h, d, torch.bfloat16, "cuda")
        f = _freqs(1, d, "cuda")
        q4, k4 = q.view(b, s, h, d), k.view(b, s, h, d)
        with ck.use_backend("triton"):
            got = ck.rms_rope(q4, k4, f, qs, ks, 1e-6, norm_over_heads=True)
        ref = eager_backend.rms_rope(q4, k4, f, qs, ks, 1e-6, norm_over_heads=True)
        for g, r in zip(got, ref, strict=True):
            assert _rel(g, r) < 1e-2

    def test_unsupported_inputs_go_to_eager(self):
        b, s, h, d = 2, 16, 4, 128
        q, k, qs, ks = _make(b, s, h, d, torch.bfloat16, "cuda")
        q4, k4 = q.view(b, s, h, d), k.view(b, s, h, d)
        cases = [
            (q4.float(), k4.float(), _freqs(s, d, "cuda")),            # fp32
            (q4.transpose(1, 2), k4.transpose(1, 2), _freqs(s, d, "cuda")),  # not [B, S, H, D] contiguous
            (q4, k4, _freqs(s, d, "cuda", batch=2)),                   # per-batch freqs
        ]
        for qq, kk, ff in cases:
            kwargs = {"q": qq, "k": kk, "freqs_cis": ff, "q_scale": qs, "k_scale": ks,
                      "epsilon": 1e-6, "norm_over_heads": True}
            assert not registry.validate_backend_for_call("triton", "rms_rope", kwargs).success
            assert registry.get_capable_backend("rms_rope", kwargs) == "eager"
