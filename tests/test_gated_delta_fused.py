# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused GatedDeltaNet decode kernels against the eager stepwise chain."""

import pytest
import torch
from torch.nn import functional

import comfy_kitchen as ck
from tests.conftest import rel_err

B, HV, HK, DK, DV, HD, KS = 2, 4, 2, 128, 128, 256, 4
KEY_DIM = HK * DK
C = 2 * KEY_DIM + HV * DV
SCALE = DK ** -0.5
EPS = 1e-6


def _conv_ref(proj, conv_state, w, b, seq):
    combined = torch.cat([conv_state, proj.transpose(1, 2)], dim=-1)
    out = functional.silu(functional.conv1d(combined, w.reshape(C, 1, KS), b, groups=C))
    snaps = torch.stack([combined[:, :, 1 + s:1 + s + KS - 1] for s in range(seq - 1)]) if seq > 1 else None
    return out, combined[:, :, seq:].contiguous(), snaps


def _decode_ref(conv_out, x, w_a, w_b, dt_bias, g_decay, state, z, norm_w, seq):
    a = functional.linear(x, w_a)
    b = functional.linear(x, w_b)
    beta = b.sigmoid().reshape(B, seq, HV)
    g = (g_decay * functional.softplus(a.float() + dt_bias)).reshape(B, seq, HV).exp()
    query, key, value = conv_out.transpose(1, 2).split([KEY_DIM, KEY_DIM, HV * DV], dim=-1)
    rep = HV // HK
    q = functional.normalize(query.reshape(B, seq, HK, DK).float(), dim=-1).repeat_interleave(rep, dim=2) * SCALE
    k = functional.normalize(key.reshape(B, seq, HK, DK).float(), dim=-1).repeat_interleave(rep, dim=2)
    v = value.reshape(B, seq, HV, DV).float()
    outs, snaps = [], []
    for s in range(seq):
        state.mul_(g[:, s, :, None, None])
        kv_mem = torch.einsum("bhk,bhkv->bhv", k[:, s], state)
        delta = (v[:, s] - kv_mem) * beta[:, s, :, None]
        state.add_(torch.einsum("bhk,bhv->bhkv", k[:, s], delta))
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, s], state))
        if s < seq - 1:
            snaps.append(state.clone())
    out = torch.stack(outs, dim=1).to(x.dtype)
    out = functional.rms_norm(out.reshape(-1, DV), (DV,), norm_w, EPS) * functional.silu(z.reshape(-1, DV))
    return out.reshape(B, seq, HV, DV), (torch.stack(snaps) if snaps else None)


@pytest.mark.skipif(not ck.gated_delta_decode_is_available(), reason="fused DeltaNet decode kernels unavailable")
class TestDeltanetConvStep:
    @pytest.mark.parametrize("seq", [1, 4, 8])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_matches_conv1d(self, seq, dtype, seed):
        proj = torch.randn(B, seq, C, device="cuda", dtype=dtype)
        state = torch.randn(B, C, KS - 1, device="cuda", dtype=dtype)
        w = torch.randn(C, 1, KS, device="cuda", dtype=dtype) * 0.5
        b = torch.randn(C, device="cuda", dtype=dtype) * 0.1
        ref_out, ref_state, ref_snaps = _conv_ref(proj, state, w, b, seq)

        got_state = state.clone()
        snaps = torch.empty((seq - 1, B, C, KS - 1), device="cuda", dtype=dtype) if seq > 1 else None
        got = ck.deltanet_conv_step(proj, got_state, w, b, snaps)
        torch.cuda.synchronize()

        tol = 1e-5 if dtype == torch.float32 else 1e-2
        assert rel_err(got, ref_out) < tol
        assert torch.equal(got_state, ref_state)
        if seq > 1:
            assert torch.equal(snaps, ref_snaps)


@pytest.mark.skipif(not ck.gated_delta_decode_is_available(), reason="fused DeltaNet decode kernels unavailable")
class TestGatedDeltaDecodeFused:
    @pytest.mark.parametrize("seq", [1, 4, 8])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_matches_stepwise(self, seq, dtype, seed):
        conv_out = torch.randn(B, C, seq, device="cuda", dtype=dtype)
        x = torch.randn(B, seq, HD, device="cuda", dtype=dtype)
        w_a = torch.randn(HV, HD, device="cuda", dtype=dtype) * 0.05
        w_b = torch.randn(HV, HD, device="cuda", dtype=dtype) * 0.05
        dt_bias = torch.randn(HV, device="cuda")
        g_decay = -torch.rand(HV, device="cuda") - 0.5
        state = torch.randn(B, HV, DK, DV, device="cuda") * 0.1
        z = torch.randn(B, seq, HV * DV, device="cuda", dtype=dtype)
        norm_w = torch.rand(DV, device="cuda", dtype=dtype) + 0.5

        ref_state = state.clone()
        ref_out, ref_snaps = _decode_ref(conv_out, x, w_a, w_b, dt_bias, g_decay, ref_state, z, norm_w, seq)

        got_state = state.clone()
        snaps = torch.empty((seq - 1, B, HV, DK, DV), device="cuda") if seq > 1 else None
        got = ck.gated_delta_decode_fused(conv_out, x, w_a, w_b, dt_bias, g_decay, got_state, KEY_DIM, HK, SCALE,
                                          z, norm_w, EPS, snaps)
        torch.cuda.synchronize()

        # bf16: the gate projections are rounded to bf16 in both paths, but the dot order differs
        tol = 1e-5 if dtype == torch.float32 else 5e-3
        assert got.shape == (B, seq, HV, DV)
        assert rel_err(got.float(), ref_out.float()) < tol
        assert rel_err(got_state, ref_state) < tol
        if seq > 1:
            assert rel_err(snaps, ref_snaps) < tol

    def test_rejects_long_sequence(self):
        seq = 9
        conv_out = torch.randn(B, C, seq, device="cuda", dtype=torch.bfloat16)
        x = torch.randn(B, seq, HD, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(HV, HD, device="cuda", dtype=torch.bfloat16)
        state = torch.zeros(B, HV, DK, DV, device="cuda")
        z = torch.zeros(B, seq, HV * DV, device="cuda", dtype=torch.bfloat16)
        norm_w = torch.ones(DV, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(RuntimeError):
            ck.gated_delta_decode_fused(conv_out, x, w, w, torch.zeros(HV, device="cuda"), -torch.ones(HV, device="cuda"),
                                        state, KEY_DIM, HK, SCALE, z, norm_w, EPS)
