# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apple Metal / MPS backend tests."""

from __future__ import annotations

import os
import time

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends import metal as metal_backend
from comfy_kitchen.registry import registry

pytestmark = pytest.mark.skipif(
    not (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and (hasattr(torch, "_int_mm") or hasattr(torch, "_weight_int8pack_mm"))
    ),
    reason="MPS with int8 matmul required",
)

# Strict speedups need a quiet GPU; suite runs are too noisy on shared MPS.
_STRICT_PERF = os.environ.get("COMFY_KITCHEN_METAL_PERF", "") == "1"


@pytest.fixture
def mps_device():
    return torch.device("mps")


def _dequant_linear(x, weight_int8, scale, bias, out_dtype):
    s = scale.reshape(-1).to(device=x.device, dtype=torch.float32)
    w_f = (weight_int8.to(dtype=torch.float32) * s[:, None]).to(dtype=x.dtype)
    out = torch.nn.functional.linear(x, w_f, bias)
    return out.to(dtype=out_dtype)


def test_metal_backend_listed(mps_device):
    backends = ck.list_backends()
    assert "metal" in backends
    assert backends["metal"]["available"]
    assert "int8_linear" in backends["metal"]["capabilities"]


def test_metal_selected_for_int8_linear(mps_device):
    m, n, k = 64, 256, 256
    x = torch.randn(m, k, device=mps_device, dtype=torch.float16)
    weight = torch.randint(-64, 64, (n, k), device=mps_device, dtype=torch.int8)
    scale = torch.rand(n, device=mps_device, dtype=torch.float32) * 0.02 + 0.001
    call = {
        "x": x,
        "weight": weight,
        "weight_scale": scale,
        "bias": None,
        "out_dtype": torch.float16,
        "convrot": False,
        "convrot_groupsize": 256,
        "input_act": None,
        "input_act_weight": None,
        "input_act_eps": 0.0,
        "residual": None,
        "residual_scale": None,
    }
    assert registry.get_capable_backend("int8_linear", call) == "metal"


def test_metal_accepts_non_aligned_w8a8_shapes(mps_device):
    x = torch.randn(16, 64, device=mps_device, dtype=torch.float16)
    weight = torch.randint(-32, 32, (17, 64), device=mps_device, dtype=torch.int8)
    scale = torch.ones(17, device=mps_device, dtype=torch.float32)
    call = {
        "x": x,
        "weight": weight,
        "weight_scale": scale,
        "bias": None,
        "out_dtype": torch.float16,
        "convrot": False,
        "convrot_groupsize": 256,
        "input_act": None,
        "input_act_weight": None,
        "input_act_eps": 0.0,
        "residual": None,
        "residual_scale": None,
    }
    assert registry.get_capable_backend("int8_linear", call) == "metal"


@pytest.mark.parametrize(
    "x_shape",
    [
        (32, 256),
        (2, 3, 8, 256),
    ],
)
def test_int8_linear_matches_eager_w8a8(mps_device, x_shape):
    from comfy_kitchen.backends.eager.quantization import int8_linear as eager_int8_linear

    n, k = 128, 256
    x = torch.randn(*x_shape, device=mps_device, dtype=torch.float16)
    weight = torch.randint(-64, 64, (n, k), device=mps_device, dtype=torch.int8)
    scale = torch.rand(n, device=mps_device, dtype=torch.float32) * 0.02 + 0.001
    bias = torch.randn(n, device=mps_device, dtype=torch.float16)

    with ck.use_backend("metal"):
        actual = ck.int8_linear(x, weight, scale, bias, out_dtype=torch.float16)
    expected = eager_int8_linear(x, weight, scale, bias, out_dtype=torch.float16)

    assert actual.device.type == "mps"
    assert actual.dtype == torch.float16
    assert actual.shape == (*x_shape[:-1], n)
    # W8A8 paths can differ by a few ULPs when row-scale rounding ties disagree.
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=2e-1)


def test_int8_linear_direct_backend(mps_device):
    x = torch.randn(16, 128, device=mps_device, dtype=torch.float16)
    weight = torch.randint(-40, 40, (64, 128), device=mps_device, dtype=torch.int8)
    scale = torch.ones(64, device=mps_device, dtype=torch.float32) * 0.01
    out = metal_backend.int8_linear(x, weight, scale, None, out_dtype=torch.float16)
    assert out.shape == (16, 64)


def test_int8_linear_faster_than_dequant(mps_device):
    m, n, k = 256, 4096, 4096
    x = torch.randn(m, k, device=mps_device, dtype=torch.float16).contiguous()
    weight = torch.randint(-64, 64, (n, k), device=mps_device, dtype=torch.int8).contiguous()
    scale = (torch.rand(n, device=mps_device, dtype=torch.float32) * 0.02 + 0.001).contiguous()
    bias = torch.randn(n, device=mps_device, dtype=torch.float16)

    for _ in range(5):
        _ = metal_backend.int8_linear(x, weight, scale, bias, out_dtype=torch.float16)
        _ = _dequant_linear(x, weight, scale, bias, torch.float16)
    torch.mps.synchronize()

    t0 = time.perf_counter()
    for _ in range(20):
        _ = metal_backend.int8_linear(x, weight, scale, bias, out_dtype=torch.float16)
    torch.mps.synchronize()
    metal_ms = (time.perf_counter() - t0) * 1000 / 20

    t0 = time.perf_counter()
    for _ in range(20):
        _ = _dequant_linear(x, weight, scale, bias, torch.float16)
    torch.mps.synchronize()
    dequant_ms = (time.perf_counter() - t0) * 1000 / 20

    if not _STRICT_PERF:
        pytest.skip("set COMFY_KITCHEN_METAL_PERF=1 for strict int8_linear speed check")
    speedup = dequant_ms / metal_ms
    assert speedup >= 1.3, (
        f"metal int8_linear not faster enough: {speedup:.2f}x ({metal_ms:.2f} vs {dequant_ms:.2f} ms)"
    )


def _rotation_freqs(T: int, pairs: int, device, dtype=torch.float32):
    angle = torch.randn(T, pairs, device=device, dtype=dtype)
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack(
        [torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)],
        dim=-2,
    ).contiguous()


@pytest.mark.skipif(
    metal_backend._kernels() is None,
    reason="metal custom kernels unavailable",
)
class TestMetalKernels:
    def test_metal_lists_kernel_ops(self, mps_device):
        caps = ck.list_backends()["metal"]["capabilities"]
        for name in (
            "apply_rope1",
            "apply_rope_split_half1",
            "rms_rope1",
            "adaln",
            "rms_adaln",
            "quantize_int8_rowwise",
            "quantize_int8_tensorwise",
            "dequantize_int8_simple",
            "dequantize_int8_simple_dtype",
            "stochastic_rounding_fp8",
            "na3d",
            "window_attention",
            "quantize_and_rotate_rowwise",
            "quantize_int8_convrot_weight",
            "dequantize_int8_convrot_weight_dtype",
            "fp16_linear",
            "rms_gated_residual",
            "convrot_w4a4_linear",
        ):
            assert name in caps

    def test_apply_rope1_matches_eager(self, mps_device):
        from comfy_kitchen.backends.eager.rope import apply_rope1 as eager_rope1

        B, H, T, D = 2, 8, 32, 64
        x = torch.randn(B, H, T, D, device=mps_device, dtype=torch.float16)
        freqs = _rotation_freqs(T, D // 2, mps_device)
        with ck.use_backend("metal"):
            actual = ck.apply_rope1(x, freqs)
        expected = eager_rope1(x, freqs)
        torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-2, atol=1e-2)

    def test_apply_rope_split_half1_matches_eager(self, mps_device):
        from comfy_kitchen.backends.eager.rope import apply_rope_split_half1 as eager_split

        B, H, T, D = 2, 8, 32, 64
        x = torch.randn(B, H, T, D, device=mps_device, dtype=torch.float16)
        freqs = _rotation_freqs(T, D // 2, mps_device)
        with ck.use_backend("metal"):
            actual = ck.apply_rope_split_half1(x, freqs)
        expected = eager_split(x, freqs)
        torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-2, atol=1e-2)

    def test_apply_rope_split_half1_ltx_per_head_bnhd(self, mps_device):
        """LTX embeddings connector: BNHD x + per-head 6D freqs (no singleton).

        Must index BNHD via strides — no Python permute/contiguous tax.
        """
        from comfy_kitchen.backends.eager.rope import apply_rope_split_half1 as eager_split

        B, T, H, D = 1, 128, 32, 64
        x = torch.randn(B, T, H, D, device=mps_device, dtype=torch.float16)
        # freqs_cis_matrix layout: (B, T, H, pairs, 2, 2)
        angle = torch.randn(B, T, H, D // 2, device=mps_device, dtype=torch.float32)
        c, s = torch.cos(angle), torch.sin(angle)
        freqs = torch.stack(
            [torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)],
            dim=-2,
        ).contiguous()
        with ck.use_backend("metal"):
            actual = ck.apply_rope_split_half1(x, freqs)
            x_in = x.clone()
            ptr = x_in.data_ptr()
            inplace = ck.apply_rope_split_half1_(x_in, freqs)
        expected = eager_split(x, freqs)
        assert actual.shape == x.shape
        assert actual.stride() == x.stride()
        assert inplace is x_in and inplace.data_ptr() == ptr
        torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(inplace.float(), expected.float(), rtol=1e-2, atol=1e-2)

    def test_rms_rope1_matches_eager(self, mps_device):
        from comfy_kitchen.backends.eager.rope import rms_rope1 as eager_rms_rope1

        B, H, T, D = 2, 8, 32, 64
        x = torch.randn(B, H, T, D, device=mps_device, dtype=torch.float16)
        freqs = _rotation_freqs(T, D // 2, mps_device)
        scale = torch.randn(D, device=mps_device, dtype=torch.float16)
        with ck.use_backend("metal"):
            actual = ck.rms_rope1(x, freqs, scale)
        expected = eager_rms_rope1(x, freqs, scale)
        torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-2, atol=2e-2)

    def test_rms_rope_inplace_ltx_vae_bnhd_view(self, mps_device):
        """LTX NA decoder: inplace rms_rope_ on BNHD views into (B,T,H,W,heads,D)."""
        from comfy_kitchen.backends.eager.rope import rms_rope_ as eager_rms_rope_

        B, T, Hs, Ws, heads, D = 1, 8, 4, 4, 16, 64
        nt = T * Hs * Ws
        q = torch.randn(B, T, Hs, Ws, heads, D, device=mps_device, dtype=torch.float16)
        k = torch.randn(B, T, Hs, Ws, heads, D, device=mps_device, dtype=torch.float16)
        angle = torch.randn(1, nt, 1, D // 2, device=mps_device, dtype=torch.float32)
        c, s = torch.cos(angle), torch.sin(angle)
        freqs = torch.stack(
            [torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)],
            dim=-2,
        ).contiguous()
        qw = torch.ones(D, device=mps_device, dtype=torch.float16)
        kw = torch.ones(D, device=mps_device, dtype=torch.float16)
        q_m, k_m = q.clone(), k.clone()
        q_e, k_e = q.clone(), k.clone()
        qv = q_m[0].view(1, nt, heads, D)
        kv = k_m[0].view(1, nt, heads, D)
        assert qv.is_contiguous()
        ptr_q, ptr_k = qv.data_ptr(), kv.data_ptr()
        with ck.use_backend("metal"):
            ck.rms_rope_(qv, kv, freqs, qw, kw)
        assert qv.data_ptr() == ptr_q and kv.data_ptr() == ptr_k
        eager_rms_rope_(
            q_e[0].view(1, nt, heads, D),
            k_e[0].view(1, nt, heads, D),
            freqs,
            qw,
            kw,
        )
        torch.testing.assert_close(q_m.float(), q_e.float(), rtol=1e-2, atol=2e-2)
        torch.testing.assert_close(k_m.float(), k_e.float(), rtol=1e-2, atol=2e-2)

    def test_rms_rope_fused_beats_eager_at_dit_shapes(self, mps_device):
        from comfy_kitchen.backends.eager.rope import rms_rope1 as eager_rms_rope1

        if not _STRICT_PERF:
            pytest.skip("set COMFY_KITCHEN_METAL_PERF=1 for strict DiT-shape speed check")
        B, H, T, D = 1, 24, 2048, 128
        x = torch.randn(B, H, T, D, device=mps_device, dtype=torch.float16).contiguous()
        freqs = _rotation_freqs(T, D // 2, mps_device)
        scale = torch.ones(D, device=mps_device, dtype=torch.float16)
        torch.mps.empty_cache()
        for _ in range(15):
            _ = metal_backend.rms_rope1(x, freqs, scale)
            _ = eager_rms_rope1(x, freqs, scale)
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            _ = metal_backend.rms_rope1(x, freqs, scale)
        torch.mps.synchronize()
        metal_ms = (time.perf_counter() - t0) * 1000 / 30
        t0 = time.perf_counter()
        for _ in range(30):
            _ = eager_rms_rope1(x, freqs, scale)
        torch.mps.synchronize()
        eager_ms = (time.perf_counter() - t0) * 1000 / 30
        assert eager_ms / metal_ms >= 1.3, f"fused rms_rope not faster: {eager_ms/metal_ms:.2f}x"

    def test_rms_rope_flux_6d_freqs(self, mps_device):
        from comfy_kitchen.backends.eager.rope import rms_rope1 as eager_rms_rope1

        B, H, T, D = 1, 4, 64, 64
        x = torch.randn(B, H, T, D, device=mps_device, dtype=torch.float16).contiguous()
        pairs = D // 2
        angle = torch.randn(B, 1, T, pairs, device=mps_device, dtype=torch.float32)
        c, s = torch.cos(angle), torch.sin(angle)
        freqs = torch.stack(
            [torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)], dim=-2
        )
        scale = torch.ones(D, device=mps_device, dtype=torch.float16)
        with ck.use_backend("metal"):
            actual = ck.rms_rope1(x, freqs, scale)
        expected = eager_rms_rope1(x, freqs, scale)
        torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-2, atol=2e-2)

    def test_fp8_per_tensor_roundtrip(self, mps_device):
        from comfy_kitchen.backends.eager.quantization import (
            dequantize_per_tensor_fp8 as eager_dequant,
            quantize_per_tensor_fp8 as eager_quant,
        )

        x = torch.randn(128, 256, device=mps_device, dtype=torch.float16)
        scale = torch.tensor(0.02, device=mps_device, dtype=torch.float32)
        with ck.use_backend("metal"):
            q = ck.quantize_per_tensor_fp8(x, scale, torch.float8_e4m3fn)
        q_ref = eager_quant(x, scale, torch.float8_e4m3fn)
        mismatch = (q.view(torch.uint8).to(torch.int16) - q_ref.view(torch.uint8).to(torch.int16)).abs()
        assert mismatch.max().item() <= 1
        with ck.use_backend("metal"):
            dq = ck.dequantize_per_tensor_fp8(q, scale, torch.float16)
        torch.testing.assert_close(
            dq, eager_dequant(q, scale, torch.float16), rtol=1e-2, atol=2e-2
        )

    def test_svdquant_w4a4_matches_eager(self, mps_device):
        from comfy_kitchen.backends.eager.svdquant import (
            quantize_svdquant_w4a4 as eager_q,
            scaled_mm_svdquant_w4a4 as eager_mm,
        )

        m, n, k, r = 64, 128, 256, 16
        x = torch.randn(m, k, device=mps_device, dtype=torch.float16)
        smooth = torch.rand(k, device=mps_device, dtype=torch.float16) * 0.5 + 0.5
        lora_down = torch.randn(k, r, device=mps_device, dtype=torch.float16) * 0.01
        wgt = torch.randint(-7, 8, (n, k // 2), device=mps_device, dtype=torch.int8)
        wscales = torch.rand(k // 64, n, device=mps_device, dtype=torch.float16) * 0.02 + 0.001
        lora_up = torch.randn(n, r, device=mps_device, dtype=torch.float16) * 0.01
        bias = torch.randn(n, device=mps_device, dtype=torch.float16)

        with ck.use_backend("metal"):
            q, ascales, lora_act = ck.quantize_svdquant_w4a4(x, smooth, lora_down, pad_size=64)
            out = ck.scaled_mm_svdquant_w4a4(
                q, wgt, ascales, wscales, lora_act, lora_up, bias=bias
            )
        q_e, ascales_e, lora_e = eager_q(x, smooth, lora_down, pad_size=64)
        out_e = eager_mm(q_e, wgt, ascales_e, wscales, lora_e, lora_up, bias=bias)

        from comfy_kitchen.backends.eager.svdquant import _unpack_int4_row_major

        # Packed bytes can differ by 16 when a single high nibble rounds ±1.
        u_m = _unpack_int4_row_major(q)
        u_e = _unpack_int4_row_major(q_e)
        assert (u_m.to(torch.int32) - u_e.to(torch.int32)).abs().max().item() <= 1
        torch.testing.assert_close(ascales, ascales_e, rtol=1e-2, atol=2e-2)
        torch.testing.assert_close(out.float(), out_e.float(), rtol=8e-2, atol=2e-1)

    def test_svdquant_faster_than_eager(self, mps_device):
        from comfy_kitchen.backends.eager.svdquant import scaled_mm_svdquant_w4a4 as eager_mm

        m, n, k, r = 256, 2048, 2048, 32
        act = torch.randint(-7, 8, (m, k // 2), device=mps_device, dtype=torch.int8)
        wgt = torch.randint(-7, 8, (n, k // 2), device=mps_device, dtype=torch.int8)
        ascales = torch.rand(k // 64, m, device=mps_device, dtype=torch.float16) * 0.02 + 0.001
        wscales = torch.rand(k // 64, n, device=mps_device, dtype=torch.float16) * 0.02 + 0.001
        lora_act = torch.randn(m, r, device=mps_device, dtype=torch.float32)
        lora_up = torch.randn(n, r, device=mps_device, dtype=torch.float16) * 0.01
        for _ in range(3):
            _ = metal_backend.scaled_mm_svdquant_w4a4(
                act, wgt, ascales, wscales, lora_act, lora_up
            )
            _ = eager_mm(act, wgt, ascales, wscales, lora_act, lora_up)
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            _ = metal_backend.scaled_mm_svdquant_w4a4(
                act, wgt, ascales, wscales, lora_act, lora_up
            )
        torch.mps.synchronize()
        metal_ms = (time.perf_counter() - t0) * 1000 / 10
        t0 = time.perf_counter()
        for _ in range(10):
            _ = eager_mm(act, wgt, ascales, wscales, lora_act, lora_up)
        torch.mps.synchronize()
        eager_ms = (time.perf_counter() - t0) * 1000 / 10
        if not _STRICT_PERF:
            pytest.skip("set COMFY_KITCHEN_METAL_PERF=1 for strict svdquant speed check")
        assert eager_ms / metal_ms >= 1.2, f"svdquant metal not faster: {eager_ms/metal_ms:.2f}x"

    @pytest.mark.parametrize("op", ["adaln", "rms_adaln"])
    def test_adaln_matches_eager(self, mps_device, op):
        from comfy_kitchen.backends.eager.adaln import adaln as eager_adaln
        from comfy_kitchen.backends.eager.adaln import rms_adaln as eager_rms_adaln

        x = torch.randn(4, 64, 256, device=mps_device, dtype=torch.float16)
        scale = torch.randn(1, 1, 256, device=mps_device, dtype=torch.float16)
        shift = torch.randn(1, 1, 256, device=mps_device, dtype=torch.float16)
        eager_fn = eager_adaln if op == "adaln" else eager_rms_adaln
        with ck.use_backend("metal"):
            actual = getattr(ck, op)(x, scale, shift)
        expected = eager_fn(x, scale, shift)
        torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-2, atol=2e-2)

    def test_quantize_int8_rowwise_matches_eager(self, mps_device):
        from comfy_kitchen.backends.eager.quantization import (
            dequantize_int8_simple as eager_dequant,
            quantize_int8_rowwise as eager_quant,
        )

        x = torch.randn(64, 512, device=mps_device, dtype=torch.float16)
        with ck.use_backend("metal"):
            q, scale = ck.quantize_int8_rowwise(x)
        q_ref, scale_ref = eager_quant(x)
        torch.testing.assert_close(scale, scale_ref, rtol=1e-5, atol=1e-5)
        assert (q.to(torch.int32) - q_ref.to(torch.int32)).abs().max().item() <= 1
        with ck.use_backend("metal"):
            dq = ck.dequantize_int8_simple(q, scale)
        torch.testing.assert_close(dq, eager_dequant(q, scale), rtol=1e-5, atol=1e-5)

    def test_quantize_int8_tensorwise_matches_eager(self, mps_device):
        from comfy_kitchen.backends.eager.quantization import (
            dequantize_int8_simple_dtype as eager_dequant_dtype,
            quantize_int8_tensorwise as eager_quant,
        )
        from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE

        x = torch.randn(32, 256, device=mps_device, dtype=torch.float16)
        with ck.use_backend("metal"):
            q, scale = ck.quantize_int8_tensorwise(x)
        q_ref, scale_ref = eager_quant(x)
        torch.testing.assert_close(scale, scale_ref, rtol=1e-5, atol=1e-5)
        assert (q.to(torch.int32) - q_ref.to(torch.int32)).abs().max().item() <= 1
        with ck.use_backend("metal"):
            dq = torch.ops.comfy_kitchen.dequantize_int8_simple_dtype(
                q, scale, DTYPE_TO_CODE[torch.float16]
            )
        torch.testing.assert_close(
            dq,
            eager_dequant_dtype(q, scale, DTYPE_TO_CODE[torch.float16]),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_stochastic_rounding_fp8_matches_eager(self, mps_device):
        from comfy_kitchen.backends.eager.quantization import (
            stochastic_rounding_fp8 as eager_sr,
        )

        x = torch.randn(64, 128, device=mps_device, dtype=torch.float16)
        rng = torch.randint(0, 256, x.shape, device=mps_device, dtype=torch.uint8)
        with ck.use_backend("metal"):
            q = ck.stochastic_rounding_fp8(x.clone(), rng.clone(), torch.float8_e4m3fn)
        ref = eager_sr(x.clone(), rng.clone(), torch.float8_e4m3fn)
        mismatch = (
            q.view(torch.uint8).to(torch.int16) - ref.view(torch.uint8).to(torch.int16)
        ).abs()
        assert mismatch.max().item() <= 1

    def test_na3d_matches_eager(self, mps_device):
        from comfy_kitchen.backends.eager.na import na3d as eager_na3d

        B, T, H, W, NH, HD = 1, 4, 8, 8, 2, 16
        q = torch.randn(B, T, H, W, NH, HD, device=mps_device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        kernel = [3, 3, 3]
        causal = [False, False, False]
        with ck.use_backend("metal"):
            actual = ck.na3d(q, k, v, kernel, causal, None)
        expected = eager_na3d(q, k, v, kernel, causal, None)
        torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)
        assert registry.get_capable_backend(
            "na3d",
            {"q": q, "k": k, "v": v, "kernel_size": kernel, "is_causal": causal, "scale": None},
        ) == "metal"

    def test_window_attention_matches_eager(self, mps_device):
        from comfy_kitchen.backends.eager.na import window_attention as eager_wa

        B, T, H, W, NH, HD = 1, 2, 8, 8, 2, 16
        q = torch.randn(B, T, H, W, NH, HD, device=mps_device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        with ck.use_backend("metal"):
            actual = ck.window_attention(q, k, v, (5, 5), temporal_window=2, clamp=True)
        expected = eager_wa(q, k, v, [5], [5], 2, True, None)
        torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)

    def test_quantize_and_rotate_rowwise_matches_eager(self, mps_device):
        from comfy_kitchen.backends.eager.quantization import (
            quantize_and_rotate_rowwise as eager_qar,
        )
        from comfy_kitchen.tensor.int8_utils import _build_hadamard

        group = 64
        x = torch.randn(32, 256, device=mps_device, dtype=torch.float16)
        h = _build_hadamard(group, device=mps_device, dtype=torch.float16)
        q, scale = metal_backend.quantize_and_rotate_rowwise(x, h, group)
        q_e, scale_e = eager_qar(x, h, group)
        torch.testing.assert_close(scale, scale_e, rtol=1e-5, atol=1e-5)
        assert (q.to(torch.int32) - q_e.to(torch.int32)).abs().max().item() <= 1

    def test_fp16_linear_matches_eager(self, mps_device):
        from comfy_kitchen.backends.eager.quantization import fp16_linear as eager_fp16

        x = torch.randn(16, 128, device=mps_device, dtype=torch.float16)
        w = torch.randn(64, 128, device=mps_device, dtype=torch.float16)
        b = torch.randn(64, device=mps_device, dtype=torch.float16)
        resid = torch.randn(16, 64, device=mps_device, dtype=torch.float16)
        rscale = torch.ones(64, device=mps_device, dtype=torch.float16)
        with ck.use_backend("metal"):
            actual = ck.fp16_linear(x, w, b, residual=resid, residual_scale=rscale)
        expected = eager_fp16(x, w, b, residual=resid, residual_scale=rscale)
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_rms_gated_residual_matches_eager(self, mps_device):
        from comfy_kitchen.backends.eager.residual import rms_gated_residual as eager_rgr

        act = torch.randn(2, 32, 128, device=mps_device, dtype=torch.float16)
        weight = torch.randn(128, device=mps_device, dtype=torch.float16)
        resid = torch.randn_like(act)
        gate = torch.randn(128, device=mps_device, dtype=torch.float16)
        with ck.use_backend("metal"):
            actual = ck.rms_gated_residual(act, weight, resid, gate)
        expected = eager_rgr(act, weight, resid, gate)
        torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)

    def test_convrot_weight_roundtrip(self, mps_device):
        from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE
        from comfy_kitchen.backends.eager.quantization import (
            dequantize_int8_convrot_weight_dtype as eager_deq,
            quantize_int8_convrot_weight as eager_q,
        )

        group = 64
        w = torch.randn(128, 256, device=mps_device, dtype=torch.float16)
        q, scale = metal_backend.quantize_int8_convrot_weight(w, group)
        q_e, scale_e = eager_q(w, group)
        torch.testing.assert_close(scale, scale_e, rtol=1e-5, atol=1e-5)
        assert (q.to(torch.int32) - q_e.to(torch.int32)).abs().max().item() <= 1
        code = DTYPE_TO_CODE[torch.float16]
        dq = metal_backend.dequantize_int8_convrot_weight_dtype(q, scale, group, code)
        dq_e = eager_deq(q, scale, group, code)
        torch.testing.assert_close(dq.float(), dq_e.float(), rtol=2e-2, atol=2e-2)

    def test_convrot_w4a4_linear_matches_eager(self, mps_device):
        from comfy_kitchen.backends.eager.convrot_w4a4 import (
            convrot_w4a4_linear as eager_linear,
            quantize_convrot_w4a4_weight as eager_q,
        )

        m, n, k, g = 16, 64, 256, 64
        x = torch.randn(m, k, device=mps_device, dtype=torch.float16)
        w = torch.randn(n, k, device=mps_device, dtype=torch.float16)
        qweight, wscales = eager_q(w, convrot_groupsize=g)
        with ck.use_backend("metal"):
            actual = ck.convrot_w4a4_linear(x, qweight, wscales, convrot_groupsize=g)
        expected = eager_linear(x, qweight, wscales, convrot_groupsize=g)
        # INT4 row-scale ties can disagree by ±1 nibble vs eager torch.round.
        torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-1, atol=1.5)

    def test_metal_selected_for_rope_and_adaln(self, mps_device):
        x = torch.randn(2, 8, 16, 64, device=mps_device, dtype=torch.float16)
        freqs = _rotation_freqs(16, 32, mps_device)
        assert registry.get_capable_backend("apply_rope1", {"x": x, "freqs_cis": freqs}) == "metal"
        x2 = torch.randn(4, 32, 128, device=mps_device, dtype=torch.float16)
        s = torch.randn(1, 1, 128, device=mps_device, dtype=torch.float16)
        assert (
            registry.get_capable_backend(
                "adaln", {"x": x2, "scale": s, "shift": s, "eps": 1e-6}
            )
            == "metal"
        )

    def test_rope_faster_than_eager(self, mps_device):
        from comfy_kitchen.backends.eager.rope import apply_rope1 as eager_rope1

        if not _STRICT_PERF:
            pytest.skip("set COMFY_KITCHEN_METAL_PERF=1 for strict DiT-shape speed check")
        B, H, T, D = 1, 24, 2048, 128
        x = torch.randn(B, H, T, D, device=mps_device, dtype=torch.float16).contiguous()
        freqs = _rotation_freqs(T, D // 2, mps_device)
        torch.mps.empty_cache()
        for _ in range(5):
            _ = metal_backend.apply_rope1(x, freqs)
            _ = eager_rope1(x, freqs)
        torch.mps.synchronize()

        t0 = time.perf_counter()
        for _ in range(30):
            _ = metal_backend.apply_rope1(x, freqs)
        torch.mps.synchronize()
        metal_ms = (time.perf_counter() - t0) * 1000 / 30

        t0 = time.perf_counter()
        for _ in range(30):
            _ = eager_rope1(x, freqs)
        torch.mps.synchronize()
        eager_ms = (time.perf_counter() - t0) * 1000 / 30

        assert eager_ms / metal_ms >= 1.3, f"metal rope not faster enough: {eager_ms/metal_ms:.2f}x"


class TestMetalMxNvSolAwq:
    """Eager-on-MPS wrappers; do not require the metallib extension."""

    def test_lists_mx_nv_sol_awq(self, mps_device):
        caps = ck.list_backends()["metal"]["capabilities"]
        for name in (
            "quantize_nvfp4",
            "dequantize_nvfp4",
            "scaled_mm_nvfp4",
            "sol_attn",
            "gemv_awq_w4a16",
        ):
            assert name in caps
        if hasattr(torch, "float8_e8m0fnu"):
            for name in ("quantize_mxfp8", "dequantize_mxfp8", "scaled_mm_mxfp8"):
                assert name in caps

    def test_mxfp8_tensorops_mm(self, mps_device):
        if not hasattr(torch, "float8_e8m0fnu"):
            pytest.skip("torch.float8_e8m0fnu required")
        from comfy_kitchen.backends.eager.quantization import (
            dequantize_mxfp8 as eager_deq,
            quantize_mxfp8 as eager_q,
        )

        m, n, k = 64, 64, 64
        x = torch.randn(m, k, device=mps_device, dtype=torch.float16)
        w = torch.randn(n, k, device=mps_device, dtype=torch.float16)
        qx, sx = eager_q(x)
        qw, sw = eager_q(w)
        with ck.use_backend("metal"):
            out = ck.scaled_mm_mxfp8(qx, qw, sx, sw, out_dtype=torch.float16)
        ref = eager_deq(qx, sx, output_type=torch.float16) @ eager_deq(
            qw, sw, output_type=torch.float16
        ).t()
        torch.testing.assert_close(out.float(), ref.float(), rtol=5e-2, atol=5e-2)

    def test_nvfp4_fused_mm(self, mps_device):
        from comfy_kitchen.backends.eager.quantization import (
            dequantize_nvfp4 as eager_deq,
            quantize_nvfp4 as eager_q,
        )

        m, n, k = 32, 64, 64
        x = torch.randn(m, k, device=mps_device, dtype=torch.float16)
        w = torch.randn(n, k, device=mps_device, dtype=torch.float16)
        scale_x = torch.tensor([1.0], device=mps_device, dtype=torch.float32)
        scale_w = torch.tensor([1.0], device=mps_device, dtype=torch.float32)
        qx, sx = eager_q(x, scale_x)
        qw, sw = eager_q(w, scale_w)
        with ck.use_backend("metal"):
            out = ck.scaled_mm_nvfp4(qx, qw, scale_x, scale_w, sx, sw, out_dtype=torch.float16)
        ref = eager_deq(qx, scale_x, sx, output_type=torch.float16) @ eager_deq(
            qw, scale_w, sw, output_type=torch.float16
        ).t()
        torch.testing.assert_close(out.float(), ref.float(), rtol=5e-2, atol=5e-2)
        assert registry.get_capable_backend(
            "scaled_mm_nvfp4",
            {
                "a": qx,
                "b": qw,
                "tensor_scale_a": scale_x,
                "tensor_scale_b": scale_w,
                "block_scale_a": sx,
                "block_scale_b": sw,
                "out_dtype": torch.float16,
            },
        ) == "metal"

    @pytest.mark.skipif(
        not hasattr(torch, "float8_e8m0fnu"), reason="torch.float8_e8m0fnu required"
    )
    def test_mxfp8_roundtrip_and_scaled_mm(self, mps_device):
        from comfy_kitchen.backends.eager.quantization import (
            dequantize_mxfp8 as eager_deq,
            quantize_mxfp8 as eager_q,
        )

        m, n, k = 32, 64, 64
        x = torch.randn(m, k, device=mps_device, dtype=torch.float16)
        w = torch.randn(n, k, device=mps_device, dtype=torch.float16)
        qx, sx = metal_backend.quantize_mxfp8(x)
        qw, sw = metal_backend.quantize_mxfp8(w)
        qx_e, sx_e = eager_q(x)
        assert torch.equal(qx.view(torch.uint8), qx_e.view(torch.uint8))
        assert torch.equal(sx.view(torch.uint8), sx_e.view(torch.uint8))
        dq = metal_backend.dequantize_mxfp8(qx, sx, output_type=torch.float16)
        torch.testing.assert_close(
            dq, eager_deq(qx, sx, output_type=torch.float16), rtol=0, atol=0
        )
        with ck.use_backend("metal"):
            out = ck.scaled_mm_mxfp8(qx, qw, sx, sw, out_dtype=torch.float16)
        a_fp = eager_deq(qx, sx, output_type=torch.float16)
        b_fp = eager_deq(qw, sw, output_type=torch.float16)
        ref = a_fp @ b_fp.t()
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
        assert (
            registry.get_capable_backend(
                "scaled_mm_mxfp8",
                {
                    "a": qx,
                    "b": qw,
                    "block_scale_a": sx,
                    "block_scale_b": sw,
                    "out_dtype": torch.float16,
                },
            )
            == "metal"
        )

    def test_sol_attn_matches_eager(self, mps_device):
        from comfy_kitchen.backends.eager.sol_attn import sol_attn as eager_sol

        B, T, H, D = 1, 64, 4, 32
        q = torch.randn(B, T, H, D, device=mps_device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        with ck.use_backend("metal"):
            out = ck.sol_attn(q, k, v, tau=1.4)
        ref = eager_sol(q, k, v, tau=1.4)
        torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)
        assert registry.get_capable_backend("sol_attn", {"q": q, "k": k, "v": v}) == "metal"

    def test_gemv_awq_w4a16_matches_eager(self, mps_device):
        from comfy_kitchen.backends.eager.awq import gemv_awq_w4a16 as eager_gemv

        m, n, k, g = 4, 64, 256, 64
        x = torch.randn(m, k, device=mps_device, dtype=torch.float16)
        qweight = torch.randint(-128, 127, (n, k // 2), device=mps_device, dtype=torch.int8)
        wscales = torch.rand(k // g, n, device=mps_device, dtype=torch.float16) * 0.02 + 0.001
        wzeros = torch.randn(k // g, n, device=mps_device, dtype=torch.float16) * 0.01
        with ck.use_backend("metal"):
            out = ck.gemv_awq_w4a16(x, qweight, wscales, wzeros, group_size=g)
        ref = eager_gemv(x, qweight, wscales, wzeros, group_size=g)
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
        assert (
            registry.get_capable_backend(
                "gemv_awq_w4a16",
                {"x": x, "qweight": qweight, "wscales": wscales, "wzeros": wzeros},
            )
            == "metal"
        )
