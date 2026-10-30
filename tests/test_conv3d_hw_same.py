# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""conv3d_hw_same -- Wan VAE CausalConv3d conv and the Upsample + Conv2d frame upsampler.

The eager backend is F.conv3d (after a nearest x2 upsample for upsample=2). The Triton
implicit GEMM accumulates in fp32; it must stay as close to an fp32 reference as the eager
half-precision conv does: rel(triton vs fp32) <= 1.1 x rel(eager half vs fp32).
"""

import pytest
import torch
from torch.nn import functional

import comfy_kitchen as ck
from comfy_kitchen.backends import eager as eager_backend
from comfy_kitchen.registry import registry


def _rel(a, b):
    return ((a.float() - b.float()).abs().mean() / b.float().abs().mean().clamp_min(1e-12)).item()


def _make(ci, co, t, h, w, kt, kh, dtype, bias, device="cuda"):
    x = torch.randn(1, ci, t, h, w, device=device).to(dtype)
    wt = (torch.randn(co, ci, kt, kh, kh, device=device) * (ci * kt * kh * kh) ** -0.5).to(dtype)
    b = (torch.randn(co, device=device) * 0.1).to(dtype) if bias else None
    return x, wt, b


class TestConv3dHwSameEager:
    @pytest.mark.parametrize("kt,kh,up", [(3, 3, 1), (1, 1, 1), (1, 3, 2)])
    def test_matches_reference(self, kt, kh, up):
        x, wt, b = _make(6, 5, 4, 7, 9, kt, kh, torch.float32, True, device="cpu")
        with ck.use_backend("eager"):
            got = ck.conv3d_hw_same(x, wt, b, up)
        xr = x.repeat_interleave(up, 3).repeat_interleave(up, 4)
        ref = functional.conv3d(xr, wt, b, padding=(0, kh // 2, kh // 2))
        assert got.shape == (1, 5, 4 - kt + 1, 7 * up, 9 * up)
        torch.testing.assert_close(got, ref)

    def test_rejects_upsample_with_temporal_kernel(self):
        x, wt, _ = _make(4, 4, 3, 5, 5, 3, 3, torch.float32, False, device="cpu")
        with pytest.raises(ValueError):
            ck.conv3d_hw_same(x, wt, None, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestConv3dHwSameTriton:
    @pytest.fixture(autouse=True)
    def _triton(self):
        pytest.importorskip("triton")
        if not registry.is_available("triton"):
            pytest.skip("triton backend not available")

    def _check(self, x, wt, b, up):
        kwargs = {"x": x, "weight": wt, "bias": b, "upsample": up}
        assert registry.validate_backend_for_call("triton", "conv3d_hw_same", kwargs).success
        with ck.use_backend("triton"):
            got = ck.conv3d_hw_same(x, wt, b, up)
        half = eager_backend.conv3d_hw_same(x, wt, b, up)
        ref = eager_backend.conv3d_hw_same(x.float(), wt.float(), None if b is None else b.float(), up)
        assert got.shape == half.shape and got.dtype == x.dtype
        assert _rel(got, ref) <= 1.1 * _rel(half, ref)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("ci,co,t,h,w,kt,kh,bias", [
        (96, 96, 4, 44, 80, 3, 3, True),     # Wan VAE residual conv, cached chunk (T = 2 cache + 2)
        (40, 72, 3, 17, 23, 3, 3, False),    # Ci not a multiple of block_k, odd H/W
        (160, 48, 3, 20, 20, 3, 3, True),    # Co <= 64 tiles
        (64, 128, 1, 30, 50, 1, 1, True),    # 1x1x1 (shortcut / head)
        (256, 512, 3, 22, 40, 3, 1, True),   # time_conv (3, 1, 1)
    ])
    def test_matches_fp32_reference(self, dtype, ci, co, t, h, w, kt, kh, bias):
        x, wt, b = _make(ci, co, t, h, w, kt, kh, dtype, bias)
        self._check(x, wt, b, 1)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_t1_sliced_weight(self, dtype):
        # comfy's T == 1 fast path: weight[:, :, -1:] of a kt = 3 weight, a strided view
        x, wt, b = _make(64, 64, 1, 24, 40, 3, 3, dtype, True)
        self._check(x, wt[:, :, -1:], b, 1)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("ci,t,h,w", [(128, 2, 11, 13), (96, 1, 22, 40), (40, 3, 9, 7)])
    def test_upsample2(self, dtype, ci, t, h, w):
        x, wt, b = _make(ci, ci, t, h, w, 1, 3, dtype, True)
        self._check(x, wt, b, 2)

    def test_non_contiguous_input(self):
        x, wt, b = _make(32, 32, 3, 10, 12, 3, 3, torch.bfloat16, True)
        self._check(x.to(memory_format=torch.channels_last_3d), wt, b, 1)

    def test_unsupported_inputs_go_to_eager(self):
        x, wt, b = _make(16, 16, 3, 8, 8, 3, 3, torch.bfloat16, True)
        cases = [
            (x.float(), wt.float()),                       # fp32
            (torch.cat([x, x]), wt),                        # batch 2
            (x, wt.half()),                                 # weight dtype differs
            (x, torch.randn(16, 16, 3, 2, 2, device="cuda", dtype=torch.bfloat16)),  # even kernel
        ]
        for xx, ww in cases:
            kwargs = {"x": xx, "weight": ww, "bias": None, "upsample": 1}
            assert registry.get_capable_backend("conv3d_hw_same", kwargs) == "eager"
