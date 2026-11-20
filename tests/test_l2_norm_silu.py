# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""l2_norm_silu -- Wan VAE RMS_norm (F.normalize over channels * sqrt(C) * gamma) + SiLU.

The eager backend is the sequence ComfyUI's Wan VAE ResidualBlock runs today. The Triton
kernel keeps the whole chain in fp32 and rounds once, so it is compared to eager with a
tolerance and to an fp32 reference.
"""

import importlib

import pytest
import torch
from torch.nn import functional

import comfy_kitchen as ck
from comfy_kitchen.backends import eager as eager_backend
from comfy_kitchen.registry import registry


def _reference(x, gamma, scale, eps=1e-12):
    xf = x.float()
    norm = xf.square().sum(1, keepdim=True).sqrt().clamp_min(eps)
    y = xf / norm * scale * gamma.float().reshape(x.shape[1], *([1] * (x.dim() - 2)))
    return y * torch.sigmoid(y)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def _randn(*shape, dtype):
    return torch.randn(shape, device="cuda", dtype=dtype)


class TestL2NormSiluEager:
    @pytest.mark.parametrize("shape", [(1, 32, 3, 5, 7), (2, 16, 9, 4)])
    def test_matches_normalize_then_silu(self, shape):
        x = torch.randn(shape)
        gamma = torch.rand(shape[1]) + 0.5
        scale = shape[1] ** 0.5
        with ck.use_backend("eager"):
            out = ck.l2_norm_silu(x, gamma, scale)
        g = gamma.reshape(shape[1], *([1] * (len(shape) - 2)))
        assert torch.equal(out, functional.silu(functional.normalize(x, dim=1) * scale * g))
        torch.testing.assert_close(out, _reference(x, gamma, scale), rtol=1e-5, atol=1e-6)

    def test_rejects_wrong_weight_size(self):
        with pytest.raises(ValueError):
            ck.l2_norm_silu(torch.randn(1, 8, 2, 2, 2), torch.ones(4), 8 ** 0.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestL2NormSiluTriton:
    @pytest.fixture(autouse=True)
    def _triton(self):
        pytest.importorskip("triton")
        if not registry.is_available("triton"):
            pytest.skip("triton backend not available")

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("shape", [
        (1, 256, 4, 44, 80),    # Wan2.2 VAE decoder residual block, 704 x 1280 / 16
        (2, 160, 2, 17, 23),    # batch 2, positions not a multiple of block_p, C % 32 == 0
        (1, 100, 3, 9, 11),     # C not a multiple of block_c
        (1, 64, 30, 50),        # 4D
    ])
    def test_matches_eager(self, dtype, shape):
        x = torch.randn(shape, dtype=dtype, device="cuda") * 3
        gamma = (torch.rand(shape[1], device="cuda") + 0.5).to(dtype)
        scale = shape[1] ** 0.5
        kwargs = {"x": x, "weight": gamma, "scale": scale, "eps": 1e-12}
        assert registry.validate_backend_for_call("triton", "l2_norm_silu", kwargs).success
        with ck.use_backend("triton"):
            got = ck.l2_norm_silu(x, gamma, scale)
        ref = eager_backend.l2_norm_silu(x, gamma, scale)
        ref32 = _reference(x, gamma, scale)
        assert got.shape == x.shape and got.dtype == dtype
        tol = 1e-2 if dtype == torch.bfloat16 else 2e-3
        assert _rel(got, ref) < tol
        assert _rel(got, ref32) < tol

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_weight_rounded_to_x_dtype(self, dtype):
        # the reference multiplies by gamma.to(x): an fp32 weight must give exactly the output of
        # the same weight rounded to x's dtype first (the tolerance tests are too loose to see it)
        x = torch.randn(1, 64, 2, 8, 8, dtype=dtype, device="cuda") * 3
        g = torch.rand(64, device="cuda") + 0.5
        assert not torch.equal(g.to(dtype).float(), g)
        with ck.use_backend("triton"):
            got = ck.l2_norm_silu(x, g, 8.0)
            rounded = ck.l2_norm_silu(x, g.to(dtype), 8.0)
        assert torch.equal(got, rounded)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_zero_channel_vector_matches_eager(self, dtype):
        # F.normalize rounds eps to x's dtype: 1e-12 is 0 in fp16, so a zero vector is 0 / 0 = NaN
        # there and 0 in bf16. The Triton kernel must give the same as eager in both.
        x = torch.randn(1, 32, 2, 4, 4, dtype=dtype, device="cuda")
        x[0, :, 1, 2, 3] = 0
        g = torch.rand(32, device="cuda") + 0.5
        with ck.use_backend("triton"):
            got = ck.l2_norm_silu(x, g, 32 ** 0.5)
        ref = eager_backend.l2_norm_silu(x, g, 32 ** 0.5)
        assert torch.equal(torch.isnan(got), torch.isnan(ref))
        assert torch.isnan(got[0, :, 1, 2, 3]).all() == (dtype == torch.float16)
        keep = ~torch.isnan(ref)
        assert _rel(got[keep], ref[keep]) < 1e-2

    def test_non_contiguous_input(self):
        x = torch.randn(1, 32, 3, 8, 8, dtype=torch.bfloat16, device="cuda")
        g = torch.rand(32, device="cuda") + 0.5
        for xx in (x.to(memory_format=torch.channels_last_3d), x[:, :, 1:]):
            kwargs = {"x": xx, "weight": g, "scale": 32 ** 0.5, "eps": 1e-12}
            assert registry.validate_backend_for_call("triton", "l2_norm_silu", kwargs).success
            with ck.use_backend("triton"):
                got = ck.l2_norm_silu(xx, g, 32 ** 0.5)
            assert _rel(got, eager_backend.l2_norm_silu(xx, g, 32 ** 0.5)) < 1e-2

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("layout", [
        "channels_last_3d",     # read in place, position stride C
        "channels_last",        # 4D, read in place
        "channels_last_b2",     # batch 2, channels last
        "last_dim_step2",       # x[..., ::2]: position stride 2
        "batch_step2",          # x[::2]: batch stride 2 * C * P
        "spatial_transposed",   # H and W swapped: no single position stride, copied
        "time_sliced",          # x[:, :, 1:] of a T-major tensor: copied
    ])
    def test_strided_input_matches_contiguous(self, dtype, layout):
        # a strided x must give the output of the same values made contiguous to within one
        # rounding step (the in-chunk reduction order follows the load layout), and match eager
        g = torch.rand(48, device="cuda") + 0.5
        cl3, cl = torch.channels_last_3d, torch.channels_last
        make = {
            "channels_last_3d": lambda: _randn(1, 48, 3, 9, 13, dtype=dtype).to(memory_format=cl3),
            "channels_last": lambda: _randn(1, 48, 17, 23, dtype=dtype).to(memory_format=cl),
            "channels_last_b2": lambda: _randn(2, 48, 2, 9, 13, dtype=dtype).to(memory_format=cl3),
            "last_dim_step2": lambda: _randn(1, 48, 3, 9, 26, dtype=dtype)[..., ::2],
            "batch_step2": lambda: _randn(4, 48, 3, 9, 13, dtype=dtype)[::2],
            "spatial_transposed": lambda: _randn(1, 48, 3, 13, 9, dtype=dtype).transpose(3, 4),
            "time_sliced": lambda: _randn(1, 3, 48, 9, 13, dtype=dtype).transpose(1, 2)[:, :, 1:],
        }
        x = make[layout]()
        assert not x.is_contiguous()
        with ck.use_backend("triton"):
            got = ck.l2_norm_silu(x, g, 48 ** 0.5)
            want = ck.l2_norm_silu(x.contiguous(), g, 48 ** 0.5)
        assert got.is_contiguous() and got.shape == x.shape
        eps = torch.finfo(dtype).eps
        torch.testing.assert_close(got, want, rtol=eps, atol=eps)
        tol = 1e-2 if dtype == torch.bfloat16 else 2e-3
        assert _rel(got, eager_backend.l2_norm_silu(x, g, 48 ** 0.5)) < tol

    def test_position_stride(self):
        # which layouts the kernel reads in place (a position stride) and which it copies (None)
        kernel = importlib.import_module("comfy_kitchen.backends.triton.l2_norm_silu")
        x = torch.empty(2, 48, 3, 9, 13)
        assert kernel._position_stride(x) == 1
        assert kernel._position_stride(x.to(memory_format=torch.channels_last_3d)) == 48
        assert kernel._position_stride(x[:, :, 0].to(memory_format=torch.channels_last)) == 48
        assert kernel._position_stride(torch.empty(1, 48, 3, 9, 26)[..., ::2]) == 2
        assert kernel._position_stride(x[..., ::2]) is None  # 13 is odd: rows do not line up
        assert kernel._position_stride(x[:, :, :1]) == 1
        assert kernel._position_stride(x.transpose(3, 4)) is None
        assert kernel._position_stride(x[:, :, 1:]) == 1  # channel stride is not P: fine
        assert kernel._position_stride(x.transpose(1, 2)[:, :, 1:]) is None

    @pytest.mark.parametrize("shape", [(0, 64, 32), (2, 0, 32), (2, 64, 0), (1, 16, 0, 4, 4)])
    def test_empty_input_goes_to_eager(self, shape):
        x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        g = torch.ones(shape[1], device="cuda")
        kwargs = {"x": x, "weight": g, "scale": 1.0, "eps": 1e-12}
        assert not registry.validate_backend_for_call("triton", "l2_norm_silu", kwargs).success
        with ck.use_backend("triton"):
            got = ck.l2_norm_silu(x, g, 1.0)
        assert got.shape == x.shape and got.dtype == x.dtype

    def test_unsupported_inputs_go_to_eager(self):
        g = torch.ones(32, device="cuda")
        for xx in (torch.randn(1, 32, 2, 4, 4, device="cuda"), torch.randn(4, 32, device="cuda", dtype=torch.bfloat16)):
            kwargs = {"x": xx, "weight": g, "scale": 1.0, "eps": 1e-12}
            assert registry.get_capable_backend("l2_norm_silu", kwargs) == "eager"
