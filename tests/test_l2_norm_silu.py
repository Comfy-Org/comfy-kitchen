# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""l2_norm_silu -- Wan VAE RMS_norm (F.normalize over channels * sqrt(C) * gamma) + SiLU.

The eager backend is the sequence ComfyUI's Wan VAE ResidualBlock runs today. The Triton
kernel keeps the whole chain in fp32 and rounds once, so it is compared to eager with a
tolerance and to an fp32 reference.
"""

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

    def test_zero_input_is_zero(self):
        x = torch.zeros(1, 32, 2, 4, 4, dtype=torch.bfloat16, device="cuda")
        with ck.use_backend("triton"):
            got = ck.l2_norm_silu(x, torch.ones(32, device="cuda"), 32 ** 0.5)
        assert torch.equal(got, torch.zeros_like(x))

    def test_non_contiguous_input(self):
        x = torch.randn(1, 32, 3, 8, 8, dtype=torch.bfloat16, device="cuda")
        g = torch.rand(32, device="cuda") + 0.5
        for xx in (x.to(memory_format=torch.channels_last_3d), x[:, :, 1:]):
            kwargs = {"x": xx, "weight": g, "scale": 32 ** 0.5, "eps": 1e-12}
            assert registry.validate_backend_for_call("triton", "l2_norm_silu", kwargs).success
            with ck.use_backend("triton"):
                got = ck.l2_norm_silu(xx, g, 32 ** 0.5)
            assert _rel(got, eager_backend.l2_norm_silu(xx, g, 32 ** 0.5)) < 1e-2

    def test_unsupported_inputs_go_to_eager(self):
        g = torch.ones(32, device="cuda")
        for xx in (torch.randn(1, 32, 2, 4, 4, device="cuda"), torch.randn(4, 32, device="cuda", dtype=torch.bfloat16)):
            kwargs = {"x": xx, "weight": g, "scale": 1.0, "eps": 1e-12}
            assert registry.get_capable_backend("l2_norm_silu", kwargs) == "eager"
