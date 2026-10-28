# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""group_norm_silu_nchw -- GroupNorm + SiLU over a packed NCHW activation."""

import pytest
import torch
import torch.nn.functional as functional

from comfy_kitchen.backends import hip as hip_backend
from tests.conftest import rel_err


def _inputs(b, c, h, w, dtype, groups):
    x = torch.randn(b, c, h, w, device="cuda", dtype=dtype) * 3 + 0.5
    weight = torch.randn(c, device="cuda", dtype=dtype) * 0.5 + 1
    bias = torch.randn(c, device="cuda", dtype=dtype) * 0.2
    return x, weight, bias


class TestGroupNormSiluNchw:
    # the SDXL resnet shapes, an odd plane count (the scalar path) and C/G == 1
    @pytest.mark.parametrize(
        "b,c,h,w,groups",
        [
            (2, 320, 128, 128, 32),
            (2, 640, 64, 64, 32),
            (2, 1280, 32, 32, 32),
            (2, 1280, 33, 7, 32),
            (2, 48, 5, 5, 48),
            (1, 96, 16, 16, 32),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_matches_torch(self, b, c, h, w, groups, dtype, seed, cuda_available):
        if not cuda_available:
            pytest.skip("CUDA required")
        x, weight, bias = _inputs(b, c, h, w, dtype, groups)
        ref = functional.silu(functional.group_norm(x, groups, weight, bias, 1e-6))
        got = hip_backend.group_norm_silu_nchw(x, weight, bias, groups)

        assert got.shape == ref.shape
        assert got.is_contiguous()
        assert got.dtype == dtype
        # statistics: fp32 sum/sumsq in one block versus torch's Welford, and the
        # normalized value rounds to x's dtype before the SiLU as torch's does,
        # so the differences are ulp-level
        assert rel_err(got, ref) < 5 * torch.finfo(dtype).eps

    def test_out_reuse(self, seed, cuda_available):
        if not cuda_available:
            pytest.skip("CUDA required")
        x, weight, bias = _inputs(2, 320, 16, 16, torch.float16, 32)
        out = torch.empty_like(x)
        got = hip_backend.group_norm_silu_nchw(x, weight, bias, 32, out=out)
        assert got.data_ptr() == out.data_ptr()
        assert rel_err(got, functional.silu(functional.group_norm(x, 32, weight, bias, 1e-6))) < 5 * torch.finfo(
            torch.float16
        ).eps

    def test_groups_must_divide_channels(self, cuda_available):
        if not cuda_available:
            pytest.skip("CUDA required")
        x, weight, bias = _inputs(1, 96, 8, 8, torch.float16, 32)
        with pytest.raises(RuntimeError):
            hip_backend.group_norm_silu_nchw(x, weight, bias, 5)
