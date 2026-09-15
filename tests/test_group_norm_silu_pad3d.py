# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""group_norm_silu_pad3d — per-frame GroupNorm + SiLU + causal conv padding."""

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends.eager.group_norm_pad3d import group_norm_silu_pad3d as eager_ref
from tests.conftest import rel_err

CL3D = torch.channels_last_3d


def _inputs(c, t, h, w, dtype, affine=True):
    x = torch.randn(1, c, t, h, w, dtype=dtype, device="cuda") * 3 + 0.5
    x = x.contiguous(memory_format=CL3D)
    if not affine:
        return x, None, None
    weight = torch.randn(c, dtype=dtype, device="cuda") * 0.5 + 1
    bias = torch.randn(c, dtype=dtype, device="cuda") * 0.2
    return x, weight, bias


class TestGroupNormSiluPad3d:
    # channel counts of the H3 encoder stages, several frame sizes, both pad shapes it uses
    @pytest.mark.parametrize(
        "c,t,h,w,pad",
        [
            (128, 3, 40, 56, (1, 1, 1, 1, 2)),
            (256, 2, 33, 17, (1, 1, 1, 1, 2)),
            (512, 2, 9, 9, (0, 1, 0, 1, 2)),
            (1024, 1, 16, 16, (1, 1, 1, 1, 0)),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_matches_eager(self, c, t, h, w, pad, dtype, seed, cuda_available):
        if not cuda_available:
            pytest.skip("CUDA required")
        x, weight, bias = _inputs(c, t, h, w, dtype)
        ref = eager_ref(x, weight, bias, 32, 1e-6, list(pad), True)
        got = ck.group_norm_silu_pad3d(x, weight, bias, 32, 1e-6, pad, silu=True)

        assert got.shape == ref.shape
        assert got.is_contiguous(memory_format=CL3D)
        # statistics: fp32 sum/sumsq (double-reduced) vs torch's Welford; outputs
        # round to the same dtype the same way, so differences are ulp-level
        assert rel_err(got.float(), ref.float()) < 5 * torch.finfo(dtype).eps
        front = pad[4]
        if front:
            assert torch.all(got[:, :, :front] == 0)

    def test_pad_only(self, seed, cuda_available):
        """weight=None is the plain causal pad (used before the downsample convs)."""
        if not cuda_available:
            pytest.skip("CUDA required")
        x, _, _ = _inputs(128, 3, 31, 31, torch.float16, affine=False)
        pad = (0, 1, 0, 1, 2)
        ref = eager_ref(x, None, None, 1, 0.0, list(pad), False)
        got = ck.group_norm_silu_pad3d(x, None, None, 1, 0.0, pad, silu=False)
        assert torch.equal(got, ref)
        assert got.is_contiguous(memory_format=CL3D)

    def test_norm_without_silu(self, seed, cuda_available):
        if not cuda_available:
            pytest.skip("CUDA required")
        x, weight, bias = _inputs(256, 2, 20, 20, torch.float16)
        ref = eager_ref(x, weight, bias, 32, 1e-6, [1, 1, 1, 1, 0], False)
        got = ck.group_norm_silu_pad3d(x, weight, bias, 32, 1e-6, (1, 1, 1, 1, 0), silu=False)
        assert rel_err(got.float(), ref.float()) < 5e-3

    def test_unsupported_channels_fall_back(self, seed, cuda_available):
        """C=3 (conv_in's input) cannot use the kernel but must still be correct."""
        if not cuda_available:
            pytest.skip("CUDA required")
        x = torch.randn(1, 3, 2, 12, 12, dtype=torch.float16, device="cuda")
        got = ck.group_norm_silu_pad3d(x, None, None, 1, 0.0, (1, 1, 1, 1, 2), silu=False)
        ref = eager_ref(x, None, None, 1, 0.0, [1, 1, 1, 1, 2], False)
        assert torch.equal(got, ref)

    def test_kernel_limits_fall_back(self, cuda_available):
        """Group counts above 1024 and batch*frames above 65535 exceed the kernel's
        launch limits and must take the eager path instead of raising."""
        if not cuda_available:
            pytest.skip("CUDA required")
        x = torch.randn(1, 2048, 1, 4, 4, dtype=torch.float16, device="cuda")
        w = torch.ones(2048, dtype=torch.float16, device="cuda")
        got = ck.group_norm_silu_pad3d(x, w, None, 2048, 1e-6, (0, 0, 0, 0, 0), silu=False)
        ref = torch.nn.functional.group_norm(x, 2048, w, None, 1e-6)
        assert torch.allclose(got.float(), ref.float(), atol=1e-2)
        x = torch.randn(1, 8, 65536, 1, 1, dtype=torch.float16, device="cuda")
        got = ck.group_norm_silu_pad3d(x, None, None, 1, 0.0, (0, 0, 0, 0, 1), silu=False)
        assert got.shape[2] == 65537 and torch.equal(got[:, :, 1:].float(), x.float())

    def test_negative_padding_is_rejected(self, cuda_available):
        if not cuda_available:
            pytest.skip("CUDA required")
        x = torch.randn(1, 64, 3, 8, 8, dtype=torch.float16, device="cuda")
        for backend in ("cuda", "eager"):
            with ck.use_backend(backend), pytest.raises(ValueError):
                ck.group_norm_silu_pad3d(x, None, None, 1, 0.0, (0, 0, 0, 0, -1), silu=False)

    def test_misaligned_input_falls_back(self, seed, cuda_available):
        """A 16-byte-misaligned view must not reach the vectorized kernel."""
        if not cuda_available:
            pytest.skip("CUDA required")
        n = 128 * 2 * 12 * 12
        base = torch.randn(n + 4, dtype=torch.float16, device="cuda")
        x = base[4:4 + n].view(1, 2, 12, 12, 128).permute(0, 4, 1, 2, 3)  # NDHWC storage, misaligned
        assert x.data_ptr() % 16 == 8 and x.is_contiguous(memory_format=CL3D)
        got = ck.group_norm_silu_pad3d(x, None, None, 1, 0.0, (1, 1, 1, 1, 2), silu=True)
        torch.cuda.synchronize()
        ref = eager_ref(x, None, None, 1, 0.0, [1, 1, 1, 1, 2], True)
        assert torch.equal(got, ref)

    def test_contiguous_input_accepted(self, seed, cuda_available):
        """A plain NCDHW input is converted rather than misread."""
        if not cuda_available:
            pytest.skip("CUDA required")
        x = torch.randn(1, 128, 2, 12, 12, dtype=torch.float16, device="cuda")
        weight = torch.ones(128, dtype=torch.float16, device="cuda")
        bias = torch.zeros(128, dtype=torch.float16, device="cuda")
        ref = eager_ref(x, weight, bias, 32, 1e-6, [1, 1, 1, 1, 2], True)
        got = ck.group_norm_silu_pad3d(x, weight, bias, 32, 1e-6, (1, 1, 1, 1, 2))
        assert got.is_contiguous(memory_format=torch.channels_last_3d)
        assert rel_err(got.float(), ref.float()) < 5e-3

    @pytest.mark.parametrize("c", [128, 96])  # 96: C/8 is not a power of two -> eager fallback
    def test_fp32_affine_params(self, c, seed, cuda_available):
        """The registry admits fp32 weight/bias with a half input (fp32 master
        norms); both the kernel and the fallback must cast rather than hand
        torch's group_norm mixed dtypes."""
        if not cuda_available:
            pytest.skip("CUDA required")
        x, weight, bias = _inputs(c, 2, 12, 12, torch.float16)
        weight, bias = weight.float(), bias.float()
        got = ck.group_norm_silu_pad3d(x, weight, bias, 32, 1e-6, (1, 1, 1, 1, 2), silu=True)
        ref = eager_ref(x, weight.half(), bias.half(), 32, 1e-6, [1, 1, 1, 1, 2], True)
        assert got.dtype == torch.float16
        assert rel_err(got.float(), ref.float()) < 5e-3

    def test_eager_backend_matches_module_semantics(self, seed):
        """The eager reference must equal GroupNorm applied to each frame separately."""
        x = torch.randn(2, 64, 3, 8, 8)
        weight = torch.randn(64)
        bias = torch.randn(64)
        with ck.use_backend("eager"):
            got = ck.group_norm_silu_pad3d(x, weight, bias, 32, 1e-6, (1, 1, 1, 1, 2), silu=False)
        per_frame = torch.stack(
            [torch.nn.functional.group_norm(x[:, :, i], 32, weight, bias, 1e-6) for i in range(3)], dim=2)
        ref = torch.nn.functional.pad(torch.nn.functional.pad(per_frame, (1, 1, 1, 1, 0, 0), mode="reflect"),
                                      (0, 0, 0, 0, 2, 0))
        assert torch.allclose(got, ref, atol=1e-5)
