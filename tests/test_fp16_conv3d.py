# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""fp16_conv3d — CUTLASS fp16-accumulate NDHWC conv3d with fused bias/residual.

Tolerances follow test_fp16_linear.py: fp16 accumulation over K = C*T*R*S
products compounds roughly with sqrt(K) * 2^-11 against torch's fp32-accumulate
reference.
"""

import math

import pytest
import torch
from torch.nn import functional

import comfy_kitchen as ck
from tests.conftest import cuda_backend_available, fp16_accum_tol, rel_err

CL3D = torch.channels_last_3d


def _inputs(c, k, d, h, w, ksize, with_bias=True, with_residual=False, stride=(1, 1, 1)):
    x = torch.randn(1, c, d, h, w, dtype=torch.float16, device="cuda").contiguous(memory_format=CL3D)
    weight = (torch.randn(k, c, *ksize, dtype=torch.float16, device="cuda") * 0.02).contiguous(memory_format=CL3D)
    bias = torch.randn(k, dtype=torch.float16, device="cuda") if with_bias else None
    residual = None
    if with_residual:
        oshape = (1, k, (d - ksize[0]) // stride[0] + 1, (h - ksize[1]) // stride[1] + 1, (w - ksize[2]) // stride[2] + 1)
        residual = torch.randn(oshape, dtype=torch.float16, device="cuda").contiguous(memory_format=CL3D)
    return x, weight, bias, residual


def _ref(x, weight, bias, residual, stride):
    out = functional.conv3d(x.float(), weight.float(), None if bias is None else bias.float(), stride=stride)
    return out if residual is None else out + residual.float()


class TestFp16Conv3d:
    # encoder stage shapes (input already padded, large enough to clear the
    # minimum-launch threshold): 3x3x3 at 128 and 256 channels, the strided
    # downsample, and the 1x1x1 shortcut projection
    @pytest.mark.parametrize(
        "c,k,d,h,w,ksize,stride",
        [
            (128, 128, 5, 130, 130, (3, 3, 3), (1, 1, 1)),
            (256, 256, 5, 130, 130, (3, 3, 3), (1, 1, 1)),
            (128, 128, 9, 129, 129, (3, 3, 3), (1, 2, 2)),
            (128, 256, 5, 128, 128, (1, 1, 1), (1, 1, 1)),
        ],
    )
    @pytest.mark.parametrize("with_bias,with_residual", [(True, False), (False, False), (True, True)])
    def test_matches_fp32_accum_reference(self, c, k, d, h, w, ksize, stride, with_bias, with_residual, seed, cuda_available):
        if not cuda_backend_available():
            pytest.skip("compiled CUDA backend required")
        from comfy_kitchen.backends import cuda as cuda_backend

        x, weight, bias, residual = _inputs(c, k, d, h, w, ksize, with_bias=with_bias, with_residual=with_residual, stride=stride)
        got = cuda_backend._cutlass_fp16_conv3d(x, weight, bias, residual, list(stride))
        assert got is not None, "fused kernel declined a shape it should serve"
        assert got.is_contiguous(memory_format=CL3D)
        ref = _ref(x, weight, bias, residual, stride)
        kdim = c * math.prod(ksize)
        rel = rel_err(got.float(), ref)
        assert rel < fp16_accum_tol(kdim), f"rel={rel:.4f} tol={fp16_accum_tol(kdim):.4f}"

    def test_deep_k_small_stage_served(self, seed, cuda_available):
        """The encoder's 512-channel 16^2 stage: too small for the 128-row tiles,
        served by the 64-row ones (which beat cuDNN there by 2x)."""
        if not cuda_backend_available():
            pytest.skip("compiled CUDA backend required")
        from comfy_kitchen.backends import cuda as cuda_backend

        x, weight, bias, residual = _inputs(512, 512, 7, 18, 18, (3, 3, 3), with_residual=True)
        got = cuda_backend._cutlass_fp16_conv3d(x, weight, bias, residual, [1, 1, 1])
        assert got is not None
        ref = _ref(x, weight, bias, residual, (1, 1, 1))
        assert rel_err(got.float(), ref) < fp16_accum_tol(512 * 27)

    def test_token_launch_declined(self, seed, cuda_available):
        """A launch below even the smallest config's threshold stays on cuDNN;
        the public op must still return the right answer."""
        if not cuda_backend_available():
            pytest.skip("compiled CUDA backend required")
        from comfy_kitchen.backends import cuda as cuda_backend

        x, weight, bias, residual = _inputs(64, 64, 3, 6, 6, (3, 3, 3), with_residual=True)
        assert cuda_backend._cutlass_fp16_conv3d(x, weight, bias, residual, [1, 1, 1]) is None
        got = ck.fp16_conv3d(x, weight, bias, residual)
        ref = _ref(x, weight, bias, residual, (1, 1, 1))
        assert rel_err(got.float(), ref) < 2e-3  # cuDNN fp32-accumulate path

    def test_pixel_channels_are_padded(self, seed, cuda_available):
        """C=3 (the pixel input) is zero-padded to the 8-channel vector width
        and served; the padded taps must contribute nothing."""
        if not cuda_backend_available():
            pytest.skip("compiled CUDA backend required")
        from comfy_kitchen.backends import cuda as cuda_backend

        x = torch.randn(1, 3, 5, 130, 130, dtype=torch.float16, device="cuda")
        weight = torch.randn(128, 3, 3, 3, 3, dtype=torch.float16, device="cuda") * 0.1
        got = cuda_backend._cutlass_fp16_conv3d(x, weight, None, None, [1, 1, 1])
        assert got is not None and got.shape == (1, 128, 3, 128, 128)
        ref = _ref(x, weight, None, None, (1, 1, 1))
        assert rel_err(got.float(), ref) < fp16_accum_tol(3 * 27)

    def test_bad_stride_is_reported_by_torch(self, seed, cuda_available):
        if not cuda_available:
            pytest.skip("CUDA required")
        x, weight, bias, _ = _inputs(16, 16, 4, 10, 10, (3, 3, 3))
        with pytest.raises(RuntimeError):
            ck.fp16_conv3d(x, weight, bias, stride=(0, 1, 1))

    def test_residual_shape_mismatch_falls_back(self, seed, cuda_available):
        if not cuda_backend_available():
            pytest.skip("compiled CUDA backend required")
        from comfy_kitchen.backends import cuda as cuda_backend

        x, weight, bias, residual = _inputs(128, 128, 5, 130, 130, (3, 3, 3), with_residual=True)
        assert cuda_backend._cutlass_fp16_conv3d(x, weight, bias, residual[:, :64], [1, 1, 1]) is None


class TestStridedViews:
    """NDHWC views in and frame windows out, so a memory-tiled conv needs no per-tile copies."""

    def test_input_row_window_matches_packed(self, seed, cuda_available):
        if not cuda_backend_available():
            pytest.skip("compiled CUDA backend required")
        from comfy_kitchen.backends import cuda as cuda_backend
        # windows large enough to clear the launch gate: 4 frames x 64 rows x 256 cols each
        x, weight, bias, _ = _inputs(128, 128, 6, 130, 258, (3, 3, 3))
        full = cuda_backend._cutlass_fp16_conv3d(x, weight, bias, None, [1, 1, 1])
        for h0, h1 in ((0, 64), (64, 128)):   # output rows; the input window carries the 2-row halo
            tile = cuda_backend._cutlass_fp16_conv3d(x[:, :, :, h0:h1 + 2, :], weight, bias, None, [1, 1, 1])
            assert tile is not None and torch.equal(tile, full[:, :, :, h0:h1, :])

    def test_frame_window_in_and_out(self, seed, cuda_available):
        if not cuda_backend_available():
            pytest.skip("compiled CUDA backend required")
        from comfy_kitchen.backends import cuda as cuda_backend
        x, weight, bias, _ = _inputs(128, 128, 6, 66, 130, (3, 3, 3))
        full = cuda_backend._cutlass_fp16_conv3d(x, weight, bias, None, [1, 1, 1])
        out = torch.zeros_like(full).contiguous(memory_format=CL3D)
        for z0, z1 in ((0, 2), (2, 4)):
            got = cuda_backend._cutlass_fp16_conv3d(x[:, :, z0:z1 + 2], weight, bias, None, [1, 1, 1], out=out[:, :, z0:z1])
            assert got is not None and got.data_ptr() == out[:, :, z0:z1].data_ptr()
        assert torch.equal(out, full)
        buf = torch.empty_like(full)
        assert ck.fp16_conv3d(x, weight, bias, out=buf).data_ptr() == buf.data_ptr()
        assert torch.equal(buf, full)

    def test_row_window_out_is_declined(self, seed, cuda_available):
        """The epilogue writes a packed 2-D matrix, so a row window of the output cannot be
        expressed: the kernel declines and the public op falls back to torch."""
        if not cuda_backend_available():
            pytest.skip("compiled CUDA backend required")
        from comfy_kitchen.backends import cuda as cuda_backend
        x, weight, bias, _ = _inputs(128, 128, 4, 34, 130, (3, 3, 3))
        full = ck.fp16_conv3d(x, weight, bias)
        out = torch.zeros_like(full).contiguous(memory_format=CL3D)
        window = out[:, :, :, :16, :]
        assert cuda_backend._cutlass_fp16_conv3d(x[:, :, :, :18, :], weight, bias, None, [1, 1, 1], out=window) is None
        ck.fp16_conv3d(x[:, :, :, :18, :], weight, bias, out=window)
        assert rel_err(window.float(), full[:, :, :, :16, :].float()) < fp16_accum_tol(128 * 27)

    def test_batch_of_two_frame_window_out(self, seed, cuda_available):
        """The epilogue packs rows across batches, so an N>1 frame window must decline rather than
        write batch 1 over batch 0's other frames."""
        if not cuda_available:
            pytest.skip("CUDA required")
        x = torch.randn(2, 128, 6, 66, 130, dtype=torch.float16, device="cuda").contiguous(memory_format=CL3D)
        weight = (torch.randn(128, 128, 3, 3, 3, dtype=torch.float16, device="cuda") * 0.02).contiguous(memory_format=CL3D)
        big = torch.full((2, 128, 4, 64, 128), 7.0, dtype=torch.float16, device="cuda").contiguous(memory_format=CL3D)
        window = big[:, :, 0:2]
        ck.fp16_conv3d(x[:, :, 0:4], weight, None, out=window)
        ref = _ref(x[:, :, 0:4], weight, None, None, (1, 1, 1))
        assert rel_err(window.float(), ref) < fp16_accum_tol(128 * 27)
        assert bool((big[:, :, 2:] == 7.0).all()), "frames outside the window were overwritten"

    def test_deep_k_large_launch_fp16_accumulates(self, seed, cuda_available):
        """512 channels x 27 taps, the depth the gate was raised to admit."""
        if not cuda_backend_available():
            pytest.skip("compiled CUDA backend required")
        from comfy_kitchen.backends import cuda as cuda_backend
        x, weight, bias, _ = _inputs(512, 512, 4, 66, 130, (3, 3, 3))
        got = cuda_backend._cutlass_fp16_conv3d(x, weight, bias, None, [1, 1, 1])
        assert got is not None
        assert rel_err(got.float(), _ref(x, weight, bias, None, (1, 1, 1))) < fp16_accum_tol(512 * 27)
