# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""temporal_interleave_upsample_nearest2x -- Wan VAE Resample upsample body up to its Conv2d.

The eager backend is the sequence ComfyUI's Wan VAE Resample runs today (reshape / stack /
reshape for the temporal interleave, rearrange to (b t) c h w, nn.Upsample nearest-exact).
For a scale of 2 the output is a copy of input pixels, so the Triton kernel must match eager
exactly (torch.equal).
"""

import importlib

import pytest
import torch
from torch import nn

import comfy_kitchen as ck
from comfy_kitchen.backends import eager as eager_backend
from comfy_kitchen.registry import registry

OP = "temporal_interleave_upsample_nearest2x"


def _comfy_resample_body(x, interleave):
    # comfy/ldm/wan/vae.py and vae2_2.py Resample.forward, from the time_conv output (interleave)
    # or the Resample input up to self.resample[1] (the Conv2d); rearrange written as
    # permute + reshape, which is what einops does for this pattern
    b, c, t, h, w = x.size()
    if interleave:
        c = c // 2
        x = x.reshape(b, 2, c, t, h, w)
        x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
        x = x.reshape(b, c, t * 2, h, w)
    t = x.shape[2]
    x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    return nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact")(x)


def _call_rule():
    triton_backend = importlib.import_module("comfy_kitchen.backends.triton")
    (rule,) = triton_backend._build_constraints()[OP].call_rules
    return rule


class TestTemporalInterleaveUpsampleEager:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("interleave", [False, True])
    @pytest.mark.parametrize("shape", [(1, 8, 1, 6, 10), (2, 6, 3, 5, 7)])
    def test_matches_comfy_chain(self, shape, interleave, dtype):
        x = torch.randn(shape).to(dtype)
        with ck.use_backend("eager"):
            out = ck.temporal_interleave_upsample_nearest2x(x, interleave)
        ref = _comfy_resample_body(x, interleave)
        assert torch.equal(out, ref)
        assert out.is_contiguous()

    def test_matches_einops_rearrange(self):
        einops = pytest.importorskip("einops")
        x = torch.randn(2, 8, 3, 5, 7)
        for interleave in (False, True):
            b, c, t, h, w = x.shape
            y = x
            if interleave:
                c = c // 2
                y = y.reshape(b, 2, c, t, h, w)
                y = torch.stack((y[:, 0], y[:, 1]), 3).reshape(b, c, t * 2, h, w)
            y = einops.rearrange(y, "b c t h w -> (b t) c h w")
            ref = nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact")(y)
            with ck.use_backend("eager"):
                assert torch.equal(ck.temporal_interleave_upsample_nearest2x(x, interleave), ref)

    def test_interleave_frame_order(self):
        # output frame 2t + k of batch item b is frame t of channel half k, every pixel 2 x 2
        b, c, t, h, w = 2, 3, 4, 2, 3
        x = torch.randn(b, 2 * c, t, h, w)
        with ck.use_backend("eager"):
            out = ck.temporal_interleave_upsample_nearest2x(x, interleave=True)
        assert out.shape == (b * 2 * t, c, 2 * h, 2 * w)
        for bb in range(b):
            for tt in range(2 * t):
                src = x[bb, (tt % 2) * c : (tt % 2 + 1) * c, tt // 2]
                want = src.repeat_interleave(2, -1).repeat_interleave(2, -2)
                assert torch.equal(out[bb * 2 * t + tt], want)

    def test_strided_input(self):
        x = torch.randn(1, 3, 8, 6, 10).transpose(1, 2)[:, :, 1:]
        assert not x.is_contiguous()
        for interleave in (False, True):
            with ck.use_backend("eager"):
                out = ck.temporal_interleave_upsample_nearest2x(x, interleave)
            assert torch.equal(out, _comfy_resample_body(x, interleave))

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            ck.temporal_interleave_upsample_nearest2x(torch.randn(2, 8, 5, 7))
        with pytest.raises(ValueError):
            ck.temporal_interleave_upsample_nearest2x(torch.randn(1, 7, 2, 4, 4), interleave=True)

    def test_fake_shape(self):
        x = torch.empty(2, 8, 3, 5, 7, device="meta")
        assert ck.temporal_interleave_upsample_nearest2x(x).shape == (6, 8, 10, 14)
        assert ck.temporal_interleave_upsample_nearest2x(x, True).shape == (12, 4, 10, 14)

    def test_cpu_goes_to_eager(self):
        kwargs = {"x": torch.randn(1, 8, 2, 4, 4), "interleave": True}
        assert registry.get_capable_backend(OP, kwargs) == "eager"


class TestTemporalInterleaveUpsampleCallRule:
    # the Triton call rule only looks at shapes, so it runs on meta tensors without a GPU
    @pytest.mark.parametrize(
        ("shape", "interleave", "ok"),
        [
            ((1, 384, 1, 60, 104), False, False),  # Wan2.1 first chunk, 9.58M out: eager faster
            ((1, 1024, 1, 44, 80), False, True),  # Wan2.2 first chunk, 14.4M out
            ((1, 1, 1, 1000, 2999), False, False),  # 11,996,000 out
            ((1, 1, 1, 1000, 3000), False, True),  # 12,000,000 out
            ((1, 8, 1, 4, 4), True, True),  # interleave is not limited by size
            ((1, 7, 1, 4, 4), True, False),  # odd channels
            ((0, 8, 2, 4, 4), True, False),  # empty
            ((1, 8, 0, 4, 4), True, False),  # empty
            ((8, 2, 4, 4), False, False),  # 4D
        ],
    )
    def test_call_rule(self, shape, interleave, ok):
        x = torch.empty(shape, device="meta", dtype=torch.bfloat16)
        assert _call_rule()({"x": x, "interleave": interleave}).success == ok


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestTemporalInterleaveUpsampleTriton:
    @pytest.fixture(autouse=True)
    def _triton(self):
        pytest.importorskip("triton")
        if not registry.is_available("triton"):
            pytest.skip("triton backend not available")

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    @pytest.mark.parametrize(
        ("shape", "interleave"),
        [
            # x of the Wan VAE decoder upsamples (Wan2.1 832 x 480 x 33, Wan2.2 1280 x 704 x 121);
            # interleave and upsample2d channel counts scaled down, H / W / T kept
            ((1, 96, 2, 60, 104), True),  # Wan2.1 upsample3d, 768 channels
            ((1, 96, 2, 120, 208), True),  # Wan2.1 upsample3d, 768 channels
            ((1, 256, 1, 44, 80), True),  # Wan2.2 upsample3d, 2048 channels
            ((1, 256, 2, 88, 160), True),  # Wan2.2 upsample3d, 2048 channels
            ((1, 384, 1, 120, 208), False),  # Wan2.1 upsample3d first chunk ('Rep'), full size
            ((1, 1024, 1, 44, 80), False),  # Wan2.2 upsample3d first chunk ('Rep'), full size
            ((1, 48, 2, 240, 416), False),  # Wan2.1 upsample2d, 192 channels
            ((1, 32, 4, 176, 320), False),  # Wan2.2 upsample2d, 512 channels
        ],
    )
    def test_matches_eager(self, shape, interleave, dtype):
        x = torch.randn(shape, device="cuda").to(dtype)
        kwargs = {"x": x, "interleave": interleave}
        assert registry.validate_backend_for_call("triton", OP, kwargs).success
        with ck.use_backend("triton"):
            got = ck.temporal_interleave_upsample_nearest2x(x, interleave)
        ref = eager_backend.temporal_interleave_upsample_nearest2x(x, interleave)
        assert got.is_contiguous() and got.dtype == dtype
        assert torch.equal(got, ref)
        assert torch.equal(got, _comfy_resample_body(x, interleave))

    @pytest.mark.parametrize("interleave", [False, True])
    @pytest.mark.parametrize(
        "layout",
        [
            "odd_sizes",  # batch 2, H / W not multiples of any tile
            "wide",  # W > block_ws of every config
            "channels_last_3d",
            "time_sliced",  # x[:, :, 1:] of a T-major tensor, storage offset
            "last_dim_step2",  # x[..., ::2]
            "batch_step2",  # x[::2]
        ],
    )
    def test_kernel_layouts(self, layout, interleave):
        # the kernel itself (no size routing) reads x through its strides
        kernel = importlib.import_module("comfy_kitchen.backends.triton." + OP)
        cl3 = torch.channels_last_3d
        make = {
            "odd_sizes": lambda: torch.randn(2, 6, 3, 5, 7, device="cuda"),
            "wide": lambda: torch.randn(1, 4, 2, 3, 1100, device="cuda"),
            "channels_last_3d": lambda: torch.randn(1, 8, 3, 6, 10, device="cuda").to(
                memory_format=cl3
            ),
            "time_sliced": lambda: torch.randn(1, 3, 8, 6, 10, device="cuda").transpose(1, 2)[
                :, :, 1:
            ],
            "last_dim_step2": lambda: torch.randn(1, 4, 2, 6, 20, device="cuda")[..., ::2],
            "batch_step2": lambda: torch.randn(4, 4, 2, 6, 10, device="cuda")[::2],
        }
        x = make[layout]().to(torch.bfloat16)
        got = kernel.temporal_interleave_upsample_nearest2x(x, interleave)
        assert got.is_contiguous()
        assert torch.equal(got, eager_backend.temporal_interleave_upsample_nearest2x(x, interleave))

    def test_strided_input_served(self):
        # a non-contiguous x large enough for the plain form is still served by triton
        x = torch.randn(1, 3, 64, 120, 208, device="cuda", dtype=torch.bfloat16).transpose(1, 2)[
            :, :, 1:
        ]
        assert not x.is_contiguous()
        for interleave in (False, True):
            kwargs = {"x": x, "interleave": interleave}
            assert registry.validate_backend_for_call("triton", OP, kwargs).success
            with ck.use_backend("triton"):
                got = ck.temporal_interleave_upsample_nearest2x(x, interleave)
            assert torch.equal(
                got, eager_backend.temporal_interleave_upsample_nearest2x(x, interleave)
            )

    def test_small_plain_goes_to_eager(self):
        # Wan2.1 first-chunk upsample, 9.58M output elements: the eager upsample is faster
        x = torch.randn(1, 384, 1, 60, 104, device="cuda", dtype=torch.bfloat16)
        kwargs = {"x": x, "interleave": False}
        assert not registry.validate_backend_for_call("triton", OP, kwargs).success
        assert registry.get_capable_backend(OP, kwargs) == "eager"
        with ck.use_backend("triton"):
            got = ck.temporal_interleave_upsample_nearest2x(x)
        assert torch.equal(got, _comfy_resample_body(x, False))

    @pytest.mark.parametrize("shape", [(0, 8, 2, 4, 4), (1, 8, 0, 4, 4)])
    def test_empty_input_goes_to_eager(self, shape):
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        kwargs = {"x": x, "interleave": True}
        assert not registry.validate_backend_for_call("triton", OP, kwargs).success
        with ck.use_backend("triton"):
            got = ck.temporal_interleave_upsample_nearest2x(x, True)
        assert got.shape == (0, 4, 8, 8) and got.dtype == x.dtype

    def test_unsupported_inputs_go_to_eager(self):
        # fp64 is outside the float dtypes of both backends; triton must decline it
        x = torch.randn(1, 8, 2, 4, 4, device="cuda", dtype=torch.float64)
        kwargs = {"x": x, "interleave": True}
        assert not registry.validate_backend_for_call("triton", OP, kwargs).success
        # CPU tensors go to eager
        x = torch.randn(1, 8, 2, 4, 4, dtype=torch.bfloat16)
        kwargs = {"x": x, "interleave": True}
        assert not registry.validate_backend_for_call("triton", OP, kwargs).success
        assert registry.get_capable_backend(OP, kwargs) == "eager"
