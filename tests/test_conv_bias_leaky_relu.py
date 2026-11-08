# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""conv_bias_leaky_relu_ -- FILM FilmConv2d: conv bias add + LeakyReLU(0.2), in place.

On CUDA and ROCm the conv runs without its bias and at::convolution adds it in place
(output.add_(bias.reshape(1, -1, 1, 1))) before FilmConv2d's nn.LeakyReLU(0.2). The eager
backend is that bias add and LeakyReLU on the bias-free conv output, so it must match the stock
ops exactly. The Triton kernel rounds the sum to x's dtype before the activation, as the stock
add_ does, so it must be equal to eager as well.
"""

import importlib

import pytest
import torch
from torch import nn
from torch.nn import functional

import comfy_kitchen as ck
from comfy_kitchen.backends import eager as eager_backend
from comfy_kitchen.registry import registry

OP = "conv_bias_leaky_relu_"


def _stock_chain(y, bias, slope=0.2):
    # at::convolution on cuDNN / MIOpen: the bias-free conv output y, then
    # output.add_(reshape_bias(input.dim(), bias)); then FilmConv2d.forward's
    # self.activation = nn.LeakyReLU(0.2) (comfy_extras/frame_interpolation_models/film_net.py,
    # ComfyUI fb2b26f)
    y.add_(bias.reshape((1, -1) + (1,) * (y.dim() - 2)))
    return nn.LeakyReLU(slope)(y)


def _conv_out(n, c, h, w, layout, *, device="cpu", dtype=torch.float16):
    x = torch.randn(n, c, h, w, device=device).to(dtype)
    if layout == "channels_last":
        x = x.contiguous(memory_format=torch.channels_last)
    return x


def _bias(c, *, device="cpu", dtype=torch.float16):
    return (torch.randn(c, device=device) * 0.1).to(dtype)


def _kwargs(x, bias, negative_slope=0.2):
    return {"x": x, "bias": bias, "negative_slope": negative_slope}


def _call_rule():
    triton_backend = importlib.import_module("comfy_kitchen.backends.triton")
    (rule,) = triton_backend._build_constraints()[OP].call_rules
    return rule


_LAYOUTS = ["contiguous", "channels_last"]

# FILM conv outputs at 1920 x 1072 are [1, C, H, W] with C in {16, 32, 64, 128, 256, 512} on
# the levels 1072x1920 ... 33x60, 16x30; scaled down here to small planes (odd sizes, a channel
# count that is not a power of two, batch 2)
_EAGER_SHAPES = [
    (1, 64, 34, 60),
    (1, 512, 16, 30),
    (1, 16, 67, 120),
    (2, 32, 9, 13),
    (1, 24, 7, 5),
]


class TestConvBiasLeakyReluEager:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("layout", _LAYOUTS)
    @pytest.mark.parametrize(("n", "c", "h", "w"), _EAGER_SHAPES)
    def test_matches_stock_chain(self, n, c, h, w, layout, dtype):
        x = _conv_out(n, c, h, w, layout, dtype=dtype)
        bias = _bias(c, dtype=dtype)
        ref = _stock_chain(x.clone(memory_format=torch.preserve_format), bias)
        with ck.use_backend("eager"):
            out = ck.conv_bias_leaky_relu_(x, bias)
        assert out is x
        assert out.dtype == dtype and out.shape == ref.shape
        assert out.stride() == ref.stride()
        assert torch.equal(out, ref)

    @pytest.mark.parametrize("slope", [0.0, 0.01, 0.2, 1.5])
    def test_negative_slope(self, slope):
        x = _conv_out(1, 64, 34, 60, "contiguous")
        bias = _bias(64)
        ref = _stock_chain(x.clone(), bias, slope)
        with ck.use_backend("eager"):
            out = ck.conv_bias_leaky_relu_(x, bias, slope)
        assert torch.equal(out, ref)

    def test_in_place(self):
        x = _conv_out(1, 64, 34, 60, "channels_last")
        ptr, stride = x.data_ptr(), x.stride()
        with ck.use_backend("eager"):
            out = ck.conv_bias_leaky_relu_(x, _bias(64))
        assert out is x and x.data_ptr() == ptr and x.stride() == stride

    def test_bias_rounded_before_activation(self):
        # fp16: 1024 + 0.4 rounds to 1024 in the add_, so the activation sees 1024 (not 1024.4);
        # -1024 + 0.4 rounds to -1024 -> -204.8 rounded to fp16
        x = torch.tensor([1024.0, -1024.0], dtype=torch.float16).reshape(1, 2, 1, 1)
        bias = torch.tensor([0.4, 0.4], dtype=torch.float16)
        ref = _stock_chain(x.clone(), bias)
        with ck.use_backend("eager"):
            out = ck.conv_bias_leaky_relu_(x, bias)
        assert torch.equal(out, ref)
        assert out.flatten()[0].item() == 1024.0

    @pytest.mark.parametrize(
        ("x_shape", "bias_shape"),
        [
            ((8,), (8,)),  # 1D x
            ((1, 8, 4, 4), (4,)),  # bias != C
            ((1, 8, 4, 4), (1, 8)),  # 2D bias
            ((1, 8, 4, 4), ()),  # 0-d bias
        ],
    )
    def test_rejects_bad_inputs(self, x_shape, bias_shape):
        with pytest.raises(ValueError):
            ck.conv_bias_leaky_relu_(torch.randn(x_shape), torch.randn(bias_shape))

    def test_fake(self):
        x = torch.empty(1, 64, 33, 60, device="meta", dtype=torch.float16)
        bias = torch.empty(64, device="meta", dtype=torch.float16)
        out = ck.conv_bias_leaky_relu_(x, bias)
        assert out is x

    def test_cpu_goes_to_eager(self):
        kwargs = _kwargs(_conv_out(1, 64, 34, 60, "contiguous"), _bias(64))
        assert registry.get_capable_backend(OP, kwargs) == "eager"


def _meta(n, c, h, w, layout="contiguous", dtype=torch.float16):
    x = torch.empty(n, c, h, w, device="meta", dtype=dtype)
    if layout == "channels_last":
        x = x.contiguous(memory_format=torch.channels_last)
    return x


def _meta_bias(c, dtype=torch.float16):
    return torch.empty(c, device="meta", dtype=dtype)


class TestConvBiasLeakyReluCallRule:
    # the Triton call rule only looks at shapes, dtypes and layouts, so it runs on meta tensors
    @pytest.mark.parametrize("layout", _LAYOUTS)
    @pytest.mark.parametrize(
        ("c", "h", "w", "ok"),
        [
            (64, 1072, 1920, True),  # FILM level 0
            (128, 536, 960, True),
            (256, 268, 480, True),
            (512, 134, 240, True),
            (256, 67, 120, True),
            (512, 33, 60, True),  # 1980 pixels per plane
            (512, 16, 30, False),  # 480: runs eager
            (256, 16, 30, False),
            (64, 16, 30, False),
            (8, 32, 32, True),  # 1024
            (8, 31, 33, False),  # 1023
        ],
    )
    def test_call_rule_sizes(self, c, h, w, layout, ok):
        kwargs = _kwargs(_meta(1, c, h, w, layout), _meta_bias(c))
        assert _call_rule()(kwargs).success == ok

    @pytest.mark.parametrize(
        ("x_dtype", "bias_dtype", "ok"),
        [
            (torch.float16, torch.float16, True),
            (torch.bfloat16, torch.bfloat16, False),  # not measured
            (torch.float32, torch.float32, False),
            (torch.float16, torch.float32, False),  # bias not cast
        ],
    )
    def test_call_rule_dtypes(self, x_dtype, bias_dtype, ok):
        kwargs = _kwargs(_meta(1, 64, 34, 60, dtype=x_dtype), _meta_bias(64, bias_dtype))
        assert _call_rule()(kwargs).success == ok

    def test_call_rule_layouts(self):
        bias = _meta_bias(64)
        strided = _meta(1, 64, 68, 60)[:, :, ::2]
        assert not _call_rule()(_kwargs(strided, bias)).success
        permuted = _meta(1, 34, 64, 60).transpose(1, 2)
        assert not _call_rule()(_kwargs(permuted, bias)).success
        bias2 = _meta_bias(128)[::2]
        assert not _call_rule()(_kwargs(_meta(1, 64, 34, 60), bias2)).success

    def test_call_rule_limits(self):
        # 2**31 elements
        assert not _call_rule()(_kwargs(_meta(1, 1024, 2048, 1024), _meta_bias(1024))).success
        assert _call_rule()(_kwargs(_meta(1, 1023, 2048, 1024), _meta_bias(1023))).success
        # contiguous: N * C is a grid dimension
        assert not _call_rule()(_kwargs(_meta(2, 32768, 32, 32), _meta_bias(32768))).success
        assert _call_rule()(_kwargs(_meta(1, 65535, 32, 32), _meta_bias(65535))).success
        # channels_last: all channels in one tile
        x = _meta(1, 2048, 32, 32, "channels_last")
        assert not _call_rule()(_kwargs(x, _meta_bias(2048))).success
        x = _meta(1, 1024, 32, 32, "channels_last")
        assert _call_rule()(_kwargs(x, _meta_bias(1024))).success

    def test_fp16_only(self):
        triton_backend = importlib.import_module("comfy_kitchen.backends.triton")
        params = triton_backend._build_constraints()[OP].params
        assert params["x"].dtypes == frozenset({torch.float16})
        assert params["bias"].dtypes == frozenset({torch.float16})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestConvBiasLeakyReluTriton:
    @pytest.fixture(autouse=True)
    def _triton(self):
        pytest.importorskip("triton")
        if not registry.is_available("triton"):
            pytest.skip("triton backend not available")

    @staticmethod
    def _check(x, bias, slope=0.2):
        kwargs = _kwargs(x, bias, slope)
        assert registry.validate_backend_for_call("triton", OP, kwargs).success
        assert registry.get_capable_backend(OP, kwargs) == "triton"
        ref = eager_backend.conv_bias_leaky_relu_(
            x.clone(memory_format=torch.preserve_format), bias, slope
        )
        ptr, stride = x.data_ptr(), x.stride()
        with ck.use_backend("triton"):
            got = ck.conv_bias_leaky_relu_(x, bias, slope)
        assert got is x and x.data_ptr() == ptr and x.stride() == stride
        assert torch.equal(got, ref)
        return got

    @pytest.mark.parametrize("layout", _LAYOUTS)
    @pytest.mark.parametrize(
        ("n", "c", "h", "w"),
        [
            (1, 64, 67, 120),
            (1, 512, 33, 60),
            (1, 256, 34, 60),
            (1, 16, 134, 240),
            (2, 32, 37, 61),  # odd plane, batch 2
            (1, 24, 40, 33),  # C not a power of two
            (1, 3, 32, 32),
        ],
    )
    def test_matches_eager(self, n, c, h, w, layout):
        x = _conv_out(n, c, h, w, layout, device="cuda")
        self._check(x, _bias(c, device="cuda"))

    @pytest.mark.parametrize("slope", [0.0, 0.01, 1.5])
    def test_negative_slope(self, slope):
        x = _conv_out(1, 64, 34, 60, "contiguous", device="cuda")
        self._check(x, _bias(64, device="cuda"), slope)

    def test_special_values(self):
        x = _conv_out(1, 8, 32, 40, "contiguous", device="cuda")
        flat = x.view(-1)
        flat[:6] = torch.tensor(
            [float("inf"), -float("inf"), float("nan"), 65504.0, -65504.0, 0.0],
            dtype=torch.float16,
        )
        bias = _bias(8, device="cuda")
        bias[0] = 65504.0  # overflows to inf in the fp16 add
        ref = _stock_chain(x.clone(), bias)
        with ck.use_backend("triton"):
            got = ck.conv_bias_leaky_relu_(x, bias)
        assert torch.equal(got.isnan(), ref.isnan())
        assert torch.equal(got.nan_to_num(), ref.nan_to_num())

    def test_unaligned_view(self):
        # a conv output view at an odd element offset (pointer not 16-byte aligned)
        base = torch.randn(1 * 64 * 34 * 60 + 1, device="cuda").to(torch.float16)
        x = base[1:].view(1, 64, 34, 60)
        self._check(x, _bias(64, device="cuda"))

    @pytest.mark.parametrize("layout", _LAYOUTS)
    def test_film_conv_matches_stock(self, layout):
        # FilmConv2d with activation on the GPU: the stock conv with bias + nn.LeakyReLU(0.2)
        # against the conv without bias + this op
        x = _conv_out(1, 32, 67, 120, layout, device="cuda")
        conv = nn.Conv2d(32, 64, 3, padding=1, device="cuda", dtype=torch.float16)
        if layout == "channels_last":
            conv = conv.to(memory_format=torch.channels_last)
        with torch.no_grad():
            ref = nn.LeakyReLU(0.2)(conv(x))
            y = functional.conv2d(x, conv.weight, None, conv.stride, conv.padding)
            assert registry.get_capable_backend(OP, _kwargs(y, conv.bias)) == "triton"
            got = ck.conv_bias_leaky_relu_(y, conv.bias)
        assert got.stride() == ref.stride()
        assert torch.equal(got, ref)

    @pytest.mark.parametrize("layout", _LAYOUTS)
    def test_real_shape_served(self, layout):
        # FILM level 0 at 1920 x 1072: [1, 64, 1072, 1920] fp16
        x = _conv_out(1, 64, 1072, 1920, layout, device="cuda")
        self._check(x, _bias(64, device="cuda"))

    def test_unsupported_inputs_go_to_eager(self):
        cases = [
            # the 16 x 30 FILM level (below the size threshold), both layouts
            (_conv_out(1, 512, 16, 30, "contiguous", device="cuda"), _bias(512, device="cuda")),
            (
                _conv_out(1, 256, 16, 30, "channels_last", device="cuda"),
                _bias(256, device="cuda"),
            ),
            # bf16 / fp32, an fp32 bias with fp16 x
            (
                _conv_out(1, 64, 34, 60, "contiguous", device="cuda", dtype=torch.bfloat16),
                _bias(64, device="cuda", dtype=torch.bfloat16),
            ),
            (
                _conv_out(1, 64, 34, 60, "contiguous", device="cuda", dtype=torch.float32),
                _bias(64, device="cuda", dtype=torch.float32),
            ),
            (
                _conv_out(1, 64, 34, 60, "contiguous", device="cuda"),
                _bias(64, device="cuda", dtype=torch.float32),
            ),
            # a strided view, a non-contiguous bias
            (
                _conv_out(1, 64, 68, 60, "contiguous", device="cuda")[:, :, ::2],
                _bias(64, device="cuda"),
            ),
            (_conv_out(1, 64, 34, 60, "contiguous", device="cuda"), _bias(128, device="cuda")[::2]),
        ]
        for x, bias in cases:
            kwargs = _kwargs(x, bias)
            assert not registry.validate_backend_for_call("triton", OP, kwargs).success
            assert registry.get_capable_backend(OP, kwargs) == "eager"
            ref = _stock_chain(x.clone(memory_format=torch.preserve_format), bias)
            with ck.use_backend("triton"):
                got = ck.conv_bias_leaky_relu_(x, bias)
            assert got is x
            assert torch.equal(got, ref)
        # CPU tensors go to eager
        kwargs = _kwargs(_conv_out(1, 64, 34, 60, "contiguous"), _bias(64))
        assert not registry.validate_backend_for_call("triton", OP, kwargs).success
        assert registry.get_capable_backend(OP, kwargs) == "eager"
