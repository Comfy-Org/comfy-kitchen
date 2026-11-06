# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""pixel_norm_silu_pad3d -- LTX VAE PixelNorm + SiLU + CausalConv3d temporal padding.

The eager backend is the sequence ComfyUI's LTX VAE runs today (PixelNorm, nn.SiLU, the
CausalConv3d repeat / cat, and the reflect F.pad that Conv3d builds for padding_mode
"reflect"), so it must match the stock code exactly. The Triton kernel computes the norm and
SiLU in fp32 and rounds once, so it is compared to eager and to an fp32 reference with a
tolerance; copied values (the temporal cache, repeated frames, the reflected border) must be
exact.
"""

import importlib

import pytest
import torch
from torch import nn
from torch.nn import functional

import comfy_kitchen as ck
from comfy_kitchen.backends import eager as eager_backend
from comfy_kitchen.registry import registry

OP = "pixel_norm_silu_pad3d"


def _pixel_norm(x, eps=1e-8):
    # comfy/ldm/lightricks/vae/pixel_norm.py PixelNorm.forward (dim=1)
    return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + eps)


def _stock_chain(x, prefix, n_front, n_back, pad_mode):
    # ResnetBlock3D norm1 -> non_linearity -> conv1 (CausalConv3d.forward up to self.conv),
    # then what Conv3d._conv_forward builds before F.conv3d for padding_mode "reflect"
    h = nn.SiLU()(_pixel_norm(x))
    cached = prefix if prefix is not None else h[:, :, :1, :, :].repeat((1, 1, n_front, 1, 1))
    pieces = [cached, h]
    if n_back:
        pieces.append(h[:, :, -1:, :, :].repeat((1, 1, n_back, 1, 1)))
    y = torch.cat(pieces, dim=2)
    if pad_mode == "reflect":
        y = functional.pad(y, (1, 1, 1, 1, 0, 0), mode="reflect")
    return y


class _StockCausalConv3dInput:
    # comfy/ldm/lightricks/vae/causal_conv3d.py CausalConv3d.forward (ComfyUI fb2b26f), one
    # thread, time kernel 3, stride 1; self.conv replaced by what Conv3d builds before its conv
    # (the reflect F.pad for padding_mode "reflect", nothing for "zeros")
    time_kernel_size, time_stride = 3, 1

    def __init__(self, pad_mode):
        self.pad_mode = pad_mode
        self.state = (None, False)

    def mark_end(self):
        self.state = (self.state[0], True)

    def forward(self, x, causal=False):
        cached, is_end = self.state
        if cached is None:
            padding_length = self.time_kernel_size - 1
            if not causal:
                padding_length = padding_length // 2
            cached = x[:, :, :1, :, :].repeat((1, 1, padding_length, 1, 1))
        pieces = [cached, x]
        if is_end and not causal:
            pieces.append(x[:, :, -1:, :, :].repeat((1, 1, (self.time_kernel_size - 1) // 2, 1, 1)))
        input_length = sum([piece.shape[2] for piece in pieces])
        cache_length = (self.time_kernel_size - self.time_stride) + (
            (input_length - self.time_kernel_size) % self.time_stride
        )
        needs_caching = not is_end
        if needs_caching and cache_length == 0:
            self.state = (x[:, :, :0, :, :], False)
            needs_caching = False
        if needs_caching and x.shape[2] >= cache_length:
            needs_caching = False
            self.state = (x[:, :, -cache_length:, :, :], False)
        x = torch.cat(pieces, dim=2)
        if needs_caching:
            self.state = (x[:, :, -cache_length:, :, :], False)
        elif is_end:
            self.state = (None, True)
        if self.pad_mode == "reflect":
            x = functional.pad(x, (1, 1, 1, 1, 0, 0), mode="reflect")
        return x


def _stock_chunked(x, chunks, pad_mode):
    # the decoder runs the time chunks of x (torch.chunk / split views) through the same
    # CausalConv3d, marking the last one as the end (mark_conv3d_ended)
    conv = _StockCausalConv3dInput(pad_mode)
    outs = []
    for i, xc in enumerate(torch.split(x, chunks, dim=2)):
        if i == len(chunks) - 1:
            conv.mark_end()
        outs.append(conv.forward(nn.SiLU()(_pixel_norm(xc))))
    return outs


def _kitchen_chunked(x, chunks, pad_mode):
    # the same chunks through the op: the caller keeps the last 2 frames of each output (the
    # unpadded interior for reflect) as the next chunk's prefix, n_front only on the first chunk
    p = 1 if pad_mode == "reflect" else 0
    prefix, outs = None, []
    for i, xc in enumerate(torch.split(x, chunks, dim=2)):
        n_back = 1 if i == len(chunks) - 1 else 0
        out = ck.pixel_norm_silu_pad3d(xc, prefix, 1 if prefix is None else 0, n_back, pad_mode)
        outs.append(out)
        prefix = out[:, :, -2:, p : out.shape[3] - p, p : out.shape[4] - p]
    return outs


def _pad_hw(t, pad_mode):
    if pad_mode == "reflect":
        return functional.pad(t, (1, 1, 1, 1, 0, 0), mode="reflect")
    return t


def _reference32(x, prefix, n_front, n_back, pad_mode):
    pre = None if prefix is None else prefix.float()
    return _stock_chain(x.float(), pre, n_front, n_back, pad_mode)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def _chunk_view(b, c, t, h, w, offset, *, device="cpu", dtype=torch.float32):
    # frames [offset, offset + t) of a longer tensor: the strides of a decoder time chunk
    base = torch.randn(b, c, t + offset + 1, h, w, device=device).to(dtype)
    return base[:, :, offset : offset + t]


def _call_rule():
    triton_backend = importlib.import_module("comfy_kitchen.backends.triton")
    (rule,) = triton_backend._build_constraints()[OP].call_rules
    return rule


# prefix / n_front / n_back forms of the decoder: first chunk (repeat the first frame), later
# chunks (temporal cache), and the last chunk of each (repeat the last frame)
_FORMS = [(False, 1, 0), (False, 1, 1), (True, 0, 0), (True, 0, 1)]


class TestPixelNormSiluPad3dEager:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("pad_mode", ["zeros", "reflect"])
    @pytest.mark.parametrize(("has_prefix", "n_front", "n_back"), _FORMS)
    def test_matches_stock_chain(self, has_prefix, n_front, n_back, pad_mode, dtype):
        x = torch.randn(1, 16, 4, 6, 7).to(dtype)
        prefix = torch.randn(1, 16, 2, 6, 7).to(dtype) if has_prefix else None
        with ck.use_backend("eager"):
            out = ck.pixel_norm_silu_pad3d(x, prefix, n_front, n_back, pad_mode)
        ref = _stock_chain(x, prefix, n_front, n_back, pad_mode)
        assert out.is_contiguous() and out.dtype == dtype
        assert torch.equal(out, ref)

    @pytest.mark.parametrize("pad_mode", ["zeros", "reflect"])
    @pytest.mark.parametrize("chunks", [[4], [3, 2, 4], [1, 1, 3], [2, 1]])
    def test_chunked_decode_matches_stock_causal_conv(self, chunks, pad_mode):
        # the prefix is the conv input of the previous chunk, already normalized and activated:
        # driving the op chunk by chunk with that cache gives what CausalConv3d builds
        x = torch.randn(1, 8, sum(chunks), 5, 6).to(torch.bfloat16)
        with ck.use_backend("eager"):
            got = _kitchen_chunked(x, chunks, pad_mode)
        ref = _stock_chunked(x, chunks, pad_mode)
        assert len(got) == len(ref)
        for g, r in zip(got, ref, strict=True):
            assert torch.equal(g, r)

    def test_prefix_is_copied_not_normalized(self):
        x = torch.randn(1, 8, 3, 4, 5)
        prefix = torch.randn(1, 8, 2, 4, 5) * 100
        with ck.use_backend("eager"):
            out = ck.pixel_norm_silu_pad3d(x, prefix, n_front=0, n_back=0)
        assert torch.equal(out[:, :, :2], prefix)
        assert torch.equal(out[:, :, 2:], functional.silu(_pixel_norm(x)))

    def test_prefix_and_n_front(self):
        # with a prefix and n_front > 0 the repeated first frame follows the prefix
        x = torch.randn(1, 8, 3, 4, 5)
        prefix = torch.randn(1, 8, 2, 4, 5)
        with ck.use_backend("eager"):
            out = ck.pixel_norm_silu_pad3d(x, prefix, n_front=2, n_back=1)
        h = functional.silu(_pixel_norm(x))
        assert out.shape == (1, 8, 8, 4, 5)
        assert torch.equal(out, torch.cat([prefix, h[:, :, :1], h[:, :, :1], h, h[:, :, -1:]], 2))

    def test_empty_prefix(self):
        # CausalConv3d caches x[:, :, :0] when its cache length is 0
        x = torch.randn(1, 8, 3, 4, 5)
        with ck.use_backend("eager"):
            out = ck.pixel_norm_silu_pad3d(x, x[:, :, :0], n_front=0, n_back=0)
        assert torch.equal(out, functional.silu(_pixel_norm(x)))

    def test_strided_inputs(self):
        x = _chunk_view(1, 16, 3, 6, 7, 2)
        prefix = _chunk_view(1, 16, 2, 6, 7, 4)
        assert not x.is_contiguous() and not prefix.is_contiguous()
        for pad_mode in ("zeros", "reflect"):
            for xx in (x, x.contiguous().to(memory_format=torch.channels_last_3d)):
                with ck.use_backend("eager"):
                    out = ck.pixel_norm_silu_pad3d(xx, prefix, 0, 1, pad_mode)
                assert out.is_contiguous()
                assert torch.equal(out, _stock_chain(xx, prefix, 0, 1, pad_mode).contiguous())

    @pytest.mark.parametrize(
        ("x_shape", "prefix_shape", "kwargs"),
        [
            ((8, 3, 4, 5), None, {}),  # 4D
            ((1, 8, 0, 4, 5), None, {}),  # no frames
            ((1, 8, 3, 4, 5), None, {"n_front": -1}),
            ((1, 8, 3, 4, 5), None, {"n_back": -1}),
            ((1, 8, 3, 4, 5), None, {"pad_mode": "replicate"}),
            ((1, 8, 3, 1, 5), None, {"pad_mode": "reflect"}),  # H too small to reflect
            ((1, 8, 3, 4, 1), None, {"pad_mode": "reflect"}),  # W too small to reflect
            ((1, 8, 3, 4, 5), (1, 4, 2, 4, 5), {"n_front": 0}),  # prefix channels
            ((1, 8, 3, 4, 5), (1, 8, 2, 4, 6), {"n_front": 0}),  # prefix width
            ((1, 8, 3, 4, 5), (8, 2, 4, 5), {"n_front": 0}),  # 4D prefix
        ],
    )
    def test_rejects_bad_inputs(self, x_shape, prefix_shape, kwargs):
        x = torch.randn(x_shape)
        prefix = None if prefix_shape is None else torch.randn(prefix_shape)
        with pytest.raises(ValueError):
            ck.pixel_norm_silu_pad3d(x, prefix, **kwargs)

    def test_rejects_prefix_dtype(self):
        x = torch.randn(1, 8, 3, 4, 5)
        with pytest.raises(ValueError):
            ck.pixel_norm_silu_pad3d(x, torch.randn(1, 8, 2, 4, 5).half(), n_front=0)

    def test_fake_shape(self):
        x = torch.empty(2, 8, 3, 5, 7, device="meta")
        pre = torch.empty(2, 8, 2, 5, 7, device="meta")
        assert ck.pixel_norm_silu_pad3d(x).shape == (2, 8, 4, 5, 7)
        assert ck.pixel_norm_silu_pad3d(x, n_back=1, pad_mode="reflect").shape == (2, 8, 5, 7, 9)
        assert ck.pixel_norm_silu_pad3d(x, pre, n_front=0).shape == (2, 8, 5, 5, 7)
        assert ck.pixel_norm_silu_pad3d(x, pre, 0, 1, "reflect").shape == (2, 8, 6, 7, 9)

    def test_cpu_goes_to_eager(self):
        kwargs = {
            "x": torch.randn(1, 8, 3, 4, 5, dtype=torch.bfloat16),
            "prefix": None,
            "n_front": 1,
            "n_back": 0,
            "pad_mode": "zeros",
            "eps": 1e-8,
        }
        assert registry.get_capable_backend(OP, kwargs) == "eager"


class TestPixelNormSiluPad3dCallRule:
    # the Triton call rule only looks at shapes and dtypes, so it runs on meta tensors
    @pytest.mark.parametrize(
        ("x_shape", "prefix_shape", "n_front", "n_back", "ok"),
        [
            ((1, 128, 8, 176, 192), None, 1, 0, True),
            ((1, 512, 16, 88, 96), (1, 512, 2, 88, 96), 0, 1, True),
            ((1, 512, 16, 88, 96), (1, 512, 0, 88, 96), 0, 0, True),  # empty prefix
            ((0, 8, 2, 4, 4), None, 1, 0, False),  # empty
            ((1, 0, 2, 4, 4), None, 1, 0, False),  # empty
            ((1, 8, 2, 0, 4), None, 1, 0, False),  # empty
            ((1, 8, 65534, 1, 1), None, 1, 0, True),  # 65535 output frames
            ((1, 8, 65534, 1, 1), None, 1, 1, False),  # 65536 output frames (grid)
            ((1, 8, 65533, 1, 1), (1, 8, 2, 1, 1), 0, 1, False),
        ],
    )
    def test_call_rule(self, x_shape, prefix_shape, n_front, n_back, ok):
        x = torch.empty(x_shape, device="meta", dtype=torch.bfloat16)
        prefix = None
        if prefix_shape is not None:
            prefix = torch.empty(prefix_shape, device="meta", dtype=torch.bfloat16)
        kwargs = {"x": x, "prefix": prefix, "n_front": n_front, "n_back": n_back}
        assert _call_rule()(kwargs).success == ok

    def test_call_rule_prefix_dtype(self):
        x = torch.empty(1, 8, 2, 4, 4, device="meta", dtype=torch.bfloat16)
        prefix = torch.empty(1, 8, 2, 4, 4, device="meta", dtype=torch.float16)
        kwargs = {"x": x, "prefix": prefix, "n_front": 0, "n_back": 0}
        assert not _call_rule()(kwargs).success

    def test_half_dtypes_only(self):
        triton_backend = importlib.import_module("comfy_kitchen.backends.triton")
        params = triton_backend._build_constraints()[OP].params
        for name in ("x", "prefix"):
            assert params[name].dtypes == frozenset({torch.float16, torch.bfloat16})


# Shape classes of the LTX-2.3 (zeros) and LTXV 0.9 (reflect) decoder calls: C, H and W as
# captured, T scaled down; "view" = x is a time chunk of a longer tensor (non-contiguous), a
# prefix is the last 2 frames of a longer tensor (a view with a storage offset).
# (C, H, W, x_view, has_prefix, n_front, n_back, pad_mode)
_SHAPE_CLASSES = [
    (1024, 22, 24, False, False, 1, 1, "zeros"),  # LTX-2.3 up_blocks.0
    (1024, 22, 18, False, False, 1, 1, "zeros"),
    (512, 44, 48, True, False, 1, 1, "zeros"),
    (512, 88, 96, True, False, 1, 0, "zeros"),
    (512, 88, 96, False, True, 0, 0, "zeros"),
    (512, 88, 72, True, True, 0, 1, "zeros"),
    (256, 88, 96, False, True, 0, 1, "zeros"),
    (256, 88, 72, True, False, 1, 0, "zeros"),
    (128, 176, 192, True, True, 0, 1, "zeros"),
    (128, 176, 144, False, False, 1, 0, "zeros"),
    (512, 16, 24, False, False, 1, 1, "reflect"),  # LTXV 0.9 up_blocks.0
    (256, 64, 96, True, False, 1, 0, "reflect"),
    (256, 64, 96, False, True, 0, 1, "reflect"),
    (128, 128, 192, True, True, 0, 0, "reflect"),
    (128, 64, 96, False, False, 1, 0, "reflect"),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestPixelNormSiluPad3dTriton:
    @pytest.fixture(autouse=True)
    def _triton(self):
        pytest.importorskip("triton")
        if not registry.is_available("triton"):
            pytest.skip("triton backend not available")

    @staticmethod
    def _inputs(c, h, w, x_view, has_prefix, dtype, t=3, b=1):
        if x_view:
            x = _chunk_view(b, c, t, h, w, 2, device="cuda", dtype=dtype)
        else:
            x = torch.randn(b, c, t, h, w, device="cuda").to(dtype)
        prefix = None
        if has_prefix:
            prefix = _chunk_view(b, c, 2, h, w, 3, device="cuda", dtype=dtype)
        return x, prefix

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize(
        ("c", "h", "w", "x_view", "has_prefix", "n_front", "n_back", "pad_mode"), _SHAPE_CLASSES
    )
    def test_matches_eager(self, c, h, w, x_view, has_prefix, n_front, n_back, pad_mode, dtype):
        x, prefix = self._inputs(c, h, w, x_view, has_prefix, dtype)
        kwargs = {
            "x": x,
            "prefix": prefix,
            "n_front": n_front,
            "n_back": n_back,
            "pad_mode": pad_mode,
            "eps": 1e-8,
        }
        assert registry.validate_backend_for_call("triton", OP, kwargs).success
        assert registry.get_capable_backend(OP, kwargs) == "triton"
        with ck.use_backend("triton"):
            got = ck.pixel_norm_silu_pad3d(x, prefix, n_front, n_back, pad_mode)
        ref = eager_backend.pixel_norm_silu_pad3d(**kwargs)
        ref32 = _reference32(x, prefix, n_front, n_back, pad_mode)
        assert got.shape == ref.shape and got.dtype == dtype and got.is_contiguous()
        tol = 1e-2 if dtype == torch.bfloat16 else 2e-3
        assert _rel(got, ref) < tol
        assert _rel(got, ref32) < tol
        # one rounding: never further from the fp32 reference than the eager chain
        assert (got.float() - ref32).abs().max() <= (ref.float() - ref32).abs().max()
        if prefix is not None:
            assert torch.equal(got[:, :, : prefix.shape[2]], _pad_hw(prefix, pad_mode))

    @pytest.mark.parametrize("pad_mode", ["zeros", "reflect"])
    def test_copies_are_exact(self, pad_mode):
        # repeated frames and the reflected border recompute their source: bit-equal copies
        x = torch.randn(1, 96, 3, 9, 11, device="cuda").to(torch.bfloat16)
        with ck.use_backend("triton"):
            out = ck.pixel_norm_silu_pad3d(x, None, 2, 2, pad_mode)
        assert torch.equal(out[:, :, 0], out[:, :, 2]) and torch.equal(out[:, :, 1], out[:, :, 2])
        assert torch.equal(out[:, :, -1], out[:, :, -3]) and torch.equal(
            out[:, :, -2], out[:, :, -3]
        )
        if pad_mode == "reflect":
            assert torch.equal(out[..., 0, :], out[..., 2, :])
            assert torch.equal(out[..., -1, :], out[..., -3, :])
            assert torch.equal(out[..., :, 0], out[..., :, 2])
            assert torch.equal(out[..., :, -1], out[..., :, -3])

    @pytest.mark.parametrize("pad_mode", ["zeros", "reflect"])
    @pytest.mark.parametrize("chunks", [[3, 2, 4], [1, 1, 3]])
    def test_chunked_decode_matches_stock_causal_conv(self, chunks, pad_mode):
        x = torch.randn(1, 128, sum(chunks), 12, 16, device="cuda").to(torch.bfloat16)
        with ck.use_backend("triton"):
            got = _kitchen_chunked(x, chunks, pad_mode)
        ref = _stock_chunked(x, chunks, pad_mode)
        for g, r in zip(got, ref, strict=True):
            assert g.shape == r.shape
            assert _rel(g, r) < 1e-2

    @pytest.mark.parametrize(
        "layout",
        [
            "odd_sizes",  # batch 2, C not a power of 2, H / W not multiples of any tile
            "channels_last_3d",
            "last_dim_step2",  # x[..., ::2]
            "batch_step2",  # x[::2]
            "empty_prefix",  # x[:, :, :0] cache
        ],
    )
    def test_kernel_layouts(self, layout):
        cl3 = torch.channels_last_3d
        prefix, n_front = None, 1
        if layout == "odd_sizes":
            x = torch.randn(2, 160, 3, 7, 13, device="cuda")
            prefix, n_front = torch.randn(2, 160, 2, 7, 13, device="cuda"), 0
        elif layout == "channels_last_3d":
            x = torch.randn(1, 64, 3, 6, 10, device="cuda").to(memory_format=cl3)
            prefix, n_front = torch.randn(1, 64, 2, 6, 10, device="cuda").to(memory_format=cl3), 0
        elif layout == "last_dim_step2":
            x = torch.randn(1, 64, 3, 6, 20, device="cuda")[..., ::2]
        elif layout == "batch_step2":
            x = torch.randn(4, 64, 3, 6, 10, device="cuda")[::2]
        else:
            x = torch.randn(1, 64, 3, 6, 10, device="cuda")
            prefix, n_front = x[:, :, :0], 0
        x = x.to(torch.bfloat16)
        prefix = None if prefix is None else prefix.to(torch.bfloat16)
        for pad_mode in ("zeros", "reflect"):
            kwargs = {
                "x": x,
                "prefix": prefix,
                "n_front": n_front,
                "n_back": 1,
                "pad_mode": pad_mode,
                "eps": 1e-8,
            }
            assert registry.get_capable_backend(OP, kwargs) == "triton"
            with ck.use_backend("triton"):
                got = ck.pixel_norm_silu_pad3d(x, prefix, n_front, 1, pad_mode)
            with ck.use_backend("triton"):
                want = ck.pixel_norm_silu_pad3d(
                    x.contiguous(),
                    None if prefix is None else prefix.contiguous(),
                    n_front,
                    1,
                    pad_mode,
                )
            assert got.is_contiguous()
            assert torch.equal(got, want)
            assert _rel(got, eager_backend.pixel_norm_silu_pad3d(**kwargs)) < 1e-2

    @pytest.mark.parametrize("hip", [False, True])
    @pytest.mark.parametrize("c", [128, 512, 1024])
    def test_both_launch_configs(self, monkeypatch, c, hip):
        # the CUDA (one pass) and HIP (two passes over C) configurations, on whichever GPU runs
        kernel = importlib.import_module("comfy_kitchen.backends.triton." + OP)
        monkeypatch.setattr(kernel, "_HIP", hip)
        assert kernel._launch_config(c)[0] == (not hip)
        x, prefix = self._inputs(c, 12, 20, True, True, torch.bfloat16)
        for pad_mode in ("zeros", "reflect"):
            got = kernel.pixel_norm_silu_pad3d(x, prefix, 0, 1, pad_mode)
            ref32 = _reference32(x, prefix, 0, 1, pad_mode)
            ref = eager_backend.pixel_norm_silu_pad3d(x, prefix, 0, 1, pad_mode)
            assert _rel(got, ref32) < 1e-2
            assert (got.float() - ref32).abs().max() <= (ref.float() - ref32).abs().max()

    def test_real_shape_served(self):
        # LTX-2.3 up_blocks.3 call at its captured size: a time chunk with a temporal cache
        x = _chunk_view(1, 128, 11, 176, 192, 2, device="cuda", dtype=torch.bfloat16)
        prefix = _chunk_view(1, 128, 2, 176, 192, 4, device="cuda", dtype=torch.bfloat16)
        kwargs = {
            "x": x,
            "prefix": prefix,
            "n_front": 0,
            "n_back": 1,
            "pad_mode": "zeros",
            "eps": 1e-8,
        }
        assert registry.get_capable_backend(OP, kwargs) == "triton"
        with ck.use_backend("triton"):
            got = ck.pixel_norm_silu_pad3d(x, prefix, 0, 1)
        assert _rel(got, eager_backend.pixel_norm_silu_pad3d(**kwargs)) < 1e-2

    def test_empty_input_goes_to_eager(self):
        x = torch.randn(0, 8, 2, 4, 4, device="cuda", dtype=torch.bfloat16)
        kwargs = {"x": x, "prefix": None, "n_front": 1, "n_back": 1, "pad_mode": "zeros"}
        assert not registry.validate_backend_for_call("triton", OP, kwargs).success
        with ck.use_backend("triton"):
            got = ck.pixel_norm_silu_pad3d(x, None, 1, 1)
        assert got.shape == (0, 8, 4, 4, 4) and got.dtype == x.dtype

    def test_unsupported_inputs_go_to_eager(self):
        # fp32 is not served by the triton kernel
        x = torch.randn(1, 8, 2, 4, 4, device="cuda")
        kwargs = {"x": x, "prefix": None, "n_front": 1, "n_back": 0, "pad_mode": "zeros"}
        assert not registry.validate_backend_for_call("triton", OP, kwargs).success
        assert registry.get_capable_backend(OP, kwargs) == "eager"
        with ck.use_backend("triton"):
            got = ck.pixel_norm_silu_pad3d(x)
        assert torch.equal(got, _stock_chain(x, None, 1, 0, "zeros"))
        # CPU tensors go to eager
        kwargs["x"] = torch.randn(1, 8, 2, 4, 4, dtype=torch.bfloat16)
        assert not registry.validate_backend_for_call("triton", OP, kwargs).success
        assert registry.get_capable_backend(OP, kwargs) == "eager"
