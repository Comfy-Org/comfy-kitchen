# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""depth_to_space3d_residual_add -- the upsampling shortcut + residual add of the video VAEs.

The eager backend is the sequence ComfyUI runs today (Wan 2.2 DupUp3D + Up_ResidualBlock,
HunyuanVideo 1.5 UpSmpl), copied below as the stock reference. The Triton kernel does the
same single add in fp32 with one rounding, so it must match eager bit for bit.
"""

import importlib

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends import eager as eager_backend
from comfy_kitchen.registry import registry

eager_module = importlib.import_module("comfy_kitchen.backends.eager.depth_to_space3d_residual_add")


# --- stock ComfyUI code (comfy/ldm/wan/vae2_2.py, comfy/ldm/hunyuan_video/vae_refiner.py) ---


def _stock_dup_up3d(x, first_chunk, factor_t, factor_s, out_channels, repeats):
    x = x.repeat_interleave(repeats, dim=1)
    x = x.view(
        x.size(0), out_channels, factor_t, factor_s, factor_s, x.size(2), x.size(3), x.size(4)
    )
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    x = x.view(
        x.size(0), out_channels, x.size(2) * factor_t, x.size(4) * factor_s, x.size(6) * factor_s
    )
    if first_chunk:
        x = x[:, :, factor_t - 1 :, :, :]
    return x


def _stock_strip_apply(fn, x, scale, strip_elems, out):
    # strip_apply(fn, x, scale=factor_s, halo=0, out=x_main) with STRIP_ELEMS = strip_elems
    n = -(-x.numel() * scale * scale // strip_elems)
    size = x.shape[-2]
    step = -(-size // n)
    for a in range(0, size, step):
        b = min(size, a + step)
        y = fn(x.narrow(-2, a, b - a)).narrow(-2, 0, (b - a) * scale)
        out.narrow(-2, a * scale, (b - a) * scale).add_(y)
    return out


def _stock_upsmpl_tail(h, x, r1, rp):
    b, c, frms, ht, wd = h.shape
    nc = c // (r1 * 2 * 2)
    h = h.reshape(b, r1, 2, 2, nc, frms, ht, wd)
    h = h.permute(0, 4, 5, 1, 6, 2, 7, 3)
    h = h.reshape(b, nc, frms * r1, ht * 2, wd * 2)

    x = x.repeat_interleave(repeats=rp, dim=1)
    b, c, frms, ht, wd = x.shape
    nc = c // (r1 * 2 * 2)
    x = x.reshape(b, r1, 2, 2, nc, frms, ht, wd)
    x = x.permute(0, 4, 5, 1, 6, 2, 7, 3)
    x = x.reshape(b, nc, frms * r1, ht * 2, wd * 2)
    return h + x


def _stock(x, residual, factor_t, factor_s, repeats, first_chunk, factors_first):
    if factors_first:
        return _stock_upsmpl_tail(residual, x, factor_t, repeats)
    oc = x.shape[1] * repeats // (factor_t * factor_s * factor_s)
    return residual + _stock_dup_up3d(x, first_chunk, factor_t, factor_s, oc, repeats)


# --- inputs ---


def _make(shape, dtype, device, t_major=False):
    # t_major: [B, T, C, H, W] in memory, the layout of the Wan 2.2 decoder's activations
    # (strides (T*C*H*W, H*W, C*H*W, W, 1) in the captured calls)
    if t_major:
        b, c, t, h, w = shape
        return torch.randn(b, t, c, h, w, device=device).to(dtype).transpose(1, 2)
    return torch.randn(shape, device=device).to(dtype)


# name: (x shape, residual shape, factor_t, factor_s, repeats, first_chunk, factors_first, t_major)
# Shapes captured from ComfyUI decodes (Wan 2.2 5B ti2v 1280x704x121 bf16, Qwen-Image 2.1
# 1024x1024 bf16, HunyuanVideo 1.5 720p tiled fp16), channels and rows scaled down; strides and
# factors as captured.
CASES = {
    "wan22_first_t2": ((1, 64, 1, 11, 20), (1, 64, 1, 22, 40), 2, 2, 8, True, False, True),
    "wan22_first_t1": ((1, 64, 1, 22, 40), (1, 32, 1, 44, 80), 1, 2, 2, True, False, True),
    "wan22_steady_t2": ((1, 64, 1, 11, 20), (1, 64, 2, 22, 40), 2, 2, 8, False, False, True),
    "wan22_steady_t2_2f": ((1, 64, 2, 11, 20), (1, 64, 4, 22, 40), 2, 2, 8, False, False, True),
    "wan22_steady_t1": ((1, 64, 4, 22, 40), (1, 32, 4, 44, 80), 1, 2, 2, False, False, True),
    "hv15_up0": ((1, 64, 2, 9, 16), (1, 512, 2, 9, 16), 2, 2, 8, False, True, False),
    "hv15_up1": ((1, 64, 4, 18, 32), (1, 256, 4, 18, 32), 2, 2, 4, False, True, False),
    "hv15_up2": ((1, 32, 8, 18, 32), (1, 64, 8, 18, 32), 1, 2, 2, False, True, False),
}

# in-place image path (Qwen-Image 2.1 decode through the Wan 2.2 VAE):
# (x shape, residual shape, factor_t, factor_s, repeats)
INPLACE_CASES = {
    "qwen21_i0": ((1, 72, 1, 16, 16), (1, 72, 1, 32, 32), 2, 2, 8),
    "qwen21_i2": ((1, 72, 1, 32, 32), (1, 36, 1, 64, 64), 2, 2, 4),
    "qwen21_i3": ((1, 36, 1, 64, 64), (1, 18, 1, 128, 128), 1, 2, 2),
}

# real captured shapes small enough for a test run (the rest are covered scaled down above)
REAL_CASES = {
    # Wan 2.2 Up_ResidualBlock 0, first chunk / steady chunk
    "wan22_W0": ((1, 1024, 1, 44, 80), (1, 1024, 1, 88, 160), 2, 2, 8, True, False, True),
    "wan22_W3": ((1, 1024, 1, 44, 80), (1, 1024, 2, 88, 160), 2, 2, 8, False, False, True),
    # HunyuanVideo 1.5 UpSmpl 0 of a 45 x 32 latent tile
    "hv15_H19": ((1, 1024, 2, 45, 32), (1, 8192, 2, 45, 32), 2, 2, 8, False, True, False),
}


def _inputs(case, dtype, device):
    xs, rs, ft, fs, rep, first, ff, t_major = case
    x = _make(xs, dtype, device, t_major)
    residual = _make(rs, dtype, device, t_major)
    kwargs = {
        "factor_t": ft,
        "factor_s": fs,
        "repeats": rep,
        "first_chunk": first,
        "factors_first": ff,
        "residual_packed": ff,
    }
    return x, residual, kwargs


def _same_layout(a, b):
    # strides of size-1 dims carry no layout
    return a.shape == b.shape and all(
        sa == sb for sa, sb, n in zip(a.stride(), b.stride(), a.shape, strict=True) if n > 1
    )


def _call_kwargs(x, residual, kwargs):
    return {"x": x, "residual": residual, **kwargs}


class TestDepthToSpace3dResidualAddEager:
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    @pytest.mark.parametrize("name", list(CASES))
    def test_matches_stock(self, name, dtype):
        x, residual, kwargs = _inputs(CASES[name], dtype, "cpu")
        with ck.use_backend("eager"):
            out = ck.depth_to_space3d_residual_add(x, residual, **kwargs)
        ref = _stock(
            x,
            residual,
            kwargs["factor_t"],
            kwargs["factor_s"],
            kwargs["repeats"],
            kwargs["first_chunk"],
            kwargs["factors_first"],
        )
        assert torch.equal(out, ref)
        assert out.stride() == ref.stride()

    @pytest.mark.parametrize("name", list(CASES))
    def test_fake_matches_eager(self, name):
        # the registered fake (meta tensors) promises the eager output's shape and layout
        x, residual, kwargs = _inputs(CASES[name], torch.bfloat16, "cpu")
        with ck.use_backend("eager"):
            out = ck.depth_to_space3d_residual_add(x, residual, **kwargs)
        fake = ck.depth_to_space3d_residual_add(x.to("meta"), residual.to("meta"), **kwargs)
        assert fake.dtype == out.dtype
        assert _same_layout(fake, out)

    @pytest.mark.parametrize("packed", [False, True])
    def test_fake_matches_eager_mixed_dtypes(self, packed):
        # eager promotes an fp32 x added to a bf16 residual to fp32; the fake must say so
        x = torch.randn(1, 16, 2, 4, 4)
        rs = (1, 128, 2, 4, 4) if packed else (1, 16, 4, 8, 8)
        residual = torch.randn(rs).to(torch.bfloat16)
        kwargs = {
            "factor_t": 2,
            "factor_s": 2,
            "repeats": 8,
            "factors_first": packed,
            "residual_packed": packed,
        }
        with ck.use_backend("eager"):
            out = ck.depth_to_space3d_residual_add(x, residual, **kwargs)
        fake = ck.depth_to_space3d_residual_add(x.to("meta"), residual.to("meta"), **kwargs)
        assert out.dtype == torch.float32
        assert fake.dtype == out.dtype
        assert _same_layout(fake, out)

    @pytest.mark.parametrize("strip_elems", [2**24, 2**12])
    @pytest.mark.parametrize("name", list(INPLACE_CASES))
    def test_inplace_matches_strip_apply(self, name, strip_elems, monkeypatch):
        # small STRIP_ELEMS splits these inputs into row strips the way the full-size ones are
        xs, rs, ft, fs, rep = INPLACE_CASES[name]
        monkeypatch.setattr(eager_module, "STRIP_ELEMS", strip_elems)
        x = torch.randn(xs).to(torch.bfloat16)
        residual = torch.randn(rs).to(torch.bfloat16)
        oc = rs[1]
        ref = _stock_strip_apply(
            lambda s: _stock_dup_up3d(s, True, ft, fs, oc, rep),
            x,
            fs,
            strip_elems,
            residual.clone(),
        )
        got = residual.clone()
        with ck.use_backend("eager"):
            ret = ck.depth_to_space3d_residual_add_(x, got, ft, fs, rep, first_chunk=True)
        assert ret is got
        assert torch.equal(got, ref)

    def test_inplace_equals_out_of_place(self):
        x = torch.randn(1, 16, 2, 6, 7).to(torch.bfloat16)
        residual = torch.randn(1, 16, 3, 12, 14).to(torch.bfloat16)
        with ck.use_backend("eager"):
            want = ck.depth_to_space3d_residual_add(x, residual, 2, 2, 8, first_chunk=True)
            got = ck.depth_to_space3d_residual_add_(x, residual.clone(), 2, 2, 8, first_chunk=True)
        assert torch.equal(got, want)

    @pytest.mark.parametrize(
        "args",
        [
            ((1, 16, 2, 4, 4), (1, 16, 4, 8, 9), 2, 2, 8, False),  # wrong residual width
            ((1, 16, 2, 4, 4), (1, 16, 4, 8, 8), 2, 2, 8, True),  # first_chunk drops a frame
            ((1, 15, 2, 4, 4), (1, 3, 4, 8, 8), 2, 2, 2, False),  # 15 * 2 channels not / 8
            ((1, 16, 4, 4), (1, 16, 8, 8), 1, 2, 4, False),  # 4-D
        ],
    )
    def test_rejects_bad_shapes(self, args):
        xs, rs, ft, fs, rep, first = args
        with pytest.raises(ValueError):
            ck.depth_to_space3d_residual_add(torch.randn(xs), torch.randn(rs), ft, fs, rep, first)

    def test_rejects_bad_packed_residual(self):
        with pytest.raises(ValueError):
            ck.depth_to_space3d_residual_add(
                torch.randn(1, 8, 2, 4, 4),
                torch.randn(1, 32, 2, 4, 4),
                2,
                2,
                8,
                factors_first=True,
                residual_packed=True,
            )

    def test_inplace_rejects_overlapping_residual(self):
        x = torch.randn(1, 16, 2, 4, 4)
        residual = torch.randn(1, 16, 4, 8, 1).expand(1, 16, 4, 8, 8)
        with pytest.raises(ValueError, match="internal overlap"):
            ck.depth_to_space3d_residual_add_(x, residual, 2, 2, 8)

    def test_inplace_rejects_x_overlapping_residual(self):
        buf = torch.randn(16 * 4 * 8 * 8)
        residual = buf.view(1, 16, 4, 8, 8)
        x = buf[: 16 * 2 * 4 * 4].view(1, 16, 2, 4, 4)
        with pytest.raises(ValueError, match="must not overlap"):
            ck.depth_to_space3d_residual_add_(x, residual, 2, 2, 8)

    def test_packed_output_is_contiguous(self):
        # factor_t 1 and H = W = 1: the sum of the two permuted views comes out strided
        x = torch.randn(1, 2, 1, 1, 1)
        residual = torch.randn(1, 8, 1, 1, 1)
        with ck.use_backend("eager"):
            out = ck.depth_to_space3d_residual_add(
                x, residual, 1, 2, 4, factors_first=True, residual_packed=True
            )
        assert out.shape == (1, 2, 1, 2, 2)
        assert out.is_contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestDepthToSpace3dResidualAddTriton:
    @pytest.fixture(autouse=True)
    def _triton(self):
        pytest.importorskip("triton")
        if not registry.is_available("triton"):
            pytest.skip("triton backend not available")

    def _check_served_and_equal(self, x, residual, kwargs):
        call = _call_kwargs(x, residual, kwargs)
        assert registry.validate_backend_for_call(
            "triton", "depth_to_space3d_residual_add", call
        ).success
        assert registry.get_capable_backend("depth_to_space3d_residual_add", call) == "triton"
        with ck.use_backend("triton"):
            got = ck.depth_to_space3d_residual_add(x, residual, **kwargs)
        ref = eager_backend.depth_to_space3d_residual_add(x, residual, **kwargs)
        assert got.dtype == ref.dtype
        assert _same_layout(got, ref)
        assert torch.equal(got, ref)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    @pytest.mark.parametrize("name", list(CASES))
    def test_matches_eager(self, name, dtype):
        x, residual, kwargs = _inputs(CASES[name], dtype, "cuda")
        self._check_served_and_equal(x, residual, kwargs)

    @pytest.mark.parametrize("name", list(REAL_CASES))
    def test_matches_eager_real_shapes(self, name):
        dtype = torch.float16 if REAL_CASES[name][6] else torch.bfloat16
        x, residual, kwargs = _inputs(REAL_CASES[name], dtype, "cuda")
        self._check_served_and_equal(x, residual, kwargs)

    @pytest.mark.parametrize(
        "layout",
        [
            "contiguous",  # Wan 2.2 form with contiguous operands
            "residual_cl3d",  # residual channels_last_3d: output keeps that layout
            "x_cl3d",  # x channels_last_3d
            "x_last_dim_step2",  # x[..., ::2]
            "batch2",  # batch 2, T-major
            "hv_h_cl3d",  # HunyuanVideo form, packed residual channels_last_3d
            "hv_x_sliced",  # HunyuanVideo form, x a frame slice of a longer tensor
        ],
    )
    def test_strided_inputs(self, layout):
        d, dev = torch.bfloat16, "cuda"
        cl3 = torch.channels_last_3d
        wan = {"factor_t": 2, "factor_s": 2, "repeats": 8}
        hv = {
            "factor_t": 2,
            "factor_s": 2,
            "repeats": 8,
            "factors_first": True,
            "residual_packed": True,
        }
        make = {
            "contiguous": lambda: (
                _make((1, 32, 2, 7, 9), d, dev),
                _make((1, 32, 4, 14, 18), d, dev),
                wan,
            ),
            "residual_cl3d": lambda: (
                _make((1, 32, 2, 7, 9), d, dev),
                _make((1, 32, 4, 14, 18), d, dev).to(memory_format=cl3),
                wan,
            ),
            "x_cl3d": lambda: (
                _make((1, 32, 2, 7, 9), d, dev).to(memory_format=cl3),
                _make((1, 32, 4, 14, 18), d, dev),
                wan,
            ),
            "x_last_dim_step2": lambda: (
                _make((1, 32, 2, 7, 18), d, dev)[..., ::2],
                _make((1, 32, 4, 14, 18), d, dev),
                wan,
            ),
            "batch2": lambda: (
                _make((2, 32, 2, 7, 9), d, dev, True),
                _make((2, 32, 4, 14, 18), d, dev, True),
                wan,
            ),
            "hv_h_cl3d": lambda: (
                _make((1, 16, 2, 7, 9), d, dev),
                _make((1, 128, 2, 7, 9), d, dev).to(memory_format=cl3),
                hv,
            ),
            "hv_x_sliced": lambda: (
                _make((1, 16, 3, 7, 9), d, dev)[:, :, 1:],
                _make((1, 128, 2, 7, 9), d, dev),
                hv,
            ),
        }
        x, residual, kwargs = make[layout]()
        self._check_served_and_equal(x, residual, kwargs)

    @pytest.mark.parametrize("name", list(INPLACE_CASES))
    def test_inplace_matches_eager(self, name):
        xs, rs, ft, fs, rep = INPLACE_CASES[name]
        x = _make(xs, torch.bfloat16, "cuda")
        residual = _make(rs, torch.bfloat16, "cuda")
        call = {
            "x": x,
            "residual": residual,
            "factor_t": ft,
            "factor_s": fs,
            "repeats": rep,
            "first_chunk": True,
        }
        assert registry.get_capable_backend("depth_to_space3d_residual_add_", call) == "triton"
        want = eager_backend.depth_to_space3d_residual_add_(x, residual.clone(), ft, fs, rep, True)
        got = residual.clone()
        with ck.use_backend("triton"):
            # the first call of a new size autotunes; the shortcut must still be added once
            ret = ck.depth_to_space3d_residual_add_(x, got, ft, fs, rep, first_chunk=True)
        assert ret is got
        assert torch.equal(got, want)

    def test_inplace_into_row_strip_view(self):
        # the residual may be a row strip (narrow along H) of a larger tensor, as strip_apply passes
        x = _make((1, 36, 1, 32, 32), torch.bfloat16, "cuda")
        full = _make((1, 18, 1, 128, 64), torch.bfloat16, "cuda")
        want = full.clone()
        eager_backend.depth_to_space3d_residual_add_(
            x[..., :8, :], want.narrow(-2, 32, 16), 1, 2, 2, True
        )
        got = full.clone()
        with ck.use_backend("triton"):
            ck.depth_to_space3d_residual_add_(
                x[..., :8, :], got.narrow(-2, 32, 16), 1, 2, 2, first_chunk=True
            )
        assert torch.equal(got, want)

    def test_declined_inputs_go_to_eager(self):
        d, dev = torch.bfloat16, "cuda"
        x, r = _make((1, 16, 2, 4, 4), d, dev), _make((1, 16, 4, 8, 8), d, dev)
        h = _make((1, 128, 2, 4, 4), d, dev)
        x3, h3 = _make((1, 9, 2, 4, 4), d, dev), _make((1, 18, 2, 4, 4), d, dev)
        declined = [
            # mixed dtypes
            (x.float(), r, {"factor_t": 2, "factor_s": 2, "repeats": 8}),
            # factors_first without a packed residual and the other way round
            (x, r, {"factor_t": 2, "factor_s": 2, "repeats": 8, "factors_first": True}),
            (x, h, {"factor_t": 2, "factor_s": 2, "repeats": 8, "residual_packed": True}),
            # factors_first with factor_s 3, or with first_chunk
            (
                x3,
                h3,
                {
                    "factor_t": 2,
                    "factor_s": 3,
                    "repeats": 2,
                    "factors_first": True,
                    "residual_packed": True,
                },
            ),
            (
                x,
                h,
                {
                    "factor_t": 2,
                    "factor_s": 2,
                    "repeats": 8,
                    "first_chunk": True,
                    "factors_first": True,
                    "residual_packed": True,
                },
            ),
        ]
        for xx, rr, kwargs in declined:
            call = _call_kwargs(xx, rr, kwargs)
            assert not registry.validate_backend_for_call(
                "triton", "depth_to_space3d_residual_add", call
            ).success
            assert registry.get_capable_backend("depth_to_space3d_residual_add", call) == "eager"
            with ck.use_backend("triton"):
                got = ck.depth_to_space3d_residual_add(xx, rr, **kwargs)
            assert torch.equal(got, eager_backend.depth_to_space3d_residual_add(xx, rr, **kwargs))

    @pytest.mark.parametrize(
        "xs, rs", [((0, 16, 2, 4, 4), (0, 16, 4, 8, 8)), ((1, 16, 2, 0, 4), (1, 16, 4, 0, 8))]
    )
    def test_empty_input_goes_to_eager(self, xs, rs):
        x, r = _make(xs, torch.bfloat16, "cuda"), _make(rs, torch.bfloat16, "cuda")
        call = {"x": x, "residual": r, "factor_t": 2, "factor_s": 2, "repeats": 8}
        assert not registry.validate_backend_for_call(
            "triton", "depth_to_space3d_residual_add", call
        ).success
        with ck.use_backend("triton"):
            got = ck.depth_to_space3d_residual_add(x, r, 2, 2, 8)
            ck.depth_to_space3d_residual_add_(x, r, 2, 2, 8)
        assert got.shape == r.shape

    def test_inplace_sharing_storage_goes_to_eager(self):
        buf = _make((16 * 2 * 4 * 4 + 16 * 4 * 8 * 8,), torch.bfloat16, "cuda")
        x = buf[: 16 * 2 * 4 * 4].view(1, 16, 2, 4, 4)
        r = buf[16 * 2 * 4 * 4 :].view(1, 16, 4, 8, 8)
        call = {"x": x, "residual": r, "factor_t": 2, "factor_s": 2, "repeats": 8}
        assert not registry.validate_backend_for_call(
            "triton", "depth_to_space3d_residual_add_", call
        ).success
        assert registry.get_capable_backend("depth_to_space3d_residual_add_", call) == "eager"
