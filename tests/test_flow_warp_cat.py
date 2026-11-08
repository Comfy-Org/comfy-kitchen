# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""flow_warp_cat -- FILM bilinear backward warp + channel concat.

The eager backend is the sequence ComfyUI's FILM frame interpolation runs today (the
PyramidFlowEstimator step: _warp_core + the FlowEstimator torch.cat; a fusion level:
multiply_pyramid of both flows, two _warp_core and the aligned torch.cat), so it must match
the stock code exactly. The Triton kernel follows ATen's operation order for the grid and the
bilinear interpolation and rounds once to fp16, so it must be equal to the stock code too.
"""

import importlib

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

import comfy_kitchen as ck
from comfy_kitchen.backends import eager as eager_backend
from comfy_kitchen.registry import registry

OP = "flow_warp_cat"


# --- comfy_extras/frame_interpolation_models/film_net.py (ComfyUI fb2b26f), verbatim ---


def _warp_core(image, flow, grid_x, grid_y):
    dtype = image.dtype
    H, W = flow.shape[2], flow.shape[3]  # noqa: N806
    dx = flow[:, 0].float() / (W * 0.5)
    dy = flow[:, 1].float() / (H * 0.5)
    grid = torch.stack([grid_x[None, None, :] + dx, grid_y[None, :, None] + dy], dim=3)
    return F.grid_sample(
        image.float(), grid, mode="bilinear", padding_mode="border", align_corners=False
    ).to(dtype)


def multiply_pyramid(pyramid, scalar):
    return [image * scalar[:, None, None, None] for image in pyramid]


def pyramid_warp(feature_pyramid, flow_pyramid, warp_fn):
    return [warp_fn(features, flow) for features, flow in zip(feature_pyramid, flow_pyramid)]  # noqa: B905


def _grids(height, width, device="cpu"):
    # FILMNet._build_warp_grids for one level
    return (
        torch.linspace(-(1 - 1 / width), 1 - 1 / width, width, dtype=torch.float32, device=device),
        torch.linspace(
            -(1 - 1 / height), 1 - 1 / height, height, dtype=torch.float32, device=device
        ),
    )


def _stock_estimator_step(fa, fb, v, grid_x, grid_y):
    # PyramidFlowEstimator.forward: predictor(feature_pyramid_a[i], warp_fn(feature_pyramid_b[i], v))
    # with FlowEstimator.forward's net = torch.cat([features_a, features_b], dim=1)
    return torch.cat([fa, _warp_core(fb, v, grid_x, grid_y)], dim=1)


def _stock_fusion_levels(p2w0, p2w1, fwd_flow, bwd_flow, batch_dt, grids):
    # FILMNet.forward_multi_timestep, one timestep: every fusion level's aligned tensor
    def warp(image, flow):
        return _warp_core(image, flow, *grids[(flow.shape[2], flow.shape[3])])

    bwd_scaled = multiply_pyramid(bwd_flow, batch_dt)
    fwd_scaled = multiply_pyramid(fwd_flow, 1 - batch_dt)
    fwd_warped = pyramid_warp(p2w0, bwd_scaled, warp)
    bwd_warped = pyramid_warp(p2w1, fwd_scaled, warp)
    return [
        torch.cat([fw, bw, bf, ff], dim=1)
        for fw, bw, bf, ff in zip(fwd_warped, bwd_warped, bwd_scaled, fwd_scaled)  # noqa: B905
    ]


# --- inputs ---


def _flow(height, width, *, device="cpu", dtype=torch.float16, scale=4.0, layout="nchw"):
    # pixel-unit flows: mostly small, some far outside the frame (border clamp), some on
    # integer offsets (taps exactly on pixels)
    f = torch.randn(1, 2, height, width, device=device) * scale
    f[:, :, ::7] *= 50.0
    f[:, :, 1::5] = torch.round(f[:, :, 1::5])
    f = f.to(dtype)
    return f.contiguous(memory_format=torch.channels_last) if layout == "cl" else f


def _image(channels, height, width, *, device="cpu", dtype=torch.float16, layout="nchw"):
    x = torch.randn(1, channels, height, width, device=device).to(dtype)
    return x.contiguous(memory_format=torch.channels_last) if layout == "cl" else x


def _estimator_inputs(channels, height, width, **kw):
    fa = _image(channels, height, width, **kw)
    fb = _image(channels, height, width, **kw)
    v = _flow(height, width, **kw)
    return fa, fb, v


def _fusion_inputs(channels, height, width, *, dt=0.5, **kw):
    p0 = _image(channels, height, width, **kw)
    p1 = _image(channels, height, width, **kw)
    fwd = _flow(height, width, **kw)
    bwd = _flow(height, width, **kw)
    batch_dt = torch.tensor([dt], device=p0.device, dtype=p0.dtype)
    return p0, p1, fwd, bwd, batch_dt


def _kwargs(images, flows, grid_x, grid_y, prefix=None, scales=None):
    return {
        "images": images,
        "flows": flows,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "prefix": prefix,
        "scales": scales,
    }


def _call_rule():
    triton_backend = importlib.import_module("comfy_kitchen.backends.triton")
    (rule,) = triton_backend._build_constraints()[OP].call_rules
    return rule


# Real FILM shapes (film_net_fp16, 1920 x 1072 frames): flow estimator steps at
# (C, H, W) = (64, 1072, 1920), (192, 536, 960), (448, 268, 480), (960, 134, 240),
# (960, 67, 120), (960, 33, 60); fusion levels at C = 67, 195, 451, 963, 963 over the first
# five. Scaled down here: the same channel counts on a 1/16 frame (67 x 120, halving, odd sizes
# included).
_ESTIMATOR_SHAPES = [(64, 67, 120), (192, 33, 60), (448, 16, 30), (960, 8, 15), (960, 4, 7)]
_FUSION_SHAPES = [(67, 67, 120), (195, 33, 60), (451, 16, 30), (963, 8, 15), (963, 4, 7)]


class TestFlowWarpCatEager:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("layout", ["nchw", "cl"])
    @pytest.mark.parametrize(("channels", "height", "width"), _ESTIMATOR_SHAPES)
    def test_estimator_matches_stock(self, channels, height, width, layout, dtype):
        fa, fb, v = _estimator_inputs(channels, height, width, dtype=dtype, layout=layout)
        gx, gy = _grids(height, width)
        with ck.use_backend("eager"):
            out = ck.flow_warp_cat(fb, v, gx, gy, prefix=fa)
        ref = _stock_estimator_step(fa, fb, v, gx, gy)
        assert out.dtype == ref.dtype and out.shape == ref.shape
        assert out.stride() == ref.stride()
        assert torch.equal(out, ref)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("layout", ["nchw", "cl"])
    @pytest.mark.parametrize(("channels", "height", "width"), _FUSION_SHAPES)
    def test_fusion_level_matches_stock(self, channels, height, width, layout, dtype):
        p0, p1, fwd, bwd, dt = _fusion_inputs(channels, height, width, dtype=dtype, layout=layout)
        gx, gy = _grids(height, width)
        with ck.use_backend("eager"):
            out = ck.flow_warp_cat([p0, p1], [bwd, fwd], gx, gy, scales=[dt, 1 - dt])
        (ref,) = _stock_fusion_levels([p0], [p1], [fwd], [bwd], dt, {(height, width): (gx, gy)})
        assert out.dtype == ref.dtype and out.shape == ref.shape
        assert out.stride() == ref.stride()
        assert torch.equal(out, ref)

    @pytest.mark.parametrize("dt", [0.5, 0.25, 0.875])
    def test_fusion_pyramid(self, dt):
        # every level of one timestep, as forward_multi_timestep builds them
        shapes = _FUSION_SHAPES[:3]
        grids = {(h, w): _grids(h, w) for _, h, w in shapes}
        inputs = [_fusion_inputs(c, h, w, dt=dt) for c, h, w in shapes]
        p2w0, p2w1, fwd, bwd = ([x[i] for x in inputs] for i in range(4))
        batch_dt = inputs[0][4]
        refs = _stock_fusion_levels(p2w0, p2w1, fwd, bwd, batch_dt, grids)
        with ck.use_backend("eager"):
            for i, (_, h, w) in enumerate(shapes):
                out = ck.flow_warp_cat(
                    [p2w0[i], p2w1[i]],
                    [bwd[i], fwd[i]],
                    *grids[(h, w)],
                    scales=[batch_dt, 1 - batch_dt],
                )
                assert torch.equal(out, refs[i])

    def test_list_and_tensor_forms(self):
        fa, fb, v = _estimator_inputs(8, 12, 20)
        gx, gy = _grids(12, 20)
        with ck.use_backend("eager"):
            from_tensor = ck.flow_warp_cat(fb, v, gx, gy, prefix=fa)
            from_list = ck.flow_warp_cat([fb], [v], gx, gy, prefix=fa)
            no_prefix = ck.flow_warp_cat(fb, v, gx, gy)
        assert torch.equal(from_tensor, from_list)
        assert torch.equal(no_prefix, _warp_core(fb, v, gx, gy))

    @pytest.mark.parametrize(
        ("image_shapes", "flow_shapes", "prefix_shape", "n_scales"),
        [
            ([], [], None, None),  # no images
            ([(1, 8, 12, 20)], [], None, None),  # no flows
            ([(1, 8, 12, 20)] * 2, [(1, 2, 12, 20)], None, None),  # lengths differ
            ([(1, 8, 12, 20)], [(1, 3, 12, 20)], None, None),  # 3-channel flow
            ([(8, 12, 20)], [(1, 2, 12, 20)], None, None),  # 3D image
            ([(1, 8, 12, 20)] * 2, [(1, 2, 12, 20), (1, 2, 12, 18)], None, 2),  # flows disagree
            ([(2, 8, 12, 20)], [(1, 2, 12, 20)], None, None),  # batch differs
            ([(1, 8, 12, 20)], [(1, 2, 12, 20)], (1, 8, 12, 18), None),  # prefix W
            ([(1, 8, 12, 20)] * 2, [(1, 2, 12, 20)] * 2, None, 1),  # one scale for two flows
        ],
    )
    def test_rejects_bad_inputs(self, image_shapes, flow_shapes, prefix_shape, n_scales):
        images = [torch.randn(s) for s in image_shapes]
        flows = [torch.randn(s) for s in flow_shapes]
        prefix = None if prefix_shape is None else torch.randn(prefix_shape)
        scales = None if n_scales is None else [torch.ones(1)] * n_scales
        gx, gy = _grids(12, 20)
        with pytest.raises(ValueError):
            ck.flow_warp_cat(images, flows, gx, gy, prefix=prefix, scales=scales)

    def test_rejects_bad_grids_and_scales(self):
        fa, fb, v = _estimator_inputs(8, 12, 20, dtype=torch.float32)
        gx, gy = _grids(12, 20)
        with pytest.raises(ValueError):
            ck.flow_warp_cat(fb, v, gx[None], gy, prefix=fa)
        with pytest.raises(ValueError):
            ck.flow_warp_cat([fb, fb], [v, v], gx, gy, scales=[torch.tensor(0.5)] * 2)

    def test_fake_shape_and_dtype(self):
        meta = {"device": "meta", "dtype": torch.float16}
        gx = torch.empty(1920, device="meta")
        gy = torch.empty(1072, device="meta")
        fa = torch.empty(1, 64, 1072, 1920, **meta)
        v = torch.empty(1, 2, 1072, 1920, **meta)
        out = ck.flow_warp_cat(fa, v, gx, gy, prefix=fa)
        assert out.shape == (1, 128, 1072, 1920) and out.dtype == torch.float16
        assert out.is_contiguous()
        p = torch.empty(1, 67, 1072, 1920, **meta)
        dt = torch.empty(1, **meta)
        out = ck.flow_warp_cat([p, p], [v, v], gx, gy, scales=[dt, dt])
        assert out.shape == (1, 2 * 67 + 4, 1072, 1920) and out.dtype == torch.float16

    def test_cpu_goes_to_eager(self):
        fa, fb, v = _estimator_inputs(8, 12, 20)
        kwargs = _kwargs([fb], [v], *_grids(12, 20), prefix=fa)
        assert registry.get_capable_backend(OP, kwargs) == "eager"


def _meta(shape, dtype=torch.float16, layout="nchw"):
    t = torch.empty(shape, device="meta", dtype=dtype)
    return t.contiguous(memory_format=torch.channels_last) if layout == "cl" else t


def _meta_estimator(channels=64, height=1072, width=1920, layout="cl", dtype=torch.float16):
    fa = _meta((1, channels, height, width), dtype, layout)
    fb = _meta((1, channels, height, width), dtype, layout)
    v = _meta((1, 2, height, width), dtype, layout)
    gx = torch.empty(width, device="meta")
    gy = torch.empty(height, device="meta")
    return _kwargs([fb], [v], gx, gy, prefix=fa)


def _meta_fusion(channels=67, height=1072, width=1920, layout="cl", dtype=torch.float16):
    p = [_meta((1, channels, height, width), dtype, layout) for _ in range(2)]
    f = [_meta((1, 2, height, width), dtype, layout) for _ in range(2)]
    s = [torch.empty(1, device="meta", dtype=dtype) for _ in range(2)]
    gx = torch.empty(width, device="meta")
    gy = torch.empty(height, device="meta")
    return _kwargs(p, f, gx, gy, scales=s)


class TestFlowWarpCatCallRule:
    # the Triton call rule only looks at shapes, dtypes and layouts, so it runs on meta tensors
    @pytest.mark.parametrize("layout", ["nchw", "cl"])
    @pytest.mark.parametrize(
        ("channels", "height", "width"),
        [(64, 1072, 1920), (192, 536, 960), (448, 268, 480), (960, 134, 240), (960, 33, 60)],
    )
    def test_real_estimator_shapes(self, channels, height, width, layout):
        kwargs = _meta_estimator(channels, height, width, layout)
        assert _call_rule()(kwargs).success

    @pytest.mark.parametrize("layout", ["nchw", "cl"])
    @pytest.mark.parametrize(
        ("channels", "height", "width"),
        [(67, 1072, 1920), (195, 536, 960), (451, 268, 480), (963, 134, 240), (963, 67, 120)],
    )
    def test_real_fusion_shapes(self, channels, height, width, layout):
        kwargs = _meta_fusion(channels, height, width, layout)
        assert _call_rule()(kwargs).success

    def test_declines(self):
        rule = _call_rule()
        # dtypes not measured
        for dtype in (torch.float32, torch.bfloat16):
            assert not rule(_meta_estimator(dtype=dtype)).success
            assert not rule(_meta_fusion(dtype=dtype)).success
        # batch 2
        kwargs = _meta_estimator(8, 64, 96)
        kwargs["images"] = [_meta((2, 8, 64, 96))]
        kwargs["flows"] = [_meta((2, 2, 64, 96))]
        kwargs["prefix"] = _meta((2, 8, 64, 96))
        assert not rule(kwargs).success
        # forms other than (one image, prefix) and (two images, scales)
        kwargs = _meta_estimator(8, 64, 96)
        assert not rule({**kwargs, "prefix": None}).success
        kwargs = _meta_fusion(8, 64, 96)
        assert not rule({**kwargs, "prefix": _meta((1, 8, 64, 96))}).success
        assert not rule({**kwargs, "scales": None}).success
        assert not rule({**kwargs, "scales": kwargs["scales"][:1]}).success
        # an fp32 scale, a 2-element scale
        assert not rule({**kwargs, "scales": [torch.empty(1, device="meta")] * 2}).success
        assert not rule({**kwargs, "scales": [_meta((2,))] * 2}).success
        # the two images or the two flows with different layouts
        mixed = [_meta((1, 8, 64, 96), layout="cl"), _meta((1, 8, 64, 96))]
        assert not rule({**kwargs, "images": mixed}).success
        mixed = [_meta((1, 2, 64, 96), layout="cl"), _meta((1, 2, 64, 96))]
        assert not rule({**kwargs, "flows": mixed}).success
        # image and flow sizes differ (the grid follows the flow, the taps the image)
        kwargs = _meta_estimator(8, 64, 96)
        assert not rule({**kwargs, "images": [_meta((1, 8, 32, 48))]}).success
        # grids of the wrong length or strided
        assert not rule({**kwargs, "grid_x": torch.empty(95, device="meta")}).success
        assert not rule({**kwargs, "grid_y": torch.empty(1, device="meta")}).success
        assert not rule({**kwargs, "grid_x": torch.empty(192, device="meta")[::2]}).success
        # empty, and element offsets past int32
        assert not rule(_meta_estimator(0, 64, 96)).success
        assert not rule(_meta_estimator(1024, 1024, 2048)).success

    def test_fp16_and_fp32_grids_only(self):
        triton_backend = importlib.import_module("comfy_kitchen.backends.triton")
        params = triton_backend._build_constraints()[OP].params
        assert params["prefix"].dtypes == frozenset({torch.float16})
        assert params["grid_x"].dtypes == frozenset({torch.float32})
        assert params["grid_y"].dtypes == frozenset({torch.float32})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestFlowWarpCatTriton:
    @pytest.fixture(autouse=True)
    def _triton(self):
        pytest.importorskip("triton")
        if not registry.is_available("triton"):
            pytest.skip("triton backend not available")

    @staticmethod
    def _served(kwargs):
        assert registry.validate_backend_for_call("triton", OP, kwargs).success
        assert registry.get_capable_backend(OP, kwargs) == "triton"

    @pytest.mark.parametrize("layout", ["nchw", "cl"])
    @pytest.mark.parametrize(("channels", "height", "width"), _ESTIMATOR_SHAPES)
    def test_estimator_equals_stock(self, channels, height, width, layout):
        fa, fb, v = _estimator_inputs(channels, height, width, device="cuda", layout=layout)
        gx, gy = _grids(height, width, "cuda")
        self._served(_kwargs([fb], [v], gx, gy, prefix=fa))
        with ck.use_backend("triton"):
            got = ck.flow_warp_cat(fb, v, gx, gy, prefix=fa)
        ref = _stock_estimator_step(fa, fb, v, gx, gy)
        assert got.shape == ref.shape and got.stride() == ref.stride()
        assert got.dtype == torch.float16
        assert torch.equal(got, ref)

    @pytest.mark.parametrize("layout", ["nchw", "cl"])
    @pytest.mark.parametrize(("channels", "height", "width"), _FUSION_SHAPES)
    def test_fusion_level_equals_stock(self, channels, height, width, layout):
        p0, p1, fwd, bwd, dt = _fusion_inputs(channels, height, width, device="cuda", layout=layout)
        gx, gy = _grids(height, width, "cuda")
        scales = [dt, 1 - dt]
        self._served(_kwargs([p0, p1], [bwd, fwd], gx, gy, scales=scales))
        with ck.use_backend("triton"):
            got = ck.flow_warp_cat([p0, p1], [bwd, fwd], gx, gy, scales=scales)
        (ref,) = _stock_fusion_levels([p0], [p1], [fwd], [bwd], dt, {(height, width): (gx, gy)})
        assert got.shape == ref.shape and got.stride() == ref.stride()
        assert torch.equal(got, ref)

    @pytest.mark.parametrize("dt", [0.25, 0.875])
    def test_other_timesteps(self, dt):
        p0, p1, fwd, bwd, batch_dt = _fusion_inputs(67, 67, 120, dt=dt, device="cuda")
        gx, gy = _grids(67, 120, "cuda")
        with ck.use_backend("triton"):
            got = ck.flow_warp_cat([p0, p1], [bwd, fwd], gx, gy, scales=[batch_dt, 1 - batch_dt])
        (ref,) = _stock_fusion_levels([p0], [p1], [fwd], [bwd], batch_dt, {(67, 120): (gx, gy)})
        assert torch.equal(got, ref)

    def test_prefix_layout_differs(self):
        # the copied prefix is read through its own strides
        fa, fb, v = _estimator_inputs(64, 33, 60, device="cuda", layout="cl")
        fa = fa.contiguous()
        gx, gy = _grids(33, 60, "cuda")
        with ck.use_backend("triton"):
            got = ck.flow_warp_cat(fb, v, gx, gy, prefix=fa)
        assert torch.equal(got, _stock_estimator_step(fa, fb, v, gx, gy))

    @pytest.mark.parametrize("layout", ["nchw", "cl"])
    def test_real_shapes_served(self, layout):
        # the largest calls of a 1920 x 1072 interpolation: estimator level 0, fusion level 0
        fa, fb, v = _estimator_inputs(64, 1072, 1920, device="cuda", layout=layout)
        gx, gy = _grids(1072, 1920, "cuda")
        self._served(_kwargs([fb], [v], gx, gy, prefix=fa))
        with ck.use_backend("triton"):
            got = ck.flow_warp_cat(fb, v, gx, gy, prefix=fa)
        assert got.shape == (1, 128, 1072, 1920) and got.is_contiguous()
        assert torch.equal(got, _stock_estimator_step(fa, fb, v, gx, gy))
        del fa, fb, v, got
        p0, p1, fwd, bwd, dt = _fusion_inputs(67, 1072, 1920, device="cuda", layout=layout)
        scales = [dt, 1 - dt]
        self._served(_kwargs([p0, p1], [bwd, fwd], gx, gy, scales=scales))
        with ck.use_backend("triton"):
            got = ck.flow_warp_cat([p0, p1], [bwd, fwd], gx, gy, scales=scales)
        (ref,) = _stock_fusion_levels([p0], [p1], [fwd], [bwd], dt, {(1072, 1920): (gx, gy)})
        assert got.shape == (1, 138, 1072, 1920) and got.is_contiguous()
        assert torch.equal(got, ref)

    def test_unsupported_inputs_go_to_eager(self):
        gx, gy = _grids(16, 30, "cuda")
        cases = []
        # fp32 / bf16 are not served by the triton kernel
        for dtype in (torch.float32, torch.bfloat16):
            fa, fb, v = _estimator_inputs(8, 16, 30, device="cuda", dtype=dtype)
            cases.append(_kwargs([fb], [v], gx, gy, prefix=fa))
        fa, fb, v = _estimator_inputs(8, 16, 30, device="cuda")
        # no prefix, batch 2, an fp16 grid
        cases.append(_kwargs([fb], [v], gx, gy))
        batch2 = [t.expand(2, -1, -1, -1) for t in (fa, fb, v)]
        cases.append(_kwargs([batch2[1]], [batch2[2]], gx, gy, prefix=batch2[0]))
        cases.append(_kwargs([fb], [v], gx.half(), gy, prefix=fa))
        p0, p1, fwd, bwd, dt = _fusion_inputs(8, 16, 30, device="cuda")
        # an fp32 scale, images with different layouts
        cases.append(_kwargs([p0, p1], [bwd, fwd], gx, gy, scales=[dt.float(), 1 - dt.float()]))
        cases.append(
            _kwargs(
                [p0, p1.contiguous(memory_format=torch.channels_last)],
                [bwd, fwd],
                gx,
                gy,
                scales=[dt, 1 - dt],
            )
        )
        for kwargs in cases:
            assert not registry.validate_backend_for_call("triton", OP, kwargs).success
            assert registry.get_capable_backend(OP, kwargs) == "eager"
            with ck.use_backend("triton"):
                got = ck.flow_warp_cat(**kwargs)
            assert torch.equal(got, eager_backend.flow_warp_cat(**kwargs))
        # CPU tensors go to eager
        fa, fb, v = _estimator_inputs(8, 16, 30)
        kwargs = _kwargs([fb], [v], *_grids(16, 30), prefix=fa)
        assert not registry.validate_backend_for_call("triton", OP, kwargs).success
        assert registry.get_capable_backend(OP, kwargs) == "eager"
