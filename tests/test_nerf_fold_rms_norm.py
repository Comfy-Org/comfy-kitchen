# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nerf_fold_rms_norm -- Chroma Radiance NeRF head tail: tile cat + fold + channel RMSNorm.

The eager backend is the sequence ComfyUI's Chroma Radiance runs today (the torch.cat of the
NeRF tiles, the transpose / reshape / transpose and F.fold of forward_nerf, and
NerfFinalLayerConv's RMSNorm on movedim(1, -1)), so it must match the stock code exactly,
including the output layout the final conv reads. The Triton kernel computes the norm in fp32
and rounds once, so it is compared to eager and to an fp32 reference with a tolerance and must
never be further from the fp32 reference than eager.
"""

import contextlib
import importlib

import pytest
import torch
from torch import nn
from torch.nn import functional

import comfy_kitchen as ck
from comfy_kitchen.backends import eager as eager_backend
from comfy_kitchen.registry import registry

OP = "nerf_fold_rms_norm"


def _stock_tail(tiles, norm, batch, height, width, patch_size):
    # comfy/ldm/chroma_radiance/model.py (ComfyUI fb2b26f): forward_tiled_nerf's torch.cat,
    # then forward_nerf's reassembly and fold; layers.py NerfFinalLayerConv.forward up to
    # self.conv (norm = comfy.ops RMSNorm, a torch.nn.RMSNorm)
    img_dct = torch.cat(tiles, dim=0)
    num_patches = img_dct.shape[0] // batch
    img_dct = img_dct.transpose(1, 2)
    img_dct = img_dct.reshape(batch, num_patches, -1)
    img_dct = img_dct.transpose(1, 2)
    img_dct = nn.functional.fold(
        img_dct,
        output_size=(height, width),
        kernel_size=patch_size,
        stride=patch_size,
    )
    return norm(img_dct.movedim(1, -1)).movedim(-1, 1)


def _stock_norm(weight, eps=None):
    norm = nn.RMSNorm(weight.shape[0], eps=eps, device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        norm.weight.copy_(weight)
    return norm


def _cast_norm(weight, eps=None):
    # comfy.ops RMSNorm with comfy_cast_weights: F.rms_norm(input, normalized_shape, weight, eps)
    return lambda x: functional.rms_norm(x, (weight.shape[0],), weight, eps)


def _tiles(batch, height, width, patch_size, channels, tile, *, device="cpu", dtype=None):
    # NeRF block outputs [rows, P*P, C] of the tiled loop: rows [i * batch, end * batch) per
    # tile of `tile` patches, the last tile partial when the patch count is not a multiple
    rows = batch * (height // patch_size) * (width // patch_size)
    x = torch.randn(rows, patch_size * patch_size, channels, device=device)
    if dtype is not None:
        x = x.to(dtype)
    sizes = [min(tile * batch, rows - r) for r in range(0, rows, tile * batch)]
    return [t.contiguous() for t in torch.split(x, sizes, dim=0)]


def _reference32(tiles, weight, batch, height, width, patch_size, eps=None):
    tiles32 = [t.float() for t in tiles]
    w32 = weight.float()
    return _stock_tail(tiles32, _cast_norm(w32, eps), batch, height, width, patch_size)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def _kwargs(tiles, weight, batch, height, width, patch_size, eps=None):
    return {
        "tiles": tiles,
        "weight": weight,
        "batch": batch,
        "height": height,
        "width": width,
        "patch_size": patch_size,
        "eps": eps,
    }


def _call_rule():
    triton_backend = importlib.import_module("comfy_kitchen.backends.triton")
    (rule,) = triton_backend._build_constraints()[OP].call_rules
    return rule


# (batch, height, width, patch_size, channels, tile): the template call is batch 1, 1024 x 1024,
# patch 16, C 64, tile 512 (8 tiles); scaled down here, with a partial last tile and batch 2
_EAGER_SHAPES = [
    (1, 128, 128, 16, 64, 16),  # 4 full tiles
    (1, 128, 112, 16, 64, 16),  # 56 patches: 3 full tiles + 8
    (2, 64, 96, 16, 64, 10),  # batch 2
    (1, 64, 64, 16, 64, 1024),  # one tile (the untiled head)
    (1, 24, 40, 4, 12, 5),  # other P and C
]


class TestNerfFoldRmsNormEager:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    @pytest.mark.parametrize(
        ("batch", "height", "width", "patch", "channels", "tile"), _EAGER_SHAPES
    )
    def test_matches_stock_chain(self, batch, height, width, patch, channels, tile, dtype):
        tiles = _tiles(batch, height, width, patch, channels, tile, dtype=dtype)
        weight = torch.randn(channels).to(dtype)
        with ck.use_backend("eager"):
            out = ck.nerf_fold_rms_norm(tiles, weight, batch, height, width, patch)
        for norm in (_stock_norm(weight), _cast_norm(weight)):
            ref = _stock_tail(tiles, norm, batch, height, width, patch)
            assert out.dtype == dtype and out.shape == ref.shape
            assert out.stride() == ref.stride()
            assert torch.equal(out, ref)

    def test_output_layout_is_channels_innermost(self):
        tiles = _tiles(1, 64, 96, 16, 64, 8)
        with ck.use_backend("eager"):
            out = ck.nerf_fold_rms_norm(tiles, torch.randn(64), 1, 64, 96, 16)
        # what the final conv receives in the stock path: [B, C, H, W], strides [H*W*C, 1, W*C, C]
        assert out.shape == (1, 64, 64, 96)
        assert out.stride() == (64 * 96 * 64, 1, 96 * 64, 64)

    def test_eps(self):
        tiles = _tiles(1, 64, 64, 16, 64, 4, dtype=torch.bfloat16)
        weight = torch.randn(64).to(torch.bfloat16)
        with ck.use_backend("eager"):
            out = ck.nerf_fold_rms_norm(tiles, weight, 1, 64, 64, 16, eps=1e-2)
        ref = _stock_tail(tiles, _stock_norm(weight, eps=1e-2), 1, 64, 64, 16)
        assert torch.equal(out, ref)

    def test_tensor_and_list_forms(self):
        tiles = _tiles(1, 64, 64, 16, 64, 4)
        weight = torch.randn(64)
        with ck.use_backend("eager"):
            from_list = ck.nerf_fold_rms_norm(tiles, weight, 1, 64, 64, 16)
            from_tensor = ck.nerf_fold_rms_norm(torch.cat(tiles), weight, 1, 64, 64, 16)
        assert torch.equal(from_list, from_tensor)

    @pytest.mark.parametrize(
        ("tile_shapes", "weight_shape", "args"),
        [
            ([(16, 256)], (64,), (1, 64, 64, 16)),  # 2D tile
            ([(8, 256, 64), (8, 256, 32)], (64,), (1, 64, 64, 16)),  # tiles disagree on C
            ([(16, 256, 64)], (32,), (1, 64, 64, 16)),  # weight != C
            ([(16, 256, 64)], (1, 64), (1, 64, 64, 16)),  # 2D weight
            ([(16, 196, 64)], (64,), (1, 64, 64, 16)),  # 196 pixels for patch 16
            ([(16, 256, 64)], (64,), (1, 64, 60, 16)),  # width not a multiple of P
            ([(16, 256, 64)], (64,), (1, 64, 80, 16)),  # 16 rows for 20 patches
            ([(16, 256, 64)], (64,), (0, 64, 64, 16)),  # batch 0
            ([], (64,), (1, 64, 64, 16)),  # no tiles
        ],
    )
    def test_rejects_bad_inputs(self, tile_shapes, weight_shape, args):
        tiles = [torch.randn(s) for s in tile_shapes]
        with pytest.raises(ValueError):
            ck.nerf_fold_rms_norm(tiles, torch.randn(weight_shape), *args)

    def test_rejects_mixed_tile_dtypes(self):
        tiles = [torch.randn(8, 256, 64), torch.randn(8, 256, 64).to(torch.bfloat16)]
        with pytest.raises(ValueError):
            ck.nerf_fold_rms_norm(tiles, torch.randn(64), 1, 64, 64, 16)

    def test_fake_shape_and_strides(self):
        tiles = [torch.empty(s, 256, 64, device="meta") for s in (24, 24, 8)]
        weight = torch.empty(64, device="meta")
        out = ck.nerf_fold_rms_norm(tiles, weight, 1, 112, 128, 16)
        assert out.shape == (1, 64, 112, 128)
        assert out.stride() == (112 * 128 * 64, 1, 128 * 64, 64)

    def test_cpu_goes_to_eager(self):
        tiles = _tiles(1, 64, 64, 16, 64, 4, dtype=torch.bfloat16)
        kwargs = _kwargs(tiles, torch.randn(64).to(torch.bfloat16), 1, 64, 64, 16)
        assert registry.get_capable_backend(OP, kwargs) == "eager"


def _meta_tiles(rows, pp=256, channels=64, dtype=torch.bfloat16):
    return [torch.empty(r, pp, channels, device="meta", dtype=dtype) for r in rows]


class TestNerfFoldRmsNormCallRule:
    # the Triton call rule only looks at shapes, dtypes and layouts, so it runs on meta tensors
    @pytest.mark.parametrize(
        ("rows", "channels", "dtype", "weight_dtype", "ok"),
        [
            ([512] * 8, 64, torch.bfloat16, torch.bfloat16, True),  # the template call
            ([512, 512, 128], 64, torch.bfloat16, torch.bfloat16, True),  # partial last tile
            ([4096], 64, torch.bfloat16, torch.bfloat16, True),  # one tile
            ([512] * 8, 64, torch.float16, torch.float16, False),  # not measured
            ([512] * 8, 64, torch.float32, torch.float32, False),
            ([512] * 8, 64, torch.bfloat16, torch.float32, False),  # weight not cast
            ([512] * 8, 32, torch.bfloat16, torch.bfloat16, False),  # C != 64
            ([512, 0], 64, torch.bfloat16, torch.bfloat16, False),  # empty tile
            ([2**31 // (256 * 64)], 64, torch.bfloat16, torch.bfloat16, False),  # 2**31 elements
            ([2**31 // (256 * 64) - 1], 64, torch.bfloat16, torch.bfloat16, True),
        ],
    )
    def test_call_rule(self, rows, channels, dtype, weight_dtype, ok):
        tiles = _meta_tiles(rows, channels=channels, dtype=dtype)
        weight = torch.empty(channels, device="meta", dtype=weight_dtype)
        assert _call_rule()(_kwargs(tiles, weight, 1, 1024, 1024, 16)).success == ok

    def test_call_rule_layouts(self):
        weight = torch.empty(64, device="meta", dtype=torch.bfloat16)
        strided = torch.empty(512, 64, 256, device="meta", dtype=torch.bfloat16).transpose(1, 2)
        tiles = [*_meta_tiles([512]), strided]
        assert not _call_rule()(_kwargs(tiles, weight, 1, 512, 512, 16)).success
        weight2 = torch.empty(128, device="meta", dtype=torch.bfloat16)[::2]
        assert not _call_rule()(_kwargs(_meta_tiles([512]), weight2, 1, 256, 512, 16)).success

    def test_bf16_only(self):
        triton_backend = importlib.import_module("comfy_kitchen.backends.triton")
        params = triton_backend._build_constraints()[OP].params
        assert params["weight"].dtypes == frozenset({torch.bfloat16})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestNerfFoldRmsNormTriton:
    @pytest.fixture(autouse=True)
    def _triton(self):
        pytest.importorskip("triton")
        if not registry.is_available("triton"):
            pytest.skip("triton backend not available")

    @staticmethod
    @contextlib.contextmanager
    def _served(expected="triton"):
        # the backend of the implementation the registry returns for each call in the block
        # (get_implementation wrapped), so the check is on the call that ran, not on a query
        served = []
        get_implementation = registry.get_implementation

        def recording(func_name, backend=None, kwargs=None):
            impl = get_implementation(func_name, backend, kwargs)
            if func_name == OP:
                served.append(impl.__module__.rsplit(".", 2)[-2])
            return impl

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(registry, "get_implementation", recording)
            yield
        assert served and all(b == expected for b in served), served

    def _check(self, tiles, weight, batch, height, width, patch, eps=None):
        kwargs = _kwargs(tiles, weight, batch, height, width, patch, eps)
        assert registry.validate_backend_for_call("triton", OP, kwargs).success
        with self._served("triton"), ck.use_backend("triton"):
            got = ck.nerf_fold_rms_norm(tiles, weight, batch, height, width, patch, eps)
        ref = eager_backend.nerf_fold_rms_norm(**kwargs)
        ref32 = _reference32(tiles, weight, batch, height, width, patch, eps)
        assert got.shape == ref.shape and got.stride() == ref.stride()
        assert got.dtype == torch.bfloat16
        assert _rel(got, ref) < 1e-2
        assert _rel(got, ref32) < 1e-2
        # one rounding: never further from the fp32 reference than the eager chain
        assert (got.float() - ref32).abs().max() <= (ref.float() - ref32).abs().max()
        return got, ref

    @pytest.mark.parametrize(
        ("batch", "height", "width", "tile"),
        [
            (1, 256, 256, 64),  # 4 full tiles
            (1, 256, 224, 64),  # 224 patches: 3 full tiles + 32
            (2, 128, 192, 40),  # batch 2, partial last tile
            (1, 128, 128, 512),  # one tile (the untiled head)
            (1, 16, 16, 512),  # a single patch
        ],
    )
    def test_matches_eager(self, batch, height, width, tile):
        tiles = _tiles(batch, height, width, 16, 64, tile, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(64, device="cuda").to(torch.bfloat16)
        self._check(tiles, weight, batch, height, width, 16)

    @pytest.mark.parametrize("patch", [2, 8, 32])
    def test_other_patch_sizes(self, patch):
        tiles = _tiles(1, 4 * patch, 6 * patch, patch, 64, 7, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(64, device="cuda").to(torch.bfloat16)
        self._check(tiles, weight, 1, 4 * patch, 6 * patch, patch)

    def test_eps(self):
        tiles = _tiles(1, 128, 128, 16, 64, 16, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(64, device="cuda").to(torch.bfloat16)
        self._check(tiles, weight, 1, 128, 128, 16, eps=1e-3)
        # rows of zeros: rsqrt(eps) * 0 = 0 on both paths
        tiles = [torch.zeros_like(t) for t in tiles]
        got, ref = self._check(tiles, weight, 1, 128, 128, 16)
        assert torch.equal(got, ref)

    def test_conv_on_output(self):
        # the stock path hands this layout to NerfFinalLayerConv.conv (3x3, padding 1)
        tiles = _tiles(1, 128, 128, 16, 64, 16, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(64, device="cuda").to(torch.bfloat16)
        conv = nn.Conv2d(64, 3, 3, padding=1, device="cuda", dtype=torch.bfloat16)
        got, ref = self._check(tiles, weight, 1, 128, 128, 16)
        with torch.no_grad():
            assert _rel(conv(got), conv(ref)) < 1e-2

    def test_real_shape_served(self):
        # the template call at its captured size: 8 tiles [512, 256, 64] -> [1, 64, 1024, 1024]
        tiles = _tiles(1, 1024, 1024, 16, 64, 512, device="cuda", dtype=torch.bfloat16)
        assert [tuple(t.shape) for t in tiles] == [(512, 256, 64)] * 8
        weight = torch.randn(64, device="cuda").to(torch.bfloat16)
        got, _ = self._check(tiles, weight, 1, 1024, 1024, 16)
        assert got.stride() == (67108864, 1, 65536, 64)

    def test_unsupported_inputs_go_to_eager(self):
        weight = torch.randn(64, device="cuda")
        # fp16 / fp32 are not served by the triton kernel
        for dtype in (torch.float16, torch.float32):
            tiles = _tiles(1, 64, 64, 16, 64, 4, device="cuda", dtype=dtype)
            kwargs = _kwargs(tiles, weight.to(dtype), 1, 64, 64, 16)
            assert not registry.validate_backend_for_call("triton", OP, kwargs).success
            assert registry.get_capable_backend(OP, kwargs) == "eager"
            with ck.use_backend("triton"):
                got = ck.nerf_fold_rms_norm(tiles, weight.to(dtype), 1, 64, 64, 16)
            ref = _stock_tail(tiles, _stock_norm(weight.to(dtype)), 1, 64, 64, 16)
            assert torch.equal(got, ref)
        # C != 64, an fp32 weight with bf16 tiles, a non-contiguous tile
        bf16 = torch.bfloat16
        cases = [
            (_tiles(1, 64, 64, 16, 32, 4, device="cuda", dtype=bf16), weight[:32].to(bf16)),
            (_tiles(1, 64, 64, 16, 64, 4, device="cuda", dtype=bf16), weight),
            (
                [torch.randn(16, 64, 256, device="cuda").to(bf16).transpose(1, 2)],
                weight.to(bf16),
            ),
        ]
        for tiles, w in cases:
            kwargs = _kwargs(tiles, w, 1, 64, 64, 16)
            assert not registry.validate_backend_for_call("triton", OP, kwargs).success
            assert registry.get_capable_backend(OP, kwargs) == "eager"
            with ck.use_backend("triton"):
                got = ck.nerf_fold_rms_norm(tiles, w, 1, 64, 64, 16)
            assert torch.equal(got, _stock_tail(tiles, _cast_norm(w), 1, 64, 64, 16))
        # CPU tensors go to eager
        tiles = _tiles(1, 64, 64, 16, 64, 4, dtype=bf16)
        kwargs = _kwargs(tiles, weight.cpu().to(bf16), 1, 64, 64, 16)
        assert not registry.validate_backend_for_call("triton", OP, kwargs).success
        assert registry.get_capable_backend(OP, kwargs) == "eager"
