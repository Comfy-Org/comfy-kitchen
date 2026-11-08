# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""sum_indexed_values -- per-group channel sums of the VOID warped-noise regaussianize.

The eager backend is ComfyUI's sum_indexed_values as it is (max().item(), torch.zeros,
index_add_ over the (h w) c view), so on CPU it must match the stock code exactly. The Triton
kernel sums runs of equal consecutive indices with a scan and adds each run with an atomic, so
on the GPU it is compared to eager and to an fp64 reference with a tolerance, exactly where the
order of the additions cannot matter, and at the real shape it must not be further from the
fp64 reference than eager.
"""

import importlib

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends import eager as eager_backend
from comfy_kitchen.registry import registry

OP = "sum_indexed_values"


# comfy_extras/void_noise_warp.py (ComfyUI fb2b26f), with rearrange(image, "c h w -> (h w) c")
# written as permute / reshape (einops is not a dependency here; test_rearrange_is_the_same
# checks the two agree)
def _stock_unique_pixels(image):
    _, h, w = image.shape
    flat = image.permute(1, 2, 0).reshape(h * w, -1)
    unique_colors, inverse_indices, counts = torch.unique(
        flat,
        dim=0,
        return_inverse=True,
        return_counts=True,
        sorted=False,
    )
    index_matrix = inverse_indices.reshape(h, w)
    return unique_colors, counts, index_matrix


def _stock_sum_indexed_values(image, index_matrix):
    _, h, w = image.shape
    u = int(index_matrix.max().item()) + 1
    flat = image.permute(1, 2, 0).reshape(h * w, -1)
    out = torch.zeros((u, flat.shape[1]), dtype=flat.dtype, device=flat.device)
    out.index_add_(0, index_matrix.view(-1), flat)
    return out


def _stock_indexed_to_image(index_matrix, unique_colors):
    h, w = index_matrix.shape
    flat = unique_colors[index_matrix.view(-1)]
    return flat.reshape(h, w, -1).permute(2, 0, 1)


def _stock_regaussianize(noise, sum_indexed_values):
    _, counts, index_matrix = _stock_unique_pixels(noise[:1])

    foreign_noise = torch.randn_like(noise)
    summed = sum_indexed_values(foreign_noise, index_matrix)
    meaned = _stock_indexed_to_image(index_matrix, summed / counts.reshape(-1, 1))
    zeroed_foreign = foreign_noise - meaned

    counts_image = _stock_indexed_to_image(index_matrix, counts.reshape(-1, 1))

    output = noise / counts_image**0.5 + zeroed_foreign
    return output, counts_image


def _warped_noise(height, width, hot, *, seed=0, dup=0.03):
    # one channel of warped noise as regaussianize sees it: a "hot" group of zeros (the zero
    # pixels warp_state starts from: about half of the frame in the VOID template, in long
    # runs of a row, or scattered), randn elsewhere, with a few pixels copied onto others as
    # a nearest-neighbour warp does
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(1, height, width, generator=g)
    if hot == "runs":
        noise[:, :, : width // 2] = 0.0
    elif hot == "scattered":
        noise[0][torch.rand(height, width, generator=g) < 0.5] = 0.0
    elif hot == "all":
        noise.zero_()
    flat = noise.view(-1)
    k = int(dup * flat.numel())
    src = torch.randint(0, flat.numel(), (k,), generator=g)
    dst = torch.randint(0, flat.numel(), (k,), generator=g)
    flat[dst] = flat[src]
    return noise


def _index(height, width, hot, *, device="cpu", seed=0):
    _, _, index_matrix = _stock_unique_pixels(_warped_noise(height, width, hot, seed=seed))
    return index_matrix.to(device)


def _image(channels, height, width, *, device="cpu", dtype=torch.float32, seed=1):
    # foreign_noise = torch.randn_like(noise)
    g = torch.Generator().manual_seed(seed)
    return torch.randn(channels, height, width, generator=g).to(device=device, dtype=dtype)


def _reference64(image, index_matrix):
    return _stock_sum_indexed_values(image.double(), index_matrix)


def _errors(out, ref64):
    d = (out.double() - ref64).abs()
    return d.max().item(), d.mean().item()


def _within(out, ref64, counts):
    # summation-order error bound per group: 1e-6 per pixel of the group (about 8 fp32 ulps of
    # one per pixel; the measured error at the real shape was well under a tenth of it)
    tol = 1e-6 * counts.double()[:, None]
    return bool(((out.double() - ref64).abs() <= tol).all())


def _kwargs(image, index_matrix):
    return {"image": image, "index_matrix": index_matrix}


def _call_rule():
    triton_backend = importlib.import_module("comfy_kitchen.backends.triton")
    (rule,) = triton_backend._build_constraints()[OP].call_rules
    return rule


# (channels, height, width, hot): the template call is C 16, 1536 x 5376 (the 384 x 672 frame
# upscaled for the 32x downscale), about half of the pixels in one group; scaled down here
_EAGER_SHAPES = [
    (16, 48, 168, "runs"),  # 1/32 of the real H and W
    (16, 48, 168, "scattered"),
    (16, 37, 53, None),  # no hot group
    (4, 16, 24, "all"),  # one group
    (3, 1, 1, None),  # one pixel
]


class TestSumIndexedValuesEager:
    @pytest.mark.parametrize("index_dtype", [torch.int64, torch.int32])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    @pytest.mark.parametrize(("channels", "height", "width", "hot"), _EAGER_SHAPES)
    def test_matches_stock(self, channels, height, width, hot, dtype, index_dtype):
        image = _image(channels, height, width, dtype=dtype)
        index_matrix = _index(height, width, hot).to(index_dtype)
        with ck.use_backend("eager"):
            out = ck.sum_indexed_values(image, index_matrix)
        ref = _stock_sum_indexed_values(image, index_matrix)
        assert out.dtype == dtype and out.shape == ref.shape
        assert torch.equal(out, ref)

    def test_rearrange_is_the_same(self):
        einops = pytest.importorskip("einops")
        image = _image(16, 48, 168)
        flat = image.permute(1, 2, 0).reshape(48 * 168, -1)
        assert torch.equal(flat, einops.rearrange(image, "c h w -> (h w) c"))

    def test_in_regaussianize(self):
        noise = _warped_noise(48, 168, "runs").expand(16, -1, -1).contiguous()
        torch.manual_seed(3)
        ref = _stock_regaussianize(noise, _stock_sum_indexed_values)
        torch.manual_seed(3)
        with ck.use_backend("eager"):
            got = _stock_regaussianize(noise, ck.sum_indexed_values)
        assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])

    def test_unused_index_gives_zero_row(self):
        image = _image(16, 4, 6)
        index_matrix = torch.zeros(4, 6, dtype=torch.int64)
        index_matrix[2:] = 5
        with ck.use_backend("eager"):
            out = ck.sum_indexed_values(image, index_matrix)
        assert out.shape == (6, 16)
        assert torch.equal(out[1:5], torch.zeros(4, 16))
        assert torch.equal(out, _stock_sum_indexed_values(image, index_matrix))

    @pytest.mark.parametrize(
        ("image_shape", "index_shape", "index_dtype"),
        [
            ((16, 8), (16, 8), torch.int64),  # 2D image
            ((16, 8, 12), (8, 12, 1), torch.int64),  # 3D index
            ((16, 8, 12), (12, 8), torch.int64),  # index not [H, W]
            ((16, 8, 12), (8, 12), torch.float32),  # float index
            ((16, 8, 12), (8, 12), torch.int16),
            ((16, 0, 12), (0, 12), torch.int64),  # no pixels
        ],
    )
    def test_rejects_bad_inputs(self, image_shape, index_shape, index_dtype):
        image = torch.randn(image_shape)
        index_matrix = torch.zeros(index_shape, dtype=index_dtype)
        with pytest.raises(ValueError):
            ck.sum_indexed_values(image, index_matrix)

    def test_fake_shape(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        with FakeTensorMode(shape_env=ShapeEnv()):
            image = torch.empty(16, 48, 168)
            index_matrix = torch.empty(48, 168, dtype=torch.int64)
            out = ck.sum_indexed_values(image, index_matrix)
        # U = index_matrix.max() + 1 is data-dependent
        assert isinstance(out.shape[0], torch.SymInt)
        assert out.shape[1] == 16 and out.dtype == torch.float32

    def test_compile_fullgraph(self):
        image = _image(16, 48, 168)
        index_matrix = _index(48, 168, "runs")

        def fn(image, index_matrix):
            return ck.sum_indexed_values(image, index_matrix) * 2

        compiled = torch.compile(fn, backend="aot_eager", fullgraph=True)
        with ck.use_backend("eager"):
            assert torch.equal(compiled(image, index_matrix), fn(image, index_matrix))

    def test_cpu_goes_to_eager(self):
        kwargs = _kwargs(_image(16, 48, 168), _index(48, 168, "runs"))
        assert registry.get_capable_backend(OP, kwargs) == "eager"


def _meta(channels, height, width, dtype=torch.float32, index_dtype=torch.int64):
    image = torch.empty(channels, height, width, device="meta", dtype=dtype)
    index_matrix = torch.empty(height, width, device="meta", dtype=index_dtype)
    return image, index_matrix


class TestSumIndexedValuesCallRule:
    # the Triton call rule only looks at shapes, dtypes and layouts, so it runs on meta tensors
    @pytest.mark.parametrize(
        ("channels", "height", "width", "dtype", "index_dtype", "ok"),
        [
            (16, 1536, 5376, torch.float32, torch.int64, True),  # the template call
            (16, 37, 53, torch.float32, torch.int64, True),  # partial last tile
            (1, 1, 1, torch.float32, torch.int64, True),
            (3, 48, 168, torch.float32, torch.int64, True),
            (17, 48, 168, torch.float32, torch.int64, False),  # > 16 channels
            (16, 1536, 5376, torch.float16, torch.int64, False),  # not measured
            (16, 1536, 5376, torch.bfloat16, torch.int64, False),
            (16, 1536, 5376, torch.float32, torch.int32, False),
            (1, 1, 2**31 - 1024, torch.float32, torch.int64, True),  # int32 pixel offsets
            (1, 1, 2**31 - 1023, torch.float32, torch.int64, False),
        ],
    )
    def test_call_rule(self, channels, height, width, dtype, index_dtype, ok):
        image, index_matrix = _meta(channels, height, width, dtype, index_dtype)
        assert _call_rule()(_kwargs(image, index_matrix)).success == ok

    def test_call_rule_layouts(self):
        image, index_matrix = _meta(16, 48, 168)
        assert _call_rule()(_kwargs(image, index_matrix)).success
        # [H, W, C] memory read as [C, H, W]
        hwc = torch.empty(48, 168, 16, device="meta").permute(2, 0, 1)
        assert not _call_rule()(_kwargs(hwc, index_matrix)).success
        index_t = torch.empty(168, 48, device="meta", dtype=torch.int64).t()
        assert not _call_rule()(_kwargs(image, index_t)).success
        other = torch.empty(48, 160, device="meta", dtype=torch.int64)
        assert not _call_rule()(_kwargs(image, other)).success

    def test_call_rule_deterministic(self):
        image, index_matrix = _meta(16, 1536, 5376)
        was = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            assert not _call_rule()(_kwargs(image, index_matrix)).success
        finally:
            torch.use_deterministic_algorithms(was)
        assert _call_rule()(_kwargs(image, index_matrix)).success

    def test_fp32_int64_only(self):
        triton_backend = importlib.import_module("comfy_kitchen.backends.triton")
        params = triton_backend._build_constraints()[OP].params
        assert params["image"].dtypes == frozenset({torch.float32})
        assert params["index_matrix"].dtypes == frozenset({torch.int64})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestSumIndexedValuesTriton:
    @pytest.fixture(autouse=True)
    def _triton(self):
        pytest.importorskip("triton")
        if not registry.is_available("triton"):
            pytest.skip("triton backend not available")

    @staticmethod
    def _check(image, index_matrix):
        kwargs = _kwargs(image, index_matrix)
        assert registry.validate_backend_for_call("triton", OP, kwargs).success
        assert registry.get_capable_backend(OP, kwargs) == "triton"
        with ck.use_backend("triton"):
            got = ck.sum_indexed_values(image, index_matrix)
        ref = eager_backend.sum_indexed_values(**kwargs)
        ref64 = _reference64(image, index_matrix)
        assert got.shape == ref.shape and got.dtype == torch.float32
        assert not torch.isnan(got).any()
        counts = torch.bincount(index_matrix.view(-1), minlength=got.shape[0])
        assert _within(got, ref64, counts)
        assert _within(ref, ref64, counts)
        # a group of one pixel is one add onto zero on both paths
        assert torch.equal(got[counts == 1], ref[counts == 1])
        return got, ref, ref64

    @pytest.mark.parametrize(
        ("channels", "height", "width", "hot"),
        [
            (16, 192, 672, "runs"),  # 1/8 of the real H and W
            (16, 192, 672, "scattered"),
            (16, 37, 53, "runs"),  # partial last tile
            (16, 48, 168, None),  # no hot group
            (16, 48, 168, "all"),  # one group
            (4, 48, 168, "runs"),  # fewer channels
            (3, 48, 168, "runs"),
            (1, 48, 168, "runs"),
            (16, 1, 1, None),  # one pixel
        ],
    )
    def test_matches_eager(self, channels, height, width, hot):
        image = _image(channels, height, width, device="cuda")
        self._check(image, _index(height, width, hot, device="cuda"))

    @pytest.mark.parametrize("hot", ["runs", "scattered", "all"])
    def test_exact_with_integer_values(self, hot):
        # small integers add exactly in fp32 in any order, so both paths give the same bits
        g = torch.Generator().manual_seed(2)
        image = torch.randint(-4, 5, (16, 192, 672), generator=g).float().cuda()
        got, ref, _ = self._check(image, _index(192, 672, hot, device="cuda"))
        assert torch.equal(got, ref)

    def test_unused_index_gives_zero_row(self):
        image = _image(16, 48, 168, device="cuda")
        index_matrix = _index(48, 168, "runs", device="cuda") * 2
        got, _, _ = self._check(image, index_matrix)
        assert torch.equal(got[1::2], torch.zeros_like(got[1::2]))

    def test_real_shape_served(self):
        # the template call at its captured size: image [16, 1536, 5376] fp32, index [1536, 5376]
        # int64 with about half of the pixels in one group
        image = _image(16, 1536, 5376, device="cuda")
        index_matrix = _index(1536, 5376, "runs", device="cuda")
        counts = torch.bincount(index_matrix.view(-1))
        assert counts.max().item() > 1536 * 5376 // 2
        got, ref, ref64 = self._check(image, index_matrix)
        # never further from the fp64 reference than the current path
        got_max, got_mean = _errors(got, ref64)
        ref_max, ref_mean = _errors(ref, ref64)
        assert got_max <= ref_max and got_mean <= ref_mean

    def test_unsupported_inputs_go_to_eager(self):
        # values in {-1, 0, 1} on 16 x 16 pixels: every partial sum is an integer of at most
        # 256, exact in fp32, fp16 and bf16, so the order of the atomics cannot change the bits
        g = torch.Generator().manual_seed(4)
        index_matrix = _index(16, 16, "runs", device="cuda")
        image = torch.randint(-1, 2, (16, 16, 16), generator=g).float().cuda()
        wide = torch.randint(-1, 2, (17, 16, 16), generator=g).float().cuda()
        cases = [
            (image.half(), index_matrix),  # fp16 / bf16 are not served by the triton kernel
            (image.bfloat16(), index_matrix),
            (image, index_matrix.int()),  # int32 index
            (wide, index_matrix),  # > 16 channels
            (image.permute(1, 2, 0).contiguous().permute(2, 0, 1), index_matrix),  # strided
            (image, index_matrix.t().contiguous().t()),
        ]
        for img, idx in cases:
            kwargs = _kwargs(img, idx)
            assert not registry.validate_backend_for_call("triton", OP, kwargs).success
            assert registry.get_capable_backend(OP, kwargs) == "eager"
            with ck.use_backend("triton"):
                got = ck.sum_indexed_values(img, idx)
            assert torch.equal(got, _stock_sum_indexed_values(img, idx))
        # deterministic algorithms: eager, whose index_add_ is deterministic on CUDA then
        image = _image(16, 48, 168, device="cuda")
        index_matrix = _index(48, 168, "runs", device="cuda")
        kwargs = _kwargs(image, index_matrix)
        was = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            assert not registry.validate_backend_for_call("triton", OP, kwargs).success
            assert registry.get_capable_backend(OP, kwargs) == "eager"
            with ck.use_backend("triton"):
                got = ck.sum_indexed_values(image, index_matrix)
            assert torch.equal(got, _stock_sum_indexed_values(image, index_matrix))
        finally:
            torch.use_deterministic_algorithms(was)
        # CPU tensors go to eager
        kwargs = _kwargs(image.cpu(), index_matrix.cpu())
        assert not registry.validate_backend_for_call("triton", OP, kwargs).success
        assert registry.get_capable_backend(OP, kwargs) == "eager"
