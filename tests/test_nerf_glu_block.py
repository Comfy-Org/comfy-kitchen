# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nerf_glu_block -- Chroma Radiance NerfGLUBlock after its parameter generator.

The eager backend is the sequence ComfyUI's ``NerfGLUBlock.forward`` runs today (chunk / view
of the generated parameters, three ``F.normalize(dim=-2)``, the RMSNorm, three bmm, SiLU, the
product and the residual add), so it must match the stock code exactly. The Triton kernel
keeps the norms, the gate / value activations and the accumulator in fp32 and rounds the
hidden activation and the output once, so it is compared to eager and to an fp32 reference
with a tolerance, and its error against the fp32 reference must not exceed eager's.
"""

import importlib

import pytest
import torch
from torch import nn

import comfy_kitchen as ck
from comfy_kitchen.backends import eager as eager_backend
from comfy_kitchen.registry import registry

OP = "nerf_glu_block"


class _StockNerfGLUBlock(nn.Module):
    # comfy/ldm/chroma_radiance/layers.py NerfGLUBlock (ComfyUI fb2b26f), with torch.nn layers
    # in place of comfy.ops (the same F.linear / F.rms_norm calls; RMSNorm eps None)
    def __init__(self, hidden_size_s, hidden_size_x, mlp_ratio, dtype=None):
        super().__init__()
        total_params = 3 * hidden_size_x**2 * mlp_ratio
        self.param_generator = nn.Linear(hidden_size_s, total_params, dtype=dtype)
        self.norm = nn.RMSNorm(hidden_size_x, dtype=dtype)
        self.mlp_ratio = mlp_ratio

    def forward(self, x, s):
        mlp_params = self.param_generator(s)
        return _stock_after_generator(self, x, mlp_params)


def _stock_after_generator(block, x, mlp_params):
    # NerfGLUBlock.forward from the chunk on, verbatim
    batch_size, _, hidden_size_x = x.shape
    fc1_gate_params, fc1_value_params, fc2_params = mlp_params.chunk(3, dim=-1)
    fc1_gate = fc1_gate_params.view(batch_size, hidden_size_x, hidden_size_x * block.mlp_ratio)
    fc1_value = fc1_value_params.view(batch_size, hidden_size_x, hidden_size_x * block.mlp_ratio)
    fc2 = fc2_params.view(batch_size, hidden_size_x * block.mlp_ratio, hidden_size_x)
    fc1_gate = torch.nn.functional.normalize(fc1_gate, dim=-2)
    fc1_value = torch.nn.functional.normalize(fc1_value, dim=-2)
    fc2 = torch.nn.functional.normalize(fc2, dim=-2)
    res_x = x
    x = block.norm(x)
    x = torch.bmm(torch.nn.functional.silu(torch.bmm(x, fc1_gate)) * torch.bmm(x, fc1_value), fc2)
    return x + res_x


def _stock_chain(x, mlp_params, norm_weight):
    hidden_size_x = x.shape[-1]
    mlp_ratio = mlp_params.shape[-1] // (3 * hidden_size_x * hidden_size_x)
    block = _StockNerfGLUBlock(1, hidden_size_x, mlp_ratio, dtype=x.dtype).to(x.device)
    with torch.no_grad():
        block.norm.weight.copy_(norm_weight)
    return _stock_after_generator(block, x, mlp_params)


def _reference32(x, mlp_params, norm_weight):
    # the same math in fp32; eps as F.rms_norm resolves eps=None for fp16 / bf16 input
    eps = torch.finfo(torch.float32).eps
    return eager_backend.nerf_glu_block(x.float(), mlp_params.float(), norm_weight.float(), eps)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def _inputs(b, n, h, ratio, *, device="cpu", dtype=torch.float32):
    x = torch.randn(b, n, h, device=device).to(dtype)
    mlp_params = torch.randn(b, 3 * h * h * ratio, device=device).to(dtype)
    norm_weight = (1 + 0.1 * torch.randn(h, device=device)).to(dtype)
    return x, mlp_params, norm_weight


def _kwargs(x, mlp_params, norm_weight, eps=None):
    return {"x": x, "mlp_params": mlp_params, "norm_weight": norm_weight, "eps": eps}


def _call_rule():
    triton_backend = importlib.import_module("comfy_kitchen.backends.triton")
    (rule,) = triton_backend._build_constraints()[OP].call_rules
    return rule


class TestNerfGluBlockEager:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    @pytest.mark.parametrize(
        ("b", "n", "h", "ratio"), [(3, 16, 8, 4), (2, 256, 64, 4), (2, 5, 4, 2)]
    )
    def test_matches_stock_chain(self, b, n, h, ratio, dtype):
        x, mlp_params, norm_weight = _inputs(b, n, h, ratio, dtype=dtype)
        with ck.use_backend("eager"):
            out = ck.nerf_glu_block(x, mlp_params, norm_weight)
        ref = _stock_chain(x, mlp_params, norm_weight)
        assert out.shape == x.shape and out.dtype == dtype
        assert torch.equal(out, ref)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_matches_stock_module(self, dtype):
        # param_generator, then the op: one whole NerfGLUBlock.forward
        torch.manual_seed(0)
        block = _StockNerfGLUBlock(48, 16, 4, dtype=dtype)
        with torch.no_grad():
            block.norm.weight.copy_(1 + 0.1 * torch.randn(16))
        x = torch.randn(6, 32, 16).to(dtype)
        s = torch.randn(6, 48).to(dtype)
        with torch.no_grad(), ck.use_backend("eager"):
            out = ck.nerf_glu_block(x, block.param_generator(s), block.norm.weight)
            ref = block(x, s)
        assert torch.equal(out, ref)

    def test_tile_views(self):
        # forward_nerf feeds tiles of the patch batch: row slices of x and of the generator
        # output, and the generator input is a slice of the hidden states
        x_all, p_all, w = _inputs(10, 16, 8, 4, dtype=torch.bfloat16)
        x, p = x_all[4:7], p_all[4:7]
        with ck.use_backend("eager"):
            out = ck.nerf_glu_block(x, p, w)
        assert torch.equal(out, _stock_chain(x, p, w))

    def test_explicit_eps(self):
        x, mlp_params, norm_weight = _inputs(2, 16, 8, 4)
        with ck.use_backend("eager"):
            out = ck.nerf_glu_block(x, mlp_params, norm_weight, 1e-3)
        block = _StockNerfGLUBlock(1, 8, 4)
        block.norm.eps = 1e-3
        with torch.no_grad():
            block.norm.weight.copy_(norm_weight)
            ref = _stock_after_generator(block, x, mlp_params)
        assert torch.equal(out, ref)
        with ck.use_backend("eager"):
            assert not torch.equal(out, ck.nerf_glu_block(x, mlp_params, norm_weight))

    @pytest.mark.parametrize(
        ("x_shape", "p_shape", "w_shape"),
        [
            ((16, 8), (2, 768), (8,)),  # 2D x
            ((2, 16, 8), (2, 3, 256), (8,)),  # 3D mlp_params
            ((2, 16, 8), (2, 768), (1, 8)),  # 2D norm_weight
            ((2, 16, 8), (3, 768), (8,)),  # rows != patches
            ((2, 16, 8), (2, 770), (8,)),  # not 3 * H * D
            ((2, 16, 8), (2, 0), (8,)),  # no MLP
            ((2, 16, 8), (2, 768), (4,)),  # norm_weight size
            ((2, 16, 0), (2, 768), (0,)),  # H 0
        ],
    )
    def test_rejects_bad_inputs(self, x_shape, p_shape, w_shape):
        with pytest.raises(ValueError):
            ck.nerf_glu_block(torch.randn(x_shape), torch.randn(p_shape), torch.randn(w_shape))

    def test_rejects_dtype_mismatch(self):
        x, mlp_params, norm_weight = _inputs(2, 16, 8, 4)
        with pytest.raises(ValueError):
            ck.nerf_glu_block(x, mlp_params.bfloat16(), norm_weight)
        with pytest.raises(ValueError):
            ck.nerf_glu_block(x, mlp_params, norm_weight.half())

    def test_fake_shape(self):
        x = torch.empty(512, 256, 64, device="meta", dtype=torch.bfloat16)
        p = torch.empty(512, 49152, device="meta", dtype=torch.bfloat16)
        w = torch.empty(64, device="meta", dtype=torch.bfloat16)
        out = ck.nerf_glu_block(x, p, w)
        assert out.shape == (512, 256, 64) and out.dtype == torch.bfloat16

    def test_cpu_goes_to_eager(self):
        x, mlp_params, norm_weight = _inputs(2, 256, 64, 4, dtype=torch.bfloat16)
        assert registry.get_capable_backend(OP, _kwargs(x, mlp_params, norm_weight)) == "eager"


class TestNerfGluBlockCallRule:
    # the Triton call rule only looks at shapes and strides, so it runs on meta tensors
    @pytest.mark.parametrize(
        ("x_shape", "p_cols", "ok"),
        [
            ((512, 256, 64), 49152, True),  # Chroma Radiance, 1024 x 1024
            ((1, 256, 64), 49152, True),
            ((3, 1024, 64), 49152, True),  # patch 32
            ((4, 256, 32), 3 * 32 * 128, True),
            ((4, 256, 16), 3 * 16 * 64, True),
            ((4, 256, 64), 3 * 64 * 96, True),  # D 96
            ((4, 256, 64), 3 * 64 * 80, False),  # D not a multiple of 32
            ((4, 100, 64), 49152, False),  # pixels not a multiple of 256
            ((4, 512 + 256, 64), 49152, True),
            ((4, 256, 128), 3 * 128 * 512, False),  # H 128
            ((4, 256, 48), 3 * 48 * 192, False),  # H 48
            ((0, 256, 64), 49152, False),  # empty
            ((4, 0, 64), 49152, False),  # empty
            ((65536, 256, 64), 49152, False),  # grid
        ],
    )
    def test_call_rule(self, x_shape, p_cols, ok):
        x = torch.empty(x_shape, device="meta", dtype=torch.bfloat16)
        p = torch.empty(x_shape[0], p_cols, device="meta", dtype=torch.bfloat16)
        w = torch.empty(x_shape[2], device="meta", dtype=torch.bfloat16)
        assert _call_rule()(_kwargs(x, p, w)).success == ok

    def test_call_rule_strides(self):
        x = torch.empty(4, 256, 128, device="meta", dtype=torch.bfloat16)
        p = torch.empty(4, 2 * 49152, device="meta", dtype=torch.bfloat16)
        w = torch.empty(128, device="meta", dtype=torch.bfloat16)
        rule = _call_rule()
        # row-strided views are read through their strides
        assert rule(_kwargs(x[..., :64], p[:, :49152], w[:64])).success
        assert rule(_kwargs(x[::2, :, :64], p[::2, :49152], w[:64])).success
        # a non-unit last stride is not
        assert not rule(_kwargs(x[..., ::2], p[:, :49152], w[:64])).success
        assert not rule(_kwargs(x[..., :64], p[:, ::2], w[:64])).success
        assert not rule(_kwargs(x[..., :64], p[:, :49152], w[::2])).success

    def test_call_rule_int32_offsets(self):
        # offsets within a patch of x / out and within an mlp_params row are int32 in the
        # kernel; meta tensors give the sizes and strides without allocating them
        rule = _call_rule()
        w = torch.empty(64, device="meta", dtype=torch.bfloat16)
        p = torch.empty(1, 49152, device="meta", dtype=torch.bfloat16)

        def x_strided(n, row_stride):
            return torch.empty_strided(
                (1, n, 64), (n * row_stride, row_stride, 1), device="meta", dtype=torch.bfloat16
            )

        # contiguous x (and out): N * H elements per patch
        assert rule(_kwargs(x_strided(2**25, 64), p, w)).success
        assert not rule(_kwargs(x_strided(2**25 + 256, 64), p, w)).success
        # wide pixel stride: the last element of the patch is at 255 * stride + 63
        last_ok = (2**31 - 64) // 255
        assert rule(_kwargs(x_strided(256, last_ok), p, w)).success
        assert not rule(_kwargs(x_strided(256, last_ok + 1), p, w)).success
        # mlp_params row length (a multiple of 3 * H * 32)
        x = torch.empty(1, 256, 64, device="meta", dtype=torch.bfloat16)
        cols_ok = 2**31 // 6144 * 6144
        p_ok = torch.empty(1, cols_ok, device="meta", dtype=torch.bfloat16)
        p_big = torch.empty(1, cols_ok + 6144, device="meta", dtype=torch.bfloat16)
        assert rule(_kwargs(x, p_ok, w)).success
        assert not rule(_kwargs(x, p_big, w)).success

    def test_bf16_only(self):
        triton_backend = importlib.import_module("comfy_kitchen.backends.triton")
        constraints = triton_backend._build_constraints()[OP]
        for name in ("x", "mlp_params", "norm_weight"):
            assert constraints.params[name].dtypes == frozenset({torch.bfloat16})
        assert constraints.default_devices == frozenset({"cuda"})


# Shape classes of the Chroma Radiance NeRF head call: N, H and D as captured (16 x 16 patches,
# nerf_hidden_size 64, mlp_ratio 4), patches per tile scaled down from 512; plus the other
# sizes the call rule takes. (B, N, H, mlp_ratio)
_SHAPE_CLASSES = [
    (8, 256, 64, 4),  # Chroma Radiance, one tile scaled down
    (3, 512, 64, 4),
    (5, 256, 32, 4),
    (4, 256, 16, 4),
    (2, 256, 64, 2),
    (1, 256, 64, 4),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestNerfGluBlockTriton:
    @pytest.fixture(autouse=True)
    def _triton(self):
        pytest.importorskip("triton")
        if not registry.is_available("triton"):
            pytest.skip("triton backend not available")
        torch.manual_seed(0)

    @staticmethod
    def _check_close(got, x, mlp_params, norm_weight):
        ref = eager_backend.nerf_glu_block(x, mlp_params, norm_weight)
        ref32 = _reference32(x, mlp_params, norm_weight)
        assert got.shape == ref.shape and got.dtype == ref.dtype and got.is_contiguous()
        assert not got.isnan().any()
        assert _rel(got, ref) < 1e-2
        assert _rel(got, ref32) < 1e-2
        # fp32 inside, one rounding: never further from the fp32 reference than the eager chain
        err, err_eager = (got.float() - ref32).abs(), (ref.float() - ref32).abs()
        assert err.max() <= err_eager.max()
        assert err.mean() <= err_eager.mean()

    @pytest.mark.parametrize(("b", "n", "h", "ratio"), _SHAPE_CLASSES)
    def test_matches_eager(self, b, n, h, ratio):
        x, mlp_params, norm_weight = _inputs(b, n, h, ratio, device="cuda", dtype=torch.bfloat16)
        kwargs = _kwargs(x, mlp_params, norm_weight)
        assert registry.validate_backend_for_call("triton", OP, kwargs).success
        assert registry.get_capable_backend(OP, kwargs) == "triton"
        with ck.use_backend("triton"):
            got = ck.nerf_glu_block(x, mlp_params, norm_weight)
        self._check_close(got, x, mlp_params, norm_weight)

    def test_matches_stock_module(self):
        # param_generator (bf16 Linear), then the op: one whole NerfGLUBlock.forward
        torch.manual_seed(0)
        block = _StockNerfGLUBlock(3072, 64, 4, dtype=torch.bfloat16).cuda()
        with torch.no_grad():
            block.norm.weight.copy_(1 + 0.1 * torch.randn(64))
        s_all = torch.randn(12, 3072, device="cuda").to(torch.bfloat16)
        s = s_all[4:8]  # a tile of the hidden states (a view with a storage offset)
        x = torch.randn(4, 256, 64, device="cuda").to(torch.bfloat16)
        with torch.no_grad():
            params = block.param_generator(s)
            assert registry.get_capable_backend(OP, _kwargs(x, params, block.norm.weight)) == (
                "triton"
            )
            with ck.use_backend("triton"):
                got = ck.nerf_glu_block(x, params, block.norm.weight)
            ref = block(x, s)
            assert torch.equal(eager_backend.nerf_glu_block(x, params, block.norm.weight), ref)
            self._check_close(got, x, params, block.norm.weight)

    @pytest.mark.parametrize(
        "layout",
        [
            "batch_step2",  # x[::2], mlp_params[::2]
            "tile_rows",  # rows [3, 7) of a longer patch batch (storage offsets)
            "wide_rows",  # x and mlp_params are the leading columns of wider rows
        ],
    )
    def test_kernel_layouts(self, layout):
        x, p, w = _inputs(8, 256, 64, 4, device="cuda", dtype=torch.bfloat16)
        if layout == "batch_step2":
            x, p = x[::2], p[::2]
        elif layout == "tile_rows":
            x, p = x[3:7], p[3:7]
        else:
            xw = torch.randn(8, 256, 96, device="cuda").to(torch.bfloat16)
            pw = torch.randn(8, p.shape[1] + 64, device="cuda").to(torch.bfloat16)
            xw[..., :64].copy_(x)
            pw[:, : p.shape[1]].copy_(p)
            x, p = xw[..., :64], pw[:, : p.shape[1]]
        assert registry.get_capable_backend(OP, _kwargs(x, p, w)) == "triton"
        with ck.use_backend("triton"):
            got = ck.nerf_glu_block(x, p, w)
            want = ck.nerf_glu_block(x.contiguous(), p.contiguous(), w)
        assert got.is_contiguous()
        assert torch.equal(got, want)
        self._check_close(got, x, p, w)

    def test_eps(self):
        x, p, w = _inputs(2, 256, 64, 4, device="cuda", dtype=torch.bfloat16)
        # ComfyUI passes eps=None: F.rms_norm on bf16 then uses finfo(float32).eps, the value
        # the kernel resolves None to
        eps32 = torch.finfo(torch.float32).eps
        ref_none = eager_backend.nerf_glu_block(x, p, w)
        assert torch.equal(ref_none, eager_backend.nerf_glu_block(x, p, w, eps32))
        with ck.use_backend("triton"):
            got_none = ck.nerf_glu_block(x, p, w)
            got_eps32 = ck.nerf_glu_block(x, p, w, eps32)
            got_big = ck.nerf_glu_block(x * 1e-2, p, w, 1e-2)
        assert torch.equal(got_none, got_eps32)
        ref_big = eager_backend.nerf_glu_block(x * 1e-2, p, w, 1e-2)
        assert _rel(got_big, ref_big) < 1e-2

    def test_zero_columns(self):
        # an all-zero generated column normalizes to zero (norm clamped to 1e-12), not NaN
        x, p, w = _inputs(2, 256, 64, 4, device="cuda", dtype=torch.bfloat16)
        p.view(2, 3, 64, 256)[:, 0, :, :5] = 0  # five gate columns
        with ck.use_backend("triton"):
            got = ck.nerf_glu_block(x, p, w)
        self._check_close(got, x, p, w)

    def test_real_shape_served(self):
        # Chroma Radiance at 1024 x 1024: one tile of 512 patches, 16 x 16 pixels, H 64, D 256
        x, p, w = _inputs(512, 256, 64, 4, device="cuda", dtype=torch.bfloat16)
        assert p.shape == (512, 49152)
        assert registry.get_capable_backend(OP, _kwargs(x, p, w)) == "triton"
        with ck.use_backend("triton"):
            got = ck.nerf_glu_block(x, p, w)
        self._check_close(got, x, p, w)

    def test_empty_input_goes_to_eager(self):
        x, p, w = _inputs(0, 256, 64, 4, device="cuda", dtype=torch.bfloat16)
        assert not registry.validate_backend_for_call("triton", OP, _kwargs(x, p, w)).success
        with ck.use_backend("triton"):
            got = ck.nerf_glu_block(x, p, w)
        assert got.shape == (0, 256, 64) and got.dtype == torch.bfloat16

    def test_int32_offset_bound_goes_to_eager(self):
        # an expanded x has a patch of 2**25 + 256 pixels without the memory; its output
        # patch would pass int32 offsets, so dispatch picks eager (the op is not run here)
        _, p, w = _inputs(1, 256, 64, 4, device="cuda", dtype=torch.bfloat16)
        x = torch.empty(1, 1, 64, device="cuda", dtype=torch.bfloat16).expand(1, 2**25 + 256, 64)
        kwargs = _kwargs(x, p, w)
        assert not registry.validate_backend_for_call("triton", OP, kwargs).success
        assert registry.get_capable_backend(OP, kwargs) == "eager"
        x = torch.empty(1, 1, 64, device="cuda", dtype=torch.bfloat16).expand(1, 2**25, 64)
        assert registry.validate_backend_for_call("triton", OP, _kwargs(x, p, w)).success

    @pytest.mark.parametrize(
        ("dtype", "b", "n", "h", "ratio", "view"),
        [
            (torch.float16, 2, 256, 64, 4, None),  # fp16 not measured
            (torch.float32, 2, 256, 64, 4, None),
            (torch.bfloat16, 2, 100, 64, 4, None),  # pixels not a multiple of 256
            (torch.bfloat16, 2, 256, 48, 4, None),  # H 48
            (torch.bfloat16, 2, 256, 64, 4, "x_step2"),  # non-unit last stride
        ],
    )
    def test_unsupported_inputs_go_to_eager(self, dtype, b, n, h, ratio, view):
        x, p, w = _inputs(b, n, h, ratio, device="cuda", dtype=dtype)
        if view == "x_step2":
            x = torch.randn(b, n, 2 * h, device="cuda").to(dtype)[..., ::2]
        kwargs = _kwargs(x, p, w)
        assert not registry.validate_backend_for_call("triton", OP, kwargs).success
        assert registry.get_capable_backend(OP, kwargs) == "eager"
        with ck.use_backend("triton"):
            got = ck.nerf_glu_block(x, p, w)
        assert torch.equal(got, _stock_chain(x, p, w))
        # CPU tensors go to eager
        kwargs = _kwargs(*_inputs(2, 256, 64, 4, dtype=torch.bfloat16))
        assert not registry.validate_backend_for_call("triton", OP, kwargs).success
        assert registry.get_capable_backend(OP, kwargs) == "eager"
