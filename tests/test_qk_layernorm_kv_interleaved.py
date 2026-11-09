# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""qk_layernorm_kv_interleaved -- Hunyuan3D VAE cross-attention prep: QK LayerNorm, k from the fused kv.

The eager backend is the sequence ComfyUI's Hunyuan3D VAE QKVMultiheadCrossAttention.forward
runs today (the q / kv views, torch.split of kv, q_norm / k_norm), so it must match the stock
code exactly, including the output layout. The Triton kernel computes the norm in fp32 and
rounds once, so q and k are compared to eager and to an fp32 reference with a tolerance and
must never be further from the fp32 reference than eager.
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

OP = "qk_layernorm_kv_interleaved"
_NAMES = ("q", "kv", "q_weight", "q_bias", "k_weight", "k_bias")


def _stock_prep(q, kv, q_norm, k_norm, heads):
    # comfy/ldm/hunyuan3d/vae.py QKVMultiheadCrossAttention.forward (ComfyUI fb2b26f) up to
    # the permutes; q / kv are the c_q / c_kv outputs
    _, n_ctx, _ = q.shape
    bs, n_data, width = kv.shape

    attn_ch = width // heads // 2
    q = q.view(bs, n_ctx, heads, -1)

    kv = kv.view(bs, n_data, heads, -1)
    k, v = torch.split(kv, attn_ch, dim=-1)

    q = q_norm(q)
    k = k_norm(k)

    return q, k, v


def _stock_norm(weight, bias, eps=1e-6):
    # comfy.ops LayerNorm without cast: torch.nn.LayerNorm.forward
    norm = nn.LayerNorm(weight.shape[0], eps=eps, device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        norm.weight.copy_(weight)
        norm.bias.copy_(bias)
    return norm


def _cast_norm(weight, bias, eps=1e-6):
    # comfy.ops LayerNorm with comfy_cast_weights:
    # F.layer_norm(input, normalized_shape, weight, bias, eps)
    return lambda x: functional.layer_norm(x, (weight.shape[0],), weight, bias, eps)


def _inputs(
    batch, n_ctx, n_data, heads, head_dim, *, device="cpu", dtype=torch.float32, fused=False
):
    # q [B, S_q, C] (c_q output), kv [B, S_kv, 2 * C] (c_kv output); fused=True gives q as a
    # chunk of a wider tensor (row stride 2 * C, not contiguous)
    channels = heads * head_dim
    if fused:
        q = (torch.randn(batch, n_ctx, 2 * channels, device=device) * 2).to(dtype)[..., :channels]
    else:
        q = (torch.randn(batch, n_ctx, channels, device=device) * 2).to(dtype)
    kv = (torch.randn(batch, n_data, 2 * channels, device=device) * 2).to(dtype)
    params = []
    for _ in range(2):
        params.append((1 + 0.1 * torch.randn(head_dim, device=device)).to(dtype))
        params.append((0.1 * torch.randn(head_dim, device=device)).to(dtype))
    return [q, kv, *params]


def _kwargs(inputs, heads, eps=1e-6):
    return {**dict(zip(_NAMES, inputs, strict=True)), "heads": heads, "eps": eps}


def _kitchen(inputs, heads, eps=1e-6):
    return ck.qk_layernorm_kv_interleaved(*inputs, heads, eps)


def _stock(inputs, heads, eps=1e-6, make_norm=None):
    make_norm = make_norm or _cast_norm
    q, kv, wq, bq, wk, bk = inputs
    return _stock_prep(q, kv, make_norm(wq, bq, eps), make_norm(wk, bk, eps), heads)


def _reference32(inputs, heads, eps=1e-6):
    return _stock([t.float() for t in inputs], heads, eps)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def _err(a, ref32):
    d = (a.float() - ref32).abs()
    return d.max().item(), d.mean().item()


def _call_rule():
    triton_backend = importlib.import_module("comfy_kitchen.backends.triton")
    (rule,) = triton_backend._build_constraints()[OP].call_rules
    return rule


# (batch, S_q, S_kv, heads, head_dim): the Hunyuan3D-2.1 geometry decoder calls are batch 1,
# 8000 query points per chunk (6593 in the last chunk) against 4096 latent tokens, 16 heads
# of 64; scaled down here in S, plus batch 2 and other head counts and sizes
_EAGER_SHAPES = [
    (1, 125, 64, 16, 64),
    (1, 103, 64, 16, 64),
    (2, 37, 19, 4, 64),
    (1, 1, 1, 1, 64),
    (2, 21, 33, 3, 32),
    (1, 9, 5, 2, 16),
]


class TestQkLayernormKvInterleavedEager:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    @pytest.mark.parametrize(("batch", "n_ctx", "n_data", "heads", "head_dim"), _EAGER_SHAPES)
    def test_matches_stock_chain(self, batch, n_ctx, n_data, heads, head_dim, dtype):
        inputs = _inputs(batch, n_ctx, n_data, heads, head_dim, dtype=dtype)
        with ck.use_backend("eager"):
            out = _kitchen(inputs, heads)
        for make_norm in (_stock_norm, _cast_norm):
            ref = _stock(inputs, heads, make_norm=make_norm)
            for o, r in zip(out, ref[:2], strict=True):
                assert o.dtype == dtype and o.shape == r.shape
                assert o.stride() == r.stride()
                assert torch.equal(o, r)

    def test_output_layout(self):
        inputs = _inputs(2, 9, 5, 4, 32)
        with ck.use_backend("eager"):
            out_q, out_k = _kitchen(inputs, 4)
        # what the permutes receive in the stock path: contiguous [B, S, heads, head_dim]
        assert out_q.shape == (2, 9, 4, 32) and out_q.is_contiguous()
        assert out_k.shape == (2, 5, 4, 32) and out_k.is_contiguous()

    def test_k_is_the_first_half_of_each_head_block(self):
        inputs = _inputs(1, 3, 4, 2, 16)
        kv, wk, bk = inputs[1], inputs[4], inputs[5]
        with ck.use_backend("eager"):
            _, out_k = _kitchen(inputs, 2)
        k = kv.view(1, 4, 2, 32)[..., :16]
        assert torch.equal(out_k, functional.layer_norm(k, (16,), wk, bk, 1e-6))

    def test_strided_q(self):
        inputs = _inputs(2, 13, 7, 4, 32, dtype=torch.float16, fused=True)
        assert inputs[0].stride() == (13 * 256, 256, 1)
        with ck.use_backend("eager"):
            out = _kitchen(inputs, 4)
        for o, r in zip(out, _stock(inputs, 4)[:2], strict=True):
            assert torch.equal(o, r) and o.is_contiguous()

    def test_eps(self):
        inputs = _inputs(1, 9, 5, 4, 32, dtype=torch.float16)
        with ck.use_backend("eager"):
            out = _kitchen(inputs, 4, eps=1e-2)
        ref = _stock(inputs, 4, eps=1e-2, make_norm=_stock_norm)
        for o, r in zip(out, ref[:2], strict=True):
            assert torch.equal(o, r)

    @pytest.mark.parametrize(
        ("q_shape", "kv_shape", "param_shape", "heads"),
        [
            ((1, 9, 64), (1, 5, 128), (16,), 3),  # 64 channels, 3 heads
            ((1, 9, 64), (1, 5, 128), (16,), 0),  # no heads
            ((1, 9, 64), (2, 5, 128), (16,), 4),  # batch differs
            ((1, 9, 64), (1, 5, 64), (16,), 4),  # kv not 2 * channels
            ((9, 64), (5, 128), (16,), 4),  # 2D projections
            ((1, 9, 64), (1, 5, 128), (32,), 4),  # params != head_dim
            ((1, 9, 64), (1, 5, 128), (1, 16), 4),  # 2D params
        ],
    )
    def test_rejects_bad_inputs(self, q_shape, kv_shape, param_shape, heads):
        q, kv = torch.randn(q_shape), torch.randn(kv_shape)
        params = [torch.randn(param_shape) for _ in range(4)]
        with pytest.raises(ValueError):
            _kitchen([q, kv, *params], heads)

    def test_rejects_mixed_dtypes(self):
        inputs = _inputs(1, 9, 5, 4, 16)
        bad = list(inputs)
        bad[1] = bad[1].to(torch.float16)  # kv
        with pytest.raises(ValueError):
            _kitchen(bad, 4)
        # weight and bias in q's dtype, as ComfyUI's LayerNorm casts them
        bad = list(inputs)
        bad[5] = bad[5].to(torch.float16)  # k_bias
        with pytest.raises(ValueError):
            _kitchen(bad, 4)

    def test_fake_shape_and_strides(self):
        inputs = _inputs(2, 9, 5, 4, 16, dtype=torch.float16)
        out_q, out_k = ck.qk_layernorm_kv_interleaved(*[t.to("meta") for t in inputs], 4)
        assert out_q.shape == (2, 9, 4, 16) and out_q.is_contiguous()
        assert out_k.shape == (2, 5, 4, 16) and out_k.is_contiguous()
        assert out_q.dtype == out_k.dtype == torch.float16

    def test_cpu_goes_to_eager(self):
        inputs = _inputs(1, 9, 5, 16, 64, dtype=torch.float16)
        assert registry.get_capable_backend(OP, _kwargs(inputs, 16)) == "eager"


def _meta(batch, n_ctx, n_data, heads=16, head_dim=64, dtype=torch.float16, strided_q=False):
    channels = heads * head_dim
    if strided_q:
        q = torch.empty(batch, n_ctx, 2 * channels, device="meta", dtype=dtype)[..., :channels]
    else:
        q = torch.empty(batch, n_ctx, channels, device="meta", dtype=dtype)
    kv = torch.empty(batch, n_data, 2 * channels, device="meta", dtype=dtype)
    params = [torch.empty(head_dim, device="meta", dtype=dtype) for _ in range(4)]
    return [q, kv, *params]


class TestQkLayernormKvInterleavedCallRule:
    # the Triton call rule only looks at shapes and layouts, so it runs on meta tensors
    @pytest.mark.parametrize(
        ("batch", "n_ctx", "n_data", "heads", "head_dim", "ok"),
        [
            (1, 8000, 4096, 16, 64, True),  # Hunyuan3D-2.1 geometry decoder chunk
            (1, 6593, 4096, 16, 64, True),  # last chunk
            (2, 8000, 4096, 16, 64, True),
            (1, 300, 20, 1, 64, True),
            (1, 300, 20, 12, 64, True),
            (1, 300, 20, 8, 128, False),  # head_dim 128
            (1, 300, 20, 32, 32, False),  # head_dim 32
            (1, 0, 20, 16, 64, False),  # no queries
            (1, 300, 0, 16, 64, False),  # no kv tokens
            (0, 300, 20, 16, 64, False),  # empty batch
            (1, 2**27, 20, 16, 64, False),  # 2**31 q rows
            (1, 2**27 - 1, 20, 16, 64, True),
        ],
    )
    def test_call_rule(self, batch, n_ctx, n_data, heads, head_dim, ok):
        inputs = _meta(batch, n_ctx, n_data, heads, head_dim)
        assert _call_rule()(_kwargs(inputs, heads)).success == ok

    def test_call_rule_layouts(self):
        rule = _call_rule()
        # q as a chunk of a wider tensor: not contiguous, declined
        assert not rule(_kwargs(_meta(1, 300, 20, strided_q=True), 16)).success
        inputs = _meta(1, 300, 20)
        # bias with a stride
        b2 = torch.empty(128, device="meta", dtype=torch.float16)[::2]
        assert not rule(_kwargs([*inputs[:3], b2, *inputs[4:]], 16)).success

    def test_dtypes(self):
        triton_backend = importlib.import_module("comfy_kitchen.backends.triton")
        params = triton_backend._build_constraints()[OP].params
        assert set(params) == set(_NAMES)
        for name in _NAMES:
            assert params[name].dtypes == frozenset({torch.float16})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestQkLayernormKvInterleavedTriton:
    @pytest.fixture(autouse=True)
    def _triton(self):
        pytest.importorskip("triton")
        if not registry.is_available("triton"):
            pytest.skip("triton backend not available")
        torch.manual_seed(0)

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

    def _check(self, inputs, heads, eps=1e-6):
        kwargs = _kwargs(inputs, heads, eps)
        assert registry.validate_backend_for_call("triton", OP, kwargs).success
        with self._served(), ck.use_backend("triton"):
            got = _kitchen(inputs, heads, eps)
        ref = eager_backend.qk_layernorm_kv_interleaved(**kwargs)
        ref32 = _reference32(inputs, heads, eps)
        for g, r, r32 in zip(got, ref, ref32[:2], strict=True):
            assert g.shape == r.shape and g.stride() == r.stride()
            assert g.dtype == torch.float16
            assert not torch.isnan(g).any()
            assert _rel(g, r) < 1e-2
            assert _rel(g, r32) < 1e-2
            # one rounding: never further from the fp32 reference than the eager chain
            g_max, g_mean = _err(g, r32)
            r_max, r_mean = _err(r, r32)
            assert g_max <= r_max and g_mean <= r_mean
        return got, ref

    @pytest.mark.parametrize(
        ("batch", "n_ctx", "n_data"),
        [
            (1, 500, 256),
            (1, 413, 256),  # partial last q program
            (2, 500, 256),
            (1, 1, 1),
            (1, 3, 7),
        ],
    )
    def test_matches_eager(self, batch, n_ctx, n_data):
        inputs = _inputs(batch, n_ctx, n_data, 16, 64, device="cuda", dtype=torch.float16)
        self._check(inputs, 16)

    @pytest.mark.parametrize("heads", [1, 3, 8, 32])
    def test_other_head_counts(self, heads):
        inputs = _inputs(1, 200, 64, heads, 64, device="cuda", dtype=torch.float16)
        self._check(inputs, heads)

    def test_eps(self):
        inputs = _inputs(1, 200, 64, 16, 64, device="cuda", dtype=torch.float16)
        self._check(inputs, 16, eps=1e-3)

    @pytest.mark.parametrize("hip", [False, True])
    def test_both_launch_configs(self, request, monkeypatch, hip):
        # the CUDA and HIP launch configurations, on whichever GPU
        kernel = importlib.import_module("comfy_kitchen.backends.triton." + OP)
        monkeypatch.setattr(kernel, "_HIP", hip)
        # _rcp3 is cached per device and reads _HIP: drop what an earlier test cached, and
        # what this one caches under the patched flag
        kernel._rcp3.cache_clear()
        request.addfinalizer(kernel._rcp3.cache_clear)
        assert kernel._launch_config() == ((16, 4) if hip else (32, 4))
        inputs = _inputs(1, 500, 256, 16, 64, device="cuda", dtype=torch.float16)
        kwargs = _kwargs(inputs, 16)
        got = kernel.qk_layernorm_kv_interleaved(**kwargs)
        ref = eager_backend.qk_layernorm_kv_interleaved(**kwargs)
        ref32 = _reference32(inputs, 16)
        for g, r, r32 in zip(got, ref, ref32[:2], strict=True):
            assert _rel(g, r32) < 1e-2
            assert _err(g, r32)[0] <= _err(r, r32)[0]

    def test_count3_reciprocal(self):
        # the reciprocal the torch build of this GPU uses in the Welford update
        kernel = importlib.import_module("comfy_kitchen.backends.triton." + OP)
        rcp3 = kernel._rcp3(torch.cuda.current_device())
        arch = torch.cuda.get_device_properties(0).gcnArchName if torch.version.hip else ""
        if torch.version.hip and arch.split(":")[0] != "gfx90a":
            assert rcp3 == kernel._RCP3_AMDGCN
        else:
            assert rcp3 == kernel._RCP3_IEEE
        for value in (kernel._RCP3_IEEE, kernel._RCP3_AMDGCN):
            assert torch.tensor(value, dtype=torch.float32).item() == value  # exact in fp32

    @pytest.mark.parametrize("n_ctx", [8000, 6593])
    def test_real_shape_served(self, n_ctx):
        # the Hunyuan3D-2.1 geometry decoder calls at their captured size: q [1, 8000, 1024]
        # (6593 in the last chunk), kv [1, 4096, 2048], 16 heads of 64, fp16
        inputs = _inputs(1, n_ctx, 4096, 16, 64, device="cuda", dtype=torch.float16)
        got, _ = self._check(inputs, 16)
        assert got[0].shape == (1, n_ctx, 16, 64)
        assert got[1].shape == (1, 4096, 16, 64)

    def test_unsupported_inputs_go_to_eager(self):
        def expect_eager(inputs, heads):
            kwargs = _kwargs(inputs, heads)
            assert not registry.validate_backend_for_call("triton", OP, kwargs).success
            assert registry.get_capable_backend(OP, kwargs) == "eager"
            with ck.use_backend("triton"):
                got = _kitchen(inputs, heads)
            for g, r in zip(got, _stock(inputs, heads)[:2], strict=True):
                assert torch.equal(g, r)

        # bf16 / fp32 are not served by the triton kernel
        for dtype in (torch.bfloat16, torch.float32):
            expect_eager(_inputs(1, 9, 5, 4, 64, device="cuda", dtype=dtype), 4)
        fp16 = torch.float16
        # head_dim 128
        expect_eager(_inputs(1, 9, 5, 4, 128, device="cuda", dtype=fp16), 4)
        # strided q
        expect_eager(_inputs(1, 9, 5, 4, 64, device="cuda", dtype=fp16, fused=True), 4)
        # CPU tensors go to eager
        inputs = _inputs(1, 9, 5, 4, 64, dtype=fp16)
        kwargs = _kwargs(inputs, 4)
        assert not registry.validate_backend_for_call("triton", OP, kwargs).success
        assert registry.get_capable_backend(OP, kwargs) == "eager"
