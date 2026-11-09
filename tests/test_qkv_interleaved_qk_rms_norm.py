# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""qkv_interleaved_qk_rms_norm -- Hunyuan3D-2.1 DiT attention prep: q / k / v relayout + QK RMSNorm.

The eager backend is the sequence ComfyUI's Hunyuan3D-2.1 Attention.forward and
CrossAttention.forward run today (torch.cat of the projections, view, torch.split per head,
reshape, q_norm / k_norm), so it must match the stock code exactly, including the output
layout the attention call reads. The Triton kernel computes the norm in fp32 and rounds once,
so q and k are compared to eager and to an fp32 reference with a tolerance and must never be
further from the fp32 reference than eager; v is a copy and must be equal.
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

OP = "qkv_interleaved_qk_rms_norm"
_NAMES = ("q", "k", "v", "q_weight", "k_weight")


def _stock_self(query, key, value, q_norm, k_norm, num_heads):
    # comfy/ldm/hunyuan3dv2_1/hunyuandit.py Attention.forward (ComfyUI fb2b26f) from the
    # to_q / to_k / to_v outputs to the attention call
    B, N, _ = query.shape  # noqa: N806
    head_dim = query.shape[-1] // num_heads

    qkv_combined = torch.cat((query, key, value), dim=-1)
    split_size = qkv_combined.shape[-1] // num_heads // 3

    qkv = qkv_combined.view(B, -1, num_heads, split_size * 3)
    query, key, value = torch.split(qkv, split_size, dim=-1)

    query = query.reshape(B, N, num_heads, head_dim)
    key = key.reshape(B, N, num_heads, head_dim)
    value = value.reshape(B, N, num_heads * head_dim)

    query = q_norm(query)
    key = k_norm(key)

    return query.reshape(B, N, -1), key.reshape(B, N, -1), value


def _stock_cross(q, k, v, q_norm, k_norm, num_heads):
    # comfy/ldm/hunyuan3dv2_1/hunyuandit.py CrossAttention.forward (ComfyUI fb2b26f) from the
    # to_q / to_k / to_v outputs to the attention call
    b, s1, _ = q.shape
    s2 = k.shape[1]
    head_dim = q.shape[-1] // num_heads

    kv = torch.cat((k, v), dim=-1)
    split_size = kv.shape[-1] // num_heads // 2

    kv = kv.view(b, -1, num_heads, split_size * 2)
    k, v = torch.split(kv, split_size, dim=-1)

    q = q.view(b, s1, num_heads, head_dim)
    k = k.view(b, s2, num_heads, head_dim)
    v = v.reshape(b, s2, num_heads * head_dim)

    q = q_norm(q)
    k = k_norm(k)

    return q.reshape(b, s1, -1), k.reshape(b, s2, -1), v


def _stock_norm(weight, eps=1e-6):
    # comfy.ops RMSNorm without cast: torch.nn.RMSNorm.forward
    norm = nn.RMSNorm(weight.shape[0], eps=eps, device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        norm.weight.copy_(weight)
    return norm


def _cast_norm(weight, eps=1e-6):
    # comfy.ops RMSNorm with comfy_cast_weights:
    # F.rms_norm(input, normalized_shape, weight, eps)
    return lambda x: functional.rms_norm(x, (weight.shape[0],), weight, eps)


def _inputs(batch, seq, seq_kv, heads, head_dim, *, device="cpu", dtype=torch.float32, fused=False):
    # projection outputs: q [B, S, C], k / v [B, S_kv, C]; fused=True gives q, k, v as chunks
    # of one [B, S, 3 * C] tensor (views with row stride 3 * C, self-attention only)
    channels = heads * head_dim
    if fused:
        qkv = (torch.randn(batch, seq, 3 * channels, device=device) * 2).to(dtype)
        q, k, v = qkv.chunk(3, dim=-1)
    else:
        q = (torch.randn(batch, seq, channels, device=device) * 2).to(dtype)
        k, v = (
            (torch.randn(batch, seq_kv, channels, device=device) * 2).to(dtype) for _ in range(2)
        )
    weights = [(1 + 0.1 * torch.randn(head_dim, device=device)).to(dtype) for _ in range(2)]
    return [q, k, v, *weights]


def _kwargs(inputs, heads, eps=1e-6, q_interleaved=True):
    return {
        **dict(zip(_NAMES, inputs, strict=True)),
        "heads": heads,
        "eps": eps,
        "q_interleaved": q_interleaved,
    }


def _kitchen(inputs, heads, eps=1e-6, q_interleaved=True):
    return ck.qkv_interleaved_qk_rms_norm(*inputs, heads, eps, q_interleaved)


def _stock(inputs, heads, eps=1e-6, q_interleaved=True, make_norm=None):
    make_norm = make_norm or _cast_norm
    q, k, v, wq, wk = inputs
    stock = _stock_self if q_interleaved else _stock_cross
    return stock(q, k, v, make_norm(wq, eps), make_norm(wk, eps), heads)


def _reference32(inputs, heads, eps=1e-6, q_interleaved=True):
    return _stock([t.float() for t in inputs], heads, eps, q_interleaved)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def _err(a, ref32):
    d = (a.float() - ref32).abs()
    return d.max().item(), d.mean().item()


def _call_rule():
    triton_backend = importlib.import_module("comfy_kitchen.backends.triton")
    (rule,) = triton_backend._build_constraints()[OP].call_rules
    return rule


# (batch, S, S_kv, heads, head_dim, q_interleaved): the Hunyuan3D-2.1 template calls are
# batch 2 (cond and uncond batched), 16 heads of 128, self-attention over 4097 latent tokens
# and cross-attention from them to 1370 image tokens; scaled down here in S, plus batch 1,
# fewer heads and other head sizes
_EAGER_SHAPES = [
    (2, 65, 65, 16, 128, True),
    (1, 33, 33, 4, 128, True),
    (2, 17, 17, 3, 32, True),
    (1, 1, 1, 1, 16, True),
    (2, 65, 23, 16, 128, False),
    (1, 33, 70, 4, 128, False),
    (2, 17, 5, 3, 32, False),
    (1, 1, 1, 1, 16, False),
]


class TestQkvInterleavedQkRmsNormEager:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    @pytest.mark.parametrize(
        ("batch", "seq", "seq_kv", "heads", "head_dim", "q_interleaved"), _EAGER_SHAPES
    )
    def test_matches_stock_chain(self, batch, seq, seq_kv, heads, head_dim, q_interleaved, dtype):
        inputs = _inputs(batch, seq, seq_kv, heads, head_dim, dtype=dtype)
        with ck.use_backend("eager"):
            out = _kitchen(inputs, heads, q_interleaved=q_interleaved)
        for make_norm in (_stock_norm, _cast_norm):
            ref = _stock(inputs, heads, q_interleaved=q_interleaved, make_norm=make_norm)
            for o, r in zip(out, ref, strict=True):
                assert o.dtype == dtype and o.shape == r.shape
                assert o.stride() == r.stride()
                assert torch.equal(o, r)

    @pytest.mark.parametrize("q_interleaved", [True, False])
    def test_output_layout(self, q_interleaved):
        inputs = _inputs(2, 9, 9 if q_interleaved else 5, 4, 32)
        with ck.use_backend("eager"):
            out = _kitchen(inputs, 4, q_interleaved=q_interleaved)
        # what the attention call receives in the stock path: contiguous [B, S, heads * head_dim]
        seqs = (9, 9, 9) if q_interleaved else (9, 5, 5)
        for o, s in zip(out, seqs, strict=True):
            assert o.shape == (2, s, 128)
            assert o.is_contiguous()

    def test_head_blocks_are_interleaved(self):
        # head h of the self-attention output takes block 3h (q), 3h + 1 (k), 3h + 2 (v) of the
        # concatenated row, not the h-th slice of each projection
        inputs = _inputs(1, 3, 3, 2, 16)
        q, k, v = inputs[:3]
        with ck.use_backend("eager"):
            _, _, out_v = _kitchen(inputs, 2)
        cat = torch.cat((q, k, v), dim=-1).view(1, 3, 6, 16)
        assert torch.equal(out_v.view(1, 3, 2, 16)[:, :, 0], cat[:, :, 2])
        assert torch.equal(out_v.view(1, 3, 2, 16)[:, :, 1], cat[:, :, 5])

    def test_fused_projection_views(self):
        inputs = _inputs(2, 13, 13, 4, 32, dtype=torch.float16, fused=True)
        assert inputs[0].stride() == (13 * 384, 384, 1)
        with ck.use_backend("eager"):
            out = _kitchen(inputs, 4)
        ref = _stock(inputs, 4)
        for o, r in zip(out, ref, strict=True):
            assert torch.equal(o, r) and o.is_contiguous()

    @pytest.mark.parametrize("eps", [1e-2, 1.0 / 65504])
    def test_eps(self, eps):
        # 1 / 65504 is the model's eps with use_fp16
        inputs = _inputs(1, 9, 9, 4, 32, dtype=torch.float16)
        for q_interleaved in (True, False):
            with ck.use_backend("eager"):
                out = _kitchen(inputs, 4, eps=eps, q_interleaved=q_interleaved)
            ref = _stock(inputs, 4, eps, q_interleaved, make_norm=_stock_norm)
            for o, r in zip(out, ref, strict=True):
                assert torch.equal(o, r)

    @pytest.mark.parametrize(
        ("q_shape", "kv_shape", "w_shape", "heads", "q_interleaved"),
        [
            ((1, 9, 64), (1, 9, 64), (16,), 3, True),  # 64 channels, 3 heads
            ((1, 9, 64), (1, 9, 64), (16,), 0, True),  # no heads
            ((1, 9, 64), (1, 8, 64), (16,), 4, True),  # self: k / v length differs from q
            ((1, 9, 64), (2, 9, 64), (16,), 4, False),  # batch differs
            ((1, 9, 64), (1, 9, 32), (16,), 4, False),  # channels differ
            ((9, 64), (9, 64), (16,), 4, True),  # 2D projections
            ((1, 9, 64), (1, 9, 64), (32,), 4, True),  # weights != head_dim
            ((1, 9, 64), (1, 9, 64), (1, 16), 4, True),  # 2D weights
        ],
    )
    def test_rejects_bad_inputs(self, q_shape, kv_shape, w_shape, heads, q_interleaved):
        q, k, v = torch.randn(q_shape), torch.randn(kv_shape), torch.randn(kv_shape)
        weights = [torch.randn(w_shape) for _ in range(2)]
        with pytest.raises(ValueError):
            _kitchen([q, k, v, *weights], heads, q_interleaved=q_interleaved)

    def test_rejects_k_v_shapes_differ(self):
        q, k = torch.randn(1, 9, 64), torch.randn(1, 5, 64)
        v = torch.randn(1, 6, 64)
        weights = [torch.randn(16) for _ in range(2)]
        with pytest.raises(ValueError):
            _kitchen([q, k, v, *weights], 4, q_interleaved=False)

    def test_rejects_mixed_dtypes(self):
        inputs = _inputs(1, 9, 9, 4, 16)
        bad = list(inputs)
        bad[2] = bad[2].to(torch.float16)  # v
        with pytest.raises(ValueError):
            _kitchen(bad, 4)
        # weights in q's dtype, as ComfyUI's RMSNorm casts them
        bad = list(inputs)
        bad[4] = bad[4].to(torch.float16)  # k_weight
        with pytest.raises(ValueError):
            _kitchen(bad, 4)

    @pytest.mark.parametrize("q_interleaved", [True, False])
    def test_fake_shape_and_strides(self, q_interleaved):
        inputs = _inputs(2, 9, 9 if q_interleaved else 5, 4, 16, dtype=torch.float16)
        out = ck.qkv_interleaved_qk_rms_norm(
            *[t.to("meta") for t in inputs], 4, q_interleaved=q_interleaved
        )
        seqs = (9, 9, 9) if q_interleaved else (9, 5, 5)
        for o, s in zip(out, seqs, strict=True):
            assert o.shape == (2, s, 64) and o.is_contiguous()
            assert o.dtype == torch.float16

    # one head is the case where the stock v is a strided view of the concatenation, so the
    # fake has to give that layout (strides and storage offset), not a contiguous tensor
    @pytest.mark.parametrize("heads", [1, 4])
    @pytest.mark.parametrize(("batch", "seq", "seq_kv"), [(2, 9, 5), (1, 1, 1), (2, 1, 3)])
    @pytest.mark.parametrize("q_interleaved", [True, False])
    def test_fake_matches_eager_layout(self, heads, batch, seq, seq_kv, q_interleaved):
        inputs = _inputs(batch, seq, seq if q_interleaved else seq_kv, heads, 16)
        with ck.use_backend("eager"):
            real = _kitchen(inputs, heads, q_interleaved=q_interleaved)
        fake = _kitchen([t.to("meta") for t in inputs], heads, q_interleaved=q_interleaved)
        for f, r in zip(fake, real, strict=True):
            assert f.shape == r.shape and f.dtype == r.dtype
            assert f.stride() == r.stride()
            assert f.storage_offset() == r.storage_offset()

    @pytest.mark.parametrize("heads", [1, 4])
    @pytest.mark.parametrize("q_interleaved", [True, False])
    def test_opcheck(self, heads, q_interleaved):
        inputs = _inputs(2, 9, 9 if q_interleaved else 5, heads, 16)
        torch.library.opcheck(
            torch.ops.comfy_kitchen.qkv_interleaved_qk_rms_norm.default,
            (*inputs, heads, 1e-6, q_interleaved),
            test_utils=("test_schema", "test_faketensor"),
        )

    def test_cpu_goes_to_eager(self):
        inputs = _inputs(1, 9, 9, 4, 128, dtype=torch.float16)
        assert registry.get_capable_backend(OP, _kwargs(inputs, 4)) == "eager"


def _meta(batch, seq, seq_kv, heads=16, head_dim=128, dtype=torch.float16, fused=False):
    channels = heads * head_dim
    if fused:
        q, k, v = torch.empty(batch, seq, 3 * channels, device="meta", dtype=dtype).chunk(3, -1)
    else:
        q = torch.empty(batch, seq, channels, device="meta", dtype=dtype)
        k, v = (torch.empty(batch, seq_kv, channels, device="meta", dtype=dtype) for _ in range(2))
    weights = [torch.empty(head_dim, device="meta", dtype=dtype) for _ in range(2)]
    return [q, k, v, *weights]


class TestQkvInterleavedQkRmsNormCallRule:
    # the Triton call rule only looks at shapes and layouts, so it runs on meta tensors
    @pytest.mark.parametrize(
        ("batch", "seq", "seq_kv", "heads", "head_dim", "q_interleaved", "ok"),
        [
            (2, 4097, 4097, 16, 128, True, True),  # Hunyuan3D-2.1 self-attention
            (2, 4097, 1370, 16, 128, False, True),  # Hunyuan3D-2.1 cross-attention
            (1, 4097, 4097, 16, 128, True, True),  # batch 1
            (1, 300, 300, 2, 128, True, True),
            (1, 300, 300, 1, 128, True, False),  # one head (stock v is a strided view)
            (1, 300, 77, 32, 128, False, True),
            (1, 300, 300, 64, 128, True, False),  # more than 32 heads
            (1, 300, 300, 12, 128, True, False),  # heads not a power of 2
            (1, 300, 300, 16, 64, True, False),  # head_dim 64
            (1, 300, 300, 8, 256, True, False),  # head_dim 256
            (1, 0, 0, 16, 128, True, False),  # empty sequence
            (0, 300, 300, 16, 128, True, False),  # empty batch
            (1, 300, 0, 16, 128, False, False),  # no k / v tokens
            (1, 2**31, 77, 2, 128, False, False),  # 2**31 tokens
            (1, 2**31 - 1, 77, 2, 128, False, True),
        ],
    )
    def test_call_rule(self, batch, seq, seq_kv, heads, head_dim, q_interleaved, ok):
        inputs = _meta(batch, seq, seq_kv, heads, head_dim)
        kwargs = _kwargs(inputs, heads, q_interleaved=q_interleaved)
        assert _call_rule()(kwargs).success == ok

    def test_call_rule_layouts(self):
        rule = _call_rule()
        # q, k, v as chunks of one projection (row stride 3 * C): not contiguous, declined
        assert not rule(_kwargs(_meta(1, 300, 300, fused=True), 16)).success
        inputs = _meta(1, 300, 300)
        # weight with a stride
        w2 = torch.empty(256, device="meta", dtype=torch.float16)[::2]
        assert not rule(_kwargs([*inputs[:3], w2, inputs[4]], 16)).success

    def test_dtypes(self):
        triton_backend = importlib.import_module("comfy_kitchen.backends.triton")
        params = triton_backend._build_constraints()[OP].params
        assert set(params) == set(_NAMES)
        for name in _NAMES:
            assert params[name].dtypes == frozenset({torch.float16})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestQkvInterleavedQkRmsNormTriton:
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

    def _check(self, inputs, heads, eps=1e-6, q_interleaved=True):
        kwargs = _kwargs(inputs, heads, eps, q_interleaved)
        assert registry.validate_backend_for_call("triton", OP, kwargs).success
        with self._served(), ck.use_backend("triton"):
            got = _kitchen(inputs, heads, eps, q_interleaved)
        ref = eager_backend.qkv_interleaved_qk_rms_norm(**kwargs)
        ref32 = _reference32(inputs, heads, eps, q_interleaved)
        for i, (g, r, r32) in enumerate(zip(got, ref, ref32, strict=True)):
            assert g.shape == r.shape and g.stride() == r.stride()
            assert g.dtype == torch.float16
            assert not torch.isnan(g).any()
            if i == 2:
                assert torch.equal(g, r)  # v is a copy
                continue
            assert _rel(g, r) < 1e-2
            assert _rel(g, r32) < 1e-2
            # one rounding: never further from the fp32 reference than the eager chain
            g_max, g_mean = _err(g, r32)
            r_max, r_mean = _err(r, r32)
            assert g_max <= r_max and g_mean <= r_mean
        return got, ref

    @pytest.mark.parametrize(
        ("batch", "seq", "seq_kv", "q_interleaved"),
        [
            (2, 257, 257, True),
            (1, 257, 257, True),
            (2, 3, 3, True),  # short sequence
            (2, 257, 86, False),
            (1, 257, 86, False),
            (2, 3, 1, False),  # one k / v token
            (1, 5, 9, False),  # partial last program of the k / v part
        ],
    )
    def test_matches_eager(self, batch, seq, seq_kv, q_interleaved):
        inputs = _inputs(batch, seq, seq_kv, 16, 128, device="cuda", dtype=torch.float16)
        self._check(inputs, 16, q_interleaved=q_interleaved)

    @pytest.mark.parametrize("heads", [2, 8, 32])
    @pytest.mark.parametrize("q_interleaved", [True, False])
    def test_other_head_counts(self, heads, q_interleaved):
        inputs = _inputs(
            2, 65, 65 if q_interleaved else 21, heads, 128, device="cuda", dtype=torch.float16
        )
        self._check(inputs, heads, q_interleaved=q_interleaved)

    @pytest.mark.parametrize("eps", [1e-3, 1.0 / 65504])
    def test_eps(self, eps):
        for q_interleaved in (True, False):
            inputs = _inputs(
                2, 65, 65 if q_interleaved else 21, 16, 128, device="cuda", dtype=torch.float16
            )
            self._check(inputs, 16, eps, q_interleaved)

    @pytest.mark.parametrize("hip", [False, True])
    @pytest.mark.parametrize("q_interleaved", [True, False])
    def test_both_launch_configs(self, monkeypatch, hip, q_interleaved):
        # the CUDA and HIP launch configurations, on whichever GPU
        kernel = importlib.import_module("comfy_kitchen.backends.triton." + OP)
        monkeypatch.setattr(kernel, "_HIP", hip)
        expected = (1, 8) if hip or q_interleaved else (4, 8)
        assert kernel._launch_config(q_interleaved) == expected
        inputs = _inputs(
            2, 257, 257 if q_interleaved else 86, 16, 128, device="cuda", dtype=torch.float16
        )
        kwargs = _kwargs(inputs, 16, q_interleaved=q_interleaved)
        got = kernel.qkv_interleaved_qk_rms_norm(**kwargs)
        ref = eager_backend.qkv_interleaved_qk_rms_norm(**kwargs)
        ref32 = _reference32(inputs, 16, q_interleaved=q_interleaved)
        assert torch.equal(got[2], ref[2])
        for g, r, r32 in zip(got[:2], ref[:2], ref32[:2], strict=True):
            assert _rel(g, r32) < 1e-2
            assert _err(g, r32)[0] <= _err(r, r32)[0]

    @pytest.mark.parametrize(
        ("seq_kv", "q_interleaved"),
        [(4097, True), (1370, False)],
    )
    def test_real_shape_served(self, seq_kv, q_interleaved):
        # the Hunyuan3D-2.1 template calls at their captured size: B 2, 4097 latent tokens,
        # 1370 image tokens for cross-attention, 16 heads of 128, fp16
        inputs = _inputs(2, 4097, seq_kv, 16, 128, device="cuda", dtype=torch.float16)
        got, _ = self._check(inputs, 16, q_interleaved=q_interleaved)
        assert got[0].shape == (2, 4097, 2048)
        assert got[1].shape == got[2].shape == (2, seq_kv, 2048)

    def test_unsupported_inputs_go_to_eager(self):
        def expect_eager(inputs, heads, q_interleaved=True):
            kwargs = _kwargs(inputs, heads, q_interleaved=q_interleaved)
            assert not registry.validate_backend_for_call("triton", OP, kwargs).success
            assert registry.get_capable_backend(OP, kwargs) == "eager"
            with ck.use_backend("triton"):
                got = _kitchen(inputs, heads, q_interleaved=q_interleaved)
            ref = _stock(inputs, heads, q_interleaved=q_interleaved)
            for g, r in zip(got, ref, strict=True):
                assert torch.equal(g, r)

        # bf16 / fp32 are not served by the triton kernel
        for dtype in (torch.bfloat16, torch.float32):
            expect_eager(_inputs(1, 9, 9, 4, 128, device="cuda", dtype=dtype), 4)
        fp16 = torch.float16
        # head_dim 64
        expect_eager(_inputs(1, 9, 5, 4, 64, device="cuda", dtype=fp16), 4, q_interleaved=False)
        # one head (the stock v is a strided view of the concatenation)
        for q_interleaved in (True, False):
            expect_eager(
                _inputs(1, 9, 9 if q_interleaved else 5, 1, 128, device="cuda", dtype=fp16),
                1,
                q_interleaved=q_interleaved,
            )
        # 12 heads
        expect_eager(_inputs(1, 9, 9, 12, 128, device="cuda", dtype=fp16), 12)
        # q, k, v as chunks of one projection (not contiguous)
        expect_eager(_inputs(1, 9, 9, 4, 128, device="cuda", dtype=fp16, fused=True), 4)
        # CPU tensors go to eager
        inputs = _inputs(1, 9, 9, 4, 128, dtype=fp16)
        kwargs = _kwargs(inputs, 4)
        assert not registry.validate_backend_for_call("triton", OP, kwargs).success
        assert registry.get_capable_backend(OP, kwargs) == "eager"
