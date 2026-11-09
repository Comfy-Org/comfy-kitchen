# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""qk_layernorm_rope_partial -- CogVideoX attention prep: QK LayerNorm + RoPE on the video tokens.

The eager backend is the sequence ComfyUI's CogVideoXBlock.forward runs today (the view of the
q / k projections, norm_q / norm_k, apply_rotary_emb on the video-token slice, torch.cat with the
text tokens and the reshape back to [B, S, C]), so it must match the stock code exactly,
including the output layout the attention call reads. The Triton kernel computes the norm and the
rotation in fp32 and rounds once, so q and k are compared to eager and to an fp32 reference with
a tolerance and must never be further from the fp32 reference than eager.
"""

import importlib

import pytest
import torch
from torch import nn
from torch.nn import functional

import comfy_kitchen as ck
from comfy_kitchen.backends import eager as eager_backend
from comfy_kitchen.registry import registry

OP = "qk_layernorm_rope_partial"
_NAMES = ("q", "k", "q_weight", "q_bias", "k_weight", "k_bias", "cos", "sin")


def _apply_rotary_emb(x, freqs_cos_sin):
    # comfy/ldm/cogvideo/model.py apply_rotary_emb (ComfyUI fb2b26f)
    cos, sin = freqs_cos_sin
    cos = cos[None, None, :, :].to(x.device)
    sin = sin[None, None, :, :].to(x.device)

    # Interleaved pairs: [B, H, S, D] -> [B, H, S, D//2, 2] -> (real, imag)
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)

    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


def _stock_prep(q_lin, k_lin, norm_q, norm_k, image_rotary_emb, text_seq_length, heads):
    # comfy/ldm/cogvideo/model.py CogVideoXBlock.forward (ComfyUI fb2b26f) from the q / k
    # projections to the attention call; q_lin / k_lin are self.q(qkv_input) / self.k(qkv_input)
    b, s, c = q_lin.shape
    n, d = heads, c // heads

    q = q_lin.view(b, s, n, d)
    k = k_lin.view(b, s, n, d)

    q = norm_q(q).view(b, s, n, d)
    k = norm_k(k).view(b, s, n, d)

    # Apply rotary embeddings to image tokens only (diffusers format: [B, heads, seq, head_dim])
    if image_rotary_emb is not None:
        q_img = q[:, text_seq_length:].transpose(1, 2)  # [B, heads, img_seq, head_dim]
        k_img = k[:, text_seq_length:].transpose(1, 2)
        q_img = _apply_rotary_emb(q_img, image_rotary_emb)
        k_img = _apply_rotary_emb(k_img, image_rotary_emb)
        q = torch.cat([q[:, :text_seq_length], q_img.transpose(1, 2)], dim=1)
        k = torch.cat([k[:, :text_seq_length], k_img.transpose(1, 2)], dim=1)
        del q_img, k_img

    return q.reshape(b, s, n * d), k.reshape(b, s, n * d)


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


def _tables(seq_vid, head_dim, *, device="cpu", dtype=torch.float32):
    # [S - text, head_dim] cos / sin, interleaved pairs share an angle (repeat_interleave(2))
    # as CogVideoXTransformer3DModel._get_rotary_emb builds them, fp32
    ang = torch.rand(seq_vid, head_dim // 2, device=device) * 6.283
    cos = ang.cos().repeat_interleave(2, dim=-1).to(dtype).contiguous()
    sin = ang.sin().repeat_interleave(2, dim=-1).to(dtype).contiguous()
    return cos, sin


def _inputs(
    batch,
    seq,
    text_len,
    heads,
    head_dim,
    *,
    device="cpu",
    dtype=torch.float32,
    table_dtype=torch.float32,
    fused=False,
):
    # projection outputs [B, S, heads * head_dim]; fused=True gives q, k as chunks of one
    # [B, S, 2 * C] tensor (views with row stride 2 * C)
    channels = heads * head_dim
    if fused:
        qk = (torch.randn(batch, seq, 2 * channels, device=device) * 2).to(dtype)
        q, k = qk.chunk(2, dim=-1)
    else:
        q, k = ((torch.randn(batch, seq, channels, device=device) * 2).to(dtype) for _ in range(2))
    params = []
    for _ in range(2):
        params.append((1 + 0.1 * torch.randn(head_dim, device=device)).to(dtype))
        params.append((0.1 * torch.randn(head_dim, device=device)).to(dtype))
    cos, sin = _tables(seq - text_len, head_dim, device=device, dtype=table_dtype)
    return [q, k, *params, cos, sin]


def _kwargs(inputs, text_len, heads, eps=1e-6):
    return {
        **dict(zip(_NAMES, inputs, strict=True)),
        "text_len": text_len,
        "heads": heads,
        "eps": eps,
    }


def _kitchen(inputs, text_len, heads, eps=1e-6):
    return ck.qk_layernorm_rope_partial(*inputs, text_len, heads, eps)


def _stock(inputs, text_len, heads, eps=1e-6, make_norm=None):
    make_norm = make_norm or _cast_norm
    q, k, wq, bq, wk, bk, cos, sin = inputs
    norm_q, norm_k = make_norm(wq, bq, eps), make_norm(wk, bk, eps)
    return _stock_prep(q, k, norm_q, norm_k, (cos, sin), text_len, heads)


def _reference32(inputs, text_len, heads, eps=1e-6):
    return _stock([t.float() for t in inputs], text_len, heads, eps)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def _err(a, ref32):
    d = (a.float() - ref32).abs()
    return d.max().item(), d.mean().item()


def _call_rule():
    triton_backend = importlib.import_module("comfy_kitchen.backends.triton")
    (rule,) = triton_backend._build_constraints()[OP].call_rules
    return rule


# (batch, S, text_len, heads, head_dim): the VOID template call is batch 2 (cond and uncond
# batched), S 15346 with 226 text tokens, 48 heads of 64; scaled down here in S and heads, plus
# batch 1, no text tokens, one video token and another head_dim
_EAGER_SHAPES = [
    (2, 346, 226, 48, 64),
    (2, 130, 26, 8, 64),
    (1, 130, 26, 8, 64),
    (2, 60, 0, 4, 64),
    (1, 9, 8, 4, 64),
    (1, 40, 7, 3, 32),
]


class TestQkLayernormRopePartialEager:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    @pytest.mark.parametrize(("batch", "seq", "text_len", "heads", "head_dim"), _EAGER_SHAPES)
    def test_matches_stock_chain(self, batch, seq, text_len, heads, head_dim, dtype):
        inputs = _inputs(batch, seq, text_len, heads, head_dim, dtype=dtype)
        with ck.use_backend("eager"):
            out = _kitchen(inputs, text_len, heads)
        for make_norm in (_stock_norm, _cast_norm):
            ref = _stock(inputs, text_len, heads, make_norm=make_norm)
            for o, r in zip(out, ref, strict=True):
                assert o.dtype == dtype and o.shape == r.shape
                assert o.stride() == r.stride()
                assert torch.equal(o, r)

    def test_output_layout(self):
        inputs = _inputs(2, 37, 7, 4, 32)
        with ck.use_backend("eager"):
            out = _kitchen(inputs, 7, 4)
        # what the attention call receives in the stock path: contiguous [B, S, heads * head_dim]
        for o in out:
            assert o.shape == (2, 37, 128)
            assert o.is_contiguous()

    def test_text_rows_only_normalized(self):
        inputs = _inputs(1, 13, 4, 2, 16)
        q, _, wq, bq = inputs[:4]
        with ck.use_backend("eager"):
            out_q, _ = _kitchen(inputs, 4, 2)
        normed = functional.layer_norm(q.view(1, 13, 2, 16), (16,), wq, bq, 1e-6)
        assert torch.equal(out_q.view(1, 13, 2, 16)[:, :4], normed[:, :4])
        assert not torch.equal(out_q.view(1, 13, 2, 16)[:, 4:], normed[:, 4:])

    def test_fused_projection_views(self):
        inputs = _inputs(2, 33, 6, 4, 32, dtype=torch.bfloat16, fused=True)
        assert inputs[0].stride() == (33 * 256, 256, 1)
        with ck.use_backend("eager"):
            out = _kitchen(inputs, 6, 4)
        ref = _stock(inputs, 6, 4)
        for o, r in zip(out, ref, strict=True):
            assert torch.equal(o, r) and o.is_contiguous()

    def test_eps(self):
        inputs = _inputs(1, 37, 7, 4, 32, dtype=torch.bfloat16)
        with ck.use_backend("eager"):
            out = _kitchen(inputs, 7, 4, eps=1e-2)
        ref = _stock(inputs, 7, 4, eps=1e-2, make_norm=_stock_norm)
        for o, r in zip(out, ref, strict=True):
            assert torch.equal(o, r)

    def test_bf16_table(self):
        inputs = _inputs(1, 37, 7, 4, 32, dtype=torch.bfloat16, table_dtype=torch.bfloat16)
        with ck.use_backend("eager"):
            out = _kitchen(inputs, 7, 4)
        for o, r in zip(out, _stock(inputs, 7, 4), strict=True):
            assert torch.equal(o, r)

    @pytest.mark.parametrize(
        ("q_shape", "k_shape", "param_shape", "table_shape", "text_len", "heads"),
        [
            ((1, 37, 64), (1, 37, 64), (16,), (30, 16), 7, 3),  # 64 channels, 3 heads
            ((1, 37, 48), (1, 37, 48), (3,), (30, 3), 7, 16),  # odd head_dim 3
            ((1, 37, 64), (1, 37, 64), (16,), (30, 16), 7, 0),  # no heads
            ((1, 37, 64), (1, 36, 64), (16,), (30, 16), 7, 4),  # k shape differs
            ((37, 64), (37, 64), (16,), (30, 16), 7, 4),  # 2D projections
            ((1, 37, 64), (1, 37, 64), (32,), (30, 16), 7, 4),  # params != head_dim
            ((1, 37, 64), (1, 37, 64), (1, 16), (30, 16), 7, 4),  # 2D params
            ((1, 37, 64), (1, 37, 64), (16,), (37, 16), 7, 4),  # table for all tokens
            ((1, 37, 64), (1, 37, 64), (16,), (30, 8), 7, 4),  # table [.., head_dim / 2]
            ((1, 37, 64), (1, 37, 64), (16,), (1, 30, 16), 7, 4),  # 3D table
            ((1, 37, 64), (1, 37, 64), (16,), (0, 16), 37, 4),  # no video tokens (stock fails)
            ((1, 37, 64), (1, 37, 64), (16,), (0, 16), 38, 4),  # text_len > S
            ((1, 37, 64), (1, 37, 64), (16,), (38, 16), -1, 4),  # negative text_len
        ],
    )
    def test_rejects_bad_inputs(self, q_shape, k_shape, param_shape, table_shape, text_len, heads):
        q, k = torch.randn(q_shape), torch.randn(k_shape)
        params = [torch.randn(param_shape) for _ in range(4)]
        tables = [torch.randn(table_shape) for _ in range(2)]
        with pytest.raises(ValueError):
            _kitchen([q, k, *params, *tables], text_len, heads)

    def test_rejects_mixed_dtypes(self):
        inputs = _inputs(1, 37, 7, 4, 16)
        bad = list(inputs)
        bad[1] = bad[1].to(torch.bfloat16)  # k
        with pytest.raises(ValueError):
            _kitchen(bad, 7, 4)
        # weight and bias in q's dtype, as ComfyUI's LayerNorm casts them
        bad = list(inputs)
        bad[3] = bad[3].to(torch.bfloat16)  # q_bias
        with pytest.raises(ValueError):
            _kitchen(bad, 7, 4)
        # integer table
        bad = list(inputs)
        bad[6] = bad[6].to(torch.int32)
        with pytest.raises(ValueError):
            _kitchen(bad, 7, 4)

    def test_fake_shape_and_strides(self):
        inputs = _inputs(2, 37, 7, 4, 16, dtype=torch.bfloat16)
        out = ck.qk_layernorm_rope_partial(*[t.to("meta") for t in inputs], 7, 4)
        for o in out:
            assert o.shape == (2, 37, 64) and o.is_contiguous()
            assert o.dtype == torch.bfloat16

    def test_cpu_goes_to_eager(self):
        inputs = _inputs(1, 37, 7, 4, 32, dtype=torch.bfloat16)
        assert registry.get_capable_backend(OP, _kwargs(inputs, 7, 4)) == "eager"


def _meta(batch, seq, text_len, heads=48, head_dim=64, fused=False):
    channels = heads * head_dim
    bf16 = torch.bfloat16
    if fused:
        q, k = torch.empty(batch, seq, 2 * channels, device="meta", dtype=bf16).chunk(2, -1)
    else:
        q, k = (torch.empty(batch, seq, channels, device="meta", dtype=bf16) for _ in range(2))
    params = [torch.empty(head_dim, device="meta", dtype=bf16) for _ in range(4)]
    tables = [
        torch.empty(seq - text_len, head_dim, device="meta", dtype=torch.float32) for _ in range(2)
    ]
    return [q, k, *params, *tables]


class TestQkLayernormRopePartialCallRule:
    # the Triton call rule only looks at shapes and layouts, so it runs on meta tensors
    @pytest.mark.parametrize(
        ("batch", "seq", "text_len", "heads", "head_dim", "ok"),
        [
            (2, 15346, 226, 48, 64, True),  # VOID template, cond + uncond batched
            (1, 15346, 226, 48, 64, True),  # batch 1
            (2, 300, 0, 48, 64, True),  # no text tokens
            (1, 300, 20, 8, 16, True),
            (1, 300, 20, 8, 128, True),
            (1, 300, 20, 4, 256, False),  # head_dim > 128
            (1, 300, 20, 8, 96, False),  # head_dim not a power of 2
            (1, 0, 0, 48, 64, False),  # empty sequence
            (0, 300, 20, 48, 64, False),  # empty batch
            (1, 44_739_243, 20, 48, 64, False),  # 2**31 + 16 rows
            (1, 44_739_242, 20, 48, 64, True),
        ],
    )
    def test_call_rule(self, batch, seq, text_len, heads, head_dim, ok):
        inputs = _meta(batch, seq, text_len, heads, head_dim)
        assert _call_rule()(_kwargs(inputs, text_len, heads)).success == ok

    def test_call_rule_layouts(self):
        rule = _call_rule()
        # q and k as chunks of one projection (row stride 2 * C): not contiguous, declined
        assert not rule(_kwargs(_meta(1, 300, 20, fused=True), 20, 48)).success
        inputs = _meta(1, 300, 20)
        # weight with a stride
        w2 = torch.empty(128, device="meta", dtype=torch.bfloat16)[::2]
        assert not rule(_kwargs([*inputs[:2], w2, *inputs[3:]], 20, 48)).success
        # table rows not contiguous
        t2 = torch.empty(280, 128, device="meta", dtype=torch.float32)[:, :64]
        assert not rule(_kwargs([*inputs[:6], t2, inputs[7]], 20, 48)).success

    def test_dtypes(self):
        triton_backend = importlib.import_module("comfy_kitchen.backends.triton")
        params = triton_backend._build_constraints()[OP].params
        assert set(params) == set(_NAMES)
        for name in _NAMES[:6]:
            assert params[name].dtypes == frozenset({torch.bfloat16})
        for name in ("cos", "sin"):
            assert params[name].dtypes == frozenset({torch.float32})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestQkLayernormRopePartialTriton:
    @pytest.fixture(autouse=True)
    def _triton(self):
        pytest.importorskip("triton")
        if not registry.is_available("triton"):
            pytest.skip("triton backend not available")

    @staticmethod
    def _check(inputs, text_len, heads, eps=1e-6):
        kwargs = _kwargs(inputs, text_len, heads, eps)
        assert registry.validate_backend_for_call("triton", OP, kwargs).success
        assert registry.get_capable_backend(OP, kwargs) == "triton"
        with ck.use_backend("triton"):
            got = _kitchen(inputs, text_len, heads, eps)
        ref = eager_backend.qk_layernorm_rope_partial(**kwargs)
        ref32 = _reference32(inputs, text_len, heads, eps)
        for g, r, r32 in zip(got, ref, ref32, strict=True):
            assert g.shape == r.shape and g.stride() == r.stride()
            assert g.dtype == torch.bfloat16
            assert not torch.isnan(g).any()
            assert _rel(g, r) < 1e-2
            assert _rel(g, r32) < 1e-2
            # one rounding: never further from the fp32 reference than the eager chain
            g_max, g_mean = _err(g, r32)
            r_max, r_mean = _err(r, r32)
            assert g_max <= r_max and g_mean <= r_mean
        return got, ref

    @pytest.mark.parametrize(
        ("batch", "seq", "text_len"),
        [
            (2, 346, 226),
            (1, 346, 226),
            (2, 1024, 26),
            (2, 300, 0),  # no text tokens
            (1, 50, 49),  # one video token
            (1, 3, 1),  # short sequence, partial last program
        ],
    )
    def test_matches_eager(self, batch, seq, text_len):
        inputs = _inputs(batch, seq, text_len, 48, 64, device="cuda", dtype=torch.bfloat16)
        self._check(inputs, text_len, 48)

    @pytest.mark.parametrize(("heads", "head_dim"), [(8, 128), (16, 32), (4, 16)])
    def test_other_head_dims(self, heads, head_dim):
        inputs = _inputs(2, 150, 11, heads, head_dim, device="cuda", dtype=torch.bfloat16)
        self._check(inputs, 11, heads)

    def test_eps(self):
        inputs = _inputs(2, 150, 11, 48, 64, device="cuda", dtype=torch.bfloat16)
        self._check(inputs, 11, 48, eps=1e-3)

    def test_text_rows_only_normalized(self):
        inputs = _inputs(2, 150, 11, 48, 64, device="cuda", dtype=torch.bfloat16)
        got, _ = self._check(inputs, 11, 48)
        ref32 = _reference32(inputs, 11, 48)
        # text rows: LayerNorm only (the fp32 reference leaves them unrotated)
        for g, r32 in zip(got, ref32, strict=True):
            torch.testing.assert_close(g[:, :11].float(), r32[:, :11], rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("hip", [False, True])
    def test_both_launch_configs(self, monkeypatch, hip):
        # the CUDA and HIP launch configurations, on whichever GPU
        kernel = importlib.import_module("comfy_kitchen.backends.triton." + OP)
        monkeypatch.setattr(kernel, "_HIP", hip)
        assert kernel._launch_config() == ((64, 8) if hip else (32, 4))
        inputs = _inputs(2, 346, 226, 48, 64, device="cuda", dtype=torch.bfloat16)
        kwargs = _kwargs(inputs, 226, 48)
        got = kernel.qk_layernorm_rope_partial(**kwargs)
        ref = eager_backend.qk_layernorm_rope_partial(**kwargs)
        ref32 = _reference32(inputs, 226, 48)
        for g, r, r32 in zip(got, ref, ref32, strict=True):
            assert _rel(g, r32) < 1e-2
            assert _err(g, r32)[0] <= _err(r, r32)[0]

    def test_real_shape_served(self):
        # the VOID template call at its captured size: B 2, S 15346, 226 text tokens, 48 x 64
        inputs = _inputs(2, 15346, 226, 48, 64, device="cuda", dtype=torch.bfloat16)
        got, _ = self._check(inputs, 226, 48)
        assert got[0].shape == (2, 15346, 3072)

    def test_unsupported_inputs_go_to_eager(self):
        def expect_eager(inputs, text_len, heads):
            kwargs = _kwargs(inputs, text_len, heads)
            assert not registry.validate_backend_for_call("triton", OP, kwargs).success
            assert registry.get_capable_backend(OP, kwargs) == "eager"
            with ck.use_backend("triton"):
                got = _kitchen(inputs, text_len, heads)
            ref = _stock(inputs, text_len, heads)
            for g, r in zip(got, ref, strict=True):
                assert torch.equal(g, r)

        # fp16 / fp32 are not served by the triton kernel
        for dtype in (torch.float16, torch.float32):
            expect_eager(_inputs(1, 37, 7, 4, 64, device="cuda", dtype=dtype), 7, 4)
        bf16 = torch.bfloat16
        # a bf16 table
        expect_eager(_inputs(1, 37, 7, 4, 64, device="cuda", dtype=bf16, table_dtype=bf16), 7, 4)
        # head_dim 256
        expect_eager(_inputs(1, 37, 7, 1, 256, device="cuda", dtype=bf16), 7, 1)
        # q and k as chunks of one projection (not contiguous)
        expect_eager(_inputs(1, 37, 7, 4, 64, device="cuda", dtype=bf16, fused=True), 7, 4)
        # CPU tensors go to eager
        inputs = _inputs(1, 37, 7, 4, 64, dtype=bf16)
        kwargs = _kwargs(inputs, 7, 4)
        assert not registry.validate_backend_for_call("triton", OP, kwargs).success
        assert registry.get_capable_backend(OP, kwargs) == "eager"
