# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""joint_qk_rms_rope_cat -- Qwen-Image dual-stream attention prep: QK RMSNorm + concat + RoPE.

The eager backend is the sequence ComfyUI's Qwen-Image Attention.forward runs today (the
view / transpose / contiguous of the six projections, the four RMSNorms, the txt-then-img
torch.cat of q, k and v, and apply_rope1 on the joint q and k), so it must match the stock
code exactly, including the output layout the attention call reads. On CUDA the Triton kernel
computes the norm and the rotation in fp32 and rounds once, so q and k are compared to eager
and to an fp32 reference with a tolerance and must never be further from the fp32 reference
than eager; on HIP it rounds like the unfused chain (F.rms_norm, then the HIP apply_rope1,
which rounds as eager does), so q and k must equal that chain. v is a copy and must be exact.
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

OP = "joint_qk_rms_rope_cat"
_NAMES = (
    "img_q",
    "img_k",
    "img_v",
    "txt_q",
    "txt_k",
    "txt_v",
    "img_q_weight",
    "img_k_weight",
    "txt_q_weight",
    "txt_k_weight",
    "freqs_cis",
)


def _stock_prep(projections, norms, freqs_cis, heads):
    # comfy/ldm/qwen_image/model.py Attention.forward (ComfyUI fb2b26f) from the projections
    # to the attention call, no attn1_patch; apply_rope1 is comfy.ldm.flux.math.apply_rope1,
    # which calls comfy-kitchen's apply_rope1 outside training
    img_q, img_k, img_v, txt_q, txt_k, txt_v = projections
    norm_q, norm_k, norm_added_q, norm_added_k = norms
    batch_size = img_q.shape[0]
    seq_img = img_q.shape[1]
    seq_txt = txt_q.shape[1]

    img_query = img_q.view(batch_size, seq_img, heads, -1).transpose(1, 2).contiguous()
    img_key = img_k.view(batch_size, seq_img, heads, -1).transpose(1, 2).contiguous()
    img_value = img_v.view(batch_size, seq_img, heads, -1).transpose(1, 2)

    txt_query = txt_q.view(batch_size, seq_txt, heads, -1).transpose(1, 2).contiguous()
    txt_key = txt_k.view(batch_size, seq_txt, heads, -1).transpose(1, 2).contiguous()
    txt_value = txt_v.view(batch_size, seq_txt, heads, -1).transpose(1, 2)

    img_query = norm_q(img_query)
    img_key = norm_k(img_key)
    txt_query = norm_added_q(txt_query)
    txt_key = norm_added_k(txt_key)

    joint_query = torch.cat([txt_query, img_query], dim=2)
    joint_key = torch.cat([txt_key, img_key], dim=2)
    joint_value = torch.cat([txt_value, img_value], dim=2)

    joint_query = ck.apply_rope1(joint_query, freqs_cis)
    joint_key = ck.apply_rope1(joint_key, freqs_cis)
    return joint_query, joint_key, joint_value


def _stock_norm(weight, eps=1e-6):
    norm = nn.RMSNorm(weight.shape[0], eps=eps, device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        norm.weight.copy_(weight)
    return norm


def _cast_norm(weight, eps=1e-6):
    # comfy.ops RMSNorm with comfy_cast_weights: F.rms_norm(input, normalized_shape, weight, eps)
    return lambda x: functional.rms_norm(x, (weight.shape[0],), weight, eps)


def _freqs(batch, seq, head_dim, *, device="cpu", dtype=torch.float32):
    # the [B, 1, L, head_dim / 2, 2, 2] rotation table of comfy.ldm.flux.math.rope, then
    # .to(x.dtype).contiguous() as QwenImageTransformer2DModel does
    ang = torch.rand(batch, 1, seq, head_dim // 2, device=device) * 6.283
    out = torch.stack([ang.cos(), -ang.sin(), ang.sin(), ang.cos()], dim=-1)
    return out.view(batch, 1, seq, head_dim // 2, 2, 2).to(dtype).contiguous()


def _inputs(
    batch,
    seq_img,
    seq_txt,
    heads,
    head_dim,
    *,
    device="cpu",
    dtype=torch.float32,
    freqs_batch=1,
    fused=False,
):
    # projection outputs [B, L, heads * head_dim]; fused=True gives q, k, v as chunks of one
    # [B, L, 3 * C] projection (views with row stride 3 * C)
    channels = heads * head_dim

    def stream(seq):
        if fused:
            qkv = torch.randn(batch, seq, 3 * channels, device=device).to(dtype)
            return list(qkv.chunk(3, dim=-1))
        return [torch.randn(batch, seq, channels, device=device).to(dtype) for _ in range(3)]

    projections = stream(seq_img) + stream(seq_txt)
    weights = [(1 + 0.1 * torch.randn(head_dim, device=device)).to(dtype) for _ in range(4)]
    freqs = _freqs(freqs_batch, seq_txt + seq_img, head_dim, device=device, dtype=dtype)
    return projections, weights, freqs


def _kwargs(projections, weights, freqs, heads, eps=1e-6):
    return {
        **dict(zip(_NAMES, [*projections, *weights, freqs], strict=True)),
        "heads": heads,
        "eps": eps,
    }


def _kitchen(projections, weights, freqs, heads, eps=1e-6):
    return ck.joint_qk_rms_rope_cat(*projections, *weights, freqs, heads, eps)


def _reference32(projections, weights, freqs, heads, eps=1e-6):
    norms = [_cast_norm(w.float(), eps) for w in weights]
    with ck.use_backend("eager"):
        return _stock_prep([p.float() for p in projections], norms, freqs.float(), heads)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def _err(a, ref32):
    d = (a.float() - ref32).abs()
    return d.max().item(), d.mean().item()


def _call_rule():
    triton_backend = importlib.import_module("comfy_kitchen.backends.triton")
    (rule,) = triton_backend._build_constraints()[OP].call_rules
    return rule


# (batch, L_img, L_txt, heads, head_dim, freqs batch): the template calls are batch 1, 24 heads
# of 128, L_img 6889 / 12336 and L_txt 46 to 425 (cond and uncond as separate calls); scaled
# down here in L, plus batch 2 with a shared and a per-item table and another head_dim
_EAGER_SHAPES = [
    (1, 289, 21, 24, 128, 1),
    (1, 289, 5, 24, 128, 1),
    (2, 64, 13, 4, 128, 1),
    (2, 64, 13, 4, 128, 2),
    (1, 40, 7, 3, 32, 1),
]


class TestJointQkRmsRopeCatEager:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    @pytest.mark.parametrize(
        ("batch", "seq_img", "seq_txt", "heads", "head_dim", "freqs_batch"), _EAGER_SHAPES
    )
    def test_matches_stock_chain(
        self, batch, seq_img, seq_txt, heads, head_dim, freqs_batch, dtype
    ):
        projections, weights, freqs = _inputs(
            batch, seq_img, seq_txt, heads, head_dim, dtype=dtype, freqs_batch=freqs_batch
        )
        with ck.use_backend("eager"):
            out = _kitchen(projections, weights, freqs, heads)
        for make_norm in (_stock_norm, _cast_norm):
            ref = _stock_prep(projections, [make_norm(w) for w in weights], freqs, heads)
            for o, r in zip(out, ref, strict=True):
                assert o.dtype == dtype and o.shape == r.shape
                assert o.stride() == r.stride()
                assert torch.equal(o, r)

    def test_output_layout(self):
        projections, weights, freqs = _inputs(2, 30, 7, 4, 32)
        with ck.use_backend("eager"):
            out = _kitchen(projections, weights, freqs, 4)
        # what the attention call receives in the stock path: contiguous [B, heads, L, head_dim]
        for o in out:
            assert o.shape == (2, 4, 37, 32)
            assert o.is_contiguous()

    def test_text_rows_first(self):
        projections, weights, freqs = _inputs(1, 9, 4, 2, 16)
        with ck.use_backend("eager"):
            _, _, v = _kitchen(projections, weights, freqs, 2)
        img_v, txt_v = projections[2], projections[5]
        assert torch.equal(v[:, :, :4], txt_v.view(1, 4, 2, 16).transpose(1, 2))
        assert torch.equal(v[:, :, 4:], img_v.view(1, 9, 2, 16).transpose(1, 2))

    def test_fused_projection_views(self):
        projections, weights, freqs = _inputs(2, 33, 6, 4, 32, dtype=torch.bfloat16, fused=True)
        assert projections[0].stride() == (33 * 384, 384, 1)
        with ck.use_backend("eager"):
            out = _kitchen(projections, weights, freqs, 4)
        ref = _stock_prep(projections, [_cast_norm(w) for w in weights], freqs, 4)
        for o, r in zip(out, ref, strict=True):
            assert torch.equal(o, r)

    def test_eps(self):
        projections, weights, freqs = _inputs(1, 30, 7, 4, 32, dtype=torch.bfloat16)
        with ck.use_backend("eager"):
            out = _kitchen(projections, weights, freqs, 4, eps=1e-2)
        ref = _stock_prep(projections, [_stock_norm(w, eps=1e-2) for w in weights], freqs, 4)
        for o, r in zip(out, ref, strict=True):
            assert torch.equal(o, r)

    @pytest.mark.parametrize(
        ("img_shape", "txt_shape", "weight_shape", "freqs_shape", "heads"),
        [
            ((1, 30, 64), (1, 7, 64), (16,), (1, 1, 37, 8, 2, 2), 3),  # 64 channels, 3 heads
            ((1, 30, 48), (1, 7, 48), (16,), (1, 1, 37, 8, 2, 2), 16),  # odd head_dim 3
            ((1, 30, 64), (1, 7, 64), (16,), (1, 1, 37, 8, 2, 2), 0),  # no heads
            ((1, 30, 64), (2, 7, 64), (16,), (1, 1, 37, 8, 2, 2), 4),  # batch differs
            ((1, 30, 64), (1, 7, 32), (16,), (1, 1, 37, 8, 2, 2), 4),  # channels differ
            ((30, 64), (7, 64), (16,), (1, 1, 37, 8, 2, 2), 4),  # 2D projections
            ((1, 30, 64), (1, 7, 64), (32,), (1, 1, 37, 8, 2, 2), 4),  # weight != head_dim
            ((1, 30, 64), (1, 7, 64), (1, 16), (1, 1, 37, 8, 2, 2), 4),  # 2D weight
            ((1, 30, 64), (1, 7, 64), (16,), (1, 1, 30, 8, 2, 2), 4),  # table for img only
            ((1, 30, 64), (1, 7, 64), (16,), (1, 1, 38, 8, 2, 2), 4),  # longer table
            ((1, 30, 64), (1, 7, 64), (16,), (1, 37, 8, 2, 2), 4),  # 5D table
            ((1, 30, 64), (1, 7, 64), (16,), (1, 1, 37, 16, 2), 4),  # [.., D, 2] table
            ((2, 30, 64), (2, 7, 64), (16,), (3, 1, 37, 8, 2, 2), 4),  # table batch 3
        ],
    )
    def test_rejects_bad_inputs(self, img_shape, txt_shape, weight_shape, freqs_shape, heads):
        img = [torch.randn(img_shape) for _ in range(3)]
        txt = [torch.randn(txt_shape) for _ in range(3)]
        weights = [torch.randn(weight_shape) for _ in range(4)]
        with pytest.raises(ValueError):
            _kitchen(img + txt, weights, torch.randn(freqs_shape), heads)

    def test_rejects_mixed_dtypes(self):
        projections, weights, freqs = _inputs(1, 30, 7, 4, 16)
        bad = list(projections)
        bad[4] = bad[4].to(torch.bfloat16)  # txt_k
        with pytest.raises(ValueError):
            _kitchen(bad, weights, freqs, 4)
        # the weights in the projections' dtype, as ComfyUI's RMSNorm casts them
        bad = [w.to(torch.bfloat16) for w in weights]
        with pytest.raises(ValueError):
            _kitchen(projections, bad, freqs, 4)

    def test_fake_shape_and_strides(self):
        projections, weights, freqs = _inputs(2, 30, 7, 4, 16, dtype=torch.bfloat16)
        to_meta = [t.to("meta") for t in (*projections, *weights, freqs)]
        out = ck.joint_qk_rms_rope_cat(*to_meta, 4)
        for o in out:
            assert o.shape == (2, 4, 37, 16) and o.is_contiguous()
            assert o.dtype == torch.bfloat16

    def test_cpu_goes_to_eager(self):
        projections, weights, freqs = _inputs(1, 30, 7, 4, 32, dtype=torch.bfloat16)
        kwargs = _kwargs(projections, weights, freqs, 4)
        assert registry.get_capable_backend(OP, kwargs) == "eager"


def _meta(batch, seq_img, seq_txt, heads=24, head_dim=128, dtype=torch.bfloat16, fused=False):
    channels = heads * head_dim

    def stream(seq):
        if fused:
            return list(
                torch.empty(batch, seq, 3 * channels, device="meta", dtype=dtype).chunk(3, -1)
            )
        return [torch.empty(batch, seq, channels, device="meta", dtype=dtype) for _ in range(3)]

    weights = [torch.empty(head_dim, device="meta", dtype=dtype) for _ in range(4)]
    freqs = torch.empty(1, 1, seq_txt + seq_img, head_dim // 2, 2, 2, device="meta", dtype=dtype)
    return stream(seq_img) + stream(seq_txt), weights, freqs


class TestJointQkRmsRopeCatCallRule:
    # the Triton call rule only looks at shapes and layouts, so it runs on meta tensors
    @pytest.mark.parametrize(
        ("batch", "seq_img", "seq_txt", "heads", "head_dim", "ok"),
        [
            (1, 6889, 210, 24, 128, True),  # Qwen-Image 2512, 1328 x 1328, cond
            (1, 6889, 46, 24, 128, True),  # uncond
            (1, 12336, 425, 24, 128, True),  # Qwen-Image Edit 2511
            (2, 6889, 210, 24, 128, True),  # batch 2
            (1, 300, 20, 8, 64, True),
            (1, 300, 20, 8, 16, True),
            (1, 300, 20, 4, 256, False),  # head_dim > 128
            (1, 300, 20, 8, 96, False),  # head_dim not a power of 2
            (1, 300, 0, 24, 128, False),  # empty text stream
            (1, 0, 20, 24, 128, False),  # empty image stream
            (0, 300, 20, 24, 128, False),  # empty batch
            (1, 699_031, 20, 24, 128, False),  # 699_051 rows x 3072 >= 2**31 elements
            (1, 699_030, 20, 24, 128, True),
        ],
    )
    def test_call_rule(self, batch, seq_img, seq_txt, heads, head_dim, ok):
        projections, weights, freqs = _meta(batch, seq_img, seq_txt, heads, head_dim)
        assert _call_rule()(_kwargs(projections, weights, freqs, heads)).success == ok

    def test_call_rule_layouts(self):
        rule = _call_rule()
        # q, k, v as chunks of one fused projection (row stride 3 * C): served
        projections, weights, freqs = _meta(1, 300, 20, fused=True)
        assert rule(_kwargs(projections, weights, freqs, 24)).success
        # last dim not contiguous
        projections, weights, freqs = _meta(1, 300, 20)
        strided = torch.empty(1, 300, 6144, device="meta", dtype=torch.bfloat16)[..., ::2]
        assert not rule(
            _kwargs([strided, strided, strided, *projections[3:]], weights, freqs, 24)
        ).success
        # k with other strides than q in the same stream
        other = torch.empty(1, 300, 4096, device="meta", dtype=torch.bfloat16)[..., :3072]
        bad = [projections[0], other, *projections[2:]]
        assert not rule(_kwargs(bad, weights, freqs, 24)).success
        # weight with a stride
        w2 = torch.empty(256, device="meta", dtype=torch.bfloat16)[::2]
        assert not rule(_kwargs(projections, [w2, *weights[1:]], freqs, 24)).success
        # table rows not contiguous
        f2 = torch.empty(1, 1, 320, 64, 2, 4, device="meta", dtype=torch.bfloat16)[..., :2]
        assert not rule(_kwargs(projections, weights, f2, 24)).success

    def test_bf16_only(self):
        triton_backend = importlib.import_module("comfy_kitchen.backends.triton")
        params = triton_backend._build_constraints()[OP].params
        assert set(params) == set(_NAMES)
        for name in _NAMES:
            assert params[name].dtypes == frozenset({torch.bfloat16})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestJointQkRmsRopeCatTriton:
    @pytest.fixture(autouse=True)
    def _triton(self):
        pytest.importorskip("triton")
        if not registry.is_available("triton"):
            pytest.skip("triton backend not available")

    @staticmethod
    @contextlib.contextmanager
    def _served(expected):
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

    @staticmethod
    def _numerics(got, ref, projections, weights, freqs, heads, eps=1e-6):
        # ref: the eager backend's output for the same inputs
        kernel = importlib.import_module("comfy_kitchen.backends.triton." + OP)
        if kernel._STOCK_ROUNDING:
            # the unfused chain: F.rms_norm, then the HIP apply_rope1 (eager rounds the same)
            rope = "hip" if registry.is_available("hip") else "eager"
            norms = [_cast_norm(w, eps) for w in weights]
            with ck.use_backend(rope):
                stock = _stock_prep(projections, norms, freqs, heads)
            for g, s in zip(got, stock, strict=True):
                assert torch.equal(g, s)
            return
        ref32 = _reference32(projections, weights, freqs, heads, eps)
        for g, r, r32 in zip(got[:2], ref[:2], ref32[:2], strict=True):
            assert _rel(g, r) < 1e-2
            assert _rel(g, r32) < 1e-2
            # one rounding: never further from the fp32 reference than the eager chain
            g_max, g_mean = _err(g, r32)
            r_max, r_mean = _err(r, r32)
            assert g_max <= r_max and g_mean <= r_mean
        # v is a copy
        assert torch.equal(got[2], ref[2])

    @classmethod
    def _check(cls, projections, weights, freqs, heads, eps=1e-6):
        kwargs = _kwargs(projections, weights, freqs, heads, eps)
        with cls._served("triton"), ck.use_backend("triton"):
            got = _kitchen(projections, weights, freqs, heads, eps)
        ref = eager_backend.joint_qk_rms_rope_cat(**kwargs)
        for g, r in zip(got, ref, strict=True):
            assert g.shape == r.shape and g.stride() == r.stride()
            assert g.dtype == torch.bfloat16
        cls._numerics(got, ref, projections, weights, freqs, heads, eps)
        return got, ref

    @pytest.mark.parametrize(
        ("batch", "seq_img", "seq_txt", "freqs_batch"),
        [
            (1, 289, 21, 1),
            (1, 289, 5, 1),  # short text (uncond)
            (1, 1024, 77, 1),
            (2, 100, 13, 1),  # batch 2, shared table
            (2, 100, 13, 2),  # batch 2, per-item table
            (1, 7, 3, 1),  # fewer rows than one program
        ],
    )
    def test_matches_eager(self, batch, seq_img, seq_txt, freqs_batch):
        projections, weights, freqs = _inputs(
            batch,
            seq_img,
            seq_txt,
            24,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            freqs_batch=freqs_batch,
        )
        self._check(projections, weights, freqs, 24)

    @pytest.mark.parametrize(("heads", "head_dim"), [(8, 64), (16, 32), (4, 16)])
    def test_other_head_dims(self, heads, head_dim):
        projections, weights, freqs = _inputs(
            1, 150, 11, heads, head_dim, device="cuda", dtype=torch.bfloat16
        )
        self._check(projections, weights, freqs, heads)

    def test_eps(self):
        projections, weights, freqs = _inputs(
            1, 150, 11, 24, 128, device="cuda", dtype=torch.bfloat16
        )
        self._check(projections, weights, freqs, 24, eps=1e-3)

    def test_large_values(self):
        # outputs in the hundreds (bf16 step 2 to 4), where a different rounding shows most:
        # a few large weight entries and rows with large inputs
        projections, weights, freqs = _inputs(
            1, 1024, 77, 24, 128, device="cuda", dtype=torch.bfloat16
        )
        for w in weights:
            w[::29] = 48
        for p in projections:
            p[:, ::5] *= 100
        self._check(projections, weights, freqs, 24)

    @pytest.mark.parametrize("layout", ["fused", "batch_step2"])
    def test_kernel_layouts(self, layout):
        if layout == "fused":
            projections, weights, freqs = _inputs(
                2, 150, 11, 24, 128, device="cuda", dtype=torch.bfloat16, fused=True
            )
        else:
            projections, weights, freqs = _inputs(
                4, 150, 11, 24, 128, device="cuda", dtype=torch.bfloat16
            )
            projections = [p[::2] for p in projections]
        got, _ = self._check(projections, weights, freqs, 24)
        with ck.use_backend("triton"):
            want = _kitchen([p.contiguous() for p in projections], weights, freqs, 24)
        for g, w in zip(got, want, strict=True):
            assert torch.equal(g, w)

    @pytest.mark.parametrize("hip", [False, True])
    def test_both_launch_configs(self, monkeypatch, hip):
        # the CUDA (one head per program) and HIP (head loop) configurations, on whichever GPU
        kernel = importlib.import_module("comfy_kitchen.backends.triton." + OP)
        monkeypatch.setattr(kernel, "_HIP", hip)
        assert kernel._launch_config()[2] == hip
        projections, weights, freqs = _inputs(
            2, 150, 11, 24, 128, device="cuda", dtype=torch.bfloat16, freqs_batch=2
        )
        kwargs = _kwargs(projections, weights, freqs, 24)
        got = kernel.joint_qk_rms_rope_cat(**kwargs)
        ref = eager_backend.joint_qk_rms_rope_cat(**kwargs)
        self._numerics(got, ref, projections, weights, freqs, 24)

    def test_real_shape_served(self):
        # the Qwen-Image 2512 cond call at its captured size: L_img 6889, L_txt 210
        projections, weights, freqs = _inputs(
            1, 6889, 210, 24, 128, device="cuda", dtype=torch.bfloat16
        )
        got, _ = self._check(projections, weights, freqs, 24)
        assert got[0].shape == (1, 24, 7099, 128)

    def test_unsupported_inputs_go_to_eager(self):
        def expect_eager(projections, weights, freqs, heads):
            kwargs = _kwargs(projections, weights, freqs, heads)
            assert not registry.validate_backend_for_call("triton", OP, kwargs).success
            assert registry.get_capable_backend(OP, kwargs) == "eager"
            with self._served("eager"), ck.use_backend("triton"):
                got = _kitchen(projections, weights, freqs, heads)
            with ck.use_backend("eager"):
                ref = _stock_prep(projections, [_cast_norm(w) for w in weights], freqs, heads)
            for g, r in zip(got, ref, strict=True):
                assert torch.equal(g, r)

        # fp16 / fp32 are not served by the triton kernel
        for dtype in (torch.float16, torch.float32):
            expect_eager(*_inputs(1, 30, 7, 4, 64, device="cuda", dtype=dtype), 4)
        bf16 = torch.bfloat16
        projections, weights, freqs = _inputs(1, 30, 7, 4, 64, device="cuda", dtype=bf16)
        # an fp32 table with bf16 projections
        expect_eager(projections, weights, freqs.float(), 4)
        # head_dim 256
        w256 = [torch.randn(256, device="cuda").to(bf16) for _ in range(4)]
        expect_eager(projections, w256, _freqs(1, 37, 256, device="cuda", dtype=bf16), 1)
        # last dim not contiguous, k with other strides than q
        strided = [torch.randn(1, 30, 512, device="cuda").to(bf16)[..., ::2] for _ in range(3)]
        expect_eager(strided + projections[3:], weights, freqs, 4)
        other = torch.randn(1, 30, 300, device="cuda").to(bf16)[..., :256]
        expect_eager([projections[0], other, *projections[2:]], weights, freqs, 4)
        # CPU tensors go to eager
        projections, weights, freqs = _inputs(1, 30, 7, 4, 64, dtype=bf16)
        kwargs = _kwargs(projections, weights, freqs, 4)
        assert not registry.validate_backend_for_call("triton", OP, kwargs).success
        assert registry.get_capable_backend(OP, kwargs) == "eager"
