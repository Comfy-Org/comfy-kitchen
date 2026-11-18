# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Nested 3/4/5-bit W4A8 codebook and the packed LSB tier: a weight narrows by dropping
the low bit of its codes, and the record stream without its tier tensor is that narrower
weight."""

import dataclasses

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends import cuda as cuda_backend
from comfy_kitchen.backends.eager import w4a8_int8 as eager_w4a8
from comfy_kitchen.tensor import AsymW4A8Int8Layout, QuantizedTensor
from comfy_kitchen.tensor.w4a8_int8 import cat_packed_weights
from comfy_kitchen.tensor.w4a8_packed import (
    _mma_bits,
    _pack_codes,
    _unpack_codes,
    nested_w4a8_codebook,
    pack_w4a8_mma_weight,
    unpack_w4a8_mma_weight,
    w4a8_level_table,
    w4a8_mma_packed_numel,
    w4a8_mma_tier_numel,
)
from tests.conftest import requires_cuda_backend
from tests.test_w6a8 import rel_l2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N, K = 64, 512


@pytest.fixture
def weight(seed):
    return torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16) * 0.02


def _scales(n, k):
    return (torch.rand(n, k // 16, device=DEVICE) * 2).to(torch.float8_e4m3fn)


def _nested(weight, bits):
    return eager_w4a8.quantize_w4a8_int8_weight(weight, bits=bits, codebook_tensor=nested_w4a8_codebook())


@pytest.mark.parametrize("bits", [4, 5])
@pytest.mark.parametrize("compact", [False, True])
def test_cat_packed_weights_preserves_rows_and_tiers(bits, compact, seed):
    weights = []
    for _ in range(2):
        weight = torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16) * 0.02
        canonical = QuantizedTensor.from_float(weight, "AsymW4A8Int8Layout", bits=bits, codebook_tensor=nested_w4a8_codebook())
        stream, tier = pack_w4a8_mma_weight(canonical._qdata, canonical._params.scale, lsb_tier=True)
        params = dataclasses.replace(canonical._params, scale=canonical._params.scale.new_empty(0),
                                     mma_packed=True, lsb_tier=None if compact else tier)
        weights.append(QuantizedTensor(stream, "AsymW4A8Int8Layout", params))
    original = [weight._qdata.clone() for weight in weights]
    combined = cat_packed_weights(tuple(weights))
    assert combined.shape == (2 * N, K)
    assert AsymW4A8Int8Layout.bits(combined) == bits - compact
    assert torch.equal(combined.dequantize(), torch.cat([weight.dequantize() for weight in weights]))
    assert all(torch.equal(weight._qdata, saved) for weight, saved in zip(weights, original, strict=True))
    if not compact:
        narrow = QuantizedTensor(weights[1]._qdata, "AsymW4A8Int8Layout", dataclasses.replace(weights[1]._params, lsb_tier=None))
        assert cat_packed_weights((weights[0], narrow)) is None
        assert all(torch.equal(weight._qdata, saved) for weight, saved in zip(weights, original, strict=True))


class TestCodebook:
    def test_tables_nest_by_code_prefix(self):
        cb = nested_w4a8_codebook()
        for bits in (3, 4):
            parent, child = w4a8_level_table(cb, bits), w4a8_level_table(cb, bits + 1)
            assert parent.numel() == 1 << bits and child.numel() == 2 << bits
            # levels ascend with the code and each parent sits between its two children
            assert torch.all(child[1:] > child[:-1]) and torch.all(parent[1:] > parent[:-1])
            assert torch.all(child[0::2] < parent) and torch.all(parent < child[1::2])
        with pytest.raises(ValueError):
            w4a8_level_table(cb, 6)

    def test_narrowing_codes_is_quantizing_narrower(self, weight):
        q5, s5, c5, corr, cb = _nested(weight, 5)
        assert corr is None and cb.numel() == 56 and q5.shape == (N, K * 5 // 8)
        codes = _unpack_codes(q5, K, 5)
        errors = {}
        for bits in (5, 4, 3):
            qb, sb, cb_, _, _ = _nested(weight, bits)
            assert torch.equal(qb, _pack_codes(codes >> (5 - bits), bits))
            assert torch.equal(sb.view(torch.uint8), s5.view(torch.uint8)) and torch.equal(cb_, c5)
            deq = eager_w4a8.dequantize_w4a8_int8_weight(qb, sb, cb_, codebook=cb, output_dtype=torch.float32)
            errors[bits] = rel_l2(deq, weight)
        assert errors[5] < errors[4] < errors[3] < 0.2, errors

    def test_five_bit_needs_the_nested_symmetric_codebook(self, weight):
        with pytest.raises(ValueError):
            eager_w4a8.quantize_w4a8_int8_weight(weight, bits=5)
        with pytest.raises(ValueError):
            eager_w4a8.quantize_w4a8_int8_weight(weight, bits=5, codebook_tensor=nested_w4a8_codebook(), symmetric=False)
        with pytest.raises(ValueError):
            eager_w4a8.quantize_w4a8_int8_weight(weight, bits=4, codebook_tensor=nested_w4a8_codebook(), stochastic_rounding=1)
        q = torch.zeros(N, K * 5 // 8, device=DEVICE, dtype=torch.int8)
        s = torch.ones(N, K // 16, device=DEVICE)
        with pytest.raises(ValueError):  # 5-bit codes need the nested table
            eager_w4a8.validate_w4a8_operands(q, s, torch.ones(N, device=DEVICE), torch.zeros(32, device=DEVICE), None, 16, 256)


class TestPackedTier:
    @pytest.mark.parametrize("bits", [4, 5])
    def test_pack_unpack_roundtrip_and_stream_is_narrower_weight(self, bits, seed):
        codes = torch.randint(0, 1 << bits, (N, K), device=DEVICE, dtype=torch.int32)
        qdata, s_rel = _pack_codes(codes, bits), _scales(N, K)
        stream, tier = pack_w4a8_mma_weight(qdata, s_rel, lsb_tier=True)
        assert stream.numel() == w4a8_mma_packed_numel(N, K, bits - 1) == N * K * (2 * bits - 1) // 16
        assert tier.dtype == torch.uint8 and tier.numel() == w4a8_mma_tier_numel(N, K) == N * K // 8
        assert _mma_bits(stream.numel(), N, K) == bits - 1
        q2, s2 = unpack_w4a8_mma_weight(stream, N, K, lsb_tier=tier)
        assert torch.equal(q2, qdata) and torch.equal(s2.view(torch.uint8), s_rel.view(torch.uint8))
        qn, sn = unpack_w4a8_mma_weight(stream, N, K)
        assert torch.equal(_unpack_codes(qn, K, bits - 1), codes >> 1)
        assert torch.equal(sn.view(torch.uint8), s_rel.view(torch.uint8))

    def test_tier_widths(self, seed):
        s_rel = _scales(N, K)
        with pytest.raises(ValueError):  # 3-bit codes have no low bit to split off
            pack_w4a8_mma_weight(_pack_codes(torch.zeros(N, K, device=DEVICE, dtype=torch.int32), 3), s_rel, lsb_tier=True)
        with pytest.raises(ValueError):  # 5-bit codes only exist with a tier
            pack_w4a8_mma_weight(_pack_codes(torch.zeros(N, K, device=DEVICE, dtype=torch.int32), 5), s_rel)

    @pytest.mark.parametrize("bits", [4, 5])
    def test_eager_packed_linear_matches_canonical(self, weight, bits):
        qdata, s_rel, s_channel, _, cb = _nested(weight, bits)
        stream, tier = pack_w4a8_mma_weight(qdata, s_rel, lsb_tier=True)
        x = torch.randn(4, K, device=DEVICE, dtype=torch.bfloat16)
        ref = eager_w4a8.w4a8_int8_linear(x, qdata, s_rel, s_channel, codebook=cb)
        got = eager_w4a8.w4a8_packed_linear(x, stream, s_rel.new_empty(0), s_channel, codebook=cb, lsb_tier=tier)
        assert torch.equal(got, ref)
        # the stream alone decodes as the plain narrower weight
        narrow = _pack_codes(_unpack_codes(qdata, K, bits) >> 1, bits - 1)
        ref = eager_w4a8.w4a8_int8_linear(x, narrow, s_rel, s_channel, codebook=cb)
        got = eager_w4a8.w4a8_packed_linear(x, stream, s_rel.new_empty(0), s_channel, codebook=cb)
        assert torch.equal(got, ref)

    def test_layout_params_and_state_dict(self, weight):
        qdata, s_rel, _, _, cb = _nested(weight, 5)
        qt = QuantizedTensor.from_float(weight, "AsymW4A8Int8Layout", bits=5, codebook_tensor=cb)
        assert AsymW4A8Int8Layout.bits(qt) == 5 and qt._params.codebook.numel() == 56
        canonical = qt.dequantize()
        stream, tier = pack_w4a8_mma_weight(qdata, s_rel, lsb_tier=True)
        params = dataclasses.replace(qt._params, scale=s_rel.new_empty(0), mma_packed=True, lsb_tier=tier)
        pt = QuantizedTensor(stream, "AsymW4A8Int8Layout", params)
        assert AsymW4A8Int8Layout.bits(pt) == 5
        assert torch.equal(pt.dequantize(), canonical)
        sd = pt.state_dict("w")
        assert set(sd) == {"w", "w_s_channel", "w_codebook", "w_lsb_tier"}
        assert sd["w"].data_ptr() == stream.data_ptr() and sd["w_lsb_tier"].data_ptr() == tier.data_ptr()
        # the tier is the last flattened tensor, so a copy without it is a prefix of one with it
        inner, _ = pt.__tensor_flatten__()
        assert inner[-1] == "_param_lsb_tier"
        narrow = QuantizedTensor(stream, "AsymW4A8Int8Layout", dataclasses.replace(params, lsb_tier=None))
        assert AsymW4A8Int8Layout.bits(narrow) == 4
        assert torch.equal(narrow.dequantize(), QuantizedTensor.from_float(weight, "AsymW4A8Int8Layout", bits=4, codebook_tensor=cb).dequantize())
        with pytest.raises(ValueError):  # a tier only exists on packed nested storage
            dataclasses.replace(qt._params, lsb_tier=tier)
        with pytest.raises(ValueError):
            dataclasses.replace(params, codebook=w4a8_level_table(cb, 5))
        with pytest.raises(ValueError):
            dataclasses.replace(params, lsb_tier=tier[:-64])


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="packed W4A8 MMA decode needs sm_90+",
)
@requires_cuda_backend
class TestCudaBackend:
    @pytest.mark.parametrize("bits", [3, 4])
    def test_canonical_nested_codes_decode_bit_exactly(self, weight, bits):
        q, s, c, _, cb = _nested(weight, bits)
        got = cuda_backend.dequantize_w4a8_int8_weight(q, s, c, codebook=cb, output_dtype=torch.float32)
        ref = eager_w4a8.dequantize_w4a8_int8_weight(q, s, c, codebook=cb, output_dtype=torch.float32)
        assert rel_l2(got, ref) < 1e-5
        x = torch.randn(3, K, device="cuda", dtype=torch.bfloat16)
        got = cuda_backend.w4a8_int8_linear(x, q, s, c, codebook=cb)
        ref = eager_w4a8.w4a8_int8_linear(x, q, s, c, codebook=cb)
        assert rel_l2(got, ref) < 2e-2

    @pytest.mark.parametrize("bits", [4, 5])
    def test_nested_quantize_matches_eager(self, weight, bits):
        got = cuda_backend.quantize_w4a8_int8_weight(weight, bits=bits, codebook_tensor=nested_w4a8_codebook())
        ref = _nested(weight, bits)
        assert torch.equal(got[4], ref[4])
        deq = [eager_w4a8.dequantize_w4a8_int8_weight(*r[:3], codebook=r[4], output_dtype=torch.float32) for r in (got, ref)]
        assert rel_l2(deq[0], deq[1]) < 1e-2

    def test_five_bit_canonical_takes_the_eager_route(self, weight):
        q, s, c, _, cb = _nested(weight, 5)
        x = torch.randn(3, K, device="cuda", dtype=torch.bfloat16)
        assert torch.equal(
            cuda_backend.dequantize_w4a8_int8_weight(q, s, c, codebook=cb, output_dtype=torch.float32),
            eager_w4a8.dequantize_w4a8_int8_weight(q, s, c, codebook=cb, output_dtype=torch.float32),
        )
        assert torch.equal(
            cuda_backend.w4a8_int8_linear(x, q, s, c, codebook=cb),
            eager_w4a8.w4a8_int8_linear(x, q, s, c, codebook=cb),
        )

    # 8 and 16 record splits, M on both sides of the MMA / chunked GEMM switch
    @pytest.mark.parametrize(("n", "k"), [(320, 1280), (4096, 4096)])
    @pytest.mark.parametrize("m", [1, 8, 9, 600])
    @pytest.mark.parametrize("bits", [4, 5])
    def test_packed_tier_matches_eager(self, n, k, m, bits, seed):
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
        qdata, s_rel, s_channel, _, cb = _nested(w, bits)
        stream, tier = pack_w4a8_mma_weight(qdata, s_rel, lsb_tier=True)
        empty = torch.empty(0, device="cuda", dtype=torch.float8_e4m3fn)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, device="cuda", dtype=torch.float32)
        got = ck.w4a8_int8_linear(x, stream, empty, s_channel, codebook=cb, bias=b, mma_packed=True, lsb_tier=tier)
        ref = eager_w4a8.w4a8_int8_linear(x, qdata, s_rel, s_channel, codebook=cb, bias=b)
        assert rel_l2(got, ref) < 2e-2
        # and the stream without its tier is the plain weight one bit narrower
        narrow = _pack_codes(_unpack_codes(qdata, k, bits) >> 1, bits - 1)
        got = ck.w4a8_int8_linear(x, stream, empty, s_channel, codebook=cb, bias=b, mma_packed=True)
        ref = eager_w4a8.w4a8_int8_linear(x, narrow, s_rel, s_channel, codebook=cb, bias=b)
        assert rel_l2(got, ref) < 2e-2

    @pytest.mark.parametrize("bits", [4, 5])
    def test_packed_tier_chunked_gemm_integer_reference(self, bits, seed):
        # Full-range codes and every finite fp8 scale; integer sums exact in fp32 so a
        # misplaced tier bit cannot hide behind tolerances.
        n, k, m, chunk = 272, 1280, 9, 128
        codes = torch.randint(0, 1 << bits, (n, k), device="cuda", dtype=torch.int32)
        qdata = _pack_codes(codes, bits)
        s_rel = (torch.arange(n * k // 16, device="cuda") % 127).to(torch.uint8).view(torch.float8_e4m3fn).view(n, k // 16)
        table = w4a8_level_table(nested_w4a8_codebook(), bits).cuda()
        decoded = (table[codes.long()] * s_rel.float().repeat_interleave(16, dim=1)).round().clamp(-127, 127)
        xq = torch.randint(-3, 4, (m, k), device="cuda", dtype=torch.int8)
        expected = xq.double() @ decoded.double().t()
        stream, tier = pack_w4a8_mma_weight(qdata, s_rel, lsb_tier=True)
        workspace = torch.empty(chunk, k, device="cuda", dtype=torch.int8)
        out = torch.empty(m, n, device="cuda", dtype=torch.float32)
        wrap = cuda_backend._wrap_for_dlpack
        assert cuda_backend._C.w4a8_codebook_gemm_chunked(
            wrap(xq), wrap(stream), wrap(torch.empty(0, device="cuda", dtype=torch.uint8)), wrap(table),
            wrap(torch.ones(n, device="cuda")), wrap(torch.ones(m, device="cuda")), None,
            wrap(workspace), wrap(out), 16, chunk, 0, torch.cuda.current_stream().cuda_stream, True, wrap(tier),
        )
        assert torch.equal(out.double(), expected)
