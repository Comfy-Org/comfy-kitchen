# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""W3A8: 8-level codebook codes in the AsymW4A8Int8 storage contract as two planes (2-bit
plane then 1-bit plane, row width 3K/8). Eager is the reference; CUDA must decode bit-exactly
and Triton falls back to eager."""

import pytest
import torch

from comfy_kitchen.backends import cuda as cuda_backend
from comfy_kitchen.backends.eager import w4a8_int8 as eager_w4a8
from comfy_kitchen.tensor import AsymW4A8Int8Layout, QuantizedTensor
from comfy_kitchen.tensor.w4a8_packed import default_w4a8_codebook
from tests.conftest import requires_cuda_backend
from tests.test_w6a8 import raw, rel_l2

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture
def weight(seed):
    return torch.randn(384, 1024, device="cuda", dtype=torch.bfloat16) * 0.02


class TestStorage:
    def test_pack_unpack_roundtrip(self, seed):
        codes = torch.randint(0, 8, (5, 128), device="cuda", dtype=torch.int32)
        packed = eager_w4a8._pack_codes(codes, 3)
        assert packed.shape == (5, 48) and packed.dtype == torch.int8
        assert torch.equal(eager_w4a8._unpack_codes(packed, 128, 3), codes)

    def test_row_is_two_bit_plane_then_one_bit_plane(self, seed):
        codes = torch.randint(0, 8, (2, 32), device="cuda", dtype=torch.int32)
        packed = eager_w4a8._pack_codes(codes, 3).to(torch.int32) & 0xFF
        for c in range(32):
            assert torch.equal((packed[:, c // 4] >> (2 * (c % 4))) & 3, codes[:, c] & 3)
            assert torch.equal((packed[:, 8 + c // 8] >> (c % 8)) & 1, codes[:, c] >> 2)

    def test_geometry_infers_bits(self, seed):
        s_rel = torch.ones(384, 64, device="cuda")
        q = torch.zeros(384, 384, device="cuda", dtype=torch.int8)
        assert eager_w4a8._w4a8_geometry(q, s_rel, 16) == (384, 1024, 3)

    def test_requires_symmetric_codebook_and_k_multiple_of_32(self, seed):
        w = torch.randn(64, 256, device="cuda")
        with pytest.raises(ValueError):
            eager_w4a8.quantize_w4a8_int8_weight(torch.randn(64, 272, device="cuda"), convrot_groupsize=16, bits=3)
        with pytest.raises(ValueError):
            eager_w4a8.quantize_w4a8_int8_weight(w, bits=3, symmetric=False)
        with pytest.raises(ValueError):
            eager_w4a8.quantize_w4a8_int8_weight(w, bits=3, codebook=False)
        q = torch.zeros(64, 96, device="cuda", dtype=torch.int8)
        s_rel = torch.ones(64, 16, device="cuda")
        with pytest.raises(ValueError):  # no codebook
            eager_w4a8.validate_w4a8_operands(q, s_rel, torch.ones(64, device="cuda"), None, None, 16, 256)
        with pytest.raises(ValueError):  # 4-bit codebook on 3-bit storage
            eager_w4a8.validate_w4a8_operands(
                q, s_rel, torch.ones(64, device="cuda"), default_w4a8_codebook(4).cuda(), None, 16, 256)


class TestQuantizer:
    def test_three_bit_uses_the_fixed_eight_level_codebook(self, weight):
        q3, s3, c3, corr, cb3 = eager_w4a8.quantize_w4a8_int8_weight(weight, bits=3)
        q4, s4, c4, _, cb4 = eager_w4a8.quantize_w4a8_int8_weight(weight, bits=4)
        assert q3.shape == (384, 384) and corr is None and s3.dtype == torch.float8_e4m3fn
        assert torch.equal(cb3.cpu(), default_w4a8_codebook(3))
        codes = eager_w4a8._unpack_codes(q3, 1024, 3)
        assert codes.min() >= 0 and codes.max() <= 7
        e3 = rel_l2(eager_w4a8.dequantize_w4a8_int8_weight(q3, s3, c3, codebook=cb3, output_dtype=torch.float32), weight)
        e4 = rel_l2(eager_w4a8.dequantize_w4a8_int8_weight(q4, s4, c4, codebook=cb4, output_dtype=torch.float32), weight)
        # one bit fewer: roughly twice the error of the 4-bit codebook, still a usable weight
        assert 1.6 * e4 < e3 < 2.6 * e4, (e3, e4)
        assert e3 < 0.17

    def test_layout_and_state_dict(self, weight):
        qt = QuantizedTensor.from_float(weight, "AsymW4A8Int8Layout", bits=3)
        assert AsymW4A8Int8Layout.bits(qt) == 3 and qt._qdata.shape == (384, 384)
        sd = qt.state_dict("w")
        assert set(sd) == {"w", "w_s_rel", "w_s_channel", "w_codebook"} and sd["w_codebook"].shape == (8,)
        kw = AsymW4A8Int8Layout.requantize_kwargs(qt)
        assert kw["bits"] == 3 and kw["codebook"] is True
        rq = qt.requantize_from_float(weight.float())
        assert rq._qdata.shape == (384, 384) and rel_l2(rq.dequantize(), weight) < 0.17


@requires_cuda_backend
class TestCudaBackend:
    @pytest.mark.parametrize("scale_dtype", [torch.float8_e4m3fn, torch.float32])
    @pytest.mark.parametrize("group_size", [16, 64])
    def test_dequant_is_bit_exact_with_eager(self, weight, scale_dtype, group_size):
        q, s, c, _, cb = eager_w4a8.quantize_w4a8_int8_weight(weight, bits=3, group_size=group_size, scale_dtype=scale_dtype)
        out = cuda_backend._dequant_int4_grouped_to_int8(q, s, cb, group_size)
        assert torch.equal(out, eager_w4a8._dequant_int4_grouped_to_int8(q, s, cb, group_size))
        got = cuda_backend.dequantize_w4a8_int8_weight(q, s, c, codebook=cb, group_size=group_size, output_dtype=torch.float32)
        ref = eager_w4a8.dequantize_w4a8_int8_weight(q, s, c, codebook=cb, group_size=group_size, output_dtype=torch.float32)
        assert rel_l2(got, ref) < 1e-5

    @pytest.mark.parametrize("m", [1, 8, 9, 256, 1024])  # gemv (<=8), chunked GEMM, fast act-quant (>=512)
    @pytest.mark.parametrize("with_bias", [False, True])
    def test_linear_matches_eager(self, weight, m, with_bias):
        q, s, c, _, cb = eager_w4a8.quantize_w4a8_int8_weight(weight, bits=3)
        x = torch.randn(m, 1024, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(384, device="cuda", dtype=torch.bfloat16) if with_bias else None
        got = cuda_backend.w4a8_int8_linear(x, q, s, c, codebook=cb, bias=bias, out_dtype=torch.bfloat16)
        ref = eager_w4a8.w4a8_int8_linear(x, q, s, c, codebook=cb, bias=bias, out_dtype=torch.bfloat16)
        exact = x.float() @ weight.float().t() + (bias.float() if with_bias else 0)
        assert rel_l2(got, ref) < 2e-2          # int8 activation rounding between backends
        assert rel_l2(got, exact) < 0.2         # and both carry the 3-bit weight error

    @pytest.mark.parametrize("m", [64, 1024])
    def test_linear_is_the_same_whole_or_chunked(self, m, seed, monkeypatch):
        w = torch.randn(4352, 512, device="cuda", dtype=torch.bfloat16) * 0.02
        q, s, c, _, cb = eager_w4a8.quantize_w4a8_int8_weight(w, bits=3)
        x = torch.randn(m, 512, device="cuda", dtype=torch.bfloat16)
        outs = []
        for chunk in (1024, 4096, 4352):
            monkeypatch.setattr(cuda_backend, "_w4a8_weight_chunk_cols", lambda m, n, d, c=chunk: c)
            outs.append(cuda_backend.w4a8_int8_linear(x, q, s, c, codebook=cb, out_dtype=torch.bfloat16))
        assert torch.equal(outs[0], outs[2]) and torch.equal(outs[1], outs[2])

    def test_quantize_takes_the_eager_route(self, weight):
        # No fused 3-bit requant kernel: the CUDA quantizer must return the eager result.
        got = cuda_backend.quantize_w4a8_int8_weight(weight, bits=3)
        ref = eager_w4a8.quantize_w4a8_int8_weight(weight, bits=3)
        assert torch.equal(got[0], ref[0]) and torch.equal(raw(got[1]), raw(ref[1]))
        assert torch.equal(got[2], ref[2]) and torch.equal(got[4], ref[4])


class TestTritonBackend:
    def test_dequant_matches_eager(self, weight):
        triton_mod = pytest.importorskip("comfy_kitchen.backends.triton.w4a8_int8")
        q, s, _c, _, cb = eager_w4a8.quantize_w4a8_int8_weight(weight, bits=3)
        got = triton_mod._dequant_int4_grouped_to_int8(q, s, cb, 16)
        assert torch.equal(got, eager_w4a8._dequant_int4_grouped_to_int8(q, s, cb, 16))
