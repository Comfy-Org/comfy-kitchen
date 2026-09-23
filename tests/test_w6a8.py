# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""W6A8: uniform 6-bit codes in the AsymW4A8Int8 storage contract (nibble plane + 2-bit high
plane, row width 3K/4). Eager is the reference; CUDA and Triton must decode bit-exactly."""

import pytest
import torch

from comfy_kitchen.backends import cuda as cuda_backend
from comfy_kitchen.backends.eager import w4a8_int8 as eager_w4a8
from comfy_kitchen.tensor import AsymW4A8Int8Layout, QuantizedTensor
from tests.conftest import requires_cuda_backend

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def rel_l2(got, ref):
    """Relative L2 error: the quantization-quality metric (conftest's rel_err is max-based)."""
    got, ref = got.float(), ref.float()
    return ((got - ref).norm() / ref.norm().clamp(min=1e-9)).item()


def raw(t):
    """Bit view for exact comparison of fp8 tensors."""
    return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t


@pytest.fixture
def weight(seed):
    return torch.randn(384, 1024, device="cuda", dtype=torch.bfloat16) * 0.02


class TestStorage:
    def test_pack_unpack_roundtrip(self, seed):
        for bits, hi in ((4, 16), (6, 64)):
            codes = torch.randint(0, hi, (5, 128), device="cuda", dtype=torch.int32)
            packed = eager_w4a8._pack_codes(codes, bits)
            assert packed.shape == (5, 128 * bits // 8) and packed.dtype == torch.int8
            assert torch.equal(eager_w4a8._unpack_codes(packed, 128, bits), codes)

    def test_six_bit_row_is_nibble_plane_then_high_plane(self, seed):
        codes = torch.randint(0, 64, (2, 32), device="cuda", dtype=torch.int32)
        packed = eager_w4a8._pack_codes(codes, 6)
        # the first K/2 bytes are exactly the 4-bit packing of the low nibbles
        assert torch.equal(packed[:, :16], eager_w4a8._pack_codes(codes & 0xF, 4))
        hi = (packed[:, 16:].to(torch.int32) & 0xFF)
        for c in range(32):
            assert torch.equal((hi[:, c // 4] >> (2 * (c % 4))) & 3, codes[:, c] >> 4)

    def test_geometry_infers_bits_and_rejects_bad_widths(self, weight):
        s_rel = torch.ones(384, 64, device="cuda")  # g16 for K=1024
        for bits in (4, 6):
            q = torch.zeros(384, 1024 * bits // 8, device="cuda", dtype=torch.int8)
            assert eager_w4a8._w4a8_geometry(q, s_rel, 16) == (384, 1024, bits)
        with pytest.raises(ValueError):
            eager_w4a8._w4a8_geometry(torch.zeros(384, 640, device="cuda", dtype=torch.int8), s_rel, 16)

    def test_six_bit_rejects_codebook_correction_and_k_not_multiple_of_32(self, seed):
        with pytest.raises(ValueError):
            eager_w4a8.quantize_w4a8_int8_weight(
                torch.randn(64, 272, device="cuda"), group_size=16, convrot_groupsize=16, bits=6
            )
        with pytest.raises(ValueError):
            eager_w4a8.quantize_w4a8_int8_weight(torch.randn(64, 256, device="cuda"), bits=6, symmetric=False)
        q = torch.zeros(64, 192, device="cuda", dtype=torch.int8)
        s_rel = torch.ones(64, 16, device="cuda")
        with pytest.raises(ValueError):
            eager_w4a8.validate_w4a8_operands(q, s_rel, torch.ones(64, device="cuda"),
                                              torch.zeros(16, device="cuda"), None, 16, 256)


class TestQuantizer:
    def test_six_bit_is_uniform_symmetric_and_much_cleaner_than_four(self, weight):
        q4, s4, c4, _corr4, cb4 = eager_w4a8.quantize_w4a8_int8_weight(weight, bits=4)
        q6, s6, c6, corr6, cb6 = eager_w4a8.quantize_w4a8_int8_weight(weight, bits=6)
        assert q4.shape == (384, 512) and q6.shape == (384, 768)
        assert cb6 is None and corr6 is None and s6.dtype == torch.float8_e4m3fn
        codes = eager_w4a8._unpack_codes(q6, 1024, 6)
        assert codes.min() >= 1 and codes.max() <= 63  # q + 32 for q in -31..31
        e4 = rel_l2(eager_w4a8.dequantize_w4a8_int8_weight(q4, s4, c4, codebook=cb4, output_dtype=torch.float32), weight)
        e6 = rel_l2(eager_w4a8.dequantize_w4a8_int8_weight(q6, s6, c6, output_dtype=torch.float32), weight)
        assert e6 < e4 / 2.5, (e4, e6)
        assert e6 < 0.03

    def test_scale_search_improves_quality_and_requantize_skips_it(self, weight):
        q, s, c, _, _ = eager_w4a8.quantize_w4a8_int8_weight(weight, bits=6, scale_search=True)
        qn, sn, cn, _, _ = eager_w4a8.quantize_w4a8_int8_weight(weight, bits=6, scale_search=False)
        e = rel_l2(eager_w4a8.dequantize_w4a8_int8_weight(q, s, c, output_dtype=torch.float32), weight)
        en = rel_l2(eager_w4a8.dequantize_w4a8_int8_weight(qn, sn, cn, output_dtype=torch.float32), weight)
        assert e < 0.95 * en, (e, en)
        qt = QuantizedTensor.from_float(weight, "AsymW4A8Int8Layout", bits=6)
        assert AsymW4A8Int8Layout.bits(qt) == 6
        kw = AsymW4A8Int8Layout.requantize_kwargs(qt)
        assert kw["bits"] == 6 and kw["scale_search"] is False and kw["codebook"] is False

    @pytest.mark.parametrize("group_size", [16, 32, 64])
    def test_group_sizes(self, weight, group_size):
        q, s, c, _, _ = eager_w4a8.quantize_w4a8_int8_weight(weight, bits=6, group_size=group_size)
        assert s.shape == (384, 1024 // group_size)
        e = rel_l2(eager_w4a8.dequantize_w4a8_int8_weight(q, s, c, group_size=group_size, output_dtype=torch.float32), weight)
        assert e < 0.035

    def test_chunked_rows_match_single_shot(self, seed, monkeypatch):
        w = torch.randn(96, 512, device="cuda", dtype=torch.bfloat16) * 0.02
        ref = eager_w4a8.quantize_w4a8_int8_weight(w, bits=6)
        monkeypatch.setattr(eager_w4a8, "_QUANT_ROW_ELEM_BUDGET", 512 * 20)
        got = eager_w4a8.quantize_w4a8_int8_weight(w, bits=6)
        for a, b in zip(ref[:3], got[:3], strict=True):
            assert torch.equal(raw(a), raw(b))


@requires_cuda_backend
class TestCudaBackend:
    @pytest.mark.parametrize("scale_dtype", [torch.float8_e4m3fn, torch.float32])
    @pytest.mark.parametrize("group_size", [16, 64])
    def test_dequant_is_bit_exact_with_eager(self, weight, scale_dtype, group_size):
        q, s, c, _, _ = eager_w4a8.quantize_w4a8_int8_weight(weight, bits=6, group_size=group_size, scale_dtype=scale_dtype)
        # the int8 grid the GEMM consumes must match eager exactly
        out = cuda_backend._dequant_int4_grouped_to_int8(q, s, None, group_size)
        assert torch.equal(out, eager_w4a8._dequant_int4_grouped_to_int8(q, s, None, group_size))
        # and the full dequantize differs only by the fp un-rotation
        got = cuda_backend.dequantize_w4a8_int8_weight(q, s, c, group_size=group_size, output_dtype=torch.float32)
        ref = eager_w4a8.dequantize_w4a8_int8_weight(q, s, c, group_size=group_size, output_dtype=torch.float32)
        assert rel_l2(got, ref) < 1e-5

    @pytest.mark.parametrize("m", [1, 8, 9, 256, 1024])  # gemv (<=8), chunked GEMM, fast act-quant (>=512)
    @pytest.mark.parametrize("with_bias", [False, True])
    def test_linear_matches_eager(self, weight, m, with_bias):
        q, s, c, _, _ = eager_w4a8.quantize_w4a8_int8_weight(weight, bits=6)
        x = torch.randn(m, 1024, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(384, device="cuda", dtype=torch.bfloat16) if with_bias else None
        got = cuda_backend.w4a8_int8_linear(x, q, s, c, bias=bias, out_dtype=torch.bfloat16)
        ref = eager_w4a8.w4a8_int8_linear(x, q, s, c, bias=bias, out_dtype=torch.bfloat16)
        exact = x.float() @ weight.float().t() + (bias.float() if with_bias else 0)
        assert rel_l2(got, ref) < 2e-2          # int8 activation rounding between backends
        assert rel_l2(got, exact) < 4e-2        # and both are close to the bf16 matmul

    def test_two_pass_fp32_scale_route(self, weight):
        q, s, c, _, _ = eager_w4a8.quantize_w4a8_int8_weight(weight, bits=6, scale_dtype=torch.float32)
        x = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16)
        got = cuda_backend.w4a8_int8_linear(x, q, s, c, out_dtype=torch.bfloat16)
        ref = eager_w4a8.w4a8_int8_linear(x, q, s, c, out_dtype=torch.bfloat16)
        assert rel_l2(got, ref) < 2e-2

    def test_four_bit_path_unchanged(self, weight):
        q, s, c, _, cb = eager_w4a8.quantize_w4a8_int8_weight(weight, bits=4)
        got = cuda_backend.dequantize_w4a8_int8_weight(q, s, c, codebook=cb, output_dtype=torch.float32)
        ref = eager_w4a8.dequantize_w4a8_int8_weight(q, s, c, codebook=cb, output_dtype=torch.float32)
        assert rel_l2(got, ref) < 1e-5


class TestTritonBackend:
    def test_dequant_is_bit_exact_with_eager(self, weight):
        triton_mod = pytest.importorskip("comfy_kitchen.backends.triton.w4a8_int8")
        for bits in (4, 6):
            q, s, _c, _, cb = eager_w4a8.quantize_w4a8_int8_weight(weight, bits=bits)
            got = triton_mod._dequant_int4_grouped_to_int8(q, s, cb, 16)
            ref = eager_w4a8._dequant_int4_grouped_to_int8(q, s, cb, 16)
            assert torch.equal(got, ref), bits


class TestLayout:
    def test_quantized_tensor_linear_and_state_dict(self, weight):
        qt = QuantizedTensor.from_float(weight, "AsymW4A8Int8Layout", bits=6)
        assert qt._qdata.shape == (384, 768)
        sd = qt.state_dict("w")
        assert set(sd) == {"w", "w_s_rel", "w_s_channel"}  # no codebook key at 6 bits
        x = torch.randn(32, 1024, device="cuda", dtype=torch.bfloat16)
        y = torch.nn.functional.linear(x, qt)
        assert rel_l2(y, x.float() @ weight.float().t()) < 4e-2
        assert rel_l2(qt.dequantize(), weight) < 0.03
        # requantize keeps the width and stays close to the fresh quantization
        rq = qt.requantize_from_float(weight.float(), stochastic_rounding=3)
        assert rq._qdata.shape == (384, 768) and rel_l2(rq.dequantize(), weight) < 0.035
