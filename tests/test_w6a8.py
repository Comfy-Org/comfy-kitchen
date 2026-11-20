# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The multi-plane widths of the AsymW4A8Int8 storage contract: W6A8 (nibble plane + 2-bit high
plane, row width 3K/4), W5A8 (nibble plane + 1-bit plane, 5K/8) and W3A8 (2-bit plane + 1-bit
plane, 3K/8, 8-entry codebook). Eager is the reference; CUDA and Triton must decode bit-exactly."""

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends import cuda as cuda_backend
from comfy_kitchen.backends.eager import w4a8_int8 as eager_w4a8
from comfy_kitchen.tensor import AsymW4A8Int8Layout, QuantizedTensor
from tests.conftest import requires_cuda_backend

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# relL2 of the dequantized weight vs the bf16 original on the Gaussian fixture, per width
_QUALITY_BOUND = {3: 0.2, 4: 0.1, 5: 0.055, 6: 0.035}


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
        for bits in (3, 4, 5, 6):
            codes = torch.randint(0, 1 << bits, (5, 128), device="cuda", dtype=torch.int32)
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

    def test_three_and_five_bit_planes(self):
        """3 bits: 2-bit plane then 1-bit plane; 5 bits: nibble plane then 1-bit plane."""
        codes = torch.zeros(1, 64, device="cuda", dtype=torch.int32)
        codes[0, 5] = 0b101  # col 5: low 2 bits 01, top bit 1
        p3 = eager_w4a8._pack_codes(codes, 3).view(torch.uint8)
        assert p3.shape[1] == 24 and p3[0, 5 // 4].item() == (0b01 << (2 * (5 % 4))) and p3[0, 16 + 5 // 8].item() == (1 << (5 % 8))
        codes[0, 5] = 0b10111  # nibble 0111, top bit 1
        p5 = eager_w4a8._pack_codes(codes, 5).view(torch.uint8)
        assert p5.shape[1] == 40 and p5[0, 5 // 2].item() == (0b0111 << 4) and p5[0, 32 + 5 // 8].item() == (1 << (5 % 8))

    def test_geometry_infers_bits_and_rejects_bad_widths(self, weight):
        s_rel = torch.ones(384, 64, device="cuda")  # g16 for K=1024
        for bits in (3, 4, 5, 6):
            q = torch.zeros(384, 1024 * bits // 8, device="cuda", dtype=torch.int8)
            assert eager_w4a8._w4a8_geometry(q, s_rel, 16) == (384, 1024, bits)
        with pytest.raises(ValueError):  # 7-bit width is not a supported storage
            eager_w4a8._w4a8_geometry(torch.zeros(384, 896, device="cuda", dtype=torch.int8), s_rel, 16)
        with pytest.raises(ValueError):  # 3/5-bit planes need K % 64 and groups of >= 16
            eager_w4a8._w4a8_geometry(torch.zeros(4, 96 * 3 // 8, device="cuda", dtype=torch.int8), torch.ones(4, 6, device="cuda"), 16)
        with pytest.raises(ValueError):
            eager_w4a8._w4a8_geometry(torch.zeros(4, 1024 * 5 // 8, device="cuda", dtype=torch.int8), torch.ones(4, 128, device="cuda"), 8)

    # K % 32 at 6 bits, K % 64 at 3/5 (288 passes the first rule, not the second)
    @pytest.mark.parametrize("bits, bad_k", [(3, 288), (5, 288), (6, 272)])
    def test_plane_widths_reject_misaligned_k_asymmetric_and_wrong_codebook(self, seed, bits, bad_k):
        with pytest.raises(ValueError):
            eager_w4a8.quantize_w4a8_int8_weight(
                torch.randn(64, bad_k, device="cuda"), group_size=16, convrot_groupsize=16, bits=bits
            )
        with pytest.raises(ValueError):
            eager_w4a8.quantize_w4a8_int8_weight(torch.randn(64, 256, device="cuda"), bits=bits, symmetric=False)
        q = torch.zeros(64, 256 * bits // 8, device="cuda", dtype=torch.int8)
        s_rel = torch.ones(64, 16, device="cuda")
        wrong_codebook = None if bits == 3 else torch.zeros(16, device="cuda")  # 3 needs one, 5/6 have none
        with pytest.raises(ValueError):
            eager_w4a8.validate_w4a8_operands(q, s_rel, torch.ones(64, device="cuda"), wrong_codebook, None, 16, 256)


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

    def test_widths_order_by_error_and_shapes(self, weight):
        errs = {}
        for bits in (3, 4, 5, 6):
            q, s, c, corr, cb = eager_w4a8.quantize_w4a8_int8_weight(weight, bits=bits)
            assert q.shape == (384, 1024 * bits // 8) and corr is None
            assert (cb is not None and cb.shape == (8,)) if bits == 3 else (cb is None if bits in (5, 6) else cb.shape == (16,))
            errs[bits] = rel_l2(eager_w4a8.dequantize_w4a8_int8_weight(q, s, c, codebook=cb, output_dtype=torch.float32), weight)
            assert errs[bits] < _QUALITY_BOUND[bits], errs
        assert errs[3] > errs[4] > errs[5] > errs[6]

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

    @pytest.mark.parametrize("bits", [3, 5, 6])
    @pytest.mark.parametrize("group_size", [16, 32, 64])
    def test_group_sizes(self, weight, bits, group_size):
        q, s, c, _, cb = eager_w4a8.quantize_w4a8_int8_weight(weight, bits=bits, group_size=group_size)
        assert s.shape == (384, 1024 // group_size)
        e = rel_l2(eager_w4a8.dequantize_w4a8_int8_weight(q, s, c, codebook=cb, group_size=group_size, output_dtype=torch.float32), weight)
        assert e < _QUALITY_BOUND[bits]

    @pytest.mark.parametrize("bits", [3, 4, 5, 6])
    def test_chunked_rows_match_single_shot(self, bits, seed, monkeypatch):
        w = torch.randn(96, 512, device="cuda", dtype=torch.bfloat16) * 0.02
        ref = eager_w4a8.quantize_w4a8_int8_weight(w, bits=bits)
        monkeypatch.setattr(eager_w4a8, "_QUANT_ROW_ELEM_BUDGET", 512 * 20)
        got = eager_w4a8.quantize_w4a8_int8_weight(w, bits=bits)
        for a, b in zip(ref[:3], got[:3], strict=True):
            assert torch.equal(raw(a), raw(b))


@requires_cuda_backend
class TestCudaBackend:
    @pytest.mark.parametrize("bits", [3, 5, 6])
    @pytest.mark.parametrize("scale_dtype", [torch.float8_e4m3fn, torch.float32])
    @pytest.mark.parametrize("group_size", [16, 64])
    def test_dequant_is_bit_exact_with_eager(self, weight, bits, scale_dtype, group_size):
        q, s, c, _, cb = eager_w4a8.quantize_w4a8_int8_weight(weight, bits=bits, group_size=group_size, scale_dtype=scale_dtype)
        # the int8 grid the GEMM consumes must match eager exactly
        out = cuda_backend._dequant_int4_grouped_to_int8(q, s, cb, group_size)
        assert torch.equal(out, eager_w4a8._dequant_int4_grouped_to_int8(q, s, cb, group_size))
        # and the full dequantize differs only by the fp un-rotation
        got = cuda_backend.dequantize_w4a8_int8_weight(q, s, c, codebook=cb, group_size=group_size, output_dtype=torch.float32)
        ref = eager_w4a8.dequantize_w4a8_int8_weight(q, s, c, codebook=cb, group_size=group_size, output_dtype=torch.float32)
        assert rel_l2(got, ref) < 1e-5

    @pytest.mark.parametrize("bits", [3, 5, 6])
    @pytest.mark.parametrize("m", [1, 8, 9, 256, 1024])  # gemv (<=8), chunked GEMM, fast act-quant (>=512)
    @pytest.mark.parametrize("with_bias", [False, True])
    def test_linear_matches_eager(self, weight, bits, m, with_bias):
        q, s, c, _, cb = eager_w4a8.quantize_w4a8_int8_weight(weight, bits=bits)
        x = torch.randn(m, 1024, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(384, device="cuda", dtype=torch.bfloat16) if with_bias else None
        got = cuda_backend.w4a8_int8_linear(x, q, s, c, codebook=cb, bias=bias, out_dtype=torch.bfloat16)
        ref = eager_w4a8.w4a8_int8_linear(x, q, s, c, codebook=cb, bias=bias, out_dtype=torch.bfloat16)
        exact = x.float() @ weight.float().t() + (bias.float() if with_bias else 0)
        assert rel_l2(got, ref) < 2e-2                          # int8 activation rounding between backends
        assert rel_l2(got, exact) < _QUALITY_BOUND[bits] + 0.01  # and both are close to the bf16 matmul

    @pytest.mark.parametrize("bits", [3, 4, 5, 6])
    def test_gemv_is_bit_exact_with_chunked(self, weight, bits):
        # M<=8 runs the in-register GEMV; rows are quantized independently, so a row's
        # output must equal the same row computed by the chunked path at M=16
        q, s, c, _, cb = eager_w4a8.quantize_w4a8_int8_weight(weight, bits=bits)
        x = torch.randn(16, 1024, device="cuda", dtype=torch.bfloat16)
        full = ck.w4a8_int8_linear(x, q, s, c, codebook=cb)
        for m in (1, 8):
            assert torch.equal(ck.w4a8_int8_linear(x[:m].contiguous(), q, s, c, codebook=cb), full[:m])

    # a small L2 decodes the weight whole, a large one in chunks; 4352 columns leave a short last chunk
    @pytest.mark.parametrize("bits", [3, 4, 5, 6])
    @pytest.mark.parametrize("m", [64, 1024])
    def test_linear_is_the_same_whole_or_chunked(self, bits, m, seed, monkeypatch):
        w = torch.randn(4352, 512, device="cuda", dtype=torch.bfloat16) * 0.02
        q, s, c, _, cb = eager_w4a8.quantize_w4a8_int8_weight(w, bits=bits)
        x = torch.randn(m, 512, device="cuda", dtype=torch.bfloat16)
        outs = []
        for chunk in (1024, 4096, 4352):
            monkeypatch.setattr(cuda_backend, "_w4a8_weight_chunk_cols", lambda m, n, d, c=chunk: c)
            outs.append(cuda_backend.w4a8_int8_linear(x, q, s, c, codebook=cb, out_dtype=torch.bfloat16))
        assert torch.equal(outs[0], outs[2]) and torch.equal(outs[1], outs[2])

    def test_two_pass_fp32_scale_route(self, weight):
        q, s, c, _, _ = eager_w4a8.quantize_w4a8_int8_weight(weight, bits=6, scale_dtype=torch.float32)
        x = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16)
        got = cuda_backend.w4a8_int8_linear(x, q, s, c, out_dtype=torch.bfloat16)
        ref = eager_w4a8.w4a8_int8_linear(x, q, s, c, out_dtype=torch.bfloat16)
        assert rel_l2(got, ref) < 2e-2

    @pytest.mark.parametrize("group_size", [16, 32, 64])
    def test_fused_requant_matches_eager(self, weight, group_size, monkeypatch):
        """The fused ConvRot+requant kernel reproduces the eager quantizer (search off): same
        scales and int8 grid up to fp32 rounding-order effects, nothing systematic."""
        if not cuda_backend._WXA8_FUSED_QUANT:
            pytest.skip("fused requant not built")
        kw = {"bits": 6, "group_size": group_size, "codebook": False, "scale_search": False}
        q, s, c, _, _ = eager_w4a8.quantize_w4a8_int8_weight(weight, **kw)
        monkeypatch.setattr(cuda_backend, "_quantize_w4a8_chunked", lambda *a, **k: pytest.fail("eager path used"))
        qf, sf, cf, corr, cb = cuda_backend.quantize_w4a8_int8_weight(weight, **kw)
        assert corr is None and cb is None and qf.shape == q.shape and sf.dtype == s.dtype
        assert torch.allclose(cf, c, rtol=1e-5, atol=0)
        sb, sfb = raw(s).int(), raw(sf).int()
        assert (sb != sfb).float().mean().item() < 1e-3 and (sb - sfb).abs().max().item() <= 1
        grid = eager_w4a8._dequant_int4_grouped_to_int8(q, s, None, group_size)
        grid_f = eager_w4a8._dequant_int4_grouped_to_int8(qf, sf, None, group_size)
        assert (grid_f != grid).float().mean().item() < 1e-3
        e = rel_l2(eager_w4a8.dequantize_w4a8_int8_weight(q, s, c, group_size=group_size, output_dtype=torch.float32), weight)
        ef = rel_l2(eager_w4a8.dequantize_w4a8_int8_weight(qf, sf, cf, group_size=group_size, output_dtype=torch.float32), weight)
        assert abs(ef - e) < 1e-4 * e

    def test_fused_requant_stochastic_rounding(self, weight):
        if not cuda_backend._WXA8_FUSED_QUANT:
            pytest.skip("fused requant not built")
        kw = {"bits": 6, "group_size": 32, "codebook": False, "scale_search": False, "stochastic_rounding": 7}
        q, s, c, _, _ = cuda_backend.quantize_w4a8_int8_weight(weight, **kw)
        q2, _, _, _, _ = cuda_backend.quantize_w4a8_int8_weight(weight, **kw)
        assert torch.equal(q, q2)  # seeded, deterministic
        e = rel_l2(cuda_backend.dequantize_w4a8_int8_weight(q, s, c, group_size=32, output_dtype=torch.float32), weight)
        assert e < 0.035

    def test_requantize_uses_fused_kernel(self, weight, monkeypatch):
        if not cuda_backend._WXA8_FUSED_QUANT:
            pytest.skip("fused requant not built")
        qt = QuantizedTensor.from_float(weight, "AsymW4A8Int8Layout", bits=6, group_size=32)
        monkeypatch.setattr(cuda_backend, "_quantize_w4a8_chunked", lambda *a, **k: pytest.fail("eager path used"))
        rq = qt.requantize_from_float(weight)   # ComfyUI hands over compute-dtype (bf16/fp16) weights
        assert rel_l2(rq.dequantize(), weight) < 0.03

    def test_declined_shape_falls_back_to_eager(self, weight, monkeypatch):
        """When the fused launcher declines (a row too long for shared memory, which depends on the
        GPU), the chunked eager quantizer runs and its result is returned unchanged."""
        if not cuda_backend._WXA8_FUSED_QUANT:
            pytest.skip("fused requant not built")
        monkeypatch.setattr(cuda_backend._C, "quantize_wxa8_convrot_fused", lambda *a, **k: False)
        calls = []
        chunked = cuda_backend._quantize_w4a8_chunked
        monkeypatch.setattr(cuda_backend, "_quantize_w4a8_chunked", lambda *a, **k: calls.append(1) or chunked(*a, **k))
        kw = {"bits": 6, "group_size": 32, "codebook": False, "scale_search": False}
        q, s, c, _, _ = cuda_backend.quantize_w4a8_int8_weight(weight, **kw)
        qe, se, ce, _, _ = eager_w4a8.quantize_w4a8_int8_weight(weight, **kw)
        assert calls and torch.equal(q, qe) and torch.equal(raw(s), raw(se)) and torch.equal(c, ce)

    def test_four_bit_path_unchanged(self, weight):
        q, s, c, _, cb = eager_w4a8.quantize_w4a8_int8_weight(weight, bits=4)
        got = cuda_backend.dequantize_w4a8_int8_weight(q, s, c, codebook=cb, output_dtype=torch.float32)
        ref = eager_w4a8.dequantize_w4a8_int8_weight(q, s, c, codebook=cb, output_dtype=torch.float32)
        assert rel_l2(got, ref) < 1e-5


class TestTritonBackend:
    def test_dequant_is_bit_exact_with_eager(self, weight):
        triton_mod = pytest.importorskip("comfy_kitchen.backends.triton.w4a8_int8")
        for bits in (3, 4, 5, 6):
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

    @pytest.mark.parametrize("bits", [3, 5])
    def test_three_and_five_bit_layout_roundtrip_and_dispatch(self, weight, bits):
        qt = QuantizedTensor.from_float(weight, "AsymW4A8Int8Layout", bits=bits)
        assert AsymW4A8Int8Layout.bits(qt) == bits and qt._qdata.shape == (384, 1024 * bits // 8)
        assert ("w_codebook" in qt.state_dict("w")) == (bits == 3)
        kw = AsymW4A8Int8Layout.requantize_kwargs(qt)
        assert kw["bits"] == bits and kw["scale_search"] is False
        assert rel_l2(qt.dequantize(), weight) < _QUALITY_BOUND[bits]
        x = torch.randn(300, 1024, device="cuda", dtype=torch.bfloat16)
        assert rel_l2(qt.linear_input_act(x, None), torch.nn.functional.linear(x, weight)) < _QUALITY_BOUND[bits] + 0.01
