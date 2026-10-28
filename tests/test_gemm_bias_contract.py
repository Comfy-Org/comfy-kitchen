# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fused CUTLASS GEMMs read bias (and the residual scale) in the OUTPUT
dtype. Every route that lands on them -- int8, int4 via int8, W4A8 chunked and
2-pass -- must hand them that, and must never pass a pointer the epilogue's
16-byte vector loads cannot dereference."""

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends import cuda as cuda_backend
from comfy_kitchen.backends.eager import w4a8_int8 as eager_w4a8
from tests.conftest import rel_err, requires_cuda_backend

pytestmark = requires_cuda_backend


def _offset_view(t: torch.Tensor) -> torch.Tensor:
    """`t`'s values in a contiguous buffer whose data pointer is 8 mod 16."""
    base = torch.empty(t.numel() + 4, dtype=t.dtype, device=t.device)
    v = base[4:4 + t.numel()].view(t.shape)
    v.copy_(t)
    assert v.data_ptr() % 16 == 8 and v.is_contiguous()
    return v


class TestW4A8Bias:
    @pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("route", ["chunked", "two_pass"])
    def test_bias_is_added_in_the_output_dtype(self, seed, out_dtype, route):
        n, k, m = 512, 512, 256
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
        # an asymmetric quant carries a correction, which forces the 2-pass route
        qdata, s_rel, s_channel, correction, cb = eager_w4a8.quantize_w4a8_int8_weight(
            w, symmetric=(route == "chunked"), codebook=(route == "chunked"))
        if route == "chunked":  # the preconditions of the chunked branch, so both routes really run
            assert cuda_backend._W4A8_CHUNKED and correction is None and s_rel.dtype == torch.float8_e4m3fn
            assert hasattr(cuda_backend._C, "w4a8_codebook_linear_chunked")
        else:
            assert correction is not None
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(n, device="cuda", dtype=torch.float32)  # fp32, as a module may keep it
        kwargs = {"codebook": cb, "correction": correction, "out_dtype": out_dtype}

        plain = cuda_backend.w4a8_int8_linear(x, qdata, s_rel, s_channel, **kwargs)
        got = cuda_backend.w4a8_int8_linear(x, qdata, s_rel, s_channel, bias=bias, **kwargs)
        # (the eager reference wants an fp32 input when a correction is present)
        ref = eager_w4a8.w4a8_int8_linear(x.float(), qdata, s_rel, s_channel, bias=bias, **kwargs)

        assert got.dtype == out_dtype
        # the bias add is the only difference from `plain`; rounding-level
        assert rel_err(got, plain.float() + bias) < 1e-2
        # and the whole thing agrees with eager at int8 quantizer tolerance
        scale = ref.float().abs().max().item()
        assert (got.float() - ref.float()).abs().max().item() < 0.05 * scale


class TestInt4ViaInt8Bias:
    def test_chunked_strided_gemm_reads_bias_in_the_output_dtype(self, seed):
        """M >= 1024, N >= 4096, per-channel scale: the chunked int4-weight GEMM
        hands each column chunk to the strided CUTLASS int8 GEMM."""
        m, k, n = 1024, 256, 4096
        x_int8 = torch.randint(-127, 127, (m, k), dtype=torch.int8, device="cuda")
        x_scale = torch.rand(m, device="cuda") * 0.01 + 0.005
        w_qdata, w_scale = cuda_backend.quantize_int4_rowwise(
            torch.randn(n, k, device="cuda", dtype=torch.float16))
        bias = torch.randn(n, device="cuda", dtype=torch.float32)
        for out_dtype in (torch.float16, torch.bfloat16):
            plain = cuda_backend._int4_weight_int8_act_gemm_dequant_chunked(
                x_int8, w_qdata, x_scale, w_scale, None, out_dtype)
            got = cuda_backend._int4_weight_int8_act_gemm_dequant_chunked(
                x_int8, w_qdata, x_scale, w_scale, bias, out_dtype)
            assert rel_err(got, plain.float() + bias) < 1e-2

    @pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
    def test_int4_linear_accepts_a_model_dtype_bias(self, seed, out_dtype):
        m, k, n = 1024, 256, 4096
        x = torch.randn(m, k, device="cuda", dtype=out_dtype)
        w = torch.randn(n, k, device="cuda", dtype=out_dtype)
        x_q, x_s = cuda_backend.quantize_int4_rowwise(x)
        w_q, w_s = cuda_backend.quantize_int4_rowwise(w)
        bias = torch.randn(n, device="cuda", dtype=out_dtype)
        plain = cuda_backend.int4_linear(x_q, w_q, x_s, w_s, None, out_dtype)
        got = cuda_backend.int4_linear(x_q, w_q, x_s, w_s, bias, out_dtype)
        assert rel_err(got, plain.float() + bias.float()) < 1e-2


class TestVectorAlignment:
    def test_int8_linear_realigns_bias_and_residual_scale(self, seed):
        m, k, n = 512, 2048, 512
        x = torch.randn(m, k, dtype=torch.float16, device="cuda")
        weight = torch.randint(-127, 127, (n, k), dtype=torch.int8, device="cuda")
        wscale = torch.tensor(0.01, dtype=torch.float32, device="cuda")
        bias = torch.randn(n, dtype=torch.float16, device="cuda")
        resid = torch.randn(m, n, dtype=torch.float16, device="cuda")
        rscale = torch.randn(n, dtype=torch.float16, device="cuda")

        ref = ck.int8_linear(x, weight, wscale, bias, torch.float16,
                             residual=resid, residual_scale=rscale)
        got = ck.int8_linear(x, weight, wscale, _offset_view(bias), torch.float16,
                             residual=resid, residual_scale=_offset_view(rscale))
        torch.cuda.synchronize()
        assert torch.equal(got, ref)

    def test_fp16_linear_realigns_bias_and_residual_scale(self, seed):
        m, k, n = 512, 2048, 512
        x = torch.randn(m, k, dtype=torch.float16, device="cuda")
        w = torch.randn(n, k, dtype=torch.float16, device="cuda") * 0.02
        bias = torch.randn(n, dtype=torch.float16, device="cuda")
        resid = torch.randn(m, n, dtype=torch.float16, device="cuda")
        rscale = torch.randn(n, dtype=torch.float16, device="cuda")

        ref = ck.fp16_linear(x, w, bias, resid, rscale)
        got = ck.fp16_linear(x, w, _offset_view(bias), resid, _offset_view(rscale))
        torch.cuda.synchronize()
        assert torch.equal(got, ref)
