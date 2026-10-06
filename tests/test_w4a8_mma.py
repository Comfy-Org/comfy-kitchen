# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed W4A8 tensor-core decode (M <= 8, one 8-token MMA beat) against the unpacked
chunked route on the same int8 grid."""

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends import cuda as cuda_backend
from comfy_kitchen.backends.eager import w4a8_int8 as eager_w4a8
from comfy_kitchen.tensor import AsymW4A8Int8Layout, QuantizedTensor
from tests.conftest import requires_cuda_backend

pytestmark = requires_cuda_backend


def _packed_weight(n, k, seed=0):
    torch.manual_seed(seed)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    qdata, s_rel, s_channel, correction, cb = eager_w4a8.quantize_w4a8_int8_weight(w)
    assert correction is None
    rows = ck.w4a8_mma_stream_rows(n, k)
    assert rows
    packed = ck.pack_w4a8_mma_weight(qdata, s_rel, rows)
    return qdata, s_rel, s_channel, cb, packed, rows


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="packed W4A8 MMA decode needs sm_90+ (cluster quantizer)",
)
class TestW4A8PackedMMA:
    # (n, k) covering each stream_rows the geometry rule picks: 8 (one run per warp,
    # including K = 1280 with 40 records = 5 splits of 8), 16 and 32 (multi-run
    # warps, where the run-major layout interleaves splits).
    @pytest.mark.parametrize(("n", "k"), [(4096, 4096), (320, 1280), (17408, 2560), (16384, 4096)])
    @pytest.mark.parametrize("m", [1, 7, 8])
    @pytest.mark.parametrize("bias", [False, True])
    def test_matches_unpacked_route(self, n, k, m, bias, seed):
        qdata, s_rel, s_channel, cb, packed, rows = _packed_weight(n, k)
        assert rows == {(4096, 4096): 8, (320, 1280): 8, (17408, 2560): 16, (16384, 4096): 32}[(n, k)]
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, device="cuda", dtype=torch.float32) if bias else None
        empty = torch.empty(0, device="cuda", dtype=torch.float8_e4m3fn)
        got = cuda_backend.w4a8_int8_linear(x, packed, empty, s_channel, bias=b, stream_rows=rows)
        ref = cuda_backend.w4a8_int8_linear(x, qdata, s_rel, s_channel, codebook=cb, bias=b)
        assert got.shape == (m, n)
        # same int8 grid and int32 accumulation; the only difference is fp32 epilogue order
        torch.testing.assert_close(got.float(), ref.float(), atol=2e-2, rtol=1e-2)

    @pytest.mark.parametrize("m", [1, 7])
    @pytest.mark.parametrize("bias", [False, True])
    def test_fused_residual_matches_addcmul(self, m, bias, seed):
        # The epilogue's residual + scale * linear must be bit-identical to the unfused
        # addcmul on the same bf16-rounded linear; a scale of exactly 1 must reduce
        # to torch.add(residual, linear).
        n, k = 4096, 4096
        _, _, s_channel, _, packed, rows = _packed_weight(n, k)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, device="cuda", dtype=torch.float32) if bias else None
        empty = torch.empty(0, device="cuda", dtype=torch.float8_e4m3fn)
        residual = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        scale = torch.rand(n, device="cuda", dtype=torch.bfloat16) + 0.5
        linear = cuda_backend.w4a8_int8_linear(x, packed, empty, s_channel, bias=b, stream_rows=rows)
        got = cuda_backend.w4a8_int8_linear(x, packed, empty, s_channel, bias=b, stream_rows=rows,
                                            residual=residual, residual_scale=scale)
        assert torch.equal(got, torch.addcmul(residual, linear, scale))
        ones = torch.ones(n, device="cuda", dtype=torch.bfloat16)
        got = cuda_backend.w4a8_int8_linear(x, packed, empty, s_channel, bias=b, stream_rows=rows,
                                            residual=residual, residual_scale=ones)
        assert torch.equal(got, residual + linear)
        # the prefill-shaped (unfused) route applies the same contract
        big = torch.randn(9, k, device="cuda", dtype=torch.bfloat16)
        big_res = torch.randn(9, n, device="cuda", dtype=torch.bfloat16)
        linear = cuda_backend.w4a8_int8_linear(big, packed, empty, s_channel, bias=b, stream_rows=rows)
        got = cuda_backend.w4a8_int8_linear(big, packed, empty, s_channel, bias=b, stream_rows=rows,
                                            residual=big_res, residual_scale=scale)
        assert torch.equal(got, torch.addcmul(big_res, linear, scale))

    @pytest.mark.parametrize("m", [1, 5, 8])
    def test_shared_quantized_input(self, m, seed):
        # One quantizer pass, two linears: each must be bit-identical to the linear that
        # folds the norm into its own quantizer, and the activated row it hands back must
        # be the bf16 rms_norm the fused quantizer rounds from.
        k = 4096
        _, _, s_channel_a, _, packed_a, rows_a = _packed_weight(4096, k, seed=1)
        _, _, s_channel_b, _, packed_b, rows_b = _packed_weight(320 * 4, k, seed=2)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 3.0
        w = torch.randn(k, device="cuda", dtype=torch.bfloat16)
        eps = 1e-6
        empty = torch.empty(0, device="cuda", dtype=torch.float8_e4m3fn)
        xq, xs, xn = cuda_backend.w4a8_quantize_input(x, "rms_norm", w, eps, activated=True)
        assert xq.shape == (m, k) and xs.shape == (m, 1) and xn.shape == (m, k) and xn.dtype == x.dtype
        assert torch.equal(xn, torch.nn.functional.rms_norm(x, (k,), weight=w, eps=eps))
        for packed, s_channel, rows in ((packed_a, s_channel_a, rows_a), (packed_b, s_channel_b, rows_b)):
            fused = cuda_backend.w4a8_int8_linear(x, packed, empty, s_channel, stream_rows=rows,
                                                  input_act="rms_norm", input_act_weight=w, input_act_eps=eps)
            got = cuda_backend.w4a8_int8_linear_prequantized(xq, xs, packed, s_channel, rows)
            assert torch.equal(got, fused)
        # no activation: the plain quantizer's image, and no activated row to hand back
        xq, xs, none = cuda_backend.w4a8_quantize_input(x)
        assert none is None
        plain = cuda_backend.w4a8_int8_linear(x, packed_a, empty, s_channel_a, stream_rows=rows_a)
        assert torch.equal(cuda_backend.w4a8_int8_linear_prequantized(xq, xs, packed_a, s_channel_a, rows_a), plain)
        assert cuda_backend.w4a8_quantize_input(torch.randn(9, k, device="cuda", dtype=torch.bfloat16)) is None

    def test_rows_are_independent(self, seed):
        # Row r of a 7-row call must equal row 0 of a 1-row call on the same input: the MMA
        # beat's padding rows must not leak into the real ones.
        n, k = 4096, 4096
        _, _, s_channel, _, packed, rows = _packed_weight(n, k)
        x = torch.randn(7, k, device="cuda", dtype=torch.bfloat16)
        empty = torch.empty(0, device="cuda", dtype=torch.float8_e4m3fn)
        full = cuda_backend.w4a8_int8_linear(x, packed, empty, s_channel, stream_rows=rows)
        for r in (0, 3, 6):
            single = cuda_backend.w4a8_int8_linear(x[r:r + 1], packed, empty, s_channel, stream_rows=rows)
            assert torch.equal(full[r], single[0]), f"row {r}"

    def test_scratch_returns_to_zero(self, seed):
        # The streaming kernel's split-K workspace and counters must be zero after every
        # call, for all 8 rows, or the next call accumulates onto stale partials.
        n, k = 4096, 4096
        _, _, s_channel, _, packed, rows = _packed_weight(n, k)
        x = torch.randn(8, k, device="cuda", dtype=torch.bfloat16)
        empty = torch.empty(0, device="cuda", dtype=torch.float8_e4m3fn)
        cuda_backend.w4a8_int8_linear(x, packed, empty, s_channel, stream_rows=rows)
        torch.cuda.synchronize()
        workspace, counters = cuda_backend._w4a8_mma_scratch(x.device, n)
        assert workspace.shape[0] == 8
        assert not workspace.any() and not counters.any()

    def test_decode_layout_roundtrip(self, seed):
        # decode_layout packs a resident canonical weight in place of its qdata + s_rel
        # bytes; the layout still computes the same linear for decode and prefill rows,
        # refuses to serialize the packed form, and leaves non-streaming shapes alone.
        n, k = 4096, 4096
        qdata, s_rel, s_channel, cb, _, rows = _packed_weight(n, k)
        params = AsymW4A8Int8Layout.Params(
            scale=s_rel, s_channel=s_channel, codebook=cb,
            orig_dtype=torch.bfloat16, orig_shape=(n, k),
        )
        packed, packed_params = AsymW4A8Int8Layout.decode_layout(qdata, params)
        assert packed.device == qdata.device and packed_params.stream_rows == rows
        assert packed_params.scale.numel() == 0
        assert packed.nbytes == qdata.nbytes + s_rel.nbytes
        with pytest.raises(ValueError):
            AsymW4A8Int8Layout.state_dict_tensors(packed, packed_params)
        saved = AsymW4A8Int8Layout.state_dict_tensors(qdata, params)
        assert saved[""] is qdata and saved["_s_rel"] is s_rel

        weight = QuantizedTensor(packed, "AsymW4A8Int8Layout", packed_params)
        reference = QuantizedTensor(qdata, "AsymW4A8Int8Layout", params)
        for m in (1, 8, 33):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            got = torch.nn.functional.linear(x, weight)
            ref = torch.nn.functional.linear(x, reference)
            torch.testing.assert_close(got.float(), ref.float(), atol=2e-2, rtol=1e-2)
        torch.testing.assert_close(weight.dequantize(), reference.dequantize())

        odd = AsymW4A8Int8Layout.Params(
            scale=s_rel[:4088], s_channel=s_channel[:4088], codebook=cb,
            orig_dtype=torch.bfloat16, orig_shape=(4088, k),
        )
        assert AsymW4A8Int8Layout.decode_layout(qdata[:4088], odd)[1].stream_rows == 0
