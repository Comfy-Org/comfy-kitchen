# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed W4A8 tensor-core decode (M <= 8, one 8-token MMA beat) against the unpacked
chunked route on the same int8 grid."""

import dataclasses

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends import cuda as cuda_backend
from comfy_kitchen.backends.eager import w4a8_int8 as eager_w4a8
from comfy_kitchen.tensor import AsymW4A8Int8Layout, QuantizedTensor
from comfy_kitchen.tensor.w4a8_packed import unpack_w4a8_mma_weight
from tests.conftest import requires_cuda_backend

pytestmark = requires_cuda_backend


def _packed_weight(n, k, seed=0, bits=4):
    torch.manual_seed(seed)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    qdata, s_rel, s_channel, correction, cb = eager_w4a8.quantize_w4a8_int8_weight(w, bits=bits)
    assert correction is None
    packed = torch.cat((qdata.flatten(), s_rel.view(torch.int8).flatten()))
    cuda_backend.pack_w4a8_mma_weight_(packed, s_rel)
    return qdata, s_rel, s_channel, cb, packed


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="packed W4A8 MMA decode needs sm_90+ (cluster quantizer)",
)
class TestW4A8PackedMMA:
    # (n, k) covering each native split: 8 records (K = 1280, 40 records = 5 splits)
    # and 16 records (K % 512 == 0, single and multi-split).
    @pytest.mark.parametrize(("n", "k"), [(4096, 4096), (320, 1280), (17408, 2560), (16384, 512)])
    @pytest.mark.parametrize("m", [1, 7, 8])
    @pytest.mark.parametrize("bias", [False, True])
    @pytest.mark.parametrize("bits", [4, 6])
    def test_matches_unpacked_route(self, n, k, m, bias, bits, seed):
        qdata, s_rel, s_channel, cb, packed = _packed_weight(n, k, bits=bits)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, device="cuda", dtype=torch.float32) if bias else None
        empty = torch.empty(0, device="cuda", dtype=torch.float8_e4m3fn)
        got = ck.w4a8_int8_linear(x, packed, empty, s_channel, bias=b, mma_packed=True)
        ref = ck.w4a8_int8_linear(x, qdata, s_rel, s_channel, codebook=cb, bias=b)
        assert got.shape == (m, n)
        # same int8 grid and int32 accumulation; the only difference is fp32 epilogue order
        torch.testing.assert_close(got.float(), ref.float(), atol=2e-2, rtol=1e-2)

    @pytest.mark.parametrize("k", [1280, 5120, 17408])
    @pytest.mark.parametrize("bits", [4, 6])
    def test_packed_chunked_gemm_integer_reference(self, k, bits, seed):
        # Two full chunks and a tail, full-range asymmetric codes, and every finite
        # nonnegative FP8 scale encoding. Keep the integer sum exactly representable
        # in FP32 so layout mistakes cannot hide behind output tolerances.
        n, m, chunk = 272, 9, 128
        codes = torch.randint(0, 1 << bits, (n, k), device="cuda", dtype=torch.int32)
        qdata = eager_w4a8._pack_codes(codes, bits)
        s_rel = (torch.arange(n * k // 16, device="cuda") % 127).to(torch.uint8).view(torch.float8_e4m3fn).view(n, k // 16)
        cb = eager_w4a8._FIXED_LUT if bits == 4 else None
        cb = torch.tensor(cb, device="cuda", dtype=torch.float32) if cb is not None else None
        levels = cb[codes.long()] if bits == 4 else codes.float() - 32
        decoded = (levels * s_rel.float().repeat_interleave(16, dim=1)).round().clamp(-127, 127)
        xq = torch.randint(-3, 4, (m, k), device="cuda", dtype=torch.int8)
        expected = xq.double() @ decoded.double().t()
        packed = torch.cat((qdata.flatten(), s_rel.view(torch.int8).flatten()))
        cuda_backend.pack_w4a8_mma_weight_(packed, s_rel)
        original = packed.clone()
        workspace = torch.empty(chunk, k, device="cuda", dtype=torch.int8)
        out = torch.empty(m, n, device="cuda", dtype=torch.float32)
        wrap = cuda_backend._wrap_for_dlpack
        assert cuda_backend._C.w4a8_codebook_gemm_chunked(
            wrap(xq), wrap(packed), wrap(torch.empty(0, device="cuda", dtype=torch.uint8)),
            wrap(cb) if cb is not None else None,
            wrap(torch.ones(n, device="cuda")), wrap(torch.ones(m, device="cuda")), None,
            wrap(workspace), wrap(out), 16, chunk, 0, torch.cuda.current_stream().cuda_stream, True,
        )
        assert torch.equal(out.double(), expected)
        assert torch.equal(workspace[:n % chunk], decoded[-(n % chunk):].to(torch.int8))
        assert torch.equal(packed, original)

    @pytest.mark.parametrize("m", [1, 7])
    @pytest.mark.parametrize("bias", [False, True])
    @pytest.mark.parametrize("bits", [4, 6])
    def test_fused_residual_matches_addcmul(self, m, bias, bits, seed):
        # The epilogue's residual + scale * linear must be bit-identical to the unfused
        # addcmul on the same bf16-rounded linear; a scale of exactly 1 must reduce
        # to torch.add(residual, linear).
        n, k = 4096, 4096
        _, _, s_channel, _, packed = _packed_weight(n, k, bits=bits)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, device="cuda", dtype=torch.float32) if bias else None
        empty = torch.empty(0, device="cuda", dtype=torch.float8_e4m3fn)
        residual = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        scale = torch.rand(n, device="cuda", dtype=torch.bfloat16) + 0.5
        linear = ck.w4a8_int8_linear(x, packed, empty, s_channel, bias=b, mma_packed=True)
        got = ck.w4a8_int8_linear(x, packed, empty, s_channel, bias=b, mma_packed=True,
                                            residual=residual, residual_scale=scale)
        assert torch.equal(got, torch.addcmul(residual, linear, scale))
        ones = torch.ones(n, device="cuda", dtype=torch.bfloat16)
        got = ck.w4a8_int8_linear(x, packed, empty, s_channel, bias=b, mma_packed=True,
                                            residual=residual, residual_scale=ones)
        assert torch.equal(got, residual + linear)
        # the prefill-shaped (unfused) route applies the same contract
        big = torch.randn(9, k, device="cuda", dtype=torch.bfloat16)
        big_res = torch.randn(9, n, device="cuda", dtype=torch.bfloat16)
        linear = ck.w4a8_int8_linear(big, packed, empty, s_channel, bias=b, mma_packed=True)
        got = ck.w4a8_int8_linear(big, packed, empty, s_channel, bias=b, mma_packed=True,
                                            residual=big_res, residual_scale=scale)
        assert torch.equal(got, torch.addcmul(big_res, linear, scale))

    @pytest.mark.parametrize("m", [1, 5, 8])
    @pytest.mark.parametrize("act", ["rms_norm", "swiglu"])
    @pytest.mark.parametrize("bits", [4, 6])
    def test_fused_activation_matches_eager(self, m, act, bits, seed):
        k = 4096
        _, _, s_channel, _, packed = _packed_weight(4096, k, seed=1, bits=bits)
        x = torch.randn(m, k * (2 if act == "swiglu" else 1), device="cuda", dtype=torch.bfloat16) * 3.0
        w = torch.randn(k, device="cuda", dtype=torch.bfloat16)
        eps = 1e-6
        empty = torch.empty(0, device="cuda", dtype=torch.float8_e4m3fn)
        activated = (torch.nn.functional.silu(x[:, :k]) * x[:, k:] if act == "swiglu"
                     else torch.nn.functional.rms_norm(x, (k,), weight=w, eps=eps))
        expected = ck.w4a8_int8_linear(activated, packed, empty, s_channel, mma_packed=True)
        got = ck.w4a8_int8_linear(x, packed, empty, s_channel, mma_packed=True,
                                           input_act=act, input_act_weight=w, input_act_eps=eps)
        assert torch.equal(got, expected)

    def test_rows_are_independent(self, seed):
        # Row r of a 7-row call must equal row 0 of a 1-row call on the same input: the MMA
        # beat's padding rows must not leak into the real ones.
        n, k = 4096, 4096
        _, _, s_channel, _, packed = _packed_weight(n, k)
        x = torch.randn(7, k, device="cuda", dtype=torch.bfloat16)
        empty = torch.empty(0, device="cuda", dtype=torch.float8_e4m3fn)
        full = ck.w4a8_int8_linear(x, packed, empty, s_channel, mma_packed=True)
        for r in (0, 3, 6):
            single = ck.w4a8_int8_linear(x[r:r + 1], packed, empty, s_channel, mma_packed=True)
            assert torch.equal(full[r], single[0]), f"row {r}"

    def test_concurrent_graphs_match_serial(self, seed):
        # Capture on the same stream, then replay independently: neither graph may use
        # another invocation's split-K sums or arrival counters, even for identical N.
        n, k = 4096, 4096
        _, _, s_channel, _, packed = _packed_weight(n, k)
        inputs = [torch.randn(8, k, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        empty = torch.empty(0, device="cuda", dtype=torch.float8_e4m3fn)

        def linear(x):
            return ck.w4a8_int8_linear(x, packed, empty, s_channel, mma_packed=True)

        expected = [linear(x) for x in inputs]
        torch.cuda.synchronize()
        graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
        outputs = []
        capture_stream = torch.cuda.Stream()
        for graph, x in zip(graphs, inputs, strict=False):
            with torch.cuda.graph(graph, stream=capture_stream):
                outputs.append(linear(x))
        torch.cuda.synchronize()
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        for _ in range(16):
            for stream, graph in zip(streams, graphs, strict=False):
                with torch.cuda.stream(stream):
                    graph.replay()
            torch.cuda.synchronize()
            for got, ref in zip(outputs, expected, strict=False):
                assert torch.equal(got, ref)

    @pytest.mark.parametrize("bits", [4, 6])
    def test_gemv_optimization_roundtrip(self, bits, seed):
        # Kitchen packs a resident canonical weight in place of its qdata + s_rel
        # bytes; the layout still computes the same linear for decode and prefill rows,
        # refuses to serialize the packed form, and leaves non-streaming shapes alone.
        n, k = 4096, 4096
        qdata, s_rel, s_channel, cb, _ = _packed_weight(n, k, bits=bits)
        assert (cb is None) == (bits == 6)
        params = AsymW4A8Int8Layout.Params(
            scale=s_rel, s_channel=s_channel, codebook=cb,
            orig_dtype=torch.bfloat16, orig_shape=(n, k),
        )
        prefix, gap, suffix = 64, 1024, 128
        storage = torch.full((prefix + qdata.nbytes + gap + s_rel.nbytes + suffix,), 73, dtype=torch.uint8, device="cuda")
        resident_q = storage[prefix:prefix + qdata.nbytes].view(torch.int8).view_as(qdata)
        resident_s = storage[-suffix - s_rel.nbytes:-suffix].view(s_rel.dtype).view_as(s_rel)
        resident_q.copy_(qdata)
        resident_s.copy_(s_rel)
        resident = QuantizedTensor(resident_q, "AsymW4A8Int8Layout", dataclasses.replace(params, scale=resident_s))
        weight = resident.optimize_for_gemv(storage)
        assert weight.data_ptr() == resident_q.data_ptr()
        assert weight.params.mma_packed and weight.params.scale.numel() == 0
        assert weight._qdata.nbytes == qdata.nbytes + s_rel.nbytes
        assert weight.optimize_for_gemv(storage) is weight
        assert torch.all(storage[:prefix] == 73) and torch.all(storage[-suffix:] == 73)
        with pytest.raises(ValueError):
            weight.state_dict()
        saved = AsymW4A8Int8Layout.state_dict_tensors(qdata, params)
        assert saved[""] is qdata and saved["_s_rel"] is s_rel

        reference = QuantizedTensor(qdata, "AsymW4A8Int8Layout", params)
        for m in (1, 8, 9, 33, 512):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            got = torch.nn.functional.linear(x, weight)
            ref = torch.nn.functional.linear(x, reference)
            torch.testing.assert_close(got.float(), ref.float(), atol=2e-2, rtol=1e-2)
        assert torch.equal(weight.dequantize(), reference.dequantize())

        odd = AsymW4A8Int8Layout.Params(
            scale=s_rel[:4088], s_channel=s_channel[:4088], codebook=cb,
            orig_dtype=torch.bfloat16, orig_shape=(4088, k),
        )
        odd_weight = QuantizedTensor(qdata[:4088], "AsymW4A8Int8Layout", odd)
        assert odd_weight.optimize_for_gemv(storage) is odd_weight

    @pytest.mark.parametrize(("n", "k"), [(16, 256), (272, 1280), (17408, 5120), (5120, 17408)])
    @pytest.mark.parametrize("bits", [4, 6])
    def test_inplace_pack_roundtrip_and_stream_order(self, n, k, bits, seed):
        # Full-range codes/scales exercise every bit, including W6's high plane.
        # Cover asymmetric projection shapes and a single slab. Independent
        # streams must not share scratch or overwrite adjacent gathered tensors.
        q = torch.randint(-128, 128, (n, k * bits // 8), device="cuda", dtype=torch.int8)
        scales = torch.randint(0, 256, (n, k // 16), device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn)
        params = AsymW4A8Int8Layout.Params(
            scale=scales, s_channel=torch.ones(n, device="cuda"),
            codebook=torch.tensor(eager_w4a8._FIXED_LUT) if bits == 4 else None,
            orig_dtype=torch.bfloat16, orig_shape=(n, k),
        )
        streams, results = [torch.cuda.Stream(), torch.cuda.Stream()], []
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                storage = torch.full((64 + q.nbytes + 1024 + scales.nbytes + 128,), 73, device="cuda", dtype=torch.uint8)
                resident_q = storage[64:64 + q.nbytes].view(q.dtype).view_as(q)
                resident_s = storage[-128 - scales.nbytes:-128].view(scales.dtype).view_as(scales)
                resident_q.copy_(q)
                resident_s.copy_(scales)
                weight = QuantizedTensor(resident_q, "AsymW4A8Int8Layout", dataclasses.replace(params, scale=resident_s))
                if not results:
                    torch.cuda.reset_peak_memory_stats()
                    allocated = torch.cuda.memory_allocated()
                packed = weight.optimize_for_gemv(storage)
                if not results:
                    # No full-weight temporary: saved scales plus one slab.
                    assert torch.cuda.max_memory_allocated() - allocated <= scales.nbytes + (1 << 20)
                results.append((storage, packed))
        for stream in streams:
            torch.cuda.current_stream().wait_stream(stream)
        for storage, packed in results:
            assert packed.data_ptr() == storage.data_ptr() + 64
            unpacked_q, unpacked_scales = unpack_w4a8_mma_weight(packed._qdata, n, k)
            assert torch.equal(unpacked_q, q)
            assert torch.equal(unpacked_scales.view(torch.uint8), scales.view(torch.uint8))
            assert torch.all(storage[:64] == 73) and torch.all(storage[-128:] == 73)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="CUDA W4A8 linear needs sm_80+",
)
@requires_cuda_backend
@pytest.mark.parametrize("act", ["rms_norm", "swiglu", "situ"])
@pytest.mark.parametrize("bits", [4, 6])
def test_prefill_fused_activation_matches_eager(act, bits, seed):
    """M >= 512 quantizes with the 64-lane ConvRot kernel, which fuses the activation."""
    m, n, k = 600, 512, 4096
    qdata, s_rel, s_channel, cb, _ = _packed_weight(n, k, seed=1, bits=bits)
    x = torch.randn(m, k * (1 if act == "rms_norm" else 2), device="cuda", dtype=torch.bfloat16) * 3.0
    w = torch.randn(k, device="cuda", dtype=torch.bfloat16)
    if act == "rms_norm":
        activated = torch.nn.functional.rms_norm(x, (k,), weight=w, eps=1e-6)
    elif act == "swiglu":
        activated = torch.nn.functional.silu(x[:, :k]) * x[:, k:]
    else:
        hidden, gate = x.chunk(2, dim=-1)
        activated = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate) * (25.0 * torch.tanh(hidden / 25.0))
    eager = ck.w4a8_int8_linear(activated, qdata, s_rel, s_channel, codebook=cb)
    fused = ck.w4a8_int8_linear(x, qdata, s_rel, s_channel, codebook=cb,
                                input_act=act, input_act_weight=w, input_act_eps=1e-6)
    # the kernel evaluates the activation in fp32, eager rounds it to bf16 first
    assert ((fused.float() - eager.float()).norm() / eager.float().norm()).item() < 0.03


@pytest.mark.parametrize("m", [1, 9])
def test_canonical_residual_matches_addcmul(m, seed):
    # the canonical layout's decode GEMV and prefill routes apply the residual after the linear
    n, k = 4096, 4096
    qdata, s_rel, s_channel, cb, _ = _packed_weight(n, k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    scale = torch.rand(n, device="cuda", dtype=torch.bfloat16) + 0.5
    linear = ck.w4a8_int8_linear(x, qdata, s_rel, s_channel, codebook=cb)
    got = ck.w4a8_int8_linear(x, qdata, s_rel, s_channel, codebook=cb,
                                        residual=residual, residual_scale=scale)
    assert torch.equal(got, torch.addcmul(residual, linear, scale))
