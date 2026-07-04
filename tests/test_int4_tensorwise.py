# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for INT4 tensor-wise (int4_tensorwise, W4A4 + optional ConvRot) quantization."""

import pytest
import torch

from comfy_kitchen.backends.eager.quantization import (
    _build_hadamard,
    _int4_pack_rowwise,
    _int4_unpack_rowwise,
    _rotate_activation,
    _rotate_weight,
    dequantize_int4_convrot_weight,
    dequantize_int4_simple,
    int4_linear,
    quantize_int4_convrot_weight,
    quantize_int4_rowwise,
)


def test_int4_pack_unpack_roundtrip(seed):
    codes = torch.randint(-8, 8, (16, 64), dtype=torch.int8)
    packed = _int4_pack_rowwise(codes)
    assert packed.dtype == torch.int8
    assert packed.shape == (16, 32)
    assert torch.equal(_int4_unpack_rowwise(packed), codes)


def test_int4_pack_low_nibble_first():
    packed = _int4_pack_rowwise(torch.tensor([[1, -2]], dtype=torch.int8))
    # low nibble = element 0 (0x1), high nibble = element 1 (-2 -> 0xE)
    assert packed.contiguous().view(torch.uint8).item() == (0x1 | (0xE << 4))


def test_int4_pack_rejects_odd_last_dim():
    with pytest.raises(ValueError, match="even last dim"):
        _int4_pack_rowwise(torch.zeros(4, 7, dtype=torch.int8))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_quantize_int4_rowwise_recipe(seed, dtype):
    """Recipe parity with the int8_tensorwise convention at 4-bit: fp32 scale =
    absmax/7 clamped 1e-30, DIVISION with the scale cast to the source dtype,
    round-half-even, clamp [-7, 7]."""
    x = torch.randn(8, 128, dtype=dtype)
    q, scale = quantize_int4_rowwise(x)
    assert q.shape == (8, 64)
    assert q.dtype == torch.int8
    assert scale.shape == (8, 1)
    assert scale.dtype == torch.float32

    expected_scale = (x.abs().amax(dim=-1, keepdim=True).float() / 7.0).clamp(min=1e-30)
    assert torch.equal(scale, expected_scale)

    codes = _int4_unpack_rowwise(q)
    assert codes.min().item() >= -7
    assert codes.max().item() <= 7
    expected = (x / expected_scale.to(dtype)).round().clamp(-7, 7).to(torch.int8)
    assert torch.equal(codes, expected)


def test_quantize_int4_rowwise_rejects_odd_k():
    with pytest.raises(ValueError, match="even last dim"):
        quantize_int4_rowwise(torch.randn(4, 7))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_convrot_weight_matches_rotate_then_rowwise(seed, dtype):
    """quantize_int4_convrot_weight == Hadamard at the SOURCE dtype, then rowwise."""
    w = torch.randn(8, 128, dtype=dtype)
    q, scale = quantize_int4_convrot_weight(w, 64)
    h = _build_hadamard(64, device=w.device, dtype=w.dtype)
    q_ref, scale_ref = quantize_int4_rowwise(_rotate_weight(w, h, 64))
    assert torch.equal(q, q_ref)
    assert torch.equal(scale, scale_ref)


def test_convrot_weight_group_must_divide():
    with pytest.raises((ValueError, RuntimeError)):
        quantize_int4_convrot_weight(torch.randn(8, 96), 64)


def test_dequant_roundtrip_error_bound(seed):
    """|W - dequant(quant(W))| <= scale/2 per element (fp32, deterministic rounding)."""
    w = torch.randn(16, 128, dtype=torch.float32)
    q, scale = quantize_int4_rowwise(w)
    deq = dequantize_int4_simple(q, scale)
    assert deq.dtype == torch.float32
    assert deq.shape == w.shape
    assert ((w - deq).abs() <= scale * 0.5 + 1e-6).all()


def test_convrot_dequant_returns_original_basis(seed):
    """dequantize_int4_convrot_weight rotates back: result approximates W, not rot(W)."""
    w = torch.randn(16, 128, dtype=torch.float32)
    q, scale = quantize_int4_convrot_weight(w, 64)
    deq = dequantize_int4_convrot_weight(q, scale, 64)
    rel = (w - deq).norm() / w.norm()
    assert rel.item() < 0.2, f"relative error too high: {rel:.4f}"
    # and it must NOT be the rotated-basis reconstruction
    h = _build_hadamard(64, device=w.device, dtype=torch.float32)
    rel_rot = (_rotate_weight(w, h, 64) - deq).norm() / w.norm()
    assert rel_rot > rel


def test_int4_linear_exact_integer_case(seed):
    """With exactly-representable codes/scales the whole W4A4 path is exact."""
    w_codes = torch.randint(-7, 8, (16, 64), dtype=torch.int8)
    w_codes[:, 0] = 7  # pin per-row absmax so scale is exactly 0.5
    w = w_codes.float() * 0.5
    x_codes = torch.randint(-7, 8, (4, 64), dtype=torch.int8)
    x_codes[:, 0] = 7  # pin per-row absmax so scale is exactly 0.25
    x = x_codes.float() * 0.25

    wq, ws = quantize_int4_rowwise(w)
    assert torch.equal(_int4_unpack_rowwise(wq), w_codes)

    y = int4_linear(x, wq, ws, None, torch.float32)
    assert torch.equal(y, x @ w.T)

    bias = torch.arange(16, dtype=torch.float32)
    y_bias = int4_linear(x, wq, ws, bias, torch.float32)
    assert torch.equal(y_bias, x @ w.T + bias)


def test_int4_linear_convrot_matches_eager_composition(seed):
    """int4_linear(convrot=True) == rotate act -> rowwise quant -> exact int mm ->
    rank-1 scale epilogue, composed from the public pieces."""
    w = torch.randn(16, 128, dtype=torch.float32)
    x = torch.randn(4, 128, dtype=torch.float32)
    wq, ws = quantize_int4_convrot_weight(w, 64)

    y = int4_linear(x, wq, ws, None, torch.float32, convrot=True, convrot_groupsize=64)

    h = _build_hadamard(64, device=x.device, dtype=x.dtype)
    x_rot = _rotate_activation(x, h, 64)
    xq, xs = quantize_int4_rowwise(x_rot)
    y_ref = (_int4_unpack_rowwise(xq).float() * xs) @ (_int4_unpack_rowwise(wq).float() * ws).T
    torch.testing.assert_close(y, y_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("convrot", [False, True])
def test_int4_linear_close_to_dense(seed, convrot):
    """W4A4 output stays within coarse-4-bit tolerance of the dense linear."""
    w = torch.randn(32, 256, dtype=torch.float32)
    x = torch.randn(8, 256, dtype=torch.float32)
    if convrot:
        wq, ws = quantize_int4_convrot_weight(w, 64)
    else:
        wq, ws = quantize_int4_rowwise(w)
    y = int4_linear(x, wq, ws, None, torch.float32, convrot=convrot, convrot_groupsize=64)
    y_ref = x @ w.T
    rel = (y - y_ref).norm() / y_ref.norm()
    assert rel.item() < 0.25, f"relative error too high (convrot={convrot}): {rel:.4f}"


def test_int4_linear_batch_dims(seed):
    w = torch.randn(16, 64, dtype=torch.float32)
    x = torch.randn(2, 3, 64, dtype=torch.float32)
    wq, ws = quantize_int4_rowwise(w)
    y = int4_linear(x, wq, ws, None, torch.float32)
    assert y.shape == (2, 3, 16)
    y_2d = int4_linear(x.reshape(6, 64), wq, ws, None, torch.float32)
    torch.testing.assert_close(y, y_2d.reshape(2, 3, 16))


def test_int4_linear_shape_mismatch_raises():
    wq, ws = quantize_int4_rowwise(torch.randn(16, 64))
    with pytest.raises(ValueError, match="inner dimensions"):
        int4_linear(torch.randn(4, 128), wq, ws, None, torch.float32)


def test_int4_linear_convrot_group_must_divide():
    wq, ws = quantize_int4_rowwise(torch.randn(16, 96))
    with pytest.raises(ValueError, match="does not divide"):
        int4_linear(torch.randn(4, 96), wq, ws, None, torch.float32, convrot=True, convrot_groupsize=64)


class TestInt4LinearCudaParity:
    """CUDA int4_linear (fused ConvRot + s4 MMA via the svdquant GEMM) vs eager."""

    @pytest.fixture(autouse=True)
    def cuda_kernel_only(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        try:
            from comfy_kitchen.backends.cuda import _C  # noqa: F401
        except Exception:
            pytest.skip("compiled comfy_kitchen CUDA extension required")
        if torch.cuda.get_device_capability() < (8, 0):
            pytest.skip("int4 MMA path requires SM >= 8.0")

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("convrot", [False, True])
    @pytest.mark.parametrize("shape", [(1, 512, 256), (37, 3072, 1024), (300, 12288, 512)])
    def test_matches_eager(self, seed, dtype, convrot, shape):
        from comfy_kitchen.backends.cuda import int4_linear as cuda_int4_linear

        m, k, n = shape
        x = torch.randn(m, k, dtype=dtype, device="cuda")
        w = torch.randn(n, k, dtype=dtype, device="cuda")
        bias = torch.randn(n, dtype=dtype, device="cuda")
        if convrot:
            wq, ws = quantize_int4_convrot_weight(w, 256)
        else:
            wq, ws = quantize_int4_rowwise(w)
        y_ref = int4_linear(x, wq, ws, bias, dtype, convrot, 256).float()
        y_cuda = cuda_int4_linear(x, wq, ws, bias, dtype, convrot, 256).float()
        # The GEMM epilogue uses 16-bit (a/w)scales vs eager's fp32; the ConvRot
        # path additionally rounds the fp32 FHT output through the source dtype
        # to reproduce eager's rotated values. Residual divergence is boundary
        # code flips at ~1e-3 relative.
        rel = (y_cuda - y_ref).norm() / y_ref.norm().clamp(min=1e-9)
        assert rel.item() < 0.02, f"cuda/eager divergence {rel:.5f} (convrot={convrot}, {shape})"

    def test_batch_dims_and_dispatch(self, seed):
        from comfy_kitchen.tensor import QuantizedTensor

        x = torch.randn(2, 3, 512, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda")
        qt_w = QuantizedTensor.from_float(w, "TensorWiseINT4Layout", convrot=True)
        out = torch.nn.functional.linear(x, qt_w)
        assert out.shape == (2, 3, 256)
        ref = x.float() @ w.float().transpose(0, 1)
        rel = (out.float() - ref).norm() / ref.norm()
        assert rel.item() < 0.3


class TestTensorWiseINT4Layout:
    """Tests for the TensorWiseINT4Layout quantized tensor format."""

    @pytest.fixture(autouse=True)
    def cuda_only(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for TensorWiseINT4Layout tests")

    def test_weight_quantize_shape_dtype(self, seed):
        from comfy_kitchen.tensor import QuantizedTensor

        w = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(w, "TensorWiseINT4Layout")

        assert qt._qdata.dtype == torch.int8
        assert qt._qdata.shape == (256, 256)  # packed [N, K/2]
        assert qt._params.scale.shape == (256, 1)
        assert qt._params.scale.dtype == torch.float32
        assert qt._params.orig_shape == (256, 512)

    def test_convrot_quantize_params(self, seed):
        from comfy_kitchen.tensor import TensorWiseINT4Layout

        w = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
        qdata, params = TensorWiseINT4Layout.quantize(w, convrot=True, convrot_groupsize=256)
        assert qdata.shape == (64, 256)
        assert params.convrot is True
        assert params.convrot_groupsize == 256

    def test_weight_dequantize_dtype_and_logical_shape(self, seed):
        from comfy_kitchen.tensor import QuantizedTensor

        for dtype in (torch.float16, torch.bfloat16):
            w = torch.randn(64, 128, device="cuda", dtype=dtype)
            qt = QuantizedTensor.from_float(w, "TensorWiseINT4Layout")
            dq = qt.dequantize()
            assert dq.dtype == dtype
            assert dq.shape == w.shape

    @pytest.mark.parametrize("convrot", [False, True])
    def test_weight_roundtrip_error(self, seed, convrot):
        from comfy_kitchen.tensor import QuantizedTensor

        w = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(w, "TensorWiseINT4Layout", convrot=convrot)
        dq = qt.dequantize()

        rel = (w.float() - dq.float()).norm() / w.float().norm()
        assert rel.item() < 0.2, f"relative roundtrip error too high: {rel:.4f}"

    def test_state_dict_tensors_keys(self, seed):
        from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT4Layout

        w = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(w, "TensorWiseINT4Layout")
        sd = TensorWiseINT4Layout.state_dict_tensors(qt._qdata, qt._params)

        assert set(sd.keys()) == {"", "_scale"}
        assert sd[""].dtype == torch.int8
        assert sd[""].shape == (64, 32)
        assert sd["_scale"].shape == (64, 1)

    def test_requantize_kwargs_preserve_convrot(self, seed):
        from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT4Layout

        w = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(w, "TensorWiseINT4Layout", convrot=True, convrot_groupsize=256)
        kwargs = TensorWiseINT4Layout.requantize_kwargs(qt)
        assert kwargs["convrot"] is True
        assert kwargs["convrot_groupsize"] == 256
        assert kwargs["per_channel"] is True

    def test_transpose_flips_flags(self, seed):
        from comfy_kitchen.tensor import QuantizedTensor

        w = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(w, "TensorWiseINT4Layout")
        qt_t = torch.ops.aten.t.default(qt)
        assert qt_t._params.transposed is True
        assert qt_t._params.orig_shape == (128, 64)

    @pytest.mark.parametrize("convrot", [False, True])
    def test_linear_dispatch(self, seed, convrot):
        from comfy_kitchen.tensor import QuantizedTensor

        x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
        qt_w = QuantizedTensor.from_float(w, "TensorWiseINT4Layout", convrot=convrot)

        out = torch.nn.functional.linear(x, qt_w)

        assert out.shape == (4, 64)
        assert out.dtype == torch.bfloat16
        rel = (out.float() - (x.float() @ w.float().T)).norm() / (x.float() @ w.float().T).norm()
        assert rel.item() < 0.3, f"linear dispatch error too high (convrot={convrot}): {rel:.4f}"

    def test_linear_dispatch_with_bias(self, seed):
        from comfy_kitchen.tensor import QuantizedTensor

        x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(64, device="cuda", dtype=torch.bfloat16)
        qt_w = QuantizedTensor.from_float(w, "TensorWiseINT4Layout")

        out = torch.nn.functional.linear(x, qt_w, bias)
        assert out.shape == (4, 64)
        assert out.dtype == torch.bfloat16

    def test_logical_shape_and_no_padding(self, seed):
        """padded_shape reverses the 2-per-byte packing; int4 never pads."""
        from comfy_kitchen.tensor import QuantizedTensor

        w = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(w, "TensorWiseINT4Layout")
        assert qt._qdata.shape == (64, 64)  # packed storage
        assert qt.padded_shape == (64, 128)  # logical, unpacked
        assert qt.is_padded is False
        assert qt.shape == (64, 128)

    def test_mm_dispatch_transposed_and_fallback(self, seed):
        """mm(x, qt.t()) rides int4_linear; non-transposed mm falls back to dequant."""
        from comfy_kitchen.tensor import QuantizedTensor

        x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        qt_w = QuantizedTensor.from_float(w, "TensorWiseINT4Layout")

        out = torch.mm(x, qt_w.t())
        assert out.shape == (4, 64)
        ref = x.float() @ w.float().T
        rel = (out.float() - ref).norm() / ref.norm()
        assert rel.item() < 0.3, f"transposed mm error too high: {rel:.4f}"

        # Non-transposed: per-row scales cannot ride int4_linear -> dequant fallback.
        w_sq = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        qt_sq = QuantizedTensor.from_float(w_sq, "TensorWiseINT4Layout")
        out_sq = torch.mm(x, qt_sq)
        ref_sq = x.float() @ qt_sq.dequantize().float()
        rel_sq = (out_sq.float() - ref_sq).norm() / ref_sq.norm()
        assert rel_sq.item() < 0.05, f"non-transposed mm fallback error too high: {rel_sq:.4f}"

    def test_addmm_dispatch_transposed_and_fallback(self, seed):
        from comfy_kitchen.tensor import QuantizedTensor

        x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(64, device="cuda", dtype=torch.bfloat16)
        qt_w = QuantizedTensor.from_float(w, "TensorWiseINT4Layout")

        out = torch.addmm(bias, x, qt_w.t())
        assert out.shape == (4, 64)
        ref = bias.float() + x.float() @ w.float().T
        rel = (out.float() - ref).norm() / ref.norm()
        assert rel.item() < 0.3, f"transposed addmm error too high: {rel:.4f}"

        w_sq = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        qt_sq = QuantizedTensor.from_float(w_sq, "TensorWiseINT4Layout")
        bias_sq = torch.randn(128, device="cuda", dtype=torch.bfloat16)
        out_sq = torch.addmm(bias_sq, x, qt_sq)
        ref_sq = bias_sq.float() + x.float() @ qt_sq.dequantize().float()
        rel_sq = (out_sq.float() - ref_sq).norm() / ref_sq.norm()
        assert rel_sq.item() < 0.05, f"non-transposed addmm fallback error too high: {rel_sq:.4f}"

    @pytest.mark.parametrize("convrot", [False, True])
    def test_mm_dispatch_convrot(self, seed, convrot):
        from comfy_kitchen.tensor import QuantizedTensor

        x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
        qt_w = QuantizedTensor.from_float(w, "TensorWiseINT4Layout", convrot=convrot)
        out = torch.mm(x, qt_w.t())
        ref = x.float() @ w.float().T
        rel = (out.float() - ref).norm() / ref.norm()
        assert rel.item() < 0.3, f"mm convrot={convrot} error too high: {rel:.4f}"
