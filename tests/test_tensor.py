"""Unit tests for comfy_kitchen.tensor module."""
import pytest
import torch

import comfy_kitchen
from comfy_kitchen.tensor import (
    BaseLayoutParams,
    QuantizedTensor,
    TensorCoreFP8Layout,
    TensorCoreMXFP8Layout,
    TensorCoreNVFP4Layout,
    get_cuda_capability,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestTensorCoreFP8Layout:
    """Tests for FP8 quantization layout."""

    def test_quantize_basic(self):
        x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        qdata, params = TensorCoreFP8Layout.quantize(x)

        assert qdata.dtype == torch.float8_e4m3fn
        assert qdata.shape == x.shape
        assert params.orig_dtype == torch.bfloat16
        assert params.orig_shape == (128, 256)
        assert params.scale.dtype == torch.float32

    def test_quantize_with_scale(self):
        x = torch.randn(64, 64, device="cuda", dtype=torch.float16)
        scale = torch.tensor(0.5, device="cuda")
        _qdata, params = TensorCoreFP8Layout.quantize(x, scale=scale)

        assert torch.allclose(params.scale, scale)

    def test_dequantize_roundtrip(self):
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16) * 10
        qdata, params = TensorCoreFP8Layout.quantize(x)
        dq = TensorCoreFP8Layout.dequantize(qdata, params)

        assert dq.dtype == torch.bfloat16
        assert dq.shape == x.shape
        # FP8 has limited precision, so use loose tolerance
        assert torch.allclose(dq, x, rtol=0.1, atol=0.1)

    def test_params_to_device(self):
        x = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
        _, params = TensorCoreFP8Layout.quantize(x)

        params_cpu = params.to_device(torch.device("cpu"))
        assert params_cpu.scale.device.type == "cpu"
        assert params_cpu.orig_dtype == params.orig_dtype
        assert params_cpu.orig_shape == params.orig_shape

    def test_params_clone(self):
        x = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
        _, params = TensorCoreFP8Layout.quantize(x)

        params_clone = params.clone()
        assert params_clone.scale is not params.scale
        assert torch.equal(params_clone.scale, params.scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestTensorCoreNVFP4Layout:
    """Tests for NVFP4 quantization layout."""

    def test_quantize_aligned(self):
        x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        qdata, params = TensorCoreNVFP4Layout.quantize(x)

        assert qdata.dtype == torch.uint8
        assert qdata.shape == (128, 128)  # cols / 2 due to packing
        assert params.orig_shape == (128, 256)
        assert params.block_scale.dtype == torch.float8_e4m3fn

    def test_quantize_unaligned_pads(self):
        x = torch.randn(129, 130, device="cuda", dtype=torch.bfloat16)
        qdata, params = TensorCoreNVFP4Layout.quantize(x)

        # Should be padded to 144x144, then packed to 144x72
        assert qdata.shape == (144, 72)
        assert params.orig_shape == (129, 130)

    def test_dequantize_roundtrip(self):
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16) * 4
        qdata, params = TensorCoreNVFP4Layout.quantize(x)
        dq = TensorCoreNVFP4Layout.dequantize(qdata, params)

        assert dq.dtype == torch.bfloat16
        # Dequantize returns padded shape, caller handles slicing
        assert dq.shape == (128, 128)

    def test_get_padded_shape(self):
        assert TensorCoreNVFP4Layout.get_padded_shape((128, 256)) == (128, 256)
        assert TensorCoreNVFP4Layout.get_padded_shape((129, 128)) == (144, 128)
        assert TensorCoreNVFP4Layout.get_padded_shape((100, 100)) == (112, 112)

    def test_get_storage_shape(self):
        assert TensorCoreNVFP4Layout.get_storage_shape((128, 256)) == (128, 128)
        assert TensorCoreNVFP4Layout.get_storage_shape((129, 128)) == (144, 64)

    def test_requires_2d(self):
        x = torch.randn(10, 10, 10, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="2D tensor"):
            TensorCoreNVFP4Layout.quantize(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestQuantizedTensor:
    """Tests for QuantizedTensor class."""

    def test_from_float_fp8(self):
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")

        assert isinstance(qt, QuantizedTensor)
        assert qt.shape == (64, 64)
        assert qt.storage_shape == (64, 64)  # FP8 is not packed
        assert qt.padded_shape == (64, 64)   # no packing to reverse
        assert not qt.is_padded  # FP8 doesn't require padding
        assert qt.layout_cls is TensorCoreFP8Layout

    def test_from_float_nvfp4(self):
        x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreNVFP4Layout")

        assert isinstance(qt, QuantizedTensor)
        assert qt.shape == (128, 256)
        assert qt.layout_cls is TensorCoreNVFP4Layout

    def test_shape_vs_storage_shape_aligned(self):
        x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreNVFP4Layout")

        assert qt.shape == (128, 256)
        assert qt.storage_shape == (128, 128)  # packed (cols / 2)
        assert qt.padded_shape == (128, 256)   # logical shape (unpacked)
        assert not qt.is_padded  # no padding needed for 16-aligned dims

    def test_shape_vs_storage_shape_unaligned(self):
        x = torch.randn(129, 130, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreNVFP4Layout")

        assert qt.shape == (129, 130)
        assert qt.storage_shape == (144, 72)   # padded to 144x144, then packed
        assert qt.padded_shape == (144, 144)   # logical shape after padding
        assert qt.is_padded  # padding was applied

    def test_dequantize_fp8(self):
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16) * 5
        qt = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")
        dq = qt.dequantize()

        assert dq.dtype == torch.bfloat16
        assert dq.shape == x.shape
        assert torch.allclose(dq, x, rtol=0.1, atol=0.1)

    def test_dequantize_nvfp4_unpadded(self):
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16) * 4
        qt = QuantizedTensor.from_float(x, "TensorCoreNVFP4Layout")
        dq = qt.dequantize()

        assert dq.shape == (128, 128)
        # NVFP4 has limited precision
        assert torch.allclose(dq, x, rtol=0.5, atol=0.5)

    def test_dequantize_nvfp4_with_padding(self):
        x = torch.randn(129, 130, device="cuda", dtype=torch.bfloat16) * 4
        qt = QuantizedTensor.from_float(x, "TensorCoreNVFP4Layout")
        dq = qt.dequantize()

        # Should return original shape, not padded
        assert dq.shape == (129, 130)

    def test_detach(self):
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")
        qt_detached = qt.detach()

        assert isinstance(qt_detached, QuantizedTensor)
        assert qt_detached._qdata is not qt._qdata

    def test_clone(self):
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")
        qt_cloned = qt.clone()

        assert isinstance(qt_cloned, QuantizedTensor)
        assert qt_cloned._qdata is not qt._qdata
        assert torch.equal(qt_cloned._qdata, qt._qdata)

    def test_to_device_roundtrip(self):
        """Test device transfer: cuda -> cpu -> cuda preserves data."""
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16) * 5
        qt = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")

        # Move to CPU
        qt_cpu = qt.to("cpu")
        assert isinstance(qt_cpu, QuantizedTensor)
        assert qt_cpu._qdata.device.type == "cpu"
        assert qt_cpu._params.scale.device.type == "cpu"

        # Move back to CUDA
        qt_cuda = qt_cpu.to("cuda")
        assert qt_cuda._qdata.device.type == "cuda"
        assert qt_cuda._params.scale.device.type == "cuda"

        # Verify data integrity
        dq_original = qt.dequantize()
        dq_roundtrip = qt_cuda.dequantize()
        assert torch.allclose(dq_original, dq_roundtrip)

    def test_to_dtype_changes_orig_dtype(self):
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")

        assert qt._params.orig_dtype == torch.bfloat16

        # .float() should change orig_dtype without dequantizing
        qt_float = qt.float()
        assert isinstance(qt_float, QuantizedTensor)
        assert qt_float._params.orig_dtype == torch.float32
        assert torch.equal(qt_float._qdata, qt._qdata)  # qdata unchanged

        # .half() should change orig_dtype
        qt_half = qt.half()
        assert isinstance(qt_half, QuantizedTensor)
        assert qt_half._params.orig_dtype == torch.float16

    def test_to_dtype_dequantize_uses_new_dtype(self):
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")

        qt_float = qt.float()
        dq = qt_float.dequantize()

        assert dq.dtype == torch.float32

    def test_to_device_and_dtype(self):
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")

        # Move to CPU and change dtype
        qt_cpu_float = qt.to("cpu", dtype=torch.float32)

        assert isinstance(qt_cpu_float, QuantizedTensor)
        assert qt_cpu_float._qdata.device.type == "cpu"
        assert qt_cpu_float._params.orig_dtype == torch.float32

    def test_to_dtype_positional_arg(self):
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")

        qt_copy = qt.to(torch.float32, copy=True)

        assert isinstance(qt_copy, QuantizedTensor)
        assert qt_copy._qdata.dtype == torch.float8_e4m3fn
        assert qt_copy._params.orig_dtype == torch.float32
        assert qt_copy._qdata.data_ptr() != qt._qdata.data_ptr()
        assert torch.equal(qt_copy._qdata, qt._qdata)

    def test_to_copy_without_dtype_change(self):
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")

        qt_copy = qt.to(copy=True)

        assert isinstance(qt_copy, QuantizedTensor)
        assert qt_copy._qdata.data_ptr() != qt._qdata.data_ptr()
        assert torch.equal(qt_copy._qdata, qt._qdata)
        assert qt_copy._params.orig_dtype == qt._params.orig_dtype

    def test_empty_like_with_dtype_preserves_qdata_format(self):
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")

        qt_empty = torch.empty_like(qt, dtype=torch.float32)

        assert isinstance(qt_empty, QuantizedTensor)
        assert qt_empty._qdata.dtype == torch.float8_e4m3fn
        assert qt_empty._params.orig_dtype == torch.float32

    def test_empty_like_copy_pattern_preserves_dtype(self):
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")

        r = torch.empty_like(qt, dtype=torch.float16, device="cuda")
        r.copy_(qt)

        assert isinstance(r, QuantizedTensor)
        assert r._qdata.dtype == torch.float8_e4m3fn
        assert r._params.orig_dtype == torch.float16
        assert r.dequantize().dtype == torch.float16

    def test_repr(self):
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")
        r = repr(qt)

        assert "QuantizedTensor" in r
        assert "TensorCoreFP8Layout" in r
        assert "(64, 64)" in r

    def test_params_property(self):
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")

        assert qt.params is qt._params
        assert hasattr(qt.params, "scale")
        assert hasattr(qt.params, "orig_dtype")
        assert hasattr(qt.params, "orig_shape")

    def test_contiguous(self):
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")

        # Already contiguous - should return self
        qt_contig = qt.contiguous()
        assert isinstance(qt_contig, QuantizedTensor)
        assert qt_contig._qdata.is_contiguous()

    def test_is_contiguous(self):
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")

        assert qt.is_contiguous()

    def test_copy_(self):
        x1 = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        x2 = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16) * 2

        qt1 = QuantizedTensor.from_float(x1, "TensorCoreFP8Layout")
        qt2 = QuantizedTensor.from_float(x2, "TensorCoreFP8Layout")

        # Copy qt2 into qt1
        qt1.copy_(qt2)

        assert torch.equal(qt1._qdata, qt2._qdata)
        assert torch.equal(qt1._params.scale, qt2._params.scale)

    def test_empty_like(self):
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")

        qt_empty = torch.empty_like(qt)

        assert isinstance(qt_empty, QuantizedTensor)
        assert qt_empty._qdata.shape == qt._qdata.shape
        assert qt_empty._params.orig_dtype == qt._params.orig_dtype

    def test_empty_like_different_device(self):
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")

        qt_empty_cpu = torch.empty_like(qt, device="cpu")

        assert isinstance(qt_empty_cpu, QuantizedTensor)
        assert qt_empty_cpu._qdata.device.type == "cpu"
        assert qt_empty_cpu._params.scale.device.type == "cpu"

    def test_fallback_to_dequantize(self):
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")

        # Operations not explicitly handled should dequantize and proceed
        result = qt + 1.0
        assert isinstance(result, torch.Tensor)
        assert not isinstance(result, QuantizedTensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestQuantizedTensorFlatten:
    """Tests for tensor flattening protocol (device movement)."""

    def test_tensor_flatten_unflatten_fp8(self):
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")

        inner_tensors, ctx = qt.__tensor_flatten__()

        assert "_qdata" in inner_tensors
        assert "layout_cls" in ctx
        assert "params_class" in ctx

    def test_tensor_flatten_unflatten_nvfp4(self):
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreNVFP4Layout")

        inner_tensors, _ctx = qt.__tensor_flatten__()

        assert "_qdata" in inner_tensors
        assert "_param_scale" in inner_tensors
        assert "_param_block_scale" in inner_tensors

    def test_tensor_flatten_unflatten_mxfp8(self):
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreMXFP8Layout")

        inner_tensors, ctx = qt.__tensor_flatten__()

        assert "_qdata" in inner_tensors
        assert "_param_scale" in inner_tensors
        assert "layout_cls" in ctx



@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestCapabilityChecking:
    """Tests for hardware capability checking."""

    def test_get_cuda_capability(self):
        cap = get_cuda_capability()
        assert cap is not None
        assert isinstance(cap, tuple)
        assert len(cap) == 2
        assert cap[0] >= 1  # Major version

    def test_fp8_min_sm_version(self):
        assert TensorCoreFP8Layout.MIN_SM_VERSION == (8, 9)

    def test_nvfp4_min_sm_version(self):
        assert TensorCoreNVFP4Layout.MIN_SM_VERSION == (10, 0)

    def test_mxfp8_min_sm_version(self):
        assert TensorCoreMXFP8Layout.MIN_SM_VERSION == (10, 0)

    def test_supports_fast_matmul_returns_bool(self):
        result = TensorCoreFP8Layout.supports_fast_matmul()
        assert isinstance(result, bool)

    def test_get_requirements_fp8(self):
        reqs = TensorCoreFP8Layout.get_requirements()

        assert reqs["layout"] == "TensorCoreFP8Layout"
        assert reqs["min_sm_version"] == (8, 9)
        assert reqs["current_sm_version"] is not None
        assert isinstance(reqs["fast_matmul_supported"], bool)

    def test_get_requirements_nvfp4(self):
        reqs = TensorCoreNVFP4Layout.get_requirements()

        assert reqs["layout"] == "TensorCoreNVFP4Layout"
        assert reqs["min_sm_version"] == (10, 0)
        assert reqs["current_sm_version"] is not None
        assert isinstance(reqs["fast_matmul_supported"], bool)

    def test_supports_fast_matmul_consistent_with_requirements(self):
        for layout_cls in [TensorCoreFP8Layout, TensorCoreNVFP4Layout, TensorCoreMXFP8Layout]:
            reqs = layout_cls.get_requirements()
            assert reqs["fast_matmul_supported"] == layout_cls.supports_fast_matmul()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestCopyValidation:
    """Tests for copy operation validation."""

    def test_copy_non_quantized_tensor_raises(self):
        """Test that copying a regular tensor to QuantizedTensor raises TypeError."""
        x1 = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        x2 = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)

        qt1 = QuantizedTensor.from_float(x1, "TensorCoreFP8Layout")

        with pytest.raises(TypeError, match=r"Cannot copy.*to QuantizedTensor"):
            qt1.copy_(x2)

    def test_copy_mismatched_layouts_raises(self):
        """Test that copying between different layouts raises TypeError."""
        x1 = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        x2 = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)

        qt1 = QuantizedTensor.from_float(x1, "TensorCoreFP8Layout")
        qt2 = QuantizedTensor.from_float(x2, "TensorCoreNVFP4Layout")

        with pytest.raises(TypeError, match="Layout mismatch"):
            qt1.copy_(qt2)



@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestBaseLayoutParams:
    """Tests for BaseLayoutParams functionality."""

    def test_nvfp4_params_tensor_fields(self):
        """Test that NVFP4 Params correctly lists tensor fields."""
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        _, params = TensorCoreNVFP4Layout.quantize(x)

        tensor_fields = params._tensor_fields()

        assert "scale" in tensor_fields
        assert "block_scale" in tensor_fields
        assert len(tensor_fields) == 2

    def test_nvfp4_params_to_device_moves_all_tensors(self):
        """Test that NVFP4 Params.to_device moves all tensor fields."""
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        _, params = TensorCoreNVFP4Layout.quantize(x)

        params_cpu = params.to_device(torch.device("cpu"))

        assert params_cpu.scale.device.type == "cpu"
        assert params_cpu.block_scale.device.type == "cpu"
        # Original unchanged
        assert params.scale.device.type == "cuda"
        assert params.block_scale.device.type == "cuda"

    def test_nvfp4_params_clone_copies_all_tensors(self):
        """Test that NVFP4 Params.clone creates independent copies of all tensors."""
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        _, params = TensorCoreNVFP4Layout.quantize(x)

        params_clone = params.clone()

        # All tensors should be different objects
        assert params_clone.scale is not params.scale
        assert params_clone.block_scale is not params.block_scale
        # But equal values
        assert torch.equal(params_clone.scale, params.scale)
        assert torch.equal(params_clone.block_scale, params.block_scale)

    @pytest.mark.parametrize("layout_cls", [TensorCoreFP8Layout, TensorCoreNVFP4Layout, TensorCoreMXFP8Layout])
    def test_params_inherits_from_base(self, layout_cls):
        assert issubclass(layout_cls.Params, BaseLayoutParams)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestParamsDtypeValidation:
    """Tests for automatic dtype validation and conversion in Params."""

    def test_fp8_params_scale_dtype_auto_conversion(self):
        """Test that FP8 Params automatically converts scale to float32."""
        # Create scale with wrong dtype (float16)
        scale_f16 = torch.tensor([1.0], device="cuda", dtype=torch.float16)

        # Creating Params should auto-convert to float32
        params = TensorCoreFP8Layout.Params(
            scale=scale_f16,
            orig_dtype=torch.bfloat16,
            orig_shape=(128, 128),
        )

        # Verify scale was converted to float32
        assert params.scale.dtype == torch.float32
        assert params.scale.device.type == "cuda"
        assert params.scale.item() == 1.0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestFP8LinearOperations:
    """Tests for FP8 linear, mm, addmm operations."""

    def test_fp8_linear_both_quantized(self):
        """Test FP8 linear with both input and weight quantized."""
        if not TensorCoreFP8Layout.supports_fast_matmul():
            pytest.skip("FP8 matmul not supported on this hardware")

        batch, in_features, out_features = 32, 64, 128
        x = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16)

        qt_x = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")
        qt_w = QuantizedTensor.from_float(w, "TensorCoreFP8Layout")

        result = torch.nn.functional.linear(qt_x, qt_w)
        expected = torch.nn.functional.linear(qt_x.dequantize(), qt_w.dequantize())

        assert result.shape == expected.shape
        assert torch.allclose(result, expected, rtol=0.15, atol=0.15)

    def test_fp8_linear_with_bias(self):
        """Test FP8 linear with bias."""
        if not TensorCoreFP8Layout.supports_fast_matmul():
            pytest.skip("FP8 matmul not supported on this hardware")

        batch, in_features, out_features = 16, 32, 64
        x = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(out_features, device="cuda", dtype=torch.bfloat16)

        qt_x = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")
        qt_w = QuantizedTensor.from_float(w, "TensorCoreFP8Layout")

        result = torch.nn.functional.linear(qt_x, qt_w, b)
        expected = torch.nn.functional.linear(qt_x.dequantize(), qt_w.dequantize(), b)

        assert result.shape == (batch, out_features)
        assert torch.allclose(result, expected, rtol=0.15, atol=0.15)

    def test_fp8_linear_3d_input(self):
        """Test FP8 linear with 3D input (batch, seq, features)."""
        if not TensorCoreFP8Layout.supports_fast_matmul():
            pytest.skip("FP8 matmul not supported on this hardware")

        batch, seq_len, in_features, out_features = 4, 16, 64, 128
        x = torch.randn(batch, seq_len, in_features, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16)

        qt_x = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")
        qt_w = QuantizedTensor.from_float(w, "TensorCoreFP8Layout")

        result = torch.nn.functional.linear(qt_x, qt_w)
        expected = torch.nn.functional.linear(qt_x.dequantize(), qt_w.dequantize())

        assert result.shape == (batch, seq_len, out_features)
        assert torch.allclose(result, expected, rtol=0.15, atol=0.15)

    def test_fp8_linear_single_quantized_fallback(self):
        """Test FP8 linear with only one operand quantized (fallback path)."""
        batch, in_features, out_features = 16, 32, 64
        x = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16)

        qt_x = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")
        qt_w = QuantizedTensor.from_float(w, "TensorCoreFP8Layout")

        result_w = torch.nn.functional.linear(x, qt_w)
        expected_w = torch.nn.functional.linear(x, qt_w.dequantize())
        assert result_w.shape == (batch, out_features)
        assert torch.equal(result_w, expected_w)

        result_x = torch.nn.functional.linear(qt_x, w)
        expected_x = torch.nn.functional.linear(qt_x.dequantize(), w)
        assert result_x.shape == (batch, out_features)
        assert torch.equal(result_x, expected_x)

    def test_fp8_mm(self):
        """Test FP8 matrix multiplication."""
        if not TensorCoreFP8Layout.supports_fast_matmul():
            pytest.skip("FP8 matmul not supported on this hardware")

        m, k, n = 64, 128, 256
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)

        qt_a = QuantizedTensor.from_float(a, "TensorCoreFP8Layout")
        qt_b = QuantizedTensor.from_float(b, "TensorCoreFP8Layout")

        result = torch.mm(qt_a, qt_b)
        expected = torch.mm(qt_a.dequantize(), qt_b.dequantize())

        assert result.shape == expected.shape
        assert torch.allclose(result, expected, rtol=0.15, atol=0.15)

    def test_fp8_mm_single_quantized_fallback(self):
        """Test FP8 mm with only one operand quantized (fallback path)."""
        m, k, n = 32, 64, 128
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)

        qt_a = QuantizedTensor.from_float(a, "TensorCoreFP8Layout")

        result = torch.mm(qt_a, b)
        expected = torch.mm(qt_a.dequantize(), b)

        assert result.shape == expected.shape
        assert torch.equal(result, expected)

    def test_fp8_addmm(self):
        """Test FP8 addmm: bias + input @ weight."""
        if not TensorCoreFP8Layout.supports_fast_matmul():
            pytest.skip("FP8 matmul not supported on this hardware")

        m, k, n = 32, 64, 128
        bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)

        qt_a = QuantizedTensor.from_float(a, "TensorCoreFP8Layout")
        qt_b = QuantizedTensor.from_float(b, "TensorCoreFP8Layout")

        result = torch.addmm(bias, qt_a, qt_b)
        expected = torch.addmm(bias, qt_a.dequantize(), qt_b.dequantize())

        assert result.shape == expected.shape
        assert torch.allclose(result, expected, rtol=0.15, atol=0.15)

    def test_fp8_addmm_fallback(self):
        """Test FP8 addmm with single quantized operand (fallback path)."""
        m, k, n = 32, 64, 128
        bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)

        qt_a = QuantizedTensor.from_float(a, "TensorCoreFP8Layout")

        # Both paths dequantize qt_a, so results should match exactly
        result = torch.addmm(bias, qt_a, b)
        expected = torch.addmm(bias, qt_a.dequantize(), b)

        assert result.shape == expected.shape
        assert torch.equal(result, expected)

    def test_fp8_linear_output_dtype(self):
        """Test that FP8 linear output has correct dtype."""
        if not TensorCoreFP8Layout.supports_fast_matmul():
            pytest.skip("FP8 matmul not supported on this hardware")

        batch, in_features, out_features = 16, 32, 64
        x = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16)

        qt_x = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")
        qt_w = QuantizedTensor.from_float(w, "TensorCoreFP8Layout")

        result = torch.nn.functional.linear(qt_x, qt_w)

        # Output should be in orig_dtype (bfloat16), not FP8
        assert result.dtype == torch.bfloat16

    def test_fp8_linear_different_scales(self):
        """Test FP8 linear with different input/weight scales."""
        if not TensorCoreFP8Layout.supports_fast_matmul():
            pytest.skip("FP8 matmul not supported on this hardware")

        batch, in_features, out_features = 16, 32, 64
        # Use different value ranges to get different scales
        x = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16) * 10
        w = torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16) * 0.1

        qt_x = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")
        qt_w = QuantizedTensor.from_float(w, "TensorCoreFP8Layout")

        # Scales should be different
        assert not torch.allclose(qt_x._params.scale, qt_w._params.scale)

        result = torch.nn.functional.linear(qt_x, qt_w)
        expected = torch.nn.functional.linear(qt_x.dequantize(), qt_w.dequantize())

        assert torch.allclose(result, expected, rtol=0.2, atol=0.2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestFP8ViewOperations:
    """Tests for FP8 view, transpose, reshape operations."""

    def test_fp8_view(self):
        """Test FP8 view preserves quantization."""
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")

        qt_viewed = qt.view(16, 256)

        assert isinstance(qt_viewed, QuantizedTensor)
        assert qt_viewed.shape == (16, 256)
        assert qt_viewed._layout_cls == "TensorCoreFP8Layout"
        # Scale should be shared (same tensor, not cloned)
        assert qt_viewed._params.scale is qt._params.scale

    def test_fp8_view_multiple_shapes(self):
        """Test FP8 view with various shape transformations."""
        x = torch.randn(2, 3, 4, 5, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")

        # Flatten
        qt_flat = qt.view(-1)
        assert qt_flat.shape == (120,)
        assert isinstance(qt_flat, QuantizedTensor)

        # Reshape to 2D
        qt_2d = qt.view(6, 20)
        assert qt_2d.shape == (6, 20)
        assert isinstance(qt_2d, QuantizedTensor)

        # Keep some dims, merge others
        qt_merged = qt.view(2, 12, 5)
        assert qt_merged.shape == (2, 12, 5)
        assert isinstance(qt_merged, QuantizedTensor)

    def test_fp8_transpose(self):
        """Test FP8 transpose preserves quantization."""
        x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")

        qt_t = qt.t()

        assert isinstance(qt_t, QuantizedTensor)
        assert qt_t.shape == (64, 32)
        assert qt_t._layout_cls == "TensorCoreFP8Layout"
        # Scale should be shared
        assert qt_t._params.scale is qt._params.scale

    def test_fp8_reshape(self):
        """Test FP8 reshape preserves quantization."""
        x = torch.randn(4, 8, 16, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")

        qt_reshaped = qt.reshape(32, 16)

        assert isinstance(qt_reshaped, QuantizedTensor)
        assert qt_reshaped.shape == (32, 16)
        assert qt_reshaped._layout_cls == "TensorCoreFP8Layout"

    def test_fp8_reshape_with_minus_one(self):
        """Test FP8 reshape with inferred dimension."""
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")

        qt_reshaped = qt.reshape(-1, 128)

        assert qt_reshaped.shape == (32, 128)
        assert isinstance(qt_reshaped, QuantizedTensor)

    @pytest.mark.parametrize("op_name,input_shape,op_args", [
        ("view", (64, 64), ((16, 256),)),
        ("t", (32, 64), ()),
        ("reshape", (4, 8, 16), ((32, 16),)),
    ])
    def test_fp8_shape_op_dequantize_consistency(self, op_name, input_shape, op_args):
        """Test that shape op then dequantize matches dequantize then shape op."""
        x = torch.randn(*input_shape, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")

        qt_op = getattr(qt, op_name)
        dq_op = getattr(qt.dequantize(), op_name)

        result1 = qt_op(*op_args).dequantize()
        result2 = dq_op(*op_args)

        assert torch.allclose(result1, result2)

    def test_fp8_chained_shape_ops(self):
        """Test chaining multiple shape operations."""
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")

        # Chain: view -> reshape -> view
        qt_chain = qt.view(16, 256).reshape(32, 128).view(4096)

        assert isinstance(qt_chain, QuantizedTensor)
        assert qt_chain.shape == (4096,)
        # Scale should still be shared through all ops
        assert qt_chain._params.scale is qt._params.scale

    def test_fp8_shape_op_then_linear(self):
        """Test shape op followed by linear operation."""
        if not TensorCoreFP8Layout.supports_fast_matmul():
            pytest.skip("FP8 matmul not supported on this hardware")

        batch, seq, features = 4, 8, 64
        out_features = 128
        x = torch.randn(batch, seq, features, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(out_features, features, device="cuda", dtype=torch.bfloat16)

        qt_x = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")
        qt_w = QuantizedTensor.from_float(w, "TensorCoreFP8Layout")

        # Reshape to 2D, do linear, reshape back
        qt_flat = qt_x.reshape(-1, features)
        result_flat = torch.nn.functional.linear(qt_flat, qt_w)
        result = result_flat.reshape(batch, seq, out_features)

        expected = torch.nn.functional.linear(qt_x.dequantize(), qt_w.dequantize())

        assert result.shape == expected.shape
        assert torch.allclose(result, expected, rtol=0.15, atol=0.15)

    def test_fp8_orig_shape_updated(self):
        """Test that orig_shape is updated after shape operations."""
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")

        assert qt._params.orig_shape == (64, 64)

        qt_viewed = qt.view(16, 256)
        assert qt_viewed._params.orig_shape == (16, 256)

        qt_t = qt.t()
        assert qt_t._params.orig_shape == (64, 64)

        qt_reshaped = qt.reshape(8, 8, 64)
        assert qt_reshaped._params.orig_shape == (8, 8, 64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestNVFP4LinearOperations:
    """Tests for NVFP4 linear and matmul operations."""

    def test_nvfp4_linear_both_quantized(self):
        """Test NVFP4 linear with both input and weight quantized."""
        if not TensorCoreNVFP4Layout.supports_fast_matmul():
            pytest.skip("NVFP4 matmul not supported on this hardware (requires SM >= 10.0)")

        batch, in_features, out_features = 32, 64, 128
        x = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16)

        qt_x = QuantizedTensor.from_float(x, "TensorCoreNVFP4Layout")
        qt_w = QuantizedTensor.from_float(w, "TensorCoreNVFP4Layout")
        with comfy_kitchen.use_backend("eager"):
            result = torch.nn.functional.linear(qt_x, qt_w)
        expected = torch.nn.functional.linear(qt_x.dequantize(), qt_w.dequantize())

        # Verify output shape matches original (non-padded)
        assert result.shape == (batch, out_features)
        # NVFP4 has limited precision
        assert torch.allclose(result, expected, rtol=0.3, atol=0.3)

    def test_nvfp4_linear_with_bias(self):
        """Test NVFP4 linear with bias."""
        if not TensorCoreNVFP4Layout.supports_fast_matmul():
            pytest.skip("NVFP4 matmul not supported on this hardware")

        batch, in_features, out_features = 16, 32, 64
        x = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(out_features, device="cuda", dtype=torch.bfloat16)

        qt_x = QuantizedTensor.from_float(x, "TensorCoreNVFP4Layout")
        qt_w = QuantizedTensor.from_float(w, "TensorCoreNVFP4Layout")

        result = torch.nn.functional.linear(qt_x, qt_w, b)
        expected = torch.nn.functional.linear(qt_x.dequantize(), qt_w.dequantize(), b)

        assert result.shape == (batch, out_features)
        assert torch.allclose(result, expected, rtol=0.3, atol=0.3)

    def test_nvfp4_linear_padded_input(self):
        """Test NVFP4 linear with non-16-aligned shapes (requires padding)."""
        if not TensorCoreNVFP4Layout.supports_fast_matmul():
            pytest.skip("NVFP4 matmul not supported on this hardware")

        # Use shapes that require padding (not divisible by 16)
        batch, in_features, out_features = 49, 81, 113
        x = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16)

        qt_x = QuantizedTensor.from_float(x, "TensorCoreNVFP4Layout")
        qt_w = QuantizedTensor.from_float(w, "TensorCoreNVFP4Layout")

        # Verify tensors were padded
        assert qt_x.is_padded
        assert qt_w.is_padded

        result = torch.nn.functional.linear(qt_x, qt_w)

        # Output should be sliced back to original shape
        assert result.shape == (batch, out_features)

    def test_nvfp4_linear_weight_only_quantized(self):
        """Test NVFP4 linear with only weight quantized (fallback path)."""
        batch, in_features, out_features = 16, 32, 64
        x = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16)

        qt_w = QuantizedTensor.from_float(w, "TensorCoreNVFP4Layout")

        # Should work via dequantize fallback
        result = torch.nn.functional.linear(x, qt_w)
        expected = torch.nn.functional.linear(x, qt_w.dequantize())

        assert result.shape == (batch, out_features)
        assert torch.equal(result, expected)

    def test_nvfp4_linear_input_only_quantized(self):
        """Test NVFP4 linear with only input quantized (fallback path)."""
        batch, in_features, out_features = 16, 32, 64
        x = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16)

        qt_x = QuantizedTensor.from_float(x, "TensorCoreNVFP4Layout")

        # Should work via dequantize fallback
        result = torch.nn.functional.linear(qt_x, w)
        expected = torch.nn.functional.linear(qt_x.dequantize(), w)

        assert result.shape == (batch, out_features)
        assert torch.equal(result, expected)

    def test_nvfp4_mm_fallback(self):
        """Test NVFP4 mm falls back to dequantization when b is not transposed."""
        m, k, n = 32, 64, 128
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)

        qt_a = QuantizedTensor.from_float(a, "TensorCoreNVFP4Layout")
        qt_b = QuantizedTensor.from_float(b, "TensorCoreNVFP4Layout")

        # mm with non-transposed b should fall back to dequantization
        # (scaled_mm_nvfp4 expects b.T semantics)
        result = torch.mm(qt_a, qt_b)
        expected = torch.mm(qt_a.dequantize(), qt_b.dequantize())

        assert result.shape == expected.shape
        assert torch.equal(result, expected)

    def test_nvfp4_mm_with_transposed_b(self):
        """Test NVFP4 mm with transposed b (the torch.compile linear decomposition case)."""
        if not TensorCoreNVFP4Layout.supports_fast_matmul():
            pytest.skip("NVFP4 matmul not supported on this hardware (requires SM >= 10.0)")

        m, k, n = 32, 64, 128
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        # b has shape (n, k) - like a linear weight
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)

        qt_a = QuantizedTensor.from_float(a, "TensorCoreNVFP4Layout")
        qt_b = QuantizedTensor.from_float(b, "TensorCoreNVFP4Layout")

        # mm(a, b.t()) is the common torch.compile decomposition of linear
        # This should use the quantized mm path since b.t() has transposed=True
        result = torch.mm(qt_a, qt_b.t())
        expected = torch.mm(qt_a.dequantize(), qt_b.dequantize().t())

        assert result.shape == expected.shape
        # NVFP4 has limited precision
        assert torch.allclose(result, expected, rtol=0.3, atol=0.3)

    def test_nvfp4_addmm_fallback(self):
        """Test NVFP4 addmm falls back to dequantization."""
        m, k, n = 32, 64, 128
        bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)

        qt_a = QuantizedTensor.from_float(a, "TensorCoreNVFP4Layout")
        qt_b = QuantizedTensor.from_float(b, "TensorCoreNVFP4Layout")

        # addmm should fall back to dequantization
        result = torch.addmm(bias, qt_a, qt_b)
        expected = torch.addmm(bias, qt_a.dequantize(), qt_b.dequantize())

        assert result.shape == expected.shape
        assert torch.equal(result, expected)

    def test_nvfp4_linear_output_dtype(self):
        """Test that NVFP4 linear output has correct dtype."""
        if not TensorCoreNVFP4Layout.supports_fast_matmul():
            pytest.skip("NVFP4 matmul not supported on this hardware")

        batch, in_features, out_features = 16, 32, 64
        x = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16)

        qt_x = QuantizedTensor.from_float(x, "TensorCoreNVFP4Layout")
        qt_w = QuantizedTensor.from_float(w, "TensorCoreNVFP4Layout")

        result = torch.nn.functional.linear(qt_x, qt_w)

        # Output should be in orig_dtype (bfloat16)
        assert result.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestNVFP4ShapeOperationsFallback:
    """Tests verifying NVFP4 shape operations fall back to dequantization."""

    def test_nvfp4_view_falls_back(self):
        """Test that view on NVFP4 tensor dequantizes (no native support)."""
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreNVFP4Layout")

        # view should trigger dequantize fallback
        result = qt.view(16, 256)

        # Result should be a regular tensor, not QuantizedTensor
        assert not isinstance(result, QuantizedTensor)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (16, 256)

    def test_nvfp4_transpose_is_noop(self):
        """Test that transpose on NVFP4 tensor is a no-op that tracks transposition."""
        x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreNVFP4Layout")

        # t() should return a QuantizedTensor with transposed flag
        result = qt.t()

        # Result should still be a QuantizedTensor
        assert isinstance(result, QuantizedTensor)
        assert result.shape == (64, 32)
        assert result._params.transposed is True
        # qdata should be the same (no physical transpose)
        assert result._qdata.data_ptr() == qt._qdata.data_ptr()

    def test_nvfp4_double_transpose_restores_state(self):
        """Test that double transpose restores original state."""
        x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreNVFP4Layout")

        result = qt.t().t()

        assert isinstance(result, QuantizedTensor)
        assert result.shape == (32, 64)
        assert result._params.transposed is False

    def test_nvfp4_reshape_falls_back(self):
        """Test that reshape on NVFP4 tensor dequantizes (no native support)."""
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreNVFP4Layout")

        # reshape should trigger dequantize fallback
        result = qt.reshape(32, 128)

        # Result should be a regular tensor, not QuantizedTensor
        assert not isinstance(result, QuantizedTensor)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (32, 128)

    def test_nvfp4_view_values_correct(self):
        """Test that NVFP4 view fallback produces correct values."""
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreNVFP4Layout")

        result = qt.view(-1)
        expected = qt.dequantize().view(-1)

        assert torch.equal(result, expected)

    def test_nvfp4_transpose_dequantize_values_correct(self):
        """Test that dequantizing a transposed NVFP4 tensor produces correct values."""
        x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreNVFP4Layout")

        result = qt.t().dequantize()
        expected = qt.dequantize().t()

        assert torch.equal(result, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestTensorCoreMXFP8Layout:

    def test_quantize_aligned(self):
        x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        qdata, params = TensorCoreMXFP8Layout.quantize(x)

        assert qdata.dtype == torch.float8_e4m3fn
        assert qdata.shape == (128, 256)
        assert params.orig_shape == (128, 256)
        assert params.scale.dtype == torch.float8_e8m0fnu

    def test_quantize_unaligned_pads(self):
        x = torch.randn(129, 130, device="cuda", dtype=torch.bfloat16)
        qdata, params = TensorCoreMXFP8Layout.quantize(x)

        assert qdata.shape == (160, 160)
        assert params.orig_shape == (129, 130)

    def test_dequantize_roundtrip(self):
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16) * 4
        qdata, params = TensorCoreMXFP8Layout.quantize(x)
        dq = TensorCoreMXFP8Layout.dequantize(qdata, params)

        assert dq.dtype == torch.bfloat16
        assert dq.shape == (128, 128)

    def test_get_padded_shape(self):
        assert TensorCoreMXFP8Layout.get_padded_shape((128, 256)) == (128, 256)
        assert TensorCoreMXFP8Layout.get_padded_shape((129, 128)) == (160, 128)
        assert TensorCoreMXFP8Layout.get_padded_shape((100, 100)) == (128, 128)

    def test_get_storage_shape(self):
        assert TensorCoreMXFP8Layout.get_storage_shape((128, 256)) == (128, 256)
        assert TensorCoreMXFP8Layout.get_storage_shape((129, 128)) == (160, 128)

    def test_requires_2d(self):
        x = torch.randn(10, 10, 10, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="2D tensor"):
            TensorCoreMXFP8Layout.quantize(x)

    def test_mxfp8_min_sm_version(self):
        assert TensorCoreMXFP8Layout.MIN_SM_VERSION == (10, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestMXFP8LinearOperations:

    def test_mxfp8_linear_both_quantized(self):
        if not TensorCoreMXFP8Layout.supports_fast_matmul():
            pytest.skip("MXFP8 matmul not supported on this hardware (requires SM >= 10.0)")

        batch, in_features, out_features = 32, 64, 128
        x = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16)

        qt_x = QuantizedTensor.from_float(x, "TensorCoreMXFP8Layout")
        qt_w = QuantizedTensor.from_float(w, "TensorCoreMXFP8Layout")

        result = torch.nn.functional.linear(qt_x, qt_w)
        expected = torch.nn.functional.linear(qt_x.dequantize(), qt_w.dequantize())

        assert result.shape == (batch, out_features)
        assert torch.allclose(result, expected, rtol=0.2, atol=0.2)

    def test_mxfp8_linear_with_bias(self):
        if not TensorCoreMXFP8Layout.supports_fast_matmul():
            pytest.skip("MXFP8 matmul not supported on this hardware")

        batch, in_features, out_features = 16, 32, 64
        x = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(out_features, device="cuda", dtype=torch.bfloat16)

        qt_x = QuantizedTensor.from_float(x, "TensorCoreMXFP8Layout")
        qt_w = QuantizedTensor.from_float(w, "TensorCoreMXFP8Layout")

        result = torch.nn.functional.linear(qt_x, qt_w, b)
        expected = torch.nn.functional.linear(qt_x.dequantize(), qt_w.dequantize(), b)

        assert result.shape == (batch, out_features)
        assert torch.allclose(result, expected, rtol=0.2, atol=0.2)

    def test_mxfp8_linear_padded_input(self):
        if not TensorCoreMXFP8Layout.supports_fast_matmul():
            pytest.skip("MXFP8 matmul not supported on this hardware")

        batch, in_features, out_features = 49, 65, 97
        x = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16)

        qt_x = QuantizedTensor.from_float(x, "TensorCoreMXFP8Layout")
        qt_w = QuantizedTensor.from_float(w, "TensorCoreMXFP8Layout")

        assert qt_x.is_padded
        assert qt_w.is_padded

        result = torch.nn.functional.linear(qt_x, qt_w)

        assert result.shape == (batch, out_features)

    def test_mxfp8_linear_weight_only_quantized(self):
        batch, in_features, out_features = 16, 32, 64
        x = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16)

        qt_w = QuantizedTensor.from_float(w, "TensorCoreMXFP8Layout")

        result = torch.nn.functional.linear(x, qt_w)
        expected = torch.nn.functional.linear(x, qt_w.dequantize())

        assert result.shape == (batch, out_features)
        assert torch.equal(result, expected)

    def test_mxfp8_mm_fallback(self):
        m, k, n = 32, 64, 128
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)

        qt_a = QuantizedTensor.from_float(a, "TensorCoreMXFP8Layout")
        qt_b = QuantizedTensor.from_float(b, "TensorCoreMXFP8Layout")

        result = torch.mm(qt_a, qt_b)
        expected = torch.mm(qt_a.dequantize(), qt_b.dequantize())

        assert result.shape == expected.shape
        assert torch.equal(result, expected)

    def test_mxfp8_mm_with_transposed_b(self):
        if not TensorCoreMXFP8Layout.supports_fast_matmul():
            pytest.skip("MXFP8 matmul not supported on this hardware (requires SM >= 10.0)")

        m, k, n = 32, 64, 128
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)

        qt_a = QuantizedTensor.from_float(a, "TensorCoreMXFP8Layout")
        qt_b = QuantizedTensor.from_float(b, "TensorCoreMXFP8Layout")

        result = torch.mm(qt_a, qt_b.t())
        expected = torch.mm(qt_a.dequantize(), qt_b.dequantize().t())

        assert result.shape == expected.shape
        assert torch.allclose(result, expected, rtol=0.2, atol=0.2)

    def test_mxfp8_linear_output_dtype(self):
        if not TensorCoreMXFP8Layout.supports_fast_matmul():
            pytest.skip("MXFP8 matmul not supported on this hardware")

        batch, in_features, out_features = 16, 32, 64
        x = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16)

        qt_x = QuantizedTensor.from_float(x, "TensorCoreMXFP8Layout")
        qt_w = QuantizedTensor.from_float(w, "TensorCoreMXFP8Layout")

        result = torch.nn.functional.linear(qt_x, qt_w)

        assert result.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestMXFP8ShapeOperationsFallback:

    def test_mxfp8_view_falls_back(self):
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreMXFP8Layout")

        result = qt.view(16, 256)

        assert not isinstance(result, QuantizedTensor)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (16, 256)

    def test_mxfp8_transpose_is_noop(self):
        x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreMXFP8Layout")

        result = qt.t()

        assert isinstance(result, QuantizedTensor)
        assert result.shape == (64, 32)
        assert result._params.transposed is True
        assert result._qdata.data_ptr() == qt._qdata.data_ptr()

    def test_mxfp8_double_transpose_restores_state(self):
        x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreMXFP8Layout")

        result = qt.t().t()

        assert isinstance(result, QuantizedTensor)
        assert result.shape == (32, 64)
        assert result._params.transposed is False

    def test_mxfp8_reshape_falls_back(self):
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreMXFP8Layout")

        result = qt.reshape(32, 128)

        assert not isinstance(result, QuantizedTensor)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (32, 128)

    def test_mxfp8_transpose_dequantize_values_correct(self):
        x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "TensorCoreMXFP8Layout")

        result = qt.t().dequantize()
        expected = qt.dequantize().t()

        assert torch.equal(result, expected)


