import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.float_utils import F4_E2M1_MAX, F8_E4M3_MAX


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDAGraphCompatibility:
    """Test CUDA graph capture compatibility with comfy_kitchen CUDA backend.

    These tests verify the DLPack stream workaround works correctly during
    CUDA graph capture on non-default streams.
    See: https://github.com/pytorch/pytorch/pull/163242
    """

    def test_quantize_fp8_cuda_graph(self):
        """Test FP8 quantization inside CUDA graph capture."""
        try:
            from comfy_kitchen.backends import cuda as cuda_backend
            if not cuda_backend._EXT_AVAILABLE:
                pytest.skip("CUDA extension not available")
        except ImportError:
            pytest.skip("CUDA backend not available")

        device = "cuda"
        dtype = torch.bfloat16

        # Static tensors for graph capture
        static_input = torch.randn(128, 1024, device=device, dtype=dtype)
        scale = torch.tensor([1.0], device=device, dtype=torch.float32)

        # Use a side stream (non-default) - this is where the bug manifests
        stream = torch.cuda.Stream()

        # Warmup on the side stream
        with torch.cuda.stream(stream), ck.use_backend("cuda"):
            ck.quantize_per_tensor_fp8(static_input, scale)
        stream.synchronize()

        # Capture graph on the side stream
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph), ck.use_backend("cuda"):
            static_output = ck.quantize_per_tensor_fp8(static_input, scale)
        stream.synchronize()

        # Replay graph
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()

        # Verify output is valid FP8
        assert static_output.dtype == torch.float8_e4m3fn
        assert static_output.shape == static_input.shape
        assert not torch.isnan(static_output.view(torch.uint8).float()).any()

    def test_dequantize_fp8_cuda_graph(self):
        """Test FP8 dequantization inside CUDA graph capture."""
        try:
            from comfy_kitchen.backends import cuda as cuda_backend
            if not cuda_backend._EXT_AVAILABLE:
                pytest.skip("CUDA extension not available")
        except ImportError:
            pytest.skip("CUDA backend not available")

        device = "cuda"
        dtype = torch.bfloat16

        # Create FP8 input
        x_fp16 = torch.randn(128, 1024, device=device, dtype=torch.float16)
        scale = torch.tensor([1.0], device=device, dtype=torch.float32)
        with ck.use_backend("cuda"):
            static_input = ck.quantize_per_tensor_fp8(x_fp16, scale)

        # Use a side stream
        stream = torch.cuda.Stream()

        # Warmup
        with torch.cuda.stream(stream), ck.use_backend("cuda"):
            ck.dequantize_per_tensor_fp8(static_input, scale, output_type=dtype)
        stream.synchronize()

        # Capture graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph), ck.use_backend("cuda"):
            static_output = ck.dequantize_per_tensor_fp8(static_input, scale, output_type=dtype)
        stream.synchronize()

        # Replay
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()

        # Verify
        assert static_output.dtype == dtype
        assert static_output.shape == static_input.shape
        assert not torch.isnan(static_output).any()

    def test_quantize_nvfp4_cuda_graph(self):
        """Test NVFP4 quantization inside CUDA graph capture."""
        try:
            from comfy_kitchen.backends import cuda as cuda_backend
            if not cuda_backend._EXT_AVAILABLE:
                pytest.skip("CUDA extension not available")
        except ImportError:
            pytest.skip("CUDA backend not available")

        device = "cuda"
        dtype = torch.bfloat16

        # Static tensors for graph capture (must be 16-aligned for NVFP4)
        static_input = torch.randn(128, 1024, device=device, dtype=dtype)
        scale = (torch.amax(static_input.abs()) / (F8_E4M3_MAX * F4_E2M1_MAX)).to(torch.float32)

        stream = torch.cuda.Stream()

        # Warmup on side stream
        with torch.cuda.stream(stream), ck.use_backend("cuda"):
            ck.quantize_nvfp4(static_input, scale)
        stream.synchronize()

        # Capture graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph), ck.use_backend("cuda"):
            static_qdata, static_scales = ck.quantize_nvfp4(static_input, scale)
        stream.synchronize()

        # Replay graph
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()

        # Verify outputs are valid
        assert static_qdata.dtype == torch.uint8
        assert static_qdata.shape == (128, 512)  # packed: cols / 2
        assert static_scales.dtype == torch.float8_e4m3fn

    def test_scaled_mm_nvfp4_cuda_graph(self):
        """Test NVFP4 scaled matmul inside CUDA graph capture."""
        try:
            from comfy_kitchen.backends import cuda as cuda_backend
            if not cuda_backend._EXT_AVAILABLE:
                pytest.skip("CUDA extension not available")
        except ImportError:
            pytest.skip("CUDA backend not available")

        # Check SM version for NVFP4 support
        cap = torch.cuda.get_device_capability()
        if cap < (10, 0):
            pytest.skip("NVFP4 matmul requires SM >= 10.0 (Blackwell)")

        device = "cuda"
        dtype = torch.bfloat16
        m, k, n = 128, 256, 512

        # Create and quantize inputs
        a = torch.randn(m, k, device=device, dtype=dtype)
        b = torch.randn(n, k, device=device, dtype=dtype)

        scale_a = (torch.amax(a.abs()) / (F8_E4M3_MAX * F4_E2M1_MAX)).to(torch.float32)
        scale_b = (torch.amax(b.abs()) / (F8_E4M3_MAX * F4_E2M1_MAX)).to(torch.float32)

        with ck.use_backend("cuda"):
            qa, sa = ck.quantize_nvfp4(a, scale_a)
            qb, sb = ck.quantize_nvfp4(b, scale_b)

        stream = torch.cuda.Stream()

        # Warmup
        with torch.cuda.stream(stream), ck.use_backend("cuda"):
            ck.scaled_mm_nvfp4(qa, qb, scale_a, scale_b, sa, sb, out_dtype=dtype)
        stream.synchronize()

        # Capture graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph), ck.use_backend("cuda"):
            static_output = ck.scaled_mm_nvfp4(
                qa, qb, scale_a, scale_b, sa, sb, out_dtype=dtype
            )
        stream.synchronize()

        # Replay graph
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()

        # Verify output
        assert static_output.dtype == dtype
        assert static_output.shape == (m, n)
        assert not torch.isnan(static_output).any()
