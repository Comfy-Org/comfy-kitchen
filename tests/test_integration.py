"""Integration tests for CUDA graph capture and torch.compile with quantized models."""
import pytest
import torch

from comfy_kitchen.tensor import (
    QuantizedTensor,
    TensorCoreFP8Layout,
    TensorCoreNVFP4Layout,
    TensorWiseINT8Layout,
)

from .conftest import assert_values_close, cuda_extension_runtime_available


class DummyQuantizedModel(torch.nn.Module):
    """Simple model for testing CUDA graph capture and torch.compile with quantized weights.

    This model demonstrates the pattern of:
    1. Pre-quantized weights stored as QuantizedTensors
    2. Runtime quantization of inputs
    3. Quantized matmul operations
    4. Activation functions between layers
    """

    def __init__(
        self,
        in_features: int = 64,
        hidden: int = 128,
        out_features: int = 32,
        layout_cls="TensorCoreNVFP4Layout",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.layout_cls = layout_cls
        self.dtype = dtype

        # Pre-quantized weights
        w1 = torch.randn(hidden, in_features, device=device, dtype=dtype)
        w2 = torch.randn(out_features, hidden, device=device, dtype=dtype)

        self.w1 = QuantizedTensor.from_float(w1, layout_cls)
        self.w2 = QuantizedTensor.from_float(w2, layout_cls)

        self.register_buffer("input_scale", torch.tensor(48.0, device=device, dtype=torch.float32))
        self.register_buffer("hidden_scale", torch.tensor(96.0, device=device, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize input -> matmul -> activation -> quantize -> matmul
        x_q = QuantizedTensor.from_float(x, self.layout_cls, scale=self.input_scale)
        h = torch.nn.functional.linear(x_q, self.w1)
        h = torch.nn.functional.gelu(h)
        h_q = QuantizedTensor.from_float(h, self.layout_cls, scale=self.hidden_scale)
        return torch.nn.functional.linear(h_q, self.w2)


# Layout configurations for parametrized tests
LAYOUT_CONFIGS = [
    pytest.param(TensorCoreNVFP4Layout, id="nvfp4"),
    pytest.param(TensorCoreFP8Layout, id="fp8"),
    pytest.param(TensorWiseINT8Layout, id="int8"),
]


@pytest.mark.skipif(not cuda_extension_runtime_available(), reason="CUDA extension backend required")
class TestQuantizedCUDAGraph:
    """Tests for CUDA graph capture with quantized models."""

    @pytest.fixture
    def model(self, request):
        """Create a DummyQuantizedModel for testing."""
        layout_cls = request.param
        if not layout_cls.supports_fast_matmul():
            reqs = layout_cls.get_requirements()
            pytest.skip(f"{layout_cls.__name__} matmul not supported (requires SM >= {reqs['min_sm_version']}, have {reqs['current_sm_version']})")
        return DummyQuantizedModel(
            in_features=64, hidden=128, out_features=32, layout_cls=layout_cls.__name__, device="cuda"
        )

    @pytest.fixture
    def static_input(self):
        """Create static input tensor for CUDA graph capture."""
        return torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)

    @pytest.mark.parametrize("model", LAYOUT_CONFIGS, indirect=True)
    def test_cuda_graph_capture(self, model, static_input):
        """Test CUDA graph capture with quantize + linear + activation."""
        stream = torch.cuda.Stream()

        # Warmup on side stream
        with torch.cuda.stream(stream):
            _ = model(static_input)
        stream.synchronize()

        # Capture graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph):
            static_output = model(static_input)
        stream.synchronize()

        # Verify capture succeeded
        assert static_output is not None
        assert static_output.shape == (16, 32)
        assert static_output.dtype == torch.bfloat16

    @pytest.mark.parametrize("model", LAYOUT_CONFIGS, indirect=True)
    def test_cuda_graph_replay(self, model, static_input):
        """Test CUDA graph replay produces consistent results."""
        stream = torch.cuda.Stream()

        # Warmup
        with torch.cuda.stream(stream):
            _ = model(static_input)
        stream.synchronize()

        # Capture graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph):
            static_output = model(static_input)
        stream.synchronize()

        # Replay multiple times and verify consistency
        results = []
        for _ in range(3):
            with torch.cuda.stream(stream):
                graph.replay()
            stream.synchronize()
            results.append(static_output.clone())

        # All replays should produce identical results
        for i in range(1, len(results)):
            assert torch.equal(results[0], results[i]), f"Replay {i} differs from replay 0"

    @pytest.mark.parametrize("model", LAYOUT_CONFIGS, indirect=True)
    def test_cuda_graph_with_new_input(self, model, static_input):
        """Test CUDA graph with updated input data."""
        stream = torch.cuda.Stream()

        # Warmup
        with torch.cuda.stream(stream):
            _ = model(static_input)
        stream.synchronize()

        # Capture graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph):
            static_output = model(static_input)
        stream.synchronize()

        # Get initial output
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        output1 = static_output.clone()

        # Update static input and replay
        # We use a large enough difference to avoid numerical issues with small changes
        new_data = torch.randn_like(static_input) * 10.0
        static_input.copy_(new_data)

        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        output2 = static_output.clone()

        # Outputs should differ since input changed
        # Use a small tolerance for comparison to avoid flaky failures due to identical rounded values
        assert not torch.allclose(output1, output2, rtol=1e-5, atol=1e-5), "Outputs should differ with different inputs"


@pytest.mark.skipif(not cuda_extension_runtime_available(), reason="CUDA extension backend required")
class TestQuantizedCompile:
    """Tests for torch.compile with quantized models."""

    @pytest.fixture
    def model(self, request):
        """Create a DummyQuantizedModel for testing."""
        layout_cls = request.param
        if not layout_cls.supports_fast_matmul():
            reqs = layout_cls.get_requirements()
            pytest.skip(f"{layout_cls.__name__} matmul not supported (requires SM >= {reqs['min_sm_version']}, have {reqs['current_sm_version']})")
        return DummyQuantizedModel(
            in_features=64, hidden=128, out_features=32, layout_cls=layout_cls.__name__, device="cuda"
        )

    @pytest.mark.parametrize("model", LAYOUT_CONFIGS, indirect=True)
    def test_compile_model(self, model):
        """Test torch.compile on quantized model."""
        x = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)

        # Get reference output
        with torch.no_grad():
            ref_output = model(x)

        # Compile and run
        compiled_model = torch.compile(model)
        with torch.no_grad():
            compiled_output = compiled_model(x)

        # Verify outputs match
        assert compiled_output.shape == ref_output.shape
        assert compiled_output.dtype == ref_output.dtype
        # INT8 has slightly lower precision due to quantization noise
        is_int8 = "INT8" in model.layout_cls
        tol = 0.2 if is_int8 else 1e-3
        max_mismatch_ratio = 0.01 if is_int8 else 0.0
        assert_values_close(
            compiled_output,
            ref_output,
            rtol=tol,
            atol=tol,
            name="compiled_output",
            max_mismatch_ratio=max_mismatch_ratio,
        )

    @pytest.mark.parametrize("model", LAYOUT_CONFIGS, indirect=True)
    def test_compile_model_multiple_runs(self, model):
        """Test compiled model produces consistent results across multiple runs."""
        x = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)

        compiled_model = torch.compile(model)

        # Run multiple times
        results = []
        with torch.no_grad():
            for _ in range(3):
                results.append(compiled_model(x).clone())

        # All results should be identical
        for i in range(1, len(results)):
            assert torch.equal(results[0], results[i])
