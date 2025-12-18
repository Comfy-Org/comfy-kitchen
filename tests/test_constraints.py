import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.constraints import (
    DivisibleBy,
    ExactDims,
    FunctionConstraints,
    MinDims,
    ParamConstraint,
    validate_function_call,
    validate_param,
)
from comfy_kitchen.exceptions import NoCapableBackendError


class TestShapeRules:
    """Tests for shape rule validation."""

    def test_divisible_by_passes(self):
        tensor = torch.randn(16, 32)
        rule = DivisibleBy(dim=1, factor=16)
        assert rule.check(tensor) is True

    def test_divisible_by_fails(self):
        tensor = torch.randn(16, 30)
        rule = DivisibleBy(dim=1, factor=16)
        assert rule.check(tensor) is False

    def test_divisible_by_wrong_dim(self):
        tensor = torch.randn(16)  # 1D tensor
        rule = DivisibleBy(dim=1, factor=16)
        assert rule.check(tensor) is False  # dim 1 doesn't exist

    def test_min_dims_passes(self):
        tensor = torch.randn(2, 4, 8, 16)
        rule = MinDims(ndim=4)
        assert rule.check(tensor) is True

    def test_min_dims_fails(self):
        tensor = torch.randn(2, 4)
        rule = MinDims(ndim=4)
        assert rule.check(tensor) is False

    def test_exact_dims_passes(self):
        tensor = torch.randn(2, 4, 8, 16)
        rule = ExactDims(ndim=4)
        assert rule.check(tensor) is True

    def test_exact_dims_fails(self):
        tensor = torch.randn(2, 4, 8)
        rule = ExactDims(ndim=4)
        assert rule.check(tensor) is False

    def test_shape_rule_describe(self):
        """Test that describe() returns human-readable strings."""
        assert "16" in DivisibleBy(dim=1, factor=16).describe()
        assert "4" in MinDims(ndim=4).describe()
        assert "2" in ExactDims(ndim=2).describe()


class TestParamConstraint:
    """Tests for parameter constraint validation."""

    def test_dtype_passes(self):
        constraint = ParamConstraint(dtypes=frozenset({torch.float32, torch.float16}))
        tensor = torch.randn(10, dtype=torch.float32)
        assert constraint.check_dtype(tensor) is True

    def test_dtype_fails(self):
        constraint = ParamConstraint(dtypes=frozenset({torch.float32}))
        tensor = torch.randn(10, dtype=torch.float16)
        assert constraint.check_dtype(tensor) is False

    def test_dtype_none_allows_any(self):
        constraint = ParamConstraint(dtypes=None)
        tensor = torch.randn(10, dtype=torch.bfloat16)
        assert constraint.check_dtype(tensor) is True

    def test_dtype_value_not_tensor(self):
        """Test checking dtype value directly (for output_type params)."""
        constraint = ParamConstraint(dtypes=frozenset({torch.float32, torch.bfloat16}))
        assert constraint.check_dtype(torch.float32) is True
        assert constraint.check_dtype(torch.float16) is False

    def test_device_passes(self, device):
        default_devices = frozenset({"cuda", "cpu"})
        constraint = ParamConstraint(devices=None)  # Inherit default
        tensor = torch.randn(10, device=device)
        assert constraint.check_device(tensor, default_devices) is True

    def test_device_fails(self, device):
        if device != "cuda":
            pytest.skip("Need CUDA to test device failure")
        constraint = ParamConstraint(devices=frozenset({"cpu"}))  # CPU only
        tensor = torch.randn(10, device="cuda")
        assert constraint.check_device(tensor, frozenset({"cuda", "cpu"})) is False

    def test_shape_rules_pass(self):
        constraint = ParamConstraint(
            shape_rules=(ExactDims(2), DivisibleBy(dim=1, factor=16))
        )
        tensor = torch.randn(8, 32)
        assert constraint.check_shape(tensor) is True

    def test_shape_rules_fail(self):
        constraint = ParamConstraint(
            shape_rules=(ExactDims(2), DivisibleBy(dim=1, factor=16))
        )
        tensor = torch.randn(8, 30)  # Not divisible by 16
        assert constraint.check_shape(tensor) is False


class TestFunctionConstraints:
    """Tests for function-level constraint validation."""

    def test_validate_param_success(self):
        constraint = ParamConstraint(
            dtypes=frozenset({torch.float32}),
        )
        tensor = torch.randn(10, dtype=torch.float32)
        result = validate_param("x", tensor, constraint, frozenset({"cuda", "cpu"}))
        assert result.success is True

    def test_validate_param_dtype_failure(self):
        constraint = ParamConstraint(dtypes=frozenset({torch.float32}))
        tensor = torch.randn(10, dtype=torch.float16)
        result = validate_param("x", tensor, constraint, frozenset({"cuda", "cpu"}))
        assert result.success is False
        assert result.failed_param == "x"
        assert "dtype" in result.failure_reason

    def test_validate_param_none_passes(self):
        """None values (optional params) should always pass."""
        constraint = ParamConstraint(dtypes=frozenset({torch.float32}))
        result = validate_param("bias", None, constraint, frozenset({"cuda", "cpu"}))
        assert result.success is True

    def test_validate_function_call_success(self, device):
        constraints = FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=frozenset({torch.float32})),
                "scale": ParamConstraint(dtypes=frozenset({torch.float32})),
            },
            default_devices=frozenset({"cuda", "cpu"}),
        )
        kwargs = {
            "x": torch.randn(10, dtype=torch.float32, device=device),
            "scale": torch.tensor([1.0], dtype=torch.float32, device=device),
        }
        result = validate_function_call(constraints, kwargs)
        assert result.success is True

    def test_validate_function_call_failure(self, device):
        constraints = FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=frozenset({torch.float32})),
            },
            default_devices=frozenset({"cuda", "cpu"}),
        )
        kwargs = {
            "x": torch.randn(10, dtype=torch.float16, device=device),  # Wrong dtype
        }
        result = validate_function_call(constraints, kwargs)
        assert result.success is False
        assert result.failed_param == "x"

    def test_compute_capability_check(self):
        """Test hardware capability validation."""
        constraints = FunctionConstraints(
            params={},
            min_compute_capability=(9, 0),  # Hopper+
        )
        # With low compute capability
        result = validate_function_call(constraints, {}, compute_capability=(8, 0))
        assert result.success is False
        assert "__hardware__" in result.failed_param

        # With high enough compute capability
        result = validate_function_call(constraints, {}, compute_capability=(9, 0))
        assert result.success is True

    def test_compute_capability_none_required(self):
        """Test that CUDA requirement fails when no CUDA available."""
        constraints = FunctionConstraints(
            params={},
            min_compute_capability=(8, 0),
        )
        result = validate_function_call(constraints, {}, compute_capability=None)
        assert result.success is False


class TestRegistryConstraintValidation:
    """Tests for registry-level constraint validation."""

    def test_validate_backend_for_call(self, device):
        """Test validating a specific backend can handle a call."""
        kwargs = {
            "x": torch.randn(10, dtype=torch.float32, device=device),
            "scale": torch.tensor([1.0], dtype=torch.float32, device=device),
            "output_type": torch.float8_e4m3fn,
        }
        result = ck.registry.validate_backend_for_call(
            "eager", "quantize_per_tensor_fp8", kwargs
        )
        assert result.success is True

    def test_validate_backend_wrong_dtype(self, device):
        """Test validation fails for wrong dtype."""
        kwargs = {
            "x": torch.randint(0, 10, (10,), dtype=torch.int32, device=device),  # Wrong dtype
            "scale": torch.tensor([1.0], dtype=torch.float32, device=device),
            "output_type": torch.float8_e4m3fn,
        }
        result = ck.registry.validate_backend_for_call(
            "eager", "quantize_per_tensor_fp8", kwargs
        )
        assert result.success is False

    def test_get_capable_backend_fallback(self, device):
        """Test automatic fallback when first backend can't handle the call."""
        # Create kwargs that would work on eager but may require specific setup on CUDA
        kwargs = {
            "x": torch.randn(10, dtype=torch.float32, device=device),
            "scale": torch.tensor([1.0], dtype=torch.float32, device=device),
            "output_type": torch.float8_e4m3fn,
        }
        backend = ck.registry.get_capable_backend("quantize_per_tensor_fp8", kwargs)
        assert backend is not None
        assert backend in ["cuda", "triton", "eager"]

    def test_get_capable_backend_no_match(self, device):
        """Test NoCapableBackendError when no backend can handle the call."""
        # Pass completely wrong dtypes
        kwargs = {
            "x": torch.randint(0, 10, (10,), dtype=torch.int64, device=device),  # int64
            "scale": torch.tensor([1.0], dtype=torch.float32, device=device),
            "output_type": torch.float8_e4m3fn,
        }
        with pytest.raises(NoCapableBackendError) as exc_info:
            ck.registry.get_capable_backend("quantize_per_tensor_fp8", kwargs)

        assert exc_info.value.func_name == "quantize_per_tensor_fp8"
        assert len(exc_info.value.failures) > 0

    def test_get_implementation_with_kwargs(self, device):
        """Test get_implementation with constraint validation."""
        kwargs = {
            "x": torch.randn(10, dtype=torch.float32, device=device),
            "scale": torch.tensor([1.0], dtype=torch.float32, device=device),
            "output_type": torch.float8_e4m3fn,
        }
        impl = ck.registry.get_implementation("quantize_per_tensor_fp8", kwargs=kwargs)
        assert callable(impl)

    def test_constraints_stored_correctly(self):
        """Test that constraints are stored in the registry."""
        # Check that eager backend has constraints registered
        constraints = ck.registry.get_constraints("eager", "quantize_per_tensor_fp8")
        assert constraints is not None
        assert "x" in constraints.params
        assert torch.float32 in constraints.params["x"].dtypes


class TestIntegrationWithBackends:
    """Integration tests for constraint validation with actual backends."""

    def test_eager_allows_cpu_tensors(self):
        """Test that eager backend accepts CPU tensors."""
        x = torch.randn(10, dtype=torch.float32, device="cpu")
        scale = torch.tensor([1.0], dtype=torch.float32, device="cpu")

        result = ck.registry.validate_backend_for_call(
            "eager",
            "quantize_per_tensor_fp8",
            {"x": x, "scale": scale, "output_type": torch.float8_e4m3fn},
        )
        assert result.success is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_rejects_cpu_tensors(self):
        """Test that CUDA backend rejects CPU tensors."""
        x = torch.randn(10, dtype=torch.float32, device="cpu")
        scale = torch.tensor([1.0], dtype=torch.float32, device="cpu")

        backends = ck.list_backends()
        if not backends.get("cuda", {}).get("available", False):
            pytest.skip("CUDA backend not available")

        result = ck.registry.validate_backend_for_call(
            "cuda",
            "quantize_per_tensor_fp8",
            {"x": x, "scale": scale, "output_type": torch.float8_e4m3fn},
        )
        assert result.success is False
        assert "device" in result.failure_reason.lower()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_automatic_fallback_cpu_to_eager(self):
        """Test that CPU tensors automatically fall back to eager."""
        x = torch.randn(10, dtype=torch.float32, device="cpu")
        scale = torch.tensor([1.0], dtype=torch.float32, device="cpu")

        backend = ck.registry.get_capable_backend(
            "quantize_per_tensor_fp8",
            {"x": x, "scale": scale, "output_type": torch.float8_e4m3fn},
        )
        # Should fall back to eager since CPU tensors
        assert backend == "eager"

    def test_shape_constraint_validation(self, device):
        """Test shape constraints are validated correctly."""
        # NVFP4 quantize requires 2D input
        x_3d = torch.randn(2, 16, 32, dtype=torch.float32, device=device)
        scale = torch.tensor([1.0], dtype=torch.float32, device=device)

        result = ck.registry.validate_backend_for_call(
            "eager",
            "quantize_nvfp4",
            {"x": x_3d, "per_tensor_scale": scale},
        )
        assert result.success is False
        assert "shape" in result.failure_reason.lower() or "2D" in result.failure_reason

    def test_valid_shape_passes(self, device):
        """Test valid shapes pass validation."""
        x_2d = torch.randn(16, 32, dtype=torch.float32, device=device)
        scale = torch.tensor([1.0], dtype=torch.float32, device=device)

        result = ck.registry.validate_backend_for_call(
            "eager",
            "quantize_nvfp4",
            {"x": x_2d, "per_tensor_scale": scale},
        )
        assert result.success is True
