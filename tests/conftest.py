import pytest
import torch

import comfy_kitchen as ck


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: mark test as requiring CUDA")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "cupy: mark test as requiring CuPy")


@pytest.fixture(scope="session")
def cuda_available():
    return torch.cuda.is_available()


@pytest.fixture
def seed():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    yield


@pytest.fixture
def device(cuda_available):
    return "cuda" if cuda_available else "cpu"


@pytest.fixture
def small_tensor(cuda_available):
    device = "cuda" if cuda_available else "cpu"
    return torch.randn(128, 128, device=device, dtype=torch.float32)


# =============================================================================
# Constraint-Aware Test Utilities
# =============================================================================


def get_capable_backends(func_name: str, device: str | None = None) -> list[str]:
    """Get list of backends capable of running a function on a given device.

    Args:
        func_name: Function name to check
        device: Device type ("cuda", "cpu", or None for any)

    Returns:
        List of backend names that can handle the function
    """
    capable = []
    backends = ck.list_backends()

    for backend_name in ["cuda", "triton", "eager"]:
        if not backends.get(backend_name, {}).get("available", False):
            continue

        if func_name not in backends[backend_name].get("capabilities", []):
            continue

        # Check device constraint if specified
        if device is not None:
            constraints = ck.registry.get_constraints(backend_name, func_name)
            if constraints is not None and device not in constraints.default_devices:
                continue

        capable.append(backend_name)

    return capable


def get_supported_devices(func_name: str) -> set[str]:
    devices = set()
    backends = ck.list_backends()

    for backend_name in ["cuda", "triton", "eager"]:
        if not backends.get(backend_name, {}).get("available", False):
            continue

        if func_name not in backends[backend_name].get("capabilities", []):
            continue

        constraints = ck.registry.get_constraints(backend_name, func_name)
        if constraints is not None:
            devices.update(constraints.default_devices)

    return devices


class ConstraintAwareTestInputs:
    """Generate valid test inputs based on constraints.

    Usage:
        inputs = ConstraintAwareTestInputs("quantize_per_tensor_fp8", "eager")
        x = inputs.tensor("x", shape=(100, 100))
        scale = inputs.scalar("scale")
    """

    def __init__(self, func_name: str, backend_name: str, device: str | None = None):
        self.func_name = func_name
        self.backend_name = backend_name
        self.constraints = ck.registry.get_constraints(backend_name, func_name)

        # Determine device
        if device is not None:
            self.device = device
        elif self.constraints is not None:
            # Pick first available device from constraints
            if "cuda" in self.constraints.default_devices and torch.cuda.is_available():
                self.device = "cuda"
            elif "cpu" in self.constraints.default_devices:
                self.device = "cpu"
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_valid_dtype(self, param_name: str, prefer: torch.dtype = None) -> torch.dtype:
        """Get a valid dtype for a parameter."""
        if self.constraints is None:
            return prefer or torch.float32

        param_constraint = self.constraints.params.get(param_name)
        if param_constraint is None or param_constraint.dtypes is None:
            return prefer or torch.float32

        if prefer is not None and prefer in param_constraint.dtypes:
            return prefer

        # Return first valid dtype
        return next(iter(param_constraint.dtypes))

    def tensor(
        self,
        param_name: str,
        shape: tuple,
        dtype: torch.dtype = None,
        fill: str = "randn",
    ) -> torch.Tensor:
        """Generate a tensor that satisfies constraints for a parameter.

        Args:
            param_name: Parameter name
            shape: Tensor shape
            dtype: Preferred dtype (will use constraint-valid dtype if not specified)
            fill: How to fill tensor ("randn", "rand", "ones", "zeros")

        Returns:
            Tensor on correct device with valid dtype
        """
        actual_dtype = dtype or self.get_valid_dtype(param_name, dtype)

        # Generate tensor
        if fill == "randn":
            if actual_dtype in (torch.float32, torch.float16, torch.bfloat16):
                tensor = torch.randn(shape, dtype=actual_dtype, device=self.device)
            else:
                # For non-float types, generate in float32 then convert
                tensor = torch.randn(shape, dtype=torch.float32, device=self.device)
                tensor = tensor.to(actual_dtype)
        elif fill == "rand":
            tensor = torch.rand(shape, dtype=torch.float32, device=self.device)
            if actual_dtype != torch.float32:
                tensor = tensor.to(actual_dtype)
        elif fill == "ones":
            tensor = torch.ones(shape, dtype=actual_dtype, device=self.device)
        elif fill == "zeros":
            tensor = torch.zeros(shape, dtype=actual_dtype, device=self.device)
        else:
            raise ValueError(f"Unknown fill type: {fill}")

        return tensor

    def scalar(self, param_name: str, value: float = 1.0) -> torch.Tensor:
        """Generate a scalar tensor."""
        dtype = self.get_valid_dtype(param_name)
        return torch.tensor([value], dtype=dtype, device=self.device)


def assert_values_close(values, ref_values, rtol, atol, name="values", max_mismatch_ratio=0.0):
    delta_idx = torch.isclose(values, ref_values, rtol=rtol, atol=atol)

    if not torch.all(delta_idx):
        n = 10
        failing_indices = torch.nonzero(~delta_idx, as_tuple=False)
        num_failures = len(failing_indices)
        total_elements = values.numel()
        mismatch_ratio = num_failures / total_elements

        # If max_mismatch_ratio is set and we're below it, pass with a warning
        if max_mismatch_ratio > 0 and mismatch_ratio <= max_mismatch_ratio:
            print(f"Warning: {num_failures} / {total_elements} ({mismatch_ratio*100:.4f}%) {name} "
                  f"differ (rtol={rtol}, atol={atol}), but within allowed ratio ({max_mismatch_ratio*100}%)")
            return

        print(f"Failed: \n{num_failures} {name} are not close.")

        for i, idx in enumerate(failing_indices[:n]):
            idx_tuple = tuple(idx.tolist())
            val = values[idx_tuple].item()
            ref_val = ref_values[idx_tuple].item()
            diff = abs(val - ref_val)
            print(
                f"  [{i}] Index {idx_tuple}: got {val:.6f}, expected {ref_val:.6f}, diff={diff:.6f}"
            )

        raise ValueError(
            f"{num_failures} {name} are not close (rtol={rtol}, atol={atol})"
        )
