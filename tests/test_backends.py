import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.exceptions import (
    BackendNotFoundError,
    BackendNotImplementedError,
    NoCapableBackendError,
)


class TestBackendSystem:
    def test_list_backends(self):
        import comfy_kitchen as ck

        backends = ck.list_backends()

        assert isinstance(backends, dict)
        assert "eager" in backends
        assert "cuda" in backends
        assert "triton" in backends

        # Eager backend should always be available
        assert backends["eager"]["available"] is True
        assert "capabilities" in backends["eager"]

    def test_backend_priority(self):
        import comfy_kitchen as ck

        ck.set_backend_priority(["eager", "cuda", "triton"])
        ck.set_backend_priority(["cuda", "triton", "eager"])

    def test_disable_enable_backend(self):
        import comfy_kitchen as ck

        # Disable triton
        ck.disable_backend("triton")
        backends = ck.list_backends()
        assert backends["triton"]["disabled"] is True

        # Re-enable
        ck.enable_backend("triton")
        backends = ck.list_backends()
        assert backends["triton"]["disabled"] is False

    def test_backend_context_manager_override(self, small_tensor):
        """Test that use_backend context manager correctly overrides backend selection."""
        import comfy_kitchen as ck

        scale = torch.tensor([1.0], device=small_tensor.device)

        with ck.use_backend("eager"):
            result = ck.quantize_per_tensor_fp8(small_tensor, scale)

        assert isinstance(result, torch.Tensor)
        assert result.shape == small_tensor.shape


class TestBackendExceptions:
    """Tests for backend exception handling."""

    def test_backend_not_found_error_unregistered(self):
        """Test BackendNotFoundError when requesting unregistered backend."""
        with pytest.raises(BackendNotFoundError, match="not_a_real_backend"), \
             ck.use_backend("not_a_real_backend"):
            pass

    def test_backend_not_found_error_disabled(self):
        """Test BackendNotFoundError when backend is disabled."""
        # Disable eager backend temporarily
        ck.disable_backend("eager")
        try:
            with pytest.raises(BackendNotFoundError, match="disabled"), \
                 ck.use_backend("eager"):
                pass
        finally:
            # Re-enable for other tests
            ck.enable_backend("eager")

    def test_backend_not_implemented_error(self):
        """Test BackendNotImplementedError when backend doesn't implement function."""
        # Request a function that doesn't exist from eager backend
        with pytest.raises(BackendNotImplementedError, match="nonexistent_function"):
            ck.registry.get_implementation("nonexistent_function", backend="eager")

    def test_no_capable_backend_error(self):
        """Test NoCapableBackendError when no backend implements function."""
        with pytest.raises(NoCapableBackendError, match="totally_fake_function"):
            ck.registry.get_implementation("totally_fake_function")

    def test_backend_not_found_error_attributes(self):
        """Test BackendNotFoundError has correct attributes."""
        try:
            with ck.use_backend("fake_backend"):
                pass
        except BackendNotFoundError as e:
            assert e.backend_name == "fake_backend"

    def test_backend_not_implemented_error_attributes(self):
        """Test BackendNotImplementedError has correct attributes."""
        try:
            ck.registry.get_implementation("fake_func", backend="eager")
        except BackendNotImplementedError as e:
            assert e.backend_name == "eager"
            assert e.func_name == "fake_func"

    def test_no_capable_backend_error_attributes(self):
        """Test NoCapableBackendError has correct attributes."""
        try:
            ck.registry.get_implementation("fake_function_xyz")
        except NoCapableBackendError as e:
            assert e.func_name == "fake_function_xyz"
            assert isinstance(e.failures, dict)
