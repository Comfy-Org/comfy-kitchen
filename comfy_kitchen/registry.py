import logging
import threading
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

from .constraints import (
    INT8_MAX_COMPUTE_CAPABILITY,
    FunctionConstraints,
    ValidationResult,
    cuda_compute_capabilities,
    guard_compute_capability,
    validate_function_call,
)
from .exceptions import (
    BackendNotFoundError,
    BackendNotImplementedError,
    NoCapableBackendError,
)

logger = logging.getLogger("comfy_kitchen.dispatch")


class BackendRegistry:
    def __init__(self):
        self._backends = {}  # {name: backend_module}
        self._capabilities = {}  # {name: set of function names}
        self._constraints = {}  # {(backend_name, func_name): FunctionConstraints}
        self._priority = ["ascend", "cuda", "triton", "eager"]
        self._disabled = set()  # Manually disabled backends
        self._unavailable = {}  # {name: error_message} for failed backends
        self._lock = threading.Lock()
        self._thread_local = threading.local()

    def register(
        self,
        name: str,
        module,
        capabilities: dict[str, FunctionConstraints],
    ):
        """Register a backend with its capabilities and constraints.

        Args:
            name: Backend name (e.g., "cuda", "eager", "triton")
            module: The backend module containing implementations
            capabilities: Dict mapping function names to FunctionConstraints
        """
        with self._lock:
            self._backends[name] = module
            self._capabilities[name] = set(capabilities.keys())
            for func_name, constraints in capabilities.items():
                int8 = (
                    "int8" in func_name.split("_")
                    and func_name != "rotate_int8_convrot_weight"
                ) or func_name in {"w4a8_packed_linear", "w6a8_linear", "quantize_and_rotate_rowwise"}
                if int8 or (name == "cuda" and func_name in {
                    "int4_linear", "convrot_w4a4_linear", "sol_attn",
                }):
                    maximum = min(
                        constraints.max_compute_capability or INT8_MAX_COMPUTE_CAPABILITY,
                        INT8_MAX_COMPUTE_CAPABILITY,
                    )
                    constraints = replace(constraints, max_compute_capability=maximum)
                self._constraints[(name, func_name)] = constraints
                if constraints.max_compute_capability is not None:
                    func = getattr(module, func_name, None)
                    if callable(func):
                        guarded = guard_compute_capability(constraints.max_compute_capability)(func)
                        setattr(module, func_name, guarded)
            # Some CUDA helper entries are called directly by mixed-precision paths.
            for func_name, func in list(vars(module).items()):
                if (func_name not in capabilities and callable(func)
                        and not func_name.startswith("_")
                        and "int8" in func_name.split("_")
                        and func_name != "rotate_int8_convrot_weight"):
                    setattr(module, func_name, guard_compute_capability(INT8_MAX_COMPUTE_CAPABILITY)(func))
            self._unavailable.pop(name, None)

    def get_constraints(
        self, backend_name: str, func_name: str
    ) -> FunctionConstraints | None:
        """Get constraints for a specific backend/function pair."""
        return self._constraints.get((backend_name, func_name))

    def mark_unavailable(self, name: str, reason: str):
        """Mark a backend as unavailable with a reason.

        Args:
            name: Backend name
            reason: Why the backend is unavailable (e.g., "ImportError: ...")
        """
        with self._lock:
            self._unavailable[name] = reason

    def disable(self, backend_name: str):
        """Manually disable a backend.

        Args:
            backend_name: Name of backend to disable
        """
        with self._lock:
            self._disabled.add(backend_name)

    def enable(self, backend_name: str):
        """Re-enable a previously disabled backend.

        Args:
            backend_name: Name of backend to enable
        """
        with self._lock:
            self._disabled.discard(backend_name)

    def set_priority(self, priority_list: list[str]):
        """Set backend selection priority order.

        Args:
            priority_list: List of backend names in priority order
        """
        with self._lock:
            self._priority = list(priority_list)

    def is_available(self, backend_name: str) -> bool:
        """Check if backend is available and not disabled.

        Args:
            backend_name: Name of backend to check

        Returns:
            True if backend is registered and not disabled
        """
        return backend_name in self._backends and backend_name not in self._disabled

    def list_backends(self) -> dict:
        """Return dict of all backends with their status.

        Returns:
            Dict mapping backend names to status info:
            {
                "backend_name": {
                    "available": bool,
                    "disabled": bool,
                    "unavailable_reason": str or None,
                    "capabilities": list[str]
                }
            }
        """
        result = {}
        all_backend_names = (
            set(self._priority) | set(self._backends.keys()) | set(self._unavailable.keys())
        )

        for name in all_backend_names:
            result[name] = {
                "available": name in self._backends,
                "disabled": name in self._disabled,
                "unavailable_reason": self._unavailable.get(name),
                "capabilities": sorted(self._capabilities.get(name, [])),
            }

        return result

    def validate_backend_for_call(
        self,
        backend_name: str,
        func_name: str,
        kwargs: Mapping[str, Any],
    ) -> ValidationResult:
        """Validate if a backend can handle a function call with given args.

        Args:
            backend_name: Name of backend to validate
            func_name: Function name
            kwargs: Keyword arguments for the function call

        Returns:
            ValidationResult with success/failure details
        """
        # Check backend is available
        if not self.is_available(backend_name):
            return ValidationResult.fail(
                "__backend__",
                self._unavailable.get(backend_name, "not available"),
            )

        if func_name not in self._capabilities.get(backend_name, set()):
            return ValidationResult.fail("__function__", "not implemented")

        constraints = self._constraints.get((backend_name, func_name))
        if constraints is None:
            return ValidationResult.ok()

        capabilities = (None,)
        if constraints.min_compute_capability is not None or constraints.max_compute_capability is not None:
            capabilities = cuda_compute_capabilities(
                kwargs.values(), include_hip=constraints.min_compute_capability is not None
            ) or (None,)
        for capability in capabilities:
            result = validate_function_call(constraints, kwargs, compute_capability=capability)
            if not result.success:
                return result
        return result

    def get_capable_backend(
        self,
        func_name: str,
        kwargs: Mapping[str, Any] | None = None,
    ) -> str:
        """Find the best backend that can handle a function call.

        Args:
            func_name: Function name
            kwargs: Keyword arguments for constraint validation (empty/None skips validation)

        Returns:
            Backend name that will handle the function

        Raises:
            NoCapableBackendError: If no backend can handle the call
        """
        failures: dict[str, str] = {}
        validate = bool(kwargs)  # Skip validation if kwargs is empty/None

        # Check for thread-local override first
        override = getattr(self._thread_local, "backend_override", None)
        if override:
            if not self.is_available(override):
                failures[override] = "not available"
            elif func_name not in self._capabilities.get(override, set()):
                failures[override] = "not implemented"
            elif validate:
                result = self.validate_backend_for_call(override, func_name, kwargs)
                if result.success:
                    logger.debug("Backend %s selected for %s (override)", override, func_name)
                    return override
                failures[override] = f"{result.failed_param}: {result.failure_reason}"
            else:
                logger.debug("Backend %s selected for %s (override)", override, func_name)
                return override

        # Try backends in priority order
        for backend_name in self._priority:
            if not self.is_available(backend_name):
                continue
            if func_name not in self._capabilities.get(backend_name, set()):
                continue

            if validate:
                result = self.validate_backend_for_call(backend_name, func_name, kwargs)
                if not result.success:
                    failures[backend_name] = f"{result.failed_param}: {result.failure_reason}"
                    continue

            logger.debug("Backend %s selected for %s", backend_name, func_name)
            return backend_name

        raise NoCapableBackendError(func_name, failures)

    def get_implementation(
        self,
        func_name: str,
        backend: str | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> Callable:
        """Get the best implementation for a function.

        Args:
            func_name: Name of the function to get
            backend: Explicit backend to use, or None for auto-select
            kwargs: Kwargs for constraint validation (empty/None skips validation)

        Returns:
            The function implementation

        Raises:
            BackendNotFoundError: If explicit backend is not available
            BackendNotImplementedError: If explicit backend doesn't implement function
            NoCapableBackendError: If no backend can handle the call
        """
        if backend:
            if backend not in self._backends:
                raise BackendNotFoundError(backend, self._unavailable.get(backend, "not registered"))
            if backend in self._disabled:
                raise BackendNotFoundError(backend, "disabled")
            if func_name not in self._capabilities.get(backend, set()):
                raise BackendNotImplementedError(backend, func_name)
            if kwargs:
                result = self.validate_backend_for_call(backend, func_name, kwargs)
                if not result.success:
                    raise NoCapableBackendError(func_name, {backend: f"{result.failed_param}: {result.failure_reason}"})
            return getattr(self._backends[backend], func_name)

        selected_backend = self.get_capable_backend(func_name, kwargs)
        return getattr(self._backends[selected_backend], func_name)

    @contextmanager
    def use_backend(self, backend_name: str):
        """Context manager to temporarily use a specific backend.

        Args:
            backend_name: Name of backend to use in this context

        Example:
            with registry.use_backend("eager"):
                result = some_function()
        """
        # Validate backend exists and is available
        if not self.is_available(backend_name):
            if backend_name in self._unavailable:
                reason = self._unavailable[backend_name]
                raise BackendNotFoundError(backend_name, reason)
            elif backend_name in self._disabled:
                raise BackendNotFoundError(backend_name, "disabled")
            else:
                raise BackendNotFoundError(backend_name, "not registered")

        # Store previous override (in case of nested contexts)
        previous = getattr(self._thread_local, "backend_override", None)
        self._thread_local.backend_override = backend_name

        try:
            yield
        finally:
            # Restore previous state
            if previous is None:
                delattr(self._thread_local, "backend_override")
            else:
                self._thread_local.backend_override = previous


registry = BackendRegistry()
