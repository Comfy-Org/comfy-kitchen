class BackendError(RuntimeError):
    """Base exception for backend-related errors."""

    pass


class BackendNotFoundError(BackendError):
    def __init__(self, backend_name: str, reason: str | None = None):
        self.backend_name = backend_name
        msg = f"Backend '{backend_name}' is not available"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class BackendNotImplementedError(BackendError):
    def __init__(self, backend_name: str, func_name: str):
        self.backend_name = backend_name
        self.func_name = func_name
        msg = f"Backend '{backend_name}' does not implement '{func_name}'"
        super().__init__(msg)


class NoCapableBackendError(BackendError):
    """Raised when no backend can handle the request due to constraint violations."""

    def __init__(
        self,
        func_name: str,
        failures: dict[str, str],
    ):
        self.func_name = func_name
        self.failures = failures  # {backend_name: failure_reason}

        if not failures:
            msg = f"No backend available for '{func_name}'"
        else:
            details = "; ".join(f"{name}: {reason}" for name, reason in failures.items())
            msg = f"No backend can handle '{func_name}': {details}"
        super().__init__(msg)
