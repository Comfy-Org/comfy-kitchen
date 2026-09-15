from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import torch

__all__ = [
    "DivisibleBy",
    "ExactDims",
    "FunctionConstraints",
    "MinDims",
    "ParamConstraint",
    "ShapeRule",
    "ValidationResult",
    "validate_function_call",
    "validate_param",
]


# =============================================================================
# Shape Rules
# =============================================================================


class ShapeRule(ABC):
    """Base class for shape validation rules."""

    @abstractmethod
    def check(self, tensor: torch.Tensor) -> bool:
        """Check if tensor satisfies this shape rule."""
        ...

    @abstractmethod
    def describe(self) -> str:
        """Human-readable description of the rule."""
        ...


@dataclass(frozen=True)
class DivisibleBy(ShapeRule):
    """Check that a tensor dimension is divisible by a factor."""

    dim: int
    factor: int

    def check(self, tensor: torch.Tensor) -> bool:
        if tensor.dim() <= self.dim:
            return False
        return tensor.shape[self.dim] % self.factor == 0

    def describe(self) -> str:
        return f"dim[{self.dim}] divisible by {self.factor}"


@dataclass(frozen=True)
class MinDims(ShapeRule):
    """Check that tensor has at least N dimensions."""

    ndim: int

    def check(self, tensor: torch.Tensor) -> bool:
        return tensor.dim() >= self.ndim

    def describe(self) -> str:
        return f"at least {self.ndim}D"


@dataclass(frozen=True)
class ExactDims(ShapeRule):
    """Check that tensor has exactly N dimensions."""

    ndim: int

    def check(self, tensor: torch.Tensor) -> bool:
        return tensor.dim() == self.ndim

    def describe(self) -> str:
        return f"exactly {self.ndim}D"


# =============================================================================
# Parameter and Function Constraints
# =============================================================================


@dataclass(frozen=True)
class ParamConstraint:
    """Constraints for a single parameter.

    Attributes:
        dtypes: Allowed dtypes for this parameter. None means any dtype.
        devices: Allowed device types. None means inherit from function default.
        shape_rules: Tuple of shape rules that must all pass.
    """

    dtypes: frozenset[torch.dtype] | None = None
    devices: frozenset[str] | None = None
    shape_rules: tuple[ShapeRule, ...] = ()

    def check_dtype(self, value: torch.Tensor | torch.dtype) -> bool:
        """Check if value's dtype is allowed."""
        if self.dtypes is None:
            return True
        if isinstance(value, torch.Tensor):
            return value.dtype in self.dtypes
        # value is a dtype directly (e.g., output_type parameter) or a Python built-in type instance
        return value in self.dtypes or type(value) in self.dtypes

    def check_device(self, tensor: torch.Tensor, default_devices: frozenset[str]) -> bool:
        """Check if tensor's device is allowed."""
        devices = self.devices if self.devices is not None else default_devices
        if "*" in devices:
            return True
        return tensor.device.type in devices

    def check_shape(self, tensor: torch.Tensor) -> bool:
        """Check if tensor satisfies all shape rules."""
        return all(rule.check(tensor) for rule in self.shape_rules)


@dataclass(frozen=True)
class FunctionConstraints:
    """Constraints for an entire function.

    Attributes:
        params: Per-parameter constraints keyed by parameter name.
        default_devices: Default allowed devices when param doesn't specify.
        min_compute_capability: Minimum CUDA compute capability (major, minor).
    """

    params: dict[str, ParamConstraint] = field(default_factory=dict)
    default_devices: frozenset[str] = field(default_factory=lambda: frozenset({"cuda", "cpu"}))
    min_compute_capability: tuple[int, int] | None = None
    call_rules: tuple[Callable[[Mapping[str, object]], ValidationResult], ...] = ()

    def __hash__(self) -> int:
        return hash((
            tuple(sorted(self.params.items(), key=lambda x: x[0])),
            self.default_devices,
            self.min_compute_capability,
            self.call_rules,
        ))


# =============================================================================
# Validation Logic
# =============================================================================


@dataclass
class ValidationResult:
    """Result of constraint validation."""

    success: bool
    failed_param: str | None = None
    failure_reason: str | None = None

    @staticmethod
    def ok() -> ValidationResult:
        return ValidationResult(success=True)

    @staticmethod
    def fail(param: str, reason: str) -> ValidationResult:
        return ValidationResult(success=False, failed_param=param, failure_reason=reason)


def validate_param(
    name: str,
    value: torch.Tensor | torch.dtype | None,
    constraint: ParamConstraint,
    default_devices: frozenset[str],
) -> ValidationResult:
    """Validate a single parameter against its constraint.

    Args:
        name: Parameter name (for error messages)
        value: The parameter value
        constraint: Constraint to validate against
        default_devices: Default devices from FunctionConstraints

    Returns:
        ValidationResult indicating success or failure with details
    """
    if value is None:
        return ValidationResult.ok()

    if not constraint.check_dtype(value):
        if isinstance(value, torch.Tensor):
            return ValidationResult.fail(
                name, f"dtype {value.dtype} not in {set(constraint.dtypes)}"
            )
        else:
            return ValidationResult.fail(
                name, f"dtype {value} not in {set(constraint.dtypes)}"
            )

    if not isinstance(value, torch.Tensor):
        return ValidationResult.ok()

    if not constraint.check_device(value, default_devices):
        allowed = constraint.devices if constraint.devices is not None else default_devices
        return ValidationResult.fail(
            name, f"device {value.device.type} not in {set(allowed)}"
        )

    for rule in constraint.shape_rules:
        if not rule.check(value):
            return ValidationResult.fail(name, f"shape {list(value.shape)} fails: {rule.describe()}")

    return ValidationResult.ok()


def validate_function_call(
    constraints: FunctionConstraints,
    kwargs: Mapping[str, torch.Tensor | torch.dtype | None],
    compute_capability: tuple[int, int] | None = None,
) -> ValidationResult:
    """Validate all parameters for a function call.

    Args:
        constraints: Function constraints to validate against
        kwargs: Keyword arguments passed to the function
        compute_capability: Current device's compute capability (major, minor)

    Returns:
        ValidationResult indicating success or first failure
    """
    if constraints.min_compute_capability is not None:
        if compute_capability is None:
            return ValidationResult.fail(
                "__hardware__", "CUDA compute capability required but not available"
            )
        min_major, min_minor = constraints.min_compute_capability
        curr_major, curr_minor = compute_capability
        if (curr_major, curr_minor) < (min_major, min_minor):
            return ValidationResult.fail(
                "__hardware__",
                f"compute capability {curr_major}.{curr_minor} < required {min_major}.{min_minor}",
            )

    for param_name, param_constraint in constraints.params.items():
        value = kwargs.get(param_name)
        result = validate_param(param_name, value, param_constraint, constraints.default_devices)
        if not result.success:
            return result

    for rule in constraints.call_rules:
        result = rule(kwargs)
        if not result.success:
            return result

    return ValidationResult.ok()


def sol_attn_common_call_rule(kwargs):
    """Shared ``sol_attn`` validation. The fused backends size everything from
    ``q`` and reach ``k``/``v`` as bare pointers, so a mismatch becomes an
    out-of-bounds read; head_dim 128 is baked into the kernels' layout."""
    q = kwargs.get("q")
    if q is not None:
        for name in ("k", "v"):
            other = kwargs.get(name)
            if other is None:
                continue
            if tuple(other.shape) != tuple(q.shape):
                return ValidationResult.fail(
                    name, f"must have the same shape as q {tuple(q.shape)}, got {tuple(other.shape)}"
                )
            if other.dtype != q.dtype:
                return ValidationResult.fail(
                    name, f"must have the same dtype as q ({q.dtype}), got {other.dtype}"
                )
            if other.device != q.device:
                return ValidationResult.fail(
                    name, f"must be on the same device as q ({q.device}), got {other.device}"
                )
        if q.shape[-1] != 128:
            return ValidationResult.fail("q", "head_dim must be 128")
        block_len = kwargs.get("block_len")
        if block_len is not None:
            n = (q.shape[1] + 63) // 64
            if block_len.dtype != torch.int32 or block_len.dim() != 1 or block_len.numel() != n:
                return ValidationResult.fail(
                    "block_len", f"must be int32 of shape ({n},), got {block_len.dtype} {tuple(block_len.shape)}")
            if block_len.device != q.device:
                return ValidationResult.fail("block_len", f"must be on {q.device}, got {block_len.device}")
        coarse_gate = kwargs.get("coarse_gate")
        if coarse_gate is not None:
            if tuple(coarse_gate.shape) != tuple(q.shape):
                return ValidationResult.fail(
                    "coarse_gate", f"must have q's shape {tuple(q.shape)}, got {tuple(coarse_gate.shape)}")
            if coarse_gate.device != q.device:
                return ValidationResult.fail("coarse_gate", f"must be on {q.device}, got {coarse_gate.device}")
    # Both sinks are half-open [start, end) block ranges.
    for name in ("sink_blocks", "sink_q"):
        value = kwargs.get(name)
        if value is None:
            continue
        if len(value) != 2:
            return ValidationResult.fail(name, f"must have 2 elements [start, end), got {len(value)}")
        start, end = value
        if start < 0 or end < 0:
            return ValidationResult.fail(name, f"entries must be non-negative, got {list(value)}")
        if end < start:
            return ValidationResult.fail(name, f"must be [start, end) with end >= start, got {list(value)}")
    # 0 means "tau threshold"; anything else is a fraction of the key blocks.
    topk_ratio = kwargs.get("topk_ratio")
    if topk_ratio and not 0.0 < topk_ratio < 1.0:
        return ValidationResult.fail("topk_ratio", f"must be 0 or in (0, 1), got {topk_ratio}")
    # token routing budget: 64-token tiles, at most the kernels' list capacity
    token_aug = kwargs.get("token_aug") or 0
    if token_aug < 0 or token_aug > 256 or token_aug % 64:
        return ValidationResult.fail("token_aug", f"must be 0 or a multiple of 64 up to 256, got {token_aug}")
    return ValidationResult.ok()


def na3d_common_call_rule(kwargs):
    """Shared ``na3d`` validation (used by every backend's entry).

    The fused backends size everything from ``q`` and pass ``k``/``v`` as bare
    pointers, so a mismatch becomes an out-of-bounds read.
    """
    q = kwargs.get("q")
    if q is not None:
        for name in ("k", "v"):
            other = kwargs.get(name)
            if other is None:
                continue
            if tuple(other.shape) != tuple(q.shape):
                return ValidationResult.fail(
                    name, f"must have the same shape as q {tuple(q.shape)}, got {tuple(other.shape)}"
                )
            if other.dtype != q.dtype:
                return ValidationResult.fail(
                    name, f"must have the same dtype as q ({q.dtype}), got {other.dtype}"
                )
            if other.device != q.device:
                return ValidationResult.fail(
                    name, f"must be on the same device as q ({q.device}), got {other.device}"
                )
    for name in ("kernel_size", "is_causal"):
        value = kwargs.get(name)
        if value is not None and len(value) != 3:
            return ValidationResult.fail(name, f"must have 3 elements, got {len(value)}")
    kernel_size = kwargs.get("kernel_size")
    if kernel_size is not None and any(k < 1 for k in kernel_size):
        return ValidationResult.fail(
            "kernel_size", f"entries must be positive, got {list(kernel_size)}"
        )
    return ValidationResult.ok()
