"""Base classes for quantized tensors with typed layout parameters."""
from __future__ import annotations

import contextlib
import dataclasses
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
import torch._dynamo

LAYOUTS = {}


def register_layout_class(name: str, cls):
    LAYOUTS[name] = cls


def get_layout_class(name: str):
    return LAYOUTS[name]


# ==================== Capability Utilities ====================

@lru_cache(maxsize=1)
def get_cuda_capability() -> tuple[int, int] | None:
    """Get CUDA compute capability (SM version), cached."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability()


# ==================== Base Params Dataclass ====================

@dataclass(frozen=True)
class BaseLayoutParams:
    """Base dataclass for layout parameters with common functionality.

    Subclasses should define additional fields and override _tensor_fields()
    if they have additional tensor fields beyond 'scale'.
    """
    scale: torch.Tensor
    orig_dtype: torch.dtype
    orig_shape: tuple[int, ...]

    def __post_init__(self):
        self._validate_tensor_fields()

    def _validate_tensor_fields(self):
        # Implemented in subclasses
        pass

    def _tensor_fields(self) -> list[str]:
        """Return list of field names that are tensors. Override in subclass."""
        return ["scale"]

    def to_device(self, device: torch.device) -> BaseLayoutParams:
        """Move all tensor fields to the specified device."""
        kwargs = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        for field in self._tensor_fields():
            kwargs[field] = kwargs[field].to(device=device)
        return type(self)(**kwargs)

    def clone(self) -> BaseLayoutParams:
        """Clone all tensor fields."""
        kwargs = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        for field in self._tensor_fields():
            kwargs[field] = kwargs[field].clone()
        return type(self)(**kwargs)

    def copy_from(self, src: BaseLayoutParams, non_blocking: bool = False) -> None:
        """Copy tensor fields in-place from src, reusing existing memory.

        Args:
            src: Source params to copy from (must be same type).
            non_blocking: If True, use non-blocking copy for tensors.
        """
        for field in dataclasses.fields(self):
            src_val = getattr(src, field.name)
            if field.name in self._tensor_fields():
                getattr(self, field.name).copy_(src_val, non_blocking=non_blocking)
            else:
                object.__setattr__(self, field.name, src_val)


class QuantizedLayout(ABC):
    """Base class for quantization layouts. Subclasses define inner Params dataclass."""

    Params: type[Any]

    # Minimum SM version for fast matmul. None means no requirement.
    MIN_SM_VERSION: tuple[int, int] | None = None

    @classmethod
    @abstractmethod
    def quantize(cls, tensor: torch.Tensor, **kwargs) -> tuple[torch.Tensor, Any]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def dequantize(cls, qdata: torch.Tensor, params: Any) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_plain_tensors(cls, qtensor: QuantizedTensor) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def state_dict_tensors(cls, qdata: torch.Tensor, params: Any) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @classmethod
    def supports_fast_matmul(cls) -> bool:
        """Check if fast quantized matmul is supported on current hardware."""
        if cls.MIN_SM_VERSION is None:
            return True
        cap = get_cuda_capability()
        if cap is None:
            return False
        return cap >= cls.MIN_SM_VERSION

    @classmethod
    def get_requirements(cls) -> dict[str, Any]:
        """Return hardware/software requirements for this layout."""
        cap = get_cuda_capability()
        return {
            "layout": cls.__name__,
            "min_sm_version": cls.MIN_SM_VERSION,
            "current_sm_version": cap,
            "fast_matmul_supported": cls.supports_fast_matmul(),
        }


class QuantizedTensor(torch.Tensor):
    """
    Quantized tensor with typed layout parameters.

    Properties:
        shape: Original (unpadded) shape
        storage_shape: Actual shape of quantized data (may be padded)
        is_padded: True if storage_shape != original shape
    """

    @staticmethod
    def __new__(
        cls,
        qdata: torch.Tensor,
        layout_cls: str,
        params: Any,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            params.orig_shape,
            device=qdata.device,
            dtype=params.orig_dtype,
            requires_grad=False,
        )

    def __init__(
        self,
        qdata: torch.Tensor,
        layout_cls: str,
        params: Any,
    ):
        assert isinstance(layout_cls, str)
        self._qdata = qdata
        self._layout_cls = layout_cls
        self._params = params

    def __repr__(self) -> str:
        return (
            f"QuantizedTensor(shape={tuple(self.shape)}, "
            f"storage_shape={self.storage_shape}, "
            f"layout={self._layout_cls}, "
            f"dtype={self._params.orig_dtype})"
        )

    # ==================== Properties ====================

    @property
    def storage_shape(self) -> tuple[int, ...]:
        return tuple(self._qdata.shape)

    @property
    def storage_dtype(self) -> torch.dtype:
        """The dtype of the underlying quantized storage (e.g., float8_e4m3fn, uint8)."""
        return self._qdata.dtype

    @property
    def nbytes(self) -> int:
        """Return the actual storage size in bytes.

        Note: This returns the true memory footprint of the quantized data,
        not the logical size based on orig_dtype. For example, an FP8 quantized
        tensor with logical dtype bfloat16 returns the FP8 storage size.
        """
        return self._qdata.nbytes

    @property
    def padded_shape(self) -> tuple[int, ...]:
        layout_cls = get_layout_class(self._layout_cls)
        if hasattr(layout_cls, "get_logical_shape_from_storage"):
            return layout_cls.get_logical_shape_from_storage(self.storage_shape)
        return self.storage_shape

    @property
    def is_padded(self) -> bool:
        return self._params.orig_shape != self.padded_shape

    @property
    def layout_cls(self) -> type[QuantizedLayout]:
        return get_layout_class(self._layout_cls)

    @property
    def params(self) -> Any:
        return self._params

    # ==================== Factory Methods ====================

    @classmethod
    def from_float(
        cls,
        tensor: torch.Tensor,
        layout_cls: str,
        **kwargs,
    ) -> QuantizedTensor:
        qdata, params = get_layout_class(layout_cls).quantize(tensor, **kwargs)
        return cls(qdata, layout_cls, params)

    def _copy_with(
        self,
        qdata: torch.Tensor | None = None,
        params: Any | None = None,
        clone_params: bool = True,
    ) -> QuantizedTensor:
        """Create a copy with optionally modified qdata/params.

        Args:
            qdata: New quantized data tensor. If None, uses self._qdata.
            params: New parameters. If None, uses self._params (cloned if clone_params=True).
            clone_params: If True and params is None, clone self._params. Set to False
                when you know params don't need cloning (e.g., they're already new).
        """
        if params is None:
            params = self._params.clone() if clone_params else self._params
        return QuantizedTensor(
            qdata if qdata is not None else self._qdata,
            self._layout_cls,
            params,
        )

    # ==================== Core Operations ====================

    def data_ptr(self):
        return self._qdata.data_ptr()

    def is_pinned(self):
        return self._qdata.is_pinned()

    def storage(self):
        return self._qdata.storage()

    def dequantize(self) -> torch.Tensor:
        # Ensure qdata is contiguous - backends may not handle non-contiguous views
        # (e.g., after transpose/view operations)
        qdata = self._qdata.contiguous() if not self._qdata.is_contiguous() else self._qdata

        # Check if this is a logically transposed tensor (e.g., NVFP4 with deferred transpose)
        is_transposed = getattr(self._params, "transposed", False)

        if is_transposed:
            physical_shape = (self._params.orig_shape[1], self._params.orig_shape[0])
            full = self.layout_cls.dequantize(qdata, self._params)
            if full.shape[:2] != physical_shape:
                slices = tuple(slice(0, s) for s in physical_shape)
                full = full[slices]
            return full.t()

        full = self.layout_cls.dequantize(qdata, self._params)
        orig = self._params.orig_shape
        if full.shape != orig:
            slices = tuple(slice(0, s) for s in orig)
            return full[slices]
        return full

    def state_dict(self, prefix: str = "") -> dict[str, torch.Tensor]:
        tensors = self.layout_cls.state_dict_tensors(self._qdata, self._params)
        return {f"{prefix}{suffix}": tensor for suffix, tensor in tensors.items()}

    # ==================== Flatten/Unflatten Protocol ====================

    def __tensor_flatten__(self):
        inner_tensors = ["_qdata"]
        tensor_fields = {}
        non_tensor_fields = {}

        for field in dataclasses.fields(self._params):
            value = getattr(self._params, field.name)
            if isinstance(value, torch.Tensor):
                attr_name = f"_param_{field.name}"
                object.__setattr__(self, attr_name, value)
                inner_tensors.append(attr_name)
                tensor_fields[field.name] = attr_name
            else:
                non_tensor_fields[field.name] = value

        return inner_tensors, {
            "layout_cls": self._layout_cls,
            "params_class": type(self._params),
            "tensor_fields": tensor_fields,
            "non_tensor_fields": non_tensor_fields,
        }

    @staticmethod
    def __tensor_unflatten__(inner_tensors, ctx, outer_size, outer_stride):
        params_kwargs = dict(ctx["non_tensor_fields"])
        for field_name, attr_name in ctx["tensor_fields"].items():
            params_kwargs[field_name] = inner_tensors[attr_name]

        params = ctx["params_class"](**params_kwargs)
        return QuantizedTensor(inner_tensors["_qdata"], ctx["layout_cls"], params)

    # ==================== Torch Dispatch ====================

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        qt = args[0] if args else None

        # Step 1: Check generic dispatch table (layout-agnostic operations)
        handler = _DISPATCH_TABLE.get(func)
        if handler is not None:
            return handler(qt, args, kwargs)

        # Step 2: Check layout-specific dispatch table (with MRO lookup for subclasses)
        layout_cls = _get_layout_from_args(args)
        if layout_cls and func in _LAYOUT_DISPATCH_TABLE:
            op_handlers = _LAYOUT_DISPATCH_TABLE[func]
            for parent_cls in layout_cls.__mro__:
                if parent_cls in op_handlers:
                    return op_handlers[parent_cls](qt, args, kwargs)

        # Step 3: Fallback to dequantization
        logging.debug(f"QuantizedTensor: Unhandled op {func}, falling back to dequantize")
        return cls._dequant_and_fallback(func, args, kwargs)

    @classmethod
    def _dequant_and_fallback(cls, func, args, kwargs):
        return func(*dequantize_args(args), **dequantize_args(kwargs))


# ==================== Dispatch Utilities ====================

def dequantize_args(args):
    """Recursively dequantize QuantizedTensors in args/kwargs.

    Useful for fallback implementations that need to convert quantized
    tensors back to regular tensors before calling the underlying op.
    """
    if isinstance(args, QuantizedTensor):
        return args.dequantize()
    elif isinstance(args, dict):
        return {k: dequantize_args(v) for k, v in args.items()}
    elif isinstance(args, (list, tuple)):
        return type(args)(dequantize_args(a) for a in args)
    return args


# ==================== Dispatch Handlers ====================

def _parse_to_args(args, kwargs):
    """Extract device and dtype from .to() arguments."""
    device = kwargs.get("device")
    dtype = kwargs.get("dtype")
    for arg in args[1:]:
        if isinstance(arg, torch.device):
            device = arg
        elif isinstance(arg, torch.dtype):
            dtype = arg
        elif isinstance(arg, str):
            with contextlib.suppress(Exception):
                device = torch.device(arg)
    if isinstance(device, str):
        device = torch.device(device)
    return device, dtype


def _handle_detach(qt, args, kwargs):
    return qt._copy_with(qdata=qt._qdata.detach())


def _handle_clone(qt, args, kwargs):
    return qt._copy_with(qdata=qt._qdata.clone())


def _handle_to(qt, args, kwargs, force_copy=False):
    target_device, target_dtype = _parse_to_args(args, kwargs)

    needs_device = target_device is not None and target_device != qt._qdata.device
    needs_dtype = target_dtype is not None and target_dtype != qt._params.orig_dtype

    if not needs_device and not needs_dtype and not force_copy:
        return qt

    if needs_device:
        new_qdata = qt._qdata.to(device=target_device)
        new_params = qt._params.to_device(target_device)
    else:
        new_qdata = qt._qdata.clone() if force_copy else qt._qdata
        new_params = qt._params.clone()

    if needs_dtype:
        new_params = dataclasses.replace(new_params, orig_dtype=target_dtype)

    return qt._copy_with(qdata=new_qdata, params=new_params, clone_params=False)


def _handle_to_copy(qt, args, kwargs):
    return _handle_to(qt, args, kwargs, force_copy=True)


def _handle_contiguous(qt, args, kwargs):
    if qt._qdata.is_contiguous():
        return qt
    return qt._copy_with(qdata=qt._qdata.contiguous())


def _handle_is_contiguous(qt, args, kwargs):
    return qt._qdata.is_contiguous()


def _handle_copy_(qt, args, kwargs):
    dst, src = args[0], args[1]
    if not isinstance(src, QuantizedTensor):
        raise TypeError(f"Cannot copy {type(src).__name__} to QuantizedTensor")
    if dst._layout_cls != src._layout_cls:
        raise TypeError(f"Layout mismatch: {dst._layout_cls} vs {src._layout_cls}")

    dst_orig_dtype = dst._params.orig_dtype
    non_blocking = kwargs.get("non_blocking", len(args) >= 3)

    dst._qdata.copy_(src._qdata, non_blocking=non_blocking)
    dst._params.copy_from(src._params, non_blocking=non_blocking)
    dst._params = dataclasses.replace(dst._params, orig_dtype=dst_orig_dtype)
    return dst


def _handle_empty_like(qt, args, kwargs):
    target_dtype = kwargs.pop("dtype", None)
    target_device = kwargs.get("device")

    new_qdata = torch.empty_like(qt._qdata, device=target_device)
    new_params = qt._params.clone()

    if target_device is not None:
        new_params = new_params.to_device(target_device)
    if target_dtype is not None:
        new_params = dataclasses.replace(new_params, orig_dtype=target_dtype)

    return qt._copy_with(qdata=new_qdata, params=new_params, clone_params=False)


_DISPATCH_TABLE = {
    torch.ops.aten.detach.default: _handle_detach,
    torch.ops.aten.clone.default: _handle_clone,
    torch.ops.aten._to_copy.default: _handle_to_copy,
    torch.ops.aten.to.dtype_layout: _handle_to,
    torch.ops.aten.contiguous.default: _handle_contiguous,
    torch.ops.aten.is_contiguous.default: _handle_is_contiguous,
    torch.ops.aten.copy_.default: _handle_copy_,
    torch.ops.aten.empty_like.default: _handle_empty_like,
    torch.ops.aten._has_compatible_shallow_copy_type.default: lambda qt, args, kwargs: True,
}

# Layout-specific dispatch table: {torch_op: {layout_cls: handler}}
_LAYOUT_DISPATCH_TABLE: dict[Any, dict[type[QuantizedLayout], Any]] = {}


def register_layout_op(torch_op: Any, layout_cls: type[QuantizedLayout]):
    """Decorator to register a layout-specific operation handler.

    Args:
        torch_op: PyTorch operation (e.g., torch.ops.aten.linear.default)
        layout_cls: Layout class (e.g., TensorCoreFP8Layout)
    """
    def decorator(handler_func):
        if torch_op not in _LAYOUT_DISPATCH_TABLE:
            _LAYOUT_DISPATCH_TABLE[torch_op] = {}
        _LAYOUT_DISPATCH_TABLE[torch_op][layout_cls] = handler_func
        return handler_func
    return decorator


def _get_layout_from_args(args) -> type[QuantizedLayout] | None:
    """Extract layout class from operation arguments."""
    for arg in args:
        if isinstance(arg, QuantizedTensor):
            return get_layout_class(arg._layout_cls)
        elif isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, QuantizedTensor):
                    return get_layout_class(item._layout_cls)
    return None
