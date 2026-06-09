import importlib.util
import os
import sys

import torch

__all__ = [
    "dequantize_per_tensor_fp8",
    "quantize_per_tensor_fp8",
    "stochastic_rounding_fp8",
]


try:
    _C = None  # type: ignore
    _module_path = os.path.join(os.path.dirname(__file__), "_C.abi3.so")

    if not os.path.exists(_module_path):
        directory = os.path.dirname(__file__)
        for filename in os.listdir(directory):
            if filename.startswith("_C.") and filename.endswith(".so"):
                _module_path = os.path.join(directory, filename)

    if os.path.exists(_module_path):
        _spec = importlib.util.spec_from_file_location(
            "comfy_kitchen.backends.hip._C", _module_path
        )
        if _spec and _spec.loader:
            _C = importlib.util.module_from_spec(_spec)
            sys.modules["comfy_kitchen.backends.hip._C"] = _C
            _spec.loader.exec_module(_C)
            _EXT_AVAILABLE = True
            _EXT_ERROR = None
        else:
            _EXT_AVAILABLE = False
            _EXT_ERROR = f"Could not create module spec for {_module_path}"
    else:
        _EXT_AVAILABLE = False
        _EXT_ERROR = f"Extension file not found: {_module_path}"
except ImportError as e:
    _EXT_AVAILABLE = False
    _EXT_ERROR = str(e)
    _C = None  # type: ignore
except Exception as e:
    _EXT_AVAILABLE = False
    _EXT_ERROR = f"Failed to load extension: {e}"
    _C = None  # type: ignore

from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE


def _wrap_for_dlpack(tensor: torch.Tensor):
    # stream=-1 tells PyTorch to skip synchronization (DLPack spec)
    return tensor.__dlpack__(stream=-1)


def quantize_per_tensor_fp8(
    x: torch.Tensor,
    scale: torch.Tensor,
    output_type: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    input_dtype_code = DTYPE_TO_CODE[x.dtype]
    output_dtype_code = DTYPE_TO_CODE[output_type]

    if not x.is_contiguous():
        x = x.contiguous()
    if scale.dim() == 0:
        scale = scale.reshape(1)

    result_uint8 = torch.empty(x.shape, dtype=torch.uint8, device=x.device)
    stream_ptr = torch.cuda.current_stream(x.device).cuda_stream
    _C.quantize_per_tensor_fp8(
        _wrap_for_dlpack(x),
        _wrap_for_dlpack(scale),
        _wrap_for_dlpack(result_uint8),
        input_dtype_code,
        output_dtype_code,
        x.numel(),
        stream_ptr,
    )
    return result_uint8.view(output_type)


def dequantize_per_tensor_fp8(
    x: torch.Tensor,
    scale: torch.Tensor,
    output_type: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    input_dtype_code = DTYPE_TO_CODE[x.dtype]
    output_dtype_code = DTYPE_TO_CODE[output_type]

    if not x.is_contiguous():
        x = x.contiguous()
    if scale.dim() == 0:
        scale = scale.reshape(1)

    result = torch.empty(x.shape, dtype=output_type, device=x.device)
    stream_ptr = torch.cuda.current_stream(x.device).cuda_stream
    _C.dequantize_per_tensor_fp8(
        _wrap_for_dlpack(x.view(torch.uint8)),
        _wrap_for_dlpack(scale),
        _wrap_for_dlpack(result),
        input_dtype_code,
        output_dtype_code,
        x.numel(),
        stream_ptr,
    )
    return result


def stochastic_rounding_fp8(
    x: torch.Tensor,
    rng: torch.Tensor,
    output_type: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    output_dtype_code = DTYPE_TO_CODE[output_type]

    if rng.device != x.device:
        raise ValueError("rng must be on the same device as x")
    if rng.shape != x.shape:
        raise ValueError("rng must have the same shape as x")

    if not x.is_contiguous():
        x = x.contiguous()
    if not rng.is_contiguous():
        rng = rng.contiguous()

    stream_ptr = torch.cuda.current_stream(x.device).cuda_stream
    _C.stochastic_round_fp8(
        _wrap_for_dlpack(rng),
        _wrap_for_dlpack(x),
        output_dtype_code,
        x.numel(),
        stream_ptr,
    )

    return rng.view(output_type)


def _build_constraints() -> dict:
    from comfy_kitchen.constraints import FunctionConstraints, ParamConstraint

    # PyTorch exposes ROCm tensors as device type "cuda"; keep "hip" too for
    # future compatibility with runtimes that expose a distinct device type.
    hip_devices = frozenset({"cuda", "hip"})

    return {
        "quantize_per_tensor_fp8": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
                ),
                "scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                "output_type": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn, torch.float8_e5m2}),
                ),
            },
            default_devices=hip_devices,
        ),
        "dequantize_per_tensor_fp8": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn, torch.float8_e5m2}),
                ),
                "scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                "output_type": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
                ),
            },
            default_devices=hip_devices,
        ),
        "stochastic_rounding_fp8": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
                ),
                "rng": ParamConstraint(dtypes=frozenset({torch.uint8})),
                "output_type": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn, torch.float8_e5m2}),
                ),
            },
            default_devices=hip_devices,
        ),
    }


def _register():
    from comfy_kitchen.registry import registry

    if not _EXT_AVAILABLE:
        registry.mark_unavailable("hip", _EXT_ERROR)
        return

    if not getattr(torch.version, "hip", None):
        registry.mark_unavailable("hip", "PyTorch ROCm/HIP runtime not available")
        return

    if not torch.cuda.is_available():
        registry.mark_unavailable("hip", "HIP device not available")
        return

    registry.register(
        name="hip",
        module=__import__(__name__, fromlist=__all__),
        capabilities=_build_constraints(),
    )


_register()
