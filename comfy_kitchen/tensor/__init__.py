"""Quantized tensor types with typed layout parameters."""
from .base import (
    BaseLayoutParams,
    QuantizedLayout,
    QuantizedTensor,
    dequantize_args,
    get_cuda_capability,
    get_layout_class,
    register_layout_class,
    register_layout_op,
)
from .fp8 import TensorCoreFP8Layout
from .mxfp8 import TensorCoreMXFP8Layout
from .nvfp4 import TensorCoreNVFP4Layout

__all__ = [
    "BaseLayoutParams",
    "QuantizedLayout",
    "QuantizedTensor",
    "TensorCoreFP8Layout",
    "TensorCoreMXFP8Layout",
    "TensorCoreNVFP4Layout",
    "dequantize_args",
    "get_cuda_capability",
    "get_layout_class",
    "register_layout_class",
    "register_layout_op",
]

register_layout_class("TensorCoreFP8Layout", TensorCoreFP8Layout)
register_layout_class("TensorCoreMXFP8Layout", TensorCoreMXFP8Layout)
register_layout_class("TensorCoreNVFP4Layout", TensorCoreNVFP4Layout)
