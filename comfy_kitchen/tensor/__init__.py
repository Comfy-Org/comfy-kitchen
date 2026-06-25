"""Quantized tensor types with typed layout parameters."""
from .awq_w4a16 import TensorCoreAWQW4A16Layout
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
from .int8 import TensorWiseINT8Layout
from .mxfp8 import TensorCoreMXFP8Layout
from .nvfp4 import TensorCoreNVFP4Layout
from .svdquant_w4a4 import (
    TensorCoreSVDQuantW4A4Layout,
    svdquant_w4a4_can_share_quant,
    svdquant_w4a4_fuse_linear_weights,
    svdquant_w4a4_fused_grouped_linear,
    svdquant_w4a4_grouped_linear,
)

__all__ = [
    "BaseLayoutParams",
    "QuantizedLayout",
    "QuantizedTensor",
    "TensorCoreAWQW4A16Layout",
    "TensorCoreFP8Layout",
    "TensorCoreMXFP8Layout",
    "TensorCoreNVFP4Layout",
    "TensorWiseINT8Layout",
    "TensorCoreSVDQuantW4A4Layout",
    "dequantize_args",
    "get_cuda_capability",
    "get_layout_class",
    "register_layout_class",
    "register_layout_op",
    "svdquant_w4a4_can_share_quant",
    "svdquant_w4a4_fuse_linear_weights",
    "svdquant_w4a4_fused_grouped_linear",
    "svdquant_w4a4_grouped_linear",
]

register_layout_class("TensorCoreAWQW4A16Layout", TensorCoreAWQW4A16Layout)
register_layout_class("TensorCoreFP8Layout", TensorCoreFP8Layout)
register_layout_class("TensorCoreMXFP8Layout", TensorCoreMXFP8Layout)
register_layout_class("TensorCoreNVFP4Layout", TensorCoreNVFP4Layout)
register_layout_class("TensorWiseINT8Layout", TensorWiseINT8Layout)
register_layout_class("TensorCoreSVDQuantW4A4Layout", TensorCoreSVDQuantW4A4Layout)
