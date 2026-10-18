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
from .convrot_w4a4 import (
    TensorCoreConvRotW4A4Layout,
    convrot_w4a4_linear,
    dequantize_convrot_w4a4_weight,
    quantize_convrot_w4a4_weight,
)
from .fp8 import TensorCoreFP8Layout
from .int8 import TensorWiseINT8Layout, try_relayout_int8_weights
from .mxfp8 import TensorCoreMXFP8Layout
from .nvfp4 import TensorCoreNVFP4Layout
from .svdquant_w4a4 import (
    TensorCoreSVDQuantW4A4Layout,
    svdquant_w4a4_can_share_quant,
    svdquant_w4a4_fuse_linear_weights,
    svdquant_w4a4_fused_grouped_linear,
    svdquant_w4a4_grouped_linear,
)
from .w4a8_int8 import (
    AsymW4A8Int8Layout,
    dequantize_w4a8_int8_weight,
    quantize_w4a8_int8_weight,
    try_relayout_w4a8_weights,
    w4a8_int8_linear,
)

__all__ = [
    "AsymW4A8Int8Layout",
    "BaseLayoutParams",
    "QuantizedLayout",
    "QuantizedTensor",
    "TensorCoreAWQW4A16Layout",
    "TensorCoreConvRotW4A4Layout",
    "TensorCoreFP8Layout",
    "TensorCoreMXFP8Layout",
    "TensorCoreNVFP4Layout",
    "TensorWiseINT8Layout",
    "TensorCoreSVDQuantW4A4Layout",
    "convrot_w4a4_linear",
    "dequantize_args",
    "dequantize_convrot_w4a4_weight",
    "dequantize_w4a8_int8_weight",
    "get_cuda_capability",
    "get_layout_class",
    "register_layout_class",
    "register_layout_op",
    "quantize_convrot_w4a4_weight",
    "quantize_w4a8_int8_weight",
    "svdquant_w4a4_can_share_quant",
    "svdquant_w4a4_fuse_linear_weights",
    "svdquant_w4a4_fused_grouped_linear",
    "svdquant_w4a4_grouped_linear",
    "try_relayout_weights",
    "w4a8_int8_linear",
]


def try_relayout_weights(weights, storage):
    """Relayout consecutive resident weights into one combined weight in ``storage``.

    ``weights`` is an ordered sequence of canonical quantized weights whose bytes lie
    in ``storage``, a contiguous caller-owned uint8 span that the caller replaces as
    a group. Returns ``(combined, views)``, where ``views`` slices ``combined`` by
    output row in input order, or NotImplemented with ``storage`` unchanged.
    """
    weights = tuple(weights)
    if weights and all(
        isinstance(w, QuantizedTensor) and w._layout_cls == "AsymW4A8Int8Layout" for w in weights
    ):
        return try_relayout_w4a8_weights(weights, storage)
    if weights and all(
        isinstance(w, QuantizedTensor) and w._layout_cls == "TensorWiseINT8Layout" for w in weights
    ):
        return try_relayout_int8_weights(weights, storage)
    return NotImplemented


register_layout_class("AsymW4A8Int8Layout", AsymW4A8Int8Layout)
register_layout_class("TensorCoreAWQW4A16Layout", TensorCoreAWQW4A16Layout)
register_layout_class("TensorCoreConvRotW4A4Layout", TensorCoreConvRotW4A4Layout)
register_layout_class("TensorCoreFP8Layout", TensorCoreFP8Layout)
register_layout_class("TensorCoreMXFP8Layout", TensorCoreMXFP8Layout)
register_layout_class("TensorCoreNVFP4Layout", TensorCoreNVFP4Layout)
register_layout_class("TensorWiseINT8Layout", TensorWiseINT8Layout)
register_layout_class("TensorCoreSVDQuantW4A4Layout", TensorCoreSVDQuantW4A4Layout)
