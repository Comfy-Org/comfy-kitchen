"""Route the wide one-row linear layers to comfy-kitchen's fp16 GEMM.

A ConvRot text encoder's lm_head is M=1, K=4096, N=151936: 1.24 GB of weight
read per token, and the largest single kernel in the decode. On the 6-WGP
780M cuBLAS reads that weight at 52.8 GB/s; comfy-kitchen's one-row GEMV
reads it at 79.0 GB/s, 1.49x, on a kernel that is 21% of the decode.

The gate is the measured crossover of the two kitchen paths (see
comfy_kitchen/backends/hip/ops/gemm_fp16.hip): the GEMV wins from 24576
output columns up and loses below it, so the projections stay on cuBLAS.

Nothing about the arithmetic changes: both paths accumulate in fp32, and the
relative error against an fp32 reference is 2.7e-04 for fp16 and 2.3e-03 for
bf16 on either of them. The order the products are summed in differs, so the
last bits of the logits do.

The hook is on the layer class's _forward, which is where ComfyUI's mixed
precision linear calls F.linear with the weight it has already cast to the
input's dtype, so the kitchen sees the same weight the framework would use::

    from kitchen_fp16_linear import patch_model_with_kitchen_fp16

    clip = ...                       # the loaded text encoder
    patch_model_with_kitchen_fp16(clip.cond_stage_model)
"""
from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn

import comfy_kitchen

logger = logging.getLogger(__name__)

# The crossover of the two comfy-kitchen paths, measured on the 6-WGP 780M:
# the one-row GEMV reaches the read roofline from 24576 output columns up and
# is slower than the WMMA path below it.
MIN_OUT_FEATURES = int(os.getenv("COMFY_KITCHEN_FP16_MIN_OUT", "24576"))


def _kitchen_wants(input: torch.Tensor, weight: torch.Tensor, out_features: int,
                   min_out_features: int) -> bool:
    return (
        input.shape[0] == 1
        and out_features >= min_out_features
        and input.is_cuda
        and input.dtype in (torch.float16, torch.bfloat16)
        and weight.dtype == input.dtype
        and weight.is_contiguous()
    )


def _patch_class(cls: type, min_out_features: int) -> None:
    """Route one layer class's _forward to the kitchen's fp16 GEMM."""
    if getattr(cls, "_kitchen_fp16_patched", False):
        return
    original = cls._forward

    def _forward(self, input, weight, bias):
        if _kitchen_wants(input, weight, self.out_features, min_out_features):
            return comfy_kitchen.fp16_linear(input, weight, bias)
        return original(self, input, weight, bias)

    cls._forward = _forward
    cls._kitchen_fp16_patched = True


def patch_model_with_kitchen_fp16(
    model: nn.Module,
    min_out_features: int = MIN_OUT_FEATURES,
    verbose: bool = False,
) -> tuple[nn.Module, int]:
    """Route the wide linear layers of ``model`` to comfy-kitchen's fp16 GEMM.

    Args:
        model: The model to patch.
        min_out_features: Layers with fewer output columns are left alone.
        verbose: Print what was patched.

    Returns:
        The model and the number of layer classes patched.
    """
    classes: set[type] = set()
    for module in model.modules():
        if (hasattr(module, "out_features") and hasattr(module, "weight")
                and module.out_features >= min_out_features
                and hasattr(type(module), "_forward")):
            classes.add(type(module))
    for cls in classes:
        _patch_class(cls, min_out_features)
        if verbose:
            logger.info("kitchen fp16: %s", cls.__name__)
    return model, len(classes)
