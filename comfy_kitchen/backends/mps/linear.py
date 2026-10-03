# SPDX-License-Identifier: Apache-2.0
"""Weight-only quantized linear layers for Apple GPUs."""
from __future__ import annotations

import os

import torch

from comfy_kitchen.backends._activations import apply_input_act, apply_residual
from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation

_SHADERS = """
#include <metal_stdlib>
using namespace metal;

// out[n, k] = half(w[n, k] * scale[n]); one pass, int8 read + fp16 write.
kernel void dequant_i8_rowscale_f16(device half* out,
                                    constant char* w,
                                    constant float* scale,
                                    constant long& K,
                                    uint idx [[thread_position_in_grid]]) {
    out[idx] = half(float(w[idx]) * scale[idx / uint(K)]);
}

// Same decode, written transposed: out[k, n] = half(w[n, k] * scale[n]).
kernel void dequant_i8_rowscale_f16_t(device half* out,
                                      constant char* w,
                                      constant float* scale,
                                      constant long& K,
                                      constant long& N,
                                      uint idx [[thread_position_in_grid]]) {
    uint k = idx / uint(N);
    uint n = idx - k * uint(N);
    out[idx] = half(float(w[n * uint(K) + k]) * scale[n]);
}
"""

_lib = None

# Decode at most this many bytes of fp16 weight at once. Bounds the transient
# buffer for very wide layers; the GEMM then runs per row-chunk of the weight.
_MAX_DECODE_BYTES = int(os.environ.get("COMFY_KITCHEN_MPS_DECODE_BYTES", 512 * 1024 * 1024))


def _shaders():
    global _lib
    if _lib is None:
        _lib = torch.mps.compile_shader(_SHADERS)
    return _lib


def _compute_dtype(x: torch.Tensor) -> torch.dtype:
    # bf16 is emulated on M1-M4 GPUs and measurably slower than fp16.
    forced = os.environ.get("COMFY_KITCHEN_MPS_COMPUTE_DTYPE")
    if forced:
        return getattr(torch, forced)
    return torch.float32 if x.dtype == torch.float32 else torch.float16


# MPS matmul against a lazily transposed weight (what F.linear does) falls onto a
# slow path once the input width passes ~8k: 2x slower at K=16384 on an M1 Max.
# Decoding straight into [K, N] layout costs a strided read but avoids it.
_TRANSPOSED_DECODE_MIN_K = int(os.environ.get("COMFY_KITCHEN_MPS_TRANSPOSED_MIN_K", 12288))


def _decode_int8(weight: torch.Tensor, scale: torch.Tensor, transposed: bool = False) -> torch.Tensor:
    n, k = weight.shape
    if transposed:
        out = torch.empty((k, n), dtype=torch.float16, device=weight.device)
        _shaders().dequant_i8_rowscale_f16_t(out, weight, scale, k, n)
    else:
        out = torch.empty((n, k), dtype=torch.float16, device=weight.device)
        _shaders().dequant_i8_rowscale_f16(out, weight, scale, k)
    return out


def _gemm(xc: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, cdtype: torch.dtype) -> torch.Tensor:
    transposed = weight.shape[1] >= _TRANSPOSED_DECODE_MIN_K
    w = _decode_int8(weight, scale, transposed)
    if cdtype != torch.float16:
        w = w.to(cdtype)
    return xc @ w if transposed else torch.nn.functional.linear(xc, w)


def _to_compute(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.dtype]:
    """Cast activations to the compute dtype without losing range.

    fp16 tops out at 65504 while bf16/fp32 callers routinely carry larger
    activations (text-encoder states in the 1e4 range overflow after one GEMM).
    Each row is divided by a power of two that brings its max below 1, which is
    exact in every float format, and the result is multiplied back in the output
    dtype by ``_from_compute``.
    """
    cdtype = _compute_dtype(x)
    if cdtype != torch.float16 or x.dtype == torch.float16:
        return x.to(cdtype), None, cdtype
    row_max = x.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-30)
    row_scale = torch.exp2(torch.ceil(torch.log2(row_max)))
    return (x * torch.reciprocal(row_scale).to(x.dtype)).to(cdtype), row_scale, cdtype


def _from_compute(result: torch.Tensor, row_scale: torch.Tensor | None,
                  bias: torch.Tensor | None, out_dtype: torch.dtype) -> torch.Tensor:
    result = result.to(out_dtype)
    if row_scale is not None:
        result = result * row_scale.to(out_dtype)
    if bias is not None:
        result = result + bias.to(device=result.device, dtype=out_dtype)
    return result


def int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    convrot: bool = False,
    convrot_groupsize: int = 256,
    input_act: str | None = None,
    input_act_weight: torch.Tensor | None = None,
    input_act_eps: float = 0.0,
    residual: torch.Tensor | None = None,
    residual_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """INT8-weight linear, computed as fp16 GEMM over on-the-fly decoded weights."""
    x = apply_input_act(x, input_act, input_act_weight, input_act_eps)
    n, k = weight.shape
    if x.shape[-1] != k:
        raise ValueError(
            f"Input and weight inner dimensions must match, got {x.shape[-1]} and {k}"
        )
    weight = weight.to(device=x.device).contiguous()
    scale = weight_scale.to(device=x.device, dtype=torch.float32).reshape(-1)
    if scale.numel() == 1:
        scale = scale.expand(n)
    elif scale.numel() != n:
        raise ValueError(
            f"INT8 weight scale must be scalar or per-output-channel, got {tuple(weight_scale.shape)} "
            f"for weight shape {tuple(weight.shape)}"
        )
    scale = scale.contiguous()

    xc, row_scale, cdtype = _to_compute(x)
    if convrot:
        if k % convrot_groupsize != 0:
            raise ValueError(
                f"ConvRot group size {convrot_groupsize} does not divide input features {k}"
            )
        h = _build_hadamard(convrot_groupsize, device=x.device, dtype=cdtype)
        xc = _rotate_activation(xc, h, convrot_groupsize)

    rows_per_chunk = max(1, _MAX_DECODE_BYTES // (2 * k))
    if rows_per_chunk >= n:
        result = _gemm(xc, weight, scale, cdtype)
    else:
        parts = []
        for i in range(0, n, rows_per_chunk):
            parts.append(_gemm(xc, weight[i:i + rows_per_chunk].contiguous(),
                               scale[i:i + rows_per_chunk].contiguous(), cdtype))
        result = torch.cat(parts, dim=-1)

    result = _from_compute(result, row_scale, bias, out_dtype)
    return apply_residual(result, residual, residual_scale)
