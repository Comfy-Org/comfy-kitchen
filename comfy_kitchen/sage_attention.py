# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pure INT8 scaled dot-product attention for NVIDIA tensor-core GPUs."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass

import torch
from torch.nn import functional

from .backends import cuda as _cuda_backend
from .backends.eager.quantization import DTYPE_TO_CODE

if getattr(torch.version, "hip", None):
    from .backends import hip as _hip_backend
else:
    _hip_backend = None

CTA_K = 64
LARGE_CTA_K = 128
_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_NATIVE_MINIMUM_CAPABILITY = (7, 5)


@dataclass(frozen=True, slots=True)
class PrequantizedInt8Attention:
    """Packed Q/K/V and immutable launch metadata for split INT8 attention.

    Instances own only the quantized tensors, their scales, and an optional
    attention mask. They never retain the floating-point Q, K, or V inputs.
    Create instances with :func:`prequantize_int8_attention` rather than
    constructing them directly.
    """

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    original_head_dim: int
    input_dtype: torch.dtype
    attention_scale: float
    cta_k: int
    attn_mask: torch.Tensor | None


def _pad_to_cta_k(length: int, cta_k: int = CTA_K) -> int:
    return ((length + cta_k - 1) // cta_k) * cta_k


def _select_cta_k(
    kernel_head_dim: int,
    kv_length: int,
    *,
    has_mask: bool,
) -> int:
    if not has_mask and kernel_head_dim >= 128 and kv_length > 1024:
        return LARGE_CTA_K
    return CTA_K


def is_available(device: torch.device | int | None = None) -> bool:
    """Return whether the compiled INT8 attention kernel supports this GPU."""
    if not torch.cuda.is_available():
        return False
    if _hip_backend is not None:
        # torch.cuda is the ROCm API here, and get_device_capability reports
        # something SM-shaped for a gfx part, so the compute capability test
        # below would wave AMD hardware through to a CUDA extension that never
        # loaded. Ask the HIP backend instead.
        return _hip_backend.int8_attention_is_available()
    capability = torch.cuda.get_device_capability(device)
    return capability >= _NATIVE_MINIMUM_CAPABILITY and _cuda_backend._EXT_AVAILABLE


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [batch, heads, sequence, head_dim]")
    if q.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"q, k, and v must be float32, float16, or bfloat16, got {q.dtype}")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError(
            f"q, k, and v must have the same dtype, got {q.dtype}, {k.dtype}, and {v.dtype}"
        )
    if not q.is_cuda or q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same CUDA device")
    if not is_available(q.device):
        raise RuntimeError(
            "INT8 attention requires the comfy-kitchen CUDA extension on SM75 or newer, "
            "or the HIP extension on an AMD device with matrix cores (RDNA3 or newer)"
        )

    batch, q_heads, q_length, head_dim = q.shape
    k_batch, kv_heads, kv_length, k_head_dim = k.shape
    if v.shape != (batch, kv_heads, kv_length, head_dim):
        raise ValueError(
            f"v must have shape [q.batch, k.heads, k.sequence, q.head_dim], got {tuple(v.shape)}"
        )
    if k_batch != batch or k_head_dim != head_dim:
        raise ValueError(
            f"q and k batch/head dimensions must match, got {tuple(q.shape)} and {tuple(k.shape)}"
        )
    if batch == 0 or q_heads == 0 or kv_heads == 0 or q_length == 0 or kv_length == 0:
        raise ValueError("batch, head counts, and sequence lengths must be positive")
    if q_heads % kv_heads != 0:
        raise ValueError(
            f"q head count ({q_heads}) must be divisible by k/v head count ({kv_heads})"
        )
    if head_dim <= 0 or head_dim > 256:
        raise ValueError(f"head_dim must be in [1, 256], got {head_dim}")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("the last dimension of q, k, and v must be contiguous")
    if attn_mask is None:
        return None
    if attn_mask.device != q.device:
        raise ValueError("attn_mask must be on the same CUDA device as q, k, and v")
    if attn_mask.dtype not in (torch.bool, torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("attn_mask must be bool, float16, bfloat16, or float32")
    try:
        return torch.broadcast_to(attn_mask, (batch, q_heads, q_length, kv_length))
    except RuntimeError as error:
        raise ValueError(
            "attn_mask must be broadcastable to "
            f"[{batch}, {q_heads}, {q_length}, {kv_length}], got {tuple(attn_mask.shape)}"
        ) from error


_LOG_INT8_PATH = os.environ.get("COMFY_KITCHEN_LOG_INT8_PATH", "0") == "1"
_logger = logging.getLogger("comfy_kitchen.sage_attention")
_attn_path_logged: set[tuple] = set()


def _log_attn_once(key: tuple, message: str) -> None:
    if not _LOG_INT8_PATH or key in _attn_path_logged:
        return
    _attn_path_logged.add(key)
    _logger.warning("[int8_attn_path] %s", message)


def _int8_attention_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    attn_mask = _validate_inputs(q, k, v, attn_mask)

    original_head_dim = q.shape[-1]
    if original_head_dim <= 64:
        kernel_head_dim = 64
    elif original_head_dim <= 128:
        kernel_head_dim = 128
    else:
        kernel_head_dim = 256
    if kernel_head_dim != original_head_dim:
        padding = (0, kernel_head_dim - original_head_dim)
        q = functional.pad(q, padding)
        k = functional.pad(k, padding)
        v = functional.pad(v, padding)

    attention_scale = original_head_dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(attention_scale):
        raise ValueError(f"scale must be finite, got {attention_scale}")

    if _hip_backend is not None:
        output = _hip_backend.sage_int8_sdpa(
            q,
            k,
            v,
            attention_scale=attention_scale,
            attn_mask=attn_mask,
        )
        output = output[..., :original_head_dim]
        return output.float() if q.dtype == torch.float32 else output

    batch, q_heads, q_length, _ = q.shape
    _, kv_heads, kv_length, _ = k.shape
    cta_k = _select_cta_k(
        kernel_head_dim,
        kv_length,
        has_mask=attn_mask is not None,
    )
    _log_attn_once(
        (batch, q_heads, q_length, kv_length, kernel_head_dim, cta_k, attn_mask is not None),
        f"int8_attention: B={batch} Hq={q_heads} Hkv={kv_heads} Sq={q_length} Skv={kv_length} "
        f"D={original_head_dim} kernel_D={kernel_head_dim} cta_k={cta_k} has_mask={attn_mask is not None} "
        f"cap={torch.cuda.get_device_capability(q.device)}",
    )
    padded_k_length = _pad_to_cta_k(kv_length, cta_k)
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)
    q_scales_per_block = 64 if kernel_head_dim == 256 else 32
    q_scale = torch.empty(
        batch,
        q_heads,
        ((q_length + 127) // 128) * q_scales_per_block,
        dtype=torch.float32,
        device=q.device,
    )
    k_scale = torch.empty(
        batch,
        kv_heads,
        ((kv_length + cta_k - 1) // cta_k) * 4,
        dtype=torch.float32,
        device=q.device,
    )
    v_int8 = torch.empty(
        batch * kv_heads * kernel_head_dim,
        padded_k_length,
        dtype=torch.int8,
        device=q.device,
    )
    v_scale = torch.empty(batch * kv_heads * kernel_head_dim, dtype=torch.float32, device=q.device)

    output_dtype = torch.bfloat16 if q.dtype == torch.float32 else q.dtype
    output = torch.empty(
        batch, q_heads, q_length, kernel_head_dim, dtype=output_dtype, device=q.device
    )

    anchor_indices = torch.empty(
        batch, kv_heads, dtype=torch.int32, device=q.device
    )
    anchor_indices_ptr = anchor_indices.data_ptr()

    stream_ptr = torch.cuda.current_stream(q.device).cuda_stream
    if attn_mask is None:
        _cuda_backend._C.sage_sdpa(
            _cuda_backend._wrap_for_dlpack(q),
            _cuda_backend._wrap_for_dlpack(k),
            _cuda_backend._wrap_for_dlpack(v),
            _cuda_backend._wrap_for_dlpack(output),
            _cuda_backend._wrap_for_dlpack(q_int8),
            _cuda_backend._wrap_for_dlpack(q_scale),
            _cuda_backend._wrap_for_dlpack(k_int8),
            _cuda_backend._wrap_for_dlpack(k_scale),
            _cuda_backend._wrap_for_dlpack(v_int8),
            _cuda_backend._wrap_for_dlpack(v_scale),
            attention_scale,
            DTYPE_TO_CODE[q.dtype],
            DTYPE_TO_CODE[output_dtype],
            stream_ptr,
            anchor_indices_ptr,
            cta_k=cta_k,
        )
    else:
        _cuda_backend._C.sage_sdpa(
            _cuda_backend._wrap_for_dlpack(q),
            _cuda_backend._wrap_for_dlpack(k),
            _cuda_backend._wrap_for_dlpack(v),
            _cuda_backend._wrap_for_dlpack(output),
            _cuda_backend._wrap_for_dlpack(q_int8),
            _cuda_backend._wrap_for_dlpack(q_scale),
            _cuda_backend._wrap_for_dlpack(k_int8),
            _cuda_backend._wrap_for_dlpack(k_scale),
            _cuda_backend._wrap_for_dlpack(v_int8),
            _cuda_backend._wrap_for_dlpack(v_scale),
            attention_scale,
            DTYPE_TO_CODE[q.dtype],
            DTYPE_TO_CODE[output_dtype],
            stream_ptr,
            anchor_indices_ptr,
            _cuda_backend._wrap_for_dlpack(attn_mask),
            cta_k=cta_k,
        )

    output = output[..., :original_head_dim]
    return output.float() if q.dtype == torch.float32 else output


def prequantize_int8_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    attn_mask: torch.Tensor | None = None,
) -> PrequantizedInt8Attention:
    """Quantize Q, K, and V without allocating the attention output.

    The returned object does not retain the floating-point inputs, so model
    code can delete those tensors before calling
    :func:`int8_attention_from_prequantized`. Quantization and consumption use
    the current CUDA stream and preserve normal PyTorch stream ordering; no
    host synchronization is introduced.

    This is an inference and peak-memory API. The regular :func:`int8_attention`
    remains the lower-overhead single-call path when early Q/K/V release is not
    needed.
    """
    attn_mask = _validate_inputs(q, k, v, attn_mask)

    original_head_dim = q.shape[-1]
    input_dtype = q.dtype
    if original_head_dim <= 64:
        kernel_head_dim = 64
    elif original_head_dim <= 128:
        kernel_head_dim = 128
    else:
        kernel_head_dim = 256
    if kernel_head_dim != original_head_dim:
        padding = (0, kernel_head_dim - original_head_dim)
        q = functional.pad(q, padding)
        k = functional.pad(k, padding)
        v = functional.pad(v, padding)

    attention_scale = original_head_dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(attention_scale):
        raise ValueError(f"scale must be finite, got {attention_scale}")

    if _hip_backend is not None:
        # The packed V row width follows this cta_k, so the value that packed the
        # buffers is the one that has to come back to attend over them. Taking the
        # CUDA-side constant here would only agree with the HIP choice by accident.
        hip_cta_k = _hip_backend._sage_cta_k(
            kernel_head_dim, k.shape[2], attn_mask is not None
        )
        packed = _hip_backend.sage_int8_quantize(q, k, v, cta_k=hip_cta_k)
        return PrequantizedInt8Attention(
            q=packed["q_int8"],
            k=packed["k_int8"],
            v=packed["v_int8"],
            q_scale=packed["q_scale"],
            k_scale=packed["k_scale"],
            v_scale=packed["v_scale"],
            original_head_dim=original_head_dim,
            input_dtype=input_dtype,
            attention_scale=attention_scale,
            cta_k=hip_cta_k,
            attn_mask=attn_mask,
        )

    batch, q_heads, q_length, _ = q.shape
    _, kv_heads, kv_length, _ = k.shape
    cta_k = _select_cta_k(
        kernel_head_dim,
        kv_length,
        has_mask=attn_mask is not None,
    )
    _log_attn_once(
        (batch, q_heads, q_length, kv_length, kernel_head_dim, cta_k, attn_mask is not None),
        f"int8_attention: B={batch} Hq={q_heads} Hkv={kv_heads} Sq={q_length} Skv={kv_length} "
        f"D={original_head_dim} kernel_D={kernel_head_dim} cta_k={cta_k} has_mask={attn_mask is not None} "
        f"cap={torch.cuda.get_device_capability(q.device)}",
    )
    padded_k_length = _pad_to_cta_k(kv_length, cta_k)
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)
    q_scales_per_block = 64 if kernel_head_dim == 256 else 32
    q_scale = torch.empty(
        batch,
        q_heads,
        ((q_length + 127) // 128) * q_scales_per_block,
        dtype=torch.float32,
        device=q.device,
    )
    k_scale = torch.empty(
        batch,
        kv_heads,
        ((kv_length + cta_k - 1) // cta_k) * 4,
        dtype=torch.float32,
        device=q.device,
    )
    v_int8 = torch.empty(
        batch * kv_heads * kernel_head_dim,
        padded_k_length,
        dtype=torch.int8,
        device=q.device,
    )
    v_scale = torch.empty(
        batch * kv_heads * kernel_head_dim,
        dtype=torch.float32,
        device=q.device,
    )

    anchor_indices = torch.empty(
        batch, kv_heads, dtype=torch.int32, device=q.device
    )
    anchor_indices_ptr = anchor_indices.data_ptr()

    stream_ptr = torch.cuda.current_stream(q.device).cuda_stream
    _cuda_backend._C.sage_sdpa_quantize(
        _cuda_backend._wrap_for_dlpack(q),
        _cuda_backend._wrap_for_dlpack(k),
        _cuda_backend._wrap_for_dlpack(v),
        _cuda_backend._wrap_for_dlpack(q_int8),
        _cuda_backend._wrap_for_dlpack(q_scale),
        _cuda_backend._wrap_for_dlpack(k_int8),
        _cuda_backend._wrap_for_dlpack(k_scale),
        _cuda_backend._wrap_for_dlpack(v_int8),
        _cuda_backend._wrap_for_dlpack(v_scale),
        cta_k,
        DTYPE_TO_CODE[input_dtype],
        stream_ptr,
        anchor_indices_ptr,
    )

    return PrequantizedInt8Attention(
        q=q_int8,
        k=k_int8,
        v=v_int8,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        original_head_dim=original_head_dim,
        input_dtype=input_dtype,
        attention_scale=attention_scale,
        cta_k=cta_k,
        attn_mask=attn_mask,
    )


def int8_attention_from_prequantized(
    quantized: PrequantizedInt8Attention,
) -> torch.Tensor:
    """Run INT8 attention after the floating-point Q/K/V inputs are released."""
    if not isinstance(quantized, PrequantizedInt8Attention):
        raise TypeError(
            "quantized must be returned by prequantize_int8_attention, got "
            f"{type(quantized).__name__}"
        )

    packed_tensors = (
        quantized.q,
        quantized.k,
        quantized.v,
        quantized.q_scale,
        quantized.k_scale,
        quantized.v_scale,
    )
    if not quantized.q.is_cuda:
        raise ValueError("prequantized INT8 attention tensors must be on a CUDA device")
    if any(tensor.device != quantized.q.device for tensor in packed_tensors[1:]):
        raise ValueError("prequantized INT8 attention tensors must be on the same CUDA device")
    if quantized.attn_mask is not None and quantized.attn_mask.device != quantized.q.device:
        raise ValueError("attn_mask must be on the same CUDA device as the packed tensors")
    if not is_available(quantized.q.device):
        raise RuntimeError(
            "INT8 attention requires the comfy-kitchen CUDA extension on SM75 or newer, "
            "or the HIP extension on an AMD device with matrix cores (RDNA3 or newer)"
        )

    batch, q_heads, q_length, kernel_head_dim = quantized.q.shape
    output_dtype = (
        torch.bfloat16 if quantized.input_dtype == torch.float32 else quantized.input_dtype
    )

    if _hip_backend is not None:
        output = _hip_backend.sage_int8_attend(
            quantized.q,
            quantized.k,
            quantized.v,
            quantized.q_scale,
            quantized.k_scale,
            quantized.v_scale,
            attention_scale=quantized.attention_scale,
            attn_mask=quantized.attn_mask,
            output_dtype=output_dtype,
            cta_k=quantized.cta_k,
        )
        output = output[..., : quantized.original_head_dim]
        return output.float() if quantized.input_dtype == torch.float32 else output

    output = torch.empty(
        batch,
        q_heads,
        q_length,
        kernel_head_dim,
        dtype=output_dtype,
        device=quantized.q.device,
    )

    stream_ptr = torch.cuda.current_stream(quantized.q.device).cuda_stream
    arguments = (
        _cuda_backend._wrap_for_dlpack(quantized.q),
        _cuda_backend._wrap_for_dlpack(quantized.k),
        _cuda_backend._wrap_for_dlpack(quantized.v),
        _cuda_backend._wrap_for_dlpack(output),
        _cuda_backend._wrap_for_dlpack(quantized.q_scale),
        _cuda_backend._wrap_for_dlpack(quantized.k_scale),
        _cuda_backend._wrap_for_dlpack(quantized.v_scale),
        quantized.cta_k,
        quantized.attention_scale,
        DTYPE_TO_CODE[output_dtype],
        stream_ptr,
    )
    if quantized.attn_mask is None:
        _cuda_backend._C.sage_sdpa_prequantized(*arguments)
    else:
        _cuda_backend._C.sage_sdpa_prequantized(
            *arguments,
            _cuda_backend._wrap_for_dlpack(quantized.attn_mask),
        )

    output = output[..., : quantized.original_head_dim]
    return output.float() if quantized.input_dtype == torch.float32 else output


@torch.library.custom_op("comfy_kitchen::int8_attention", mutates_args=())
def _op_int8_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None,
) -> torch.Tensor:
    return _int8_attention_cuda(
        q,
        k,
        v,
        scale=scale,
        attn_mask=None,
    )


@_op_int8_attention.register_fake
def _op_int8_attention_fake(
    q,
    k,
    v,
    scale,
):
    return q.new_empty(q.shape)


@torch.library.custom_op("comfy_kitchen::int8_attention_masked", mutates_args=())
def _op_int8_attention_masked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
    scale: float | None,
) -> torch.Tensor:
    return _int8_attention_cuda(
        q,
        k,
        v,
        scale=scale,
        attn_mask=attn_mask,
    )


@_op_int8_attention_masked.register_fake
def _op_int8_attention_masked_fake(
    q,
    k,
    v,
    attn_mask,
    scale,
):
    return q.new_empty(q.shape)


def int8_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute inference SDPA with signed INT8 Q/K/V and unsigned INT8 P.

    Inputs use ``[batch, heads, sequence, head_dim]`` layout. Grouped-query
    attention and unequal non-causal Q/K sequence lengths are supported. Head
    dimensions are padded to the kernel's 64-, 128-, or 256-wide tile and
    sliced back on return; 64, 128, and 256 take the zero-copy dimension path.
    Q and K receive the same fused block-Hadamard rotation before INT8
    quantization, preserving their exact dot products while reducing
    quantization outliers. K lengths up to 256 use low-overhead H4 blocks and
    longer D64 attention uses H64. The common D128 path uses a fixed signed H128
    transform, while padded D256 uses plain H128. The kernel samples K on the
    GPU and subtracts a representative key only when doing so improves the
    quantization range. This model-independent shift is exactly
    softmax-invariant and uses one int32 of temporary storage per batch/KV-head.
    Softmax score, maximum, exponential, denominator, reciprocal, and V-scale
    arithmetic is FP32. This path does not allocate FP8 tensors or execute FP8
    MMA instructions.
    """
    if attn_mask is None:
        return torch.ops.comfy_kitchen.int8_attention(
            q,
            k,
            v,
            scale,
        )
    return torch.ops.comfy_kitchen.int8_attention_masked(
        q,
        k,
        v,
        attn_mask,
        scale,
    )
