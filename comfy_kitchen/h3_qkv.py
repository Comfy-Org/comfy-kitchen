# SPDX-License-Identifier: Apache-2.0
"""Prequantized H3 projection, RMS/RoPE and INT8 attention preparation."""

import math

import torch

from .backends import cuda
from .sage_attention import PrequantizedInt8Attention, prequantize_int8_attention


def _validate(a, weight, a_scale, weight_scale, rope, q_weight, k_weight, eps):
    if a.ndim != 2:
        raise ValueError("a must be an INT8 matrix [M, 5376]")
    m = a.shape[0]
    expected = (
        (a, (m, 5376), torch.int8),
        (weight, (21504, 5376), torch.int8),
        (a_scale, (m, 1), torch.float32),
        (weight_scale, (21504, 1), torch.float32),
        (rope, (1, m, 1, 48, 2, 2), torch.bfloat16),
        (q_weight, (128,), torch.bfloat16),
        (k_weight, (128,), torch.bfloat16),
    )
    if not a.is_cuda or torch.version.hip or m < 1:
        raise ValueError("H3 QKV preparation requires nonempty CUDA tensors")
    for tensor, shape, dtype in expected:
        if tensor.shape != shape or tensor.dtype != dtype or tensor.device != a.device:
            raise ValueError("H3 QKV shape/dtype/device does not match the 56x128 geometry")
        if tensor.requires_grad:
            raise ValueError("H3 QKV preparation is inference-only")
    if (
        not math.isfinite(eps)
        or not torch.finfo(torch.float32).tiny <= eps <= torch.finfo(torch.float32).max
    ):
        raise ValueError("eps must be finite positive normal FP32")


def _fallback(a, weight, a_scale, weight_scale, rope, q_weight, k_weight, eps):
    from . import rms_rope_split_half_
    from .backends.eager.quantization import mm_int8

    out = torch.empty((a.shape[0], 21504), device=a.device, dtype=torch.bfloat16)
    empty = out.new_empty(0)
    used = False
    if (
        cuda._C is not None
        and not torch.cuda.is_current_stream_capturing()
        and all(t.is_contiguous() for t in (a, weight, a_scale, weight_scale))
    ):
        used = cuda._C.cutlass_int8_dequant(
            *(cuda._wrap_for_dlpack(t) for t in (a, weight, a_scale, weight_scale, empty, out)),
            2,
            torch.cuda.current_stream(a.device).cuda_stream,
        )
    if not used:
        acc = mm_int8(a, weight.T).float()
        out = ((acc * a_scale) * weight_scale.T + 0.0).to(torch.bfloat16)
    q, k, v = out.split(7168, dim=-1)
    q = q.reshape(1, len(a), 56, 128)
    k = k.reshape(1, len(a), 56, 128)
    rms_rope_split_half_(q, k, rope, q_weight, k_weight, epsilon=eps, rot_dim=96)
    v = v.reshape(1, len(a), 56, 128)
    packed = prequantize_int8_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return packed.q, packed.k, packed.v, packed.q_scale, packed.k_scale, packed.v_scale


@torch.library.custom_op("comfy_kitchen::h3_qkv_quant", mutates_args=())
def _op(
    a: torch.Tensor,
    weight: torch.Tensor,
    a_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    rope: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate(a, weight, a_scale, weight_scale, rope, q_weight, k_weight, eps)
    inputs = (a, weight, a_scale, weight_scale, rope, q_weight, k_weight)
    m = len(a)
    if (
        8192 <= m <= 200000
        and cuda._C is not None
        and hasattr(cuda._C, "h3_qkv_quant")
        and torch.cuda.get_device_capability(a.device) == (12, 0)
        and all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in inputs)
    ):
        parts = (m + 127) // 128

        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=a.device)

        qi = empty((1, 56, m, 128), torch.int8)
        ki = torch.empty_like(qi)
        qs = empty((1, 56, parts * 32), torch.float32)
        ks = empty((1, 56, parts * 4), torch.float32)
        partial = empty((56, parts, 128), torch.float32)
        anchors = empty((1, 56), torch.int32)
        sample_q = empty((9, 5376), torch.int8)
        sample_scale = empty((9, 1), torch.float32)
        sample_rope = empty((1, 9, 1, 48, 2, 2), torch.bfloat16)
        sample_out = empty((9, 21504), torch.bfloat16)
        projected = empty((m, 21504), torch.bfloat16)
        vi = empty((56 * 128, parts * 128), torch.int8)
        vs = empty((56 * 128,), torch.float32)
        inverse = torch.empty_like(vs)
        tensors = (
            *inputs,
            qi,
            ki,
            qs,
            ks,
            partial,
            anchors,
            sample_q,
            sample_scale,
            sample_rope,
            sample_out,
            projected,
            vi,
            vs,
            inverse,
        )
        with torch.cuda.device(a.device):
            used = cuda._C.h3_qkv_quant(
                [cuda._wrap_for_dlpack(t) for t in tensors],
                eps,
                torch.cuda.current_stream(a.device).cuda_stream,
            )
        if used:
            return qi, ki, vi, qs, ks, vs
        # A no-CUTLASS build declines before launching. Do not retain scratch
        # and unused result buffers while executing the fallback.
        del tensors, qi, ki, vi, qs, ks, vs, partial, anchors
        del sample_q, sample_scale, sample_rope, sample_out, projected, inverse
    return _fallback(*inputs, eps)


@_op.register_fake
def _fake(a, weight, a_scale, weight_scale, rope, q_weight, k_weight, eps):
    _validate(a, weight, a_scale, weight_scale, rope, q_weight, k_weight, eps)
    m = a.shape[0]
    cta_k = 128 if m > 1024 else 64
    parts = (m + 127) // 128
    kv_parts = (m + cta_k - 1) // cta_k
    return (
        a.new_empty((1, 56, m, 128)),
        a.new_empty((1, 56, m, 128)),
        a.new_empty((56 * 128, kv_parts * cta_k)),
        a.new_empty((1, 56, parts * 32), dtype=torch.float32),
        a.new_empty((1, 56, kv_parts * 4), dtype=torch.float32),
        a.new_empty((56 * 128,), dtype=torch.float32),
    )


def h3_qkv_prequantize(a, weight, a_scale, weight_scale, rope, q_weight, k_weight, *, eps=1e-5):
    """Prepare dense H3 attention from ConvRot-quantized activations/weights.

    Specialized geometry: hidden5376, 56 heads of D128, partial split-half
    RoPE D96; no bias. BF16 projection/RMS/RoPE rounding and Kitchen's current
    nine-key centering criterion are retained. No reuse across steps.
    Returns the usual prequantized attention container, with every key/value.
    """
    q, k, v, qs, ks, vs = _op(a, weight, a_scale, weight_scale, rope, q_weight, k_weight, eps)
    return PrequantizedInt8Attention(
        q, k, v, qs, ks, vs, 128, torch.bfloat16, 128**-0.5, 128 if len(a) > 1024 else 64, None
    )
