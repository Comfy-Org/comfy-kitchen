# SPDX-License-Identifier: Apache-2.0
"""Prequantized INT8 GEMM with a compact per-token gate and residual."""

import torch


def _validate(a, b, x_scale, w_scale, gate, row_indices, residual):
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("a and b must have shapes [M, K] and [N, K]")
    m, n = a.shape[0], b.shape[0]
    if a.dtype != torch.int8 or b.dtype != torch.int8:
        raise ValueError("a and b must be int8")
    if x_scale.dtype != torch.float32 or w_scale.dtype != torch.float32:
        raise ValueError("scales must be float32")
    if x_scale.shape not in ((m,), (m, 1)) or w_scale.shape != (n,):
        raise ValueError("x_scale must be [M] or [M, 1]; w_scale must be [N]")
    if gate.ndim != 2 or gate.shape[1] != n or residual.shape != (m, n):
        raise ValueError("gate must be [G, N] and residual must be [M, N]")
    if gate.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
        raise ValueError("gate and residual must be bfloat16")
    if row_indices.dtype != torch.int32 or row_indices.shape != (m,):
        raise ValueError("row_indices must be int32 [M]")
    for tensor in (b, x_scale, w_scale, gate, row_indices, residual):
        if tensor.device != a.device:
            raise ValueError("all tensors must be on the same device")
    if any(t.requires_grad for t in (x_scale, w_scale, gate, residual)):
        raise ValueError("int8_gemm_indexed_gate is inference-only")
    # INT8 dot products must fit int32, independently of the chosen backend.
    if a.shape[1] > 131071:
        raise ValueError("K exceeds the worst-case int32 accumulator bound")


def _gather_gate(gate, row_indices):
    if gate.shape[0] == 0:
        return gate.new_zeros((row_indices.numel(), gate.shape[1]))
    valid = (row_indices >= 0) & (row_indices < gate.shape[0])
    selected = gate[row_indices.clamp(0, gate.shape[0] - 1).long()]
    return torch.where(valid[:, None], selected, 0)


def _fallback(a, b, x_scale, w_scale, gate, row_indices, residual):
    from .backends.eager.quantization import mm_int8

    if a.shape[0] == 0 or b.shape[0] == 0 or a.shape[1] == 0:
        acc = torch.zeros((a.shape[0], b.shape[0]), device=a.device, dtype=torch.float32)
    else:
        acc = mm_int8(a, b.t()).float()
    branch = ((acc * x_scale.reshape(-1, 1)) * w_scale + 0.0).to(torch.bfloat16)
    return torch.addcmul(residual, branch, _gather_gate(gate, row_indices)).contiguous()


def _cuda_plain_fallback(a, b, x_scale, w_scale, gate, row_indices, residual):
    from .backends import cuda

    if torch.cuda.is_current_stream_capturing():
        return _fallback(a, b, x_scale, w_scale, gate, row_indices, residual)
    branch = torch.empty_like(residual)
    empty_bias = torch.empty(0, dtype=torch.bfloat16, device=a.device)
    stream = torch.cuda.current_stream(a.device).cuda_stream
    used = cuda._C.cutlass_int8_dequant(
        *(cuda._wrap_for_dlpack(t) for t in (a, b, x_scale, w_scale, empty_bias, branch)),
        2,
        stream,
    )
    if used:
        return torch.addcmul(residual, branch, _gather_gate(gate, row_indices)).contiguous()
    return _fallback(a, b, x_scale, w_scale, gate, row_indices, residual)


@torch.library.custom_op("comfy_kitchen::int8_gemm_indexed_gate", mutates_args=())
def _op(
    a: torch.Tensor,
    b: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    gate: torch.Tensor,
    row_indices: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    from .backends import cuda

    _validate(a, b, x_scale, w_scale, gate, row_indices, residual)
    tensors = (a, b, x_scale, w_scale, gate, row_indices, residual)
    if a.shape[0] == 0 or b.shape[0] == 0 or a.shape[1] == 0:
        return _fallback(*tensors)
    if (
        a.is_cuda
        and not torch.version.hip
        and cuda._C is not None
        and hasattr(cuda._C, "cutlass_int8_indexed_gate")
        and all(t.is_contiguous() for t in tensors)
    ):
        out = torch.empty_like(residual)
        with torch.cuda.device(a.device):
            stream = torch.cuda.current_stream(a.device).cuda_stream
            if (a.shape[0] >= 4096 or a.shape[0] <= 64) and cuda._C.cutlass_int8_indexed_gate(
                *(cuda._wrap_for_dlpack(t) for t in (*tensors, out)), stream
            ):
                return out
            return _cuda_plain_fallback(*tensors)
    return _fallback(*tensors)


@_op.register_fake
def _fake(a, b, x_scale, w_scale, gate, row_indices, residual):
    _validate(a, b, x_scale, w_scale, gate, row_indices, residual)
    return residual.new_empty(residual.shape)


def int8_gemm_indexed_gate(a, b, x_scale, w_scale, gate, row_indices, residual):
    """Compute a prequantized linear branch, indexed gate and residual.

    ``branch = bf16(float32(A @ B.T) * x_scale[:, None] * w_scale)``;
    ``out = addcmul(residual, branch, gate[row_indices])``. The BF16 branch
    rounding happens before the FP32 FMA and final BF16 rounding. No bias,
    quantization or rotation is performed here. Existing ``int8_linear`` and
    its column-broadcast residual semantics are unchanged.

    A: INT8 [M,K]; B: INT8 [N,K]; scales: FP32 [M] (or [M,1]) and [N];
    gate: BF16 [G,N]; row_indices: INT32 [M]; residual: BF16 [M,N]. All tensors
    must share a device. Gate indices outside [0,G) select zero, including
    G=0; this avoids an index-validation host synchronization. Inputs are not
    modified. The output is contiguous. Noncontiguous inputs and unsupported devices use a PyTorch
    fallback. The fused initial implementation targets SM120, K%16=0, N%8=0.
    Inference only; an opaque custom op supports torch.compile and CUDA Graphs.
    """
    return _op(a, b, x_scale, w_scale, gate, row_indices, residual)


def _validate_linear(x, weight, weight_scale, gate, row_indices, residual, input_act):
    if x.ndim != 2 or x.dtype != torch.bfloat16:
        raise ValueError("x must be a BF16 matrix")
    if input_act not in (None, "swiglu"):
        raise ValueError("input_act must be None or swiglu")
    k = x.shape[1] // (2 if input_act == "swiglu" else 1)
    if k == 0 or k % 256 or k > 131071 or x.shape[1] != k * (2 if input_act == "swiglu" else 1):
        raise ValueError("activated K must be positive and divisible by 256")
    if x.requires_grad:
        raise ValueError("int8_linear_indexed_gate is inference-only")
    if weight.ndim != 2 or weight.shape[1] != k or weight.dtype != torch.int8:
        raise ValueError("weight must be INT8 [N,K]")
    m, n = x.shape[0], weight.shape[0]
    if weight_scale.dtype != torch.float32 or weight_scale.shape != (n,):
        raise ValueError("weight_scale must be FP32 [N]")
    if gate.ndim != 2 or gate.shape[1] != n or residual.shape != (m, n):
        raise ValueError("gate must be [G,N] and residual [M,N]")
    if gate.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
        raise ValueError("gate and residual must be BF16")
    if row_indices.dtype != torch.int32 or row_indices.shape != (m,):
        raise ValueError("row_indices must be INT32 [M]")
    if any(t.device != x.device for t in (weight, weight_scale, gate, row_indices, residual)):
        raise ValueError("all tensors must share a device")
    if any(t.requires_grad for t in (weight_scale, gate, residual)):
        raise ValueError("int8_linear_indexed_gate is inference-only")


@torch.library.custom_op("comfy_kitchen::int8_linear_indexed_gate", mutates_args=())
def int8_linear_indexed_gate(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    gate: torch.Tensor,
    row_indices: torch.Tensor,
    residual: torch.Tensor,
    input_act: str | None = None,
) -> torch.Tensor:
    """ConvRot256 INT8 linear followed by an indexed BF16 gate and residual.

    No bias; x is BF16 [M,K] (or [M,2K] for ``input_act="swiglu"``).
    Weight is an already ConvRot-quantized INT8 [N,K] matrix. The optional
    SwiGLU retains the BF16 SiLU and product roundings of ``int8_linear``.
    """
    from .backends import cuda
    from . import int8_linear

    _validate_linear(x, weight, weight_scale, gate, row_indices, residual, input_act)
    k = x.shape[1] // (2 if input_act == "swiglu" else 1)
    if (
        x.is_cuda
        and not torch.version.hip
        and cuda._C is not None
        and k <= cuda._CONVROT_FUSED_MAX_K
        and cuda._convrot_fused_shared_memory_fits(x, k, 256)
    ):
        q, qs = cuda.quantize_int8_rowwise_convrot64(x.contiguous(), 256, input_act=input_act)
    else:
        branch = int8_linear(x, weight, weight_scale, convrot=True, input_act=input_act)
        return torch.addcmul(residual, branch, _gather_gate(gate, row_indices)).contiguous()
    return int8_gemm_indexed_gate(
        q, weight, qs.reshape(-1, 1), weight_scale, gate, row_indices, residual
    )


@int8_linear_indexed_gate.register_fake
def _linear_fake(x, weight, weight_scale, gate, row_indices, residual, input_act=None):
    _validate_linear(x, weight, weight_scale, gate, row_indices, residual, input_act)
    return residual.new_empty(residual.shape)
