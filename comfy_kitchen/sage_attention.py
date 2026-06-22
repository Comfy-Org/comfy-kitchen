# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SageAttention: INT8 Q/K, FP8 V scaled dot-product attention.

Public API
----------
- ``sage_sdpa``  — end-to-end fused SDPA (the only function most callers need).
- ``is_available`` — runtime check for SM89+ and the compiled CUDA extension.

Internal helpers (exported for component-level testing only):
- ``_quantize_v_fp8``
- ``_quantize_v_fp8_eager``
- ``CTA_K``
"""

from __future__ import annotations

import torch

# Tiling constants — must match the CUDA kernels in
# sage_attention/sage_attn_launcher.cu and dlpack_bindings.cpp.
CTA_K = 64

_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _pad_to_cta_k(n: int) -> int:
    return ((n + CTA_K - 1) // CTA_K) * CTA_K


def is_available() -> bool:
    """Return True if the SageAttention CUDA kernels can run on this device."""
    if not torch.cuda.is_available():
        return False
    try:
        from comfy_kitchen.backends.cuda import _EXT_AVAILABLE

        if not _EXT_AVAILABLE:
            return False
    except ImportError:
        return False
    return torch.cuda.get_device_capability() >= (8, 9)


def _quantize_v_fp8_eager(
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager (pure-PyTorch) V → FP8 E4M3 quantization — reference implementation.

    Output layout matches the CUDA kernel: [B, H, D, padded_N] with FP8 MMA
    16-element permutation applied along padded_N.
    """
    _b, _h, n, _d = v.shape
    padded_n = _pad_to_cta_k(n)

    vt = v.permute(0, 1, 3, 2).contiguous().float()  # [B, H, D, N]
    if padded_n > n:
        vt = torch.nn.functional.pad(vt, (0, padded_n - n))

    amax = vt.abs().amax(dim=-1)  # [B, H, D]
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = amax / fp8_max
    scale = scale.clamp(min=1e-12)

    inv_scale = 1.0 / scale
    vq = (vt * inv_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)

    # FP8 MMA 16-element permutation: for each output position j, read from
    # source position src within the same 16-element group.
    idx = torch.arange(padded_n, device=vq.device)
    w = idx & 15
    src = (idx & ~15) | ((w >> 2) * 2 + ((w >> 1) & 1) * 8 + (w & 1))
    vq = vq[..., src]

    return vq.view(torch.int8), scale


def _quantize_v_fp8(
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize V [B,H,N,D] → FP8 E4M3 [B,H,D,padded_N] + per-channel scale [B,H,D].

    Uses the CUDA kernel ``_C._quant_v_fp8``.
    """
    from comfy_kitchen.backends.cuda import _C, _wrap_for_dlpack
    from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE

    b, h, n, d = v.shape
    padded_n = _pad_to_cta_k(n)

    out = torch.empty(b * h * d, padded_n, dtype=torch.float8_e4m3fn, device=v.device)
    scale = torch.empty(b * h * d, device=v.device, dtype=torch.float32)

    code = DTYPE_TO_CODE[v.dtype]
    stream_ptr = torch.cuda.current_stream(v.device).cuda_stream
    _C._quant_v_fp8(
        _wrap_for_dlpack(v),
        _wrap_for_dlpack(out),
        _wrap_for_dlpack(scale),
        padded_n,
        code,
        stream_ptr,
    )
    return (
        out.view(torch.int8).reshape(b, h, d, padded_n),
        scale.reshape(b, h, d),
    )


def sage_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    smooth_k: bool = True,
) -> torch.Tensor:
    """SageAttention scaled dot-product attention.

    Parameters
    ----------
    q : Tensor [B, H_Q, N_Q, D]   (fp16 or bf16)
    k : Tensor [B, H_K, N_K, D]   (same dtype as q)
    v : Tensor [B, H_K, N_K, D]   (same dtype as q)
    is_causal : bool
    smooth_k : bool  — subtract per-head K mean before quantisation

    Returns
    -------
    Tensor [B, H_Q, N_Q, D]  (same dtype as q)
    """
    from comfy_kitchen.backends.cuda import _C, _wrap_for_dlpack
    from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE

    b, h_q, n_q, d = q.shape
    _, h_k, n_k, _ = k.shape
    if q.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"q.dtype must be float32, float16, or bfloat16, got {q.dtype}")
    if d not in (64, 128):
        raise ValueError(f"head_dim must be 64 or 128, got {d}")
    if h_q % h_k != 0:
        raise ValueError(f"num_qo_heads ({h_q}) must be divisible by num_kv_heads ({h_k})")

    # Tiling parameters — must match sage_attn_launcher.cu and dlpack_bindings.cpp.
    blkq, warpq, blkk, warpk = 128, 32, 64, 64
    padded_n_k = _pad_to_cta_k(n_k)

    q_int8 = torch.empty_like(q, dtype=torch.int8)
    k_int8 = torch.empty_like(k, dtype=torch.int8)
    q_scale = torch.empty(
        b,
        h_q,
        ((n_q + blkq - 1) // blkq) * (blkq // warpq) * 8,
        device=q.device,
        dtype=torch.float32,
    )
    k_scale = torch.empty(
        b,
        h_k,
        ((n_k + blkk - 1) // blkk) * (blkk // warpk) * 4,
        device=q.device,
        dtype=torch.float32,
    )
    v_quant = torch.empty(
        b * h_k * d,
        padded_n_k,
        dtype=torch.float8_e4m3fn,
        device=q.device,
    )
    v_scale = torch.empty(b * h_k * d, device=q.device, dtype=torch.float32)

    original_dtype = q.dtype
    if q.dtype == torch.float32:
        output_dtype = torch.bfloat16
    else:
        output_dtype = q.dtype

    output = torch.empty(b, h_q, n_q, d, dtype=output_dtype, device=q.device)

    km_scratch_ptr = 0
    km_done_ptr = 0
    if smooth_k:
        km_scratch = torch.empty(b * h_k * d, device=q.device, dtype=torch.float32)
        km_done = torch.empty(b * h_k, device=q.device, dtype=torch.int32)
        km_scratch_ptr = km_scratch.data_ptr()
        km_done_ptr = km_done.data_ptr()

    input_dtype_code = DTYPE_TO_CODE[original_dtype]
    output_dtype_code = DTYPE_TO_CODE[output_dtype]
    stream_ptr = torch.cuda.current_stream(q.device).cuda_stream

    _C.sage_sdpa(
        _wrap_for_dlpack(q),
        _wrap_for_dlpack(k),
        _wrap_for_dlpack(v),
        _wrap_for_dlpack(output),
        _wrap_for_dlpack(q_int8),
        _wrap_for_dlpack(q_scale),
        _wrap_for_dlpack(k_int8),
        _wrap_for_dlpack(k_scale),
        _wrap_for_dlpack(v_quant),
        _wrap_for_dlpack(v_scale),
        int(is_causal),
        d**-0.5,
        input_dtype_code,
        output_dtype_code,
        stream_ptr,
        int(smooth_k),
        km_scratch_ptr,
        km_done_ptr,
    )
    if original_dtype == torch.float32:
        return output.to(torch.float32)
    return output
