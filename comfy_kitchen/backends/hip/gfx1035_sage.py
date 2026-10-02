# SPDX-FileCopyrightText: Copyright (c) 2024 SageAttention team.
# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RDNA2 (gfx103x) SageAttention: INT8 QK^T + FP16 P·V on VALU.

RDNA2 has no matrix cores, so the main HIP backend's sage_attention/*.hip
sources (built on WMMA) do not build or run for gfx103x at all. This module
loads the ported RDNA2 kernels and exposes the same call shape as the external
SageAttention package, so callers can treat it as a drop-in backend.

The gate in :func:`is_available` is deliberately architecture-strict, and it
covers the whole RDNA2 family (gfx1030-1036), not just the gfx1035 the module
and build suffix are named after: every gfx103x part lacks matrix cores, so
every one of them needs this port. The extension is compiled for gfx103x only,
and this module refuses to run on any other target rather than silently
producing wrong results.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
from collections.abc import Sequence

import torch

from . import _visible_gfx_arches, _gfx_arch

# Supported head dimensions for the ported kernels.
_SUPPORTED_HEAD_DIMS = (64, 128)

_module = None
_import_error: Exception | None = None
_ARCH_LOADED: str | None = None


def _detect_gfx103x_arch(device: torch.device | int | None = None) -> str | None:
    """Return the RDNA2 (gfx103x) architecture name of ``device``, or None.

    Covers the whole RDNA2 family, not just gfx1035: gfx1030-1036 all lack matrix
    cores, so all of them need this port and all of them are accepted by the
    extension's CMake target. The historical ``gfx1035`` name lives on in the
    directory, module and build-suffix for compatibility, but the *gate* is
    architecture-class, not a single part.
    """
    arch = _gfx_arch(device)
    return arch if isinstance(arch, str) and arch.startswith("gfx103") else None


def _load_module() -> None:
    """Import the compiled extension; caches success and the first failure."""
    global _module, _import_error, _ARCH_LOADED
    if _module is not None or _import_error is not None:
        return

    arch = _detect_gfx103x_arch()
    if arch is None:
        _import_error = RuntimeError(
            "gfx1035 sage attention: no RDNA2 (gfx103x) device is visible; "
            "this backend is compiled for RDNA2 only."
        )
        return

    directory = os.path.dirname(__file__)
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    path = None
    for name in sorted(os.listdir(directory)):
        if name.startswith("_qattn_gfx1035.") and name.endswith(suffixes):
            path = os.path.join(directory, name)
            break

    if path is None:
        _import_error = RuntimeError(
            "gfx1035 sage attention: extension not built (no _qattn_gfx1035 module "
            "in backends/hip). Rebuild with COMFY_HIP_ARCHS set to this device's RDNA2 "
            "target, e.g. COMFY_HIP_ARCHS=gfx1035."
        )
        return

    try:
        spec = importlib.util.spec_from_file_location(
            "comfy_kitchen.backends.hip._qattn_gfx1035", path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"no extension loader for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["comfy_kitchen.backends.hip._qattn_gfx1035"] = module
        spec.loader.exec_module(module)
    except Exception as e:  # a broken extension must not break import
        if sys.modules.get("comfy_kitchen.backends.hip._qattn_gfx1035") is not None:
            del sys.modules["comfy_kitchen.backends.hip._qattn_gfx1035"]
        _import_error = e
        return

    _module = module
    _ARCH_LOADED = arch


def is_available(device: torch.device | int | None = None) -> bool:
    """Whether the gfx1035 SageAttention path can run for this device."""
    if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
        return False
    if _detect_gfx103x_arch(device) is None:
        return False
    _load_module()
    return _module is not None


def import_error() -> str | None:
    """Why the extension is unavailable, for error messages."""
    _load_module()
    return None if _module is not None else str(_import_error)


def _ops():
    if _module is None:
        raise RuntimeError(f"gfx1035 sage attention unavailable: {_import_error}")
    return torch.ops.comfy_kitchen_qattn_gfx1035


# Mask representations, mirroring sageattn_gfx10::MaskMode in mma_gfx10.h and
# the WMMA backend's own MaskMode. The values are shared on purpose: the RDNA2
# port reads the buffers the main HIP backend's sage_prepare_* kernels produce,
# so a prepared mask is the same object on either backend.
_MASK_NONE = 0
_MASK_RAW = 2
_MASK_PREPARED_KEY = 4
_MASK_PREPARED_DENSE = 5
_MASK_PREPARED_DENSE_BOOL = 7
_MASK_PREPARED_DENSE_BF16 = 8
_MASK_PREPARED_DENSE_F16 = 9

# Dtype codes, matching the main HIP backend's mask helpers: 0 fp32, 1 fp16,
# 2 bf16, 3 bool.
_MASK_DTYPE_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.bool: 3,
}

# The producer's key tile, which is the WMMA backend's, not this kernel's: one
# 64-key tile contributes 64 biases plus one "kept anything" descriptor.
_MASK_TILE_K = 64


def _prepared_mask_width(kv_length: int) -> int:
    """Element width of the 3D key-mask buffer for ``kv_length`` keys.

    Each 64-key tile contributes 64 biases plus one descriptor saying whether
    the tile kept anything at all, and the row is padded so the producer's
    vector loads stay aligned. Mirrors sage_attention.py's allocation.
    """
    tiles = (kv_length + _MASK_TILE_K - 1) // _MASK_TILE_K
    return ((tiles * (_MASK_TILE_K + 1) + 3) // 4) * 4


def _vector_readable(t: torch.Tensor) -> bool:
    """Whether a 16-byte vector load can address every element of ``t``.

    The kernels read whole rows with 16-byte loads whose offsets are computed
    from the strides, so every stride has to be a multiple of 8 halfs and the
    base has to be 16-byte aligned. A tensor from permute/slice often is and
    often is not, so this is asked rather than assumed -- it is what decides
    whether a strided Q can be read in place instead of copied.
    """
    if t.data_ptr() % 16 != 0:
        return False
    return all(stride % 8 == 0 for stride in t.stride()[:-1])


def _prepare_key_mask(mask: torch.Tensor) -> torch.Tensor:
    """Pack a key-broadcast mask into the 3D fp32 layout the kernel reads.

    Delegates to the main HIP backend's producer rather than packing it here: the
    layout is defined by that kernel, and two copies of the same packing would be
    two things to keep in step. It is a pure memory shuffle with no matrix cores,
    so it runs on RDNA2.

    The producer sizes the output buffer from the mask's own batch and head
    strides, so it is handed [B,H,1,K] with a query axis that never had more than
    one row. A mask that arrived already broadcast to the full query length is
    compacted first, because slicing a stride-0 view leaves a stride-0 view and
    the producer would size the buffer from the wrong extents.
    """
    from . import _C, _dl, _stream

    mask_batch = 1 if mask.stride(0) == 0 else mask.shape[0]
    mask_heads = 1 if mask.stride(1) == 0 else mask.shape[1]
    packed = torch.empty(
        (mask_batch, mask_heads, _prepared_mask_width(mask.shape[-1])),
        dtype=torch.float32,
        device=mask.device,
    )
    key_only = mask[:mask_batch, :mask_heads, :1, :]
    if key_only.stride(2) == 0:
        # Still a broadcast on the query axis: materialise the row so the
        # producer sees a real address rather than a zero stride it rejects.
        key_only = key_only.contiguous()
    _C.sage_prepare_key_mask(_dl(key_only), _dl(packed), _stream(mask))
    return packed


def _prepare_attn_mask(
    mask: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[int, int, torch.Tensor]:
    """Turn a caller-supplied mask into a (mode, dtype code, buffer) triple.

    A mask that varies along the query axis is read raw: the kernel indexes it
    with the caller's own strides, so there is no packing pass at all. That is
    the trade the RDNA2 port can make and the WMMA backends cannot -- their MMA
    fragment layout only sees a key's bias from one specific slot, so they have
    to permute the mask into a packed tile first, which is what
    ``sage_int8_sdpa``'s fused short-dense-mask path exists to hide. A strided
    read inside the attention loop replaces that pass outright here.

    A mask that does *not* vary along queries is still packed, because there the
    packed form is strictly less work: one element per key, read once, instead of
    a strided read per key per query block.

    Reading raw and reading the packed form give bit-identical results. Both
    decode the same bias in the same units and add it to the same rounded score
    product, so a fused call and a prequantized snapshot of one mask agree
    exactly -- which test_hip_compact_dense_bool_matches_float_bias pins.
    """
    if mask.dtype not in _MASK_DTYPE_CODE:
        raise TypeError(
            f"attn_mask must be bool, float16, bfloat16 or float32, got {mask.dtype}"
        )
    if mask.device != q.device:
        raise ValueError("attn_mask must be on the same device as q, k and v")
    kv_length = k.shape[2]
    if mask.shape[-1] not in (kv_length, 1):
        raise ValueError(
            f"attn_mask's last extent must be {kv_length} or 1, got {mask.shape[-1]}"
        )

    # A key-only mask is one whose query axis carries no per-query structure:
    # either a single row, or a stride-0 broadcast. Checked before any broadcast
    # view is taken, which would flatten both cases into [B,H,Q,K] and hide the
    # distinction.
    key_only = mask.ndim == 4 and (mask.shape[2] == 1 or mask.stride(2) == 0)
    if key_only and kv_length > _MASK_TILE_K:
        return _MASK_PREPARED_KEY, _MASK_DTYPE_CODE[mask.dtype], _prepare_key_mask(mask)
    # Narrower than one mask tile: there is nothing a packed tile would save,
    # and the kernel already treats the keys past the end as dropped.
    return _MASK_RAW, _MASK_DTYPE_CODE[mask.dtype], mask


def pack_attn_mask(
    mask: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    attention_scale: float,
) -> torch.Tensor:
    """Pack ``mask`` into the snapshot form a prequantized call stores.

    The result is a 3D fp32 buffer in the key layout, or a 5D dense buffer whose
    trailing width says which of the dense forms it is. Callers pass it back with
    ``prequantized=True``, and the rank plus trailing width are what the consumer
    reads to decide how.
    """
    from comfy_kitchen.sage_attention import _prepare_attn_mask as _shared_prepare

    return _shared_prepare(mask, attention_scale)


def _unpack_prepared_mask(
    packed: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[int, int, torch.Tensor]:
    """Recognise a packed mask buffer and return the mode to read it with.

    The packed forms are self-describing by rank and trailing width, which is
    what lets PrequantizedInt8Attention carry a mask as a bare tensor: a 3D
    buffer is the key layout, a 5D buffer is dense with the last extent saying
    whether it holds keep bits or 16-bit values, and a 4D buffer is still a raw
    mask -- the shared preparer declines to pack one that is too narrow to tile.
    """
    if packed.dim() == 4:
        return _MASK_RAW, _MASK_DTYPE_CODE[packed.dtype], packed
    if packed.dim() == 3:
        expected = _prepared_mask_width(k.shape[2])
        if packed.shape[-1] != expected:
            raise ValueError(
                f"prepared key mask must be {expected} wide for kv_length "
                f"{k.shape[2]}, got {packed.shape[-1]}"
            )
        if packed.dtype != torch.float32:
            raise ValueError("prepared key mask must be float32")
        return _MASK_PREPARED_KEY, 0, packed
    if packed.dim() == 5:
        if packed.shape[-1] == 32:
            if packed.dtype != torch.int32:
                raise ValueError("a bit-packed dense mask must be int32")
            return _MASK_PREPARED_DENSE_BOOL, 3, packed
        if packed.shape[-1] != 1024:
            raise ValueError(
                f"prepared dense mask must be 32 (bit-packed) or 1024 wide, "
                f"got {packed.shape[-1]}"
            )
        if packed.dtype == torch.bfloat16:
            return _MASK_PREPARED_DENSE_BF16, 2, packed
        if packed.dtype == torch.float16:
            return _MASK_PREPARED_DENSE_F16, 1, packed
        if packed.dtype == torch.float32:
            return _MASK_PREPARED_DENSE, 0, packed
        raise ValueError(
            f"prepared dense mask must be float32, float16 or bfloat16, got {packed.dtype}"
        )
    raise ValueError(
        "a prepared attn_mask must be 3D (key layout) or 5D (dense layout), got "
        f"{packed.dim()}D"
    )


def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: float | None = None,
    attn_mask: torch.Tensor | None = None,
    prequantized: bool = False,
) -> torch.Tensor:
    """SDPA with INT8 Q·K^T and FP16 P·V on RDNA2.

    Shapes and layout semantics match SageAttention's ``sageattn``:
    ``tensor_layout="HND"`` takes ``[batch, heads, seq, head_dim]`` and
    ``"NHD"`` takes ``[batch, seq, heads, head_dim]``. Only head_dim 64 and 128
    are supported, matching the ported kernels.

    ``attn_mask`` takes the same mask ComfyUI hands to the other backends: bool
    or a float mask broadcastable to ``[batch, heads, q_len, kv_len]``, where a
    False or non-finite entry drops the key. ``prequantized=True`` says the mask
    is already a packed buffer produced by :func:`pack_attn_mask` and is read as
    such; that is how :func:`comfy_kitchen.sage_attention.int8_attention_from_prequantized`
    consumes a snapshot without rebuilding it.
    """
    if not is_available(q.device):
        raise RuntimeError(
            f"gfx1035 sage attention is unavailable: {import_error()}"
        )

    dtype = q.dtype
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"inputs must be fp16, bf16 or fp32, got {dtype}")
    if not (q.dtype == k.dtype == v.dtype):
        raise TypeError("q, k, v must share one dtype")
    if not (q.device == k.device == v.device):
        raise ValueError("q, k, v must be on the same device")
    if not q.is_cuda:
        raise ValueError("q, k, v must be on a CUDA/HIP device")

    head_dim = q.size(-1)
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(f"head_dim must be 64 or 128, got {head_dim}")
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name} must be contiguous along head_dim")

    # Resolve the mask to (mode, dtype code, buffer) once, so the raw and packed
    # forms are the same object to the kernel.
    mask_mode = _MASK_NONE
    mask_dtype = 0
    mask_buffer = torch.empty(0, device=q.device, dtype=q.dtype)
    # True when this call built the mask itself, and so owns a temporary the
    # pending kernel still reads.
    owns_mask_buffer = False
    if attn_mask is not None:
        if prequantized:
            mask_mode, mask_dtype, mask_buffer = _unpack_prepared_mask(attn_mask, q, k)
        else:
            mask_mode, mask_dtype, mask_buffer = _prepare_attn_mask(attn_mask, q, k)
            owns_mask_buffer = True

    input_dtype = dtype
    if dtype == torch.float32:
        q, k, v = (t.half() for t in (q, k, v))
        dtype = torch.float16

    ops = _ops()
    layout_code = 1 if tensor_layout == "HND" else 0

    if sm_scale is None:
        sm_scale = head_dim ** -0.5

    # The prepass quantizer and the V staging read their operands with 16-byte
    # vector loads, so K and V have to be laid out for that.
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()

    # The result is packed [B, S, H, D] and handed back as a [B, H, S, D] view.
    # Every ComfyUI attention path ends with
    # `out.transpose(1, 2).reshape(batch, -1, heads * dim)`, and on
    # HND-contiguous storage that reshape is a full copy of the result: on the
    # short-KV cross-attention shapes it cost as much as the attention itself.
    # The kernel writes through the strides the tensor reports, so producing the
    # layout the caller wants is free, and the reshape becomes a view.
    batch = q.size(0)
    q_heads = q.size(1) if tensor_layout == "HND" else q.size(2)
    qo_len = q.size(2) if tensor_layout == "HND" else q.size(1)
    out = torch.empty(
        batch, qo_len, q_heads, head_dim,
        dtype=dtype if input_dtype != torch.float32 else torch.float16,
        device=q.device,
    )
    o = out.transpose(1, 2) if tensor_layout == "HND" else out

    kv_len = k.size(2) if tensor_layout == "HND" else k.size(1)

    # gfx103x reads native [B,H,N,D] V directly; no transpose is needed, and the
    # v_scale argument is a placeholder the kernel does not read. The kernel
    # itself is FP16-only, so bf16 V is converted here (a single cast kernel,
    # which the direct-fp16 path avoids by folding the cast into its transpose).
    if tensor_layout == "NHD":
        v16 = v if v.dtype == torch.float16 else v.half()
        if v16.is_contiguous():
            b_, n_, h_, d_ = v16.shape
            v_for_attn = v16.as_strided(
                (b_, h_, n_, d_), (n_ * h_ * d_, d_, h_ * d_, 1)
            )
        else:
            v_for_attn = v16.permute(0, 2, 1, 3).contiguous()
    else:
        v_for_attn = v if v.dtype == torch.float16 else v.half()
    # v_scale is a placeholder the ported op's signature carries and the kernel
    # never reads (see qk_int8_sv_bf16_attn_gfx103x_t), so there is no reason to
    # allocate the per-32-key float buffer a quantized-V kernel would want.
    v_scale = torch.empty(0, device=q.device, dtype=torch.float32)

    # smooth_k went with the mean-seq kernel it needed. Nothing in the tree set
    # it, the only caller of sageattn passes nothing, and int8_attention lists it
    # among the options it must reject, so the feature is absent from the public
    # surface rather than merely unused. quant_qk_int8 still takes a key_mean and
    # subtracts it when it gets one; nothing produces one now.
    k_mean = torch.empty(0, device=q.device, dtype=q.dtype)

    # In-kernel Q quantization lets the attention kernel quantize Q itself, so Q
    # never has to be materialized. It is off by default and the env override
    # exists only to measure it, for two reasons.
    #
    # It does not agree with the prepass: the kernel scales each query row
    # against its own amax while the prepass scales MIN_BLK_Q rows together, so
    # the int8 differs and a prequantized snapshot -- which is packed by the
    # prepass -- would not match the fused call it came from.
    # test_prequantized_attention_is_bitwise_identical_to_fused pins that match.
    # And with the output packed it measured no faster anyway (1.80 vs 1.78 ms
    # on SDXL cross-attention at 6144x77x10x64), the Q round-trip not being what
    # the short-KV shapes spend their time on.
    skip_inq = bool(
        head_dim in _SUPPORTED_HEAD_DIMS
        and not is_causal
        and kv_len <= 1024
        and _vector_readable(q)
        and os.getenv("CK_GFX1035_INQ") == "1"
    )
    if not _vector_readable(q):
        # The quantizer reads whole 8-element packs whose offsets come from the
        # strides, so a Q whose strides are not multiples of 8 halfs (or whose
        # base is misaligned) has to be repacked. A Q that is merely *strided* --
        # the usual [B,S,H,D].transpose(1,2) -- is read in place: on the
        # short-KV cross-attention shapes this copy was a third of the call.
        q = q.contiguous()
    q_fp = q
    if skip_inq and tensor_layout == "NHD":
        b_, s_, h_, d_ = q.shape
        q_fp = q.as_strided((b_, h_, s_, d_), (s_ * h_ * d_, d_, h_ * d_, 1))

    q_int8, q_scale, k_int8, k_scale = ops.quant_qk_int8(
        q, k, k_mean, layout_code, float(sm_scale), int(skip_inq)
    )
    ops.qk_int8_sv_bf16_attn_t(
        q_int8, k_int8, v_for_attn, o,
        q_scale, k_scale, v_scale,
        layout_code, int(is_causal), float(sm_scale),
        q_fp if skip_inq else torch.empty(0, device=q.device, dtype=q.dtype),
        mask_mode, mask_dtype, mask_buffer,
    )

    if input_dtype == torch.float32:
        o = o.to(torch.float32)
    if owns_mask_buffer:
        # The packed mask is a temporary built for this call, and the attention
        # kernel that reads it is only *enqueued* here. Dropping the last
        # reference would let the caching allocator hand the block to the next
        # call before the pending kernel has run, which made back-to-back masked
        # calls return different answers. Attaching it to the output keeps it
        # alive for exactly as long as the result that depends on it.
        o._gfx1035_mask = mask_buffer
    return o


def prequantize(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    original_head_dim: int,
    input_dtype: torch.dtype,
    attention_scale: float,
    attn_mask: torch.Tensor | None,
):
    """Quantize Q/K and snapshot the mask, without running attention.

    The RDNA2 kernel reads V straight from memory in fp16 -- it is the
    P·V product, not a quantized one -- so the snapshot keeps V as the fp16
    buffer the kernel wants rather than an int8 copy of it. That is what the
    ``v`` field holds here, and ``v_scale`` stays empty: the kernel has no
    per-row V scale to apply. The Q/K half is the ordinary packed int8 with
    per-row and per-16-key scales, produced by the same quantizer the fused path
    uses, so a snapshot and a fused call over the same inputs agree exactly.
    """
    from comfy_kitchen.sage_attention import PrequantizedInt8Attention

    if not is_available(q.device):
        raise RuntimeError(f"gfx1035 sage attention is unavailable: {import_error()}")

    head_dim = q.size(-1)
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise NotImplementedError(
            "prequantize_int8_attention on RDNA2 (gfx103x) requires head_dim 64 or "
            f"128, got {head_dim}."
        )

    mask_snapshot: torch.Tensor | None = None
    if attn_mask is not None:
        mask_snapshot = pack_attn_mask(attn_mask, q, k, attention_scale)

    work_q = q if q.dtype != torch.float32 else q.half()
    work_k = k if k.dtype != torch.float32 else k.half()
    if not work_q.is_contiguous():
        work_q = work_q.contiguous()
    if not work_k.is_contiguous():
        work_k = work_k.contiguous()

    ops = _ops()
    k_mean = torch.empty(0, device=q.device, dtype=work_k.dtype)
    # skip_q=0: the snapshot is the whole point, so Q has to be packed here
    # rather than left for the attention pass to quantize in place.
    q_int8, q_scale, k_int8, k_scale = ops.quant_qk_int8(
        work_q, work_k, k_mean, 1, float(attention_scale), 0
    )
    v_fp16 = v if v.dtype == torch.float16 else v.half()
    if not v_fp16.is_contiguous():
        v_fp16 = v_fp16.contiguous()

    return PrequantizedInt8Attention(
        q=q_int8,
        k=k_int8,
        v=v_fp16,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=torch.empty(0, device=q.device, dtype=torch.float32),
        original_head_dim=original_head_dim,
        input_dtype=input_dtype,
        attention_scale=attention_scale,
        # cta_k is a WMMA scheduling parameter; the RDNA2 kernel fixes its own
        # tile and ignores it, but the field is part of the shared dataclass.
        cta_k=64,
        attn_mask=mask_snapshot,
    )


def attend_prequantized(quantized) -> torch.Tensor:
    """Attend over a snapshot taken by :func:`prequantize`.

    The packed Q/K go back through the same op the fused path uses, with the
    snapshot's mask handed over already packed, so no part of the result depends
    on the float inputs still being alive.
    """
    if not is_available(quantized.q.device):
        raise RuntimeError(f"gfx1035 sage attention is unavailable: {import_error()}")

    if quantized.q.dtype != torch.int8:
        raise ValueError(
            "a prequantized INT8 attention snapshot on RDNA2 must carry packed "
            f"int8 Q, got {quantized.q.dtype}"
        )

    mask_mode = _MASK_NONE
    mask_dtype = 0
    mask_buffer = torch.empty(0, device=quantized.q.device, dtype=quantized.q.dtype)
    if quantized.attn_mask is not None:
        mask_mode, mask_dtype, mask_buffer = _unpack_prepared_mask(
            quantized.attn_mask, quantized.q, quantized.k
        )

    q = quantized.q
    if not q.is_contiguous():
        q = q.contiguous()
    k = quantized.k
    if not k.is_contiguous():
        k = k.contiguous()

    batch, q_heads, q_length, head_dim = q.shape
    kv_len = k.size(2)
    # The RDNA2 kernel is fp16-only, so a float32 input accumulates and writes
    # fp16 here and is widened on return -- the same thing sageattn() does. Using
    # bf16 instead (as the WMMA backends do, having a bf16 path) would round the
    # result differently, and a snapshot has to match its fused counterpart
    # bit for bit.
    output_dtype = torch.float16 if quantized.input_dtype == torch.float32 \
        else quantized.input_dtype
    o = torch.empty(batch, q_heads, q_length, head_dim, dtype=output_dtype, device=q.device)
    v_scale = torch.empty(
        batch, k.size(1), (kv_len + 31) // 32, device=q.device, dtype=torch.float32
    )
    # q_fp is empty: that is how the op is told Q is already packed int8.
    _ops().qk_int8_sv_bf16_attn_t(
        q, k, quantized.v, o,
        quantized.q_scale, quantized.k_scale, v_scale,
        1, 0, float(quantized.attention_scale),
        torch.empty(0, device=q.device, dtype=torch.float16),
        mask_mode, mask_dtype, mask_buffer,
    )
    o = o[..., : quantized.original_head_dim]
    return o.to(torch.float32) if quantized.input_dtype == torch.float32 else o
