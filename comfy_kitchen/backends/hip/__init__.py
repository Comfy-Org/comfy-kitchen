# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HIP backend for AMD RDNA2, RDNA3/3.5, RDNA4 and gfx117x.

Every matmul is a WMMA kernel compiled from the sources in this directory; the
backend does not link or call hipBLAS/hipBLASLt.

RDNA3 (gfx11xx) and RDNA4 (gfx12xx) have matrix cores and get everything. Their
fragment layouts differ, and RDNA3 has no fp8 WMMA, so it widens fp8 to bf16;
see mma.h.

RDNA2 (gfx103x) has no matrix cores. It runs the elementwise kernels (RoPE,
AdaLN and RMS-AdaLN, the quantizers, stochastic rounding, the AWQ GEMV) and
declines the GEMMs, which fall through to triton/eager.
"""
import functools
import importlib.machinery
import importlib.util
import json
import logging
import os
import pathlib
import sys
import weakref
from collections.abc import Sequence
from contextlib import nullcontext

import torch

from comfy_kitchen._rope_utils import _tensors_overlap, check_rope_inplace, trim_rope_freqs
from comfy_kitchen.allocation import allocation_context
from comfy_kitchen.backends import eager as _eager
from comfy_kitchen.backends._activations import apply_input_act as _apply_input_act
from comfy_kitchen.backends._activations import apply_residual as _apply_residual
from comfy_kitchen.backends._activations import input_act_code as _input_act_code
from comfy_kitchen.backends._activations import input_act_width as _input_act_width
from comfy_kitchen.backends.eager import rope as _eager_rope
from comfy_kitchen.backends.eager.quantization import DTYPE_CODE_TO_DTYPE, DTYPE_TO_CODE
from comfy_kitchen.backends.eager.sol_attn import (
    _LOG2E,
    _block_lengths,
    _normalize_key_bias,
    _sink_count,
    _topk_count,
    _valid_rows,
    add_coarse_,
    coarse_output,
)
from comfy_kitchen.backends.eager.w4a8_int8 import (
    _QUANT_ROW_ELEM_BUDGET,
    _decide_codebook,
    _dequantize_w4a8_int8_weight_from_int8,
    _quantize_w4a8_chunked,
    _w4a8_geometry,
    validate_w4a8_operands,
    validate_w4a8_weight_shape,
)
from comfy_kitchen.float_utils import roundup

logger = logging.getLogger("comfy_kitchen.hip")

__all__ = [
    "adaln",
    "na3d",
    "rms_adaln",
    "fp16_conv3d",
    "fp16_conv3d_out",
    "fp16_linear",
    "gemv_awq_w4a16",
    "rms_gated_residual",
    "group_norm_silu_pad3d",
    "group_norm_silu_pad3d_out",
"quantize_nvfp4",
    "dequantize_nvfp4",
    "scaled_mm_nvfp4",
    "group_norm_stats3d",
    "quantize_svdquant_w4a4",
    "scaled_mm_svdquant_w4a4",
    "apply_rope",
    "apply_rope_",
    "apply_rope1",
    "apply_rope1_",
    "apply_rope_split_half",
    "apply_rope_split_half_",
    "apply_rope_split_half1",
    "apply_rope_split_half1_",
    "convrot_w4a4_linear",
    "dequantize_convrot_w4a4_weight",
    "dequantize_int8_convrot_weight_dtype",
    "dequantize_int8_simple_dtype",
    "dequantize_per_tensor_fp8",
    "dequantize_w4a8_int8_weight",
    "has_valu_gemm",
    "has_wmma",
    "int8_linear",
    "int8_attention_is_available",
    "flash_attention_decode_is_available",
    "flash_attention_decode_gqa_is_available",
    "flash_decode",
    "int8_linear_modulated",
    "int8_linear_rms_modulated",
    "flash_decode_gqa",
    "flash_decode_tree_merge",
    "gated_delta_decode_fused",
    "gated_delta_decode_is_available",
    "deltanet_conv_step",
    "gated_delta_deferred_is_available",
    "gated_delta_decode_deferred",
    "deltanet_conv_deferred",
    "is_available",
    "sage_int8_attend",
    "sage_int8_quantize",
    "sage_int8_sdpa",
    "quantize_and_rotate_rowwise",
    "quantize_convrot_w4a4_weight",
    "quantize_int8_convrot_weight",
    "quantize_int8_rowwise",
    "quantize_int8_tensorwise",
    "quantize_per_tensor_fp8",
    "quantize_w4a8_int8_weight",
    "rms_rope",
    "rms_rope_",
    "rms_rope1",
    "rms_rope1_",
    "rms_rope_split_half",
    "rms_rope_split_half_",
    "rms_rope_split_half1",
    "rms_rope_split_half1_",
    "scaled_mm_fp8",
    "sol_attn",
    "sol_attn_chunked",
    "stochastic_rounding_fp8",
    "w4a8_int8_linear",
]

_C = None
_EXT_AVAILABLE = False
_EXT_ERROR = None

try:
    _dir = os.path.dirname(__file__)
    # Only consider suffixes this interpreter can actually load. A stale
    # cross-platform artifact (a Windows .pyd left in the tree by a cross build)
    # is otherwise picked up by listdir order, and spec_from_file_location then
    # returns None for it, which used to surface as a bare AttributeError below
    # and left the backend silently disabled.
    _ext_suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    _module_path = None
    for _fn in sorted(os.listdir(_dir)):
        if _fn.startswith("_C.") and _fn.endswith(_ext_suffixes):
            _module_path = os.path.join(_dir, _fn)
            break

    if _module_path is None:
        _EXT_ERROR = "HIP extension not built (no _C module in backends/hip)"
    else:
        _spec = importlib.util.spec_from_file_location("comfy_kitchen.backends.hip._C", _module_path)
        if _spec is None or _spec.loader is None:
            raise ImportError(f"no extension loader for {_module_path}")
        _C = importlib.util.module_from_spec(_spec)
        sys.modules["comfy_kitchen.backends.hip._C"] = _C
        _spec.loader.exec_module(_C)
        _EXT_AVAILABLE = True
except Exception as e:  # a broken extension must not break import
    # exec_module can fail after the module was cached above, leaving a
    # half-initialized _C importable from sys.modules. Drop it so a later import
    # does not pick up the broken object.
    if _C is not None and sys.modules.get("comfy_kitchen.backends.hip._C") is _C:
        del sys.modules["comfy_kitchen.backends.hip._C"]
    _EXT_ERROR = f"Failed to load HIP extension: {e}"
    _C = None


def _gfx_arch(device: torch.device | int | None = None) -> str | None:
    """gfx architecture of ``device``, e.g. "gfx1201". Defaults to the current device."""
    if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
        return None
    try:
        return torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    except Exception:
        return None


def _is_small_igpu(device: torch.device | int | None = None) -> bool:
    """True on the small RDNA3 iGPU (Radeon 780M, gfx1103) whose 6-WGP tuning
    this tree carries. dGPUs keep the upstream schedules: several block-size
    and dispatch choices measured here are a regression on 60-96 CU parts.

    Resolved for ``device`` (defaulting to the current device) and cached per
    device index (matching the C++ comfy_small_igpu), so a mixed iGPU + dGPU box
    answers for the device a tensor actually lives on, not whichever was current
    the first time this ran.
    """
    if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
        return False
    if device is None:
        index = torch.cuda.current_device()
    elif isinstance(device, int):
        index = device
    else:
        index = device.index
        if index is None:
            # torch.device("cuda") means the current device, same as None -- the
            # docstring above has always said so. Returning False here instead made
            # that a lie, and silently disabled anything gated on it: the int8
            # attention key-tile override returned 64 even on a 780M with
            # COMFY_SAGE_CTA_K=128 set. No existing caller was affected, because a
            # tensor's .device always carries an index, which is why this stayed
            # hidden.
            index = torch.cuda.current_device()
    return _is_small_igpu_index(index)


@functools.cache
def _is_small_igpu_index(index: int) -> bool:
    arch = _gfx_arch(index)
    return arch is not None and arch == "gfx1103"


@functools.lru_cache(maxsize=1)
def _visible_gfx_arches() -> tuple[str | None, ...]:
    """One entry per visible device; None where the architecture could not be read.

    Cheap, but not free and not because of anything in here: the visible device set
    is fixed for the life of the process, and torch caches get_device_properties, so
    the whole call is a current_device(), a device_count() and a tuple build. It was
    measured at 0.05 us against 2.97 us for a bare _gfx_arch(0), which is the whole
    argument for calling it once per predicate rather than once per term -- see
    has_wmma and has_valu_gemm, which both hoist it for exactly this reason.
    """
    if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
        return ()

    # ROCm lazy initialization: device_count() may return 0 until HIP is initialized.
    torch.cuda.current_device()

    return tuple(_gfx_arch(i) for i in range(torch.cuda.device_count()))


# Why the launchers below need the *current* device pinned: launch_int8_gemm_kernel,
# gemm_fp16's launcher and convrot_w4a4's all pick their tile family from
# comfy_is_gfx10(), which reads the current device, while the stream they are handed
# comes from _stream(x) and names a stream on x.device. On a box holding both an
# RDNA2 iGPU and a matrix-core dGPU those can be different devices, and the kernel
# then runs a v_dot4 tile that traps on the operand's part (or the WMMA tile on the
# iGPU).
#
# Entering torch.cuda.device() is not free, though: measured 1.87 us per call on
# gfx1103 (1.42 us of it just constructing the context manager, against 0.03 us for
# a bare `pass`). The quantized GEMMs call this once per layer, so on an SDXL
# txt2img that is ~700 pin/unpin pairs per step and ~39 ms per image of pure
# overhead. With a single visible device the pin is provably a no-op -- x.device is
# that device and current_device() is always it -- so it is only entered when the
# box can actually disagree.
@functools.cache
def _device_pin_can_matter() -> bool:
    """Whether a caller's tensor can live on a device other than the current one."""
    # torch.cuda.current_device() first, for ROCm lazy init: device_count() may
    # return 0 until HIP is initialized. Same ordering as _visible_gfx_arches.
    torch.cuda.current_device()
    return torch.cuda.device_count() > 1


def _device_pin(device: torch.device | int):
    """Context manager pinning ``device`` as current, free when it cannot matter."""
    if _device_pin_can_matter():
        return torch.cuda.device(device)
    return nullcontext()


# RDNA2 has no matrix cores; RDNA3/3.5, gfx117x and RDNA4 do. This exact manifest
# is also consumed by setup.py and CMake. Never infer support from a gfx prefix: a new
# compiler-recognized target needs its WMMA policy reviewed before it is safe.
_ARCH_MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "architectures.json")
_ARCH_GROUPS = json.loads(
    pathlib.Path(_ARCH_MANIFEST_PATH).read_text(encoding="utf-8")
)
_ARCH_ELEMENTWISE_ONLY = frozenset(_ARCH_GROUPS["elementwise_only"])
_ARCH_WMMA_GFX11 = frozenset(_ARCH_GROUPS["wmma_gfx11"])
_ARCH_WMMA_GFX117 = frozenset(_ARCH_GROUPS.get("wmma_gfx117", []))
_ARCH_WMMA_GFX12 = frozenset(_ARCH_GROUPS["wmma_gfx12"])
_ARCH_WMMA = _ARCH_WMMA_GFX11 | _ARCH_WMMA_GFX117 | _ARCH_WMMA_GFX12
_ARCH_SUPPORTED = _ARCH_ELEMENTWISE_ONLY | _ARCH_WMMA

# The RDNA2 manifest group, read back out of the same file rather than spelled as a
# literal list, because this gate has to move in lockstep with _has_wmma's: both
# derive from architectures.json, and a second hand-written gfx103x list is a second
# place for the manifest and the gate to disagree.
_ARCH_VALU_GEMM = _ARCH_ELEMENTWISE_ONLY

# The GEMMs that need matrix cores and have no RDNA2 fallback. Everything else
# either has a VALU path behind __GFX10__ (int8_linear, fp16_linear,
# w4a8_int8_linear, convrot_w4a4_linear) or runs on any supported architecture.
# This set names the registry-dispatched GEMMs that _build_constraints drops on
# RDNA2; the fp8 GEMM is not among them because it is reached through
# scaled_mm_v2's _hip_fp8_gemm, which gates on has_wmma() itself rather than
# through the registry.
#
# w4a8_int8_linear used to be here and was wrong. Its GEMM is not one of the
# WMMA-only kernels: w4a8_dequant.hip's launcher finishes each decoded column
# chunk with launch_int8_gemm_kernel, and that one already branches to the VALU
# tile on comfy_is_gfx10() (ops/gemm_int8.hip). The grouped INT4/INT6 decode is
# architecture-independent and templates on BITS, so the whole W4A8 and W6A8
# stack runs on RDNA2 as it stands -- it was only unreachable, which sent those
# models to triton and made a dequantize-the-whole-weight-then-Triton-GEMM pass
# out of a decode the HIP backend does in column chunks. Measured on the 6-CU
# gfx1035 at the Anima W4A8 shapes: 8.85 -> 6.55 ms per (4096, 2048, 2048) call.
#
# convrot_w4a4_linear is here for the same reason of arrival and not of truth:
# it reached for MmaInt4, and MmaInt4 is a __builtin_trap() without matrix
# cores. Its launcher now branches on the launching device the way
# launch_int8_gemm_kernel does, so RDNA2 reaches gemm_int4_valu_kernel -- the
# same product sum on v_dot8_i32_i4, ~1.9x the int8 tile here -- and everything
# else reaches MmaInt4 exactly as before. Removing it here only stops the whole
# op being withheld from RDNA2; the WMMA path is untouched.
#
# scaled_mm_nvfp4 is in this set for a different reason than the rest. The WMMA tile
# kernel needs matrix cores, and mma.h's no-matrix-core fallback for MmaF16::mma is
# __builtin_trap() rather than a plausible wrong answer -- so on an architecture with
# neither matrix cores nor a VALU NVFP4 kernel, that kernel is compiled (the TU must
# build in every device pass) but must never be launched.
#
# RDNA2 is no longer that architecture. nvfp4_gfx10.h gives it a VALU GEMM on
# v_dot4_i32_i8 -- a GEMV for M <= 4 and a register tile above it, both reached
# behind the same comfy_is_gfx10() gate as every other gfx103x VALU path -- so
# _WMMA_ONLY_OPS drops scaled_mm_nvfp4 only for devices outside _ARCH_VALU_GEMM, and
# _build_constraints puts it back for those inside it. The distinction the set is
# drawing is "this op's only implementation is a matrix-core kernel" and not "this op
# needs a matrix core", which is what w4a8_int8_linear and convrot_w4a4_linear were
# both wrongly filed under above.
#
# quantize_nvfp4 and dequantize_nvfp4 stay out of this set deliberately: they are
# VALU and LDS work with no matrix instruction in them, and they are the ops that
# make the packed representation usable at all when the GEMM has to fall back.
#
# int8_linear_modulated and int8_linear_rms_modulated are NOT in the set, although
# upstream lists them. Upstream's int8_linear has no RDNA2 path, so their fallback
# into it is a matrix-core kernel there and withholding them is the only safe
# answer. Here int8_linear reaches the VALU tile on gfx10, and both modulated ops
# fall back into int8_linear whenever their own fused path (gfx12 only) is
# unavailable, so RDNA2 can serve them.

def _has_nonduplicated_wmma(device: torch.device | int | None = None) -> bool:
    """Whether WMMA operands use the gfx12 128-bit layout."""
    return _gfx_arch(device) in (_ARCH_WMMA_GFX12 | _ARCH_WMMA_GFX117)


_WMMA_ONLY_OPS = frozenset({
    "fp16_conv3d",
    "fp16_conv3d_out",
    "na3d",
    "sol_attn",
    "scaled_mm_svdquant_w4a4",
    "scaled_mm_nvfp4",
})

# The subset of _WMMA_ONLY_OPS that has a VALU implementation on _ARCH_VALU_GEMM, and
# so is re-advertised there. Only scaled_mm_nvfp4 is in it: the others' VALU
# fallbacks (conv3d, na3d, sol_attn, svdquant w4a4) do not exist on this backend.
_VALU_GEMM_OPS = frozenset({
    "scaled_mm_nvfp4",
})


def _unsupported_arch_reason(arches: Sequence[str | None]) -> str | None:
    """Why the backend must not register for this set of devices, or None if it may.

    A device whose architecture cannot be read counts against it: it cannot be
    shown to be supported.
    """
    if not arches:
        return "no HIP device available"
    if any(a is None for a in arches):
        return "could not read the architecture of every visible device"
    unsupported = sorted({a for a in arches if a not in _ARCH_SUPPORTED})
    if unsupported:
        return f"architecture is not in the validated target manifest: {', '.join(unsupported)}"
    return None


def _has_wmma(arches: Sequence[str | None]) -> bool:
    """Whether every visible device has matrix cores.

    Registration is per-process while kernels launch on the tensor's own device, so
    the capability set has to be the intersection over the visible devices: one
    RDNA2 card in an otherwise RDNA4 box means no GEMM is safe to advertise.
    """
    return bool(arches) and all(a in _ARCH_WMMA for a in arches)


def _has_valu_gemm(arches: Sequence[str | None]) -> bool:
    """Whether every visible device has the RDNA2 VALU GEMM set.

    The same intersection _has_wmma takes and for the same reason: registration is
    per-process while kernels launch on the tensor's own device, so the capability
    set has to hold for all of them. False on a mixed box, which is what sends an
    NVFP4 matmul back to dequantize there -- the safe answer, and the same one
    _has_wmma gives.
    """
    return bool(arches) and all(a in _ARCH_VALU_GEMM for a in arches)


def is_available() -> bool:
    return _EXT_AVAILABLE and _unsupported_arch_reason(_visible_gfx_arches()) is None


def has_valu_gemm() -> bool:
    """Whether every visible device has the RDNA2 VALU GEMM set (gfx103x).

    False when the extension is unavailable, when the architectures are not all in
    the validated manifest, or when the process has a mix of RDNA2 and matrix-core
    cards -- the same intersection :func:`has_wmma` takes, and for the same reason:
    registration is per-process while kernels launch on the tensor's own device.

    This is what makes the NVFP4 GEMM reachable on RDNA2. ``scaled_mm_nvfp4``'s
    launcher branches on ``comfy_is_gfx10()`` and reaches nvfp4_gfx10.h's GEMV or
    register tile there, so the op is advertised on gfx103x as well as on the
    matrix-core parts; this predicate is the Python-side half of that, and
    :func:`comfy_kitchen.tensor.base.nvfp4_mm_has_fast_path` is what consults it.

    One arch snapshot per call, and the snapshot is hoisted for the same reason
    :func:`has_wmma` hoists it: this sits on the per-matmul dispatch path, and
    ``_visible_gfx_arches`` is not free -- it reads the current device and builds a
    properties object per device. Calling it twice is not a style question either.
    It is not what made a call cost 170 us on this stack, but it is free to be
    wrong here and the fix is one line.
    """
    arches = _visible_gfx_arches()
    return (
        _EXT_AVAILABLE
        and _unsupported_arch_reason(arches) is None
        and _has_valu_gemm(arches)
    )


def has_wmma() -> bool:
    """Whether the GEMM kernels can run: every visible device has matrix cores.

    is_available() is true on RDNA2 as well, where only the elementwise kernels
    exist, so callers that reach a GEMM without going through the registry (see
    scaled_mm_v2) have to test this instead. One arch snapshot per call: this sits
    on the per-GEMM dispatch path.
    """
    arches = _visible_gfx_arches()
    return (
        _EXT_AVAILABLE
        and _unsupported_arch_reason(arches) is None
        and _has_wmma(arches)
    )


def _stream(t: torch.Tensor) -> int:
    return torch.cuda.current_stream(t.device).cuda_stream


# The epilogues read a scalar per element with one dtype code.
_EPILOGUE_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _aligned(t: torch.Tensor) -> torch.Tensor:
    """``t`` on a 16-byte boundary, which the GEMMs' uint4 row loads require.

    contiguous() keeps a slice whose storage offset is not aligned, so a fresh
    allocation is the only way to move the base.
    """
    return t.clone() if t.data_ptr() % 16 else t


def _operand(t: torch.Tensor, device: torch.device, name: str, shape=None) -> torch.Tensor:
    """A contiguous view of ``t`` on ``device``, since the kernels take raw pointers.

    Every launch uses one stream and one set of extents, so an operand left on
    another device would be dereferenced there, and a mis-shaped one read past its
    end.
    """
    if shape is not None and tuple(t.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(t.shape)}")
    return _aligned(t.to(device=device).contiguous())


def _scale_operand(scale: torch.Tensor, device: torch.device) -> torch.Tensor:
    """The fp8 kernels read one float scale off a raw pointer on the launch stream."""
    scale = scale.reshape(-1)
    if scale.numel() != 1:
        raise ValueError(f"expected a single per-tensor scale, got {scale.numel()} elements")
    return scale.to(device=device, dtype=torch.float32).contiguous()


def _bias_operand(bias: torch.Tensor, n: int, device: torch.device) -> torch.Tensor:
    """The epilogue indexes bias[col], so it must be 1D of length N and decodable."""
    if bias.dim() != 1 or bias.numel() != n:
        raise ValueError(f"bias must be 1D of length {n}, got {tuple(bias.shape)}")
    if bias.dtype not in _EPILOGUE_DTYPES:
        raise ValueError(f"bias dtype {bias.dtype} is not supported, expected one of "
                         f"{[str(d) for d in _EPILOGUE_DTYPES]}")
    return bias.to(device=device).contiguous()


def _dl(t: torch.Tensor):
    # __dlpack__ refuses a tensor that requires grad, and weights arrive as
    # nn.Parameters. Neither no_grad nor inference_mode clears the flag. Detaching
    # shares storage, so it costs nothing.
    if t.requires_grad:
        t = t.detach()
    # stream=-1 tells PyTorch to skip synchronization (DLPack spec)
    return t.__dlpack__(stream=-1)


# ---------------------------------------------------------------------------
# FP8 elementwise
# ---------------------------------------------------------------------------

def quantize_per_tensor_fp8(
    x: torch.Tensor,
    scale: torch.Tensor,
    output_type: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    x = x.contiguous()
    scale = _scale_operand(scale, x.device)

    out = torch.empty(x.shape, dtype=torch.uint8, device=x.device)
    _C.quantize_per_tensor_fp8(
        _dl(x), _dl(scale), _dl(out),
        DTYPE_TO_CODE[x.dtype], DTYPE_TO_CODE[output_type], x.numel(), _stream(x),
    )
    return out.view(output_type)


def dequantize_per_tensor_fp8(
    x: torch.Tensor,
    scale: torch.Tensor,
    output_type: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    x = x.contiguous()
    scale = _scale_operand(scale, x.device)

    out = torch.empty(x.shape, dtype=output_type, device=x.device)
    _C.dequantize_per_tensor_fp8(
        _dl(x.view(torch.uint8)), _dl(scale), _dl(out),
        DTYPE_TO_CODE[x.dtype], DTYPE_TO_CODE[output_type], x.numel(), _stream(x),
    )
    return out


def stochastic_rounding_fp8(
    x: torch.Tensor,
    rng: torch.Tensor,
    output_type: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    """Quantize x to fp8 with stochastic rounding, consuming rng as the random source.

    The kernel writes the fp8 result into rng's storage, so the caller's rng
    tensor is overwritten and the returned tensor is a view of it.
    """
    if rng.device != x.device:
        raise ValueError("rng must be on the same device as x")
    if rng.shape != x.shape:
        raise ValueError("rng must have the same shape as x")
    # .contiguous() would hand the kernel a copy, leaving the caller's rng
    # untouched and the returned view backed by the wrong storage.
    if not rng.is_contiguous():
        raise ValueError("rng must be contiguous: the kernel writes the result into it")

    x = x.contiguous()
    _C.stochastic_round_fp8(
        _dl(rng), _dl(x), DTYPE_TO_CODE[output_type], x.numel(), _stream(x),
    )
    return rng.view(output_type)


# ---------------------------------------------------------------------------
# FP8 GEMM (v_wmma_f32_16x16x16_fp8_fp8)
# ---------------------------------------------------------------------------

def _weight_as_nk(b: torch.Tensor) -> torch.Tensor:
    """Return the weight as a contiguous (N, K) tensor.

    The B operand of ``a @ b`` arrives with shape (K, N). When it is a transposed
    view of a contiguous (N, K) weight, as it is for linear, the transpose is
    free; otherwise it costs a copy.
    """
    if b.dim() != 2:
        raise ValueError(f"expected a 2D weight, got {tuple(b.shape)}")
    bt = b.t()
    return bt if bt.is_contiguous() else bt.contiguous()


def scaled_mm_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """out = (a @ b) * scale_a * scale_b + bias, with a (M, K) and b (K, N) fp8."""
    # The kernel reads both operands as raw e4m3 bytes; anything else viewed as
    # uint8 would compute nonsense. _C is reachable directly, and this wrapper is a
    # public entry too, so gate here rather than trust the caller.
    if a.dtype is not torch.float8_e4m3fn or b.dtype is not torch.float8_e4m3fn:
        raise ValueError(
            f"scaled_mm_fp8 requires float8_e4m3fn operands, got a={a.dtype}, b={b.dtype}"
        )
    a = _aligned(a.contiguous())
    b_nk = _aligned(_weight_as_nk(b).to(device=a.device))

    m, k = a.shape
    n = b_nk.shape[0]
    if b_nk.shape[1] != k:
        raise ValueError(f"inner dimension mismatch: a K={k}, b K={b_nk.shape[1]}")

    # The WMMA K-step and the small-M GEMV both read a row 16 bytes at a time.
    if k % 16 != 0:
        raise ValueError(f"scaled_mm_fp8 requires K divisible by 16, got {k}")
    # The kernel takes raw pointers on a's stream, so the scales and bias have to
    # live on a's device, and the epilogue indexes bias[col].
    scale_a = _scale_operand(scale_a, a.device)
    scale_b = _scale_operand(scale_b, a.device)
    if bias is not None:
        bias = _bias_operand(bias, n, a.device)

    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    _C.scaled_mm_fp8(
        _dl(a.view(torch.uint8)), _dl(b_nk.view(torch.uint8)), _dl(out),
        _dl(scale_a), _dl(scale_b), None if bias is None else _dl(bias),
        m, n, k, DTYPE_TO_CODE[out_dtype], _stream(a),
    )
    return out


# ---------------------------------------------------------------------------
# NVFP4 (E2M1 + per-16 e4m3 block scales)
# ---------------------------------------------------------------------------

# The blocked scale buffer's shape: one byte per 16 columns, in cuBLAS's "D block
# scaling factors layout" over (RoundUp(rows, 128), RoundUp(cols/16, 4)). Mirrors
# cuda/__init__.py:663-686 and the kernel's nvfp4_scale_offset, which is the exact
# inverse of comfy_kitchen.float_utils.to_blocked.
def _nvfp4_scale_shape(rows: int, cols: int) -> tuple[int, int]:
    return (roundup(rows, 128), roundup(cols // 16, 4))


def _nvfp4_scale_operand(block_scale: torch.Tensor, rows: int, cols: int,
                         device: torch.device) -> torch.Tensor:
    """``block_scale`` as the contiguous raw e4m3 bytes the kernels index.

    The kernels address the scales through the blocked layout rather than as a 2D
    grid, so what they need is the raw byte buffer with the right length. A
    ``float8_e4m3fn`` tensor is already that storage; anything else has to be cast,
    and the cast has to be a real copy because a same-itemcount reinterpret of a
    differently shaped tensor would change which byte a logical cell lands on.
    """
    want = _nvfp4_scale_shape(rows, cols)
    if block_scale.shape != want:
        raise ValueError(
            f"block_scale must have shape {want} for a ({rows}, {cols}) NVFP4 tensor, "
            f"got {tuple(block_scale.shape)}"
        )
    # e4m3 crosses DLPack as uint8 (map_dtype_to_code has no name for codes 10/12).
    return _operand(block_scale.view(torch.uint8), device, "block_scale", shape=want)


def quantize_nvfp4(
    x: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    epsilon: float = 0.0,
    pad_16x: bool = False,
    hi_first: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """E2M1 block quantize: two nibbles per byte, one e4m3 scale per 16 columns.

    ``epsilon`` is accepted and ignored, exactly as it is on the eager and triton
    paths -- no reference implementation uses it either (see triton/quantization.py:395).
    """
    del epsilon  # unused by every implementation of this op; see above
    x = x.contiguous()
    if x.dim() != 2:
        raise ValueError(f"quantize_nvfp4 expects a 2D tensor, got {x.dim()}D")
    orig_rows, orig_cols = x.shape
    # The block scale covers 16 elements and a packed byte covers 2, so the width
    # has to be a whole number of blocks -- of the PADDED width, which is the whole
    # point of pad_16x. An unpadded call therefore still needs orig_cols % 16 == 0.
    if not pad_16x and orig_cols % 16 != 0:
        raise ValueError(
            f"quantize_nvfp4 requires the column count divisible by 16 "
            f"(or pad_16x=True), got {orig_cols}"
        )
    rows = roundup(orig_rows, 16) if pad_16x else orig_rows
    cols = roundup(orig_cols, 16) if pad_16x else orig_cols

    scale = _scale_operand(per_tensor_scale, x.device)
    out = torch.empty((rows, cols // 2), dtype=torch.uint8, device=x.device)
    # Zeroed, not empty: the blocked buffer pads both dimensions up, and the padding
    # is part of the layout rather than a hole. cuda/__init__.py:662 says so too.
    scales = torch.zeros(_nvfp4_scale_shape(rows, cols), dtype=torch.uint8, device=x.device)
    _C.quantize_nvfp4(
        _dl(x), _dl(scale), _dl(out), _dl(scales),
        rows, cols, orig_rows, orig_cols, DTYPE_TO_CODE[x.dtype], hi_first, _stream(x),
    )
    return out, scales.view(torch.float8_e4m3fn)


def dequantize_nvfp4(
    qx: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    block_scales: torch.Tensor,
    output_type: torch.dtype = torch.bfloat16,
    hi_first: bool = True,
) -> torch.Tensor:
    """``e2m1(qx) * block_scale * per_tensor_scale`` back to a float tensor."""
    qx = qx.contiguous()
    if qx.dim() != 2:
        raise ValueError(f"dequantize_nvfp4 expects a 2D tensor, got {qx.dim()}D")
    rows, packed_cols = qx.shape
    cols = packed_cols * 2
    if cols % 16 != 0:
        raise ValueError(f"dequantize_nvfp4 requires the column count divisible by 16, got {cols}")

    scale = _scale_operand(per_tensor_scale, qx.device)
    scales = _nvfp4_scale_operand(block_scales, rows, cols, qx.device)
    out = torch.empty((rows, cols), dtype=output_type, device=qx.device)
    _C.dequantize_nvfp4(
        _dl(qx), _dl(scale), _dl(scales), _dl(out),
        rows, cols, DTYPE_TO_CODE[output_type], hi_first, _stream(qx),
    )
    return out


def scaled_mm_nvfp4(
    a: torch.Tensor,
    b: torch.Tensor,
    tensor_scale_a: torch.Tensor,
    tensor_scale_b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    """out = (a @ b.T) * (tensor_scale_a * tensor_scale_b) + bias, on NVFP4 operands.

    a is (M, K/2) and b is (N, K/2) uint8, two e2m1 values per byte. Both block
    scales are in the blocked layout the kernels index; the per-tensor scales are
    the global half of the decomposition and are applied once in the epilogue.

    ``alpha`` is accepted for signature compatibility with the eager and CUDA paths
    and ignored: it means the same thing ``tensor_scale_a * tensor_scale_b`` already
    says, and cuda/__init__.py:1858-1859 defaults it to exactly that. Passing a
    different alpha would silently disagree with the reference, so the kernels take
    only the two scales.
    """
    del alpha  # equals tensor_scale_a * tensor_scale_b; see docstring
    out_dtype = torch.bfloat16 if out_dtype is None else out_dtype
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("scaled_mm_nvfp4 expects two 2D operands")

    a = _aligned(a.to(torch.uint8).contiguous())
    # The device move is not optional. Every scale and bias below is placed with
    # a.device and the kernel is launched on a's stream, so a `b` left on another
    # device hands the kernel a foreign pointer to dereference there. This is not
    # hypothetical on a part that offloads: the text-encoder runs with
    # offload_device="cpu", so weights do arrive on the host.
    # scaled_mm_fp8 already does this (b.to(device=a.device)).
    b = _aligned(b.to(device=a.device, dtype=torch.uint8).contiguous())
    m, k_packed = a.shape
    n, b_k_packed = b.shape
    if b_k_packed != k_packed:
        raise ValueError(f"inner dimension mismatch: a K={2 * k_packed}, b K={2 * b_k_packed}")
    k = 2 * k_packed
    # Both paths read a packed row a 16-byte chunk at a time, and one scale covers
    # 16 elements, so K must be a whole number of blocks and of chunks.
    if k % 32 != 0:
        raise ValueError(f"scaled_mm_nvfp4 requires K divisible by 32, got {k}")

    sa = _nvfp4_scale_operand(block_scale_a, m, k, a.device)
    sb = _nvfp4_scale_operand(block_scale_b, n, k, a.device)
    tsa = _scale_operand(tensor_scale_a, a.device)
    tsb = _scale_operand(tensor_scale_b, a.device)
    if bias is not None:
        bias = _bias_operand(bias, n, a.device)

    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    _C.scaled_mm_nvfp4(
        _dl(a), _dl(b), _dl(out), _dl(sa), _dl(sb), _dl(tsa), _dl(tsb),
        None if bias is None else _dl(bias),
        m, n, k, n, DTYPE_TO_CODE[out_dtype], _stream(a),
    )
    return out


# ---------------------------------------------------------------------------
# INT8 quantization + GEMM (v_wmma_i32_16x16x16_iu8)
# ---------------------------------------------------------------------------

def quantize_int8_rowwise(
    x: torch.Tensor,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if stochastic_rounding:
        return _eager.quantize_int8_rowwise(x, stochastic_rounding=stochastic_rounding)

    x2d = x.reshape(-1, x.shape[-1]).contiguous()
    m, k = x2d.shape
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    scales = torch.empty((m,), dtype=torch.float32, device=x.device)
    _C.quantize_int8_rowwise(_dl(x2d), _dl(q), _dl(scales), m, k, _stream(x))
    return q.reshape(x.shape), scales.reshape(*x.shape[:-1], 1)


def quantize_int8_tensorwise(
    x: torch.Tensor,
    scale: torch.Tensor | float | str | None = None,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    # A caller-supplied scale reduces to an elementwise quantize; only the
    # absmax-derived scale needs the fused reduction kernel.
    if stochastic_rounding or (scale is not None and not isinstance(scale, str)):
        return _eager.quantize_int8_tensorwise(x, scale=scale, stochastic_rounding=stochastic_rounding)

    xc = x.contiguous()
    q = torch.empty(xc.shape, dtype=torch.int8, device=x.device)
    out_scale = torch.empty((), dtype=torch.float32, device=x.device)
    scratch = torch.zeros((), dtype=torch.int32, device=x.device)
    _C.quantize_int8_tensorwise(
        _dl(xc), _dl(q), _dl(out_scale.reshape(1)), _dl(scratch.reshape(1)), xc.numel(), _stream(x)
    )
    return q, out_scale


def dequantize_int8_simple_dtype(
    q: torch.Tensor,
    scale: torch.Tensor,
    output_dtype_code: int,
) -> torch.Tensor:
    """Dequantize INT8 values into a requested floating dtype."""
    q = q.contiguous()
    output_dtype = DTYPE_CODE_TO_DTYPE[output_dtype_code]
    inner_dim = q.shape[-1] if q.dim() > 0 else 1

    if scale.numel() == 1:
        scale_mode = 0
    elif tuple(scale.shape) == tuple(q.shape):
        scale_mode = 1
    elif (
        q.dim() > 0
        and scale.dim() == q.dim()
        and tuple(scale.shape[:-1]) == tuple(q.shape[:-1])
        and scale.shape[-1] == 1
    ):
        scale_mode = 2
    else:
        return _eager.dequantize_int8_simple_dtype(q, scale, output_dtype_code)

    scale_arg = scale.to(device=q.device, dtype=torch.float32).contiguous()
    output = torch.empty(q.shape, dtype=output_dtype, device=q.device)
    _C.dequantize_int8_simple(
        _dl(q),
        _dl(scale_arg.reshape(-1)),
        _dl(output),
        inner_dim,
        scale_mode,
        _stream(q),
    )
    return output


def dequantize_int8_convrot_weight_dtype(
    q: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    output_dtype_code: int,
) -> torch.Tensor:
    """Dequantize ConvRot INT8 weights and rotate them back to the original basis."""
    if q.dim() != 2:
        raise ValueError("ConvRot INT8 weight dequantization expects a 2D q tensor")

    q_2d = q.contiguous()
    m, k = q_2d.shape
    output_dtype = DTYPE_CODE_TO_DTYPE[output_dtype_code]
    if group_size not in (16, 64, 256) or k % group_size != 0:
        return _eager.dequantize_int8_convrot_weight_dtype(
            q_2d, scale, group_size, output_dtype_code
        )

    scale_arg = scale.to(device=q.device, dtype=torch.float32).reshape(-1).contiguous()
    output = torch.empty(q_2d.shape, dtype=output_dtype, device=q.device)
    _C.dequantize_int8_convrot_weight(
        _dl(q_2d),
        _dl(scale_arg),
        _dl(output),
        m,
        k,
        group_size,
        _stream(q),
    )
    return output


# Keyed by device and dtype: convrot_max_k() reports the LDS budget of whichever
# device is current and FP32 rows consume twice the LDS of FP16/BF16 rows.
_convrot_max_k: dict[tuple[int, torch.dtype], int] = {}
_CONVROT_SPILL_WORKSPACE_BYTES = 32 << 20


def _convrot_int8_needs_spill(
    m: int, k: int, input_dtype_code: int, device: torch.device
) -> bool:
    """Query the ConvRot capability for the operand's device."""
    with torch.cuda.device(device):
        return bool(_C.convrot_int8_needs_spill(m, k, input_dtype_code))


def _convrot_row_max_k(device: torch.device, dtype: torch.dtype) -> int:
    """The widest row ``device``'s LDS can stage in ``dtype``, cached per device."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, dtype)
    max_k = _convrot_max_k.get(key)
    if max_k is None:
        with torch.cuda.device(index):
            max_k = _C.convrot_max_k(DTYPE_TO_CODE[dtype])
        _convrot_max_k[key] = max_k
    return max_k


def _convrot_supported(
    k: int, group_size: int, device: torch.device, dtype: torch.dtype,
    *, int8_global_spill: bool = False,
) -> bool:
    """Whether the HIP ConvRot quantizer can handle a row of this width on ``device``.

    INT8 G=256 with ``int8_global_spill`` uses fused LDS or global spill; K need only
    divide the group size. Other paths stage the whole row in LDS (K bounded per device).
    The kernel rotates K/G whole groups; a partial trailing group would quantize
    uninitialized LDS. The LDS budget is read for the operand's device, not the
    process-current one.
    """
    if group_size not in (16, 64, 256) or k % group_size != 0:
        return False
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    if group_size == 256 and int8_global_spill:
        return True
    max_k = _convrot_row_max_k(device, dtype)
    return max_k > 0 and k <= max_k


def _rotate_quant_int8(
    x2d: torch.Tensor, group_size: int, input_act: str | None = None,
    act_weight: torch.Tensor | None = None, act_eps: float = 0.0,
    *, tiled_output: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k_in = x2d.shape
    # swiglu and situ halve the row: the input is twice the quantized width.
    k = k_in // _input_act_width(input_act)
    q_rows = (m + 127) // 128 * 128 if tiled_output else m
    q = torch.empty((q_rows, k), dtype=torch.int8, device=x2d.device)
    scales = torch.empty((m,), dtype=torch.float32, device=x2d.device)
    x_arg = _operand(x2d, x2d.device, "x2d")
    # check_convrot_k queries the current device's LDS budget, so pin it to the
    # operand's device rather than trusting the caller thread's current device.
    with torch.cuda.device(x2d.device):
        if group_size != 256 or not _convrot_int8_needs_spill(
            m, k, DTYPE_TO_CODE[x2d.dtype], x2d.device
        ):
            _C.quantize_int8_convrot(
                _dl(x_arg), _dl(q), _dl(scales),
                None, None, m, k,
                group_size, _input_act_code(input_act), _stream(x2d),
                None if act_weight is None else _dl(act_weight),
                float(act_eps), tiled_output,
            )
            return q, scales

        workspace_per_row = x_arg.element_size() * k + 4 * (k // 256)
        # Tiled chunks start on an M-tile boundary. Reserve at least one tile
        # of workspace, just as the row-major path reserves at least one row.
        row_unit = 128 if tiled_output else 1
        units = (m + row_unit - 1) // row_unit
        unit_cap = max(1, _CONVROT_SPILL_WORKSPACE_BYTES // (workspace_per_row * row_unit))
        capacity_chunk_count = (units + unit_cap - 1) // unit_cap
        # For wide rows the native policy picks the global implementation at
        # 96 rows or more; a shorter chunk would switch to the one-wave LDS
        # schedule. Balance row-major chunks to avoid that switch when the
        # workspace cap allows it. Tiled chunks must keep 128-row boundaries,
        # so their final tail may still use the LDS schedule.
        balanced_chunk_count = min(capacity_chunk_count, max(1, m // 96))
        balanced_chunk_rows = (
            units + balanced_chunk_count - 1
        ) // balanced_chunk_count * row_unit
        chunk_count = (
            balanced_chunk_count
            if balanced_chunk_rows <= unit_cap * row_unit
            else capacity_chunk_count
        )
        chunk_rows = min(m, (units + chunk_count - 1) // chunk_count * row_unit)
        spill_rotated = torch.empty(
            (chunk_rows, k), dtype=x_arg.dtype, device=x2d.device
        )
        spill_partials = torch.empty(
            (chunk_rows, k // 256), dtype=torch.float32, device=x2d.device
        )
        start = 0
        for chunk in range(chunk_count):
            rows = min(m - start, (units // chunk_count + (chunk < units % chunk_count)) * row_unit)
            end = start + rows
            q_end = start + (rows + row_unit - 1) // row_unit * row_unit
            _C.quantize_int8_convrot(
                # Row slices of x and q start on a 256-byte multiple (K % 256 == 0);
                # the kernel writes scales[row] as scalars, so the scales slice needs
                # no alignment. Output slices must be passed as views, never copied.
                _dl(x_arg[start:end]),
                _dl(q[start:q_end]),
                _dl(scales[start:end]),
                _dl(spill_rotated),
                _dl(spill_partials),
                rows,
                k,
                group_size,
                _input_act_code(input_act),
                _stream(x2d),
                None if act_weight is None else _dl(act_weight),
                float(act_eps),
                tiled_output,
            )
            start = end
    return q, scales


def _rotate_quant_int8_modulated(
    x2d: torch.Tensor, modulation_scale: torch.Tensor, group_size: int,
    *, tiled_output: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold exact twice-rounded BF16 batch-one modulation into ConvRot."""
    m, k = x2d.shape
    q_rows = (m + 127) // 128 * 128 if tiled_output else m
    q = torch.empty((q_rows, k), dtype=torch.int8, device=x2d.device)
    scales = torch.empty((m,), dtype=torch.float32, device=x2d.device)
    with torch.cuda.device(x2d.device):
        _C.quantize_int8_convrot_modulated(
            _dl(_aligned(x2d)), _dl(_aligned(modulation_scale)),
            _dl(q), _dl(scales),
            m, k, group_size, _stream(x2d), tiled_output,
        )
    return q, scales


def _rotate_quant_int8_rms_modulated(
    x2d: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    modulation_scale: torch.Tensor,
    group_size: int,
    *, tiled_output: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold exact BF16 RMSNorm and modulation into the ConvRot producer."""
    m, k = x2d.shape
    q_rows = (m + 127) // 128 * 128 if tiled_output else m
    q = torch.empty((q_rows, k), dtype=torch.int8, device=x2d.device)
    scales = torch.empty((m,), dtype=torch.float32, device=x2d.device)
    with torch.cuda.device(x2d.device):
        _C.quantize_int8_convrot_rms_modulated_fused_stats(
            _dl(_aligned(x2d)), _dl(_aligned(norm_weight)),
            _dl(_aligned(modulation_scale)), _dl(q),
            _dl(scales), m, k, group_size, norm_eps, _stream(x2d),
            tiled_output,
        )
    return q, scales


def _fused_rms_norm_ok(x: torch.Tensor, convrot: bool, group_size: int) -> bool:
    """Whether int8_linear's quantizer will run the fused G=256 kernel, the only
    one that implements rms_norm; the global spill and G=16/64 paths do not."""
    k = x.shape[-1]
    m = x.numel() // k if k else 0
    if not (convrot and group_size == 256 and m > 0 and k % 256 == 0
            and _convrot_supported(k, group_size, x.device, x.dtype, int8_global_spill=True)):
        return False
    # the spill answer reads the current device's LDS budget, as in _rotate_quant_int8
    with torch.cuda.device(x.device):
        return not _convrot_int8_needs_spill(m, k, DTYPE_TO_CODE[x.dtype], x.device)


def quantize_and_rotate_rowwise(
    x: torch.Tensor,
    h: torch.Tensor,
    group_size: int,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused ConvRot rotation + rowwise int8 quantize.

    ``h`` is the pre-built Hadamard matrix the eager path multiplies by. The
    fused kernel synthesizes the same transform from radix-4 butterflies, so the
    argument is accepted and unused.
    """
    if stochastic_rounding or not _convrot_supported(
        x.shape[-1], group_size, x.device, x.dtype, int8_global_spill=True
    ):
        return _eager.quantize_and_rotate_rowwise(
            x, h, group_size, stochastic_rounding=stochastic_rounding
        )

    x2d = x.reshape(-1, x.shape[-1]).contiguous()
    q, scales = _rotate_quant_int8(x2d, group_size)
    return q.reshape(x.shape), scales.reshape(*x.shape[:-1], 1)


def quantize_int8_convrot_weight(
    weight: torch.Tensor,
    group_size: int,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Offline ConvRot weight rotation + rowwise int8 quantize.

    Uses the same fused kernel as the activation path.
    """
    if stochastic_rounding or not _convrot_supported(
        weight.shape[-1], group_size, weight.device, weight.dtype, int8_global_spill=True
    ):
        return _eager.quantize_int8_convrot_weight(
            weight, group_size, stochastic_rounding=stochastic_rounding
        )

    w2d = weight.reshape(-1, weight.shape[-1]).contiguous()
    q, scales = _rotate_quant_int8(w2d, group_size)
    return q.reshape(weight.shape), scales.reshape(*weight.shape[:-1], 1)


def _vector_operand(v: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """A per-channel epilogue vector on ``device`` in ``dtype``, contiguous."""
    return v.to(device=device, dtype=dtype).reshape(-1).contiguous()


def fp16_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    residual_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """FP16/BF16 WMMA GEMM with bias and an optional ``residual + residual_scale * out``
    fused into the epilogue; torch's linear where the kernel declines the shape."""
    if residual is not None and residual_scale is None:
        raise ValueError("fp16_linear: residual requires residual_scale")

    orig_shape = x.shape
    x_2d = x if x.dim() == 2 and x.is_contiguous() else x.reshape(-1, x.shape[-1]).contiguous()
    m = x_2d.shape[0]
    n, k = weight.shape

    supported = (
        x.dtype in (torch.float16, torch.bfloat16 if _is_small_igpu(x.device) else torch.float16)
        and weight.dtype == x.dtype
        and weight.device == x.device
        and x_2d.shape[1] == k
        # the tile stager issues 16-byte row loads; a misaligned view falls back, as on CUDA
        and x_2d.data_ptr() % 16 == 0
        and (bias is None or bias.dtype == torch.float16)
        and (residual is None or (residual.dtype == torch.float16
                                  and residual_scale.dtype == torch.float16))
    )
    if supported:
        weight = weight if weight.is_contiguous() else weight.contiguous()
        supported = weight.data_ptr() % 16 == 0
    if not supported:
        # the kernel path takes bias and residual from any device, so the fallback must too
        # Upstream leaves the bias dtype alone here and lets torch's linear reject a
        # mismatched pair; only the iGPU's bf16 envelope needs it cast, so the cast is
        # gated like the dtype check above and every other device keeps upstream's
        # spelling (.to(dtype=<its own dtype>) is the identity).
        bias = None if bias is None else bias.to(
            device=x.device, dtype=x.dtype if _is_small_igpu(x.device) else bias.dtype)
        if residual is not None:
            residual = residual.to(device=x.device)
            residual_scale = residual_scale.to(device=x.device)
        out = torch.nn.functional.linear(x, weight, bias)
        return _apply_residual(out, residual, residual_scale)

    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    bias_arg = None if bias is None else _vector_operand(bias, x.device, torch.float16)
    resid_arg = rscale_arg = None
    if residual is not None:
        resid_arg = residual.to(device=x.device).reshape(m, n).contiguous()
        rscale_arg = _vector_operand(residual_scale, x.device, torch.float16)
    # The stream comes from x, but both launchers pick their architecture from the
    # CURRENT device -- launch_fp16_gemm_kernel branches on comfy_is_gfx10() to reach
    # the VALU v_dot2_f32_f16 kernel and falls through to launch_gemm_wmma<MmaF16>
    # otherwise, and fp16_shape_served reads the same device. On a box where those
    # two devices differ, a gfx11 current device sends a gfx103x stream to a WMMA
    # launch, and MmaF16 does not exist there. The reverse direction is milder and
    # asymmetric by dtype: the bf16 launcher declines gfx10 outright (there is no
    # v_dot2_f32_bf16 below gfx11) while fp16 has a VALU kernel, so a gfx10 current
    # device with a gfx11 stream declines one and mis-serves the other.
    #
    # This is the same pin int8_linear, _w4a8_int8_linear_chunked and
    # convrot_w4a4_linear already apply for the same reason, and it is free where it
    # cannot matter: _device_pin is a nullcontext when only one device is visible.
    with _device_pin(x.device):
        if x.dtype == torch.float16:
            served = _C.fp16_gemm(
                _dl(x_2d), _dl(weight), _dl(out),
                None if bias_arg is None else _dl(bias_arg),
                None if rscale_arg is None else _dl(rscale_arg),
                None if resid_arg is None else _dl(resid_arg),
                m, n, k, _stream(x),
            )
        else:
            served = _C.bf16_gemm(
                _dl(x_2d), _dl(weight), _dl(out),
                None if bias_arg is None else _dl(bias_arg),
                None if rscale_arg is None else _dl(rscale_arg),
                None if resid_arg is None else _dl(resid_arg),
                m, n, k, _stream(x),
            )
    if not served:
        # The GEMM declined the shape; torch's linear needs the bias in the input
        # dtype (addmm requires compatible operand dtypes), so cast the FP16 bias
        # here while keeping it FP16 on the kernel path.
        bias_fb = None if bias_arg is None else bias_arg.to(dtype=x.dtype)
        out = _apply_residual(torch.nn.functional.linear(x_2d, weight, bias_fb), resid_arg,
                              rscale_arg)
    return out if len(orig_shape) == 2 else out.reshape(*orig_shape[:-1], n)


# Acts the HIP fused quantizer implements; anything else is applied eagerly so
# new act codes degrade gracefully instead of erroring in the kernel. rms_norm is
# folded as well, but only where _fused_rms_norm_ok holds.
_HIP_FUSED_ACTS = (None, "none", "gelu_tanh", "swiglu", "situ")


def _use_tiled_int8(m: int, n: int, k: int, device: torch.device) -> bool:
    # Keep the small-M GEMV and the narrow/partial-K row-major tiles.
    return m > 8 and n > 64 and k > 0 and k % 128 == 0 and _has_nonduplicated_wmma(device)


def _split_input_act(
    x: torch.Tensor,
    input_act: str | None,
    input_act_weight: torch.Tensor | None,
    input_act_eps: float,
    convrot: bool,
    convrot_groupsize: int,
) -> tuple[torch.Tensor, str | None, torch.Tensor | None]:
    """Apply what the ConvRot quantizer cannot fuse; return the input, the act left
    for the quantizer and the rms_norm weight operand it needs."""
    # Rejected here so every route fails the same way, not just the fused one.
    _input_act_code(input_act)
    act_weight = None
    if input_act == "rms_norm":
        if input_act_weight is None:
            raise ValueError("input_act 'rms_norm' requires act_weight")
        if (
            input_act_weight.numel() == x.shape[-1]
            and _fused_rms_norm_ok(x, convrot, convrot_groupsize)
        ):
            act_weight = _operand(
                input_act_weight.reshape(-1).to(dtype=x.dtype), x.device, "input_act_weight")
    if input_act not in _HIP_FUSED_ACTS and act_weight is None:
        x = _apply_input_act(x, input_act, input_act_weight, input_act_eps)
        input_act = None
    return x, input_act, act_weight


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
    """INT8 linear with dynamic row-wise activation quantization, on WMMA."""
    x, input_act, act_weight = _split_input_act(
        x, input_act, input_act_weight, input_act_eps, convrot, convrot_groupsize)
    # k_act is the activated (quantized) row width: swiglu and situ halve the raw row.
    in_width = _input_act_width(input_act)
    if x.shape[-1] % in_width != 0:
        raise ValueError(f"input_act {input_act!r} needs an even input width, got {x.shape[-1]}")
    k_act = x.shape[-1] // in_width
    if k_act != weight.shape[-1]:
        raise ValueError(
            f"Input and weight inner dimensions must match, got {k_act} and {weight.shape[-1]}"
        )

    weight = _aligned(weight.to(device=x.device).contiguous())
    weight_scale = weight_scale.to(device=x.device, dtype=torch.float32).reshape(-1)
    if weight_scale.numel() not in (1, weight.shape[0]):
        raise ValueError(
            f"INT8 weight scale must be scalar or per-output-channel, got {tuple(weight_scale.shape)}"
        )

    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1]).contiguous()
    m = x2d.shape[0]
    k = k_act
    n = weight.shape[0]
    tiled = _use_tiled_int8(m, n, k, x.device)

    # The WMMA K-step and the small-M GEMV both read a row 16 bytes at a time.
    if k % 16 != 0:
        raise ValueError(f"int8_linear requires K divisible by 16, got {k}")

    if convrot:
        if convrot_groupsize not in (16, 64, 256):
            raise ValueError(f"ConvRot group size must be 16, 64 or 256, got {convrot_groupsize}")
        if k % convrot_groupsize != 0:
            raise ValueError(
                f"ConvRot group size {convrot_groupsize} does not divide input features {k}"
            )
        if not _convrot_supported(
            k, convrot_groupsize, x.device, x.dtype, int8_global_spill=True
        ):
            return _eager.int8_linear(
                x2d.reshape(*orig_shape[:-1], x2d.shape[-1]),
                weight, weight_scale, bias, out_dtype, convrot, convrot_groupsize,
                input_act=input_act, residual=residual, residual_scale=residual_scale,
            )
        q, x_scale = _rotate_quant_int8(
            x2d, convrot_groupsize, input_act, act_weight, input_act_eps,
            tiled_output=tiled)
    else:
        x2d = _apply_input_act(x2d, input_act)
        q_rows = (m + 127) // 128 * 128 if tiled else m
        q = torch.empty((q_rows, k), dtype=torch.int8, device=x.device)
        x_scale = torch.empty((m,), dtype=torch.float32, device=x.device)
        _C.quantize_int8_rowwise(_dl(x2d), _dl(q), _dl(x_scale), m, k, _stream(x), tiled)

    x_scale = x_scale.reshape(-1).contiguous()
    if bias is not None:
        bias = _bias_operand(bias, n, x.device)

    out = torch.empty((m, n), dtype=out_dtype, device=x.device)
    # launch_int8_gemm_kernel picks its tile family from comfy_is_gfx10(), which
    # reads the *current* device, while _stream(x) names a stream on x.device. See
    # _device_pin for why the pin is skipped when it cannot matter.
    gemm = _C.int8_gemm_a_tiled128 if tiled else _C.int8_gemm
    with _device_pin(x.device):
        gemm(
            _dl(q), _dl(weight), _dl(out),
            _dl(x_scale), _dl(weight_scale), 0 if weight_scale.numel() == 1 else 1,
            None if bias is None else _dl(bias),
            m, n, k, DTYPE_TO_CODE[out_dtype], _stream(x),
        )
    # Unlike CUDA, the residual is not folded into the epilogue: the per-element
    # residual reads there cost more than a separate addcmul at the output widths
    # pre-norm blocks apply it to.
    return _apply_residual(out.reshape(*orig_shape[:-1], n), residual, residual_scale)


def rms_gated_residual(
    activation: torch.Tensor,
    norm_weight: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    """Exact BF16 RMSNorm, gate multiplication, and residual add."""
    width = activation.shape[-1] if activation.ndim else 0
    max_k = 0
    if activation.dtype == torch.bfloat16 and activation.device.type == "cuda":
        max_k = _convrot_row_max_k(activation.device, activation.dtype)
    use_fused = (
        activation.dtype == torch.bfloat16
        and norm_weight.dtype == torch.bfloat16
        and residual.dtype == torch.bfloat16
        and gate.dtype == torch.bfloat16
        and activation.device == norm_weight.device == residual.device == gate.device
        and activation.ndim == 3
        and activation.shape[0] == 1
        and activation.shape == residual.shape
        and width > 0
        and width % 4 == 0
        and width <= max_k
        and norm_weight.numel() == width
        and norm_weight.shape[-1] == width
        and gate.numel() == width
        and gate.shape[-1] == width
    )
    if not use_fused:
        return _eager.rms_gated_residual(
            activation, norm_weight, residual, gate, eps
        )

    shape = activation.shape
    activation_2d = activation.reshape(-1, width).contiguous()
    norm_weight_1d = norm_weight.reshape(-1).contiguous()
    residual_2d = residual.reshape(-1, width).contiguous()
    gate_1d = gate.reshape(-1).contiguous()
    output = torch.empty_like(activation_2d)
    _C.rms_gated_residual_bf16(
        _dl(_aligned(activation_2d)), _dl(_aligned(norm_weight_1d)),
        _dl(_aligned(residual_2d)), _dl(_aligned(gate_1d)), _dl(output),
        activation_2d.shape[0], width,
        float(eps), _stream(activation),
    )
    return output.reshape(shape)


def int8_linear_modulated(
    x: torch.Tensor,
    modulation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    convrot: bool = False,
    convrot_groupsize: int = 256,
) -> torch.Tensor:
    """INT8 projection with batch-one BF16 modulation in its ConvRot load."""
    k = x.shape[-1]
    if k != weight.shape[-1]:
        raise ValueError(
            f"Input and weight inner dimensions must match, got {k} and {weight.shape[-1]}"
        )
    if modulation_scale.numel() != k:
        raise ValueError(
            f"modulation_scale must contain one batch-one row of {k} values, "
            f"got {modulation_scale.numel()}"
        )

    modulation_scale = modulation_scale.to(device=x.device).reshape(-1).contiguous()
    orig_shape = x.shape
    x2d = x.reshape(-1, k).contiguous()
    m = x2d.shape[0]
    n = weight.shape[0]
    use_fused = (
        _has_nonduplicated_wmma(x.device)
        and x2d.dtype == torch.bfloat16
        and modulation_scale.dtype == torch.bfloat16
        and out_dtype == torch.bfloat16
        and convrot
        and convrot_groupsize == 256
        and k == 3840
        and n == 11520
    )
    if not use_fused:
        scale_shape = (1,) * (x.ndim - 1) + (k,)
        modulated = x * (1.0 + modulation_scale.reshape(scale_shape))
        return int8_linear(
            modulated, weight, weight_scale, bias, out_dtype,
            convrot, convrot_groupsize,
        )

    weight = _aligned(weight.to(device=x.device).contiguous())
    weight_scale = weight_scale.to(device=x.device, dtype=torch.float32).reshape(-1)
    if weight_scale.numel() not in (1, n):
        raise ValueError(
            f"INT8 weight scale must be scalar or per-output-channel, got "
            f"{tuple(weight_scale.shape)}"
        )
    if bias is not None:
        bias = _bias_operand(bias, n, x.device)

    tiled = _use_tiled_int8(m, n, k, x.device)
    q, x_scale = _rotate_quant_int8_modulated(
        x2d, modulation_scale, convrot_groupsize, tiled_output=tiled,
    )
    out = torch.empty((m, n), dtype=out_dtype, device=x.device)
    gemm = _C.int8_gemm_a_tiled128 if tiled else _C.int8_gemm
    gemm(
        _dl(q), _dl(weight), _dl(out), _dl(x_scale), _dl(weight_scale),
        0 if weight_scale.numel() == 1 else 1,
        None if bias is None else _dl(bias),
        m, n, k, DTYPE_TO_CODE[out_dtype], _stream(x),
    )
    return out.reshape(*orig_shape[:-1], n)


def int8_linear_rms_modulated(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    modulation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    convrot: bool = False,
    convrot_groupsize: int = 256,
) -> torch.Tensor:
    """INT8 QKV projection with exact RMSNorm and modulation in ConvRot."""
    k = x.shape[-1]
    if k != weight.shape[-1]:
        raise ValueError(
            f"Input and weight inner dimensions must match, got {k} and {weight.shape[-1]}"
        )
    if norm_weight.numel() != k or modulation_scale.numel() != k:
        raise ValueError(
            f"norm_weight and modulation_scale must each contain {k} values"
        )

    norm_weight = norm_weight.to(device=x.device).reshape(-1).contiguous()
    modulation_scale = modulation_scale.to(device=x.device).reshape(-1).contiguous()
    orig_shape = x.shape
    x2d = x.reshape(-1, k).contiguous()
    m = x2d.shape[0]
    n = weight.shape[0]
    use_fused = (
        _has_nonduplicated_wmma(x.device)
        and x2d.dtype == torch.bfloat16
        and norm_weight.dtype == torch.bfloat16
        and modulation_scale.dtype == torch.bfloat16
        and out_dtype == torch.bfloat16
        and convrot
        and convrot_groupsize == 256
        and k == 3840
        and n == 11520
    )
    if not use_fused:
        normalized = torch.nn.functional.rms_norm(
            x, (k,), norm_weight, norm_eps
        )
        return int8_linear_modulated(
            normalized, modulation_scale, weight, weight_scale, bias,
            out_dtype, convrot, convrot_groupsize,
        )

    weight = _aligned(weight.to(device=x.device).contiguous())
    weight_scale = weight_scale.to(
        device=x.device, dtype=torch.float32
    ).reshape(-1)
    if weight_scale.numel() not in (1, n):
        raise ValueError(
            f"INT8 weight scale must be scalar or per-output-channel, got "
            f"{tuple(weight_scale.shape)}"
        )
    if bias is not None:
        bias = _bias_operand(bias, n, x.device)

    tiled = _use_tiled_int8(m, n, k, x.device)
    q, x_scale = _rotate_quant_int8_rms_modulated(
        x2d, norm_weight, norm_eps, modulation_scale, convrot_groupsize,
        tiled_output=tiled,
    )
    out = torch.empty((m, n), dtype=out_dtype, device=x.device)
    gemm = _C.int8_gemm_a_tiled128 if tiled else _C.int8_gemm
    gemm(
        _dl(q), _dl(weight), _dl(out), _dl(x_scale), _dl(weight_scale),
        0 if weight_scale.numel() == 1 else 1,
        None if bias is None else _dl(bias),
        m, n, k, DTYPE_TO_CODE[out_dtype], _stream(x),
    )
    return out.reshape(*orig_shape[:-1], n)


# ---------------------------------------------------------------------------
# Grouped W4A8 over the INT8 GEMM
# ---------------------------------------------------------------------------

def _dequant_int4_grouped_to_int8(
    qdata: torch.Tensor,
    s_rel: torch.Tensor,
    codebook: torch.Tensor | None,
    group_size: int,
) -> torch.Tensor:
    """Decode packed INT4/INT6 weights to the grouped INT8 grid the GEMM consumes."""
    n, k, _bits = _w4a8_geometry(qdata, s_rel, group_size)
    device = qdata.device
    qdata_arg = _operand(qdata, device, "qdata")
    scale_code = DTYPE_TO_CODE[s_rel.dtype]
    # fp8 crosses the binding as raw bytes, as it does everywhere else here.
    s_rel_arg = _operand(s_rel, device, "s_rel")
    if s_rel.dtype == torch.float8_e4m3fn:
        s_rel_arg = s_rel_arg.view(torch.uint8)
    codebook_arg = (
        None
        if codebook is None
        else codebook.to(device=device, dtype=torch.float32).reshape(-1).contiguous()
    )

    out = torch.empty((n, k), dtype=torch.int8, device=device)
    _C.dequant_int4_grouped_to_int8(
        _dl(qdata_arg),
        _dl(s_rel_arg),
        scale_code,
        None if codebook_arg is None else _dl(codebook_arg),
        _dl(out),
        n,
        k,
        group_size,
        _stream(qdata),
    )
    return out


def _tuning_env_int(name: str, default: int) -> int:
    """A non-negative tuning knob from the environment, or ``default``.

    These are read at import, and the backend is imported unguarded, so letting a
    typo raise would take out ``import comfy_kitchen`` itself rather than just the
    knob. A tuning value is never worth that.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("ignoring non-integer %s=%r, using %d", name, raw, default)
        return default


# A chunk is worth decoding separately only while it is still cached when the GEMM
# reads it back, so the target is the device's own L2 rather than a tuned constant.
# The cap matches the CUDA backend's chunk width, keeping a shallow K from making a
# chunk pointlessly wide; the floor keeps a deep K from starving the GEMM's N grid,
# which costs far more than the decode saves.
_W4A8_MAX_CHUNK_COLS = _tuning_env_int("COMFY_KITCHEN_W4A8_CHUNK_COLS", 4096)
_W4A8_MIN_CHUNK_COLS = 1024
# Chunking pays while the decode, not the GEMM, sets the cost. Where that crossover
# sits depends on the part, so the limit is the conservative end of what measured as
# a win on every shape tried: above it the weight is decoded in one pass, which is
# what this path did before chunking, so a part that crosses over later loses only
# the extra win rather than regressing.
_W4A8_CHUNK_MAX_ROWS = _tuning_env_int("COMFY_KITCHEN_W4A8_CHUNK_MAX_ROWS", 64)
# Matches kW4a8GemvMaxM in ops/w4a8_gemv.hip: past it the weight pass is amortized
# over enough rows that the WMMA GEMM wins, and the kernel holds one accumulator
# per row in registers.
_W4A8_GEMV_MAX_ROWS = 8
_W4A8_FALLBACK_L2_BYTES = 4 << 20
_w4a8_l2_bytes: dict[int, int] = {}


def _w4a8_chunk_cols(m: int, n: int, k: int, device: torch.device) -> int:
    """How many weight columns to decode before running the GEMM on them.

    ``n`` decodes the weight in one pass, which is what the chunked launcher does
    with a single chunk.
    """
    if m > _W4A8_CHUNK_MAX_ROWS or not _W4A8_MAX_CHUNK_COLS:
        return n

    index = device.index if device.index is not None else torch.cuda.current_device()
    budget = _w4a8_l2_bytes.get(index)
    if budget is None:
        props = torch.cuda.get_device_properties(index)
        budget = getattr(props, "L2_cache_size", 0) or _W4A8_FALLBACK_L2_BYTES
        _w4a8_l2_bytes[index] = budget

    # Rounded to the widest N tile, so a chunk boundary never splits one.
    cols = min(_W4A8_MAX_CHUNK_COLS, max(_W4A8_MIN_CHUNK_COLS, budget // max(k, 1) // 128 * 128))
    return min(n, cols)


def _w4a8_int8_linear_chunked(
    x: torch.Tensor,
    qdata: torch.Tensor,
    s_rel: torch.Tensor,
    s_channel: torch.Tensor,
    codebook: torch.Tensor | None,
    bias: torch.Tensor | None,
    group_size: int,
    convrot_groupsize: int,
    out_dtype: torch.dtype,
    input_act: str | None = None,
    act_weight: torch.Tensor | None = None,
    act_eps: float = 0.0,
) -> torch.Tensor:
    n, k, _bits = _w4a8_geometry(qdata, s_rel, group_size)
    device = x.device
    orig_shape = x.shape
    x2d = x.reshape(-1, x.shape[-1]).contiguous()
    m = x2d.shape[0]

    # Rotated and quantized once: the chunk loop only walks the weight.
    xq, xs = _rotate_quant_int8(x2d, convrot_groupsize, input_act, act_weight, act_eps)

    scale_code = DTYPE_TO_CODE[s_rel.dtype]
    s_rel_arg = _operand(s_rel, device, "s_rel")
    if s_rel.dtype == torch.float8_e4m3fn:
        s_rel_arg = s_rel_arg.view(torch.uint8)
    codebook_arg = (
        None
        if codebook is None
        else codebook.to(device=device, dtype=torch.float32).reshape(-1).contiguous()
    )
    s_channel_arg = s_channel.to(device=device, dtype=torch.float32).reshape(-1).contiguous()
    bias_arg = None if bias is None else _bias_operand(bias, n, device)

    qdata_arg = _operand(qdata, device, "qdata")
    xs_arg = xs.reshape(-1).contiguous()
    out = torch.empty((m, n), dtype=out_dtype, device=device)

    # Decode fast path: the GEMV dequantizes in registers, so the INT8 weight never
    # reaches memory. Bit-exact with the chunked path below, and it declines any
    # shape it cannot take rather than the caller having to predict the envelope.
    if m <= _W4A8_GEMV_MAX_ROWS and s_rel.dtype == torch.float8_e4m3fn:
        used = _C.w4a8_codebook_gemv(
            _dl(xq),
            _dl(qdata_arg),
            _dl(s_rel_arg),
            None if codebook_arg is None else _dl(codebook_arg),
            _dl(s_channel_arg),
            _dl(xs_arg),
            None if bias_arg is None else _dl(bias_arg),
            _dl(out),
            m,
            n,
            k,
            group_size,
            DTYPE_TO_CODE[out_dtype],
            _stream(x),
        )
        if used:
            return out.reshape(*orig_shape[:-1], n)

    # An empty weight has no chunk to size; the loop then has nothing to walk.
    chunk_cols = max(1, _w4a8_chunk_cols(m, n, k, device))
    workspace = torch.empty((chunk_cols, k), dtype=torch.int8, device=device)
    # Same reason as int8_linear's int8_gemm: the chunked path reaches
    # launch_int8_gemm_kernel, whose gfx10 test reads the current device while the
    # stream is the one for x.device. See _device_pin.
    with _device_pin(x.device):
        _C.w4a8_int8_gemm_chunked(
            _dl(xq),
            _dl(qdata_arg),
            _dl(s_rel_arg),
            scale_code,
            None if codebook_arg is None else _dl(codebook_arg),
            _dl(s_channel_arg),
            _dl(xs_arg),
            None if bias is None else _dl(bias_arg),
            _dl(workspace),
            _dl(out),
            m,
            n,
            k,
            group_size,
            chunk_cols,
            DTYPE_TO_CODE[out_dtype],
            _stream(x),
        )
    return out.reshape(*orig_shape[:-1], n)


# Keyed by device and group size: the fused requantize holds the rotated row and one
# float per group in LDS, so the widest K it can take is a property of whichever card
# the weight lives on.
_wxa8_requant_max_k: dict[tuple[int, int], int] = {}


def _requant_supported(k: int, group_size: int, device: torch.device, dtype: torch.dtype) -> bool:
    """Whether the fused requantize can take a row of this width on ``device``.

    The budget is read for the weight's own device rather than the process-current
    one, so a multi-GPU process asks the card the kernel will actually launch on.
    fp32 weights take the staged path, as on CUDA.
    """
    if not _EXT_AVAILABLE or k % 256 != 0 or k % group_size != 0:
        return False
    if dtype not in (torch.float16, torch.bfloat16):
        return False

    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, group_size)
    max_k = _wxa8_requant_max_k.get(key)
    if max_k is None:
        with torch.cuda.device(index):
            max_k = _C.wxa8_requant_max_k(group_size)
        _wxa8_requant_max_k[key] = max_k
    return max_k > 0 and k <= max_k


def _fused_quantize_wxa8(
    weight: torch.Tensor,
    bits: int,
    group_size: int,
    codebook: torch.Tensor | None,
    stochastic_rounding: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, torch.Tensor | None]:
    """ConvRot and requantize the raw weight in one launch, with no rotated copy."""
    n, k = weight.shape
    dev = weight.device
    if bits == 4:
        cb = codebook.to(device=dev, dtype=torch.float32).contiguous()
        packed = torch.empty(n, k // 2, dtype=torch.int8, device=dev)
    else:
        cb = torch.empty(0, dtype=torch.float32, device=dev)
        packed = torch.empty(n, k * 3 // 4, dtype=torch.int8, device=dev)
    s_rel = torch.empty(n, k // group_size, dtype=torch.float8_e4m3fn, device=dev)
    s_channel = torch.empty(n, dtype=torch.float32, device=dev)
    with torch.cuda.device(dev):
        _C.quantize_wxa8_convrot_fused(
            _dl(_aligned(weight.contiguous())),
            _dl(cb),
            _dl(packed),
            _dl(s_rel.view(torch.uint8)),
            _dl(s_channel),
            n,
            k,
            bits,
            group_size,
            stochastic_rounding > 0,
            int(stochastic_rounding) if stochastic_rounding > 0 else 0,
            _stream(weight),
        )
    return packed, s_rel, s_channel, None, (cb if bits == 4 else None)


_W4A8_STAGED_QUANT = _EXT_AVAILABLE
# The staged kernel keeps only K/16 fp32 group scales in LDS. Same limit as CUDA's,
# which reserves room for its static arrays within the 48 KiB default block limit.
_W4A8_STAGED_MAX_K = 16 * (47 * 1024 // 4)


def _quantize_w4a8_staged(
    weight: torch.Tensor,
    codebook: torch.Tensor,
    convrot_groupsize: int,
    stochastic_rounding: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, torch.Tensor]:
    """Rotate, then requantize in a second launch, a row block at a time so no full
    rotated copy is held. Every output is per row, so each block writes straight into
    its slice."""
    n, k = weight.shape
    dev = weight.device
    cb = codebook.to(device=dev, dtype=torch.float32).contiguous()
    packed = torch.empty(n, k // 2, dtype=torch.int8, device=dev)
    s_rel = torch.empty(n, k // 16, dtype=torch.float8_e4m3fn, device=dev)
    s_channel = torch.empty(n, dtype=torch.float32, device=dev)
    block = max(1, _QUANT_ROW_ELEM_BUDGET // max(k, 1))
    with torch.cuda.device(dev):
        for r0 in range(0, n, block):
            r1 = min(r0 + block, n)
            rot = _eager.rotate_int8_convrot_weight(weight[r0:r1].contiguous(), convrot_groupsize)
            # Offset per block like the eager packer, so blocks decorrelate yet stay deterministic.
            seed = stochastic_rounding + r0 if stochastic_rounding > 0 else 0
            _C.quantize_w4a8_convrot(
                _dl(rot.contiguous()),
                _dl(cb),
                _dl(packed[r0:r1]),
                _dl(s_rel[r0:r1].view(torch.uint8)),
                _dl(s_channel[r0:r1]),
                r1 - r0,
                k,
                seed > 0,
                seed,
                _stream(weight),
            )
            del rot
    return packed, s_rel, s_channel, None, cb


def quantize_w4a8_int8_weight(
    weight: torch.Tensor,
    group_size: int = 16,
    convrot_groupsize: int = 256,
    symmetric: bool = True,
    scale_dtype: torch.dtype = torch.float8_e4m3fn,
    codebook: bool = True,
    codebook_tensor: torch.Tensor | None = None,
    stochastic_rounding: int = 0,
    bits: int = 4,
    scale_search: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Prepare W4A8/W6A8 weights: the fused HIP ConvRot + requantize for the default
    configs (4-bit symmetric codebook g16, or 6-bit without the quantize-time scale
    search; fp8 scales). W4A8 the fused kernel declines takes the staged HIP requantize;
    everything else the chunked eager path."""
    validate_w4a8_weight_shape(weight, group_size, convrot_groupsize, bits)
    layout_ok = (
        scale_dtype == torch.float8_e4m3fn
        and symmetric
        and (
            (bits == 4 and codebook and group_size == 16)
            or (bits == 6 and not scale_search and group_size in (16, 32, 64))
        )
    )
    fused_ok = (
        layout_ok
        and convrot_groupsize == 256
        and _requant_supported(weight.shape[1], group_size, weight.device, weight.dtype)
    )
    staged_ok = (
        layout_ok
        and not fused_ok
        and bits == 4
        and _W4A8_STAGED_QUANT
        and weight.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and weight.shape[1] <= _W4A8_STAGED_MAX_K
    )
    if fused_ok or staged_ok:
        cb = None
        if bits == 4:
            cb = (
                codebook_tensor
                if codebook_tensor is not None
                else _decide_codebook(
                    weight, _eager.rotate_int8_convrot_weight, group_size, convrot_groupsize
                )
            )
        if staged_ok:
            return _quantize_w4a8_staged(weight, cb, convrot_groupsize, stochastic_rounding)
        return _fused_quantize_wxa8(weight, bits, group_size, cb, stochastic_rounding)
    return _quantize_w4a8_chunked(
        weight,
        _eager.rotate_int8_convrot_weight,
        group_size,
        convrot_groupsize,
        symmetric=symmetric,
        scale_dtype=scale_dtype,
        codebook=codebook,
        codebook_override=codebook_tensor,
        stochastic_rounding=stochastic_rounding,
        bits=bits,
        scale_search=scale_search,
    )


def dequantize_w4a8_int8_weight(
    qdata: torch.Tensor,
    s_rel: torch.Tensor,
    s_channel: torch.Tensor,
    codebook: torch.Tensor | None = None,
    correction: torch.Tensor | None = None,
    group_size: int = 16,
    convrot_groupsize: int = 256,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode W4A8 storage into its physical [N, K] floating weight."""
    validate_w4a8_operands(
        qdata, s_rel, s_channel, codebook, correction, group_size, convrot_groupsize
    )
    int8_weight = _dequant_int4_grouped_to_int8(qdata, s_rel, codebook, group_size)
    weight_rotated = _dequantize_w4a8_int8_weight_from_int8(
        int8_weight, s_channel, correction, group_size, output_dtype
    )
    # Rotating back has no HIP kernel. Eager applies the same orthonormal ConvRot
    # transform the fused activation path uses, and it is its own inverse.
    return _eager.rotate_int8_convrot_weight(weight_rotated, convrot_groupsize).to(output_dtype)


def w4a8_int8_linear(
    x: torch.Tensor,
    qdata: torch.Tensor,
    s_rel: torch.Tensor,
    s_channel: torch.Tensor,
    codebook: torch.Tensor | None = None,
    correction: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    group_size: int = 16,
    convrot_groupsize: int = 256,
    out_dtype: torch.dtype = torch.bfloat16,
    input_act: str | None = None,
    input_act_weight: torch.Tensor | None = None,
    input_act_eps: float = 0.0,
    residual: torch.Tensor | None = None,
    residual_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x @ W.T + bias`` via the HIP INT4/INT6 decode feeding the WMMA INT8 GEMM.

    The weight is decoded a column chunk at a time so a chunk is still cached when
    the GEMM reads it back, instead of the whole [N, K] INT8 weight round-tripping
    through global memory. At decode row counts even that is more traffic than the
    packed weight itself, so a few rows take a GEMV that dequantizes in registers
    and never writes the INT8 weight at all.
    """
    if residual is not None:
        return _apply_residual(
            w4a8_int8_linear(
                x, qdata, s_rel, s_channel, codebook, correction, bias, group_size,
                convrot_groupsize, out_dtype, input_act, input_act_weight,
                input_act_eps,
            ),
            residual,
            residual_scale,
        )
    x, input_act, act_weight = _split_input_act(
        x, input_act, input_act_weight, input_act_eps, True, convrot_groupsize)
    _n, k, _bits = validate_w4a8_operands(
        qdata, s_rel, s_channel, codebook, correction, group_size, convrot_groupsize
    )
    if x.shape[-1] != k * _input_act_width(input_act):
        raise ValueError(f"Input K={x.shape[-1]} does not match qdata K={k}")
    # The layout allows any ConvRot group that divides K; INT8 G=256 can spill to
    # global memory when K exceeds the fused LDS budget. int8_linear applies the
    # same test before its own fast path.
    convrot_ok = _convrot_supported(
        k, convrot_groupsize, x.device, x.dtype, int8_global_spill=True
    )
    # Only the HIP ConvRot quantizer absorbs the activation.
    if correction is not None or not convrot_ok:
        x = _apply_input_act(x, input_act, input_act_weight, input_act_eps)
        input_act = None

    # The asymmetric zero-point correction is a rank-one term the INT8 epilogue
    # cannot express, so that layout runs off the dequantized weight instead.
    if correction is not None:
        weight = dequantize_w4a8_int8_weight(
            qdata,
            s_rel,
            s_channel,
            codebook=codebook,
            correction=correction,
            group_size=group_size,
            convrot_groupsize=convrot_groupsize,
            output_dtype=x.dtype,
        )
        return torch.nn.functional.linear(x, weight, bias).to(out_dtype)

    if not convrot_ok:
        int8_weight = _dequant_int4_grouped_to_int8(qdata, s_rel, codebook, group_size)
        return _eager.int8_linear(
            x, int8_weight, s_channel, bias, out_dtype, True, convrot_groupsize
        )

    return _w4a8_int8_linear_chunked(
        x, qdata, s_rel, s_channel, codebook, bias, group_size, convrot_groupsize, out_dtype,
        input_act, act_weight, input_act_eps,
    )


# ---------------------------------------------------------------------------
# ConvRot W4A4 (v_wmma_i32_16x16x32_iu4)
# ---------------------------------------------------------------------------

_INT4_GROUP_SIZE = 64


def quantize_convrot_w4a4_weight(
    weight: torch.Tensor,
    convrot_groupsize: int = 256,
    quant_group_size: int = _INT4_GROUP_SIZE,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if quant_group_size != _INT4_GROUP_SIZE:
        raise ValueError(f"int4 MMA kernel requires quant_group_size {_INT4_GROUP_SIZE}")
    if stochastic_rounding or not _convrot_supported(
        weight.shape[-1], convrot_groupsize, weight.device, weight.dtype
    ):
        return _eager.quantize_convrot_w4a4_weight(
            weight, convrot_groupsize, quant_group_size, stochastic_rounding
        )

    w = weight.reshape(-1, weight.shape[-1]).contiguous()
    n, k = w.shape
    q = torch.empty((n, k // 2), dtype=torch.int8, device=w.device)
    scales = torch.empty((n,), dtype=torch.float32, device=w.device)
    # Pin the current device so check_convrot_k reads this operand's LDS budget.
    with torch.cuda.device(w.device):
        _C.convrot_quant_int4(_dl(w), _dl(q), _dl(scales), n, k, convrot_groupsize, _stream(w))
    return q, scales


def dequantize_convrot_w4a4_weight(
    qdata: torch.Tensor,
    scales: torch.Tensor,
    convrot_groupsize: int = 256,
    quant_group_size: int = _INT4_GROUP_SIZE,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    # Only the nibble unpack runs on device; the inverse rotation reuses the
    # eager Hadamard.
    from comfy_kitchen.backends.eager.convrot_w4a4 import _build_hadamard, _rotate_weight

    if quant_group_size != _INT4_GROUP_SIZE:
        raise ValueError(f"int4 MMA kernel requires quant_group_size {_INT4_GROUP_SIZE}")

    n, kp = qdata.shape
    unpacked = torch.empty((n, kp * 2), dtype=torch.int8, device=qdata.device)
    _C.unpack_int4(_dl(qdata.contiguous()), _dl(unpacked), n * kp, _stream(qdata))

    w_rot = unpacked.float() * scales.to(device=qdata.device, dtype=torch.float32).reshape(-1, 1)
    h = _build_hadamard(convrot_groupsize, device=qdata.device, dtype=torch.float32)
    return _rotate_weight(w_rot, h, convrot_groupsize).to(output_dtype)


def convrot_w4a4_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    bias: torch.Tensor | None = None,
    convrot_groupsize: int = 256,
    quant_group_size: int = _INT4_GROUP_SIZE,
    linear_dtype: str = "int4",
) -> torch.Tensor:
    if linear_dtype not in {"int4", "int8"}:
        raise ValueError(f"ConvRot W4A4 linear_dtype must be 'int4' or 'int8', got {linear_dtype!r}")
    if quant_group_size != _INT4_GROUP_SIZE:
        raise ValueError(f"int4 MMA kernel requires quant_group_size {_INT4_GROUP_SIZE}")
    if x.shape[-1] != qweight.shape[-1] * 2:
        raise ValueError(f"Input K={x.shape[-1]} does not match qweight K={qweight.shape[-1] * 2}")
    # An empty output (zero rows, or zero output features) does no packed reads; the
    # launcher accepts it, so return before the alignment checks below rather than
    # tripping over them. A zero-K input is not empty: its output is a pure bias
    # broadcast (an empty contraction sums to zero), so build it here rather than
    # launch, and never leak uninitialized values through torch.empty.
    if 0 in x.shape[:-1] or qweight.shape[0] == 0:
        return torch.empty((*x.shape[:-1], qweight.shape[0]), dtype=x.dtype, device=x.device)
    if x.shape[-1] == 0:
        out = torch.zeros((*x.shape[:-1], qweight.shape[0]), dtype=x.dtype, device=x.device)
        if bias is not None:
            out = out + _bias_operand(bias, qweight.shape[0], x.device)
        return out
    # The tile loader reads the packed row (K/2 bytes) in 16-byte chunks. A group
    # size of 16 alone would allow a K that packs to a partial chunk.
    if x.shape[-1] % 32 != 0:
        raise ValueError(f"convrot_w4a4_linear requires K divisible by 32, got {x.shape[-1]}")
    # A group size the kernel does not implement falls back to eager below rather
    # than raising here, so only the accepted sizes get the divisibility check.
    # Testing membership first also keeps a zero group size off the modulo.
    if convrot_groupsize in (16, 64, 256) and x.shape[-1] % convrot_groupsize != 0:
        raise ValueError(
            f"Input K={x.shape[-1]} not divisible by convrot_groupsize {convrot_groupsize}"
        )
    if not _convrot_supported(x.shape[-1], convrot_groupsize, x.device, x.dtype):
        return _eager.convrot_w4a4_linear(
            x, qweight, wscales, bias, convrot_groupsize, quant_group_size, linear_dtype
        )

    if linear_dtype == "int8":
        # As on CUDA: the int4 weight is unpacked to int8 values and the whole
        # linear runs on the int8 WMMA kernel with int8 activations.
        qw = _operand(qweight, x.device, "qweight")
        w_int8 = torch.empty((qw.shape[0], qw.shape[1] * 2), dtype=torch.int8, device=x.device)
        _C.unpack_int4(_dl(qw), _dl(w_int8), qw.numel(), _stream(x))
        return int8_linear(
            x, w_int8, wscales, bias, x.dtype,
            convrot=True, convrot_groupsize=convrot_groupsize,
        )

    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1]).contiguous()
    m, k = x2d.shape
    n = qweight.shape[0]

    qact = torch.empty((m, k // 2), dtype=torch.int8, device=x.device)
    x_scale = torch.empty((m,), dtype=torch.float32, device=x.device)
    # Pin the current device so check_convrot_k reads this operand's LDS budget.
    with torch.cuda.device(x.device):
        _C.convrot_quant_int4(_dl(x2d), _dl(qact), _dl(x_scale), m, k, convrot_groupsize, _stream(x))

    wscales = wscales.to(device=x.device, dtype=torch.float32).reshape(-1)
    if wscales.numel() != n:
        raise ValueError(f"wscales must have {n} entries, got {wscales.numel()}")
    if bias is not None:
        bias = _bias_operand(bias, n, x.device)
    qw = _operand(qweight, x.device, "qweight", shape=(n, k // 2))

    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    # The launcher picks the tile from the launching device's own architecture:
    # MmaInt4 where there are matrix cores, the VALU v_dot8_i32_i4 tile on RDNA2.
    # Deciding here instead would be wrong on a box with both an iGPU and a dGPU,
    # because has_wmma() is the intersection over every visible device and so
    # reports false for the whole process -- which would send the dGPU's calls to
    # an instruction it does not have. That makes "the launching device" load
    # bearing, so pin it. See _device_pin for why it is skipped when there is
    # only one visible device.
    with _device_pin(x.device):
        _C.convrot_w4a4_gemm(
            _dl(qact), _dl(qw), _dl(out),
            _dl(x_scale), _dl(wscales), None if bias is None else _dl(bias),
            m, n, k, DTYPE_TO_CODE[x.dtype], _stream(x),
        )
    return out.reshape(*orig_shape[:-1], n)


# ---------------------------------------------------------------------------
# AWQ W4A16 and SVDQuant W4A4
# ---------------------------------------------------------------------------

def gemv_awq_w4a16(
    x: torch.Tensor,
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    wzeros: torch.Tensor,
    bias: torch.Tensor | None = None,
    group_size: int = 64,
) -> torch.Tensor:
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1]).contiguous()
    m, k = x2d.shape
    n = qweight.shape[0]

    # The inner loop decodes eight weights at a time and rescales the chunk once,
    # so a chunk must not straddle a group boundary.
    if group_size <= 0 or group_size % 8 != 0:
        raise ValueError(f"group_size must be a positive multiple of 8, got {group_size}")
    if k % group_size != 0:
        raise ValueError(f"K={k} not divisible by group_size={group_size}")
    if qweight.shape[1] * 2 != k:
        raise ValueError(f"qweight K//2={qweight.shape[1]} inconsistent with x K={k}")

    # The kernel decodes scales and zeros with a single dtype code, taken from
    # wscales; a wzeros of another dtype would be read as that one.
    wscales = _operand(wscales, x.device, "wscales", shape=(k // group_size, n))
    if wscales.dtype not in _EPILOGUE_DTYPES:
        raise ValueError(f"wscales dtype {wscales.dtype} is not supported")
    wzeros = _operand(wzeros, x.device, "wzeros", shape=(k // group_size, n)).to(wscales.dtype)
    qw = _operand(qweight, x.device, "qweight", shape=(n, k // 2))
    if bias is not None:
        bias = _bias_operand(bias, n, x.device)

    out_dtype = wscales.dtype
    out = torch.empty((m, n), dtype=out_dtype, device=x.device)
    _C.gemv_awq_w4a16(
        _dl(x2d), _dl(qw), _dl(wscales), _dl(wzeros),
        None if bias is None else _dl(bias), _dl(out),
        m, n, k, group_size, _stream(x),
    )
    return out.reshape(*orig_shape[:-1], n)


_SVD_GROUP = 64


def quantize_svdquant_w4a4(
    x: torch.Tensor,
    smooth: torch.Tensor,
    lora_down: torch.Tensor,
    pad_size: int = 256,
    act_unsigned: bool = False,
    lora_x: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.dim() != 2:
        raise ValueError(f"expected 2D input, got shape {tuple(x.shape)}")
    m, k = x.shape
    if k % _SVD_GROUP != 0:
        raise ValueError(f"K={k} not divisible by group_size={_SVD_GROUP}")

    if pad_size <= 0:
        raise ValueError(f"pad_size must be positive, got {pad_size}")
    if lora_down.dim() != 2:
        raise ValueError(f"lora_down must be 2D, got {tuple(lora_down.shape)}")

    r = lora_down.shape[1]
    m_pad = -(-m // pad_size) * pad_size

    # The kernels take M, K and R with no bounds of their own, so every operand
    # has to match those extents and sit on x's device.
    xc = x.contiguous()
    # The kernel decodes smooth with the same dtype code it uses for ascales, which
    # is allocated from x.dtype, so a smooth of any other dtype would be read as
    # x.dtype and misdecoded.
    smooth = _operand(smooth.reshape(-1).to(x.dtype), x.device, "smooth", shape=(k,))
    lora_down = _operand(lora_down, x.device, "lora_down", shape=(k, r))
    # The LoRA branch is defined on the un-shifted, un-smoothed activation.
    lora_src = _operand(lora_x if lora_x is not None else x, x.device, "lora_x", shape=(m, k))

    # Padded rows stay zero: q = 0 and scale = 0 contribute nothing downstream.
    q = torch.zeros((m_pad, k // 2), dtype=torch.int8, device=x.device)
    ascales = torch.zeros((k // _SVD_GROUP, m_pad), dtype=x.dtype, device=x.device)
    lora_act = torch.zeros((m_pad, r), dtype=torch.float32, device=x.device)

    _C.svdquant_quantize(
        _dl(xc), _dl(smooth), _dl(q), _dl(ascales),
        m, m_pad, k, act_unsigned, _stream(x),
    )
    _C.svdquant_lora_down(
        _dl(lora_src), _dl(lora_down), _dl(lora_act[:m]), m, k, r, _stream(x)
    )
    return q, ascales, lora_act


def scaled_mm_svdquant_w4a4(
    act: torch.Tensor,
    wgt: torch.Tensor,
    ascales: torch.Tensor,
    wscales: torch.Tensor,
    lora_act_in: torch.Tensor,
    lora_up: torch.Tensor,
    bias: torch.Tensor | None = None,
    act_unsigned: bool = False,
) -> torch.Tensor:
    # act is the padded output of quantize_svdquant_w4a4, so M here is m_pad, which
    # is also the row stride of ascales. The kernel indexes ascales as g * M + row.
    if act.dim() != 2 or wgt.dim() != 2 or lora_up.dim() != 2:
        raise ValueError("act, wgt and lora_up must be 2D")

    m, k_half = act.shape
    n = wgt.shape[0]
    k = k_half * 2
    r = lora_up.shape[1]

    # One scale per 64-element group: a partial trailing group has no scale, and the
    # (K // 64, ...) scale shapes below would silently truncate it away.
    if k % _SVD_GROUP != 0:
        raise ValueError(f"K={k} not divisible by group_size={_SVD_GROUP}")

    # The kernel gets M, N, K and R but no tensor bounds, so each operand has to
    # match those extents and share act's device.
    dev = act.device
    act = _operand(act, dev, "act")
    wgt = _operand(wgt, dev, "wgt", shape=(n, k_half))
    ascales = _operand(ascales, dev, "ascales", shape=(k // _SVD_GROUP, m))
    wscales = _operand(wscales, dev, "wscales", shape=(k // _SVD_GROUP, n))
    if wscales.dtype not in _EPILOGUE_DTYPES:
        raise ValueError(f"wscales dtype {wscales.dtype} is not supported")
    lora_act_in = _operand(lora_act_in, dev, "lora_act_in", shape=(m, r))
    lora_up = _operand(lora_up, dev, "lora_up", shape=(n, r))
    if bias is not None:
        bias = _bias_operand(bias, n, dev)

    out = torch.empty((m, n), dtype=wscales.dtype, device=dev)
    _C.svdquant_gemm(
        _dl(act), _dl(wgt), _dl(out),
        _dl(ascales), _dl(wscales),
        _dl(lora_act_in), _dl(lora_up),
        None if bias is None else _dl(bias),
        m, n, k, r, act_unsigned, _stream(act),
    )
    return out


# ---------------------------------------------------------------------------
# Normalization and positional encoding
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Neighborhood attention
# ---------------------------------------------------------------------------


def na3d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: Sequence[int],
    is_causal: Sequence[bool] | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Fused 3D neighborhood attention (NATTEN ``na3d`` semantics) over
    ``(B, T, H, W, NH, HD)`` tensors, on WMMA. See ops/na3d.hip."""
    causal = (False, False, False) if is_causal is None else tuple(is_causal)
    batch, t, h, w, num_heads, head_dim = q.shape
    if scale is None:
        scale = head_dim ** -0.5

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = torch.empty_like(q)
    _C.na3d(
        _dl(q),
        _dl(k),
        _dl(v),
        _dl(out),
        batch,
        t,
        h,
        w,
        num_heads,
        head_dim,
        int(kernel_size[0]),
        int(kernel_size[1]),
        int(kernel_size[2]),
        int(causal[0]),
        int(causal[1]),
        int(causal[2]),
        float(scale),
        DTYPE_TO_CODE[q.dtype],
        _stream(q),
    )
    return out


def _adaln_impl(kernel, x, scale, shift, eps) -> torch.Tensor:
    """``kernel`` is _C.adaln (LayerNorm) or _C.rms_adaln (RMSNorm)."""
    from comfy_kitchen.backends._modulation import adaln_prep_modulation

    orig_shape = x.shape
    d = x.shape[-1]
    n = x.numel() // d

    x_flat = x.reshape(n, d).contiguous()
    scale_flat, scale_group = adaln_prep_modulation(scale, x, n, d)
    shift_flat, shift_group = adaln_prep_modulation(shift, x, n, d)

    out = torch.empty_like(x_flat)
    kernel(
        _dl(x_flat), _dl(scale_flat), _dl(shift_flat), _dl(out),
        n, d, scale_group, shift_group, eps, _stream(x),
    )
    return out.reshape(orig_shape)


def adaln(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    return _adaln_impl(_C.adaln, x, scale, shift, eps)


def rms_adaln(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    return _adaln_impl(_C.rms_adaln, x, scale, shift, eps)


def _ndhwc_strides(t: torch.Tensor):
    """(w, h, d, n) element strides of an NDHWC-ordered view with a dense channel row, else None.
    Strides must keep the kernel's 16-byte vector loads aligned."""
    n, c = t.shape[0], t.shape[1]
    st = t.stride()
    # a zero H stride is the kernel's packed marker, so a broadcast view is copied instead
    if t.dim() != 5 or st[1] != 1 or st[4] != c or min(st[2], st[3]) <= 0:
        return None
    # the batch stride is never applied for a batch of one, and a frame window carries the
    # whole tensor's, which may not fit the kernel's 32-bit strides
    strides = (st[4], st[3], st[2], 0 if n == 1 else st[0])
    if any(s % 8 for s in strides) or any(s > 2**31 - 1 for s in strides):
        return None
    return strides


def _wmma_fp16_conv3d(x, weight, bias, residual, stride, out=None):
    """The fused kernel's result, or None when it does not apply to this call. x and out may
    be NDHWC-ordered views of larger tensors, so a tiled convolution needs no per-tile copies."""
    n, c, d, h, w = x.shape
    k, _, t, r, s = weight.shape
    sd, sh, sw = stride
    if min(sd, sh, sw) < 1:
        return None  # torch's conv reports the bad stride
    z, p, q = (d - t) // sd + 1, (h - r) // sh + 1, (w - s) // sw + 1
    supported = (
        x.dtype == torch.float16 and weight.dtype == torch.float16
        and weight.device == x.device and weight.shape[1] == c
        and (c % 8 == 0 or c < 8) and k % 8 == 0 and d >= t and h >= r and w >= s
        and (bias is None or (bias.dtype == torch.float16 and bias.shape == (k,)))
        and (residual is None or (residual.dtype == torch.float16
                                  and residual.shape == (n, k, z, p, q)))
    )
    if not supported:
        return None
    if c < 8:
        # zero-pad pixel-sized channel counts (conv_in) to the 8-channel register width
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, 0, 0, 8 - c))
        weight = torch.nn.functional.pad(weight, (0, 0, 0, 0, 0, 0, 0, 8 - c))
        c = 8
    cl = torch.channels_last_3d
    xs = _ndhwc_strides(x)
    if xs is None or x.is_contiguous(memory_format=cl):
        x = x.contiguous(memory_format=cl)
        xs = (0, 0, 0, 0)
    weight = weight.contiguous(memory_format=cl)
    # the epilogue reads both off raw pointers on x's stream
    bias = None if bias is None else bias.to(device=x.device).contiguous()
    residual = (None if residual is None
                else residual.to(device=x.device).contiguous(memory_format=cl))
    if out is None:
        out = torch.empty((n, k, z, p, q), dtype=torch.float16, device=x.device, memory_format=cl)
    else:
        if out.shape != (n, k, z, p, q) or out.dtype != torch.float16 or out.device != x.device:
            raise ValueError("fp16_conv3d: out must be an fp16 [N, K, Z, P, Q] tensor on x's device")
        # the epilogue writes a packed [N*Z*P*Q, K] matrix, so besides a packed tensor only a
        # frame window of a single batch fits
        if not out.is_contiguous(memory_format=cl) and _ndhwc_strides(out) != (
                k, q * k, p * q * k, 0 if n == 1 else z * p * q * k):
            return None
        # blocks read operands other blocks may already have overwritten
        if any(t is not None and _tensors_overlap(out, t) for t in (x, weight, bias, residual)):
            return None
    # the tile stager issues 16-byte loads from x and the weight
    if x.data_ptr() % 16 or weight.data_ptr() % 16 or out.data_ptr() % 16:
        return None
    served = _C.fp16_conv3d(
        _dl(x), _dl(weight), None if bias is None else _dl(bias),
        None if residual is None else _dl(residual), _dl(out),
        n, d, h, w, c, k, t, r, s, sd, sh, sw, _stream(x), *xs,
    )
    return out if served else None


def fp16_conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    stride: list[int],
) -> torch.Tensor:
    """FP16 conv3d with fused bias/residual, channels_last_3d in and out; torch's
    conv when the kernel declines the shape."""
    out = _wmma_fp16_conv3d(x, weight, bias, residual, stride)
    if out is not None:
        return out
    # the kernel path takes bias and residual from any device, so the fallback must too
    bias = None if bias is None else bias.to(device=x.device)
    residual = None if residual is None else residual.to(device=x.device)
    out = torch.nn.functional.conv3d(x, weight, bias, stride=stride).contiguous(
        memory_format=torch.channels_last_3d)
    return out if residual is None else out + residual


def fp16_conv3d_out(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    stride: list[int],
    out: torch.Tensor,
) -> None:
    """fp16_conv3d into ``out``; computed then copied when the kernel cannot index ``out``."""
    if _wmma_fp16_conv3d(x, weight, bias, residual, stride, out=out) is not None:
        return
    res = fp16_conv3d(x, weight, bias, residual, stride)
    if (_wants_nchw_transpose(out) and out.shape == res.shape
            and out.dtype == res.dtype
            and res.is_contiguous(memory_format=torch.channels_last_3d)):
        _C.ndhwc_to_nchw(_dl(res), _dl(out), _stream(x))
        return
    out.copy_(res)


def _wants_nchw_transpose(out: torch.Tensor) -> bool:
    """Whether filling an NCDHW ``out`` from a channels_last result wants the transpose kernel.

    The kernels here write channels_last. A caller who wants an NCDHW ``out``
    has it copied, and on this part that copy is the slow direction: the
    destination's inner dimension is the source's outer one, so every element
    lands in a 32-byte sector it cannot share with a neighbour. Measured on the
    6-WWP 780M, 268 MB: 2.3 GB/s that way round against 58.7 GB/s the other
    way. The transpose kernel does it in two coalesced passes.
    """
    return (out.is_contiguous()
            and not out.is_contiguous(memory_format=torch.channels_last_3d)
            and out.dtype in (torch.float16, torch.bfloat16)
            and out.dim() == 5
            and out.shape[1] % 8 == 0
            and out.data_ptr() % 16 == 0)


def group_norm_silu_pad3d(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    num_groups: int,
    eps: float,
    pad: list[int],
    silu: bool,
    zero_pad: bool = False,
    out: torch.Tensor | None = None,
    stats: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-frame GroupNorm + SiLU + causal conv padding in one pass; the result is
    channels_last_3d, like the CUDA kernel. Shapes it declines run the eager op. ``stats`` from
    group_norm_stats3d normalizes a row strip of a frame with that frame's statistics. ``out``
    is written in place where the kernel can index it, else copied into."""
    b, c, t, h, w = x.shape
    left, right, top, bottom, front = pad
    if min(pad) < 0:
        raise ValueError("group_norm_silu_pad3d: padding must be non-negative")
    oshape = (b, c, t + front, h + top + bottom, w + left + right)
    if out is not None and (out.shape != oshape or out.dtype != x.dtype or out.device != x.device):
        raise ValueError(f"group_norm_silu_pad3d: out must be {oshape} {x.dtype} on {x.device}")
    # the affine params may be fp32 (the registry admits it); every path wants x's dtype
    if weight is not None:
        weight = weight.to(device=x.device, dtype=x.dtype).contiguous()
        bias = (torch.zeros(c, dtype=x.dtype, device=x.device) if bias is None
                else bias.to(device=x.device, dtype=x.dtype).contiguous())
    else:
        bias = None  # pad-only ignores bias, as eager and CUDA do
    if stats is not None and (stats.shape, stats.dtype, stats.device) != ((b * t, num_groups, 2), torch.float32, x.device):
        raise ValueError(f"group_norm_silu_pad3d: stats must be float32 [{b * t}, {num_groups}, 2] on {x.device}")
    # an empty frame still has a zero border to write, which the launcher skips
    served = not (c % 8 or c > 2048 or h == 0 or w == 0
                  or (weight is not None and (c % num_groups or num_groups > 1024))
                  or (not zero_pad and (max(left, right) >= w or max(top, bottom) >= h))
                  or b * (t + front) > 65535)
    row_stride = _ndhwc_row_stride(x) if served else None
    if served and row_stride is None:
        row_stride = w * c
        x = x.contiguous(memory_format=torch.channels_last_3d)
    # the kernel loads whole 16-byte registers; a misaligned view takes eager, as on CUDA
    served = served and x.data_ptr() % 16 == 0
    if not served:
        res = _eager.group_norm_silu_pad3d(x, weight, bias, num_groups, eps, pad, silu, zero_pad, stats)
        if out is None:
            return res
        # The eager result keeps the channel axis between the time and the
        # height, so its copy into out is not the scattered direction.
        out.copy_(res)
        return out

    # frames are written in no particular order, so an out overlapping an operand goes via a copy
    dst = (out if out is not None and _writes_packed_ndhwc(out) and out.data_ptr() % 16 == 0
           and not any(t is not None and _tensors_overlap(out, t) for t in (x, weight, bias))
           else None)
    # An NCDHW out cannot be written by the kernel (it writes channels_last), and a
    # plain copy into it is the slow direction on this part: the destination's inner
    # dimension is the source's outer one, so every element lands in a sector it
    # cannot share. Measured on the 6-WGP 780M, a 268 MB copy runs at 2.3 GB/s that
    # way round against 58.7 GB/s the other way. The kernel's own transpose does it
    # in two coalesced passes.
    ncdhw_out = (out is not None and dst is None and _wants_nchw_transpose(out))
    if dst is None:
        dst = torch.empty(oshape, dtype=x.dtype, device=x.device,
                          memory_format=torch.channels_last_3d)
    workspace = None
    if weight is None:
        stats = None
    elif stats is None:
        chunks = -(-(h * w) // 1024)
        workspace = torch.empty(2 * b * t * (chunks * c + num_groups), dtype=torch.float32,
                                device=x.device)
    else:
        stats = stats.contiguous()
    _C.group_norm_silu_pad3d(
        _dl(x), None if weight is None else _dl(weight), None if bias is None else _dl(bias),
        _dl(dst), None if workspace is None else _dl(workspace), None if stats is None else _dl(stats),
        b, c, t, h, w, num_groups, eps, left, right, top, bottom, front, silu, zero_pad, row_stride,
        _stream(x),
    )
    if out is None or dst is out:
        return dst
    if ncdhw_out:
        _C.ndhwc_to_nchw(_dl(dst), _dl(out), _stream(x))
        return out
    out.copy_(dst)
    return out


def _ndhwc_row_stride(x: torch.Tensor):
    """Elements between image rows when x is a single frame with packed NHWC rows that sit farther
    apart than W * C (a row-interleaved view the kernels read in place); None otherwise."""
    b, c, t, h, w = x.shape
    if (t == 1 and x.stride(1) == 1 and x.stride(4) == c and x.stride(3) > w * c and x.stride(3) % 8 == 0
            and (b == 1 or x.stride(0) == h * x.stride(3))):
        return x.stride(3)
    return None


def group_norm_stats3d(x: torch.Tensor, num_groups: int, eps: float) -> torch.Tensor:
    """Per-frame GroupNorm (mean, rstd) of x [B, C, T, H, W] as float32 [B*T, num_groups, 2],
    the same statistics group_norm_silu_pad3d computes for itself."""
    b, c, t, h, w = x.shape
    served = not (c % 8 or c > 2048 or c % num_groups or num_groups > 1024 or b * t > 65535
                  or h == 0 or w == 0)
    row_stride = _ndhwc_row_stride(x) if served else None
    if served and row_stride is None:
        row_stride = w * c
        x = x.contiguous(memory_format=torch.channels_last_3d)
    served = served and x.data_ptr() % 16 == 0
    if not served:
        return _eager.group_norm_stats3d(x, num_groups, eps)
    chunks = -(-(h * w) // 1024)
    workspace = torch.empty(2 * b * t * chunks * c, dtype=torch.float32, device=x.device)
    stats = torch.empty((b * t, num_groups, 2), dtype=torch.float32, device=x.device)
    _C.group_norm_stats3d(_dl(x), _dl(workspace), _dl(stats), b, c, t, h, w, num_groups, eps,
                          row_stride, _stream(x))
    return stats


def _writes_packed_ndhwc(out: torch.Tensor) -> bool:
    """Packed channels_last_3d, or for a batch of one a frame-offset view of a longer buffer."""
    if out.is_contiguous(memory_format=torch.channels_last_3d):
        return True
    b, c, _, h, w = out.shape
    st = out.stride()
    return b == 1 and st[1] == 1 and st[4] == c and st[3] == w * c and st[2] == h * w * c


def group_norm_silu_pad3d_out(x, weight, bias, num_groups, eps, pad, silu, zero_pad, out, stats=None) -> None:
    """group_norm_silu_pad3d into ``out``; a frame-offset view leaves room for a caller's halo."""
    group_norm_silu_pad3d(x, weight, bias, num_groups, eps, pad, silu, zero_pad, out, stats)


def _effective_strides(t: torch.Tensor) -> tuple[int, ...]:
    """Strides of the axes the kernels walk: a length-1 axis is only indexed at 0."""
    return tuple(s for s, n in zip(t.stride(), t.shape, strict=True) if n != 1)


def _rope_rows(q: torch.Tensor, k: torch.Tensor | None):
    """``q`` and ``k`` in a layout the rope kernels can address with one stride set.

    Both walk the inputs through their strides, so a q/k pair permuted or sliced out
    of a packed qkv is read where it lies. head_dim is read one element at a time,
    so a row that is not dense would scatter every load and is copied instead.
    """
    dense = q.stride(-1) == 1 and (k is None or k.stride(-1) == 1)
    matched = k is None or _effective_strides(q) == _effective_strides(k)
    if dense and matched:
        return q, k
    return q.contiguous(), None if k is None else k.contiguous()


def _rope(xq, xk, freqs_cis, split_half, inplace=False):
    # One dtype code and one stream are passed for the whole launch, so every
    # buffer has to agree on device, and xq/xk on dtype.
    if freqs_cis.device != xq.device:
        raise ValueError("freqs_cis must be on the same device as the input")
    if xk is not None:
        if xk.device != xq.device:
            raise ValueError("xq and xk must be on the same device")
        if xk.dtype != xq.dtype:
            raise ValueError("xq and xk must have the same dtype")

    if inplace:
        # Each thread owns one (a, b) pair and loads both before storing either,
        # so rotating a view where it lies is well defined. _rope_rows must not
        # run: its copy would move the result off the caller's storage.
        xq_out, xk_out = xq, xk
    else:
        # freqs_cis is indexed through its own strides, so a strided view is free.
        xq, xk = _rope_rows(xq, xk)
        xq_out = torch.empty(xq.shape, dtype=xq.dtype, device=xq.device)
        xk_out = None if xk is None else torch.empty(xk.shape, dtype=xk.dtype, device=xk.device)

    if not split_half:
        freqs_cis = trim_rope_freqs(xq, freqs_cis)

    _C.apply_rope(
        _dl(xq), None if xk is None else _dl(xk), _dl(freqs_cis),
        _dl(xq_out), None if xk_out is None else _dl(xk_out),
        split_half, _stream(xq),
    )
    return xq_out, xk_out


def _rope_pair(xq, xk, freqs_cis, split_half, inplace=False):
    # The kernel indexes one set of strides and reads both tensors with one dtype
    # code, so a difference in either is done one at a time. Out of place a stride
    # mismatch is resolved by densifying both instead.
    if (
        xq.shape != xk.shape
        or xq.dtype != xk.dtype
        or (inplace and _effective_strides(xq) != _effective_strides(xk))
    ):
        return (
            _rope(xq, None, freqs_cis, split_half, inplace)[0],
            _rope(xk, None, freqs_cis, split_half, inplace)[0],
        )
    return _rope(xq, xk, freqs_cis, split_half, inplace)


def apply_rope1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    return _rope(x, None, freqs_cis, False)[0]


def apply_rope1_(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    check_rope_inplace(x, readonly=(freqs_cis,))
    return _rope(x, None, freqs_cis, False, inplace=True)[0]


def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return _rope_pair(xq, xk, freqs_cis, False)


def apply_rope_(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    check_rope_inplace(xq, xk, readonly=(freqs_cis,))
    return _rope_pair(xq, xk, freqs_cis, False, inplace=True)


def apply_rope_split_half1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    return _rope(x, None, freqs_cis, True)[0]


def apply_rope_split_half1_(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    check_rope_inplace(x, readonly=(freqs_cis,))
    return _rope(x, None, freqs_cis, True, inplace=True)[0]


def apply_rope_split_half(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return _rope_pair(xq, xk, freqs_cis, True)


def apply_rope_split_half_(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    check_rope_inplace(xq, xk, readonly=(freqs_cis,))
    return _rope_pair(xq, xk, freqs_cis, True, inplace=True)


def _rms_rope_weight(scale: torch.Tensor, head_dim: int) -> torch.Tensor | None:
    """The weight as a contiguous 1D tensor of length ``head_dim``, else None.

    Anything eager's rms_norm would broadcast instead has no kernel.
    """
    if scale.dim() != 1 or scale.numel() != head_dim:
        return None
    return scale.contiguous()


def _rms_rope(q, k, freqs_cis, q_scale, k_scale, epsilon, split_half, inplace=False, rot_dim=0):
    if k is not None and k_scale is None:
        k_scale = q_scale
    # One dtype code and one stream are passed for the whole launch, so every
    # buffer has to agree on device, and q/k on dtype.
    if freqs_cis.device != q.device or q_scale.device != q.device:
        raise ValueError("freqs_cis and the scales must be on the same device as the input")
    if k is not None:
        if k.device != q.device or k_scale.device != q.device:
            raise ValueError("q and k must be on the same device")
        if k.dtype != q.dtype:
            raise ValueError("q and k must have the same dtype")

    q_weight = _rms_rope_weight(q_scale, q.shape[-1])
    k_weight = None if k is None else _rms_rope_weight(k_scale, q.shape[-1])
    if q_weight is None or (k is not None and k_weight is None):
        # No fused path; eager returns fresh tensors, so in place they are
        # written back through the views.
        if k is None:
            if rot_dim:
                # Only the private eager helper takes rot_dim for a single tensor.
                out = _eager_rope._rms_rope1(
                    q, freqs_cis, q_scale, epsilon, split_half=split_half, rot_dim=rot_dim)
            else:
                impl = _eager.rms_rope_split_half1 if split_half else _eager.rms_rope1
                out = impl(q, freqs_cis, q_scale, epsilon)
            return (q.copy_(out) if inplace else out), None
        # Only the split-half pair takes rot_dim upstream; rot_dim never reaches
        # the interleaved path, which has no public rot_dim argument.
        impl = _eager.rms_rope_split_half if split_half else _eager.rms_rope
        kwargs = {"rot_dim": rot_dim} if rot_dim and split_half else {}
        q_out, k_out = impl(q, k, freqs_cis, q_scale, k_scale, epsilon, **kwargs)
        if inplace:
            return q.copy_(q_out), k.copy_(k_out)
        return q_out, k_out

    if inplace:
        # See _rope: the row is re-read only after the norm's __syncthreads, so
        # rotating it in place is well defined.
        q_out, k_out = q, k
    else:
        # freqs_cis is indexed through its own strides, so a strided view is free.
        q, k = _rope_rows(q, k)
        q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        k_out = None if k is None else torch.empty(k.shape, dtype=k.dtype, device=k.device)

    _C.rms_rope(
        _dl(q), None if k is None else _dl(k), _dl(freqs_cis),
        _dl(q_weight), None if k_weight is None else _dl(k_weight),
        _dl(q_out), None if k_out is None else _dl(k_out),
        epsilon, split_half, _stream(q), rot_dim,
    )
    return q_out, k_out


def _rms_rope_pair(q, k, freqs_cis, q_scale, k_scale, epsilon, split_half, inplace=False,
                   rot_dim=0):
    if k_scale is None:
        k_scale = q_scale
    # One dtype code covers both inputs and one more both weights, so a difference
    # in either is done one at a time.
    if (
        q.shape != k.shape
        or q.dtype != k.dtype
        or q_scale.dtype != k_scale.dtype
        or (inplace and _effective_strides(q) != _effective_strides(k))
    ):
        return (
            _rms_rope(q, None, freqs_cis, q_scale, None, epsilon, split_half, inplace,
                      rot_dim)[0],
            _rms_rope(k, None, freqs_cis, k_scale, None, epsilon, split_half, inplace,
                      rot_dim)[0],
        )
    return _rms_rope(q, k, freqs_cis, q_scale, k_scale, epsilon, split_half, inplace, rot_dim)


def rms_rope1(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    return _rms_rope(x, None, freqs_cis, scale, None, epsilon, False)[0]


def rms_rope1_(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    check_rope_inplace(x, readonly=(freqs_cis, scale))
    return _rms_rope(x, None, freqs_cis, scale, None, epsilon, False, inplace=True)[0]


def rms_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _rms_rope_pair(q, k, freqs_cis, q_scale, k_scale, epsilon, False)


def rms_rope_(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k_scale is None:
        k_scale = q_scale
    check_rope_inplace(q, k, readonly=(freqs_cis, q_scale, k_scale))
    return _rms_rope_pair(q, k, freqs_cis, q_scale, k_scale, epsilon, False, inplace=True)


def rms_rope_split_half1(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    return _rms_rope(x, None, freqs_cis, scale, None, epsilon, True)[0]


def rms_rope_split_half1_(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    check_rope_inplace(x, readonly=(freqs_cis, scale))
    return _rms_rope(x, None, freqs_cis, scale, None, epsilon, True, inplace=True)[0]


def rms_rope_split_half(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6,
    rot_dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _rms_rope_pair(q, k, freqs_cis, q_scale, k_scale, epsilon, True, rot_dim=rot_dim)


def rms_rope_split_half_(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6,
    rot_dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k_scale is None:
        k_scale = q_scale
    check_rope_inplace(q, k, readonly=(freqs_cis, q_scale, k_scale))
    return _rms_rope_pair(q, k, freqs_cis, q_scale, k_scale, epsilon, True, inplace=True,
                          rot_dim=rot_dim)



# ---------------------------------------------------------------------------
# Sol-Attn sparse attention
#
# The kernels are in sage_attention/sol_attn*.hip. Everything below mirrors the
# CUDA backend's entry points, including the workspace plan, which both backends
# read from their own C++ Plan. The one place they diverge is what the carriers
# hold: the CUDA layout permutes the contraction and key axes for its m16n8k32
# fragments and this one does not, so the top-k threshold reads cen8 in channel
# order rather than un-permuting it.
# ---------------------------------------------------------------------------

_SOL_HD = 128
_SOL_VSCALE_MARGIN = 1.1   # clip headroom on last step's V absmax


def _topk_from_pooled(c8, csc, kc, topk_ratio, log2s, sinks):
    """Per-query-block top-k as a threshold: the k-th largest pooled score (the
    kernel keeps score >= threshold, so ties over-select). ``c8``/``csc`` are the
    quantized centroids ``[BH, N, D]``/``[BH, N]``, ``kc`` the centred pooled keys.
    Scores are computed through the route kernel's own int8 quantization and
    operation order; an fp32 threshold against int8 kernel scores inflates the kept
    budget ~15%. Sink blocks are always kept, so they neither count toward nor
    consume the budget."""
    n = kc.shape[1]
    ksc = (kc.abs().amax(-1, True) / 127.0).clamp_min(1e-12)     # prep_pooled_quant
    k8 = torch.round(kc / ksc).clamp_(-127, 127)
    s = torch.bmm(c8, k8.transpose(1, 2))     # integer dot, exact in fp32
    s = s * (csc * log2s).unsqueeze(-1) * ksc.squeeze(-1).unsqueeze(-2)
    s0, s1 = sinks
    if s1 > s0:
        s[..., s0:s1] = float("-inf")
    kk = _topk_count(n - _sink_count(n, s0, s1), topk_ratio)
    if kk == 0:   # nothing beyond the forced blocks: a threshold no finite score clears
        return torch.full(s.shape[:-1], float("inf"), device=s.device, dtype=s.dtype)
    kth = s.topk(kk, dim=-1, sorted=False).values.min(-1).values
    # backed off a few ulps: this replica of the kernel's scores is not bit-exact
    return (kth - kth.abs() * 1e-5).contiguous()


def _block_means(x, lengths=None, valid=None):
    """(B, T, H, D) -> [BH, N, D] fp32 means over each 64-block's live rows, with
    fp32 accumulation but no fp32 copy of x. ``lengths``/``valid`` only for
    zero-padded tiles; the default divides by Python floats so the routing
    threshold stays bit-identical run to run."""
    b, t, h, d = x.shape
    full = (t // 64) * 64
    if lengths is None:
        m = x[:, :full].view(b, -1, 64, h, d).sum(2, dtype=torch.float32) / 64.0
        if full != t:
            tail = x[:, full:].sum(1, keepdim=True, dtype=torch.float32) / (t - full)
            m = torch.cat([m, tail], dim=1)
    else:
        if valid is not None:
            x = x * valid.view(1, -1, 1, 1).to(x.dtype)
        m = x[:, :full].view(b, -1, 64, h, d).sum(2, dtype=torch.float32)
        if full != t:
            m = torch.cat([m, x[:, full:].sum(1, keepdim=True, dtype=torch.float32)], dim=1)
        m = m / lengths.view(1, -1, 1, 1)
    return m.permute(0, 2, 1, 3).reshape(b * h, -1, d)


def _check_block_len(block_len, t, device):
    n = (t + 63) // 64
    if block_len.dtype != torch.int32 or block_len.dim() != 1 or block_len.numel() != n:
        raise ValueError(
            f"sol_attn: block_len must be int32 of shape ({n},), got {block_len.dtype} "
            f"{tuple(block_len.shape)}")
    if block_len.device != device:
        raise ValueError(f"sol_attn: block_len must be on {device}, got {block_len.device}")
    return block_len.contiguous()


def _check_coarse_gate(coarse_gate, shape, device):
    if tuple(coarse_gate.shape) != tuple(shape) or coarse_gate.device != device:
        raise ValueError(
            f"sol_attn: coarse_gate must be {tuple(shape)} on {device}, got "
            f"{tuple(coarse_gate.shape)} on {coarse_gate.device}")
    return coarse_gate


def _topk_threshold(q, k, topk_ratio, scale, lengths, valid, sinks):
    """Top-k threshold from post-rope q/k, pooled and quantized like prep_q."""
    cen = _block_means(q, lengths, valid)
    kc = _block_means(k, lengths, valid)
    kc = kc - kc.mean(dim=1, keepdim=True)
    csc = (cen.abs().amax(-1, True) / 127.0).clamp_min(1e-8)      # prep_q
    c8 = torch.round(cen / csc).clamp_(-127, 127)
    return _topk_from_pooled(c8, csc.squeeze(-1), kc, topk_ratio, scale * _LOG2E, sinks)


_ROPE_FAB_CACHE = {}


def _packed_rope_fab(freqs, t, rot):
    """[..., T, 1, rot/2, 2, 2] -> [T, rot, 2] per-channel (self, partner)
    coefficients. Cached per freqs tensor (one attention step reuses one across
    layers); the weak ref validates an id() hit without retaining freqs."""
    key = (id(freqs), t, rot)
    hit = _ROPE_FAB_CACHE.get(key)
    if hit is not None and hit[0]() is freqs:
        return hit[1]
    f = freqs.reshape(-1, rot // 2, 2, 2)
    if f.shape[0] != t:
        raise ValueError(f"sol_attn: rope_freqs covers {f.shape[0]} tokens, T={t}")
    with allocation_context():
        fab = torch.empty(t, rot, 2, device=freqs.device, dtype=torch.float32)
    fab[:, :rot // 2, 0] = f[:, :, 0, 0]
    fab[:, :rot // 2, 1] = f[:, :, 0, 1]
    fab[:, rot // 2:, 0] = f[:, :, 1, 1]
    fab[:, rot // 2:, 1] = f[:, :, 1, 0]
    _ROPE_FAB_CACHE.clear()   # one live entry
    _ROPE_FAB_CACHE[key] = (weakref.ref(freqs), fab)
    return fab


def _sink_pair(value):
    return [0, 0] if value is None else [int(value[0]), int(value[1])]


def _check_sol_args(sink_blocks, sink_q, topk_ratio, **tensors):
    """Shared by both HIP entries, since a caller may bypass the registry: the
    registry rule plus a matrix-core check.

    No device argument, unlike the CUDA entry's per-device compute-capability
    test. has_wmma() is the intersection over every visible device, so it is
    already the conservative answer for whichever one the tensors are on.
    """
    from comfy_kitchen.constraints import sol_attn_common_call_rule

    if not has_wmma():
        raise RuntimeError(
            "sol_attn: requires RDNA3 or newer matrix cores (WMMA); this device has none")
    check = sol_attn_common_call_rule(
        {"sink_blocks": sink_blocks, "sink_q": sink_q, "topk_ratio": topk_ratio, **tensors})
    if not check.success:
        raise ValueError(f"sol_attn: {check.failed_param}: {check.failure_reason}")


def sol_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tau: float = 1.0,
    scale: float | None = None,
    sink_blocks: list[int] | None = None,
    sink_q: list[int] | None = None,
    key_bias: torch.Tensor | None = None,
    topk_ratio: float = 0.0,
    tail: bool = True,
    block_len: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
    token_aug: int = 0,
) -> torch.Tensor:
    """Sol-Attn sparse attention over ``(B, T, H, 128)`` bf16 or fp16 tensors.
    See sage_attention/sol_attn.hip and the public docstring for ``tail``,
    ``block_len`` and ``coarse_gate``.

    ``topk_ratio`` > 0 switches selection from the tau threshold to SLA-style
    per-query-block top-k: keep that fraction of key blocks per query block (sinks
    and the diagonal still ride on top). tau is ignored then.

    ``token_aug`` (0, or a multiple of 64 up to 256): on top of the routed blocks,
    every row attends up to that many unrouted tokens its query block's centroid
    scores highest, and the remaining tail is exact for the centroid instead of
    pooled per block.
    """
    batch, t, h, d = q.shape
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"sol_attn: q/k/v must be bfloat16 or float16, got {q.dtype}")
    _check_sol_args(sink_blocks, sink_q, topk_ratio, q=q, k=k, v=v,
                    block_len=block_len, coarse_gate=coarse_gate)
    if scale is None:
        scale = d ** -0.5
    if block_len is not None:
        block_len = block_len.contiguous()
    lengths = _block_lengths(t, (t + 63) // 64, q.device, block_len)
    valid = _valid_rows(t, lengths) if block_len is not None else None
    mean_lengths = lengths if block_len is not None else None   # exact default arithmetic
    # Only the last dim must be contiguous (16-byte staging loads), so a BHND view
    # goes in as-is. A misaligned load faults asynchronously and poisons the
    # context, so base pointer and leading strides are checked here.
    for name, x in (("q", q), ("k", k), ("v", v)):
        if x.stride(-1) != 1:
            raise ValueError(f"sol_attn: {name} must have a contiguous last dim")
        if x.data_ptr() % 16:
            raise ValueError(
                f"sol_attn: {name} must be 16-byte aligned (storage_offset "
                f"{x.storage_offset()} leaves it at +{x.data_ptr() % 16}); "
                f"call .contiguous() on it")
        for dim in range(3):
            if x.shape[dim] > 1 and x.stride(dim) % 8:
                raise ValueError(
                    f"sol_attn: {name} stride({dim}) = {x.stride(dim)} elements "
                    f"is not a multiple of 8, so the 16-byte staging loads would "
                    f"be misaligned; call .contiguous() on it")
    sb, sq = _sink_pair(sink_blocks), _sink_pair(sink_q)
    thr = (_topk_threshold(q, k, topk_ratio, scale, mean_lengths, valid, sb)
           if topk_ratio else None)
    kb = None
    if key_bias is not None:
        # exact branch only, in log2 units; biased blocks must be sink-covered
        kb = _normalize_key_bias(key_bias, batch, t, q.device)
        kb = (kb * _LOG2E).expand(batch, t).contiguous()
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    p = _C.sol_attn_plan(batch, t, h, token_aug=int(token_aug))
    workspace = torch.empty(p["total"], dtype=torch.uint8, device=q.device)
    _C.sol_attn(
        _dl(q), _dl(k), _dl(v), _dl(out), _dl(workspace),
        batch, t, h, d,
        float(tau), float(scale),
        sb[0], sb[1], sq[0], sq[1],
        _stream(q),
        key_bias=None if kb is None else _dl(kb),
        threshold=None if thr is None else _dl(thr),
        block_len=None if block_len is None else _dl(block_len),
        tail=bool(tail), token_aug=int(token_aug),
    )
    if coarse_gate is not None:
        add_coarse_(out, coarse_output(*_ws_block_means(workspace, p, batch * h, lengths), scale),
                    coarse_gate)
    return out


def sol_attn_chunked(
    qkv_chunks,
    t: int,
    h: int,
    rope_freqs: torch.Tensor,
    qk_norm_weights: tuple[torch.Tensor, torch.Tensor],
    kmean: torch.Tensor | None = None,
    vscale: torch.Tensor | None = None,
    tau: float = 1.0,
    topk_ratio: float = 0.0,
    scale: float | None = None,
    sink_blocks: list[int] | None = None,
    sink_q: list[int] | None = None,
    rope_eps: float = 1e-6,
    tail: bool = True,
    block_len: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
    token_aug: int = 0,
):
    """Chunked-producer Sol-Attn over fused qkv projection chunks ([M, 3*H*128]
    bf16, 64-aligned starts, B=1); full Q/K/V are never materialised.
    ``tail`` / ``block_len`` / ``coarse_gate`` / ``token_aug`` as in ``sol_attn``.

    ``qkv_chunks``: an iterable of chunks or a zero-arg callable returning one.
    ``kmean``/``vscale`` are LAST step's statistics ([H,128] f32); when None the
    producer runs twice (measure, then quantize), so pass a callable to stream on
    the first call. Returns ``(out[1,T,H,128] bf16, kmean_next, vscale_next)``."""
    d = _SOL_HD
    rot = rope_freqs.shape[-3] * 2
    # the fused rope pairs channels across lanes of 4: rot/2 must be lane-aligned
    if rot % 8 or not 0 < rot <= d:
        raise ValueError(f"sol_attn_chunked: rot_dim must be a multiple of 8 in (0, {d}], got {rot}")
    fab = _packed_rope_fab(rope_freqs, t, rot)
    dev = fab.device
    _check_sol_args(sink_blocks, sink_q, topk_ratio)
    if scale is None:
        scale = d ** -0.5
    if block_len is not None:
        block_len = _check_block_len(block_len, t, dev)
    if coarse_gate is not None:
        coarse_gate = _check_coarse_gate(coarse_gate, (1, t, h, d), dev)
    lengths = _block_lengths(t, (t + 63) // 64, dev, block_len)
    qw, kw = (w.to(device=dev, dtype=torch.bfloat16).contiguous() for w in qk_norm_weights)
    factory = qkv_chunks if callable(qkv_chunks) else None
    if factory is None and (kmean is None or vscale is None):
        qkv_chunks = list(qkv_chunks)          # need two passes over it
        factory = lambda: iter(qkv_chunks)     # noqa: E731
    p = _C.sol_attn_plan(1, t, h, token_aug=int(token_aug))
    ws = torch.empty(p["total"], dtype=torch.uint8, device=dev)
    stream = torch.cuda.current_stream(dev).cuda_stream
    width = 3 * h * d

    def produce(km, vsc):
        _C.sol_producer_begin(_dl(ws), 1, t, h, stream, token_aug=int(token_aug))
        t0 = 0
        for chunk in (factory() if factory is not None else qkv_chunks):
            m = chunk.shape[-2]
            if t0 % 64 and m:
                raise ValueError("sol_attn_chunked: chunk starts must be 64-aligned")
            # bare pointers below: a wrong width reads OOB, a host tensor poisons the context
            if chunk.shape[-1] != width or chunk.dtype != torch.bfloat16 or chunk.device != dev:
                raise ValueError(
                    f"sol_attn_chunked: chunks must be [M, {width}] bfloat16 on {dev}, "
                    f"got {tuple(chunk.shape)} {chunk.dtype} on {chunk.device}")
            _C.sol_producer_chunk(
                _dl(ws), _dl(chunk.contiguous()), _dl(fab), _dl(qw), _dl(kw), _dl(km), _dl(vsc),
                float(rope_eps), rot, t0, m, 1, t, h, stream,
                block_len=None if block_len is None else _dl(block_len),
                token_aug=int(token_aug))
            t0 += m
        if t0 != t:
            raise ValueError(f"sol_attn_chunked: chunks cover {t0} tokens, T={t}")

    def vscale_of(vamax):
        return (vamax / 127.0 * _SOL_VSCALE_MARGIN).clamp_min(1e-8)

    if kmean is None or vscale is None:
        # bootstrap: harvest the scale-independent statistics (post-rope K sums, V
        # absmax) with dummy scales, then produce for real
        produce(torch.zeros(h, d, device=dev), torch.ones(h, d, device=dev))
        kmean = _ws_ksums(ws, p, h).sum(1) / lengths.sum()   # live tokens, like prep_sums_to_means
        vscale = vscale_of(ws[p["statsV"]:p["statsV"] + h * d * 4].view(torch.float32))
    kmean = kmean.to(device=dev, dtype=torch.float32).contiguous()
    # a zero scale is 1/0 in the producer and 255/0 in route: clamp like vscale_of
    vscale = vscale.to(device=dev, dtype=torch.float32).clamp_min(1e-8).contiguous()
    produce(kmean, vscale)
    sb, sq = _sink_pair(sink_blocks), _sink_pair(sink_q)
    threshold = (_topk_threshold_from_workspace(ws, p, h, topk_ratio, scale, lengths, sb)
                 if topk_ratio else None)
    out = torch.empty(1, t, h, d, dtype=torch.bfloat16, device=dev)
    kmean_next = torch.empty(h, d, device=dev, dtype=torch.float32)
    vamax = torch.empty(h, d, device=dev, dtype=torch.float32)
    _C.sol_attn_core(
        _dl(ws), _dl(out), _dl(vscale), _dl(kmean_next), _dl(vamax),
        1, t, h, float(tau), float(scale), sb[0], sb[1], sq[0], sq[1], stream,
        threshold=None if threshold is None else _dl(threshold),
        block_len=None if block_len is None else _dl(block_len), tail=bool(tail),
        token_aug=int(token_aug))
    if coarse_gate is not None:
        add_coarse_(out, coarse_output(*_ws_block_means(ws, p, h, lengths), scale), coarse_gate)
    return out, kmean_next, vscale_of(vamax)


def _ws_ksums(ws, p, bh):
    """The producer's post-rope block K sums, [BH, NTB, 128] f32 view into scratch
    (block MEANS on the direct path, and once sol_attn_core has run)."""
    d = _SOL_HD
    return ws[p["scratch"]:p["scratch"] + bh * p["NPAD"] * d * 4] \
        .view(torch.float32).view(bh, p["NPAD"], d)[:, :p["NTB"]]


def _ws_block_means(ws, p, bh, lengths):
    """(qm, km, vm) ``[BH, NTB, 128]`` fp32 block means over live rows, read from
    what the kernels already left in the workspace: the qmean slot, K means in
    scratch, V sums in vcT (bf16)."""
    ntb, npad, d = p["NTB"], p["NPAD"], _SOL_HD
    qm = ws[p["qmean"]:p["qmean"] + bh * npad * d * 4].view(torch.float32).view(bh, npad, d)[:, :ntb]
    vm = ws[p["vcT"]:p["vcT"] + bh * d * npad * 2].view(torch.bfloat16).view(bh, d, npad)[:, :, :ntb]
    return qm, _ws_ksums(ws, p, bh), vm.transpose(1, 2).float() / lengths.view(1, -1, 1)


def _topk_threshold_from_workspace(ws, p, h, topk_ratio, scale, lengths, sinks):
    """Producer-path top-k threshold from the workspace's own pooled outputs. The
    HIP carriers are in channel order, so cen8 is read as stored -- the CUDA entry
    has to gather it at perm_d first."""
    ntb, d = p["NTB"], _SOL_HD
    cen8 = ws[p["cen8"]:p["cen8"] + h * ntb * d].view(torch.int8).view(h, ntb, d).float()
    cens = ws[p["cens"]:p["cens"] + h * ntb * 4].view(torch.float32).view(h, ntb)
    kc = _ws_ksums(ws, p, h) / lengths.view(1, -1, 1)
    kc = kc - kc.mean(dim=1, keepdim=True)
    return _topk_from_pooled(cen8, cens, kc, topk_ratio, scale * _LOG2E, sinks)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def _build_constraints(has_wmma: bool = True, has_valu_gemm: bool = False) -> dict:
    from comfy_kitchen.constraints import (
        DivisibleBy,
        ExactDims,
        FunctionConstraints,
        MinDims,
        ParamConstraint,
        ValidationResult,
        na3d_common_call_rule,
        sol_attn_common_call_rule,
        with_out_param,
    )

    # PyTorch exposes ROCm tensors with device type "cuda".
    dev = frozenset({"cuda", "hip"})
    floats = frozenset({torch.float32, torch.float16, torch.bfloat16})
    half_floats = frozenset({torch.float16, torch.bfloat16})
    fp8s = frozenset({torch.float8_e4m3fn, torch.float8_e5m2})
    out_floats = frozenset({torch.float32, torch.float16, torch.bfloat16})

    def _na3d_call_rule(kwargs):
        common = na3d_common_call_rule(kwargs)
        if not common.success:
            return common
        q = kwargs.get("q")
        if q is not None:
            head_dim = q.shape[-1]
            # The WMMA K-step is 16 wide and the output accumulators are held in
            # registers, one 16-column tile each.
            if head_dim % 16 != 0 or head_dim > 64:
                return ValidationResult.fail("q", "head_dim must be a multiple of 16 and <= 64")
            if q.shape[1] * q.shape[2] > 65535 or q.shape[0] * q.shape[4] > 65535:
                return ValidationResult.fail("q", "grid dims exceed HIP limits")
        return ValidationResult.ok()

    def _fp16_linear_call_rule(kwargs):
        # bf16 for x/weight is the gfx1103 iGPU's own bf16 GEMM path. It cannot be
        # decided in the envelope below: _build_constraints runs once, at
        # registration, so the device current *then* is not necessarily the device
        # these tensors live on. A process-wide _is_small_igpu() therefore rejects
        # valid gfx1103 bf16 calls (losing the fast path outright) or admits dGPU bf16
        # ones (a wasted dispatch, since fp16_linear declines those and falls back to
        # torch). Decide per call, off x's own device, which is what fp16_linear's own
        # check already does -- so a call refused here still has somewhere correct to
        # go rather than being admitted to a fallback.
        #
        # weight is not tested separately: the kernel path requires
        # weight.dtype == x.dtype, so x alone decides the pair.
        x = kwargs.get("x")
        if (isinstance(x, torch.Tensor) and x.dtype == torch.bfloat16
                and not _is_small_igpu(x.device)):
            return ValidationResult.fail(
                "x", f"bf16 needs the gfx1103 iGPU; {x.device} would fall back to torch")
        return ValidationResult.ok()

    bf16_param = ParamConstraint(dtypes=frozenset({torch.bfloat16}))
    out_param = ParamConstraint(dtypes=out_floats)
    int8_2d = ParamConstraint(
        dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)
    )
    divisible_bf16 = ParamConstraint(
        dtypes=frozenset({torch.bfloat16}), shape_rules=(DivisibleBy(-1, 16),)
    )

    def fusion(params):
        return FunctionConstraints(params=params, default_devices=dev)

    constraints = {
        "sol_attn": FunctionConstraints(
            params={
                "q": ParamConstraint(
                    dtypes=frozenset({torch.bfloat16, torch.float16}),
                    shape_rules=(ExactDims(4),),
                ),
                "k": ParamConstraint(
                    dtypes=frozenset({torch.bfloat16, torch.float16}),
                    shape_rules=(ExactDims(4),),
                ),
                "v": ParamConstraint(
                    dtypes=frozenset({torch.bfloat16, torch.float16}),
                    shape_rules=(ExactDims(4),),
                ),
            },
            default_devices=dev,
            call_rules=(sol_attn_common_call_rule,),
        ),
        # fp32 has no 16-bit matrix path, so it goes to triton/eager.
        "na3d": FunctionConstraints(
            params={
                "q": ParamConstraint(dtypes=half_floats, shape_rules=(ExactDims(6),)),
                "k": ParamConstraint(dtypes=half_floats, shape_rules=(ExactDims(6),)),
                "v": ParamConstraint(dtypes=half_floats, shape_rules=(ExactDims(6),)),
            },
            default_devices=dev,
            call_rules=(_na3d_call_rule,),
        ),
        # NVFP4. The three ops together are the whole NVFP4 story on this backend:
        # with scaled_mm_nvfp4 registered, an NVFP4 linear no longer has to
        # dequantize its weight to a float tensor first, which is what the
        # dequantize-and-matmul fallback costs (measured on the shapes in
        # qwen3vl_32b_minimax_h3_nvfp4_awq: 12.6 ms against a 0.29 ms roofline for
        # q_proj at M=1, 38 ms against 0.90 for up_proj at M=512).
        #
        # K is not constrained here beyond what the kernels demand, because the
        # shape rules cannot see K for the GEMM (it is packed two values per byte in
        # a, b) and quantize/dequantize are told cols/rows directly. The wrappers
        # and the bindings both check the divisibility they need, so a call that
        # would read past a row is refused there rather than admitted and
        # miscompiled.
        "quantize_nvfp4": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(2),)),
                "per_tensor_scale": ParamConstraint(dtypes=frozenset({torch.float32})),
            },
            default_devices=dev,
        ),
        "dequantize_nvfp4": FunctionConstraints(
            params={
                "qx": ParamConstraint(
                    dtypes=frozenset({torch.uint8}), shape_rules=(ExactDims(2),)
                ),
                "per_tensor_scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                "block_scales": ParamConstraint(dtypes=frozenset({torch.float8_e4m3fn})),
                "output_type": ParamConstraint(dtypes=out_floats),
            },
            default_devices=dev,
        ),
        "scaled_mm_nvfp4": FunctionConstraints(
            params={
                "a": ParamConstraint(dtypes=frozenset({torch.uint8}), shape_rules=(ExactDims(2),)),
                "b": ParamConstraint(dtypes=frozenset({torch.uint8}), shape_rules=(ExactDims(2),)),
                "tensor_scale_a": ParamConstraint(dtypes=frozenset({torch.float32})),
                "tensor_scale_b": ParamConstraint(dtypes=frozenset({torch.float32})),
                "block_scale_a": ParamConstraint(dtypes=frozenset({torch.float8_e4m3fn})),
                "block_scale_b": ParamConstraint(dtypes=frozenset({torch.float8_e4m3fn})),
                "out_dtype": ParamConstraint(dtypes=out_floats),
            },
            default_devices=dev,
        ),
        "quantize_per_tensor_fp8": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=floats),
                "scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                "output_type": ParamConstraint(dtypes=fp8s),
            },
            default_devices=dev,
        ),
        "dequantize_per_tensor_fp8": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=fp8s),
                "scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                "output_type": ParamConstraint(dtypes=out_floats),
            },
            default_devices=dev,
        ),
        "stochastic_rounding_fp8": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=floats),
                "rng": ParamConstraint(dtypes=frozenset({torch.uint8})),
                "output_type": ParamConstraint(dtypes=fp8s),
            },
            default_devices=dev,
        ),
        "quantize_int8_rowwise": FunctionConstraints(
            params={"x": ParamConstraint(dtypes=floats)},
            default_devices=dev,
        ),
        "quantize_int8_tensorwise": FunctionConstraints(
            params={"x": ParamConstraint(dtypes=floats)},
            default_devices=dev,
        ),
        "quantize_and_rotate_rowwise": FunctionConstraints(
            params={"x": ParamConstraint(dtypes=floats)},
            default_devices=dev,
        ),
        "quantize_int8_convrot_weight": FunctionConstraints(
            params={"weight": ParamConstraint(dtypes=floats)},
            default_devices=dev,
        ),
        "dequantize_int8_simple_dtype": FunctionConstraints(
            params={
                "q": ParamConstraint(dtypes=frozenset({torch.int8})),
                "scale": ParamConstraint(dtypes=floats),
                "output_dtype_code": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=dev,
        ),
        "dequantize_int8_convrot_weight_dtype": FunctionConstraints(
            params={
                "q": ParamConstraint(
                    dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)
                ),
                "scale": ParamConstraint(dtypes=floats),
                "group_size": ParamConstraint(dtypes=frozenset({int})),
                "output_dtype_code": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=dev,
        ),
        # K must be a multiple of 16 so a WMMA K-step never straddles a row end.
        "int8_linear": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=floats, shape_rules=(DivisibleBy(-1, 16),)),
                "weight": ParamConstraint(dtypes=frozenset({torch.int8})),
                "out_dtype": ParamConstraint(dtypes=out_floats),
            },
            default_devices=dev,
        ),
        "rms_gated_residual": fusion({
            "activation": bf16_param, "norm_weight": bf16_param,
            "residual": bf16_param, "gate": bf16_param,
            "eps": ParamConstraint(dtypes=frozenset({float})),
        }),
        "int8_linear_modulated": fusion({
            "x": divisible_bf16, "modulation_scale": bf16_param,
            "weight": int8_2d,
            "out_dtype": out_param,
        }),
        "int8_linear_rms_modulated": fusion({
            "x": divisible_bf16, "norm_weight": bf16_param,
            "modulation_scale": bf16_param, "weight": int8_2d,
            "out_dtype": out_param,
        }),
        "quantize_w4a8_int8_weight": FunctionConstraints(
            params={
                "weight": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(2),)),
                "group_size": ParamConstraint(dtypes=frozenset({int})),
                "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                "scale_dtype": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float8_e4m3fn})
                ),
            },
            default_devices=dev,
        ),
        "dequantize_w4a8_int8_weight": FunctionConstraints(
            params={
                "qdata": ParamConstraint(
                    dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)
                ),
                "s_rel": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn, torch.float32}),
                    shape_rules=(ExactDims(2),),
                ),
                "s_channel": ParamConstraint(
                    dtypes=frozenset({torch.float32}), shape_rules=(ExactDims(1),)
                ),
                "codebook": ParamConstraint(
                    dtypes=frozenset({torch.float32}), shape_rules=(ExactDims(1),)
                ),
                "correction": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(2),)),
                "group_size": ParamConstraint(dtypes=frozenset({int})),
                "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                "output_dtype": ParamConstraint(dtypes=out_floats),
            },
            default_devices=dev,
        ),
        # K must be a multiple of 16 for the same reason as int8_linear, and so
        # that the INT4 decode's 16-column store stays inside its own row.
        "w4a8_int8_linear": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=floats, shape_rules=(DivisibleBy(-1, 16),)),
                "qdata": ParamConstraint(
                    dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)
                ),
                "s_rel": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn, torch.float32}),
                    shape_rules=(ExactDims(2),),
                ),
                "s_channel": ParamConstraint(
                    dtypes=frozenset({torch.float32}), shape_rules=(ExactDims(1),)
                ),
                "codebook": ParamConstraint(
                    dtypes=frozenset({torch.float32}), shape_rules=(ExactDims(1),)
                ),
                "correction": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(2),)),
                "bias": ParamConstraint(dtypes=floats),
                "group_size": ParamConstraint(dtypes=frozenset({int})),
                "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                "out_dtype": ParamConstraint(dtypes=out_floats),
            },
            default_devices=dev,
        ),
        "quantize_convrot_w4a4_weight": FunctionConstraints(
            params={"weight": ParamConstraint(dtypes=floats, shape_rules=(DivisibleBy(-1, 32),))},
            default_devices=dev,
        ),
        "dequantize_convrot_w4a4_weight": FunctionConstraints(
            params={"qdata": ParamConstraint(dtypes=frozenset({torch.int8}))},
            default_devices=dev,
        ),
        "convrot_w4a4_linear": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=floats, shape_rules=(DivisibleBy(-1, 32),)),
                "qweight": ParamConstraint(dtypes=frozenset({torch.int8})),
            },
            default_devices=dev,
        ),
        # 2D only: the tile-packed weight/scale variants have no HIP kernel and
        # fall through to eager.
        "gemv_awq_w4a16": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=floats),
                "qweight": ParamConstraint(
                    dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)
                ),
            },
            default_devices=dev,
        ),
        "quantize_svdquant_w4a4": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=floats, shape_rules=(ExactDims(2), DivisibleBy(-1, 64))
                ),
                "smooth": ParamConstraint(dtypes=floats),
                "lora_down": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(2),)),
            },
            default_devices=dev,
        ),
        "scaled_mm_svdquant_w4a4": FunctionConstraints(
            params={
                "act": ParamConstraint(
                    dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)
                ),
                "wgt": ParamConstraint(
                    dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)
                ),
                "wscales": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(2),)),
                "lora_up": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(2),)),
            },
            default_devices=dev,
        ),
        "adaln": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=floats),
                "scale": ParamConstraint(dtypes=floats),
                "shift": ParamConstraint(dtypes=floats),
            },
            default_devices=dev,
        ),
        "rms_adaln": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=floats),
                "scale": ParamConstraint(dtypes=floats),
                "shift": ParamConstraint(dtypes=floats),
            },
            default_devices=dev,
        ),
        "fp16_linear": FunctionConstraints(
            params={
                # bf16 is admitted in the envelope for x and weight only, and
                # _fp16_linear_call_rule decides it per call from x's device. The
                # auxiliary operands stay FP16 because the launcher requires it and
                # the fast path enforces it.
                "x": ParamConstraint(dtypes=half_floats, shape_rules=(MinDims(2),)),
                "weight": ParamConstraint(dtypes=half_floats, shape_rules=(ExactDims(2),)),
                "bias": ParamConstraint(dtypes=frozenset({torch.float16})),
                "residual": ParamConstraint(dtypes=frozenset({torch.float16})),
                "residual_scale": ParamConstraint(dtypes=frozenset({torch.float16})),
            },
            default_devices=dev,
            call_rules=(_fp16_linear_call_rule,),
        ),
        "fp16_conv3d": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=frozenset({torch.float16}), shape_rules=(ExactDims(5),)),
                "weight": ParamConstraint(
                    dtypes=frozenset({torch.float16}), shape_rules=(ExactDims(5),)
                ),
                "bias": ParamConstraint(dtypes=frozenset({torch.float16, type(None)})),
                "residual": ParamConstraint(dtypes=frozenset({torch.float16, type(None)})),
            },
            default_devices=dev,
        ),
        "group_norm_stats3d": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.float16, torch.bfloat16}),
                    shape_rules=(ExactDims(5),),
                ),
            },
            default_devices=dev,
        ),
        "group_norm_silu_pad3d": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.float16, torch.bfloat16}),
                    shape_rules=(ExactDims(5),),
                ),
                "weight": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16, type(None)}),
                ),
                "bias": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16, type(None)}),
                ),
            },
            default_devices=dev,
        ),
        # The rope kernel indexes x as 4D and freqs_cis as 6D.
        "apply_rope": FunctionConstraints(
            params={
                "xq": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(4),)),
                "xk": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(4),)),
                "freqs_cis": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(6),)),
            },
            default_devices=dev,
        ),
        "apply_rope1": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(4),)),
                "freqs_cis": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(6),)),
            },
            default_devices=dev,
        ),
        "apply_rope_split_half": FunctionConstraints(
            params={
                "xq": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(4),)),
                "xk": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(4),)),
                "freqs_cis": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(6),)),
            },
            default_devices=dev,
        ),
        "apply_rope_split_half1": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(4),)),
                "freqs_cis": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(6),)),
            },
            default_devices=dev,
        ),
        # Normalized over the last axis, so BHND and BNHD both work. A weight that
        # is not one entry per head_dim falls through to eager.
        "rms_rope": FunctionConstraints(
            params={
                "q": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(4),)),
                "k": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(4),)),
                "freqs_cis": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(6),)),
                "q_scale": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(1),)),
                "k_scale": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(1),)),
            },
            default_devices=dev,
        ),
        "rms_rope1": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(4),)),
                "freqs_cis": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(6),)),
                "scale": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(1),)),
            },
            default_devices=dev,
        ),
        "rms_rope_split_half": FunctionConstraints(
            params={
                "q": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(4),)),
                "k": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(4),)),
                "freqs_cis": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(6),)),
                "q_scale": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(1),)),
                "k_scale": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(1),)),
            },
            default_devices=dev,
        ),
        "rms_rope_split_half1": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(4),)),
                "freqs_cis": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(6),)),
                "scale": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(1),)),
            },
            default_devices=dev,
        ),
    }

    # Same operands and the same kernel as the functional siblings.
    for inplace_name, functional_name in {
        "apply_rope_": "apply_rope",
        "apply_rope1_": "apply_rope1",
        "apply_rope_split_half_": "apply_rope_split_half",
        "apply_rope_split_half1_": "apply_rope_split_half1",
        "rms_rope_": "rms_rope",
        "rms_rope1_": "rms_rope1",
        "rms_rope_split_half_": "rms_rope_split_half",
        "rms_rope_split_half1_": "rms_rope_split_half1",
    }.items():
        constraints[inplace_name] = constraints[functional_name]

    if not has_wmma:
        # RDNA2: the WMMA GEMM kernels are compiled but trap, so they must not be
        # advertised. Dropping the ops in _WMMA_ONLY_OPS here routes those to
        # triton/eager while the elementwise kernels below, and the GEMMs that have
        # a VALU path (int8_linear, fp16_linear, w4a8_int8_linear), still dispatch
        # to HIP.
        #
        # _VALU_GEMM_OPS is put back out of the pre-drop dict rather than out of the
        # filtered one, which is empty of them by construction -- reading it back out
        # of `constraints` here silently does nothing and would leave the gate closed
        # with nothing to show for it. This is the same shape w4a8_int8_linear and
        # convrot_w4a4_linear were removed for: the launcher behind
        # scaled_mm_nvfp4 branches on comfy_is_gfx10() exactly as the other RDNA2 VALU
        # paths do, so the same call reaches the WMMA tile on gfx11 and the GEMV or
        # the register tile on gfx103x with nothing else having to know which is which.
        valu_gemm = {k: v for k, v in constraints.items() if k in _VALU_GEMM_OPS}
        constraints = {k: v for k, v in constraints.items() if k not in _WMMA_ONLY_OPS}
        if has_valu_gemm:
            constraints.update(valu_gemm)

    for name in ("fp16_conv3d", "group_norm_silu_pad3d"):
        if name in constraints:
            constraints[name + "_out"] = with_out_param(constraints[name])
    return constraints


# ---------------------------------------------------------------------------
# INT8 attention
#
# The public entry points live in comfy_kitchen/sage_attention.py, which
# validates and pads the head dimension; everything below owns the packed
# layouts, which are not the same as the CUDA backend. See
# sage_attention/quant_qk_int8.hip for why the scale granularity differs.
# ---------------------------------------------------------------------------

# Must match kCtaQ and the key tiles in sage_attention/int8_attn.hip.
_SAGE_CTA_Q = 128
_SAGE_CTA_K = 64
_SAGE_LARGE_CTA_K = 128
_SAGE_KEY_GROUP = 16
_SAGE_HEAD_DIMS = (64, 128, 256)


def _sage_cta_k(head_dim: int, kv_length: int, has_mask: bool,
                device: torch.device | int | None = None) -> int:
    """Keys per attention iteration: 64. 128 is implemented, measured, and not used.

    The kernel used to discard this outright (``(void)cta_k;`` with the launch
    macro hardcoding 64), so the ~15% figure and the later 0.99x-1.01x A/B that
    retired the wide tile were both reading an identical launch -- there was
    nothing for the value to select. The argument is honoured now, and with it a
    real choice: ``COMFY_SAGE_CTA_K=128`` picks the wide tile. That override is
    **gfx1103-only**, which is not the same as "off by default": before the
    argument was honoured the 128-key tile existed nowhere, so a global override
    would hand every other architecture a schedule that has never run on it.

    Measured, that choice loses. Interleaved in one process at H=48/D=128/no
    mask, one sequence length per process: 0.997x at S=512, 1.028x at 1024,
    1.001x at 2048, 0.986x at 3072, 0.987x at 5120, 1.003x at 8192, and at the
    three sequence lengths Krea2 ships -- 1.009x (S=4138), 1.027x (6186),
    1.038x (9258). Never a win, and consistently worse where this checkpoint
    lives, by more as S grows. So 64 stays the default and this function exists
    only to keep the two sides of the contract in step.

    It stays a function because two things depend on it agreeing with the kernel:
    the V scratch is padded to a multiple of it, and the k-scale buffer is sized
    ``padded_k // _SAGE_KEY_GROUP``. A 128 here at D256 would pad buffers for a
    tile the kernel never launches -- it only instantiates the wide tile up to
    D128, 69,632 B of LDS against a 64 KB limit -- so that falls back to 64.

    The kernel additionally ignores a 128 request when a mask is present: the
    mask staging is sized for the 64-key schedule and reading it back at
    CTA_K=128 walks off the end (7.2e-01 worst relative error against a float
    softmax reference, where the 64-key tile gives 1.3e-02). Krea2's DiT is
    unmasked and this only concerns the override, but the two sides have to agree
    on *when* 128 applies as well as *whether*, or the buffer padding and the
    launch disagree.
    """
    # Module-level `os`, not a function-local import: this runs on every attention
    # call, which is the hottest path in the quantised attention stack.
    #
    # **gfx1103 only**, which is a different claim from "off by default". The
    # override exists because the 128-key tile was measured here; before this it
    # was not instantiated on *any* target, so honouring the value unconditionally
    # would hand every other architecture a schedule nothing has ever run on it --
    # an env var should not be able to do that. comfy_small_igpu() resolves for
    # the device the tensors live on, so a mixed iGPU + dGPU box answers per
    # device rather than for whichever was current.
    if ((not has_mask) and os.environ.get("COMFY_SAGE_CTA_K") == "128"
            and head_dim <= 128 and _is_small_igpu(device)):
        return _SAGE_LARGE_CTA_K
    return _SAGE_CTA_K


def int8_attention_is_available() -> bool:
    """Whether the INT8 attention kernels can run here.

    Stricter than is_available(): the kernel is built on the matrix cores, so
    RDNA2 declines it the way it declines the GEMMs.
    """
    return has_wmma()


def flash_attention_decode_is_available() -> bool:
    """Whether the BF16 decode attention kernel can run here.

    The kernel uses no matrix cores, so is_available() would be the natural
    gate, but RDNA2 has no bf16 and a caller that asks it to run there builds
    the KV cache in another dtype, which this kernel declines. has_wmma() marks
    the line where bf16 arrives, the same line the CUDA backend draws at SM80.
    The attribute test catches an extension built before the kernel existed,
    which is otherwise an AttributeError at dispatch.
    """
    return has_wmma() and hasattr(_C, "flash_attention_decode")


def flash_attention_decode_gqa_is_available() -> bool:
    """Whether the head-first GQA decode and tree-merge kernels can run here."""
    return has_wmma() and hasattr(_C, "flash_attention_decode_gqa")


# ---------------------------------------------------------------------------
# GatedDeltaNet decode
#
# The public entry points live in comfy_kitchen/gated_delta.py, which owns the
# reshaping; everything below is the raw launch. See ops/gated_delta.hip for why
# the recurrent state lives in registers here rather than in LDS.
# ---------------------------------------------------------------------------

# Matches kDeltaMaxSteps, kDeltaKeyDim and kDeltaConvMaxWindow in
# ops/gated_delta.hip.
_DELTA_MAX_STEPS = 8
_DELTA_KEY_DIM = 128
_DELTA_CONV_MAX_WINDOW = 16


def gated_delta_decode_is_available(key_head_dim: int = 128, value_head_dim: int = 128) -> bool:
    """Whether the fused DeltaNet decode kernels can run here for these head dims.

    The kernels use no matrix cores, but they do use bf16 and fp16 arithmetic
    throughout, so they draw the line where has_wmma() does, as the decode
    attention kernel next door does.
    """
    if not has_wmma() or not hasattr(_C, "gated_delta_decode_fused"):
        return False
    if key_head_dim != _DELTA_KEY_DIM or value_head_dim <= 0 or value_head_dim > 512:
        return False
    # One wave covers 32 value columns; a partial wave would leave the partition
    # reduction reading columns nobody wrote.
    return value_head_dim % 32 == 0


def gated_delta_decode_fused(
    mixed_qkv: torch.Tensor,
    x: torch.Tensor,
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    dt_bias: torch.Tensor,
    g_decay: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    snapshots: torch.Tensor | None,
    z: torch.Tensor,
    norm_w: torch.Tensor,
    eps: float,
    key_dim: int,
    num_key_heads: int,
    scale: float,
) -> bool:
    """S decode steps into ``out``; ``state`` is updated in place.

    Returns False when the shape is outside the kernel's envelope, leaving both
    outputs untouched, so the caller can fall back rather than predict it.
    """
    batch, channels, seq = mixed_qkv.shape
    heads, key_head_dim, value_head_dim = state.shape[1], state.shape[2], state.shape[3]
    return _C.gated_delta_decode_fused(
        _dl(mixed_qkv),
        _dl(x),
        _dl(w_a),
        _dl(w_b),
        _dl(dt_bias),
        _dl(g_decay),
        _dl(state),
        _dl(out),
        None if snapshots is None else _dl(snapshots),
        _dl(z),
        _dl(norm_w),
        float(eps),
        batch,
        heads,
        num_key_heads,
        seq,
        key_head_dim,
        value_head_dim,
        channels,
        x.shape[2],
        key_dim,
        float(scale),
        _stream(mixed_qkv),
    )


def deltanet_conv_step(
    proj: torch.Tensor,
    conv_state: torch.Tensor,
    conv_w: torch.Tensor,
    conv_b: torch.Tensor | None,
    conv_out: torch.Tensor,
    snapshots: torch.Tensor | None,
) -> bool:
    """Depthwise causal conv step with silu into ``conv_out``.

    ``conv_state`` is updated in place. Returns False for a shape outside the
    kernel's envelope, leaving both outputs untouched.
    """
    batch, seq, channels = proj.shape
    return _C.deltanet_conv_step(
        _dl(proj),
        _dl(conv_state),
        _dl(conv_w),
        None if conv_b is None else _dl(conv_b),
        _dl(conv_out),
        None if snapshots is None else _dl(snapshots),
        batch,
        channels,
        seq,
        conv_w.shape[-1],
        _stream(proj),
    )


def gated_delta_deferred_is_available(key_head_dim: int = 128, value_head_dim: int = 128) -> bool:
    """Whether the deferred-commit DeltaNet decode kernels can run here.

    The deferred kernel keeps the whole [DK, DV] state in one 1024-thread block
    the way the snapshot kernel does, so it is built for DK = DV = 128 only.
    """
    if not has_wmma() or not hasattr(_C, "gated_delta_decode_deferred"):
        return False
    return key_head_dim == _DELTA_KEY_DIM and value_head_dim == _DELTA_KEY_DIM


def deltanet_conv_deferred(
    proj: torch.Tensor,
    proj_buf: torch.Tensor,
    conv_state: torch.Tensor,
    conv_w: torch.Tensor,
    conv_b: torch.Tensor | None,
    qkv_buf: torch.Tensor,
    ctl: torch.Tensor,
) -> bool:
    """Deferred-commit conv step: see comfy_kitchen.gated_delta.deltanet_conv_step_deferred.

    The kernel reads packed layouts, so ``proj`` must be contiguous. Returns False
    for a shape outside the kernel's envelope, leaving every buffer untouched.
    """
    return _C.deltanet_conv_deferred(
        _dl(proj),
        _dl(proj_buf),
        _dl(conv_state),
        _dl(conv_w),
        None if conv_b is None else _dl(conv_b),
        _dl(qkv_buf),
        _dl(ctl),
        _stream(proj),
    )


def gated_delta_decode_deferred(
    x: torch.Tensor,
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    dt_bias: torch.Tensor,
    g_decay: torch.Tensor,
    qkv_buf: torch.Tensor,
    gates_buf: torch.Tensor,
    sumsq_buf: torch.Tensor,
    ctl: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    z: torch.Tensor,
    norm_w: torch.Tensor,
    eps: float,
    key_dim: int,
    num_key_heads: int,
    scale: float,
) -> bool:
    """Deferred-commit decode: see comfy_kitchen.gated_delta.gated_delta_decode_deferred.

    ``z`` is read packed as [B, S, Hv*DV], so a strided gate view is copied here.
    Returns False for a shape outside the kernel's envelope, leaving ``state``,
    ``out`` and the side buffers untouched.
    """
    return _C.gated_delta_decode_deferred(
        _dl(x),
        _dl(w_a),
        _dl(w_b),
        _dl(dt_bias),
        _dl(g_decay),
        _dl(qkv_buf),
        _dl(gates_buf),
        _dl(sumsq_buf),
        _dl(ctl),
        _dl(state),
        _dl(out),
        _dl(z.contiguous()),
        _dl(norm_w),
        float(eps),
        key_dim,
        num_key_heads,
        float(scale),
        _stream(x),
    )


def flash_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_lengths: torch.Tensor,
    output: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_lse_accum: torch.Tensor,
    output_accum: torch.Tensor,
    num_splits: int,
) -> None:
    """Decode attention into ``output``. See ops/flash_decode.hip.

    The caller owns the reshaping and the split workspace; this is the raw
    launch so both backends can share comfy_kitchen/flash_attention.py.
    """
    _C.flash_attention_decode(
        _dl(q),
        _dl(k),
        _dl(v),
        _dl(kv_lengths),
        _dl(output),
        _dl(softmax_lse),
        _dl(softmax_lse_accum),
        _dl(output_accum),
        num_splits,
        _stream(q),
    )


def flash_decode_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_lengths: torch.Tensor,
    output: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_lse_accum: torch.Tensor,
    output_accum: torch.Tensor,
    num_splits: int,
    causal: bool,
) -> None:
    """Head-first GQA decode attention into ``output``. See ops/flash_decode.hip."""
    _C.flash_attention_decode_gqa(
        _dl(q),
        _dl(k),
        _dl(v),
        _dl(kv_lengths),
        _dl(output),
        _dl(softmax_lse),
        _dl(softmax_lse_accum),
        _dl(output_accum),
        num_splits,
        causal,
        _stream(q),
    )


def flash_decode_tree_merge(
    out: torch.Tensor,
    lse: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    merged: torch.Tensor,
) -> None:
    """Fold a verify tree's own rows into a prefix-only decode result. See ops/flash_decode.hip."""
    _C.flash_attention_decode_tree_merge(
        _dl(out), _dl(lse), _dl(q), _dl(k), _dl(v), _dl(mask), _dl(merged), _stream(q))


def _sage_buffers(q: torch.Tensor, k: torch.Tensor, cta_k: int):
    batch, q_heads, q_length, head_dim = q.shape
    _, kv_heads, kv_length, _ = k.shape
    if head_dim not in _SAGE_HEAD_DIMS:
        raise ValueError(f"int8 attention head_dim must be one of {_SAGE_HEAD_DIMS}")

    padded_q = -(-q_length // _SAGE_CTA_Q) * _SAGE_CTA_Q
    padded_k = -(-kv_length // cta_k) * cta_k
    device = q.device

    buffers = {
        "q_int8": torch.empty(q.shape, dtype=torch.int8, device=device),
        "k_int8": torch.empty(k.shape, dtype=torch.int8, device=device),
        # One scale per query row and one per 16 adjacent keys; the padding exists
        # so the kernel can read a scale for every lane of its last tile.
        "q_scale": torch.empty(batch, q_heads, padded_q, dtype=torch.float32, device=device),
        "k_scale": torch.empty(
            batch, kv_heads, padded_k // _SAGE_KEY_GROUP, dtype=torch.float32, device=device
        ),
        # V is stored transposed, [B * H * D, padded_K], with the tail zero
        # filled (pure-int8 path). On the gfx1103 iGPU the buffer is twice the
        # int8 width: the int8 kernels read the first padded_K columns (int8
        # stride), while the short-key direct path writes the fp16 transposed V
        # over the full 2*padded_K byte width. dGPUs allocate the upstream
        # single-width buffer (no direct path there).
        "v_int8": torch.empty(
            batch * kv_heads * head_dim,
            padded_k * (2 if _is_small_igpu(q.device) else 1),
            dtype=torch.int8,
            device=device,
        ),
        "v_scale": torch.empty(batch * kv_heads * head_dim, dtype=torch.float32, device=device),
    }

    # One key index per batch and KV head, written by the stabilization detector
    # and read by the K quantizer. It is scratch, not part of the packed form, so
    # it stays out of the buffers the split API hands back.
    anchor_indices = torch.empty(batch, kv_heads, dtype=torch.int32, device=device)
    return buffers, anchor_indices


def _sage_buffers_direct(q: torch.Tensor, k: torch.Tensor, cta_k: int):
    """Allocate only the V scratch the short-key direct path reads.

    sage_sdpa's use_direct branch runs launch_sage_transpose_v into v_int8 and
    then launch_sage_direct_attn straight from q, k and v: the int8 prepass never
    runs, so q_int8, k_int8, the three scales and anchor_indices are written by
    nothing and require_len only looks at v_int8. The binding still demands a
    tensor per argument, so the six unfilled slots share one 16-byte stub instead
    of six real allocations (six torch.empty calls per attention).
    """
    batch, _, _, head_dim = q.shape
    _, kv_heads, kv_length, _ = k.shape
    if head_dim not in _SAGE_HEAD_DIMS:
        raise ValueError(f"int8 attention head_dim must be one of {_SAGE_HEAD_DIMS}")

    padded_k = -(-kv_length // cta_k) * cta_k
    device = q.device
    stub = torch.empty(16, dtype=torch.int8, device=device)
    buffers = {
        "q_int8": stub,
        "k_int8": stub,
        "q_scale": stub,
        "k_scale": stub,
        "v_scale": stub,
        # Always the doubled width: only the iGPU has a direct path (sage_int8_sdpa
        # gates on _is_small_igpu before choosing this allocator), and its C++
        # branch requires 2 * padded_k bytes for the fp16 transposed V.
        "v_int8": torch.empty(
            batch * kv_heads * head_dim,
            padded_k * 2,
            dtype=torch.int8,
            device=device,
        ),
    }
    anchor_indices = torch.empty(16, dtype=torch.int32, device=device)
    return buffers, anchor_indices


def _sage_can_fuse_dense_mask(attn_mask: torch.Tensor | None) -> bool:
    return (
        attn_mask is not None
        and attn_mask.ndim == 4
        and attn_mask.dtype in (torch.float16, torch.bfloat16)
        and 1 < attn_mask.shape[2] <= 256
        and 64 < attn_mask.shape[3] <= 2048
        and attn_mask.stride(2) != 0
    )


def _sage_dense_mask_buffer(attn_mask: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        (
            1 if attn_mask.stride(0) == 0 else attn_mask.shape[0],
            1 if attn_mask.stride(1) == 0 else attn_mask.shape[1],
            (attn_mask.shape[2] + 15) // 16,
            (attn_mask.shape[3] + 63) // 64,
            32 if attn_mask.dtype == torch.bool else 1024,
        ),
        dtype=torch.int32 if attn_mask.dtype == torch.bool else attn_mask.dtype,
        device=attn_mask.device,
    )


def _sage_use_direct(
    q: torch.Tensor, k: torch.Tensor, attn_mask: torch.Tensor | None
) -> bool:
    """Whether sage_int8_sdpa takes the iGPU direct-V branch.

    Mirrors sage_sdpa's use_direct in dlpack_bindings.cpp exactly; keep the two in
    step. The C++ decides which kernel runs, so the Python gate must agree or the
    buffers will not match: qo=1024 kv=4096 would allocate the direct stubs while
    the int8 kernel demands full q_int8/k_int8 and throw.

    Its own function rather than an inline expression in sage_int8_sdpa because
    the gate is a contract with two implementations on either side of it, and
    test_use_direct_gate_matches_the_direct_buffer_contract has to be able to ask
    this one. A test that re-derives the rule compares two copies of it and
    passes when only one of them drifts, which is exactly the drift it exists to
    catch.
    """
    _, _, q_length, head_dim = q.shape
    _, _, kv_length, _ = k.shape
    output_dtype = torch.bfloat16 if q.dtype == torch.float32 else q.dtype
    d64_short_keys = head_dim == 64 and kv_length <= 2048 and not (
        q_length == kv_length and kv_length <= 1024
    )
    return (
        _is_small_igpu(q.device)
        and attn_mask is None
        and q.dtype == output_dtype
        and (d64_short_keys or (head_dim == 128 and kv_length <= 256))
    )


def sage_int8_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attention_scale: float,
    attn_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Quantize and attend in one call. q, k and v are already padded to a
    supported head dimension; the output keeps that width.

    Operands are [B, H, L, D], the kernels' native packing.
    """
    batch, q_heads, q_length, head_dim = q.shape
    _, _, kv_length, _ = k.shape
    output_dtype = torch.bfloat16 if q.dtype == torch.float32 else q.dtype
    cta_k = _sage_cta_k(head_dim, kv_length, attn_mask is not None, q.device)
    use_direct = _sage_use_direct(q, k, attn_mask)

    # ComfyUI's attention_comfy_kitchen_int8 finishes with
    # out.transpose(1, 2).reshape(b, -1, heads * dim_head). HND-packed storage
    # makes that transpose non-contiguous, so the reshape materialises a copy of
    # the whole output: 0.65 ms at SDXL cross-attention 4096x77x10x64, more than
    # the attention itself. NHD-packed storage presented to the kernels as
    # [B, H, S, D] makes the transpose contiguous and the reshape a free view.
    # Both the direct kernel (explicit o strides) and sage_attend (which reads
    # o.stride(0..2) off the tensor) accept the packing.
    #
    # gfx1103 only (fork principle 1): the 0.65 ms figure is a 780M measurement, and
    # the change is not a pure win elsewhere -- it alters the contiguity of the
    # tensor returned for layout="HND", which dGPU callers may rely on. Upstream's
    # HND packing is what every other device keeps.
    if _is_small_igpu(q.device):
        out_nhd = torch.empty(
            batch, q_length, q_heads, head_dim, dtype=output_dtype, device=q.device
        )
        output = out_nhd.as_strided(
            (batch, q_heads, q_length, head_dim),
            (q_length * q_heads * head_dim, head_dim, q_heads * head_dim, 1),
        )
    else:
        output = torch.empty(
            batch, q_heads, q_length, head_dim, dtype=output_dtype, device=q.device
        )

    buffers, anchor_indices = (
        _sage_buffers_direct(q, k, cta_k) if use_direct else _sage_buffers(q, k, cta_k)
    )
    raw_dense_mask = None
    if head_dim <= 128 and _sage_can_fuse_dense_mask(attn_mask):
        raw_dense_mask = attn_mask
        attn_mask = _sage_dense_mask_buffer(raw_dense_mask)
    _C.sage_sdpa(
        _dl(q),
        _dl(k),
        _dl(v),
        _dl(output),
        _dl(buffers["q_int8"]),
        _dl(buffers["q_scale"]),
        _dl(buffers["k_int8"]),
        _dl(buffers["k_scale"]),
        _dl(buffers["v_int8"]),
        _dl(buffers["v_scale"]),
        _dl(anchor_indices),
        float(attention_scale),
        cta_k,
        DTYPE_TO_CODE[q.dtype],
        DTYPE_TO_CODE[output_dtype],
        _stream(q),
        None if attn_mask is None else _dl(attn_mask),
        None if raw_dense_mask is None else _dl(raw_dense_mask),
    )
    # The NHD-packed branch (gfx1103 only) already stores the output so that
    # ComfyUI's out.transpose(1, 2) lands contiguous; the HND branch is packed
    # [B, H, S, D], whose transpose is the correct non-contiguous view of the same
    # bytes. Either way the caller gets the bytes and decides the view, so the
    # layout='HND' contract returns the [B, H, S, D] tensor unchanged.
    return output


def sage_int8_quantize(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cta_k: int | None = None,
) -> dict:
    """Quantize q, k and v without allocating the attention output.

    **``cta_k`` here and in :func:`sage_int8_attend` must be the same number.**
    This half pads the k-scale buffer and the V scratch to a multiple of it; the
    attend half tells the kernel the tile width and the kernel then reads a scale
    for every lane of its last tile. A buffer sized for 64 read by a 128-wide
    kernel walks off the end -- four floats at Krea2's 1024x1536 sequence. That
    was harmless while the kernel discarded the argument outright; it is not now
    that the tile is really selected.

    So both halves default to ``None``: this one asks :func:`_sage_cta_k`, and
    :func:`sage_int8_attend` reads the width back out of the buffer this one built.
    A caller passing two *different* explicit values is the remaining hazard, so
    that is what the note above is for. The returned dict stays a dict of tensors;
    it is the prequantised-buffer contract and must not gain a metadata key.
    """
    batch, _, _, head_dim = q.shape
    if cta_k is None:
        cta_k = _sage_cta_k(head_dim, k.shape[2], False, q.device)
    buffers, anchor_indices = _sage_buffers(q, k, cta_k)
    _C.sage_sdpa_quantize(
        _dl(q),
        _dl(k),
        _dl(v),
        _dl(buffers["q_int8"]),
        _dl(buffers["q_scale"]),
        _dl(buffers["k_int8"]),
        _dl(buffers["k_scale"]),
        _dl(buffers["v_int8"]),
        _dl(buffers["v_scale"]),
        _dl(anchor_indices),
        cta_k,
        DTYPE_TO_CODE[q.dtype],
        _stream(q),
    )
    # _sage_buffers allocates the V scratch [B*H_kv*D, 2*padded_k] on the iGPU so
    # _C.sage_sdpa's direct fp16 transpose has the doubled width. The V quantizer
    # packs rows at padded_k stride, so the first B*H_kv*D*padded_k elements are
    # the contiguous single-width int8 V that sage_attend reads with a padded_k
    # stride. Hand that single-width view to the packed form: the prequantized
    # contract is a padded_k-wide row, and presenting the wider scratch would make
    # the attention kernel read the second half of one row as the next.
    kv_heads = k.shape[1]
    padded_k = -(-k.shape[2] // cta_k) * cta_k
    v_rows = batch * kv_heads * head_dim
    v_need = v_rows * padded_k
    buffers["v_int8"] = (
        buffers["v_int8"].reshape(-1)[:v_need].reshape(v_rows, padded_k)
        if buffers["v_int8"].numel() > v_need
        else buffers["v_int8"]
    )
    return buffers


def sage_int8_attend(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    v_int8: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    attention_scale: float,
    attn_mask: torch.Tensor | None,
    output_dtype: torch.dtype,
    cta_k: int | None = None,
) -> torch.Tensor:
    """Attend over the packed layouts sage_int8_quantize produced.

    ``cta_k`` must equal what the quantise half used. Left at ``None`` it is read
    back out of ``v_int8``'s row width rather than asked a second time, because the
    two halves cannot always be asked consistently: the quantise half runs before
    the caller has a mask, so under ``COMFY_SAGE_CTA_K=128`` it pads for 128 while
    a masked attend wants 64 -- and the C++ side checks the row against
    ``sage_padded_k(kv_len, cta_k)`` by equality, so guessing wrong there is a hard
    error rather than a slow path. Deriving it makes the buffer the single source
    of truth and every combination agree.

    **The narrow tile is tried first, and that order is load-bearing.** When
    ``kv_len`` is a multiple of 128 the two candidates pad to the *same* width, so
    the row cannot say which one built it; asking 128 first would answer 128
    always, and the 128-key schedule -- which is opt-in, and is not even
    instantiated on an architecture that never asked for it -- would become the
    default everywhere for those sequence lengths. Narrow-first pins the answer to
    ``_SAGE_CTA_K`` whenever the two are indistinguishable, which is the upstream
    behaviour. The cost is that the override is then a no-op for
    ``kv_len % 128 == 0``; the buffers are byte-identical there, so nothing is lost
    but the experiment.
    """
    batch, q_heads, q_length, head_dim = q_int8.shape
    if cta_k is None:
        kv_len = k_int8.shape[2]
        row = v_int8.shape[-1]
        cta_k = _SAGE_CTA_K
        for candidate in (_SAGE_CTA_K, _SAGE_LARGE_CTA_K):
            if row == -(-kv_len // candidate) * candidate:
                cta_k = candidate
                break
    output = torch.empty(
        batch, q_heads, q_length, head_dim, dtype=output_dtype, device=q_int8.device
    )
    _C.sage_sdpa_prequantized(
        _dl(q_int8),
        _dl(k_int8),
        _dl(v_int8),
        _dl(output),
        _dl(q_scale),
        _dl(k_scale),
        _dl(v_scale),
        cta_k,
        float(attention_scale),
        DTYPE_TO_CODE[output_dtype],
        _stream(q_int8),
        None if attn_mask is None else _dl(attn_mask),
    )
    return output


def _register():
    from comfy_kitchen.registry import registry

    # COMFY_KITCHEN_DISABLE_HIP=1 removes the backend from dispatch, leaving
    # triton/eager to handle every op.
    if os.getenv("COMFY_KITCHEN_DISABLE_HIP") == "1":
        registry.mark_unavailable("hip", "disabled by COMFY_KITCHEN_DISABLE_HIP=1")
        return

    if not _EXT_AVAILABLE:
        registry.mark_unavailable("hip", _EXT_ERROR or "HIP extension not built")
        return

    if not getattr(torch.version, "hip", None):
        registry.mark_unavailable("hip", "PyTorch ROCm/HIP runtime not available")
        return

    arches = _visible_gfx_arches()
    reason = _unsupported_arch_reason(arches)
    if reason is not None:
        registry.mark_unavailable("hip", reason)
        return

    has_wmma = _has_wmma(arches)
    has_valu_gemm = _has_valu_gemm(arches)
    registry.register(
        name="hip",
        module=sys.modules[__name__],
        capabilities=_build_constraints(has_wmma, has_valu_gemm),
    )
    logger.debug(
        "registered HIP backend for %s (%s)",
        ", ".join(sorted({a for a in arches if a})),
        "with WMMA"
        if has_wmma
        else ("no matrix cores, VALU GEMMs" if has_valu_gemm else "elementwise only"),
    )


# ---------------------------------------------------------------------------
# RDNA2 (gfx103x) SageAttention: INT8 QK^T + FP16 P*V on VALU
# ---------------------------------------------------------------------------
# RDNA2 has no matrix cores, so the WMMA sage_attention/*.hip sources above neither
# build nor run for gfx103x. These are the ported RDNA2 kernels, exposing the same
# call shape as the external SageAttention package so callers can treat them as a
# drop-in backend. Kernels in sage_attention/sage_attn_gfx103x.hip, tile in
# mma_gfx103x.h, entry points on _C as gfx103x_quant_qk_int8 and
# gfx103x_qk_int8_sv_attn.
#
# The gate in rdna2_is_available is deliberately architecture-strict, and it covers
# the whole RDNA2 family (gfx1030-1036), not just the gfx1035 this path was named
# after: every gfx103x part lacks matrix cores, so every one of them needs the port.
# Outside __GFX10__ the tile body compiles to nothing, so the entry points exist on
# every build and do nothing; this gate is what keeps them off a non-RDNA2 device.
# Supported head dimensions for the ported kernels.
_RDNA2_SUPPORTED_HEAD_DIMS = (64, 128)

_rdna2_module = None
_rdna2_import_error: Exception | None = None
_rdna2_arch_loaded: str | None = None


def _rdna2_arch(device: torch.device | int | None = None) -> str | None:
    """Return the RDNA2 (gfx103x) architecture name of ``device``, or None.

    Covers the whole RDNA2 family, not just gfx1035: gfx1030-1036 all lack matrix
    cores, so all of them need this port and all of them are accepted by the
    extension's CMake target. The historical ``gfx1035`` name lives on in the
    directory, module and build-suffix for compatibility, but the *gate* is
    architecture-class, not a single part.
    """
    arch = _gfx_arch(device)
    return arch if isinstance(arch, str) and arch.startswith("gfx103") else None


def _load_rdna2_module() -> None:
    """Import the compiled extension; caches success and the first failure."""
    global _rdna2_module, _rdna2_import_error, _rdna2_arch_loaded
    if _rdna2_module is not None or _rdna2_import_error is not None:
        return

    # The entry points live on _C with every other kernel in this backend, so there
    # is nothing to locate: one extension, built for whichever architectures
    # COMFY_HIP_ARCHS named, and registered whether or not this device can use it.
    #
    # Deliberately no architecture test of the current device here. Which device can
    # use it is already the per-device gate in rdna2_is_available, and this loader
    # caches whatever it decides for the rest of the process: on a box holding both
    # an RDNA2 iGPU and a dGPU, a first call made while the dGPU was current would
    # have cached "no RDNA2 device is visible" and every later call with a real
    # gfx1035 tensor would report the path unavailable until the process ended.
    # `from . import _C` does not raise when the extension failed to load: the
    # module body above leaves `_C = None`, and the import binds the attribute
    # rather than going looking for a submodule. Without this check a broken build
    # reaches rdna2_import_error with no recorded reason and is reported as
    # "the requested device is not RDNA2" -- the one reading the comment there
    # exists to rule out.
    if not _EXT_AVAILABLE or _C is None:
        _rdna2_import_error = RuntimeError(_EXT_ERROR or "HIP extension not built")
        return

    _rdna2_module = _C
    # Recorded for the message, not for the gate: _rdna2_arch_loaded names the
    # architecture this module was loaded for, and there is only one per process.
    _rdna2_arch_loaded = _rdna2_arch()


def rdna2_is_available(device: torch.device | int | None = None) -> bool:
    """Whether the gfx1035 SageAttention path can run for this device."""
    if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
        return False
    if _rdna2_arch(device) is None:
        return False
    _load_rdna2_module()
    return _rdna2_module is not None


def rdna2_import_error() -> str | None:
    """Why the extension is unavailable, for error messages."""
    _load_rdna2_module()
    if _rdna2_module is not None:
        return None
    if _rdna2_import_error is not None:
        return str(_rdna2_import_error)
    # The extension loaded and this call was still refused: the gate was the
    # per-device architecture check in rdna2_is_available, not the loader. Saying
    # so keeps the message from reading as a broken build.
    return "the requested device is not RDNA2 (gfx103x)"


def _rdna2_ext():
    """The shared extension module, checked.

    Named for what it returns rather than renamed outright: every call site below
    goes through here, so the availability check stays in one place.
    """
    if _rdna2_module is None:
        _load_rdna2_module()
    if _rdna2_module is None:
        raise RuntimeError(f"RDNA2 sage attention unavailable: {_rdna2_import_error}")
    return _rdna2_module


# The granularity the Q/K quantizer writes its per-block scales at: one per this
# many keys. Mirrors sageattn_gfx103x::kQuantGroupQ / kQuantGroupK, which fix the
# scale shapes the C++ used to derive itself.
_RDNA2_QUANT_GROUP_Q = 32
_RDNA2_QUANT_GROUP_K = 16

# Mask representations, mirroring sageattn_gfx10::MaskMode in mma_gfx103x.h and
# the WMMA backend's own MaskMode. The values are shared on purpose: the RDNA2
# port reads the buffers the main HIP backend's sage_prepare_* kernels produce,
# so a prepared mask is the same object on either backend.
_RDNA2_MASK_NONE = 0
_RDNA2_MASK_RAW = 2
_RDNA2_MASK_PREPARED_KEY = 4
_RDNA2_MASK_PREPARED_DENSE = 5
_RDNA2_MASK_PREPARED_DENSE_BOOL = 7
_RDNA2_MASK_PREPARED_DENSE_BF16 = 8
_RDNA2_MASK_PREPARED_DENSE_F16 = 9

# Dtype codes, matching the main HIP backend's mask helpers: 0 fp32, 1 fp16,
# 2 bf16, 3 bool.
_RDNA2_MASK_DTYPE_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.bool: 3,
}

# The producer's key tile, which is the WMMA backend's, not this kernel's: one
# 64-key tile contributes 64 biases plus one "kept anything" descriptor.
_RDNA2_MASK_TILE_K = 64


def _rdna2_prepared_mask_width(kv_length: int) -> int:
    """Element width of the 3D key-mask buffer for ``kv_length`` keys.

    Each 64-key tile contributes 64 biases plus one descriptor saying whether
    the tile kept anything at all, and the row is padded so the producer's
    vector loads stay aligned. Mirrors sage_attention.py's allocation.
    """
    tiles = (kv_length + _RDNA2_MASK_TILE_K - 1) // _RDNA2_MASK_TILE_K
    return ((tiles * (_RDNA2_MASK_TILE_K + 1) + 3) // 4) * 4


def _rdna2_vector_readable(t: torch.Tensor) -> bool:
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


def _rdna2_prepare_key_mask(mask: torch.Tensor) -> torch.Tensor:
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
    from . import _dl

    mask_batch = 1 if mask.stride(0) == 0 else mask.shape[0]
    mask_heads = 1 if mask.stride(1) == 0 else mask.shape[1]
    packed = torch.empty(
        (mask_batch, mask_heads, _rdna2_prepared_mask_width(mask.shape[-1])),
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


def _rdna2_prepare_attn_mask(
    mask: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    kv_length: int | None = None,
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
    if mask.dtype not in _RDNA2_MASK_DTYPE_CODE:
        raise TypeError(
            f"attn_mask must be bool, float16, bfloat16 or float32, got {mask.dtype}"
        )
    if mask.device != q.device:
        raise ValueError("attn_mask must be on the same device as q, k and v")
    # The key length, from the axis the caller's layout puts it on. It cannot be
    # derived from k alone: tensor_layout="HND" is [B, H, S, D] and tensor_layout=
    # "NHD" is [B, S, H, D], so k.shape[2] is the key length under the first and the
    # HEAD COUNT under the second. Reading the wrong one is not a crash and not a
    # wrong answer -- it is a wrong axis feeding two decisions below, so the failure
    # is a ValueError rejecting a valid mask whenever the head count differs from the
    # key length, and the prepared-key packing being chosen against the head count
    # rather than the key count. Defaulted rather than required so the HND behaviour
    # is unchanged for any other caller.
    if kv_length is None:
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
    if key_only and kv_length > _RDNA2_MASK_TILE_K:
        return (
            _RDNA2_MASK_PREPARED_KEY,
            _RDNA2_MASK_DTYPE_CODE[mask.dtype],
            _rdna2_prepare_key_mask(mask),
        )
    # Narrower than one mask tile: there is nothing a packed tile would save,
    # and the kernel already treats the keys past the end as dropped.
    return _RDNA2_MASK_RAW, _RDNA2_MASK_DTYPE_CODE[mask.dtype], mask


def rdna2_pack_attn_mask(
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
    # Named without the rdna2_ prefix on purpose: this is
    # sage_attention's mask packer, not the wrapper above, which is a different
    # function that happens to share its name. The prefix on the local one is
    # what keeps the two apart.


    return _shared_prepare(mask, attention_scale)


def _rdna2_unpack_prepared_mask(
    packed: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    kv_length: int | None = None,
) -> tuple[int, int, torch.Tensor]:
    """Recognise a packed mask buffer and return the mode to read it with.

    The packed forms are self-describing by rank and trailing width, which is
    what lets PrequantizedInt8Attention carry a mask as a bare tensor: a 3D
    buffer is the key layout, a 5D buffer is dense with the last extent saying
    whether it holds keep bits or 16-bit values, and a 4D buffer is still a raw
    mask -- the shared preparer declines to pack one that is too narrow to tile.
    """
    if packed.dim() == 4:
        return _RDNA2_MASK_RAW, _RDNA2_MASK_DTYPE_CODE[packed.dtype], packed
    if packed.dim() == 3:
        # Same axis question as _rdna2_prepare_attn_mask, and it is not only the
        # message here: _rdna2_prepared_mask_width derives the expected packed width
        # from the key length, so reading the head count under NHD sizes the check
        # against the wrong number of 64-key tiles and it both rejects masks packed
        # for the real key length and accepts ones packed for the head count.
        if kv_length is None:
            kv_length = k.shape[2]
        expected = _rdna2_prepared_mask_width(kv_length)
        if packed.shape[-1] != expected:
            raise ValueError(
                f"prepared key mask must be {expected} wide for kv_length "
                f"{kv_length}, got {packed.shape[-1]}"
            )
        if packed.dtype != torch.float32:
            raise ValueError("prepared key mask must be float32")
        return _RDNA2_MASK_PREPARED_KEY, 0, packed
    if packed.dim() == 5:
        if packed.shape[-1] == 32:
            if packed.dtype != torch.int32:
                raise ValueError("a bit-packed dense mask must be int32")
            return _RDNA2_MASK_PREPARED_DENSE_BOOL, 3, packed
        if packed.shape[-1] != 1024:
            raise ValueError(
                f"prepared dense mask must be 32 (bit-packed) or 1024 wide, "
                f"got {packed.shape[-1]}"
            )
        if packed.dtype == torch.bfloat16:
            return _RDNA2_MASK_PREPARED_DENSE_BF16, 2, packed
        if packed.dtype == torch.float16:
            return _RDNA2_MASK_PREPARED_DENSE_F16, 1, packed
        if packed.dtype == torch.float32:
            return _RDNA2_MASK_PREPARED_DENSE, 0, packed
        raise ValueError(
            f"prepared dense mask must be float32, float16 or bfloat16, got {packed.dtype}"
        )
    raise ValueError(
        "a prepared attn_mask must be 3D (key layout) or 5D (dense layout), got "
        f"{packed.dim()}D"
    )


def rdna2_sageattn(
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

    Shapes and layout semantics match SageAttention's ``rdna2_sageattn``:
    ``tensor_layout="HND"`` takes ``[batch, heads, seq, head_dim]`` and
    ``"NHD"`` takes ``[batch, seq, heads, head_dim]``. Only head_dim 64 and 128
    are supported, matching the ported kernels.

    ``attn_mask`` takes the same mask ComfyUI hands to the other backends: bool
    or a float mask broadcastable to ``[batch, heads, q_len, kv_len]``, where a
    False or non-finite entry drops the key. ``prequantized=True`` says the mask
    is already a packed buffer produced by :func:`rdna2_pack_attn_mask` and is read as
    such; that is how :func:`comfy_kitchen.sage_attention.int8_attention_from_prequantized`
    consumes a snapshot without rebuilding it.
    """
    if not rdna2_is_available(q.device):
        raise RuntimeError(
            f"RDNA2 sage attention is unavailable: {rdna2_import_error()}"
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
    if head_dim not in _RDNA2_SUPPORTED_HEAD_DIMS:
        raise ValueError(f"head_dim must be 64 or 128, got {head_dim}")
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name} must be contiguous along head_dim")

    # Resolve the mask to (mode, dtype code, buffer) once, so the raw and packed
    # forms are the same object to the kernel.
    mask_mode = _RDNA2_MASK_NONE
    mask_dtype = 0
    mask_buffer = torch.empty(0, device=q.device, dtype=q.dtype)
    # True when this call built the mask itself, and so owns a temporary the
    # pending kernel still reads.
    owns_mask_buffer = False
    # The key length, from the axis tensor_layout puts it on, derived ONCE here
    # because two helpers below need it and deriving it from k alone gets the head
    # count under NHD: HND is [B, H, S, D] so k.size(2) is the key length, and NHD is
    # [B, S, H, D] so it is k.size(1). The mask helpers read it to validate the mask's
    # last extent and to size the prepared-key packing.
    kv_len = k.size(2) if tensor_layout == "HND" else k.size(1)
    if attn_mask is not None:
        if prequantized:
            mask_mode, mask_dtype, mask_buffer = _rdna2_unpack_prepared_mask(
                attn_mask, q, k, kv_len
            )
        else:
            mask_mode, mask_dtype, mask_buffer = _rdna2_prepare_attn_mask(
                attn_mask, q, k, kv_len
            )
            owns_mask_buffer = True

    input_dtype = dtype
    if dtype == torch.float32:
        q, k, v = (t.half() for t in (q, k, v))
        dtype = torch.float16

    ops = _rdna2_ext()
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
    # smooth_k went with the mean-seq kernel it needed. Nothing in the tree set
    # it, the only caller of rdna2_sageattn passes nothing, and int8_attention lists it
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
        head_dim in _RDNA2_SUPPORTED_HEAD_DIMS
        and not is_causal
        and kv_len <= 1024
        and _rdna2_vector_readable(q)
        and os.getenv("CK_GFX1035_INQ") == "1"
    )
    if not _rdna2_vector_readable(q):
        # The quantizer reads whole 8-element packs whose offsets come from the
        # strides, so a Q whose strides are not multiples of 8 halfs (or whose
        # base is misaligned) has to be repacked. A Q that is merely *strided* --
        # the usual [B,S,H,D].transpose(1,2) -- is read in place: on the
        # short-KV cross-attention shapes this copy was a third of the call.
        q = q.contiguous()
    q_fp = q
    if skip_inq and tensor_layout == "NHD":
        # A transpose view, not as_strided with packed NHD strides: Q only has to
        # be vector-readable to get here, which a merely strided NHD view is, and
        # imposed strides over strided storage read the wrong elements. Same reason
        # the V branch above tests is_contiguous() before its own as_strided.
        q_fp = q.transpose(1, 2)

    # _C has no torch in it, so every buffer is built here. The quantizer always
    # writes packed [B, heads, seq, dim] whatever layout Q arrived in, which is why
    # q_int8 is HND even on the NHD path.
    kv_heads = k.size(1) if tensor_layout == "HND" else k.size(2)
    if skip_inq:
        # Q is quantized inside the attention kernel, so it is never materialized.
        q_int8 = torch.empty(0, device=q.device, dtype=torch.int8)
        q_scale = torch.empty(0, device=q.device, dtype=torch.float32)
    else:
        q_int8 = torch.empty(batch, q_heads, qo_len, head_dim,
                             device=q.device, dtype=torch.int8)
        q_scale = torch.empty(batch, q_heads,
                              (qo_len + _RDNA2_QUANT_GROUP_Q - 1) // _RDNA2_QUANT_GROUP_Q,
                              device=q.device, dtype=torch.float32)
    k_int8 = torch.empty(batch, kv_heads, kv_len, head_dim,
                         device=q.device, dtype=torch.int8)
    k_scale = torch.empty(batch, kv_heads,
                          (kv_len + _RDNA2_QUANT_GROUP_K - 1) // _RDNA2_QUANT_GROUP_K,
                          device=q.device, dtype=torch.float32)

    ops.gfx103x_quant_qk_int8(
        q, k, k_mean, q_int8, q_scale, k_int8, k_scale,
        layout_code, float(sm_scale), int(skip_inq), _stream(q),
    )
    ops.gfx103x_qk_int8_sv_attn(
        q_int8, k_int8, v_for_attn, o,
        q_scale, k_scale,
        layout_code, int(is_causal), float(sm_scale),
        q_fp if skip_inq else torch.empty(0, device=q.device, dtype=q.dtype),
        mask_mode, mask_dtype, mask_buffer, _stream(q),
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


def rdna2_prequantize(
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

    if not rdna2_is_available(q.device):
        raise RuntimeError(f"RDNA2 sage attention is unavailable: {rdna2_import_error()}")

    head_dim = q.size(-1)
    if head_dim not in _RDNA2_SUPPORTED_HEAD_DIMS:
        raise NotImplementedError(
            "prequantize_int8_attention on RDNA2 (gfx103x) requires head_dim 64 or "
            f"128, got {head_dim}."
        )

    mask_snapshot: torch.Tensor | None = None
    if attn_mask is not None:
        mask_snapshot = rdna2_pack_attn_mask(attn_mask, q, k, attention_scale)

    work_q = q if q.dtype != torch.float32 else q.half()
    work_k = k if k.dtype != torch.float32 else k.half()
    if not work_q.is_contiguous():
        work_q = work_q.contiguous()
    if not work_k.is_contiguous():
        work_k = work_k.contiguous()

    ops = _rdna2_ext()
    k_mean = torch.empty(0, device=q.device, dtype=work_k.dtype)
    # skip_q=0: the snapshot is the whole point, so Q has to be packed here
    # rather than left for the attention pass to quantize in place.
    #
    # The snapshot is always HND -- layout 1 below -- and the quantizer writes
    # packed [B, heads, seq, dim] regardless, so the shapes follow from work_q and
    # work_k directly with no layout arithmetic to get wrong.
    batch, q_heads, q_len, head_dim = work_q.shape
    k_batch, kv_heads, kv_len, _ = work_k.shape
    q_int8 = torch.empty(batch, q_heads, q_len, head_dim,
                         device=q.device, dtype=torch.int8)
    q_scale = torch.empty(batch, q_heads,
                          (q_len + _RDNA2_QUANT_GROUP_Q - 1) // _RDNA2_QUANT_GROUP_Q,
                          device=q.device, dtype=torch.float32)
    k_int8 = torch.empty(k_batch, kv_heads, kv_len, head_dim,
                         device=q.device, dtype=torch.int8)
    k_scale = torch.empty(k_batch, kv_heads,
                          (kv_len + _RDNA2_QUANT_GROUP_K - 1) // _RDNA2_QUANT_GROUP_K,
                          device=q.device, dtype=torch.float32)
    ops.gfx103x_quant_qk_int8(
        work_q, work_k, k_mean, q_int8, q_scale, k_int8, k_scale,
        1, float(attention_scale), 0, _stream(work_q),
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


def rdna2_attend_prequantized(quantized) -> torch.Tensor:
    """Attend over a snapshot taken by :func:`rdna2_prequantize`.

    The packed Q/K go back through the same op the fused path uses, with the
    snapshot's mask handed over already packed, so no part of the result depends
    on the float inputs still being alive.
    """
    if not rdna2_is_available(quantized.q.device):
        raise RuntimeError(f"RDNA2 sage attention is unavailable: {rdna2_import_error()}")

    if quantized.q.dtype != torch.int8:
        raise ValueError(
            "a prequantized INT8 attention snapshot on RDNA2 must carry packed "
            f"int8 Q, got {quantized.q.dtype}"
        )

    mask_mode = _RDNA2_MASK_NONE
    mask_dtype = 0
    mask_buffer = torch.empty(0, device=quantized.q.device, dtype=quantized.q.dtype)
    if quantized.attn_mask is not None:
        mask_mode, mask_dtype, mask_buffer = _rdna2_unpack_prepared_mask(
            quantized.attn_mask, quantized.q, quantized.k
        )

    q = quantized.q
    if not q.is_contiguous():
        q = q.contiguous()
    k = quantized.k
    if not k.is_contiguous():
        k = k.contiguous()

    batch, q_heads, q_length, head_dim = q.shape
    # The RDNA2 kernel is fp16-only, so a float32 input accumulates and writes
    # fp16 here and is widened on return -- the same thing rdna2_sageattn() does. Using
    # bf16 instead (as the WMMA backends do, having a bf16 path) would round the
    # result differently, and a snapshot has to match its fused counterpart
    # bit for bit.
    output_dtype = torch.float16 if quantized.input_dtype == torch.float32 \
        else quantized.input_dtype
    o = torch.empty(batch, q_heads, q_length, head_dim, dtype=output_dtype, device=q.device)
    # q_fp is empty: that is how the op is told Q is already packed int8.
    _rdna2_ext().gfx103x_qk_int8_sv_attn(
        q, k, quantized.v, o,
        quantized.q_scale, quantized.k_scale,
        1, 0, float(quantized.attention_scale),
        torch.empty(0, device=q.device, dtype=torch.float16),
        mask_mode, mask_dtype, mask_buffer, _stream(q),
    )
    o = o[..., : quantized.original_head_dim]
    return o.to(torch.float32) if quantized.input_dtype == torch.float32 else o

_register()
