# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HIP backend for AMD RDNA2, RDNA3/3.5 and RDNA4.

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
import importlib.util
import json
import logging
import os
import pathlib
import sys
from collections.abc import Sequence

import torch

from comfy_kitchen._rope_utils import check_rope_inplace, trim_rope_freqs
from comfy_kitchen.backends import eager as _eager
from comfy_kitchen.backends._activations import apply_input_act as _apply_input_act
from comfy_kitchen.backends._activations import input_act_code as _input_act_code
from comfy_kitchen.backends._activations import input_act_width as _input_act_width
from comfy_kitchen.backends.eager import rope as _eager_rope
from comfy_kitchen.backends.eager.quantization import DTYPE_CODE_TO_DTYPE, DTYPE_TO_CODE
from comfy_kitchen.backends.eager.w4a8_int8 import (
    _QUANT_ROW_ELEM_BUDGET,
    _decide_codebook,
    _dequantize_w4a8_int8_weight_from_int8,
    _quantize_w4a8_chunked,
    validate_w4a8_operands,
    validate_w4a8_weight_shape,
)

logger = logging.getLogger("comfy_kitchen.hip")

__all__ = [
    "adaln",
    "na3d",
    "rms_adaln",
    "gemv_awq_w4a16",
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
    "has_wmma",
    "int8_linear",
    "int8_attention_is_available",
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
    "stochastic_rounding_fp8",
    "w4a8_int8_linear",
]

_C = None
_EXT_AVAILABLE = False
_EXT_ERROR = None

try:
    _dir = os.path.dirname(__file__)
    _module_path = None
    for _fn in os.listdir(_dir):
        if _fn.startswith("_C.") and _fn.endswith((".so", ".pyd")):
            _module_path = os.path.join(_dir, _fn)
            break

    if _module_path is None:
        _EXT_ERROR = "HIP extension not built (no _C module in backends/hip)"
    else:
        _spec = importlib.util.spec_from_file_location("comfy_kitchen.backends.hip._C", _module_path)
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


@functools.lru_cache(maxsize=1)
def _visible_gfx_arches() -> tuple[str | None, ...]:
    """One entry per visible device; None where the architecture could not be read.

    Cached: the visible device set is fixed for the life of the process, and
    scaled_mm_v2 consults has_wmma() on every candidate GEMM.
    """
    if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
        return ()

    # ROCm lazy initialization: device_count() may return 0 until HIP is initialized.
    torch.cuda.current_device()

    return tuple(_gfx_arch(i) for i in range(torch.cuda.device_count()))


# RDNA2 has no matrix cores; RDNA3/3.5 and RDNA4 do. This exact manifest is also
# consumed by setup.py and CMake. Never infer support from a gfx prefix: a new
# compiler-recognized target needs its WMMA policy reviewed before it is safe.
_ARCH_MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "architectures.json")
_ARCH_GROUPS = json.loads(
    pathlib.Path(_ARCH_MANIFEST_PATH).read_text(encoding="utf-8")
)
_ARCH_ELEMENTWISE_ONLY = frozenset(_ARCH_GROUPS["elementwise_only"])
_ARCH_WMMA_GFX11 = frozenset(_ARCH_GROUPS["wmma_gfx11"])
_ARCH_WMMA_GFX12 = frozenset(_ARCH_GROUPS["wmma_gfx12"])
_ARCH_WMMA = _ARCH_WMMA_GFX11 | _ARCH_WMMA_GFX12
_ARCH_SUPPORTED = _ARCH_ELEMENTWISE_ONLY | _ARCH_WMMA

# The GEMMs, and only the GEMMs, need matrix cores. Everything else is elementwise
# or a scalar reduction and runs on any supported architecture. This set names the
# registry-dispatched GEMMs so _build_constraints can drop them on RDNA2; the fp8
# GEMM is not among them because it is reached through scaled_mm_v2's _hip_fp8_gemm,
# which gates on has_wmma() itself rather than through the registry.
_WMMA_ONLY_OPS = frozenset({
    "int8_linear",
    "na3d",
    "convrot_w4a4_linear",
    "scaled_mm_svdquant_w4a4",
    "w4a8_int8_linear",
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


def is_available() -> bool:
    return _EXT_AVAILABLE and _unsupported_arch_reason(_visible_gfx_arches()) is None


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


def _convrot_supported(
    k: int, group_size: int, device: torch.device, dtype: torch.dtype
) -> bool:
    """Whether the fused rotation can take a row of this width on ``device``.

    It stages the whole row in LDS, which bounds K per device, and it rotates K/G
    whole groups while reading back all K entries, so a partial trailing group
    would quantize uninitialized LDS.

    The budget is read for the operand's own device, not the process-current one:
    the kernels launch on the operand's stream, so in a multi-GPU process the two
    can be different cards with different budgets.
    """
    if group_size not in (16, 64, 256) or k % group_size != 0:
        return False
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False

    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, dtype)
    max_k = _convrot_max_k.get(key)
    if max_k is None:
        with torch.cuda.device(index):
            max_k = _C.convrot_max_k(DTYPE_TO_CODE[dtype])
        _convrot_max_k[key] = max_k
    return max_k > 0 and k <= max_k


def _rotate_quant_int8(
    x2d: torch.Tensor, group_size: int, input_act: str | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k_in = x2d.shape
    # swiglu halves the row: the [gate | up] input is twice the quantized width.
    k = k_in // _input_act_width(input_act)
    q = torch.empty((m, k), dtype=torch.int8, device=x2d.device)
    scales = torch.empty((m,), dtype=torch.float32, device=x2d.device)
    # check_convrot_k queries the current device's LDS budget, so pin it to the
    # operand's device rather than trusting the caller thread's current device.
    with torch.cuda.device(x2d.device):
        _C.quantize_int8_convrot(
            _dl(x2d), _dl(q), _dl(scales), m, k, group_size,
            _input_act_code(input_act), _stream(x2d),
        )
    return q, scales


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
        x.shape[-1], group_size, x.device, x.dtype
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
        weight.shape[-1], group_size, weight.device, weight.dtype
    ):
        return _eager.quantize_int8_convrot_weight(
            weight, group_size, stochastic_rounding=stochastic_rounding
        )

    w2d = weight.reshape(-1, weight.shape[-1]).contiguous()
    q, scales = _rotate_quant_int8(w2d, group_size)
    return q.reshape(weight.shape), scales.reshape(*weight.shape[:-1], 1)


def int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    convrot: bool = False,
    convrot_groupsize: int = 256,
    input_act: str | None = None,
) -> torch.Tensor:
    """INT8 linear with dynamic row-wise activation quantization, on WMMA."""
    # Rejected here so every route fails the same way, not just the fused one.
    _input_act_code(input_act)
    # k_act is the activated (quantized) row width: swiglu halves the raw row.
    k_act = x.shape[-1] // _input_act_width(input_act)
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
        if not _convrot_supported(k, convrot_groupsize, x.device, x.dtype):
            return _eager.int8_linear(
                x, weight, weight_scale, bias, out_dtype, convrot, convrot_groupsize,
                input_act=input_act,
            )
        # The only route that absorbs the activation; the rest apply it eagerly.
        q, x_scale = _rotate_quant_int8(x2d, convrot_groupsize, input_act)
    else:
        x2d = _apply_input_act(x2d, input_act)
        q = torch.empty((m, k), dtype=torch.int8, device=x.device)
        x_scale = torch.empty((m,), dtype=torch.float32, device=x.device)
        _C.quantize_int8_rowwise(_dl(x2d), _dl(q), _dl(x_scale), m, k, _stream(x))

    x_scale = x_scale.reshape(-1).contiguous()
    if bias is not None:
        bias = _bias_operand(bias, n, x.device)

    out = torch.empty((m, n), dtype=out_dtype, device=x.device)
    _C.int8_gemm(
        _dl(q), _dl(weight), _dl(out),
        _dl(x_scale), _dl(weight_scale), 0 if weight_scale.numel() == 1 else 1,
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
    """Decode packed INT4 weights to the grouped INT8 grid the GEMM consumes."""
    n, k_half = qdata.shape
    k = k_half * 2
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
) -> torch.Tensor:
    n, k_half = qdata.shape
    k = k_half * 2
    device = x.device
    orig_shape = x.shape
    x2d = x.reshape(-1, k).contiguous()
    m = x2d.shape[0]

    # Rotated and quantized once: the chunk loop only walks the weight.
    xq, xs = _rotate_quant_int8(x2d, convrot_groupsize)

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
    # An empty weight has no chunk to size; the loop then has nothing to walk.
    chunk_cols = max(1, _w4a8_chunk_cols(m, n, k, device))
    workspace = torch.empty((chunk_cols, k), dtype=torch.int8, device=device)
    out = torch.empty((m, n), dtype=out_dtype, device=device)
    _C.w4a8_int8_gemm_chunked(
        _dl(xq),
        _dl(qdata_arg),
        _dl(s_rel_arg),
        scale_code,
        None if codebook_arg is None else _dl(codebook_arg),
        _dl(s_channel_arg),
        _dl(xs_arg),
        None if bias_arg is None else _dl(bias_arg),
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


# Keyed by device: the fused requantize holds one float per 16-wide group in LDS,
# so the widest K it can take is a property of whichever card the weight lives on.
_w4a8_requant_max_k: dict[int, int] = {}


def _requant_supported(k: int, device: torch.device, dtype: torch.dtype) -> bool:
    """Whether the fused requantize can take a row of this width on ``device``.

    The budget is read for the weight's own device rather than the process-current
    one, so a multi-GPU process asks the card the kernel will actually launch on.
    """
    if not _EXT_AVAILABLE or k % 16 != 0:
        return False
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False

    index = device.index if device.index is not None else torch.cuda.current_device()
    max_k = _w4a8_requant_max_k.get(index)
    if max_k is None:
        with torch.cuda.device(index):
            max_k = _C.w4a8_requant_max_k()
        _w4a8_requant_max_k[index] = max_k
    return max_k > 0 and k <= max_k


def _fused_quantize_w4a8(
    weight: torch.Tensor,
    codebook: torch.Tensor,
    convrot_groupsize: int,
    stochastic_rounding: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, torch.Tensor]:
    """Rotate and requantize a row block at a time, so no whole rotated copy is held.

    Every output is per-row and the kernel reduces only within a row, so a block
    writes straight into its own output slices.
    """
    n, k = weight.shape
    cb = codebook.to(device=weight.device, dtype=torch.float32).contiguous()
    packed = torch.empty(n, k // 2, dtype=torch.int8, device=weight.device)
    s_rel = torch.empty(n, k // 16, dtype=torch.float8_e4m3fn, device=weight.device)
    s_channel = torch.empty(n, dtype=torch.float32, device=weight.device)
    block = max(1, _QUANT_ROW_ELEM_BUDGET // max(k, 1))
    for r0 in range(0, n, block):
        r1 = min(r0 + block, n)
        rot = _eager.rotate_int8_convrot_weight(weight[r0:r1].contiguous(), convrot_groupsize)
        # Offset the seed by the row start so blocks decorrelate but a fixed block
        # size still reproduces.
        seed = stochastic_rounding + r0 if stochastic_rounding > 0 else 0
        _C.quantize_w4a8_convrot(
            _dl(rot.contiguous()),
            _dl(cb),
            _dl(packed[r0:r1]),
            _dl(s_rel[r0:r1].view(torch.uint8)),
            _dl(s_channel[r0:r1]),
            r1 - r0,
            k,
            stochastic_rounding > 0,
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
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Prepare W4A8 weights with the fused HIP requantize and eager ConvRot."""
    validate_w4a8_weight_shape(weight, group_size, convrot_groupsize)
    # The fused kernel covers the default codebook layout only. Asymmetric, uniform,
    # fp32-scale and other group sizes stay on the chunked eager path.
    if (
        symmetric
        and codebook
        and group_size == 16
        and scale_dtype == torch.float8_e4m3fn
        and _requant_supported(weight.shape[1], weight.device, weight.dtype)
    ):
        cb = (
            codebook_tensor
            if codebook_tensor is not None
            else _decide_codebook(
                weight, _eager.rotate_int8_convrot_weight, group_size, convrot_groupsize
            )
        )
        return _fused_quantize_w4a8(weight, cb, convrot_groupsize, stochastic_rounding)
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
) -> torch.Tensor:
    """``x @ W.T + bias`` via the HIP INT4 decode feeding the WMMA INT8 GEMM.

    The weight is decoded a column chunk at a time so a chunk is still cached when
    the GEMM reads it back, instead of the whole [N, K] INT8 weight round-tripping
    through global memory.
    """
    validate_w4a8_operands(
        qdata, s_rel, s_channel, codebook, correction, group_size, convrot_groupsize
    )
    if x.shape[-1] != qdata.shape[-1] * 2:
        raise ValueError(f"Input K={x.shape[-1]} does not match qdata K={qdata.shape[-1] * 2}")

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

    # The layout allows any ConvRot group that divides K; the fused activation
    # quantizer takes only the three it has butterfly stages for, and its LDS
    # budget bounds K. int8_linear applies the same test before its own fast path.
    if not _convrot_supported(x.shape[-1], convrot_groupsize, x.device, x.dtype):
        int8_weight = _dequant_int4_grouped_to_int8(qdata, s_rel, codebook, group_size)
        return _eager.int8_linear(
            x, int8_weight, s_channel, bias, out_dtype, True, convrot_groupsize
        )

    return _w4a8_int8_linear_chunked(
        x, qdata, s_rel, s_channel, codebook, bias, group_size, convrot_groupsize, out_dtype
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
# Registration
# ---------------------------------------------------------------------------

def _build_constraints(has_wmma: bool = True) -> dict:
    from comfy_kitchen.constraints import (
        DivisibleBy,
        ExactDims,
        FunctionConstraints,
        ParamConstraint,
        ValidationResult,
        na3d_common_call_rule,
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

    constraints = {
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
        # RDNA2: the GEMM kernels are compiled but trap, so they must not be
        # advertised. Dropping them here routes those ops to triton/eager while the
        # elementwise kernels below still dispatch to HIP.
        constraints = {k: v for k, v in constraints.items() if k not in _WMMA_ONLY_OPS}

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


def _sage_cta_k(head_dim: int, kv_length: int, has_mask: bool) -> int:
    """Keys per attention iteration.

    Always 64. The CUDA backend widens this to 128 for long unmasked keys; the
    wide tile is implemented here too and measures slower on RDNA, where V is
    staged transposed and the tile doubles both LDS allocations at once. Kept as
    a function because the choice belongs with the buffer padding it decides.
    """
    del head_dim, kv_length, has_mask
    return _SAGE_CTA_K


def int8_attention_is_available() -> bool:
    """Whether the INT8 attention kernels can run here.

    Stricter than is_available(): the kernel is built on the matrix cores, so
    RDNA2 declines it the way it declines the GEMMs.
    """
    return has_wmma()


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
        # V is stored transposed, [B * H * D, padded_K], with the tail zero filled.
        "v_int8": torch.empty(
            batch * kv_heads * head_dim, padded_k, dtype=torch.int8, device=device
        ),
        "v_scale": torch.empty(batch * kv_heads * head_dim, dtype=torch.float32, device=device),
    }

    # One key index per batch and KV head, written by the stabilization detector
    # and read by the K quantizer. It is scratch, not part of the packed form, so
    # it stays out of the buffers the split API hands back.
    anchor_indices = torch.empty(batch, kv_heads, dtype=torch.int32, device=device)
    return buffers, anchor_indices


def sage_int8_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attention_scale: float,
    attn_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Quantize and attend in one call. q, k and v are already padded to a
    supported head dimension; the output keeps that width."""
    batch, q_heads, q_length, head_dim = q.shape
    output_dtype = torch.bfloat16 if q.dtype == torch.float32 else q.dtype
    output = torch.empty(
        batch, q_heads, q_length, head_dim, dtype=output_dtype, device=q.device
    )

    cta_k = _sage_cta_k(head_dim, k.shape[2], attn_mask is not None)
    buffers, anchor_indices = _sage_buffers(q, k, cta_k)
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
    )
    return output


def sage_int8_quantize(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cta_k: int = _SAGE_CTA_K,
) -> dict:
    """Quantize q, k and v without allocating the attention output."""
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
    cta_k: int = _SAGE_CTA_K,
) -> torch.Tensor:
    """Attend over the packed layouts sage_int8_quantize produced."""
    batch, q_heads, q_length, head_dim = q_int8.shape
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
    registry.register(
        name="hip",
        module=sys.modules[__name__],
        capabilities=_build_constraints(has_wmma),
    )
    logger.debug(
        "registered HIP backend for %s (%s)",
        ", ".join(sorted({a for a in arches if a})),
        "with WMMA" if has_wmma else "elementwise only, no matrix cores",
    )


_register()
