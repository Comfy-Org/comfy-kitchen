# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HIP backend for AMD RDNA4 (gfx12), built on WMMA intrinsics.

Every matmul is a WMMA kernel compiled from the sources in this directory; the
backend does not link or call hipBLAS/hipBLASLt.

The WMMA builtins used (v_wmma_*_w32_gfx12) are gfx12-only, so the backend does
not register on any other architecture, including RDNA3 (gfx11).
"""
import importlib.util
import logging
import os
import sys

import torch

from comfy_kitchen.backends import eager as _eager
from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE

logger = logging.getLogger("comfy_kitchen.hip")

__all__ = [
    "adaln",
    "gemv_awq_w4a16",
    "quantize_svdquant_w4a4",
    "scaled_mm_svdquant_w4a4",
    "apply_rope",
    "apply_rope1",
    "apply_rope_split_half",
    "apply_rope_split_half1",
    "convrot_w4a4_linear",
    "dequantize_convrot_w4a4_weight",
    "dequantize_per_tensor_fp8",
    "int8_linear",
    "is_available",
    "quantize_and_rotate_rowwise",
    "quantize_convrot_w4a4_weight",
    "quantize_int8_convrot_weight",
    "quantize_int8_rowwise",
    "quantize_int8_tensorwise",
    "quantize_per_tensor_fp8",
    "scaled_mm_fp8",
    "stochastic_rounding_fp8",
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
except Exception as e:  # noqa: BLE001 - a broken extension must not break import
    _EXT_ERROR = f"Failed to load HIP extension: {e}"
    _C = None


def _gfx_arch(device: torch.device | int | None = None) -> str | None:
    """gfx architecture of ``device``, e.g. "gfx1201". Defaults to the current device."""
    if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
        return None
    try:
        return torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    except Exception:  # noqa: BLE001
        return None


def _visible_gfx_arches() -> list[str | None]:
    """One entry per visible device; None where the architecture could not be read."""
    if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
        return []
    return [_gfx_arch(i) for i in range(torch.cuda.device_count())]


def _unsupported_arch_reason(arches: list[str | None]) -> str | None:
    """Why the backend must not register for this set of devices, or None if it may.

    Registration is per-process while kernels launch on the tensor's own device,
    so a single non-gfx12 device in the process makes any registration unsafe. A
    device whose architecture cannot be read counts against it: it cannot be
    shown to be gfx12.
    """
    if not arches:
        return "no HIP device available"
    if any(a is None for a in arches):
        return "could not read the architecture of every visible device"
    unsupported = sorted({a for a in arches if not a.startswith("gfx12")})
    if unsupported:
        return f"WMMA kernels require RDNA4 (gfx12), found {', '.join(unsupported)}"
    return None


def is_available() -> bool:
    return _EXT_AVAILABLE and _unsupported_arch_reason(_visible_gfx_arches()) is None


def _stream(t: torch.Tensor) -> int:
    return torch.cuda.current_stream(t.device).cuda_stream


# The epilogues read a scalar per element with one dtype code.
_EPILOGUE_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _operand(t: torch.Tensor, device: torch.device, name: str, shape=None) -> torch.Tensor:
    """A contiguous view of ``t`` on ``device``, since the kernels take raw pointers.

    Every launch uses one stream and one set of extents, so an operand left on
    another device would be dereferenced there, and a mis-shaped one read past its
    end.
    """
    if shape is not None and tuple(t.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(t.shape)}")
    return t.to(device=device).contiguous()


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
    a = a.contiguous()
    b_nk = _weight_as_nk(b).to(device=a.device)

    M, K = a.shape
    N = b_nk.shape[0]
    if b_nk.shape[1] != K:
        raise ValueError(f"inner dimension mismatch: a K={K}, b K={b_nk.shape[1]}")

    # The WMMA K-step and the small-M GEMV both read a row 16 bytes at a time.
    if K % 16 != 0:
        raise ValueError(f"scaled_mm_fp8 requires K divisible by 16, got {K}")
    # The kernel takes raw pointers on a's stream, so the scales and bias have to
    # live on a's device, and the epilogue indexes bias[col].
    scale_a = _scale_operand(scale_a, a.device)
    scale_b = _scale_operand(scale_b, a.device)
    if bias is not None:
        bias = _bias_operand(bias, N, a.device)

    out = torch.empty((M, N), dtype=out_dtype, device=a.device)
    _C.scaled_mm_fp8(
        _dl(a.view(torch.uint8)), _dl(b_nk.view(torch.uint8)), _dl(out),
        _dl(scale_a), _dl(scale_b), None if bias is None else _dl(bias),
        M, N, K, DTYPE_TO_CODE[out_dtype], _stream(a),
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
    M, K = x2d.shape
    q = torch.empty((M, K), dtype=torch.int8, device=x.device)
    scales = torch.empty((M,), dtype=torch.float32, device=x.device)
    _C.quantize_int8_rowwise(_dl(x2d), _dl(q), _dl(scales), M, K, _stream(x))
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


_convrot_max_k: int | None = None


def _convrot_supported(k: int, group_size: int) -> bool:
    """Whether the fused rotation can take a row of this width.

    It stages the whole row in LDS, which bounds K per device, and it rotates K/G
    whole groups while reading back all K entries, so a partial trailing group
    would quantize uninitialized LDS.
    """
    global _convrot_max_k
    if group_size not in (16, 64, 256) or k % group_size != 0:
        return False
    if _convrot_max_k is None:
        _convrot_max_k = _C.convrot_max_k()
    return _convrot_max_k > 0 and k <= _convrot_max_k


def _rotate_quant_int8(x2d: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    M, K = x2d.shape
    q = torch.empty((M, K), dtype=torch.int8, device=x2d.device)
    scales = torch.empty((M,), dtype=torch.float32, device=x2d.device)
    _C.quantize_int8_convrot(_dl(x2d), _dl(q), _dl(scales), M, K, group_size, _stream(x2d))
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
    if stochastic_rounding or not _convrot_supported(x.shape[-1], group_size):
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
    if stochastic_rounding or not _convrot_supported(weight.shape[-1], group_size):
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
) -> torch.Tensor:
    """INT8 linear with dynamic row-wise activation quantization, on WMMA."""
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"Input and weight inner dimensions must match, got {x.shape[-1]} and {weight.shape[-1]}"
        )

    weight = weight.to(device=x.device).contiguous()
    weight_scale = weight_scale.to(device=x.device, dtype=torch.float32).reshape(-1)
    if weight_scale.numel() not in (1, weight.shape[0]):
        raise ValueError(
            f"INT8 weight scale must be scalar or per-output-channel, got {tuple(weight_scale.shape)}"
        )

    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1]).contiguous()
    M, K = x2d.shape
    N = weight.shape[0]

    # The WMMA K-step and the small-M GEMV both read a row 16 bytes at a time.
    if K % 16 != 0:
        raise ValueError(f"int8_linear requires K divisible by 16, got {K}")

    if convrot:
        if K % convrot_groupsize != 0:
            raise ValueError(
                f"ConvRot group size {convrot_groupsize} does not divide input features {K}"
            )
        if convrot_groupsize not in (16, 64, 256):
            raise ValueError(f"ConvRot group size must be 16, 64 or 256, got {convrot_groupsize}")
        if not _convrot_supported(K, convrot_groupsize):
            return _eager.int8_linear(
                x, weight, weight_scale, bias, out_dtype, convrot, convrot_groupsize
            )
        q, x_scale = _rotate_quant_int8(x2d, convrot_groupsize)
    else:
        q = torch.empty((M, K), dtype=torch.int8, device=x.device)
        x_scale = torch.empty((M,), dtype=torch.float32, device=x.device)
        _C.quantize_int8_rowwise(_dl(x2d), _dl(q), _dl(x_scale), M, K, _stream(x))

    x_scale = x_scale.reshape(-1).contiguous()
    if bias is not None:
        bias = _bias_operand(bias, N, x.device)

    out = torch.empty((M, N), dtype=out_dtype, device=x.device)
    _C.int8_gemm(
        _dl(q), _dl(weight), _dl(out),
        _dl(x_scale), _dl(weight_scale), 0 if weight_scale.numel() == 1 else 1,
        None if bias is None else _dl(bias),
        M, N, K, DTYPE_TO_CODE[out_dtype], _stream(x),
    )
    return out.reshape(*orig_shape[:-1], N)


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
    if stochastic_rounding or not _convrot_supported(weight.shape[-1], convrot_groupsize):
        return _eager.quantize_convrot_w4a4_weight(
            weight, convrot_groupsize, quant_group_size, stochastic_rounding
        )

    w = weight.reshape(-1, weight.shape[-1]).contiguous()
    N, K = w.shape
    q = torch.empty((N, K // 2), dtype=torch.int8, device=w.device)
    scales = torch.empty((N,), dtype=torch.float32, device=w.device)
    _C.convrot_quant_int4(_dl(w), _dl(q), _dl(scales), N, K, convrot_groupsize, _stream(w))
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

    N, Kp = qdata.shape
    unpacked = torch.empty((N, Kp * 2), dtype=torch.int8, device=qdata.device)
    _C.unpack_int4(_dl(qdata.contiguous()), _dl(unpacked), N * Kp, _stream(qdata))

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
    # The tile loader reads the packed row (K/2 bytes) in 16-byte chunks. A group
    # size of 16 alone would allow a K that packs to a partial chunk.
    if x.shape[-1] % 32 != 0:
        raise ValueError(f"convrot_w4a4_linear requires K divisible by 32, got {x.shape[-1]}")
    if x.shape[-1] % convrot_groupsize != 0:
        raise ValueError(
            f"Input K={x.shape[-1]} not divisible by convrot_groupsize {convrot_groupsize}"
        )
    if not _convrot_supported(x.shape[-1], convrot_groupsize):
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
    M, K = x2d.shape
    N = qweight.shape[0]

    qact = torch.empty((M, K // 2), dtype=torch.int8, device=x.device)
    x_scale = torch.empty((M,), dtype=torch.float32, device=x.device)
    _C.convrot_quant_int4(_dl(x2d), _dl(qact), _dl(x_scale), M, K, convrot_groupsize, _stream(x))

    wscales = wscales.to(device=x.device, dtype=torch.float32).reshape(-1)
    if wscales.numel() != N:
        raise ValueError(f"wscales must have {N} entries, got {wscales.numel()}")
    if bias is not None:
        bias = _bias_operand(bias, N, x.device)
    qw = _operand(qweight, x.device, "qweight", shape=(N, K // 2))

    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    _C.convrot_w4a4_gemm(
        _dl(qact), _dl(qw), _dl(out),
        _dl(x_scale), _dl(wscales), None if bias is None else _dl(bias),
        M, N, K, DTYPE_TO_CODE[x.dtype], _stream(x),
    )
    return out.reshape(*orig_shape[:-1], N)


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
    M, K = x2d.shape
    N = qweight.shape[0]

    # The inner loop decodes eight weights at a time and rescales the chunk once,
    # so a chunk must not straddle a group boundary.
    if group_size <= 0 or group_size % 8 != 0:
        raise ValueError(f"group_size must be a positive multiple of 8, got {group_size}")
    if K % group_size != 0:
        raise ValueError(f"K={K} not divisible by group_size={group_size}")
    if qweight.shape[1] * 2 != K:
        raise ValueError(f"qweight K//2={qweight.shape[1]} inconsistent with x K={K}")

    # The kernel decodes scales and zeros with a single dtype code, taken from
    # wscales; a wzeros of another dtype would be read as that one.
    wscales = _operand(wscales, x.device, "wscales", shape=(K // group_size, N))
    if wscales.dtype not in _EPILOGUE_DTYPES:
        raise ValueError(f"wscales dtype {wscales.dtype} is not supported")
    wzeros = _operand(wzeros, x.device, "wzeros", shape=(K // group_size, N)).to(wscales.dtype)
    qw = _operand(qweight, x.device, "qweight", shape=(N, K // 2))
    if bias is not None:
        bias = _bias_operand(bias, N, x.device)

    out_dtype = wscales.dtype
    out = torch.empty((M, N), dtype=out_dtype, device=x.device)
    _C.gemv_awq_w4a16(
        _dl(x2d), _dl(qw), _dl(wscales), _dl(wzeros),
        None if bias is None else _dl(bias), _dl(out),
        M, N, K, group_size, _stream(x),
    )
    return out.reshape(*orig_shape[:-1], N)


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
    M, K = x.shape
    if K % _SVD_GROUP != 0:
        raise ValueError(f"K={K} not divisible by group_size={_SVD_GROUP}")

    if pad_size <= 0:
        raise ValueError(f"pad_size must be positive, got {pad_size}")
    if lora_down.dim() != 2:
        raise ValueError(f"lora_down must be 2D, got {tuple(lora_down.shape)}")

    R = lora_down.shape[1]
    m_pad = -(-M // pad_size) * pad_size

    # The kernels take M, K and R with no bounds of their own, so every operand
    # has to match those extents and sit on x's device.
    xc = x.contiguous()
    smooth = _operand(smooth.reshape(-1), x.device, "smooth", shape=(K,))
    lora_down = _operand(lora_down, x.device, "lora_down", shape=(K, R))
    # The LoRA branch is defined on the un-shifted, un-smoothed activation.
    lora_src = _operand(lora_x if lora_x is not None else x, x.device, "lora_x", shape=(M, K))

    # Padded rows stay zero: q = 0 and scale = 0 contribute nothing downstream.
    q = torch.zeros((m_pad, K // 2), dtype=torch.int8, device=x.device)
    ascales = torch.zeros((K // _SVD_GROUP, m_pad), dtype=x.dtype, device=x.device)
    lora_act = torch.zeros((m_pad, R), dtype=torch.float32, device=x.device)

    _C.svdquant_quantize(
        _dl(xc), _dl(smooth), _dl(q), _dl(ascales),
        M, m_pad, K, act_unsigned, _stream(x),
    )
    _C.svdquant_lora_down(
        _dl(lora_src), _dl(lora_down), _dl(lora_act[:M]), M, K, R, _stream(x)
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

    M, k_half = act.shape
    N = wgt.shape[0]
    K = k_half * 2
    R = lora_up.shape[1]

    # One scale per 64-element group: a partial trailing group has no scale, and the
    # (K // 64, ...) scale shapes below would silently truncate it away.
    if K % _SVD_GROUP != 0:
        raise ValueError(f"K={K} not divisible by group_size={_SVD_GROUP}")

    # The kernel gets M, N, K and R but no tensor bounds, so each operand has to
    # match those extents and share act's device.
    dev = act.device
    act = _operand(act, dev, "act")
    wgt = _operand(wgt, dev, "wgt", shape=(N, k_half))
    ascales = _operand(ascales, dev, "ascales", shape=(K // _SVD_GROUP, M))
    wscales = _operand(wscales, dev, "wscales", shape=(K // _SVD_GROUP, N))
    if wscales.dtype not in _EPILOGUE_DTYPES:
        raise ValueError(f"wscales dtype {wscales.dtype} is not supported")
    lora_act_in = _operand(lora_act_in, dev, "lora_act_in", shape=(M, R))
    lora_up = _operand(lora_up, dev, "lora_up", shape=(N, R))
    if bias is not None:
        bias = _bias_operand(bias, N, dev)

    out = torch.empty((M, N), dtype=wscales.dtype, device=dev)
    _C.svdquant_gemm(
        _dl(act), _dl(wgt), _dl(out),
        _dl(ascales), _dl(wscales),
        _dl(lora_act_in), _dl(lora_up),
        None if bias is None else _dl(bias),
        M, N, K, R, act_unsigned, _stream(act),
    )
    return out


# ---------------------------------------------------------------------------
# Normalization and positional encoding
# ---------------------------------------------------------------------------

def adaln(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    from comfy_kitchen.backends._modulation import adaln_prep_modulation

    orig_shape = x.shape
    d = x.shape[-1]
    n = x.numel() // d

    x_flat = x.reshape(n, d).contiguous()
    scale_flat, scale_group = adaln_prep_modulation(scale, x, n, d)
    shift_flat, shift_group = adaln_prep_modulation(shift, x, n, d)

    out = torch.empty_like(x_flat)
    _C.adaln(
        _dl(x_flat), _dl(scale_flat), _dl(shift_flat), _dl(out),
        n, d, scale_group, shift_group, eps, _stream(x),
    )
    return out.reshape(orig_shape)


def _rope(xq, xk, freqs_cis, split_half):
    # One dtype code and one stream are passed for the whole launch, so every
    # buffer has to agree on device, and xq/xk on dtype.
    if freqs_cis.device != xq.device:
        raise ValueError("freqs_cis must be on the same device as the input")
    if xk is not None:
        if xk.device != xq.device:
            raise ValueError("xq and xk must be on the same device")
        if xk.dtype != xq.dtype:
            raise ValueError("xq and xk must have the same dtype")

    xq = xq.contiguous()
    freqs_cis = freqs_cis.contiguous()
    xq_out = torch.empty_like(xq)

    xk_c = xk_out = None
    if xk is not None:
        xk_c = xk.contiguous()
        xk_out = torch.empty_like(xk_c)

    _C.apply_rope(
        _dl(xq), None if xk_c is None else _dl(xk_c), _dl(freqs_cis),
        _dl(xq_out), None if xk_out is None else _dl(xk_out),
        split_half, _stream(xq),
    )
    return xq_out, xk_out


def apply_rope1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    return _rope(x, None, freqs_cis, False)[0]


def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # The kernel indexes one set of strides and reads both tensors with one dtype
    # code, so a difference in either is done one at a time.
    if xq.shape != xk.shape or xq.dtype != xk.dtype:
        return apply_rope1(xq, freqs_cis), apply_rope1(xk, freqs_cis)
    q, k = _rope(xq, xk, freqs_cis, False)
    return q, k


def apply_rope_split_half1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    return _rope(x, None, freqs_cis, True)[0]


def apply_rope_split_half(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if xq.shape != xk.shape or xq.dtype != xk.dtype:
        return apply_rope_split_half1(xq, freqs_cis), apply_rope_split_half1(xk, freqs_cis)
    q, k = _rope(xq, xk, freqs_cis, True)
    return q, k


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def _build_constraints() -> dict:
    from comfy_kitchen.constraints import (
        DivisibleBy,
        ExactDims,
        FunctionConstraints,
        ParamConstraint,
    )

    # PyTorch exposes ROCm tensors with device type "cuda".
    dev = frozenset({"cuda", "hip"})
    floats = frozenset({torch.float32, torch.float16, torch.bfloat16})
    fp8s = frozenset({torch.float8_e4m3fn, torch.float8_e5m2})
    out_floats = frozenset({torch.float32, torch.float16, torch.bfloat16})

    return {
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
        # K must be a multiple of 16 so a WMMA K-step never straddles a row end.
        "int8_linear": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=floats, shape_rules=(DivisibleBy(-1, 16),)),
                "weight": ParamConstraint(dtypes=frozenset({torch.int8})),
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
    }


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

    registry.register(
        name="hip",
        module=sys.modules[__name__],
        capabilities=_build_constraints(),
    )
    logger.debug("registered HIP backend for %s", ", ".join(sorted(set(arches))))


_register()
