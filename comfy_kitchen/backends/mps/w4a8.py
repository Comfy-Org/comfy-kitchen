# SPDX-License-Identifier: Apache-2.0
"""Weight-only W4A8 / W6A8 linear layers for Apple GPUs.

The packed weight stays resident. Each call decodes it to fp16 with one Metal pass
(code unpack, level lookup, group scale, int8-grid rounding, channel scale, optional
asymmetric correction) and runs the stock fp16 GEMM. Activations are rotated online
instead of being quantized, so the result is the float linear over the decoded weight.

MPS stores fp8 tensors but cannot convert them, so nothing here does fp8 arithmetic:
dequantize reads the fp8 scale bytes in the kernel, and quantize (the LoRA requantize
path) runs eager's quantizer on CPU and moves the packed result back.
"""
from __future__ import annotations

import torch

from comfy_kitchen.backends.eager import w4a8_int8 as eager_w4a8
from comfy_kitchen.backends.eager.quantization import rotate_int8_convrot_weight
from comfy_kitchen.backends.eager.w4a8_int8 import validate_w4a8_operands
from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation

from .linear import _MAX_DECODE_BYTES, _from_compute, _to_compute

_SHADERS = """
#include <metal_stdlib>
using namespace metal;

// Bit-exact with the eager decode: the asymmetric path's multiply and add must round
// separately, as the two torch ops do, not fuse into one fma.
#pragma METAL fp contract(off)

inline float group_scale(device const uchar* s, constant float* e4m3_lut, uint i) {
    return e4m3_lut[s[i]];
}

inline float group_scale(device const float* s, constant float*, uint i) {
    return s[i];
}

// One thread decodes four consecutive columns of one row (group sizes are multiples of 4).
// Row layout: K/2 bytes of nibbles (even col = low nibble); at 6 bits a K/4-byte plane of
// top-2-bit pairs follows (col c in byte c/4 at bit 2*(c%4)).
// w = int8(clamp(rint(levels[code] * s_rel), -127, 127)) * s_channel [+ correction]
// O is the output element type: half for the GEMM, float for dequantize (exact, as eager).
template <int BITS, typename S, typename O>
kernel void w4a8_decode(device vec<O, 4>* out,
                        device const uchar* q,
                        device const S* s_rel,
                        device const float* s_channel,
                        constant float* levels,
                        constant float* e4m3_lut,
                        device const float* correction,
                        constant long& K_,
                        constant long& N_,
                        constant long& group_size_,
                        constant long& row0_,
                        constant long& has_correction,
                        uint idx [[thread_position_in_grid]]) {
    const uint K = uint(K_);
    const uint quads = K / 4;
    const uint row = uint(row0_) + idx / quads;
    const uint col = (idx % quads) * 4;
    device const uchar* qrow = q + ulong(row) * (K * BITS / 8);

    const uchar b0 = qrow[col / 2];
    const uchar b1 = qrow[col / 2 + 1];
    uint4 codes = uint4(b0 & 0xF, b0 >> 4, b1 & 0xF, b1 >> 4);
    if (BITS == 6) {
        const uint hi = qrow[K / 2 + col / 4];
        codes |= ((uint4(hi) >> uint4(0, 2, 4, 6)) & 0x3) << 4;
    }

    const uint group = col / uint(group_size_);
    const float s = group_scale(s_rel, e4m3_lut, row * (K / uint(group_size_)) + group);
    const float sc = s_channel[row];
    float4 w;
    for (int j = 0; j < 4; ++j) {
        // Through int: torch's int8 cast drops the sign of a rounded -0.
        const int grid = int(clamp(rint(levels[codes[j]] * s), -127.0f, 127.0f));
        w[j] = float(grid) * sc;
    }
    if (has_correction) {
        w += correction[group * uint(N_) + row];
    }
    out[idx] = vec<O, 4>(w);
}

#define INSTANTIATE(BITS, S, O, NAME)                                               \\
    template [[host_name(NAME)]] kernel void w4a8_decode<BITS, S, O>(               \\
        device vec<O, 4>*, device const uchar*, device const S*, device const float*, \\
        constant float*, constant float*, device const float*, constant long&,      \\
        constant long&, constant long&, constant long&, constant long&, uint);

INSTANTIATE(4, uchar, half, "w4_e4m3_decode_f16")
INSTANTIATE(4, float, half, "w4_f32_decode_f16")
INSTANTIATE(6, uchar, half, "w6_e4m3_decode_f16")
INSTANTIATE(6, float, half, "w6_f32_decode_f16")
INSTANTIATE(4, uchar, float, "w4_e4m3_decode_f32")
INSTANTIATE(4, float, float, "w4_f32_decode_f32")
INSTANTIATE(6, uchar, float, "w6_e4m3_decode_f32")
INSTANTIATE(6, float, float, "w6_f32_decode_f32")
"""

_lib = None
_tables: dict[tuple, torch.Tensor] = {}


def _shaders():
    global _lib
    if _lib is None:
        _lib = torch.mps.compile_shader(_SHADERS)
    return _lib


def _table(name: str, device: torch.device, build) -> torch.Tensor:
    key = (name, str(device))
    if key not in _tables:
        _tables[key] = build().to(device=device, dtype=torch.float32).contiguous()
    return _tables[key]


def _e4m3_lut(device: torch.device) -> torch.Tensor:
    """fp8-e4m3fn byte -> float32, taken from torch's own conversion so it cannot drift."""
    return _table(
        "e4m3", device, lambda: torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    )


def _uniform_levels(bits: int, device: torch.device) -> torch.Tensor:
    """Codebook-free levels: code - zero point, exactly what eager subtracts."""
    zero = 1 << (bits - 1)
    return _table(f"uniform{bits}", device, lambda: torch.arange(1 << bits, dtype=torch.float32) - zero)


def _decode_w4a8(
    qdata: torch.Tensor,
    s_rel: torch.Tensor,
    s_channel: torch.Tensor,
    levels: torch.Tensor,
    correction: torch.Tensor | None,
    k: int,
    bits: int,
    group_size: int,
    row0: int,
    rows: int,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Rows [row0, row0 + rows) of the rotated-basis weight, as [rows, K] fp16 or fp32."""
    out = torch.empty((rows, k), dtype=out_dtype, device=qdata.device)
    fp8 = s_rel.dtype == torch.float8_e4m3fn
    suffix = {torch.float16: "f16", torch.float32: "f32"}[out_dtype]
    kernel = getattr(_shaders(), f"w{bits}_{'e4m3' if fp8 else 'f32'}_decode_{suffix}")
    kernel(
        out,
        qdata,
        s_rel.view(torch.uint8) if fp8 else s_rel,  # MPS can store fp8 but not convert it
        s_channel,
        levels,
        _e4m3_lut(qdata.device),
        correction if correction is not None else s_channel,  # unread placeholder
        k,
        qdata.shape[0],
        group_size,
        row0,
        int(correction is not None),
        threads=rows * k // 4,
    )
    return out


def _kernel_operands(
    device: torch.device,
    qdata: torch.Tensor,
    s_rel: torch.Tensor,
    s_channel: torch.Tensor,
    codebook: torch.Tensor | None,
    correction: torch.Tensor | None,
    bits: int,
):
    """Packed operands on ``device`` in the layout the decode kernel reads. fp8 group scales
    stay fp8 (the kernel reads their bytes); fp32 scales stay fp32."""
    qdata = qdata.to(device=device).contiguous()
    s_rel = s_rel.to(device=device)
    if s_rel.dtype != torch.float8_e4m3fn:
        s_rel = s_rel.float()
    s_rel = s_rel.contiguous()
    s_channel = s_channel.to(device=device, dtype=torch.float32).contiguous()
    if codebook is not None:
        levels = codebook.to(device=device, dtype=torch.float32).contiguous()
    else:
        levels = _uniform_levels(bits, device)
    if correction is not None:
        correction = correction.to(device=device, dtype=torch.float32).contiguous()
    return qdata, s_rel, s_channel, levels, correction


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
    """W4A8/W6A8-weight linear, computed as fp16 GEMM over on-the-fly decoded weights."""
    n, k, bits = validate_w4a8_operands(
        qdata, s_rel, s_channel, codebook, correction, group_size, convrot_groupsize
    )
    if x.shape[-1] != k:
        raise ValueError(f"Input K={x.shape[-1]} does not match qdata K={k}")
    device = x.device
    qdata, s_rel, s_channel, levels, correction = _kernel_operands(
        device, qdata, s_rel, s_channel, codebook, correction, bits
    )

    xc, row_scale, cdtype = _to_compute(x)
    h = _build_hadamard(convrot_groupsize, device=device, dtype=cdtype)
    xc = _rotate_activation(xc, h, convrot_groupsize)

    def decode(row0: int, rows: int) -> torch.Tensor:
        w = _decode_w4a8(qdata, s_rel, s_channel, levels, correction, k, bits, group_size, row0, rows)
        return w if cdtype == torch.float16 else w.to(cdtype)

    rows_per_chunk = max(1, _MAX_DECODE_BYTES // (2 * k))
    if rows_per_chunk >= n:
        result = torch.nn.functional.linear(xc, decode(0, n))
    else:
        result = torch.cat(
            [
                torch.nn.functional.linear(xc, decode(i, min(rows_per_chunk, n - i)))
                for i in range(0, n, rows_per_chunk)
            ],
            dim=-1,
        )

    return _from_compute(result, row_scale, bias, out_dtype)


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
    """Decode W4A8/W6A8 storage into its physical (un-rotated) [N, K] weight, as eager does.

    Eager's contract, step for step: the rotated-basis weight in fp32 (int8 grid * s_channel
    [+ correction], which the fp32 kernel reproduces bit for bit), cast to ``output_dtype``,
    then rotated back in ``output_dtype``. The eager decode cannot run here because MPS has
    no fp8 arithmetic; the kernel reads the fp8 scale bytes through a lookup table instead.
    Row-chunked under the same budget as the linear decode, so the fp32 transient stays bounded.
    """
    n, k, bits = validate_w4a8_operands(
        qdata, s_rel, s_channel, codebook, correction, group_size, convrot_groupsize
    )
    device = qdata.device
    qdata, s_rel, s_channel, levels, correction = _kernel_operands(
        device, qdata, s_rel, s_channel, codebook, correction, bits
    )

    def decode(row0: int, rows: int) -> torch.Tensor:
        w = _decode_w4a8(
            qdata, s_rel, s_channel, levels, correction, k, bits, group_size, row0, rows, torch.float32
        ).to(output_dtype)
        return rotate_int8_convrot_weight(w, convrot_groupsize).to(output_dtype)

    rows_per_chunk = max(1, _MAX_DECODE_BYTES // (4 * k))
    if rows_per_chunk >= n:
        return decode(0, n)
    out = torch.empty((n, k), dtype=output_dtype, device=device)
    for r0 in range(0, n, rows_per_chunk):
        rows = min(rows_per_chunk, n - r0)
        out[r0 : r0 + rows] = decode(r0, rows)
    return out


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
):
    """Eager's quantizer, run on CPU, with the packed result moved back to ``weight.device``.

    MPS cannot produce fp8 group scales (no fp8 conversion), but it can hold fp8 tensors as
    storage, so the result keeps eager's dtypes exactly and a later requantize reads the same
    ``scale_dtype`` back. This is the LoRA-merge path (``requantize_from_float``), which runs
    one layer at a time; the CPU copy of the weight is dropped before the results move.
    """
    device = weight.device
    cpu_weight = weight.to("cpu")
    packed = eager_w4a8.quantize_w4a8_int8_weight(
        cpu_weight,
        group_size=group_size,
        convrot_groupsize=convrot_groupsize,
        symmetric=symmetric,
        scale_dtype=scale_dtype,
        codebook=codebook,
        codebook_tensor=None if codebook_tensor is None else codebook_tensor.to("cpu"),
        stochastic_rounding=stochastic_rounding,
        bits=bits,
        scale_search=scale_search,
    )
    del cpu_weight
    return tuple(None if t is None else t.to(device) for t in packed)
