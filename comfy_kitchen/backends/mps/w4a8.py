# SPDX-License-Identifier: Apache-2.0
"""Weight-only W4A8 / W6A8 linear layers for Apple GPUs.

The packed weight stays resident. Each call decodes it with one Metal pass (code unpack,
level lookup, group scale, int8-grid rounding, channel scale, optional asymmetric
correction, then an exact power-of-two row normalization) and runs the stock GEMM through
the same range-safe prologue/epilogue as the INT8 path (see linear.py). Activations are
rotated online instead of being quantized, so the result is the float linear over the
decoded weight.

MPS stores fp8 tensors but cannot convert them, so nothing here does fp8 arithmetic:
dequantize reads the fp8 scale bytes in the kernel, and quantize (the LoRA requantize
path) runs eager's quantizer on CPU and moves the packed result back.
"""
from __future__ import annotations

import torch

from comfy_kitchen.backends.eager import w4a8_int8 as eager_w4a8
from comfy_kitchen.backends.eager.quantization import rotate_int8_convrot_weight
from comfy_kitchen.backends.eager.w4a8_int8 import validate_w4a8_operands

from . import linear as _linear
from .linear import _MAX_DECODE_BYTES, METAL_COMMON

_SHADERS = METAL_COMMON + r"""
// Bit-exact with the eager decode: the asymmetric path's multiply and add must round
// separately, as the two torch ops do, not fuse into one fma.
#pragma METAL fp contract(off)

inline float group_scale(device const uchar* s, constant float* e4m3_lut, ulong i) {
    return e4m3_lut[s[i]];
}

inline float group_scale(device const float* s, constant float*, ulong i) {
    return s[i];
}

// Global element indices, all 64-bit: N * K passes 2^32 on real layers (review finding 5).
inline ulong w4_row_offset(ulong row, ulong K, ulong bits) { return row * (K * bits / 8); }
inline ulong w4_scale_index(ulong row, ulong K, ulong group_size, ulong group) {
    return row * (K / group_size) + group;
}
inline ulong w4_corr_index(ulong group, ulong N, ulong row) { return group * N + row; }

struct W4Args {
    long K;
    long N;
    long group_size;
    long has_correction;
};

// Four consecutive columns of one row (group sizes are multiples of 4), as eager computes
// them in fp32: w = int8(clamp(rint(levels[code] * s_rel), -127, 127)) * s_channel
// [+ correction]. Row layout: K/2 bytes of nibbles (even col = low nibble); at 6 bits a
// K/4-byte plane of top-2-bit pairs follows (col c in byte c/4 at bit 2*(c%4)).
// shift f divides the row by 2^f. It is applied to s_channel and the correction before
// the multiply and add, on raw bits, so a subnormal or huge channel scale neither flushes
// nor overflows first; in range the result is eager's value scaled exactly.
template <int BITS, typename S>
inline float4 decode_quad(device const uchar* q, device const S* s_rel, device const float* s_channel,
                          constant float* levels, constant float* e4m3_lut,
                          device const float* correction, W4Args a, ulong row, ulong col, int f) {
    const ulong K = ulong(a.K);
    device const uchar* qrow = q + w4_row_offset(row, K, BITS);
    const uchar b0 = qrow[col / 2];
    const uchar b1 = qrow[col / 2 + 1];
    uint4 codes = uint4(b0 & 0xF, b0 >> 4, b1 & 0xF, b1 >> 4);
    if (BITS == 6) {
        const uint hi = qrow[K / 2 + col / 4];
        codes |= ((uint4(hi) >> uint4(0, 2, 4, 6)) & 0x3) << 4;
    }
    const ulong group = col / ulong(a.group_size);
    const float s = group_scale(s_rel, e4m3_lut, w4_scale_index(row, K, ulong(a.group_size), group));
    const float sc = as_type<float>(ldexp_bits(as_type<uint>(s_channel[row]), -f));
    float4 w;
    for (int j = 0; j < 4; ++j) {
        // Through int: torch's int8 cast drops the sign of a rounded -0.
        const int grid = int(clamp(rint(levels[codes[j]] * s), -127.0f, 127.0f));
        w[j] = float(grid) * sc;
    }
    if (a.has_correction) {
        w += as_type<float>(ldexp_bits(as_type<uint>(correction[w4_corr_index(group, ulong(a.N), row)]), -f));
    }
    return w;
}

// Exponent f with |w| <= 2^f for every w of the row: |grid * s_channel| < 2^(E(s_channel) + 7)
// and |correction| < 2^E(max |correction|), so one more bit covers their sum.
inline int row_shift(float sc, device const float* corr_max, W4Args a, ulong row) {
    int f;
    split_pow2(sc, 7, f);
    if (a.has_correction) {
        const uint cm = as_type<uint>(corr_max[row]) & 0x7FFFFFFFu;
        if (cm != 0u && cm < 0x7F800000u) {
            f = (as_type<uint>(sc) & 0x7FFFFFFFu) == 0u ? ceil_exp(cm) + 1 : max(f, ceil_exp(cm)) + 1;
        }
    }
    return f;
}

// Rows [row0, row0 + rows) into out [rows, K]; grid (K/4, rows). With NORM each row is
// divided by 2^ch_exp[row] (exact), so |out| <= 1. O is half or float.
template <int BITS, typename S, typename O, bool NORM>
kernel void w4a8_decode(device vec<O, 4>* out,
                        device int* ch_exp,
                        device const uchar* q,
                        device const S* s_rel,
                        device const float* s_channel,
                        constant float* levels,
                        constant float* e4m3_lut,
                        device const float* correction,
                        device const float* corr_max,
                        constant long& K,
                        constant long& N,
                        constant long& group_size,
                        constant long& has_correction,
                        constant long& row0,
                        uint2 pos [[thread_position_in_grid]]) {
    const W4Args a = {K, N, group_size, has_correction};
    const ulong r = pos.y, row = ulong(row0) + r, col = ulong(pos.x) * 4;
    const int f = NORM ? row_shift(s_channel[row], corr_max, a, row) : 0;
    const float4 w = decode_quad<BITS>(q, s_rel, s_channel, levels, e4m3_lut, correction, a, row, col, f);
    if (NORM && col == 0) {
        ch_exp[row] = f;
    }
    out[r * (ulong(a.K) / 4) + pos.x] = vec<O, 4>(w);
}

// Normalized rows [row0, row0 + rows) written transposed, out [K, rows] with row stride
// ld >= rows, through a tile of 32 rows x 128 columns in threadgroup memory. Grid (ceil(K/128) * 32, ceil(rows/32) * 8),
// threadgroups of 32x8.
template <int BITS, typename S, typename O>
kernel void w4a8_decode_t(device O* out,
                          device int* ch_exp,
                          device const uchar* q,
                          device const S* s_rel,
                          device const float* s_channel,
                          constant float* levels,
                          constant float* e4m3_lut,
                          device const float* correction,
                          device const float* corr_max,
                          constant long& K_,
                          constant long& N,
                          constant long& group_size,
                          constant long& has_correction,
                          constant long& row0,
                          constant long& rows,
                          constant long& ld,
                          uint2 tg [[threadgroup_position_in_grid]],
                          uint2 lid [[thread_position_in_threadgroup]]) {
    const W4Args a = {K_, N, group_size, has_correction};
    threadgroup O tile[128][36];  // [k][r], padded
    const ulong K = ulong(a.K), R = ulong(rows);
    const ulong col = (ulong(tg.x) * 32 + lid.x) * 4;
    for (uint i = 0; i < 4u; ++i) {
        const uint rl = lid.y + 8u * i;
        const ulong r = ulong(tg.y) * 32 + rl;
        if (col < K && r < R) {
            const ulong row = ulong(row0) + r;
            const int f = row_shift(s_channel[row], corr_max, a, row);
            const float4 w = decode_quad<BITS>(q, s_rel, s_channel, levels, e4m3_lut, correction, a, row, col, f);
            for (uint j = 0; j < 4u; ++j) {
                tile[lid.x * 4u + j][rl] = O(w[j]);
            }
            if (col == 0) {
                ch_exp[row] = f;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint tid = lid.y * 32u + lid.x;
    const uint rq = (tid & 7u) * 4u;
    const ulong r0 = ulong(tg.y) * 32 + rq;
    for (uint p = 0; p < 4u; ++p) {
        const uint kl = (tid >> 3) + 32u * p;
        const ulong k = ulong(tg.x) * 128 + kl;
        if (k >= K || r0 >= R) {
            continue;
        }
        device O* o = out + k * ulong(ld) + r0;
        if (r0 + 4 <= R && R % 4 == 0 && ld % 4 == 0) {
            *(device vec<O, 4>*)o = vec<O, 4>(tile[kl][rq], tile[kl][rq + 1], tile[kl][rq + 2], tile[kl][rq + 3]);
        } else {
            for (uint j = 0; j < 4u && r0 + j < R; ++j) {
                o[j] = tile[kl][rq + j];
            }
        }
    }
}

#define W4_ROWS(BITS, S, O, NORM, NAME)                                                     \
    template [[host_name(NAME)]] kernel void w4a8_decode<BITS, S, O, NORM>(                 \
        device vec<O, 4>*, device int*, device const uchar*, device const S*,               \
        device const float*, constant float*, constant float*, device const float*,         \
        device const float*, constant long&, constant long&, constant long&, constant long&,  \
        constant long&, uint2);
#define W4_T(BITS, S, O, NAME)                                                              \
    template [[host_name(NAME)]] kernel void w4a8_decode_t<BITS, S, O>(                     \
        device O*, device int*, device const uchar*, device const S*,                       \
        device const float*, constant float*, constant float*, device const float*,         \
        device const float*, constant long&, constant long&, constant long&, constant long&,  \
        constant long&, constant long&, constant long&, uint2, uint2);
#define W4_ALL(BITS, S, NAME)                                                                   \
    W4_ROWS(BITS, S, float, false, "w" #BITS "_" NAME "_decode_f32")                           \
    W4_ROWS(BITS, S, half, true, "w" #BITS "_" NAME "_decode_norm_f16")                        \
    W4_ROWS(BITS, S, float, true, "w" #BITS "_" NAME "_decode_norm_f32")                       \
    W4_T(BITS, S, half, "w" #BITS "_" NAME "_decode_t_f16")                                    \
    W4_T(BITS, S, float, "w" #BITS "_" NAME "_decode_t_f32")
W4_ALL(4, uchar, "e4m3")
W4_ALL(4, float, "f32")
W4_ALL(6, uchar, "e4m3")
W4_ALL(6, float, "f32")
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
    out_dtype: torch.dtype = torch.float32,
    ch_exp: torch.Tensor | None = None,
    corr_max: torch.Tensor | None = None,
    transposed: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rows [row0, row0 + rows) of the rotated-basis weight.

    Without ``ch_exp``: eager's fp32 decode, value for value, as [rows, K].
    With ``ch_exp`` (int32 [N], filled here): each row divided by 2^ch_exp[row] so |w'| <= 1,
    in ``out_dtype`` (fp16 or fp32), as [rows, K] or, when ``transposed``, [K, rows]. A
    correction needs ``corr_max``, the max |correction| of each row. ``out`` (normalized
    decode only, from ``linear.decode_buffer``) receives the rows in place.
    """
    normalize = ch_exp is not None
    if not normalize and (out_dtype != torch.float32 or transposed):
        raise ValueError("the unnormalized decode is fp32 [rows, K] only")
    if normalize and correction is not None and corr_max is None:
        raise ValueError("normalized decode with a correction needs corr_max")
    fp8 = s_rel.dtype == torch.float8_e4m3fn
    suffix = {torch.float16: "f16", torch.float32: "f32"}[out_dtype]
    lib = _shaders()
    name = f"w{bits}_{'e4m3' if fp8 else 'f32'}_decode"
    args = (
        ch_exp if normalize else s_channel,  # unwritten placeholder
        qdata,
        s_rel.view(torch.uint8) if fp8 else s_rel,  # MPS can store fp8 but not convert it
        s_channel,
        levels,
        _e4m3_lut(qdata.device),
        correction if correction is not None else s_channel,  # unread placeholder
        corr_max if corr_max is not None else s_channel,  # unread placeholder
        k,
        qdata.shape[0],
        group_size,
        int(correction is not None),
        row0,
    )
    if out is None:
        shape = (k, rows) if transposed else (rows, k)
        out = torch.empty(shape, dtype=out_dtype, device=qdata.device)
    elif not normalize:
        raise ValueError("out is for the normalized decode")
    if transposed:
        getattr(lib, f"{name}_t_{suffix}")(
            out, *args, rows, out.stride(0),
            threads=(-(-k // 128) * 32, -(-rows // 32) * 8), group_size=(32, 8),
        )
    else:
        getattr(lib, f"{name}{'_norm' if normalize else ''}_{suffix}")(
            out, *args, threads=(k // 4, rows), group_size=(min(k // 4, 256), 1),
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
    lora_down: torch.Tensor | None = None,
    lora_up: torch.Tensor | None = None,
    lora_scale: float = 1.0,
) -> torch.Tensor:
    """W4A8/W6A8-weight linear, computed as fp16 GEMM over on-the-fly decoded weights.

    Bias follows eager: the asymmetric path adds it inside the linear (cast to x's dtype,
    before rounding); the symmetric path rounds to ``out_dtype`` first, as int8_linear does.
    ``lora_down`` [R, K], ``lora_up`` [N, R] and ``lora_scale`` add an unmerged LoRA, as in
    int8_linear; ``lora_down`` is in the physical (un-rotated) basis.
    """
    n, k, bits = validate_w4a8_operands(
        qdata, s_rel, s_channel, codebook, correction, group_size, convrot_groupsize
    )
    if x.shape[-1] != k:
        raise ValueError(f"Input K={x.shape[-1]} does not match qdata K={k}")
    device = x.device
    qdata, s_rel, s_channel, levels, correction = _kernel_operands(
        device, qdata, s_rel, s_channel, codebook, correction, bits
    )
    corr_max = None if correction is None else correction.abs().amax(dim=0).contiguous()
    lora = _linear.lora_operands(lora_down, lora_up, lora_scale, n, k, device)

    def decode(row0, rows, cdtype, transposed, ch_exp, out=None):
        return _decode_w4a8(
            qdata, s_rel, s_channel, levels, correction, k, bits, group_size, row0, rows,
            cdtype, ch_exp, corr_max, transposed, out,
        )

    return _linear.weight_only_linear(
        x, n, k, decode, bias=bias, out_dtype=out_dtype, group=convrot_groupsize,
        bias_in_linear=correction is not None,  # eager: asymmetric runs F.linear, symmetric int8_linear
        lora=lora,
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
