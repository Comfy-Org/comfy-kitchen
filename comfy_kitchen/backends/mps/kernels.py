# SPDX-License-Identifier: Apache-2.0
"""Metal kernels for RoPE and AdaLN on Apple GPUs.

Inputs may be strided views as long as the last dim is contiguous: the leading
dims are addressed through a small shape/stride table (merged where every
operand allows, at most four after merging), so packed-qkv slices, (B, H, N, D)
layouts and broadcast operands need no copies. ``rope_unsupported`` and
``adaln_unsupported`` are the single source of truth for what the kernels take;
the registry's call rules use them, so anything else goes to eager.
"""
import math

import torch

_SRC = r"""
#include <metal_stdlib>
using namespace metal;

// Row addressing: meta[0] = number of (collapsed) leading dims n <= 4, meta[1..4] = their
// sizes (32-bit, so the div/mod stays cheap), then 4 strides per operand (64-bit).
inline long row_offset(constant long* meta, uint row, uint operand) {
    const int n = int(meta[0]);
    constant long* st = meta + 5 + operand * 4;
    if (n == 1) return long(row) * st[0];
    long off = 0;
    uint r = row;
    for (int d = n - 1; d >= 0; --d) {
        const uint size = uint(meta[1 + d]);
        off += long(r % size) * st[d];
        r /= size;
    }
    return off;
}

// RoPE over the leading rot dims of each row, optionally preceded by an RMSNorm
// over the full row. One simdgroup per row; a lane owns the same pairs in every
// pass, so running in place (out aliasing x) is safe.
// cfg: D, rot, norm, split_half, rows
template <typename T>
kernel void rope(device const T* x [[buffer(0)]],
                 device const float* freqs [[buffer(1)]],
                 device const float* w [[buffer(2)]],
                 device T* out [[buffer(3)]],
                 constant long* meta [[buffer(4)]],
                 constant int* cfg [[buffer(5)]],
                 constant float& eps [[buffer(6)]],
                 uint tg [[threadgroup_position_in_grid]],
                 uint sg [[simdgroup_index_in_threadgroup]],
                 uint sgs [[simdgroups_per_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]]) {
    const int D = cfg[0], rot = cfg[1], norm = cfg[2], split = cfg[3], rows = cfg[4];
    const uint row = tg * sgs + sg;
    if (row >= uint(rows)) return;
    device const T* xr = x + row_offset(meta, row, 0);
    device const float* fr = freqs + row_offset(meta, row, 1);
    device T* orow = out + row_offset(meta, row, 2);
    const int P = rot / 2;

    float r = 1.f;
    if (norm) {
        float ss = 0.f;
        for (int e = lane; e < D; e += 32) { float v = float(xr[e]); ss += v * v; }
        r = rsqrt(simd_sum(ss) / float(D) + eps);
    }
    for (int p = lane; p < P; p += 32) {
        const int ia = split ? p : 2 * p;
        const int ib = split ? p + P : 2 * p + 1;
        float a = float(xr[ia]), b = float(xr[ib]);
        if (norm) {
            // F.rms_norm on a T tensor rounds to T before the rotation
            a = float(T(a * r * w[ia]));
            b = float(T(b * r * w[ib]));
        }
        device const float* f = fr + p * 4;
        orow[ia] = T(f[0] * a + f[1] * b);
        orow[ib] = T(f[2] * a + f[3] * b);
    }
    for (int e = rot + lane; e < D; e += 32) {
        const float v = float(xr[e]);
        orow[e] = norm ? T(v * r * w[e]) : T(v);
    }
}

// LayerNorm (rms == 0) or RMSNorm (rms == 1) without affine, then * (1 + scale) + shift.
// One threadgroup per row. cfg: D, rms, rows, scale_last_stride, shift_last_stride
template <typename T>
kernel void adaln(device const T* x [[buffer(0)]],
                  device const T* scale [[buffer(1)]],
                  device const T* shift [[buffer(2)]],
                  device T* out [[buffer(3)]],
                  constant long* meta [[buffer(4)]],
                  constant int* cfg [[buffer(5)]],
                  constant float& eps [[buffer(6)]],
                  uint row [[threadgroup_position_in_grid]],
                  uint tid [[thread_position_in_threadgroup]],
                  uint tgsize [[threads_per_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]],
                  uint sgs [[simdgroups_per_threadgroup]]) {
    threadgroup float red[32];
    const int D = cfg[0], rms = cfg[1], rows = cfg[2], scl = cfg[3], shl = cfg[4];
    if (row >= uint(rows)) return;
    device const T* xr = x + row_offset(meta, row, 0);
    device const T* sc = scale + row_offset(meta, row, 1);
    device const T* sh = shift + row_offset(meta, row, 2);
    device T* orow = out + row_offset(meta, row, 3);

    float mean = 0.f;
    if (!rms) {
        float s = 0.f;
        for (int e = tid; e < D; e += tgsize) s += float(xr[e]);
        s = simd_sum(s);
        if (lane == 0) red[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float t = 0.f;
        for (uint i = 0; i < sgs; ++i) t += red[i];
        mean = t / float(D);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float q = 0.f;
    for (int e = tid; e < D; e += tgsize) { float v = float(xr[e]) - mean; q += v * v; }
    q = simd_sum(q);
    if (lane == 0) red[sg] = q;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float qt = 0.f;
    for (uint i = 0; i < sgs; ++i) qt += red[i];
    const float rstd = rsqrt(qt / float(D) + eps);
    for (int e = tid; e < D; e += tgsize) {
        const float n = float(T((float(xr[e]) - mean) * rstd));
        orow[e] = T(n * (1.f + float(sc[e * scl])) + float(sh[e * shl]));
    }
}

#define INSTANTIATE(T, N) \
template [[host_name("rope_" #N)]] kernel void rope<T>(device const T*, device const float*, device const float*, device T*, constant long*, constant int*, constant float&, uint, uint, uint, uint); \
template [[host_name("adaln_" #N)]] kernel void adaln<T>(device const T*, device const T*, device const T*, device T*, constant long*, constant int*, constant float&, uint, uint, uint, uint, uint, uint);
INSTANTIATE(bfloat, bf16)
INSTANTIATE(half, f16)
INSTANTIATE(float, f32)
"""

_NAMES = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}
_MAX_LEAD = 4
_MAX_ROPE_DIM = 1024
_ROWS_PER_GROUP = 8
_lib = None
_compile_error = None
_tables: dict = {}


def _kernels():
    global _lib
    if _lib is None:
        _lib = torch.mps.compile_shader(_SRC)
    return _lib


def _kernels_unavailable():
    """Why the Metal library cannot be built here, or None. A failure is cached."""
    global _compile_error
    if _lib is None and _compile_error is None:
        try:
            _kernels()
        except Exception as exc:
            _compile_error = f"Metal shader compilation failed: {exc}"
    return _compile_error


def _table(values, dtype, device):
    """Small constant table on the device, cached by content."""
    key = (dtype, tuple(values))
    t = _tables.get(key)
    if t is None:
        if len(_tables) > 4096:
            _tables.clear()
        t = torch.tensor(values, dtype=dtype, device=device)
        _tables[key] = t
    return t


def _lead(shape, *strided):
    """Row-addressing table for the leading dims, or None past four merged dims.

    Dims every operand steps through as one are merged; the table is
    [n, sizes(4), then 4 strides per operand].
    """
    dims = [(size, [st[i] for st in strided]) for i, size in enumerate(shape) if size != 1]
    merged = []
    for size, strides in dims:
        if merged:
            psize, pstrides = merged[-1]
            if all(ps == s * size for ps, s in zip(pstrides, strides, strict=True)):
                merged[-1] = (psize * size, strides)
                continue
        merged.append((size, strides))
    if not merged:
        merged = [(1, [0] * len(strided))]
    if len(merged) > _MAX_LEAD or any(size >= 2**31 for size, _ in merged):
        return None
    n = len(merged)
    table = [n] + [size for size, _ in merged] + [1] * (4 - n)
    for j in range(len(strided)):
        table += [st[j] for _, st in merged] + [0] * (4 - n)
    return table


def _trimmed_freqs(x, freqs):
    from comfy_kitchen._rope_utils import trim_rope_freqs

    freqs = trim_rope_freqs(x, freqs)
    if freqs.stride()[-3:] != (4, 2, 1):
        freqs = freqs.contiguous()
    return freqs


def rope_unsupported(x, freqs, weight=None, rot_dim=0):
    """Why the rope kernel cannot take this call, or None when it can."""
    if not isinstance(x, torch.Tensor) or not isinstance(freqs, torch.Tensor):
        return "x and freqs_cis must be tensors"
    if x.device.type != "mps" or freqs.device != x.device:
        return "tensors must be on the same MPS device"
    if (reason := _kernels_unavailable()) is not None:
        return reason
    if x.dtype not in _NAMES:
        return f"unsupported dtype {x.dtype}"
    if x.dim() < 2 or x.numel() == 0 or x.stride(-1) != 1:
        return "x needs a contiguous last dim and at least two dims"
    dim = x.shape[-1]
    rot = rot_dim or dim
    if dim % 2 or rot % 2 or rot > dim or dim > _MAX_ROPE_DIM:
        return f"unsupported head dim {dim} / rot_dim {rot_dim}"
    if freqs.dtype != torch.float32 or freqs.dim() < 3 or tuple(freqs.shape[-3:]) != (rot // 2, 2, 2):
        return "freqs_cis must be float32 [..., rot_dim // 2, 2, 2]"
    if weight is not None and (not isinstance(weight, torch.Tensor) or weight.numel() != dim
                               or weight.device != x.device):
        return "scale must have head_dim elements on the same device"
    f = _trimmed_freqs(x, freqs)
    try:
        f = f.expand(*x.shape[:-1], rot // 2, 2, 2)
    except RuntimeError:
        return "freqs_cis does not broadcast against x"
    if _lead(x.shape[:-1], x.stride()[:-1], f.stride()[:-3], x.stride()[:-1]) is None:
        return "more than four non-mergeable leading dims"
    return None


def rope(x, freqs, *, split_half, weight=None, eps=1e-6, rot_dim=0, inplace=False):
    """RoPE, optionally preceded by an RMSNorm over the head dim (weight given)."""
    dim = x.shape[-1]
    rot = rot_dim or dim
    f = _trimmed_freqs(x, freqs).expand(*x.shape[:-1], rot // 2, 2, 2)
    out = x if inplace else torch.empty(x.shape, dtype=x.dtype, device=x.device)
    lead = x.shape[:-1]
    rows = math.prod(lead)
    table = _lead(lead, x.stride()[:-1], f.stride()[:-3], out.stride()[:-1])
    if table is None:
        raise ValueError("rope: unsupported layout (check rope_unsupported first)")
    w = (weight.reshape(-1).to(torch.float32).contiguous() if weight is not None
         else _table([0.0], torch.float32, x.device))
    meta = _table(table, torch.int64, x.device)
    cfg = _table([dim, rot, int(weight is not None), int(split_half), rows], torch.int32, x.device)
    groups = (rows + _ROWS_PER_GROUP - 1) // _ROWS_PER_GROUP
    getattr(_kernels(), "rope_" + _NAMES[x.dtype])(
        x, f, w, out, meta, cfg, float(eps),
        threads=groups * _ROWS_PER_GROUP * 32, group_size=_ROWS_PER_GROUP * 32)
    return out


def _expanded(t, shape):
    try:
        return t.expand(shape)
    except RuntimeError:
        return None


def adaln_unsupported(x, scale, shift):
    """Why the adaln kernel cannot take this call, or None when it can."""
    tensors = (x, scale, shift)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        return "x, scale and shift must be tensors"
    if any(t.device.type != "mps" or t.device != x.device for t in tensors):
        return "tensors must be on the same MPS device"
    if (reason := _kernels_unavailable()) is not None:
        return reason
    if x.dtype not in _NAMES or not (x.dtype == scale.dtype == shift.dtype):
        return "x, scale and shift must share one float dtype"
    if x.dim() < 1 or x.numel() == 0 or x.stride(-1) != 1:
        return "x needs a contiguous last dim"
    sc, sh = _expanded(scale, x.shape), _expanded(shift, x.shape)
    if sc is None or sh is None:
        return "scale / shift do not broadcast against x"
    if sc.stride(-1) not in (0, 1) or sh.stride(-1) not in (0, 1):
        return "scale / shift need a contiguous or broadcast last dim"
    if _lead(x.shape[:-1], x.stride()[:-1], sc.stride()[:-1], sh.stride()[:-1], x.stride()[:-1]) is None:
        return "more than four non-mergeable leading dims"
    return None


def adaln(x, scale, shift, eps=1e-6, rms=False):
    """LayerNorm (or RMSNorm) without affine, then * (1 + scale) + shift."""
    dim = x.shape[-1]
    sc, sh = scale.expand(x.shape), shift.expand(x.shape)
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    lead = x.shape[:-1]
    rows = math.prod(lead)
    table = _lead(lead, x.stride()[:-1], sc.stride()[:-1], sh.stride()[:-1], out.stride()[:-1])
    if table is None:
        raise ValueError("adaln: unsupported layout (check adaln_unsupported first)")
    meta = _table(table, torch.int64, x.device)
    cfg = _table([dim, int(rms), rows, sc.stride(-1), sh.stride(-1)], torch.int32, x.device)
    tg = min(256, (dim + 31) // 32 * 32)
    getattr(_kernels(), "adaln_" + _NAMES[x.dtype])(
        x, sc, sh, out, meta, cfg, float(eps), threads=rows * tg, group_size=tg)
    return out
