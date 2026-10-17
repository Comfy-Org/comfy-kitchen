// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Fused RMSNorm + RoPE. Same operand layout as ops/apply_rope.hip. The norm
// reduces over head_dim, the last axis in both the BHND and BNHD layouts, so the
// kernel never has to know which of dim1/dim2 is the sequence.
//
// One block per row, as in ops/adaln.hip, but sized to one row rather than
// adaln's 256 threads: rows are short and numerous, so a wider block would idle.
// The row is re-read for the rotation rather than held in registers, which keeps
// head_dim unbounded.
//
// The normalized value is rounded to x's dtype before the rotation reads it back.
// That is the unfused contract, which every other backend matches.
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "../hadamard.h"   // load_in / dtype codes
#include "../rope_math.h"  // rope_combine / rope_store / round_to

namespace comfy::hip_backend {

constexpr int kRmsRopeThreads = 64;

// q_scale and k_scale need not share x's dtype, so they carry their own code.
// k is optional: rms_rope1 passes nullptr for it.
template <typename T>
__global__ __launch_bounds__(kRmsRopeThreads) void rms_rope_kernel(
    const void* __restrict__ q, const void* __restrict__ k,
    const void* __restrict__ freqs, const void* __restrict__ q_scale,
    const void* __restrict__ k_scale, T* __restrict__ q_out, T* __restrict__ k_out,
    int64_t dim1, int64_t dim2, int head_dim, int rot_dim,
    int64_t fb, int64_t fd1, int64_t fd2,
    int64_t sx_b, int64_t sx_d1, int64_t sx_d2, int64_t sx_d,
    int64_t so_b, int64_t so_d1, int64_t so_d2, int64_t so_d,
    int64_t sf_b, int64_t sf_d1, int64_t sf_d2, int64_t sf_d, int64_t sf_rot,
    int64_t sf_pair,
    int x_code, int f_code, int s_code, float epsilon, bool split_half) {

    __shared__ float red_q[kRmsRopeThreads];
    __shared__ float red_k[kRmsRopeThreads];

    const int t = threadIdx.x;
    const int64_t row = blockIdx.x;
    const int64_t d2 = row % dim2;
    const int64_t d1 = (row / dim2) % dim1;
    const int64_t b = row / (dim2 * dim1);

    // Inputs keep their own strides so a qkv slice is read in place; the outputs
    // are freshly allocated, so they carry their own set.
    const int64_t xoff = b * sx_b + d1 * sx_d1 + d2 * sx_d2;
    const int64_t ooff = b * so_b + d1 * so_d1 + d2 * so_d2;
    const bool has_k = k != nullptr;

    float sq_q = 0.0f;
    float sq_k = 0.0f;
    for (int i = t; i < head_dim; i += kRmsRopeThreads) {
        const float v = load_in(q, xoff + i * sx_d, x_code);
        sq_q += v * v;
        if (has_k) {
            const float w = load_in(k, xoff + i * sx_d, x_code);
            sq_k += w * w;
        }
    }

    red_q[t] = sq_q;
    red_k[t] = sq_k;
    __syncthreads();
    for (int s = kRmsRopeThreads / 2; s > 0; s >>= 1) {
        if (t < s) {
            red_q[t] += red_q[t + s];
            red_k[t] += red_k[t + s];
        }
        __syncthreads();
    }

    const float inv_d = 1.0f / static_cast<float>(head_dim);
    const float rrms_q = rsqrtf(red_q[0] * inv_d + epsilon);
    const float rrms_k = has_k ? rsqrtf(red_k[0] * inv_d + epsilon) : 0.0f;

    const int64_t foff_row = (fb == 1 ? 0 : b) * sf_b + (fd1 == 1 ? 0 : d1) * sf_d1 +
                             (fd2 == 1 ? 0 : d2) * sf_d2;

    // Rotation covers the first rot_dim dims, in split-half pairs (p, p + rot_dim/2);
    // the reduction above always spans the full head_dim.
    const int n_pairs = rot_dim / 2;
    for (int p = t; p < n_pairs; p += kRmsRopeThreads) {
        const int64_t ia = split_half ? p : 2 * p;
        const int64_t ib = split_half ? p + n_pairs : 2 * p + 1;
        const int64_t ia_in = xoff + ia * sx_d;
        const int64_t ib_in = xoff + ib * sx_d;
        const int64_t ia_out = ooff + ia * so_d;
        const int64_t ib_out = ooff + ib * so_d;

        const int64_t foff = foff_row + static_cast<int64_t>(p) * sf_d;
        const float f00 = load_in(freqs, foff, f_code);
        const float f01 = load_in(freqs, foff + sf_pair, f_code);
        const float f10 = load_in(freqs, foff + sf_rot, f_code);
        const float f11 = load_in(freqs, foff + sf_rot + sf_pair, f_code);

        const float sa = load_in(q_scale, ia, s_code);
        const float sb = load_in(q_scale, ib, s_code);
        const float qa = round_to<T>(load_in(q, ia_in, x_code) * rrms_q * sa);
        const float qb = round_to<T>(load_in(q, ib_in, x_code) * rrms_q * sb);
        rope_store<T>(q_out, ia_out, rope_combine(f00, qa, f01, qb, f_code));
        rope_store<T>(q_out, ib_out, rope_combine_b(f10, qa, f11, qb, f_code, split_half));

        if (has_k) {
            const float ta = load_in(k_scale, ia, s_code);
            const float tb = load_in(k_scale, ib, s_code);
            const float ka = round_to<T>(load_in(k, ia_in, x_code) * rrms_k * ta);
            const float kb = round_to<T>(load_in(k, ib_in, x_code) * rrms_k * tb);
            rope_store<T>(k_out, ia_out, rope_combine(f00, ka, f01, kb, f_code));
            rope_store<T>(k_out, ib_out, rope_combine_b(f10, ka, f11, kb, f_code, split_half));
        }
    }

    // Dims past rot_dim are normalized and scaled but never rotated. The store
    // rounds to T, which is the single rounding eager's rms_norm applies. Empty
    // when rot_dim == head_dim.
    for (int d = rot_dim + t; d < head_dim; d += kRmsRopeThreads) {
        const int64_t in = xoff + d * sx_d;
        const int64_t out = ooff + d * so_d;
        rope_store<T>(q_out, out, load_in(q, in, x_code) * rrms_q * load_in(q_scale, d, s_code));
        if (has_k) {
            rope_store<T>(k_out, out, load_in(k, in, x_code) * rrms_k * load_in(k_scale, d, s_code));
        }
    }
}

// Vectorized split-half path: half/bf16 x with a contiguous head_dim of 64, 128 or 256,
// full rotation and contiguous fp32 [.., rot/2, 2, 2] freqs. TPR = head_dim / 8 lanes
// share a row: each lane holds 4 pairs (p, p + head_dim/2) of q and k in registers, so
// both are read once, and the norm reduces with shuffles. The per-element math and
// rounding are rms_rope_kernel's; only the order of the fp32 square sum differs.
// q_out may alias q (the in-place variant): every lane loads its elements before it
// stores them, and stores no others.
constexpr int kRmsRopeVecThreads = 256;

template <typename T>
__forceinline__ __device__ void load4(const void* p, int64_t i, float (&o)[4]) {
    const uint2 pack = *reinterpret_cast<const uint2*>(static_cast<const T*>(p) + i);
    const T* e = reinterpret_cast<const T*>(&pack);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if constexpr (std::is_same_v<T, __half>) {
            o[j] = __half2float(e[j]);
        } else {
            o[j] = static_cast<float>(e[j]);
        }
    }
}

template <typename T>
__forceinline__ __device__ void store4(T* p, int64_t i, const float (&v)[4]) {
    T e[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if constexpr (std::is_same_v<T, __half>) {
            e[j] = __float2half(v[j]);
        } else {
            e[j] = static_cast<__bf16>(v[j]);
        }
    }
    *reinterpret_cast<uint2*>(p + i) = *reinterpret_cast<const uint2*>(e);
}

template <typename T, int TPR>
__global__ __launch_bounds__(kRmsRopeVecThreads) void rms_rope_split_half_vec_kernel(
    const void* q, const void* k, const float* __restrict__ freqs,
    const void* __restrict__ q_scale, const void* __restrict__ k_scale, T* q_out, T* k_out,
    int64_t rows, int64_t dim1, int64_t dim2, int64_t fb, int64_t fd1, int64_t fd2,
    int64_t sx_b, int64_t sx_d1, int64_t sx_d2, int64_t so_b, int64_t so_d1, int64_t so_d2,
    int64_t sf_b, int64_t sf_d1, int64_t sf_d2, int s_code, float epsilon) {

    constexpr int kHalf = TPR * 4;
    const int lane = threadIdx.x % TPR;
    const int64_t row =
        static_cast<int64_t>(blockIdx.x) * (kRmsRopeVecThreads / TPR) + threadIdx.x / TPR;
    if (row >= rows) return;  // a row's lanes are contiguous, so they leave together

    const int64_t d2 = row % dim2;
    const int64_t d1 = (row / dim2) % dim1;
    const int64_t b = row / (dim2 * dim1);
    const int64_t xa = b * sx_b + d1 * sx_d1 + d2 * sx_d2 + 4 * lane;
    const int64_t oa = b * so_b + d1 * so_d1 + d2 * so_d2 + 4 * lane;
    const bool has_k = k != nullptr;

    float qa[4], qb[4], ka[4] = {}, kb[4] = {};
    load4<T>(q, xa, qa);
    load4<T>(q, xa + kHalf, qb);
    if (has_k) {
        load4<T>(k, xa, ka);
        load4<T>(k, xa + kHalf, kb);
    }

    float sq_q = 0.0f, sq_k = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        sq_q += qa[j] * qa[j] + qb[j] * qb[j];
        sq_k += ka[j] * ka[j] + kb[j] * kb[j];
    }
#pragma unroll
    for (int off = TPR / 2; off > 0; off >>= 1) {
        sq_q += __shfl_xor(sq_q, off, TPR);
        sq_k += __shfl_xor(sq_k, off, TPR);
    }
    constexpr float inv_d = 1.0f / static_cast<float>(TPR * 8);
    const float rrms_q = rsqrtf(sq_q * inv_d + epsilon);
    const float rrms_k = rsqrtf(sq_k * inv_d + epsilon);

    // One float4 per pair: (f00, f01, f10, f11).
    const float4* f = reinterpret_cast<const float4*>(
        freqs + (fb == 1 ? 0 : b) * sf_b + (fd1 == 1 ? 0 : d1) * sf_d1 +
        (fd2 == 1 ? 0 : d2) * sf_d2) + 4 * lane;

    float out_a[4], out_b[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float4 fr = f[j];
        const int ia = 4 * lane + j;
        const float na = round_to<T>(qa[j] * rrms_q * load_in(q_scale, ia, s_code));
        const float nb = round_to<T>(qb[j] * rrms_q * load_in(q_scale, ia + kHalf, s_code));
        out_a[j] = rope_combine(fr.x, na, fr.y, nb, 0);
        out_b[j] = rope_combine_b(fr.z, na, fr.w, nb, 0, true);
    }
    store4<T>(q_out, oa, out_a);
    store4<T>(q_out, oa + kHalf, out_b);
    if (has_k) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float4 fr = f[j];
            const int ia = 4 * lane + j;
            const float na = round_to<T>(ka[j] * rrms_k * load_in(k_scale, ia, s_code));
            const float nb = round_to<T>(kb[j] * rrms_k * load_in(k_scale, ia + kHalf, s_code));
            out_a[j] = rope_combine(fr.x, na, fr.y, nb, 0);
            out_b[j] = rope_combine_b(fr.z, na, fr.w, nb, 0, true);
        }
        store4<T>(k_out, oa, out_a);
        store4<T>(k_out, oa + kHalf, out_b);
    }
}

}  // namespace comfy::hip_backend

extern "C" void launch_rms_rope_kernel(
    const void* q, const void* k, const void* freqs, const void* q_scale, const void* k_scale,
    void* q_out, void* k_out,
    int64_t batch, int64_t dim1, int64_t dim2, int64_t head_dim, int64_t rot_dim,
    int64_t fb, int64_t fd1, int64_t fd2,
    int64_t sx_b, int64_t sx_d1, int64_t sx_d2, int64_t sx_d,
    int64_t so_b, int64_t so_d1, int64_t so_d2, int64_t so_d,
    int64_t sf_b, int64_t sf_d1, int64_t sf_d2, int64_t sf_d, int64_t sf_rot, int64_t sf_pair,
    int x_code, int f_code, int s_code, float epsilon, bool split_half, hipStream_t stream) {

    using namespace comfy::hip_backend;

    const int64_t rows = batch * dim1 * dim2;
    // A zero-block launch is hipErrorInvalidConfiguration, and head_dim == 0 would
    // divide by zero in the reduction.
    if (rows == 0 || head_dim == 0) return;
    // One block per row, and gridDim.x is 32-bit.
    if (rows > 0xffffffffLL) {
        throw std::runtime_error("rms_rope: too many rows for a single grid");
    }
    // 0 rotates everything; anything else is a partial-rotary prefix.
    const int64_t rot = rot_dim > 0 ? rot_dim : head_dim;
    if (rot % 2 != 0 || rot > head_dim) {
        throw std::runtime_error("rms_rope: rot_dim must be even and <= head_dim");
    }

    // 8-byte lane chunks of x and 16-byte freqs pairs; a stride only has to keep that
    // alignment along a dimension that is actually indexed.
    const auto stride_ok = [](int64_t size, int64_t stride, int64_t mult) {
        return size == 1 || stride % mult == 0;
    };
    const auto aligned = [](const void* p, uintptr_t a) {
        return p == nullptr || reinterpret_cast<uintptr_t>(p) % a == 0;
    };
    if ((x_code == 1 || x_code == 2) && f_code == 0 && split_half && rot == head_dim &&
        (head_dim == 64 || head_dim == 128 || head_dim == 256) && sx_d == 1 && so_d == 1 &&
        sf_d == 4 && sf_rot == 2 && sf_pair == 1 && stride_ok(batch, sx_b, 4) &&
        stride_ok(dim1, sx_d1, 4) && stride_ok(dim2, sx_d2, 4) && stride_ok(batch, so_b, 4) &&
        stride_ok(dim1, so_d1, 4) && stride_ok(dim2, so_d2, 4) && stride_ok(fb, sf_b, 4) &&
        stride_ok(fd1, sf_d1, 4) && stride_ok(fd2, sf_d2, 4) && aligned(q, 8) && aligned(k, 8) &&
        aligned(q_out, 8) && aligned(k_out, 8) && aligned(freqs, 16)) {
#define CK_RMS_ROPE_VEC(T, TPR)                                                                \
    rms_rope_split_half_vec_kernel<T, TPR>                                                     \
        <<<static_cast<unsigned int>((rows + kRmsRopeVecThreads / TPR - 1) /                   \
                                     (kRmsRopeVecThreads / TPR)),                              \
           kRmsRopeVecThreads, 0, stream>>>(                                                   \
            q, k, static_cast<const float*>(freqs), q_scale, k_scale, static_cast<T*>(q_out),  \
            static_cast<T*>(k_out), rows, dim1, dim2, fb, fd1, fd2, sx_b, sx_d1, sx_d2, so_b,  \
            so_d1, so_d2, sf_b, sf_d1, sf_d2, s_code, epsilon)
#define CK_RMS_ROPE_VEC_HD(T)                                  \
    if (head_dim == 64) {                                      \
        CK_RMS_ROPE_VEC(T, 8);                                 \
    } else if (head_dim == 128) {                              \
        CK_RMS_ROPE_VEC(T, 16);                                \
    } else {                                                   \
        CK_RMS_ROPE_VEC(T, 32);                                \
    }
        if (x_code == 1) {
            CK_RMS_ROPE_VEC_HD(__half);
        } else {
            CK_RMS_ROPE_VEC_HD(__bf16);
        }
#undef CK_RMS_ROPE_VEC_HD
#undef CK_RMS_ROPE_VEC
        return;
    }

#define CK_RMS_ROPE_LAUNCH(T)                                                                  \
    rms_rope_kernel<T><<<static_cast<unsigned int>(rows), kRmsRopeThreads, 0, stream>>>(       \
        q, k, freqs, q_scale, k_scale, static_cast<T*>(q_out), static_cast<T*>(k_out), dim1,   \
        dim2, static_cast<int>(head_dim), static_cast<int>(rot), fb, fd1, fd2, sx_b, sx_d1,    \
        sx_d2, sx_d, so_b, so_d1,                                                              \
        so_d2, so_d, sf_b, sf_d1, sf_d2, sf_d, sf_rot, sf_pair, x_code, f_code, s_code,        \
        epsilon, split_half)

    if (x_code == 0) {
        CK_RMS_ROPE_LAUNCH(float);
    } else if (x_code == 1) {
        CK_RMS_ROPE_LAUNCH(__half);
    } else if (x_code == 2) {
        CK_RMS_ROPE_LAUNCH(__bf16);
    } else {
        throw std::runtime_error("rms_rope: unsupported input dtype");
    }
#undef CK_RMS_ROPE_LAUNCH
}
