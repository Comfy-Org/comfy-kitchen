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
#include <stdexcept>

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
        rope_store<T>(q_out, ia_out, rope_combine(f00, qa, f01, qb, f_code, split_half));
        rope_store<T>(q_out, ib_out, rope_combine(f10, qa, f11, qb, f_code, split_half));

        if (has_k) {
            const float ta = load_in(k_scale, ia, s_code);
            const float tb = load_in(k_scale, ib, s_code);
            const float ka = round_to<T>(load_in(k, ia_in, x_code) * rrms_k * ta);
            const float kb = round_to<T>(load_in(k, ib_in, x_code) * rrms_k * tb);
            rope_store<T>(k_out, ia_out, rope_combine(f00, ka, f01, kb, f_code, split_half));
            rope_store<T>(k_out, ib_out, rope_combine(f10, ka, f11, kb, f_code, split_half));
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

// On wave32 RDNA, a 128-wide head fits one wave that reproduces the 64-thread
// kernel's reduction tree while several independent
// rows share a workgroup. This removes all workgroup barriers and reduction LDS.
__device__ __forceinline__ float rms_rope_ordered_fmac(float accumulator, float value) {
    asm volatile("v_fmac_f32_e32 %0, %1, %1" : "+v"(accumulator) : "v"(value));
    return accumulator;
}

__device__ __forceinline__ float rms_rope_ordered_add(float lhs, float rhs) {
    float result;
    asm volatile("v_add_f32_e32 %0, %1, %2" : "=v"(result) : "v"(lhs), "v"(rhs));
    return result;
}

// Specialize the common BF16 Q/K, FP32-frequency, BF16-scale, interleaved
// contract so uniform runtime type branches disappear from the hot loop.
template <bool STATIC_BF16_TYPES>
__device__ __forceinline__ float rms_rope_load_x(
    const void* ptr, int64_t index, int code) {
    if constexpr (STATIC_BF16_TYPES) {
        return static_cast<float>(static_cast<const __bf16*>(ptr)[index]);
    }
    return load_in(ptr, index, code);
}

template <bool STATIC_BF16_TYPES>
__device__ __forceinline__ float rms_rope_load_freq(
    const void* ptr, int64_t index, int code) {
    if constexpr (STATIC_BF16_TYPES) {
        return static_cast<const float*>(ptr)[index];
    }
    return load_in(ptr, index, code);
}

template <bool STATIC_BF16_TYPES>
__device__ __forceinline__ float rms_rope_load_scale(
    const void* ptr, int64_t index, int code) {
    if constexpr (STATIC_BF16_TYPES) {
        return static_cast<float>(static_cast<const __bf16*>(ptr)[index]);
    }
    return load_in(ptr, index, code);
}

template <typename T, int ROWS_PER_BLOCK, bool STATIC_BF16_TYPES = false>
__global__ __launch_bounds__(ROWS_PER_BLOCK * 32) void rms_rope_wave32_128_kernel(
    const void* __restrict__ q, const void* __restrict__ k,
    const void* __restrict__ freqs, const void* __restrict__ q_scale,
    const void* __restrict__ k_scale, T* __restrict__ q_out, T* __restrict__ k_out,
    int64_t rows, int64_t dim1, int64_t dim2, int rot_dim,
    int64_t fb, int64_t fd1, int64_t fd2,
    int64_t sx_b, int64_t sx_d1, int64_t sx_d2, int64_t sx_d,
    int64_t so_b, int64_t so_d1, int64_t so_d2, int64_t so_d,
    int64_t sf_b, int64_t sf_d1, int64_t sf_d2, int64_t sf_d, int64_t sf_rot,
    int64_t sf_pair,
    int x_code, int f_code, int s_code, float epsilon, bool split_half) {

    constexpr int kHeadDim = 128;
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int64_t row = static_cast<int64_t>(blockIdx.x) * ROWS_PER_BLOCK + wave;
    if (row >= rows) return;

    const int64_t d2 = row % dim2;
    const int64_t d1 = (row / dim2) % dim1;
    const int64_t b = row / (dim2 * dim1);
    const int64_t xoff = b * sx_b + d1 * sx_d1 + d2 * sx_d2;
    const int64_t ooff = b * so_b + d1 * so_d1 + d2 * so_d2;
    const bool has_k = k != nullptr;

    // Match the original 64-thread reduction association exactly. Each old
    // thread first sums d and d+64; its s=32 partner owns d+32 and d+96.
    const int dims[4] = {lane, lane + 64, lane + 32, lane + 96};
    float sq_q_lo = 0.0f;
    float sq_q_hi = 0.0f;
    float sq_k_lo = 0.0f;
    float sq_k_hi = 0.0f;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const float qv = rms_rope_load_x<STATIC_BF16_TYPES>(
            q, xoff + static_cast<int64_t>(dims[i]) * sx_d, x_code);
        sq_q_lo = rms_rope_ordered_fmac(sq_q_lo, qv);
        if (has_k) {
            const float kv = rms_rope_load_x<STATIC_BF16_TYPES>(
                k, xoff + static_cast<int64_t>(dims[i]) * sx_d, x_code);
            sq_k_lo = rms_rope_ordered_fmac(sq_k_lo, kv);
        }
    }
    #pragma unroll
    for (int i = 2; i < 4; ++i) {
        const float qv = rms_rope_load_x<STATIC_BF16_TYPES>(
            q, xoff + static_cast<int64_t>(dims[i]) * sx_d, x_code);
        sq_q_hi = rms_rope_ordered_fmac(sq_q_hi, qv);
        if (has_k) {
            const float kv = rms_rope_load_x<STATIC_BF16_TYPES>(
                k, xoff + static_cast<int64_t>(dims[i]) * sx_d, x_code);
            sq_k_hi = rms_rope_ordered_fmac(sq_k_hi, kv);
        }
    }

    float sq_q = rms_rope_ordered_add(sq_q_lo, sq_q_hi);
    float sq_k = rms_rope_ordered_add(sq_k_lo, sq_k_hi);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sq_q = rms_rope_ordered_add(sq_q, __shfl_xor(sq_q, off, 32));
        sq_k = rms_rope_ordered_add(sq_k, __shfl_xor(sq_k, off, 32));
    }
    // Only lane zero's butterfly association is identical to red_[qk][0] in
    // the reference down-tree. Broadcast that result so compiler scheduling of
    // the other lanes cannot create rare one-ULP normalization differences.
    sq_q = __shfl(sq_q, 0, 32);
    sq_k = __shfl(sq_k, 0, 32);

    const float rrms_q = rsqrtf(sq_q * (1.0f / kHeadDim) + epsilon);
    const float rrms_k = has_k ? rsqrtf(sq_k * (1.0f / kHeadDim) + epsilon) : 0.0f;
    const int64_t foff_row = (fb == 1 ? 0 : b) * sf_b +
                             (fd1 == 1 ? 0 : d1) * sf_d1 +
                             (fd2 == 1 ? 0 : d2) * sf_d2;

    const int n_pairs = rot_dim / 2;
    for (int p = lane; p < n_pairs; p += 32) {
        const int64_t ia = STATIC_BF16_TYPES ? 2 * p : (split_half ? p : 2 * p);
        const int64_t ib = STATIC_BF16_TYPES ? 2 * p + 1
                                               : (split_half ? p + n_pairs : 2 * p + 1);
        const int64_t ia_in = xoff + ia * sx_d;
        const int64_t ib_in = xoff + ib * sx_d;
        const int64_t ia_out = ooff + ia * so_d;
        const int64_t ib_out = ooff + ib * so_d;
        const int64_t foff = foff_row + static_cast<int64_t>(p) * sf_d;
        const float f00 = rms_rope_load_freq<STATIC_BF16_TYPES>(freqs, foff, f_code);
        const float f01 = rms_rope_load_freq<STATIC_BF16_TYPES>(freqs, foff + sf_pair, f_code);
        const float f10 = rms_rope_load_freq<STATIC_BF16_TYPES>(freqs, foff + sf_rot, f_code);
        const float f11 = rms_rope_load_freq<STATIC_BF16_TYPES>(
            freqs, foff + sf_rot + sf_pair, f_code);

        const float sa = rms_rope_load_scale<STATIC_BF16_TYPES>(q_scale, ia, s_code);
        const float sb = rms_rope_load_scale<STATIC_BF16_TYPES>(q_scale, ib, s_code);
        const float qa = round_to<T>(
            rms_rope_load_x<STATIC_BF16_TYPES>(q, ia_in, x_code) * rrms_q * sa);
        const float qb = round_to<T>(
            rms_rope_load_x<STATIC_BF16_TYPES>(q, ib_in, x_code) * rrms_q * sb);
        rope_store<T>(q_out, ia_out,
                      rope_combine(f00, qa, f01, qb, f_code, split_half));
        rope_store<T>(q_out, ib_out,
                      rope_combine(f10, qa, f11, qb, f_code, split_half));

        if (has_k) {
            const float ta = rms_rope_load_scale<STATIC_BF16_TYPES>(k_scale, ia, s_code);
            const float tb = rms_rope_load_scale<STATIC_BF16_TYPES>(k_scale, ib, s_code);
            const float ka = round_to<T>(
                rms_rope_load_x<STATIC_BF16_TYPES>(k, ia_in, x_code) * rrms_k * ta);
            const float kb = round_to<T>(
                rms_rope_load_x<STATIC_BF16_TYPES>(k, ib_in, x_code) * rrms_k * tb);
            rope_store<T>(k_out, ia_out,
                          rope_combine(f00, ka, f01, kb, f_code, split_half));
            rope_store<T>(k_out, ib_out,
                          rope_combine(f10, ka, f11, kb, f_code, split_half));
        }
    }

    for (int d = rot_dim + lane; d < kHeadDim; d += 32) {
        const int64_t in = xoff + static_cast<int64_t>(d) * sx_d;
        const int64_t out = ooff + static_cast<int64_t>(d) * so_d;
        rope_store<T>(q_out, out,
                      rms_rope_load_x<STATIC_BF16_TYPES>(q, in, x_code) * rrms_q *
                          rms_rope_load_scale<STATIC_BF16_TYPES>(q_scale, d, s_code));
        if (has_k) {
            rope_store<T>(k_out, out,
                          rms_rope_load_x<STATIC_BF16_TYPES>(k, in, x_code) * rrms_k *
                              rms_rope_load_scale<STATIC_BF16_TYPES>(k_scale, d, s_code));
        }
    }
}

inline bool use_rms_rope_wave32_schedule() {
    static const bool selected = [] {
        int device = 0;
        hipDeviceProp_t properties{};
        return hipGetDevice(&device) == hipSuccess &&
            hipGetDeviceProperties(&properties, device) == hipSuccess &&
            properties.warpSize == 32;
    }();
    return selected;
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

    const bool wave32_schedule = use_rms_rope_wave32_schedule();
    const bool static_bf16_types = wave32_schedule &&
        x_code == 2 && f_code == 0 && s_code == 2 && !split_half;

#define CK_RMS_ROPE_WAVE32_LAUNCH(T, ROWS, STATIC_TYPES)                                       \
    rms_rope_wave32_128_kernel<T, ROWS, STATIC_TYPES>                                          \
        <<<static_cast<unsigned int>((rows + ROWS - 1) / ROWS), ROWS * 32, 0, stream>>>(       \
            q, k, freqs, q_scale, k_scale, static_cast<T*>(q_out), static_cast<T*>(k_out),     \
            rows, dim1, dim2, static_cast<int>(rot), fb, fd1, fd2, sx_b, sx_d1, sx_d2, sx_d,  \
            so_b, so_d1, so_d2, so_d, sf_b, sf_d1, sf_d2, sf_d, sf_rot, sf_pair, x_code,      \
            f_code, s_code, epsilon, split_half)

    if (head_dim == 128 && static_bf16_types) {
        CK_RMS_ROPE_WAVE32_LAUNCH(__bf16, 8, true);
        return;
    }

#define CK_RMS_ROPE_LAUNCH(T)                                                                  \
    do {                                                                                        \
        if (head_dim == 128 && wave32_schedule) {                                              \
            CK_RMS_ROPE_WAVE32_LAUNCH(T, 8, false);                                            \
        } else {                                                                                \
            rms_rope_kernel<T>                                                                  \
                <<<static_cast<unsigned int>(rows), kRmsRopeThreads, 0, stream>>>(             \
                    q, k, freqs, q_scale, k_scale, static_cast<T*>(q_out),                     \
                    static_cast<T*>(k_out), dim1, dim2, static_cast<int>(head_dim),            \
                    static_cast<int>(rot), fb, fd1, fd2, sx_b, sx_d1, sx_d2, sx_d, so_b,      \
                    so_d1, so_d2, so_d, sf_b, sf_d1, sf_d2, sf_d, sf_rot, sf_pair, x_code,    \
                    f_code, s_code, epsilon, split_half);                                      \
        }                                                                                       \
    } while (0)

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
#undef CK_RMS_ROPE_WAVE32_LAUNCH
}
