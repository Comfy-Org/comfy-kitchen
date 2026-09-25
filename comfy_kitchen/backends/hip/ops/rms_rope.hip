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
#include <type_traits>

#include "../hadamard.h"   // load_in / dtype codes
#include "../rope_math.h"  // rope_combine / rope_store / round_to

namespace comfy::hip_backend {

constexpr int kRmsRopeThreads = 64;
// Fast path stages a row in LDS as floats; keep the footprint small (2 rows x
// 4B each) so a block leaves room for neighbors. head_dim above this falls back
// to the re-read kernel.
constexpr int kMaxStagedHd = 2048;

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

// Fast path: contiguous rows (sx_d == so_d == 1), head_dim a whole number of
// 16-byte chunks and within the LDS budget. One warp owns one row: the row is
// vector-loaded once into LDS (as floats), reduced with warp shuffles (every
// lane ends up with the total, so no block-wide reduction at all), and rotated
// out of LDS — halving the DRAM reads of q/k versus the re-read kernel above,
// cutting the block reduction to a __syncwarp, and leaving the LDS footprint at
// one row so occupancy stays high.
template <typename T>
__global__ __launch_bounds__(kRmsRopeThreads) void rms_rope_kernel_fast(
    const void* __restrict__ q, const void* __restrict__ k,
    const void* __restrict__ freqs, const void* __restrict__ q_scale,
    const void* __restrict__ k_scale, T* __restrict__ q_out, T* __restrict__ k_out,
    int64_t dim1, int64_t dim2, int head_dim, int rot_dim,
    int64_t fb, int64_t fd1, int64_t fd2,
    int64_t sx_b, int64_t sx_d1, int64_t sx_d2,
    int64_t so_b, int64_t so_d1, int64_t so_d2,
    int64_t sf_b, int64_t sf_d1, int64_t sf_d2, int64_t sf_d, int64_t sf_rot,
    int64_t sf_pair,
    int x_code, int f_code, int s_code, float epsilon, bool split_half) {

    constexpr int kWarp = 32;
    constexpr int kVec = sizeof(T) == 2 ? 8 : 4;  // elements per 16-byte load
    extern __shared__ float smem[];               // qs[head_dim] + ks[head_dim]
    float* qs = smem;
    float* ks = smem + head_dim;

    const int t = threadIdx.x;  // one warp per block, lane == row worker
    const int64_t row = blockIdx.x;
    const int64_t d2 = row % dim2;
    const int64_t d1 = (row / dim2) % dim1;
    const int64_t b = row / (dim2 * dim1);
    const int64_t xoff = b * sx_b + d1 * sx_d1 + d2 * sx_d2;
    const int64_t ooff = b * so_b + d1 * so_d1 + d2 * so_d2;
    const bool has_k = k != nullptr;

    // Phase 1: 16-byte loads into LDS (floats), accumulating each lane's own
    // partial sum. head_dim / kVec lanes load; the rest contribute zero.
    const int chunks = head_dim / kVec;
    float sq_q = 0.0f;
    float sq_k = 0.0f;
    if (t < chunks) {
        const int e0 = t * kVec;
        const char* qp = static_cast<const char*>(q) + (xoff + e0) * sizeof(T);
        const uint4 v = *reinterpret_cast<const uint4*>(qp);
        if constexpr (sizeof(T) == 2) {
            if constexpr (std::is_same_v<T, __half>) {
                const __half* hp = reinterpret_cast<const __half*>(&v);
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float f = __half2float(hp[i]);
                    qs[e0 + i] = f;
                    sq_q += f * f;
                }
            } else {
                const __bf16* bp = reinterpret_cast<const __bf16*>(&v);
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float f = static_cast<float>(bp[i]);
                    qs[e0 + i] = f;
                    sq_q += f * f;
                }
            }
        } else {
            const float* fp = reinterpret_cast<const float*>(&v);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                qs[e0 + i] = fp[i];
                sq_q += fp[i] * fp[i];
            }
        }
        if (has_k) {
            const char* kp = static_cast<const char*>(k) + (xoff + e0) * sizeof(T);
            const uint4 kv = *reinterpret_cast<const uint4*>(kp);
            if constexpr (sizeof(T) == 2) {
                if constexpr (std::is_same_v<T, __half>) {
                    const __half* hp = reinterpret_cast<const __half*>(&kv);
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const float f = __half2float(hp[i]);
                        ks[e0 + i] = f;
                        sq_k += f * f;
                    }
                } else {
                    const __bf16* bp = reinterpret_cast<const __bf16*>(&kv);
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const float f = static_cast<float>(bp[i]);
                        ks[e0 + i] = f;
                        sq_k += f * f;
                    }
                }
            } else {
                const float* fp = reinterpret_cast<const float*>(&kv);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    ks[e0 + i] = fp[i];
                    sq_k += fp[i] * fp[i];
                }
            }
        }
    }

    // Phase 2: warp-shuffle reduction. The xor butterfly leaves the total in
    // every lane, so each lane computes rrms on its own — no shared memory, no
    // block barrier. The syncwarp then orders the LDS staging writes (phase 1)
    // before the rotation reads them (phase 3).
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sq_q += __shfl_xor(sq_q, off, 32);
        if (has_k) sq_k += __shfl_xor(sq_k, off, 32);
    }
    const float inv_d = 1.0f / static_cast<float>(head_dim);
    const float rrms_q = rsqrtf(sq_q * inv_d + epsilon);
    const float rrms_k = has_k ? rsqrtf(sq_k * inv_d + epsilon) : 0.0f;
    __syncwarp();

    // Phase 3: rotate out of LDS; stores stay 2-byte (split-half pairs are not
    // adjacent), but consecutive lanes write consecutive elements so each
    // half-wave's stores coalesce into 128B runs.
    const int64_t foff_row = (fb == 1 ? 0 : b) * sf_b + (fd1 == 1 ? 0 : d1) * sf_d1 +
                             (fd2 == 1 ? 0 : d2) * sf_d2;

    const int n_pairs = rot_dim / 2;
    for (int p = t; p < n_pairs; p += kWarp) {
        const int64_t ia = split_half ? p : 2 * p;
        const int64_t ib = split_half ? p + n_pairs : 2 * p + 1;
        const int64_t ia_out = ooff + ia;
        const int64_t ib_out = ooff + ib;

        const int64_t foff = foff_row + static_cast<int64_t>(p) * sf_d;
        const float f00 = load_in(freqs, foff, f_code);
        const float f01 = load_in(freqs, foff + sf_pair, f_code);
        const float f10 = load_in(freqs, foff + sf_rot, f_code);
        const float f11 = load_in(freqs, foff + sf_rot + sf_pair, f_code);

        const float sa = load_in(q_scale, ia, s_code);
        const float sb = load_in(q_scale, ib, s_code);
        const float qa = round_to<T>(qs[ia] * rrms_q * sa);
        const float qb = round_to<T>(qs[ib] * rrms_q * sb);
        rope_store<T>(q_out, ia_out, rope_combine(f00, qa, f01, qb, f_code));
        rope_store<T>(q_out, ib_out, rope_combine_b(f10, qa, f11, qb, f_code, split_half));

        if (has_k) {
            const float ta = load_in(k_scale, ia, s_code);
            const float tb = load_in(k_scale, ib, s_code);
            const float ka = round_to<T>(ks[ia] * rrms_k * ta);
            const float kb = round_to<T>(ks[ib] * rrms_k * tb);
            rope_store<T>(k_out, ia_out, rope_combine(f00, ka, f01, kb, f_code));
            rope_store<T>(k_out, ib_out, rope_combine_b(f10, ka, f11, kb, f_code, split_half));
        }
    }

    for (int d = rot_dim + t; d < head_dim; d += kWarp) {
        const int64_t out = ooff + d;
        rope_store<T>(q_out, out, qs[d] * rrms_q * load_in(q_scale, d, s_code));
        if (has_k) {
            rope_store<T>(k_out, out, ks[d] * rrms_k * load_in(k_scale, d, s_code));
        }
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

    // Fast path: contiguous rows whose 16-byte chunks stay aligned, head_dim a
    // whole number of chunks within the LDS budget. Everything else (strided
    // views, odd head_dim, huge rows) takes the re-read kernel. The 32-thread
    // warp-per-row schedule was tuned on the 6-WGP 780M; dGPUs keep the
    // upstream 64-thread kernel.
    const int64_t kVec = x_code == 0 ? 4 : 8;
    const bool fast = comfy_small_igpu() && sx_d == 1 && so_d == 1 && head_dim <= kMaxStagedHd &&
                      head_dim % kVec == 0 && sx_b % kVec == 0 && sx_d1 % kVec == 0 &&
                      sx_d2 % kVec == 0 && so_b % kVec == 0 && so_d1 % kVec == 0 &&
                      so_d2 % kVec == 0;

#define CK_RMS_ROPE_LAUNCH(T)                                                                  \
    if (fast) {                                                                                \
        const unsigned int smem_bytes =                                                         \
            static_cast<unsigned int>(2 * head_dim * sizeof(float));                            \
        rms_rope_kernel_fast<T><<<static_cast<unsigned int>(rows), 32, smem_bytes, stream>>>(  \
            q, k, freqs, q_scale, k_scale, static_cast<T*>(q_out), static_cast<T*>(k_out),     \
            dim1, dim2, static_cast<int>(head_dim), static_cast<int>(rot), fb, fd1, fd2,       \
            sx_b, sx_d1, sx_d2, so_b, so_d1, so_d2, sf_b, sf_d1, sf_d2, sf_d, sf_rot,          \
            sf_pair, x_code, f_code, s_code, epsilon, split_half);                             \
    } else {                                                                                    \
        rms_rope_kernel<T><<<static_cast<unsigned int>(rows), kRmsRopeThreads, 0, stream>>>(   \
            q, k, freqs, q_scale, k_scale, static_cast<T*>(q_out), static_cast<T*>(k_out),     \
            dim1, dim2, static_cast<int>(head_dim), static_cast<int>(rot), fb, fd1, fd2,       \
            sx_b, sx_d1, sx_d2, sx_d, so_b, so_d1, so_d2, so_d, sf_b, sf_d1, sf_d2, sf_d,      \
            sf_rot, sf_pair, x_code, f_code, s_code, epsilon, split_half);                     \
    }

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
