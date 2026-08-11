// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Fused W4A8 requantize: an already ConvRot-rotated weight becomes packed int4 +
// fp8 s_rel + f32 s_channel in one launch (codebook assign, two ALS scale
// iterations, per-channel scale, optional stochastic rounding, pack). The
// rotated weight is read in its own dtype, so the kernel sees exactly the values
// the eager path sees. group_size is fixed at 16: one block per row, each thread
// owns whole 16-wide groups.
#include <stdexcept>
#include <string>

#include "../fp8_utils.h"
#include "../launchers.h"

namespace comfy::hip_backend {

// Group scales live in dynamic LDS, one float per group, so K is bounded per
// device the way the ConvRot row buffer is. The wrappers fall back to eager past
// the budget rather than launching something that cannot fit.
constexpr int kRequantGroup = 16;
constexpr int kRequantThreads = 256;
constexpr size_t kRequantStaticLds = (16 + kRequantThreads) * sizeof(float);

int w4a8_requant_max_k() {
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) {
        return 0;
    }
    int lds = 0;
    if (hipDeviceGetAttribute(&lds, hipDeviceAttributeMaxSharedMemoryPerBlock, device) !=
            hipSuccess ||
        lds <= static_cast<int>(kRequantStaticLds)) {
        return 0;
    }
    const size_t groups = (static_cast<size_t>(lds) - kRequantStaticLds) / sizeof(float);
    return static_cast<int>(groups * kRequantGroup);
}

namespace {

__forceinline__ __device__ float rq_to_float(float v) { return v; }
__forceinline__ __device__ float rq_to_float(__half v) { return __half2float(v); }
__forceinline__ __device__ float rq_to_float(__bf16 v) { return static_cast<float>(v); }

// PCG hash over the element's index within the launch, xored with the seed. The
// caller shifts the seed by the block's first row, so the draw is reproducible for a
// given row-block size rather than independent of it. Same scheme as the CUDA kernel
// and the eager packer, which decorrelate their chunks the same way.
__forceinline__ __device__ uint32_t rq_pcg(uint32_t x) {
    x = x * 747796405u + 2891336453u;
    const uint32_t w = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return (w >> 22u) ^ w;
}

__forceinline__ __device__ float rq_uniform(int64_t idx, uint64_t seed) {
    const uint32_t h = rq_pcg(static_cast<uint32_t>(idx) ^ static_cast<uint32_t>(seed) ^
                              static_cast<uint32_t>(seed >> 32));
    return static_cast<float>(h >> 8) * (1.0f / 16777216.0f);
}

// Nearest of 16 ascending levels, lowest index on a tie, which is how eager's
// searchsorted-plus-neighbour-compare resolves one. The array is taken by
// reference, not as a pointer: a pointer to the caller's private level table
// forces it out to scratch even after inlining.
__forceinline__ __device__ int rq_nearest(float x, const float (&levels)[16]) {
    int best = 0;
    float bd = fabsf(x - levels[0]);
    #pragma unroll
    for (int j = 1; j < 16; ++j) {
        const float d = fabsf(x - levels[j]);
        if (d < bd) {
            bd = d;
            best = j;
        }
    }
    return best;
}

// Stochastic pick between the two levels bracketing x: take the upper one when x
// clears a uniform-random point inside the bracket, so a delta below the int4 step
// survives in expectation. levels is ascending, so the last one at or below x is
// the lower bound.
__forceinline__ __device__ int rq_stochastic(float x, const float (&levels)[16], int64_t idx,
                                             uint64_t seed) {
    int lo = 0;
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        if (levels[j] <= x) {
            lo = j;
        }
    }
    lo = lo > 14 ? 14 : lo;
    // Selected rather than indexed: a dynamic index into a private array would push
    // the level table out to scratch.
    float lo_v = levels[0];
    float hi_v = levels[1];
    #pragma unroll
    for (int j = 1; j < 15; ++j) {
        if (j == lo) {
            lo_v = levels[j];
            hi_v = levels[j + 1];
        }
    }
    const float thr = lo_v + rq_uniform(idx, seed) * (hi_v - lo_v);
    return lo + (x > thr ? 1 : 0);
}

template <typename InputT, bool STOCHASTIC>
__global__ __launch_bounds__(kRequantThreads) void quantize_w4a8_convrot_kernel(
    const InputT* __restrict__ rotated,     // [N, K]
    const float* __restrict__ codebook,     // [16]
    int8_t* __restrict__ packed,            // [N, K/2]
    uint8_t* __restrict__ s_rel,            // [N, K/16], raw e4m3 bytes
    float* __restrict__ s_channel,          // [N]
    int k, uint64_t seed) {

    constexpr int G = kRequantGroup;
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const int groups = k / G;
    const int64_t row_off = static_cast<int64_t>(row) * k;

    __shared__ float cb[16];
    __shared__ float red[kRequantThreads];
    extern __shared__ float gscale[];  // [groups]

    if (t < 16) {
        cb[t] = codebook[t];
    }
    __syncthreads();

    // Per-group ALS group scale, accumulating the row's shifted amax on the way.
    float thread_amax = 0.0f;
    for (int g = t; g < groups; g += kRequantThreads) {
        const int64_t base = row_off + static_cast<int64_t>(g) * G;
        float w[G];
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < G; ++i) {
            w[i] = rq_to_float(rotated[base + i]);
            amax = fmaxf(amax, fabsf(w[i]));
        }
        float gs = fmaxf(amax, 1e-8f);
        int idx[G];
        #pragma unroll
        for (int i = 0; i < G; ++i) {
            idx[i] = rq_nearest(w[i] / gs, cb);
        }
        #pragma unroll
        for (int it = 0; it < 2; ++it) {  // matches eager _ALS_ITERS
            float num = 0.0f;
            float den = 0.0f;
            #pragma unroll
            for (int i = 0; i < G; ++i) {
                const float c = cb[idx[i]];
                num += w[i] * c;
                den += c * c;
            }
            gs = fmaxf(num / fmaxf(den, 1e-8f), 1e-8f);
            #pragma unroll
            for (int i = 0; i < G; ++i) {
                idx[i] = rq_nearest(w[i] / gs, cb);
            }
        }
        gscale[g] = gs;
        #pragma unroll
        for (int i = 0; i < G; ++i) {
            thread_amax = fmaxf(thread_amax, fabsf(cb[idx[i]] * gs));
        }
    }

    // Block max -> s_channel = row_amax / 127. An LDS tree rather than shuffles,
    // so nothing here depends on the wave width.
    red[t] = thread_amax;
    __syncthreads();
    for (int s = kRequantThreads / 2; s > 0; s >>= 1) {
        if (t < s) {
            red[t] = fmaxf(red[t], red[t + s]);
        }
        __syncthreads();
    }
    const float sc = fmaxf(red[0] / 127.0f, 1e-8f);
    if (t == 0) {
        s_channel[row] = sc;
    }

    // s_rel through fp8, then int8 levels, assign, pack. The levels are built
    // from the roundtripped s_rel, which is the value the decode side reads back.
    const int64_t prow_off = static_cast<int64_t>(row) * (k / 2);
    const int64_t srow_off = static_cast<int64_t>(row) * groups;
    for (int g = t; g < groups; g += kRequantThreads) {
        const uint8_t srel_bits = pack_fp8(gscale[g] / sc, kFp8E4M3Code);
        s_rel[srow_off + g] = srel_bits;
        const float srel = fp8_to_float(srel_bits);
        float lv[16];
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            // rintf is round-half-to-even, matching torch.round.
            lv[j] = fminf(127.0f, fmaxf(-127.0f, rintf(cb[j] * srel)));
        }
        const int64_t base = row_off + static_cast<int64_t>(g) * G;
        const int64_t base_p = prow_off + static_cast<int64_t>(g) * (G / 2);
        // Packed a pair at a time rather than assigning all 16 first: holding the
        // whole group's codes alongside the level table costs enough registers to
        // spill the fp32 and fp16 instantiations.
        #pragma unroll
        for (int p = 0; p < G / 2; ++p) {
            int code[2];
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int64_t at = base + 2 * p + h;
                const float x = rq_to_float(rotated[at]) / sc;
                if constexpr (STOCHASTIC) {
                    const int a = rq_stochastic(x, lv, at, seed);
                    code[h] = a > 15 ? 15 : a;
                } else {
                    code[h] = rq_nearest(x, lv);
                }
            }
            packed[base_p + p] =
                static_cast<int8_t>((code[0] & 0xF) | ((code[1] & 0xF) << 4));
        }
    }
}

}  // namespace
}  // namespace comfy::hip_backend

// in_dtype_code is a DTYPE_TO_CODE value: 0 float32, 1 float16, 2 bfloat16.
// s_rel is written as raw e4m3 bytes.
extern "C" void launch_quantize_w4a8_convrot_kernel(
    const void* rotated, const void* codebook, void* packed, void* s_rel, void* s_channel,
    int64_t n, int64_t k, int in_dtype_code, bool stochastic, uint64_t seed, hipStream_t stream) {

    using namespace comfy::hip_backend;
    if (k % kRequantGroup != 0) {
        throw std::runtime_error("quantize_w4a8_convrot: K must be a multiple of 16");
    }
    const int max_k = w4a8_requant_max_k();
    if (max_k <= 0 || k > max_k) {
        throw std::runtime_error("quantize_w4a8_convrot: K=" + std::to_string(k) +
                                 " does not fit in LDS (max " + std::to_string(max_k) + ")");
    }
    if (n == 0) {
        return;  // a zero-block launch is hipErrorInvalidConfiguration
    }

    const size_t shmem = static_cast<size_t>(k / kRequantGroup) * sizeof(float);
    const unsigned int blocks = static_cast<unsigned int>(n);
    const int kk = static_cast<int>(k);

#define COMFY_REQUANT_LAUNCH(InputT)                                                          \
    do {                                                                                      \
        if (stochastic) {                                                                     \
            quantize_w4a8_convrot_kernel<InputT, true>                                        \
                <<<blocks, kRequantThreads, shmem, stream>>>(                                 \
                    static_cast<const InputT*>(rotated), static_cast<const float*>(codebook), \
                    static_cast<int8_t*>(packed), static_cast<uint8_t*>(s_rel),               \
                    static_cast<float*>(s_channel), kk, seed);                                \
        } else {                                                                              \
            quantize_w4a8_convrot_kernel<InputT, false>                                       \
                <<<blocks, kRequantThreads, shmem, stream>>>(                                 \
                    static_cast<const InputT*>(rotated), static_cast<const float*>(codebook), \
                    static_cast<int8_t*>(packed), static_cast<uint8_t*>(s_rel),               \
                    static_cast<float*>(s_channel), kk, 0);                                   \
        }                                                                                     \
    } while (0)

    if (in_dtype_code == 0) {
        COMFY_REQUANT_LAUNCH(float);
    } else if (in_dtype_code == 1) {
        COMFY_REQUANT_LAUNCH(__half);
    } else if (in_dtype_code == 2) {
        COMFY_REQUANT_LAUNCH(__bf16);
    } else {
        throw std::runtime_error("quantize_w4a8_convrot: unsupported input dtype code");
    }
#undef COMFY_REQUANT_LAUNCH
}

extern "C" int w4a8_requant_max_k_kernel() { return comfy::hip_backend::w4a8_requant_max_k(); }
