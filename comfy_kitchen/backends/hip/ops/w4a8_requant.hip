// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Fused W4A8 / W6A8 requantize: a raw weight row is ConvRot-256 rotated in
// registers, parked in LDS in its own dtype, then quantized in one launch -
// 4-bit codebook (group 16) or 6-bit uniform levels (group 16/32/64), two ALS
// scale iterations, fp8 s_rel, f32 s_channel, optional stochastic rounding and
// pack. Same math as the eager quantizer up to fp32 rounding order. One block
// per row.
#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "../fp8_utils.h"
#include "../launchers.h"

namespace comfy::hip_backend {

// Static LDS: 16 codebook floats plus one max per 32-lane slice of a 512-thread block.
constexpr size_t kRequantStaticLds = (16 + 16) * sizeof(float);

// Dynamic LDS for one row: an fp32 step per group, then the rotated row in its
// 16-bit dtype with 2 elements of padding per group.
inline size_t wxa8_requant_row_lds(int64_t k, int g) {
    return static_cast<size_t>(k / g) * sizeof(float) +
           (static_cast<size_t>(k) + 2 * static_cast<size_t>(k / g)) * 2;
}

inline int requant_device_lds() {
    int device = 0;
    int lds = 0;
    if (hipGetDevice(&device) != hipSuccess ||
        hipDeviceGetAttribute(&lds, hipDeviceAttributeMaxSharedMemoryPerBlock, device) !=
            hipSuccess) {
        return 0;
    }
    return lds;
}

// Widest K (a multiple of 256) whose row fits the current device's LDS, 0 if none.
int wxa8_requant_max_k(int g) {
    const int lds = requant_device_lds();
    if (lds <= static_cast<int>(kRequantStaticLds)) {
        return 0;
    }
    const size_t budget = static_cast<size_t>(lds) - kRequantStaticLds;
    int64_t k = static_cast<int64_t>(budget / (2.0 + 8.0 / g)) / 256 * 256;
    while (k > 0 && wxa8_requant_row_lds(k, g) > budget) {
        k -= 256;
    }
    return static_cast<int>(k);
}

namespace {

// Divisions follow eager's torch semantics, the reference both paths are measured
// against: tensor by tensor is a correctly rounded divide, tensor by Python scalar a
// multiply by the rounded reciprocal. bf16 weights land w / gs exactly on a .5 tie often
// enough that one ulp decides the rounding and can move a whole row's s_channel, and the
// build's -ffast-math would turn every x / y into an approximate reciprocal. rq_div is the
// sequence LLVM emits for a precise f32 divide (hardware reciprocal, one Newton step, two
// residual corrections) minus div_scale/div_fixup, which only matter near the range
// limits these operands never reach.
__forceinline__ __device__ float rq_div(float a, float b) {
    float r = __builtin_amdgcn_rcpf(b);
    r = __builtin_fmaf(__builtin_fmaf(-b, r, 1.0f), r, r);
    float q = a * r;
    q = __builtin_fmaf(__builtin_fmaf(-b, q, a), r, q);
    return __builtin_fmaf(__builtin_fmaf(-b, q, a), r, q);
}

__forceinline__ __device__ float rq_to_float(float v) { return v; }
__forceinline__ __device__ float rq_to_float(__half v) { return __half2float(v); }
__forceinline__ __device__ float rq_to_float(__bf16 v) { return static_cast<float>(v); }
template <typename T>
__forceinline__ __device__ T rq_from_float(float v);
template <>
__forceinline__ __device__ __half rq_from_float<__half>(float v) { return __float2half(v); }
template <>
__forceinline__ __device__ __bf16 rq_from_float<__bf16>(float v) { return static_cast<__bf16>(v); }

// PCG hash over the element's flat index in the weight, xored with the seed. Same
// scheme as the CUDA kernel; eager draws from torch.Generator, so SR matches it in
// distribution, not bit for bit.
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

// Block-wide max on every thread. Reduced per 32-lane slice, so the result does not
// depend on the wave width.
__forceinline__ __device__ float rq_block_max(float v, float* slice_max) {
    const int lane = threadIdx.x & 31;
    const int slice = threadIdx.x >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v = fmaxf(v, __shfl_xor(v, o, 32));
    }
    if (lane == 0) {
        slice_max[slice] = v;
    }
    __syncthreads();
    float m = 0.0f;
    const int slices = (blockDim.x + 31) >> 5;
    for (int s = 0; s < slices; ++s) {
        m = fmaxf(m, slice_max[s]);
    }
    return m;
}

// Entries of a sorted 16-entry table below x (<= x with LE); branch-free bisection.
// Tables are callables so the level table never materializes as a private array,
// which would spill to scratch.
template <bool LE, typename Table>
__forceinline__ __device__ int rq_count16(Table tab, float x) {
    auto lt = [&](int j) { return LE ? (tab(j) <= x) : (tab(j) < x); };
    int p = lt(7) ? 8 : 0;
    p += lt(p + 3) ? 4 : 0;
    p += lt(p + 1) ? 2 : 0;
    p += lt(p) ? 1 : 0;
    return p;
}

// Nearest entry, the lower bracket on a tie; DISTINCT skips the search for the first
// copy of a repeated value.
template <bool DISTINCT, typename Table>
__forceinline__ __device__ int rq_nearest16(Table tab, float x) {
    const int pos = rq_count16<false>(tab, x);
    const int h = min(pos, 15);
    const int l = max(pos - 1, 0);
    const float th = tab(h);
    const float tl = tab(l);
    if (fabsf(x - th) < fabsf(x - tl)) {
        return h;
    }
    if (DISTINCT) {
        return l;
    }
    return rq_count16<false>(tab, tl);  // first index holding the value tl
}

template <int LOG2G>
__forceinline__ __device__ int rq_padded_index(int i) {
    return i + 2 * (i >> LOG2G);
}
template <int G>
struct RqLog2 {
    static constexpr int value = G == 16 ? 4 : G == 32 ? 5 : 6;
};

// src(i) is the rotated row's element i as fp32.
template <typename Src>
__forceinline__ __device__ void w4_quantize_row(Src src, int k, int64_t elem_base,
                                                const float* cb, int8_t* __restrict__ packed_row,
                                                uint8_t* __restrict__ s_rel_row,
                                                float* __restrict__ s_channel_out, float* gscale,
                                                float* slice_max, bool stochastic,
                                                uint64_t seed) {
    constexpr int G = 16;
    auto cbt = [&](int j) { return cb[j]; };
    const int groups = k / G;

    // Per-group ALS step; the row's shifted amax accumulates on the way.
    float thr_amax = 0.0f;
    for (int g = threadIdx.x; g < groups; g += blockDim.x) {
        const int base = g * G;
        float w[G];
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < G; ++i) {
            w[i] = src(base + i);
            amax = fmaxf(amax, fabsf(w[i]));
        }
        float gs = fmaxf(amax, 1e-8f);
        int idx[G];
        #pragma unroll
        for (int i = 0; i < G; ++i) {
            idx[i] = rq_nearest16<true>(cbt, rq_div(w[i], gs));
        }
        #pragma unroll
        for (int it = 0; it < 2; ++it) {  // eager _ALS_ITERS
            float num = 0.0f;
            float den = 0.0f;
            #pragma unroll
            for (int i = 0; i < G; ++i) {
                const float c = cb[idx[i]];
                num += w[i] * c;
                den += c * c;
            }
            gs = fmaxf(rq_div(num, fmaxf(den, 1e-8f)), 1e-8f);
            #pragma unroll
            for (int i = 0; i < G; ++i) {
                idx[i] = rq_nearest16<true>(cbt, rq_div(w[i], gs));
            }
        }
        gscale[g] = gs;
        #pragma unroll
        for (int i = 0; i < G; ++i) {
            thr_amax = fmaxf(thr_amax, fabsf(cb[idx[i]] * gs));
        }
    }
    const float sc = fmaxf(rq_block_max(thr_amax, slice_max) * (1.0f / 127.0f), 1e-8f);
    if (threadIdx.x == 0) {
        *s_channel_out = sc;
    }

    // s_rel through fp8, then the int8 levels it decodes to, assign, pack.
    auto phase2 = [&](auto sr_tag) {
        constexpr bool SR = decltype(sr_tag)::value;
        for (int g = threadIdx.x; g < groups; g += blockDim.x) {
            const uint8_t srel_bits = pack_fp8(rq_div(gscale[g], sc), kFp8E4M3Code);
            s_rel_row[g] = srel_bits;
            const float srel = fp8_to_float(srel_bits);
            // rintf is round-half-to-even, matching torch.round.
            auto lv = [&](int j) { return fminf(127.0f, fmaxf(-127.0f, rintf(cb[j] * srel))); };
            const int base = g * G;
            uint32_t words[2] = {0u, 0u};
            #pragma unroll
            for (int i = 0; i < G; ++i) {
                const float t = rq_div(src(base + i), sc);
                int a;
                if constexpr (SR) {
                    const int lo = min(max(rq_count16<true>(lv, t) - 1, 0), 14);
                    const float l0 = lv(lo);
                    const float thr = l0 + rq_uniform(elem_base + base + i, seed) * (lv(lo + 1) - l0);
                    a = min(lo + (t > thr ? 1 : 0), 15);
                } else {
                    // Rounded levels can repeat. Eager's _assign_grid returns pos - 1, the
                    // last copy of the lower level, which the plain search already lands
                    // on, so the packed codes match eager's byte for byte.
                    a = rq_nearest16<true>(lv, t);
                }
                words[i >> 3] |= static_cast<uint32_t>(a & 0xF) << (4 * (i & 7));
            }
            uint32_t* out = reinterpret_cast<uint32_t*>(packed_row + g * (G / 2));
            out[0] = words[0];
            out[1] = words[1];
        }
    };
    if (stochastic) {
        phase2(std::true_type{});
    } else {
        phase2(std::false_type{});
    }
}

// int8 value of 6-bit level index i (0..62) at group step s: eager's _grid_levels.
__forceinline__ __device__ float w6_level(int i, float s) {
    return fminf(127.0f, fmaxf(-127.0f, rintf(static_cast<float>(i - 31) * s)));
}

// First level index >= t (searchsorted): a closed-form estimate, then fixups make it exact.
__forceinline__ __device__ int w6_lower_bound(float t, float s, float inv_s) {
    int i = __float2int_rn((ceilf(t) - 0.5f) * inv_s) + 31;
    i = min(max(i, 0), 63);
    while (i > 0 && w6_level(i - 1, s) >= t) {
        --i;
    }
    while (i < 63 && w6_level(i, s) < t) {
        ++i;
    }
    return i;
}

// Eager _assign_grid: the nearer level of the bracket around t (lower on a tie), or
// an SR draw inside it.
template <bool SR>
__forceinline__ __device__ int w6_assign(float t, float s, float inv_s, int64_t idx,
                                         uint64_t seed) {
    const int pos = w6_lower_bound(t, s, inv_s);
    if constexpr (SR) {
        const int l = min(max(pos - 1, 0), 61);
        const float level_lo = w6_level(l, s);
        const float thr = level_lo + rq_uniform(idx, seed) * (w6_level(l + 1, s) - level_lo);
        return min(l + (t > thr ? 1 : 0), 62);
    }
    const int l = min(max(pos - 1, 0), 62);
    const int h = min(pos, 62);
    return (fabsf(t - w6_level(h, s)) < fabsf(t - w6_level(l, s))) ? h : l;
}

template <typename InputT, int G>
__forceinline__ __device__ void w6_quantize_row(const InputT* row_buf, int k, int64_t elem_base,
                                                int8_t* __restrict__ packed_row,
                                                uint8_t* __restrict__ s_rel_row,
                                                float* __restrict__ s_channel_out, float* gscale,
                                                float* slice_max, bool stochastic,
                                                uint64_t seed) {
    auto src = [&](int i) { return rq_to_float(row_buf[rq_padded_index<RqLog2<G>::value>(i)]); };
    const int groups = k / G;

    // Per-group step (2 ALS rounds); row amax of the decoded weight.
    float thr_amax = 0.0f;
    for (int g = threadIdx.x; g < groups; g += blockDim.x) {
        const int base = g * G;
        float w[G];
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < G; ++i) {
            w[i] = src(base + i);
            amax = fmaxf(amax, fabsf(w[i]));
        }
        float gs = fmaxf(amax * (1.0f / 31.0f), 1e-8f);
        #pragma unroll
        for (int it = 0; it < 2; ++it) {  // eager _ALS_ITERS
            float num = 0.0f;
            float den = 0.0f;
            #pragma unroll
            for (int i = 0; i < G; ++i) {
                const float q = fminf(31.0f, fmaxf(-31.0f, rintf(rq_div(w[i], gs))));
                num += w[i] * q;
                den += q * q;
            }
            gs = fmaxf(rq_div(num, fmaxf(den, 1e-8f)), 1e-8f);
        }
        gscale[g] = gs;
        #pragma unroll
        for (int i = 0; i < G; ++i) {
            thr_amax = fmaxf(thr_amax, fabsf(fminf(31.0f, fmaxf(-31.0f, rintf(rq_div(w[i], gs)))) * gs));
        }
    }
    const float sc = fmaxf(rq_block_max(thr_amax, slice_max) * (1.0f / 127.0f), 1e-8f);
    if (threadIdx.x == 0) {
        *s_channel_out = sc;
    }

    // fp8 s_rel -> grid assignment -> code q + 32 -> nibble plane then 2-bit plane.
    int8_t* hi_plane = packed_row + k / 2;
    auto phase2 = [&](auto sr_tag) {
        constexpr bool SR = decltype(sr_tag)::value;
        for (int g = threadIdx.x; g < groups; g += blockDim.x) {
            const uint8_t srel_bits = pack_fp8(rq_div(gscale[g], sc), kFp8E4M3Code);
            s_rel_row[g] = srel_bits;
            const float srel = fp8_to_float(srel_bits);
            const float inv_srel = 1.0f / fmaxf(srel, 1e-30f);  // estimate only
            const int base = g * G;
            int8_t* lo = packed_row + g * (G / 2);
            int8_t* hi = hi_plane + g * (G / 4);
            #pragma unroll
            for (int p = 0; p < G / 4; ++p) {
                const int c = 4 * p;
                int u[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    u[j] = w6_assign<SR>(rq_div(src(base + c + j), sc), srel, inv_srel,
                                         elem_base + base + c + j, seed) + 1;
                }
                lo[2 * p] = static_cast<int8_t>((u[0] & 0xF) | ((u[1] & 0xF) << 4));
                lo[2 * p + 1] = static_cast<int8_t>((u[2] & 0xF) | ((u[3] & 0xF) << 4));
                hi[p] = static_cast<int8_t>(((u[0] >> 4) & 3) | (((u[1] >> 4) & 3) << 2) |
                                            (((u[2] >> 4) & 3) << 4) | (((u[3] >> 4) & 3) << 6));
            }
        }
    };
    if (stochastic) {
        phase2(std::true_type{});
    } else {
        phase2(std::false_type{});
    }
}

// Row k of H4 in the association the CUDA backend's h4_row_dot uses.
__forceinline__ __device__ float rq_h4_row(int k, float x0, float x1, float x2, float x3) {
    return 0.5f * ((((k == 3 ? -x0 : x0) + (k == 2 ? -x1 : x1)) + (k == 1 ? -x2 : x2)) +
                   (k == 0 ? -x3 : x3));
}

// Radix-4 stage pairing register bit RB with the lane bit selected by LX.
template <int RB, int LX>
__forceinline__ __device__ void rq_fht_mixed(float (&v)[32], int lane_bit) {
    #pragma unroll
    for (int r = 0; r < 32; ++r) {
        if (r & (1 << RB)) {
            continue;
        }
        const int r1 = r | (1 << RB);
        const float a0 = v[r];
        const float a1 = v[r1];
        const float p0 = __shfl_xor(a0, LX, 32);
        const float p1 = __shfl_xor(a1, LX, 32);
        const float x0 = lane_bit ? p0 : a0;
        const float x1 = lane_bit ? p1 : a1;
        const float x2 = lane_bit ? a0 : p0;
        const float x3 = lane_bit ? a1 : p1;
        v[r] = rq_h4_row(2 * lane_bit, x0, x1, x2, x3);
        v[r1] = rq_h4_row(2 * lane_bit + 1, x0, x1, x2, x3);
    }
}

// ConvRot-256 of one group across 8 lanes x 32 values. Index bits 0,1,2,4,6 are
// register bits and 3,5,7 lane bits. The rounded result goes to the padded LDS row.
template <typename InputT, int LOG2G>
__forceinline__ __device__ void rq_rotate_group_regs(const InputT* __restrict__ xg,
                                                     InputT* row_buf, int e_base, int lane8,
                                                     bool store) {
    const int b3 = lane8 & 1;
    const int b5 = (lane8 >> 1) & 1;
    const int b7 = lane8 >> 2;
    float v[32];  // r = b0 | b1<<1 | b2<<2 | b4<<3 | b6<<4
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int off = 8 * b3 + 16 * (q & 1) + 32 * b5 + 64 * (q >> 1) + 128 * b7;
        const uint4 raw = *reinterpret_cast<const uint4*>(xg + off);  // 8 x 16-bit
        const InputT* e = reinterpret_cast<const InputT*>(&raw);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            v[q * 8 + i] = rq_to_float(e[i]);
        }
    }
    #pragma unroll
    for (int base = 0; base < 32; base += 4) {  // bits (0,1), registers only
        const float x0 = v[base];
        const float x1 = v[base + 1];
        const float x2 = v[base + 2];
        const float x3 = v[base + 3];
        v[base] = 0.5f * (x0 + x1 + x2 - x3);
        v[base + 1] = 0.5f * (x0 + x1 - x2 + x3);
        v[base + 2] = 0.5f * (x0 - x1 + x2 + x3);
        v[base + 3] = 0.5f * (-x0 + x1 + x2 + x3);
    }
    rq_fht_mixed<2, 1>(v, b3);  // bits (2,3)
    rq_fht_mixed<3, 2>(v, b5);  // bits (4,5)
    rq_fht_mixed<4, 4>(v, b7);  // bits (6,7)
    if (!store) {
        return;
    }
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int off = 8 * b3 + 16 * (q & 1) + 32 * b5 + 64 * (q >> 1) + 128 * b7;
        InputT* dst = row_buf + rq_padded_index<LOG2G>(e_base + off);  // even index: 4-byte pairs
        #pragma unroll
        for (int i = 0; i < 8; i += 2) {
            InputT pair[2] = {rq_from_float<InputT>(v[q * 8 + i]),
                              rq_from_float<InputT>(v[q * 8 + i + 1])};
            *reinterpret_cast<uint32_t*>(dst + i) = *reinterpret_cast<const uint32_t*>(pair);
        }
    }
}

template <typename InputT, int BITS, int G>
__global__ __launch_bounds__(512) void quantize_wxa8_convrot_fused_kernel(
    const InputT* __restrict__ weight,  // [N, K], unrotated
    const float* __restrict__ codebook, // [16] at 4 bits, else null
    int8_t* __restrict__ packed,        // [N, K*BITS/8]
    uint8_t* __restrict__ s_rel,        // [N, K/G], raw e4m3 bytes
    float* __restrict__ s_channel,      // [N]
    int k, bool stochastic, uint64_t seed) {

    constexpr int LOG2G = RqLog2<G>::value;
    __shared__ float slice_max[16];
    __shared__ float cb[16];
    extern __shared__ float smem[];
    float* gscale = smem;                                                  // [K/G]
    InputT* row_buf = reinterpret_cast<InputT*>(gscale + (k >> LOG2G));  // padded rotated row
    if (BITS == 4 && threadIdx.x < 16) {
        cb[threadIdx.x] = codebook[threadIdx.x];
    }

    const int64_t row = blockIdx.x;
    const InputT* wrow = weight + row * k;
    const int clusters = blockDim.x / 8;
    const int cluster = threadIdx.x / 8;
    const int lane8 = threadIdx.x % 8;
    const int n_groups = k / 256;
    // A uniform trip count keeps every lane in the shuffles; idle clusters rotate
    // group 0 and drop the result.
    for (int it = 0; it < (n_groups + clusters - 1) / clusters; ++it) {
        const int g = it * clusters + cluster;
        const bool active = g < n_groups;
        rq_rotate_group_regs<InputT, LOG2G>(wrow + (active ? g : 0) * 256, row_buf, g * 256,
                                            lane8, active);
    }
    __syncthreads();
    if constexpr (BITS == 4) {
        auto src = [&](int i) { return rq_to_float(row_buf[rq_padded_index<4>(i)]); };
        w4_quantize_row(src, k, row * k, cb, packed + row * (k / 2), s_rel + row * (k / G),
                        s_channel + row, gscale, slice_max, stochastic, seed);
    } else {
        w6_quantize_row<InputT, G>(row_buf, k, row * k, packed + row * (k * 3 / 4),
                                   s_rel + row * (k / G), s_channel + row, gscale, slice_max,
                                   stochastic, seed);
    }
}

template <int BITS, int G>
void launch_wxa8_fused(const void* weight, const float* codebook, void* packed, void* s_rel,
                       void* s_channel, int64_t n, int64_t k, int in_dtype_code, bool stochastic,
                       uint64_t seed, hipStream_t stream) {
    // Long rows hold a CU alone anyway, so they get more waves. Same split as CUDA.
    const bool wide = k > (BITS == 4 ? 16384 : 8192);
    const int groups = static_cast<int>(k / G);
    const int threads = std::min(wide ? 512 : 256, (groups + 31) / 32 * 32);
    const size_t shmem = wxa8_requant_row_lds(k, G);
    const unsigned int blocks = static_cast<unsigned int>(n);
    const int kk = static_cast<int>(k);
    const uint64_t s = stochastic ? seed : 0;
    if (in_dtype_code == 1) {
        quantize_wxa8_convrot_fused_kernel<__half, BITS, G><<<blocks, threads, shmem, stream>>>(
            static_cast<const __half*>(weight), codebook, static_cast<int8_t*>(packed),
            static_cast<uint8_t*>(s_rel), static_cast<float*>(s_channel), kk, stochastic, s);
    } else {
        quantize_wxa8_convrot_fused_kernel<__bf16, BITS, G><<<blocks, threads, shmem, stream>>>(
            static_cast<const __bf16*>(weight), codebook, static_cast<int8_t*>(packed),
            static_cast<uint8_t*>(s_rel), static_cast<float*>(s_channel), kk, stochastic, s);
    }
}

// Staged W4A8 requantize for what the fused kernel cannot hold: fp32 weights and rows
// too wide for the rotated LDS row. The caller rotates first; only the group scales
// sit in LDS, and the row is read from global memory in each phase.
template <typename InputT>
__global__ __launch_bounds__(256) void quantize_w4a8_convrot_kernel(
    const InputT* __restrict__ rotated,  // [N, K]
    const float* __restrict__ codebook,  // [16]
    int8_t* __restrict__ packed,         // [N, K/2]
    uint8_t* __restrict__ s_rel,         // [N, K/16], raw e4m3 bytes
    float* __restrict__ s_channel,       // [N]
    int k, bool stochastic, uint64_t seed) {

    __shared__ float slice_max[16];
    __shared__ float cb[16];
    extern __shared__ float gscale[];  // [K/16]
    if (threadIdx.x < 16) {
        cb[threadIdx.x] = codebook[threadIdx.x];
    }
    __syncthreads();

    const int64_t row = blockIdx.x;
    const InputT* rrow = rotated + row * k;
    auto src = [&](int i) { return rq_to_float(rrow[i]); };
    w4_quantize_row(src, k, row * k, cb, packed + row * (k / 2), s_rel + row * (k / 16),
                    s_channel + row, gscale, slice_max, stochastic, seed);
}

template <typename InputT>
void launch_w4a8_staged(const void* rotated, const float* codebook, void* packed, void* s_rel,
                        void* s_channel, int64_t n, int64_t k, bool stochastic, uint64_t seed,
                        size_t shmem, hipStream_t stream) {
    const int groups = static_cast<int>(k / 16);
    const int threads = std::min(256, (groups + 31) / 32 * 32);
    quantize_w4a8_convrot_kernel<InputT><<<static_cast<unsigned int>(n), threads, shmem, stream>>>(
        static_cast<const InputT*>(rotated), codebook, static_cast<int8_t*>(packed),
        static_cast<uint8_t*>(s_rel), static_cast<float*>(s_channel), static_cast<int>(k),
        stochastic, stochastic ? seed : 0);
}

}  // namespace
}  // namespace comfy::hip_backend

// weight is the raw [N, K] fp16 (code 1) or bf16 (code 2) weight; ConvRot group 256.
// bits 4 takes the 16-entry codebook at G 16, bits 6 uniform levels at G 16/32/64.
// s_rel is written as raw e4m3 bytes; seed is ignored unless stochastic is set.
extern "C" void launch_quantize_wxa8_convrot_fused_kernel(
    const void* weight, const void* codebook, void* packed, void* s_rel, void* s_channel,
    int64_t n, int64_t k, int bits, int g, int in_dtype_code, bool stochastic, uint64_t seed,
    hipStream_t stream) {

    using namespace comfy::hip_backend;
    constexpr const char* kFn = "quantize_wxa8_convrot_fused";
    if (in_dtype_code != 1 && in_dtype_code != 2) {
        throw std::runtime_error(std::string(kFn) + ": weight must be float16 or bfloat16");
    }
    if (k % 256 != 0 || k % g != 0) {
        throw std::runtime_error(std::string(kFn) +
                                 ": K must be a multiple of 256 and of group_size");
    }
    const int max_k = wxa8_requant_max_k(g);
    if (k > max_k) {
        throw std::runtime_error(std::string(kFn) + ": K=" + std::to_string(k) +
                                 " does not fit in LDS (max " + std::to_string(max_k) + ")");
    }
    if (n == 0) {
        return;  // a zero-block launch is hipErrorInvalidConfiguration
    }
    const float* cb = static_cast<const float*>(codebook);
    if (bits == 4 && g == 16) {
        launch_wxa8_fused<4, 16>(weight, cb, packed, s_rel, s_channel, n, k, in_dtype_code,
                                 stochastic, seed, stream);
    } else if (bits == 6 && g == 16) {
        launch_wxa8_fused<6, 16>(weight, nullptr, packed, s_rel, s_channel, n, k, in_dtype_code,
                                 stochastic, seed, stream);
    } else if (bits == 6 && g == 32) {
        launch_wxa8_fused<6, 32>(weight, nullptr, packed, s_rel, s_channel, n, k, in_dtype_code,
                                 stochastic, seed, stream);
    } else if (bits == 6 && g == 64) {
        launch_wxa8_fused<6, 64>(weight, nullptr, packed, s_rel, s_channel, n, k, in_dtype_code,
                                 stochastic, seed, stream);
    } else {
        throw std::runtime_error(std::string(kFn) +
                                 ": bits/group_size must be 4/16 or 6/{16,32,64}");
    }
}

extern "C" int wxa8_requant_max_k_kernel(int group_size) {
    return comfy::hip_backend::wxa8_requant_max_k(group_size);
}

// rotated is the ConvRot-rotated [N, K] weight: in_dtype_code 0 float32, 1 float16 or
// 2 bfloat16. Same outputs as the fused kernel at 4 bits, group 16.
extern "C" void launch_quantize_w4a8_convrot_kernel(const void* rotated, const void* codebook,
                                                    void* packed, void* s_rel, void* s_channel,
                                                    int64_t n, int64_t k, int in_dtype_code,
                                                    bool stochastic, uint64_t seed,
                                                    hipStream_t stream) {
    using namespace comfy::hip_backend;
    constexpr const char* kFn = "quantize_w4a8_convrot";
    if (in_dtype_code < 0 || in_dtype_code > 2) {
        throw std::runtime_error(std::string(kFn) +
                                 ": rotated must be float32, float16 or bfloat16");
    }
    if (k <= 0 || k % 16 != 0) {
        throw std::runtime_error(std::string(kFn) + ": K must be a positive multiple of 16");
    }
    const size_t shmem = static_cast<size_t>(k / 16) * sizeof(float);
    if (shmem + kRequantStaticLds > static_cast<size_t>(requant_device_lds())) {
        throw std::runtime_error(std::string(kFn) + ": K=" + std::to_string(k) +
                                 " group scales do not fit in LDS");
    }
    if (n == 0) {
        return;  // a zero-block launch is hipErrorInvalidConfiguration
    }
    const float* cb = static_cast<const float*>(codebook);
    if (in_dtype_code == 0) {
        launch_w4a8_staged<float>(rotated, cb, packed, s_rel, s_channel, n, k, stochastic, seed,
                                  shmem, stream);
    } else if (in_dtype_code == 1) {
        launch_w4a8_staged<__half>(rotated, cb, packed, s_rel, s_channel, n, k, stochastic, seed,
                                   shmem, stream);
    } else {
        launch_w4a8_staged<__bf16>(rotated, cb, packed, s_rel, s_channel, n, k, stochastic, seed,
                                   shmem, stream);
    }
}
