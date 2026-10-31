// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Float flash attention for gfx1010: a tiled prefill kernel and a direct KV-cache decode
// kernel. gfx1010 has no matrix unit and no dot instructions, so both follow the
// thread-level layout of the fp16-FMA Sage kernel (sage_attention/int8_attn_fp16.hip)
// rather than a tile MMA policy: every multiply-add is one vector FMA.
//
// Precision. fp32 and bf16 inputs are computed in fp32 (staged as fp32), so the result
// matches fp32 attention to rounding. fp16 inputs are staged as fp16 and every product
// is a v_fma_mix_f32 (f16 x f16 -> f32), exact products with fp32 sums. Scores,
// probabilities and O are always fp32.
//
// Rows. The G query heads that share a KV head are folded with the queries into one row
// space of G * Sq rows (row = g * Sq + position), so a block streams one KV head once
// for all of them; decode (Sq = 1) has G rows per KV head.
//
// Prefill (flash_attn_kernel). K and V tiles are staged in LDS. A group of kL lanes owns
// four rows, each lane 8 dims of their Q and O (HD 64 and 128; at HD 256 two rows of 16
// dims, kL = 16), so every K and V element a lane reads from LDS feeds four rows: LDS
// output (128 B a clock per CU) is what bounds fp32 here, and two rows of 16 dims a lane
// ran ~2.2 TFLOPS against it at HD 128. A lane's dims are its 8-wide chunks
// c * kL + part, in both kernels. For
// QK a lane computes partial scores of a 16-key step over its dims and log2(kL) DPP
// butterfly steps reduce-scatter them, leaving each lane the full scores of 16 / kL keys,
// whose softmax it runs; the probabilities are broadcast back over the group for PV.
//
// Splits. With num_splits > 1 blockIdx.y takes an equal share of the keys and writes its
// normalized O and base-2 LSE to fp32 scratch; a second kernel merges them. KV-cache
// decode needs this to put enough blocks on the GPU.
#include <hip/hip_runtime.h>

#include <cfloat>
#include <cstdint>
#include <type_traits>

#include <hip/hip_fp16.h>

#include "../epilogue.h"
#include "../launchers.h"

namespace comfy::hip_backend::flash_attn {
namespace {

constexpr float kLog2e = 1.44269504088896340736f;
// Finite, not -inf: -ffast-math is on and scores go through fma and exp2 first.
constexpr float kMaskedScore = -50000.0f;
constexpr float kMaskFloor = -1.0e30f;
constexpr float kNegInf = -FLT_MAX / 4.0f;

constexpr int kSub = 16;  // keys per softmax step

__forceinline__ __device__ int imin(int a, int b) { return a < b ? a : b; }

// DPP8 selector: lane i of each 8-lane group reads lane f(i) of the same group.
template <typename F>
constexpr unsigned dpp8_selector(F f) {
    unsigned sel = 0;
    for (int i = 0; i < 8; ++i) sel |= static_cast<unsigned>(f(i)) << (3 * i);
    return sel;
}

template <unsigned kSel>
__forceinline__ __device__ float dpp8(float v) {
    return __int_as_float(__builtin_amdgcn_mov_dpp8(__float_as_int(v), kSel));
}

template <int kCtrl>
__forceinline__ __device__ float dpp16(float v) {
    return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), kCtrl, 0xF, 0xF, false));
}

// The value of the lane `kMask` away (xor), within an aligned group of up to 16 lanes.
template <int kMask>
__forceinline__ __device__ float lane_xor(float v) {
    if constexpr (kMask == 8) {
        return dpp16<0x168>(v);  // row_xmask:8
    } else {
        return dpp8<dpp8_selector([](int i) { return i ^ kMask; })>(v);
    }
}

// The value of lane kSrc of this lane's aligned kL-lane group (kL 4, 8 or 16).
template <int kL, int kSrc>
__forceinline__ __device__ float group_lane(float v) {
    if constexpr (kL == 16) {
        return dpp16<0x150 + kSrc>(v);  // row_share:kSrc
    } else {
        return dpp8<dpp8_selector([](int i) { return (i & ~(kL - 1)) | kSrc; })>(v);
    }
}

// Sum (or max) over the aligned kL-lane group.
template <int kL, bool kMax = false>
__forceinline__ __device__ float group_reduce(float v) {
#pragma unroll
    for (int m = 1; m < kL; m *= 2) {
        const float other = m == 1 ? lane_xor<1>(v) : m == 2 ? lane_xor<2>(v) : m == 4 ? lane_xor<4>(v) : lane_xor<8>(v);
        v = kMax ? fmaxf(v, other) : v + other;
    }
    return v;
}

// First key of the 16 / kL a lane of the group owns after the reduce-scatter: step s
// (xor kL >> (s + 1)) splits the remaining keys on key bit 3 - s, and the lane keeps the
// half its own bit selects. Slot j of the lane's partial scores holds key j ^ X.
template <int kL>
constexpr int owned_key(int part) {
    int x = 0;
    for (int s = 0, m = kL / 2; m > 0; ++s, m /= 2) x |= ((part & m) ? 1 : 0) << (3 - s);
    return x;
}

// Staged elements: fp32 for fp32/bf16 inputs, fp16 for fp16 inputs.
template <typename InT>
using StageOf = std::conditional_t<std::is_same_v<InT, __half>, _Float16, float>;

// Eight elements of a stage row, 16 or 32 bytes.
template <typename S>
struct alignas(16) Vec8 {
    S v[8];
};

// Eight input elements from global memory as staged elements (16-byte loads).
template <typename InT>
__forceinline__ __device__ Vec8<StageOf<InT>> load8(const InT* p) {
    Vec8<StageOf<InT>> out;
    if constexpr (std::is_same_v<InT, float>) {
        const float4 a = *reinterpret_cast<const float4*>(p);
        const float4 b = *reinterpret_cast<const float4*>(p + 4);
        out.v[0] = a.x; out.v[1] = a.y; out.v[2] = a.z; out.v[3] = a.w;
        out.v[4] = b.x; out.v[5] = b.y; out.v[6] = b.z; out.v[7] = b.w;
    } else {
        // __half is a struct, not a vector element type: take the words apart.
        const uint4 raw = *reinterpret_cast<const uint4*>(p);
        const unsigned int words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned int word = words[i / 2];
            if constexpr (std::is_same_v<InT, __half>) {
                out.v[i] = __builtin_bit_cast(_Float16, static_cast<unsigned short>(i % 2 ? word >> 16 : word));
            } else {
                out.v[i] = __uint_as_float(i % 2 ? word & 0xffff0000u : word << 16);
            }
        }
    }
    return out;
}

template <typename S>
__forceinline__ __device__ Vec8<S> zero8() {
    Vec8<S> out;
#pragma unroll
    for (int i = 0; i < 8; ++i) out.v[i] = static_cast<S>(0);
    return out;
}

struct Args {
    const void* q;
    const void* k;
    const void* v;
    const void* mask;
    const int* kv_lengths;
    void* o;
    float* o_accum;
    float* lse_accum;
    int heads_kv, groups, q_len, kv_len, num_splits, mask_code;
    int64_t q_sb, q_sh, q_sn;
    int64_t k_sb, k_sh, k_sn;
    int64_t v_sb, v_sh, v_sn;
    int64_t o_sb, o_sh, o_sn;
    int64_t m_sb, m_sh, m_sq, m_sk;
    float scale;
    hipStream_t stream;
};

__forceinline__ __device__ float mask_value(const void* mask, int code, int64_t i) {
    if (code == 3) return static_cast<const uint8_t*>(mask)[i] ? 0.0f : kMaskFloor;
    return load_scalar(mask, code, i);
}

// Valid keys of batch b: kv_len, or the device count the caller may rewrite between
// graph replays.
__forceinline__ __device__ int valid_keys(const Args& a, int b) {
    if (a.kv_lengths == nullptr) return a.kv_len;
    const int length = a.kv_lengths[b];
    return length < 0 ? 0 : (length > a.kv_len ? a.kv_len : length);
}

// kRows rows of Q from r0 as fp32 (fp16 Q widens exactly): the lane's 8-wide chunks
// c * kChunkStride + part, and where those rows live.
template <typename InT, int kRows, int kChunks, int kChunkStride>
__forceinline__ __device__ void load_q_rows(const Args& a, int b, int kvh, int r0, int part,
                                            float (&qf)[kRows][kChunks * 8], int (&qhead)[kRows],
                                            int (&qpos)[kRows]) {
    const int rows = a.groups * a.q_len;
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
        const int row = imin(r0 + r, rows - 1);
        qhead[r] = kvh * a.groups + row / a.q_len;
        qpos[r] = row % a.q_len;
        const InT* q_row = static_cast<const InT*>(a.q) + b * a.q_sb + qhead[r] * a.q_sh + qpos[r] * a.q_sn;
#pragma unroll
        for (int c = 0; c < kChunks; ++c) {
            const Vec8<StageOf<InT>> raw = load8<InT>(q_row + (c * kChunkStride + part) * 8);
#pragma unroll
            for (int e = 0; e < 8; ++e) qf[r][c * 8 + e] = static_cast<float>(raw.v[e]);
        }
    }
}

// The row's O over the keys it attended, normalized, into o (one split) or the split
// scratch with its base-2 LSE. Element el of a lane's O is dim
// ((el / 8) * kL + part) * 8 + el % 8.
template <typename InT, int HD, int kDims, int kL>
__forceinline__ __device__ void store_row(const Args& a, int b, int split, int part, int head, int pos,
                                          const float (&o)[kDims], float m, float d, int rows_total) {
    // d is 0 only when no key was attended (an empty split, or every key masked); the
    // output is then 0 rather than NaN.
    const float inv_d = d > 0.0f ? 1.0f / d : 0.0f;
    if (a.num_splits > 1) {
        // Rows of the scratch are (b, query head, position), the output's own order.
        const int64_t row = (static_cast<int64_t>(b) * a.heads_kv * a.groups + head) * a.q_len + pos;
        const int64_t slot = static_cast<int64_t>(split) * rows_total + row;
        float* dst = a.o_accum + slot * HD;
#pragma unroll
        for (int el = 0; el < kDims; ++el) dst[((el / 8) * kL + part) * 8 + el % 8] = o[el] * inv_d;
        if (part == 0) a.lse_accum[slot] = d > 0.0f ? m + __builtin_log2f(d) : kNegInf;
    } else {
        InT* dst = static_cast<InT*>(a.o) + b * a.o_sb + head * a.o_sh + pos * a.o_sn;
#pragma unroll
        for (int el = 0; el < kDims; ++el) dst[((el / 8) * kL + part) * 8 + el % 8] = static_cast<InT>(o[el] * inv_d);
    }
}

template <typename InT, int HD, int kWaves>
__global__ __launch_bounds__(kWaves * 32) void flash_attn_kernel(Args a) {
    using S = StageOf<InT>;
    constexpr int kChunks = HD <= 128 ? 1 : 2;  // 8-wide chunks of each row a lane holds
    constexpr int kR = HD <= 128 ? 4 : 2;       // rows per lane group
    constexpr int kL = HD / (8 * kChunks);      // lanes per group
    constexpr int kOwn = kSub / kL;             // keys whose softmax a lane runs
    constexpr int kRowsPerWave = 32 / kL * kR;
    constexpr int kRowsPerBlock = kWaves * kRowsPerWave;
    constexpr int kThreads = kWaves * 32;
    // K and V tiles of 16 KiB each: 64 or 32 keys at HD 128 (fp16 or fp32 staging).
    constexpr int CTA_K = (sizeof(S) == 2 ? 64 : 32) * 128 / (HD < 128 ? 128 : HD);
    constexpr int kUnits = HD / 8;              // Vec8 units per stage row
    static_assert(kL == 8 || kL == 16, "groups of 8 lanes (HD 64) or 16 (HD 128, 256)");
    __shared__ Vec8<S> smem_k[CTA_K * kUnits];
    __shared__ Vec8<S> smem_v[CTA_K * kUnits];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int part = lane % kL;
    const int x_key = (part & 8 ? owned_key<kL>(8) : 0) | (part & 4 ? owned_key<kL>(4) : 0) |
                      (part & 2 ? owned_key<kL>(2) : 0) | (part & 1 ? owned_key<kL>(1) : 0);
    const int split = blockIdx.y;
    const int b = blockIdx.z / a.heads_kv;
    const int kvh = blockIdx.z - b * a.heads_kv;
    const int rows = a.groups * a.q_len;
    const int r0 = blockIdx.x * kRowsPerBlock + wave * kRowsPerWave + (lane / kL) * kR;

    const int length = valid_keys(a, b);
    const int num_tiles = (length + CTA_K - 1) / CTA_K;
    const int tiles_per_split = (num_tiles + a.num_splits - 1) / a.num_splits;
    const int tile_begin = imin(split * tiles_per_split, num_tiles);
    const int tile_end = imin(tile_begin + tiles_per_split, num_tiles);

    const InT* k_head = static_cast<const InT*>(a.k) + b * a.k_sb + kvh * a.k_sh;
    const InT* v_head = static_cast<const InT*>(a.v) + b * a.v_sb + kvh * a.v_sh;

    // The softmax scale, in base-2 units, is applied to each finished score so the
    // products stay those of the inputs.
    const float scale = a.scale * kLog2e;
    float qf[kR][kChunks * 8];
    int qhead[kR], qpos[kR];
    load_q_rows<InT, kR, kChunks, kL>(a, b, kvh, r0, part, qf, qhead, qpos);

    float o_val[kR][kChunks * 8];
    float m_run[kR], d_run[kR];
#pragma unroll
    for (int r = 0; r < kR; ++r) {
#pragma unroll
        for (int t = 0; t < kChunks * 8; ++t) o_val[r][t] = 0.0f;
        m_run[r] = kMaskedScore;
        d_run[r] = 0.0f;
    }

#pragma unroll 1
    for (int tile = tile_begin; tile < tile_end; ++tile) {
        const int kb0 = tile * CTA_K;
        __syncthreads();
        // Keys past the length are staged as zeros: a probability of 0 times stale V
        // would still be NaN if that memory held one. Unrolled further, the hoisted
        // global loads alone took over a hundred VGPRs.
#pragma unroll 2
        for (int idx = tid; idx < CTA_K * kUnits; idx += kThreads) {
            const int row = idx / kUnits;
            const int unit = idx % kUnits;
            const int key = kb0 + row;
            if (key < length) {
                smem_k[idx] = load8<InT>(k_head + key * a.k_sn + unit * 8);
                smem_v[idx] = load8<InT>(v_head + key * a.v_sn + unit * 8);
            } else {
                smem_k[idx] = zero8<S>();
                smem_v[idx] = zero8<S>();
            }
        }
        __syncthreads();

#pragma unroll 1
        for (int sub = 0; sub < CTA_K / kSub; ++sub) {
            const int key0 = kb0 + sub * kSub;
            if (key0 >= length) break;

            // Partial scores over this lane's dims; slot j is key j ^ x_key.
            float acc[kR][kSub];
#pragma unroll
            for (int j = 0; j < kSub; ++j) {
#pragma unroll
                for (int r = 0; r < kR; ++r) acc[r][j] = 0.0f;
#pragma unroll
                for (int c = 0; c < kChunks; ++c) {
                    const Vec8<S> kv = smem_k[(sub * kSub + (j ^ x_key)) * kUnits + c * kL + part];
#pragma unroll
                    for (int r = 0; r < kR; ++r)
#pragma unroll
                        for (int e = 0; e < 8; ++e)
                            acc[r][j] = __builtin_fmaf(qf[r][c * 8 + e], static_cast<float>(kv.v[e]), acc[r][j]);
                }
                // Keep later keys' K reads from being hoisted over these FMAs.
                if (j % 2 == 1) __builtin_amdgcn_sched_barrier(0);
            }

            // Butterfly reduce-scatter: the lane keeps the low half of its slots and adds
            // its partner's high half, which holds the same keys.
#pragma unroll
            for (int width = kSub / 2, m = kL / 2; m > 0; width /= 2, m /= 2) {
#pragma unroll
                for (int r = 0; r < kR; ++r)
#pragma unroll
                    for (int i = 0; i < width; ++i) {
                        const float high = acc[r][i + width];
                        acc[r][i] += m == 8 ? lane_xor<8>(high) : m == 4 ? lane_xor<4>(high)
                                   : m == 2 ? lane_xor<2>(high) : lane_xor<1>(high);
                    }
            }

            const int key_base = key0 + x_key;
            float p[kR][kOwn];
#pragma unroll
            for (int r = 0; r < kR; ++r) {
                float s[kOwn];
#pragma unroll
                for (int kk = 0; kk < kOwn; ++kk) s[kk] = acc[r][kk] * scale;
                if (a.mask != nullptr) {
                    const int64_t base = b * a.m_sb + qhead[r] * a.m_sh + qpos[r] * a.m_sq;
#pragma unroll
                    for (int kk = 0; kk < kOwn; ++kk) {
                        const int key = imin(key_base + kk, length - 1);
                        s[kk] = fmaxf(fmaf(mask_value(a.mask, a.mask_code, base + key * a.m_sk), kLog2e, s[kk]), kMaskFloor);
                    }
                }
                float mx = kMaskFloor;
#pragma unroll
                for (int kk = 0; kk < kOwn; ++kk) mx = fmaxf(mx, key_base + kk < length ? s[kk] : kMaskFloor);
                mx = group_reduce<kL, true>(mx);

                // Lazy rescaling (in log2 units), as in int8_attn_fp16.hip: the running max
                // only moves when a step passes it by more than 8, to 4 below that step's
                // max, so most steps skip rescaling O. O and d are rescaled together, so
                // O / d is unchanged; probabilities stay below 2^8.
                const bool update = mx > m_run[r] + 8.0f;
                if (__any(update)) {
                    const float m_next = update ? mx - 4.0f : m_run[r];
                    const float alpha = __builtin_amdgcn_exp2f(m_run[r] - m_next);
                    d_run[r] *= alpha;
#pragma unroll
                    for (int t = 0; t < kChunks * 8; ++t) o_val[r][t] *= alpha;
                    m_run[r] = m_next;
                }
#pragma unroll
                for (int kk = 0; kk < kOwn; ++kk) {
                    const float e = __builtin_amdgcn_exp2f(s[kk] - m_run[r]);
                    p[r][kk] = key_base + kk < length ? e : 0.0f;
                    d_run[r] += p[r][kk];  // own keys; summed over the group at the end
                }
            }

            // PV, one key at a time: the key's probability comes from the lane that owns it.
#pragma unroll
            for (int src = 0; src < kL; ++src) {
#pragma unroll
                for (int kk = 0; kk < kOwn; ++kk) {
                    float ph[kR];
#pragma unroll
                    for (int r = 0; r < kR; ++r) {
                        const float own = p[r][kk];
                        ph[r] = src == 0 ? group_lane<kL, 0>(own) : src == 1 ? group_lane<kL, 1>(own)
                              : src == 2 ? group_lane<kL, 2>(own) : src == 3 ? group_lane<kL, 3>(own)
                              : src == 4 ? group_lane<kL, 4>(own) : src == 5 ? group_lane<kL, 5>(own)
                              : src == 6 ? group_lane<kL, 6>(own) : src == 7 ? group_lane<kL, 7>(own)
                              : src == 8 ? group_lane<kL, 8 % kL>(own) : src == 9 ? group_lane<kL, 9 % kL>(own)
                              : src == 10 ? group_lane<kL, 10 % kL>(own) : src == 11 ? group_lane<kL, 11 % kL>(own)
                              : src == 12 ? group_lane<kL, 12 % kL>(own) : src == 13 ? group_lane<kL, 13 % kL>(own)
                              : src == 14 ? group_lane<kL, 14 % kL>(own) : group_lane<kL, 15 % kL>(own);
                    }
#pragma unroll
                    for (int c = 0; c < kChunks; ++c) {
                        const Vec8<S> vv = smem_v[(sub * kSub + owned_key<kL>(src) + kk) * kUnits + c * kL + part];
#pragma unroll
                        for (int e = 0; e < 8; ++e)
#pragma unroll
                            for (int r = 0; r < kR; ++r)
                                o_val[r][c * 8 + e] = __builtin_fmaf(ph[r], static_cast<float>(vv.v[e]), o_val[r][c * 8 + e]);
                    }
                    if ((src * kOwn + kk) % 2 == 1) __builtin_amdgcn_sched_barrier(0);
                }
            }
        }
    }

    // Each lane summed only its own keys' probabilities; every lane is still active here.
#pragma unroll
    for (int r = 0; r < kR; ++r) d_run[r] = group_reduce<kL>(d_run[r]);

#pragma unroll
    for (int r = 0; r < kR; ++r) {
        if (r0 + r >= rows) break;
        store_row<InT, HD, kChunks * 8, kL>(a, b, split, part, qhead[r], qpos[r], o_val[r], m_run[r], d_run[r],
                                            gridDim.z * rows);
    }
}

// KV-cache decode, and any call at HD 64 or 128 whose rows fit one wave. A KV head has too few rows to
// reuse a staged tile, so a block is one wave reading K and V straight from global
// memory. Its groups of kL = HD / 16 lanes (two 8-wide chunks of two rows a lane) split the keys
// instead of the rows: group gi holds rows (gi % row_groups) * 2 + {0, 1} and takes every
// slices-th 16-key step of the block's range. A group reads keys in order, so its lanes
// read one contiguous row; each score is all-reduced over the group and every lane runs
// the softmax of all 16 keys, which needs no probability broadcast. The slices merge at
// the end. Q and O are QT, K and V KVT: fp32 Q over an fp16 cache widens each K and V
// element exactly, so the math stays fp32.
template <typename QT, typename KVT, int HD>
__global__ __launch_bounds__(32) void flash_attn_direct_kernel(Args a, int row_groups) {
    using S = StageOf<KVT>;
    constexpr int kR = 2;
    constexpr int kL = HD / 16;
    constexpr int kGroups = 32 / kL;
    constexpr int kAhead = 4;  // keys whose K or V are loaded together
    const int lane = threadIdx.x;
    const int part = lane % kL;
    const int gi = lane / kL;
    const int slice = gi / row_groups;
    const int slices = kGroups / row_groups;
    const int split = blockIdx.y;
    const int b = blockIdx.z / a.heads_kv;
    const int kvh = blockIdx.z - b * a.heads_kv;
    const int rows = a.groups * a.q_len;
    const int r0 = (gi % row_groups) * kR;

    const int length = valid_keys(a, b);
    const int steps = (length + kSub - 1) / kSub;
    const int steps_per_split = (steps + a.num_splits - 1) / a.num_splits;
    const int step_begin = imin(split * steps_per_split, steps);
    const int step_end = imin(step_begin + steps_per_split, steps);

    const KVT* k_head = static_cast<const KVT*>(a.k) + b * a.k_sb + kvh * a.k_sh;
    const KVT* v_head = static_cast<const KVT*>(a.v) + b * a.v_sb + kvh * a.v_sh;
    const float scale = a.scale * kLog2e;
    float qf[kR][16];
    int qhead[kR], qpos[kR];
    load_q_rows<QT, kR, 2, kL>(a, b, kvh, r0, part, qf, qhead, qpos);

    float o_val[kR][16];
    float m_run[kR], d_run[kR];
#pragma unroll
    for (int r = 0; r < kR; ++r) {
#pragma unroll
        for (int t = 0; t < 16; ++t) o_val[r][t] = 0.0f;
        m_run[r] = kMaskedScore;
        d_run[r] = 0.0f;
    }

#pragma unroll 1
    for (int step = step_begin + slice; step < step_end; step += slices) {
        const int key0 = step * kSub;
        float sc[kR][kSub];
#pragma unroll
        for (int j0 = 0; j0 < kSub; j0 += kAhead) {
            // Clamped rather than predicated: a key past the length reads a valid row and
            // gets probability 0 below.
            Vec8<S> kv[kAhead][2];
#pragma unroll
            for (int j = 0; j < kAhead; ++j) {
                const KVT* k_row = k_head + imin(key0 + j0 + j, length - 1) * a.k_sn;
#pragma unroll
                for (int c = 0; c < 2; ++c) kv[j][c] = load8<KVT>(k_row + (kL * c + part) * 8);
            }
#pragma unroll
            for (int j = 0; j < kAhead; ++j) {
#pragma unroll
                for (int r = 0; r < kR; ++r) {
                    float dot = 0.0f;
#pragma unroll
                    for (int c = 0; c < 2; ++c)
#pragma unroll
                        for (int e = 0; e < 8; ++e) dot = __builtin_fmaf(qf[r][c * 8 + e], static_cast<float>(kv[j][c].v[e]), dot);
                    sc[r][j0 + j] = group_reduce<kL>(dot) * scale;
                }
            }
            __builtin_amdgcn_sched_barrier(0);
        }

        float p[kR][kSub];
#pragma unroll
        for (int r = 0; r < kR; ++r) {
            if (a.mask != nullptr) {
                const int64_t base = b * a.m_sb + qhead[r] * a.m_sh + qpos[r] * a.m_sq;
#pragma unroll
                for (int j = 0; j < kSub; ++j) {
                    const int key = imin(key0 + j, length - 1);
                    sc[r][j] = fmaxf(fmaf(mask_value(a.mask, a.mask_code, base + key * a.m_sk), kLog2e, sc[r][j]), kMaskFloor);
                }
            }
            float mx = kMaskFloor;
#pragma unroll
            for (int j = 0; j < kSub; ++j) mx = fmaxf(mx, key0 + j < length ? sc[r][j] : kMaskFloor);
            // Lazy rescaling as in flash_attn_kernel; mx is the same on every lane of the group.
            if (mx > m_run[r] + 8.0f) {
                const float m_next = mx - 4.0f;
                const float alpha = __builtin_amdgcn_exp2f(m_run[r] - m_next);
                d_run[r] *= alpha;
#pragma unroll
                for (int t = 0; t < 16; ++t) o_val[r][t] *= alpha;
                m_run[r] = m_next;
            }
#pragma unroll
            for (int j = 0; j < kSub; ++j) {
                const float e = __builtin_amdgcn_exp2f(sc[r][j] - m_run[r]);
                p[r][j] = key0 + j < length ? e : 0.0f;
                d_run[r] += p[r][j];
            }
        }

#pragma unroll
        for (int j0 = 0; j0 < kSub; j0 += kAhead) {
            Vec8<S> vv[kAhead][2];
#pragma unroll
            for (int j = 0; j < kAhead; ++j) {
                const KVT* v_row = v_head + imin(key0 + j0 + j, length - 1) * a.v_sn;
#pragma unroll
                for (int c = 0; c < 2; ++c) vv[j][c] = load8<KVT>(v_row + (kL * c + part) * 8);
            }
#pragma unroll
            for (int j = 0; j < kAhead; ++j)
#pragma unroll
                for (int c = 0; c < 2; ++c)
#pragma unroll
                    for (int e = 0; e < 8; ++e)
#pragma unroll
                        for (int r = 0; r < kR; ++r)
                            o_val[r][c * 8 + e] = __builtin_fmaf(p[r][j0 + j], static_cast<float>(vv[j][c].v[e]), o_val[r][c * 8 + e]);
            __builtin_amdgcn_sched_barrier(0);
        }
    }

    // Merge the slices that hold the same rows; they are row_groups * kL lanes apart.
#pragma unroll
    for (int r = 0; r < kR; ++r) {
        for (int dist = row_groups * kL; dist < 32; dist *= 2) {
            const float m_other = __shfl_xor(m_run[r], dist, 32);
            const float d_other = __shfl_xor(d_run[r], dist, 32);
            const float m_new = fmaxf(m_run[r], m_other);
            const float a_self = __builtin_amdgcn_exp2f(m_run[r] - m_new);
            const float a_other = __builtin_amdgcn_exp2f(m_other - m_new);
#pragma unroll
            for (int t = 0; t < 16; ++t)
                o_val[r][t] = o_val[r][t] * a_self + __shfl_xor(o_val[r][t], dist, 32) * a_other;
            d_run[r] = d_run[r] * a_self + d_other * a_other;
            m_run[r] = m_new;
        }
    }
    if (slice != 0) return;
#pragma unroll
    for (int r = 0; r < kR; ++r) {
        if (r0 + r >= rows) break;
        store_row<QT, HD, 16, kL>(a, b, split, part, qhead[r], qpos[r], o_val[r], m_run[r], d_run[r],
                                   gridDim.z * rows);
    }
}

// One block per output row, one lane per head dimension.
template <typename OutT, int HD>
__global__ __launch_bounds__(HD) void flash_attn_combine_kernel(Args a, int rows_total) {
    const int row = blockIdx.x;
    const int d = threadIdx.x;
    float m = kNegInf;
    for (int s = 0; s < a.num_splits; ++s) m = fmaxf(m, a.lse_accum[static_cast<int64_t>(s) * rows_total + row]);
    float denominator = 0.0f, numerator = 0.0f;
    for (int s = 0; s < a.num_splits; ++s) {
        const int64_t slot = static_cast<int64_t>(s) * rows_total + row;
        const float weight = __builtin_amdgcn_exp2f(a.lse_accum[slot] - m);
        denominator += weight;
        numerator = __builtin_fmaf(weight, a.o_accum[slot * HD + d], numerator);
    }
    const int qpos = row % a.q_len;
    const int head = (row / a.q_len) % (a.heads_kv * a.groups);
    const int b = row / (a.q_len * a.heads_kv * a.groups);
    static_cast<OutT*>(a.o)[b * a.o_sb + head * a.o_sh + qpos * a.o_sn + d] =
        static_cast<OutT>(denominator > 0.0f ? numerator / denominator : 0.0f);
}

template <typename InT, int HD>
void launch_combine(const Args& a, int batch) {
    if (a.num_splits > 1) {
        const int rows_total = batch * a.heads_kv * a.groups * a.q_len;
        flash_attn_combine_kernel<InT, HD><<<rows_total, HD, 0, a.stream>>>(a, rows_total);
    }
}

template <typename InT, int HD>
void launch_tiled(const Args& a, int batch) {
    constexpr int kWaves = 4;
    constexpr int kRowsPerBlock = kWaves * (HD <= 128 ? 32 / (HD / 8) * 4 : 4);
    const int rows = a.groups * a.q_len;
    const dim3 grid((rows + kRowsPerBlock - 1) / kRowsPerBlock, a.num_splits, batch * a.heads_kv);
    flash_attn_kernel<InT, HD, kWaves><<<grid, kWaves * 32, 0, a.stream>>>(a);
    launch_combine<InT, HD>(a, batch);
}

template <typename QT, typename KVT, int HD>
void launch_direct(const Args& a, int batch) {
    const int rows = a.groups * a.q_len;
    int row_groups = 1;
    while (row_groups * 2 < rows) row_groups *= 2;
    const dim3 grid(1, a.num_splits, batch * a.heads_kv);
    flash_attn_direct_kernel<QT, KVT, HD><<<grid, 32, 0, a.stream>>>(a, row_groups);
    launch_combine<QT, HD>(a, batch);
}

template <typename InT>
void launch_dims(const Args& a, int batch, int head_dim, bool few_rows) {
    // Rows that fit one wave of the direct kernel (decode) take it; splits make the blocks.
    if (head_dim == 64) {
        few_rows ? launch_direct<InT, InT, 64>(a, batch) : launch_tiled<InT, 64>(a, batch);
    } else if (head_dim == 128) {
        few_rows ? launch_direct<InT, InT, 128>(a, batch) : launch_tiled<InT, 128>(a, batch);
    } else {
        // The direct kernel at HD 256 (fp16) crashes LLVM's instruction selection
        // ("Cannot select: f32 = Constant<-1>"); the tiled kernel takes any row count.
        launch_tiled<InT, 256>(a, batch);
    }
}

}  // namespace
}  // namespace comfy::hip_backend::flash_attn

extern "C" void launch_flash_attn(const void* q, const void* k, const void* v, const void* mask,
                                  int mask_code, const int* kv_lengths, void* o, float* o_accum,
                                  float* lse_accum, int batch, int heads_kv, int groups, int q_len,
                                  int kv_len, int head_dim, int num_splits, int dtype_code,
                                  const int64_t* strides, float scale, bool few_rows,
                                  hipStream_t stream) {
    using namespace comfy::hip_backend::flash_attn;
    Args a{q, k, v, mask, kv_lengths, o, o_accum, lse_accum, heads_kv, groups, q_len, kv_len,
           num_splits, mask_code,
           strides[0], strides[1], strides[2], strides[3], strides[4], strides[5],
           strides[6], strides[7], strides[8], strides[9], strides[10], strides[11],
           strides[12], strides[13], strides[14], strides[15], scale, stream};
    // 3: fp32 Q and O over an fp16 KV cache, decode rows at HD 64 and 128 only
    if (dtype_code == 3) head_dim == 64 ? launch_direct<float, __half, 64>(a, batch) : launch_direct<float, __half, 128>(a, batch);
    else if (dtype_code == 0) launch_dims<float>(a, batch, head_dim, few_rows);
    else if (dtype_code == 1) launch_dims<__half>(a, batch, head_dim, few_rows);
    else launch_dims<__bf16>(a, batch, head_dim, few_rows);
}
