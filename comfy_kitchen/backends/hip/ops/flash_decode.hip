// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// BF16 decode attention over a fixed-capacity KV cache. The CUDA counterpart
// reuses FlashAttention 2.7.4's split-KV kernels, so nothing there translates;
// this is an independent kernel behind the same launcher and the same
// split-then-combine structure.
//
// The Python layer folds the GQA group count into the query sequence dimension:
// query_length == groups, one KV head per query row, no mask, no window.
//
// Arithmetic intensity is `groups` FLOP per byte of K and V, so the shape is
// picked for those reads: a wave walks one key row at a time, keeping every
// load coalesced, and closes the dot product with a butterfly reduction. The
// dot product is plain FMA, so this needs no matrix cores.
#include <hip/hip_runtime.h>

#include <cfloat>
#include <cstdint>

#include "../launchers.h"
#include "../mma.h"

namespace comfy::hip_backend {
namespace {

constexpr int kHeadDim = 128;
constexpr int kDimPerLane = kHeadDim / kWave;
constexpr int kCombineThreads = kHeadDim;

// wave_sum below is a fixed wave32 DPP sequence, and four elements per lane is
// what makes the head dimension divide evenly.
static_assert(kWave == 32, "the reduction and the lane layout assume wave32");
static_assert(kDimPerLane == 4, "a lane holds four elements of the head dim");

// One wave per (batch, head, split) leaves too few waves to cover a cache miss,
// and the split count is the caller's and capped. The waves of a block share a
// key range instead and merge their running softmax at the end. Eight measured
// best: four costs 24%, sixteen exceeds LDS at the widest group tile.
constexpr int kWavesPerBlock = 8;
constexpr int kBlockThreads = kWavesPerBlock * kWave;

// Keys in flight per wave, measured. Wider group tiles reuse each row more and
// cannot spare the registers. Halving costs 5% at the narrowest tile, doubling
// is within noise, doubling twice spills and costs 2.6x.
template <int GroupsPerPass>
constexpr int keys_per_iter() {
    return GroupsPerPass >= 8 ? 4 : (GroupsPerPass >= 4 ? 8 : 16);
}

// 1/sqrt(128) folded into log2 so the softmax can use exp2. Mirrors
// scale_softmax_log2 in the CUDA launcher.
constexpr float kSoftmaxScaleLog2 = 0.12751743082871335f;
constexpr float kLn2 = 0.693147180559945309417f;

// Finite, not -inf: under -ffast-math the first m - m would be NaN.
constexpr float kNegInf = -FLT_MAX / 4.0f;

typedef __bf16 v4bf __attribute__((ext_vector_type(4)));
typedef float v4f __attribute__((ext_vector_type(4)));

__forceinline__ __device__ v4f zero4() { return v4f{0.0f, 0.0f, 0.0f, 0.0f}; }

__forceinline__ __device__ v4f load_bf16x4(const __bf16* p) {
    const v4bf raw = *reinterpret_cast<const v4bf*>(p);
    return v4f{static_cast<float>(raw[0]), static_cast<float>(raw[1]),
               static_cast<float>(raw[2]), static_cast<float>(raw[3])};
}

// Sum across the wave without touching LDS. mma.h's wave_reduce_sum lowers
// __shfl_xor to ds_bpermute plus an LDS wait, five times per key per group,
// which dominates once the group tile is wide. Bit-identical to that butterfly:
// each step equalizes a wider span, so the next mirror acts as an xor swap.
template <int Ctrl>
__forceinline__ __device__ float dpp_add(float v) {
    const int bits = __builtin_bit_cast(int, v);
    return v + __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, bits, Ctrl, 0xf, 0xf, true));
}

__forceinline__ __device__ float wave_sum(float v) {
    v = dpp_add<0xB1>(v);
    v = dpp_add<0x4E>(v);
    v = dpp_add<0x141>(v);
    v = dpp_add<0x140>(v);
    const int bits = __builtin_bit_cast(int, v);
    return v + __builtin_bit_cast(
                   float, __builtin_amdgcn_permlanex16(bits, bits, 0x76543210, 0xfedcba98, false,
                                                       false));
}

// Splits partition [0, length). A split past the end still runs the epilogue:
// the combine pass reads every slot and would otherwise read uninitialized
// memory.
__forceinline__ __device__ void split_range(int length, int split, int num_splits, int* begin,
                                            int* end) {
    const int per_split = (length + num_splits - 1) / num_splits;
    const int lo = split * per_split;
    const int hi = lo + per_split;
    *begin = lo < length ? lo : length;
    *end = hi < length ? hi : length;
}

// GroupsPerPass query rows share every K and V read. Templated because the
// per-group state has to stay in registers; a runtime bound spills it.
template <int GroupsPerPass, bool Split>
__global__ __launch_bounds__(kBlockThreads) void flash_decode_kernel(
    const __bf16* __restrict__ q, const __bf16* __restrict__ k, const __bf16* __restrict__ v,
    const int* __restrict__ kv_lengths, __bf16* __restrict__ out,
    float* __restrict__ softmax_lse, float* __restrict__ out_accum,
    float* __restrict__ lse_accum, int rows, int query_length, int heads, int kv_capacity,
    int num_splits,
    int64_t q_batch_stride, int64_t q_row_stride, int64_t q_head_stride, int64_t k_batch_stride,
    int64_t k_row_stride, int64_t k_head_stride) {
    constexpr int kKeysPerIter = keys_per_iter<GroupsPerPass>();
    const int lane = static_cast<int>(threadIdx.x) % kWave;
    const int wave = static_cast<int>(threadIdx.x) / kWave;
    const int group_base = static_cast<int>(blockIdx.x) * GroupsPerPass;
    const int split = static_cast<int>(blockIdx.y);
    const int bh = static_cast<int>(blockIdx.z);
    const int b = bh / heads;
    const int h = bh - b * heads;

    // A device value the caller may rewrite between graph replays.
    int length = kv_lengths[b];
    length = length < 0 ? 0 : (length > kv_capacity ? kv_capacity : length);

    int begin = 0;
    int end = length;
    if constexpr (Split) split_range(length, split, num_splits, &begin, &end);

    const int64_t q_base = static_cast<int64_t>(b) * q_batch_stride +
                           static_cast<int64_t>(h) * q_head_stride + lane * kDimPerLane;

    v4f q_vec[GroupsPerPass];
    v4f o_acc[GroupsPerPass];
    float m_run[GroupsPerPass];
    float d_run[GroupsPerPass];
#pragma unroll
    for (int i = 0; i < GroupsPerPass; ++i) {
        const int g = group_base + i;
        q_vec[i] = g < query_length ? load_bf16x4(q + q_base + g * q_row_stride) : zero4();
        o_acc[i] = zero4();
        m_run[i] = kNegInf;
        d_run[i] = 0.0f;
    }

    const int64_t kv_base = static_cast<int64_t>(b) * k_batch_stride +
                            static_cast<int64_t>(h) * k_head_stride + lane * kDimPerLane;

    const int stride = kWavesPerBlock * kKeysPerIter;
    for (int j0 = begin + wave * kKeysPerIter; j0 < end; j0 += stride) {
        // This prefetch is the only memory-level parallelism available. The
        // index is clamped rather than predicated so no load sits behind a
        // branch the scheduler cannot hoist past.
        v4f k_vec[kKeysPerIter];
        v4f v_vec[kKeysPerIter];
#pragma unroll
        for (int c = 0; c < kKeysPerIter; ++c) {
            const int j = j0 + c;
            const int clamped = j < end ? j : end - 1;
            const int64_t off = kv_base + static_cast<int64_t>(clamped) * k_row_stride;
            k_vec[c] = load_bf16x4(k + off);
            v_vec[c] = load_bf16x4(v + off);
        }

#pragma unroll
        for (int c = 0; c < kKeysPerIter; ++c) {
            const bool valid = (j0 + c) < end;
#pragma unroll
            for (int i = 0; i < GroupsPerPass; ++i) {
                float dot = 0.0f;
#pragma unroll
                for (int d = 0; d < kDimPerLane; ++d) dot += q_vec[i][d] * k_vec[c][d];
                dot = wave_sum(dot) * kSoftmaxScaleLog2;

                // A padded key must not raise the running maximum, and needs
                // probability zero rather than exp2(0).
                const float score = valid ? dot : kNegInf;
                const float m_new = fmaxf(m_run[i], score);
                const float rescale = exp2f(m_run[i] - m_new);
                const float p = valid ? exp2f(score - m_new) : 0.0f;
                d_run[i] = fmaf(d_run[i], rescale, p);
#pragma unroll
                for (int d = 0; d < kDimPerLane; ++d) {
                    o_acc[i][d] = fmaf(o_acc[i][d], rescale, p * v_vec[c][d]);
                }
                m_run[i] = m_new;
            }
        }
    }

    // Merge the waves' running softmaxes. Each covered a disjoint key set, so
    // this is the same rescale-by-exp2 the split combine does.
    __shared__ float s_max[kWavesPerBlock][GroupsPerPass];
    __shared__ float s_denominator[kWavesPerBlock][GroupsPerPass];
    __shared__ float s_out[kWavesPerBlock][GroupsPerPass][kHeadDim];
#pragma unroll
    for (int i = 0; i < GroupsPerPass; ++i) {
        if (lane == 0) {
            s_max[wave][i] = m_run[i];
            s_denominator[wave][i] = d_run[i];
        }
#pragma unroll
        for (int d = 0; d < kDimPerLane; ++d) {
            s_out[wave][i][lane * kDimPerLane + d] = o_acc[i][d];
        }
    }
    __syncthreads();
    if (wave != 0) return;

#pragma unroll
    for (int i = 0; i < GroupsPerPass; ++i) {
        float m = s_max[0][i];
#pragma unroll
        for (int w = 1; w < kWavesPerBlock; ++w) m = fmaxf(m, s_max[w][i]);
        float denominator = 0.0f;
        v4f accum = zero4();
#pragma unroll
        for (int w = 0; w < kWavesPerBlock; ++w) {
            const float rescale = exp2f(s_max[w][i] - m);
            denominator = fmaf(s_denominator[w][i], rescale, denominator);
#pragma unroll
            for (int d = 0; d < kDimPerLane; ++d) {
                accum[d] = fmaf(s_out[w][i][lane * kDimPerLane + d], rescale, accum[d]);
            }
        }

        const int g = group_base + i;
        if (g >= query_length) continue;
        // An empty range leaves the denominator at zero; write zeros, not 0/0.
        const float inv = denominator > 0.0f ? 1.0f / denominator : 0.0f;
        const int row = (b * heads + h) * query_length + g;
        if constexpr (Split) {
            const int64_t slot = static_cast<int64_t>(split) * rows + row;
            float* dst = out_accum + slot * kHeadDim + lane * kDimPerLane;
#pragma unroll
            for (int d = 0; d < kDimPerLane; ++d) dst[d] = accum[d] * inv;
            if (lane == 0) {
                lse_accum[slot] = denominator > 0.0f ? m + log2f(denominator) : kNegInf;
            }
        } else {
            __bf16* dst = out + b * q_batch_stride + g * q_row_stride + h * q_head_stride +
                          lane * kDimPerLane;
#pragma unroll
            for (int d = 0; d < kDimPerLane; ++d) dst[d] = static_cast<__bf16>(accum[d] * inv);
            // Natural log, matching the CUDA path, though the softmax is base 2.
            if (lane == 0) {
                softmax_lse[row] =
                    denominator > 0.0f ? (m + log2f(denominator)) * kLn2 : -FLT_MAX;
            }
        }
    }
}

// One block per output row, one lane per head dimension.
__global__ __launch_bounds__(kCombineThreads) void flash_decode_combine_kernel(
    const float* __restrict__ out_accum, const float* __restrict__ lse_accum,
    __bf16* __restrict__ out, float* __restrict__ softmax_lse, int num_splits, int rows,
    int query_length, int heads, int64_t q_batch_stride, int64_t q_row_stride,
    int64_t q_head_stride) {
    const int row = static_cast<int>(blockIdx.x);
    const int d = static_cast<int>(threadIdx.x);

    float m = kNegInf;
    for (int s = 0; s < num_splits; ++s) {
        m = fmaxf(m, lse_accum[static_cast<int64_t>(s) * rows + row]);
    }

    float denominator = 0.0f;
    float numerator = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
        const int64_t slot = static_cast<int64_t>(s) * rows + row;
        const float weight = exp2f(lse_accum[slot] - m);
        denominator += weight;
        numerator = fmaf(weight, out_accum[slot * kHeadDim + d], numerator);
    }

    const int g = row % query_length;
    const int h = (row / query_length) % heads;
    const int b = row / (query_length * heads);
    out[b * q_batch_stride + g * q_row_stride + h * q_head_stride + d] =
        static_cast<__bf16>(denominator > 0.0f ? numerator / denominator : 0.0f);
    if (d == 0) {
        softmax_lse[row] = denominator > 0.0f ? (m + log2f(denominator)) * kLn2 : -FLT_MAX;
    }
}

// Largest power of two up to kMaxGroupsPerPass dividing query_length, so no
// pass has to mask a partial group tile.
constexpr int kMaxGroupsPerPass = 8;

int groups_per_pass(int query_length) {
    int gpp = 1;
    while (gpp < kMaxGroupsPerPass && query_length % (gpp * 2) == 0) gpp *= 2;
    return gpp;
}

template <int GroupsPerPass>
void launch_typed(const void* q, const void* k, const void* v, const int* kv_lengths, void* out,
                  float* softmax_lse, float* out_accum, float* lse_accum, int batch,
                  int query_length, int heads, int kv_capacity, int num_splits,
                  int64_t q_batch_stride, int64_t q_row_stride, int64_t q_head_stride,
                  int64_t k_batch_stride, int64_t k_row_stride, int64_t k_head_stride,
                  hipStream_t stream) {
    const dim3 grid(query_length / GroupsPerPass, num_splits, batch * heads);
    const int rows = batch * heads * query_length;
    const bool split = num_splits > 1;
    if (split) {
        flash_decode_kernel<GroupsPerPass, true><<<grid, kBlockThreads, 0, stream>>>(
            static_cast<const __bf16*>(q), static_cast<const __bf16*>(k),
            static_cast<const __bf16*>(v), kv_lengths, static_cast<__bf16*>(out), softmax_lse,
            out_accum, lse_accum, rows, query_length, heads, kv_capacity, num_splits,
            q_batch_stride, q_row_stride, q_head_stride, k_batch_stride, k_row_stride,
            k_head_stride);
        flash_decode_combine_kernel<<<rows, kCombineThreads, 0, stream>>>(
            out_accum, lse_accum, static_cast<__bf16*>(out), softmax_lse, num_splits, rows,
            query_length, heads, q_batch_stride, q_row_stride, q_head_stride);
    } else {
        flash_decode_kernel<GroupsPerPass, false><<<grid, kBlockThreads, 0, stream>>>(
            static_cast<const __bf16*>(q), static_cast<const __bf16*>(k),
            static_cast<const __bf16*>(v), kv_lengths, static_cast<__bf16*>(out), softmax_lse,
            out_accum, lse_accum, rows, query_length, heads, kv_capacity, num_splits,
            q_batch_stride, q_row_stride, q_head_stride, k_batch_stride, k_row_stride,
            k_head_stride);
    }
}

}  // namespace
}  // namespace comfy::hip_backend

extern "C" void launch_flash_decode(const void* q, const void* k, const void* v,
                                    const int* kv_lengths, void* out, float* softmax_lse,
                                    float* out_accum, float* lse_accum, int batch,
                                    int query_length, int heads, int kv_capacity, int num_splits,
                                    int64_t q_batch_stride, int64_t q_row_stride,
                                    int64_t q_head_stride, int64_t k_batch_stride,
                                    int64_t k_row_stride, int64_t k_head_stride,
                                    hipStream_t stream) {
    using namespace comfy::hip_backend;
    switch (groups_per_pass(query_length)) {
        case 8:
            launch_typed<8>(q, k, v, kv_lengths, out, softmax_lse, out_accum, lse_accum, batch,
                            query_length, heads, kv_capacity, num_splits, q_batch_stride,
                            q_row_stride, q_head_stride, k_batch_stride, k_row_stride,
                            k_head_stride, stream);
            break;
        case 4:
            launch_typed<4>(q, k, v, kv_lengths, out, softmax_lse, out_accum, lse_accum, batch,
                            query_length, heads, kv_capacity, num_splits, q_batch_stride,
                            q_row_stride, q_head_stride, k_batch_stride, k_row_stride,
                            k_head_stride, stream);
            break;
        case 2:
            launch_typed<2>(q, k, v, kv_lengths, out, softmax_lse, out_accum, lse_accum, batch,
                            query_length, heads, kv_capacity, num_splits, q_batch_stride,
                            q_row_stride, q_head_stride, k_batch_stride, k_row_stride,
                            k_head_stride, stream);
            break;
        default:
            launch_typed<1>(q, k, v, kv_lengths, out, softmax_lse, out_accum, lse_accum, batch,
                            query_length, heads, kv_capacity, num_splits, q_batch_stride,
                            q_row_stride, q_head_stride, k_batch_stride, k_row_stride,
                            k_head_stride, stream);
            break;
    }
}
