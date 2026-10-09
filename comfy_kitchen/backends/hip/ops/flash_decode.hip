// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// BF16 decode attention over a fixed-capacity KV cache. The CUDA counterpart
// reuses FlashAttention 2.7.4's split-KV kernels, so nothing there translates;
// this is an independent kernel behind the same launcher and the same
// split-then-combine structure.
//
// The query rows of one KV head are a (group, step) grid: `groups` query heads
// share the head and each carries `steps` tokens (a speculative verify step;
// steps == 1 is plain decode). The seq-first entry folds the groups into the
// row dimension with steps == 1; the head-first GQA entry passes both. With
// `causal` row (g, s) attends slots < length - steps + s + 1, the bottom-right
// staircase the CUDA GQA kernel uses; otherwise every row attends slots < length.
//
// Arithmetic intensity is `groups` FLOP per byte of K and V, so the shape is
// picked for those reads: a wave walks one key row at a time, keeping every
// load coalesced, and closes the dot product with a butterfly reduction. The
// dot product is plain FMA, so this needs no matrix cores.
#include <hip/hip_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

#include "../launchers.h"
#include "../mma.h"

namespace comfy::hip_backend {
namespace {

// wave_sum below is a fixed wave32 DPP sequence.
static_assert(kWave == 32, "the reduction and the lane layout assume wave32");

// One wave per (batch, head, split) leaves too few waves to cover a cache miss,
// and the split count is the caller's and capped. The waves of a block share a
// key range instead and merge their running softmax at the end. Eight measured
// best: four costs 24%, sixteen exceeds LDS at the widest group tile.
constexpr int kWavesPerBlock = 8;
constexpr int kBlockThreads = kWavesPerBlock * kWave;

// Wider heads and group tiles leave fewer registers for prefetched K/V.
template <int HeadDim, int GroupsPerPass>
constexpr int keys_per_iter() {
    return (GroupsPerPass >= 8 ? 4 : (GroupsPerPass >= 4 ? 8 : 16)) * 128 / HeadDim;
}

constexpr float kLn2 = 0.693147180559945309417f;

// Finite, not -inf: under -ffast-math the first m - m would be NaN.
constexpr float kNegInf = -FLT_MAX / 4.0f;

typedef __bf16 v4bf __attribute__((ext_vector_type(4)));
template <int Width>
using FloatVec = float __attribute__((ext_vector_type(Width)));

template <int Width>
__forceinline__ __device__ FloatVec<Width> load_bf16(const __bf16* p) {
    FloatVec<Width> result;
#pragma unroll
    for (int i = 0; i < Width; i += 4) {
        const v4bf raw = *reinterpret_cast<const v4bf*>(p + i);
#pragma unroll
        for (int d = 0; d < 4; ++d) result[i + d] = static_cast<float>(raw[d]);
    }
    return result;
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
template <int HeadDim, int GroupsPerPass, bool Split>
__global__ __launch_bounds__(kBlockThreads) void flash_decode_kernel(
    const __bf16* __restrict__ q, const __bf16* __restrict__ k, const __bf16* __restrict__ v,
    const int* __restrict__ kv_lengths, __bf16* __restrict__ out,
    float* __restrict__ softmax_lse, float* __restrict__ out_accum,
    float* __restrict__ lse_accum, int rows, int query_length, int steps, int heads,
    int kv_capacity, int num_splits, bool causal,
    int64_t q_batch_stride, int64_t q_group_stride, int64_t q_step_stride, int64_t q_head_stride,
    int64_t k_batch_stride, int64_t k_row_stride, int64_t k_head_stride,
    int64_t o_batch_stride, int64_t o_group_stride, int64_t o_step_stride, int64_t o_head_stride) {
    constexpr int kDimPerLane = HeadDim / kWave;
    constexpr int kKeysPerIter = keys_per_iter<HeadDim, GroupsPerPass>();
    constexpr float kSoftmaxScaleLog2 = HeadDim == 128 ? 0.12751743082871335f : 0.09016844005556021f;
    using Vec = FloatVec<kDimPerLane>;
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

    Vec q_vec[GroupsPerPass];
    Vec o_acc[GroupsPerPass];
    float m_run[GroupsPerPass];
    float d_run[GroupsPerPass];
    int limit[GroupsPerPass];
#pragma unroll
    for (int i = 0; i < GroupsPerPass; ++i) {
        const int r = group_base + i;
        const int g = r / steps;
        const int s = r - g * steps;
        q_vec[i] = r < query_length
                       ? load_bf16<kDimPerLane>(q + q_base + g * q_group_stride + s * q_step_stride)
                       : Vec{};
        o_acc[i] = Vec{};
        m_run[i] = kNegInf;
        d_run[i] = 0.0f;
        limit[i] = causal ? length - steps + s + 1 : length;
    }

    const int64_t kv_base = static_cast<int64_t>(b) * k_batch_stride +
                            static_cast<int64_t>(h) * k_head_stride + lane * kDimPerLane;

    const int stride = kWavesPerBlock * kKeysPerIter;
    for (int j0 = begin + wave * kKeysPerIter; j0 < end; j0 += stride) {
        // This prefetch is the only memory-level parallelism available. The
        // index is clamped rather than predicated so no load sits behind a
        // branch the scheduler cannot hoist past.
        Vec k_vec[kKeysPerIter];
        Vec v_vec[kKeysPerIter];
#pragma unroll
        for (int c = 0; c < kKeysPerIter; ++c) {
            const int j = j0 + c;
            const int clamped = j < end ? j : end - 1;
            const int64_t off = kv_base + static_cast<int64_t>(clamped) * k_row_stride;
            k_vec[c] = load_bf16<kDimPerLane>(k + off);
            v_vec[c] = load_bf16<kDimPerLane>(v + off);
        }

#pragma unroll
        for (int c = 0; c < kKeysPerIter; ++c) {
            const int j = j0 + c;
#pragma unroll
            for (int i = 0; i < GroupsPerPass; ++i) {
                const bool valid = j < end && j < limit[i];
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
    __shared__ float s_out[kWavesPerBlock][GroupsPerPass][HeadDim];
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
        Vec accum = Vec{};
#pragma unroll
        for (int w = 0; w < kWavesPerBlock; ++w) {
            const float rescale = exp2f(s_max[w][i] - m);
            denominator = fmaf(s_denominator[w][i], rescale, denominator);
#pragma unroll
            for (int d = 0; d < kDimPerLane; ++d) {
                accum[d] = fmaf(s_out[w][i][lane * kDimPerLane + d], rescale, accum[d]);
            }
        }

        const int r = group_base + i;
        if (r >= query_length) continue;
        const int g = r / steps;
        const int s = r - g * steps;
        // An empty range leaves the denominator at zero; write zeros, not 0/0.
        const float inv = denominator > 0.0f ? 1.0f / denominator : 0.0f;
        const int row = (b * heads + h) * query_length + r;
        if constexpr (Split) {
            const int64_t slot = static_cast<int64_t>(split) * rows + row;
            float* dst = out_accum + slot * HeadDim + lane * kDimPerLane;
#pragma unroll
            for (int d = 0; d < kDimPerLane; ++d) dst[d] = accum[d] * inv;
            if (lane == 0) {
                lse_accum[slot] = denominator > 0.0f ? m + log2f(denominator) : kNegInf;
            }
        } else {
            __bf16* dst = out + b * o_batch_stride + g * o_group_stride + s * o_step_stride +
                          h * o_head_stride + lane * kDimPerLane;
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
template <int HeadDim>
__global__ __launch_bounds__(HeadDim) void flash_decode_combine_kernel(
    const float* __restrict__ out_accum, const float* __restrict__ lse_accum,
    __bf16* __restrict__ out, float* __restrict__ softmax_lse, int num_splits, int rows,
    int query_length, int steps, int heads, int64_t o_batch_stride, int64_t o_group_stride,
    int64_t o_step_stride, int64_t o_head_stride) {
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
        numerator = fmaf(weight, out_accum[slot * HeadDim + d], numerator);
    }

    const int r = row % query_length;
    const int g = r / steps;
    const int s = r - g * steps;
    const int h = (row / query_length) % heads;
    const int b = row / (query_length * heads);
    out[b * o_batch_stride + g * o_group_stride + s * o_step_stride + h * o_head_stride + d] =
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

// Every launch below carries the same operand list; it is wide because the
// kernel addresses q, k and out through independent strides.
struct DecodeArgs {
    const void* q;
    const void* k;
    const void* v;
    const int* kv_lengths;
    void* out;
    float* softmax_lse;
    float* out_accum;
    float* lse_accum;
    int batch;
    int query_length;
    int steps;
    int heads;
    int kv_capacity;
    int num_splits;
    bool causal;
    int64_t q_batch_stride, q_group_stride, q_step_stride, q_head_stride;
    int64_t k_batch_stride, k_row_stride, k_head_stride;
    int64_t o_batch_stride, o_group_stride, o_step_stride, o_head_stride;
    hipStream_t stream;
};

template <int HeadDim, int GroupsPerPass>
void launch_typed(const DecodeArgs& a) {
    const dim3 grid(a.query_length / GroupsPerPass, a.num_splits, a.batch * a.heads);
    const int rows = a.batch * a.heads * a.query_length;
    const bool split = a.num_splits > 1;
    if (split) {
        flash_decode_kernel<HeadDim, GroupsPerPass, true><<<grid, kBlockThreads, 0, a.stream>>>(
            static_cast<const __bf16*>(a.q), static_cast<const __bf16*>(a.k),
            static_cast<const __bf16*>(a.v), a.kv_lengths, static_cast<__bf16*>(a.out),
            a.softmax_lse, a.out_accum, a.lse_accum, rows, a.query_length, a.steps, a.heads,
            a.kv_capacity, a.num_splits, a.causal, a.q_batch_stride, a.q_group_stride,
            a.q_step_stride, a.q_head_stride, a.k_batch_stride, a.k_row_stride, a.k_head_stride,
            a.o_batch_stride, a.o_group_stride, a.o_step_stride, a.o_head_stride);
        flash_decode_combine_kernel<HeadDim><<<rows, HeadDim, 0, a.stream>>>(
            a.out_accum, a.lse_accum, static_cast<__bf16*>(a.out), a.softmax_lse, a.num_splits,
            rows, a.query_length, a.steps, a.heads, a.o_batch_stride, a.o_group_stride,
            a.o_step_stride, a.o_head_stride);
    } else {
        flash_decode_kernel<HeadDim, GroupsPerPass, false><<<grid, kBlockThreads, 0, a.stream>>>(
            static_cast<const __bf16*>(a.q), static_cast<const __bf16*>(a.k),
            static_cast<const __bf16*>(a.v), a.kv_lengths, static_cast<__bf16*>(a.out),
            a.softmax_lse, a.out_accum, a.lse_accum, rows, a.query_length, a.steps, a.heads,
            a.kv_capacity, a.num_splits, a.causal, a.q_batch_stride, a.q_group_stride,
            a.q_step_stride, a.q_head_stride, a.k_batch_stride, a.k_row_stride, a.k_head_stride,
            a.o_batch_stride, a.o_group_stride, a.o_step_stride, a.o_head_stride);
    }
}

template <int HeadDim>
void launch_groups(const DecodeArgs& a) {
    switch (groups_per_pass(a.query_length)) {
        case 8:
            // Cap the 256-dimensional tile at four groups to stay within the LDS budget.
            launch_typed<HeadDim, HeadDim == 128 ? 8 : 4>(a);
            break;
        case 4:
            launch_typed<HeadDim, 4>(a);
            break;
        case 2:
            launch_typed<HeadDim, 2>(a);
            break;
        default:
            launch_typed<HeadDim, 1>(a);
            break;
    }
}

}  // namespace
}  // namespace comfy::hip_backend

extern "C" void launch_flash_decode(const void* q, const void* k, const void* v,
                                    const int* kv_lengths, void* out, float* softmax_lse,
                                    float* out_accum, float* lse_accum, int batch,
                                    int query_length, int steps, int heads, int head_dim,
                                    int kv_capacity, int num_splits, bool causal,
                                    int64_t q_batch_stride, int64_t q_group_stride,
                                    int64_t q_step_stride, int64_t q_head_stride,
                                    int64_t k_batch_stride, int64_t k_row_stride,
                                    int64_t k_head_stride, int64_t o_batch_stride,
                                    int64_t o_group_stride, int64_t o_step_stride,
                                    int64_t o_head_stride, hipStream_t stream) {
    const comfy::hip_backend::DecodeArgs a{
        q, k, v, kv_lengths, out, softmax_lse, out_accum, lse_accum, batch, query_length, steps,
        heads, kv_capacity, num_splits, causal, q_batch_stride, q_group_stride, q_step_stride,
        q_head_stride, k_batch_stride, k_row_stride, k_head_stride, o_batch_stride,
        o_group_stride, o_step_stride, o_head_stride, stream};
    if (head_dim == 128) {
        comfy::hip_backend::launch_groups<128>(a);
    } else {
        comfy::hip_backend::launch_groups<256>(a);
    }
}

// Current-step rows: the prefix decode pass returned its output and natural-log LSE; row j's own token and
// the earlier rows of this step are the slots that prefix skips. Fold them in as
// further softmax columns, the same arithmetic as the CUDA kernel:
//   s_t = scale * q . k_t;  m = max(lse, max_t s_t)
//   o = (out * e^(lse - m) + sum_t v_t * e^(s_t - m)) / (e^(lse - m) + sum_t e^(s_t - m))
// over the step rows t <= j.
// One wave per (batch, row, head), eight dims per lane, fp32 math, bf16 out.
namespace comfy::hip_backend {
namespace {

constexpr int kMergeHeadDim = 256;
constexpr int kMergeWaves = 8;
constexpr int kMergeMaxRows = 8;
constexpr int kMergeDims = kMergeHeadDim / kWave;

__global__ __launch_bounds__(kMergeWaves * kWave) void flash_decode_step_merge_kernel(
    const __bf16* __restrict__ out, const float* __restrict__ lse, const __bf16* __restrict__ q,
    const __bf16* __restrict__ k, const __bf16* __restrict__ v, __bf16* __restrict__ merged,
    int total, int rows, int heads, int groups, float scale,
    int64_t q_bs, int64_t q_hs, int64_t q_rs, int64_t k_bs, int64_t k_hs, int64_t k_rs,
    int64_t v_bs, int64_t v_hs, int64_t v_rs, int64_t o_bs, int64_t o_rs, int64_t m_bs,
    int64_t m_rs) {
    const int lane = static_cast<int>(threadIdx.x) % kWave;
    const int idx = static_cast<int>(blockIdx.x) * kMergeWaves + static_cast<int>(threadIdx.x) / kWave;
    if (idx >= total) return;
    const int h = idx % heads;
    const int j = (idx / heads) % rows;
    const int b = idx / (heads * rows);
    const int kh = h / groups;
    const int d0 = lane * kMergeDims;

    const FloatVec<kMergeDims> qv = load_bf16<kMergeDims>(q + b * q_bs + h * q_hs + j * q_rs + d0);
    float s[kMergeMaxRows];
    const float l = lse[(b * heads + h) * rows + j];
    float m = l;
#pragma unroll
    for (int t = 0; t < kMergeMaxRows; ++t) {
        s[t] = -INFINITY;
        if (t <= j) {
            const FloatVec<kMergeDims> kv = load_bf16<kMergeDims>(k + b * k_bs + kh * k_hs + t * k_rs + d0);
            float dot = 0.0f;
#pragma unroll
            for (int d = 0; d < kMergeDims; ++d) dot += qv[d] * kv[d];
            s[t] = wave_sum(dot) * scale;
            m = fmaxf(m, s[t]);
        }
    }

    const float w_flash = expf(l - m);
    float denom = w_flash;
    FloatVec<kMergeDims> acc = load_bf16<kMergeDims>(out + b * o_bs + j * o_rs + h * kMergeHeadDim + d0);
#pragma unroll
    for (int d = 0; d < kMergeDims; ++d) acc[d] *= w_flash;
#pragma unroll
    for (int t = 0; t < kMergeMaxRows; ++t) {
        if (t <= j) {
            const float w = expf(s[t] - m);
            denom += w;
            const FloatVec<kMergeDims> vv = load_bf16<kMergeDims>(v + b * v_bs + kh * v_hs + t * v_rs + d0);
#pragma unroll
            for (int d = 0; d < kMergeDims; ++d) acc[d] = fmaf(vv[d], w, acc[d]);
        }
    }
    const float inv = 1.0f / denom;
    __bf16* dst = merged + b * m_bs + j * m_rs + h * kMergeHeadDim + d0;
#pragma unroll
    for (int d = 0; d < kMergeDims; ++d) dst[d] = static_cast<__bf16>(acc[d] * inv);
}

}  // namespace
}  // namespace comfy::hip_backend

extern "C" void launch_flash_decode_step_merge(
    const void* out, const float* lse, const void* q, const void* k, const void* v,
    void* merged, int batch, int rows, int heads, int kv_heads, int64_t q_batch_stride,
    int64_t q_head_stride, int64_t q_row_stride, int64_t k_batch_stride, int64_t k_head_stride,
    int64_t k_row_stride, int64_t v_batch_stride, int64_t v_head_stride, int64_t v_row_stride,
    int64_t o_batch_stride, int64_t o_row_stride, int64_t m_batch_stride, int64_t m_row_stride,
    hipStream_t stream) {
    using namespace comfy::hip_backend;
    const int waves = batch * rows * heads;
    flash_decode_step_merge_kernel<<<(waves + kMergeWaves - 1) / kMergeWaves, kMergeWaves * kWave, 0,
                                     stream>>>(
        static_cast<const __bf16*>(out), lse, static_cast<const __bf16*>(q),
        static_cast<const __bf16*>(k), static_cast<const __bf16*>(v),
        static_cast<__bf16*>(merged), waves, rows, heads, heads / kv_heads,
        1.0f / sqrtf(static_cast<float>(kMergeHeadDim)), q_batch_stride, q_head_stride,
        q_row_stride, k_batch_stride, k_head_stride, k_row_stride, v_batch_stride, v_head_stride,
        v_row_stride, o_batch_stride, o_row_stride, m_batch_stride, m_row_stride);
}
