// SPDX-License-Identifier: BSD-3-Clause

#include <cuda_runtime.h>

#include "utils.cuh"

#ifndef M_LOG2E
#define M_LOG2E 1.4426950408889634074
#endif

#include "flash.h"
#include "flash_fwd_kernel.h"

namespace flash {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define COMFY_FLASH_BODY(...) __VA_ARGS__
#define COMFY_FLASH_PARAM __grid_constant__
#else
#define COMFY_FLASH_BODY(...)
#define COMFY_FLASH_PARAM
#endif

// Single-query decode and bottom-right causal multi-query decode (a speculative verify
// step, seqlen_q <= kBlockM) share the split-KV kernel; both predicate their Q/K/V loads
// and O stores (Is_even_MN=false) since seqlen_q and kv_length are never tile multiples.
using Traits128 = Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass::bfloat16_t>;
using Traits256 = Flash_fwd_kernel_traits<256, 64, 64, 4, false, false, cutlass::bfloat16_t>;

template<typename Traits, bool Is_causal, bool Split>
__global__ void flash_decode_kernel(COMFY_FLASH_PARAM const Flash_fwd_params params) {
    COMFY_FLASH_BODY((compute_attn_splitkv<Traits, Is_causal, false, false, false, true, false, Split, false>(params));)
}

template<typename Traits, int LogMaxSplits>
__global__ void flash_decode_combine_kernel(COMFY_FLASH_PARAM const Flash_fwd_params params) {
    COMFY_FLASH_BODY((combine_attn_seqk_parallel<Traits, 4, LogMaxSplits, true>(params));)
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// One warp folds the num_splits partials of query row `row` of (bidb, bidh) into the bf16 output,
// reproducing combine_attn_seqk_parallel<Traits, 4, LogMaxSplits> arithmetic exactly: lane s holds
// split s's lse (-inf beyond num_splits), the max and the exp-sum run through the same
// Width-lane xor butterflies (so each lane's logsumexp carries that lane's rounding, as there),
// and the output accumulates split 0..n-1 in order from zero. Width is the kMaxSplits the
// separate launch would have chosen for this num_splits.
template<typename Traits, int Width>
__device__ __forceinline__ void combine_row_warp(const Flash_fwd_params& params, const int bidb, const int bidh, const int row, const int lane) {
    constexpr int kHeadDim = Traits::kHeadDim;
    constexpr int kPerLane = kHeadDim / 32;
    const int lse_size = params.b * params.h * params.seqlen_q;
    const int row_index = (bidb * params.h + bidh) * params.seqlen_q + row;
    const float* lse_accum = reinterpret_cast<const float*>(params.softmax_lseaccum_ptr) + row_index;
    const float* o_accum = reinterpret_cast<const float*>(params.oaccum_ptr) + static_cast<int64_t>(row_index) * params.d_rounded;
    const int64_t split_stride = static_cast<int64_t>(lse_size) * params.d_rounded;

    const int split = lane % Width;
    float lse = split < params.num_splits ? __ldcg(lse_accum + static_cast<int64_t>(split) * lse_size) : -INFINITY;
    MaxOp<float> max_op;
    float lse_max = Allreduce<Width>::run(lse, max_op);
    lse_max = lse_max == -INFINITY ? 0.0f : lse_max;
    float lse_sum = expf(lse - lse_max);
    SumOp<float> sum_op;
    lse_sum = Allreduce<Width>::run(lse_sum, sum_op);
    const float lse_logsum = (lse_sum == 0.f || lse_sum != lse_sum) ? INFINITY : logf(lse_sum) + lse_max;
    if (lane == 0) {
        reinterpret_cast<float*>(params.softmax_lse_ptr)[row_index] = lse_logsum;
    }
    const float scale = expf(lse - lse_logsum);

    float acc[kPerLane];
    #pragma unroll
    for (int i = 0; i < kPerLane; ++i) { acc[i] = 0.0f; }
    for (int s = 0; s < params.num_splits; ++s) {
        const float lse_scale = __shfl_sync(0xffffffffu, scale, s % Width);
        const float4* src = reinterpret_cast<const float4*>(o_accum + s * split_stride);
        #pragma unroll
        for (int i = 0; i < kPerLane / 4; ++i) {
            const float4 o = __ldcg(src + lane * (kPerLane / 4) + i);
            acc[i * 4 + 0] += lse_scale * o.x;
            acc[i * 4 + 1] += lse_scale * o.y;
            acc[i * 4 + 2] += lse_scale * o.z;
            acc[i * 4 + 3] += lse_scale * o.w;
        }
    }
    cutlass::bfloat16_t* out = reinterpret_cast<cutlass::bfloat16_t*>(params.o_ptr)
        + bidb * params.o_batch_stride + bidh * params.o_head_stride + row * params.o_row_stride + lane * kPerLane;
    #pragma unroll
    for (int i = 0; i < kPerLane; ++i) { out[i] = cutlass::bfloat16_t(acc[i]); }
}
#endif

// Split-KV decode with the fold of the partials done by the last-arriving split CTA of each
// (batch, head, m_block) instead of a second launch. `counters` holds one int per
// (blockIdx.z, blockIdx.x), zero on entry; the folding CTA rearms it, so one buffer serves every
// stream-ordered launch (including graph replays).
template<typename Traits, bool Is_causal>
__global__ void flash_decode_fused_combine_kernel(COMFY_FLASH_PARAM const Flash_fwd_params params, int* __restrict__ counters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    compute_attn_splitkv<Traits, Is_causal, false, false, false, true, false, true, false>(params);

    __shared__ int s_last;
    __threadfence();
    __syncthreads();
    const int slot = blockIdx.z * gridDim.x + blockIdx.x;
    if (threadIdx.x == 0) {
        const int arrived = atomicAdd(counters + slot, 1);
        s_last = arrived == static_cast<int>(gridDim.y) - 1;
        if (s_last) { counters[slot] = 0; }
    }
    __syncthreads();
    if (!s_last) { return; }
    __threadfence();

    const int bidb = blockIdx.z / params.h;
    const int bidh = blockIdx.z - bidb * params.h;
    const int lane = threadIdx.x & 31;
    const int row_end = min(params.seqlen_q, static_cast<int>((blockIdx.x + 1) * Traits::kBlockM));
    for (int row = blockIdx.x * Traits::kBlockM + (threadIdx.x >> 5); row < row_end; row += Traits::kNThreads / 32) {
        if (params.num_splits <= 2) {
            combine_row_warp<Traits, 2>(params, bidb, bidh, row, lane);
        } else if (params.num_splits <= 4) {
            combine_row_warp<Traits, 4>(params, bidb, bidh, row, lane);
        } else if (params.num_splits <= 8) {
            combine_row_warp<Traits, 8>(params, bidb, bidh, row, lane);
        } else if (params.num_splits <= 16) {
            combine_row_warp<Traits, 16>(params, bidb, bidh, row, lane);
        } else {
            combine_row_warp<Traits, 32>(params, bidb, bidh, row, lane);
        }
    }
#endif
}

template<typename Traits, bool Is_causal>
void launch_flash_decode_typed(Flash_fwd_params& params, int* combine_counters, cudaStream_t stream) {
    constexpr size_t smem_size = Traits::kSmemSize;
    const int m_blocks = (params.seqlen_q + Traits::kBlockM - 1) / Traits::kBlockM;
    dim3 grid(m_blocks, params.num_splits > 1 ? params.num_splits : params.b, params.num_splits > 1 ? params.b * params.h : params.h);

    if (params.num_splits > 1 && combine_counters != nullptr) {
        auto kernel = &flash_decode_fused_combine_kernel<Traits, Is_causal>;
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        kernel<<<grid, Traits::kNThreads, smem_size, stream>>>(params, combine_counters);
    } else if (params.num_splits > 1) {
        auto kernel = &flash_decode_kernel<Traits, Is_causal, true>;
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        kernel<<<grid, Traits::kNThreads, smem_size, stream>>>(params);

        const dim3 combine_grid((params.b * params.h * params.seqlen_q + 3) / 4);
        if (params.num_splits <= 2) {
            flash_decode_combine_kernel<Traits, 1><<<combine_grid, Traits::kNThreads, 0, stream>>>(params);
        } else if (params.num_splits <= 4) {
            flash_decode_combine_kernel<Traits, 2><<<combine_grid, Traits::kNThreads, 0, stream>>>(params);
        } else if (params.num_splits <= 8) {
            flash_decode_combine_kernel<Traits, 3><<<combine_grid, Traits::kNThreads, 0, stream>>>(params);
        } else if (params.num_splits <= 16) {
            flash_decode_combine_kernel<Traits, 4><<<combine_grid, Traits::kNThreads, 0, stream>>>(params);
        } else {
            flash_decode_combine_kernel<Traits, 5><<<combine_grid, Traits::kNThreads, 0, stream>>>(params);
        }
    } else {
        auto kernel = &flash_decode_kernel<Traits, Is_causal, false>;
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        kernel<<<grid, Traits::kNThreads, smem_size, stream>>>(params);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace flash

static flash::Flash_fwd_params decode_params(
    const void* q, const void* k, const void* v, const int* kv_lengths,
    void* output, float* softmax_lse, float* softmax_lse_accum, float* output_accum,
    int batch, int query_length, int heads, int kv_heads, int kv_capacity, int head_dim, int num_splits,
    int64_t q_batch_stride, int64_t q_row_stride, int64_t q_head_stride,
    int64_t k_batch_stride, int64_t k_row_stride, int64_t k_head_stride,
    int64_t o_batch_stride, int64_t o_row_stride, int64_t o_head_stride) {
    flash::Flash_fwd_params params{};
    params.q_ptr = const_cast<void*>(q);
    params.k_ptr = const_cast<void*>(k);
    params.v_ptr = const_cast<void*>(v);
    params.o_ptr = output;
    params.softmax_lse_ptr = softmax_lse;
    params.softmax_lseaccum_ptr = softmax_lse_accum;
    params.oaccum_ptr = output_accum;
    params.q_batch_stride = q_batch_stride;
    params.q_row_stride = q_row_stride;
    params.q_head_stride = q_head_stride;
    params.o_batch_stride = o_batch_stride;
    params.o_row_stride = o_row_stride;
    params.o_head_stride = o_head_stride;
    params.k_batch_stride = params.v_batch_stride = k_batch_stride;
    params.k_row_stride = params.v_row_stride = k_row_stride;
    params.k_head_stride = params.v_head_stride = k_head_stride;
    params.seqused_k = const_cast<int*>(kv_lengths);
    params.b = batch;
    params.h = heads;
    params.h_k = kv_heads;
    params.h_h_k_ratio = heads / kv_heads;
    params.seqlen_q = query_length;
    params.seqlen_k = kv_capacity;
    params.d = params.d_rounded = head_dim;
    params.seqlen_q_rounded = ((query_length + 127) / 128) * 128;
    params.total_q = batch * query_length;
    params.scale_softmax = head_dim == 128 ? 0.08838834764831845f : 0.0625f;
    params.scale_softmax_log2 = head_dim == 128 ? 0.12751743082871335f : 0.09016844005556021f;
    params.window_size_left = params.window_size_right = -1;
    params.num_splits = num_splits;
    params.is_bf16 = true;
    return params;
}

extern "C" void launch_flash_decode(
    const void* q, const void* k, const void* v, const int* kv_lengths,
    void* output, float* softmax_lse, float* softmax_lse_accum, float* output_accum,
    int batch, int query_length, int heads, int head_dim, int kv_capacity, int num_splits,
    int64_t q_batch_stride, int64_t q_row_stride, int64_t q_head_stride,
    int64_t k_batch_stride, int64_t k_row_stride, int64_t k_head_stride,
    int* combine_counters, cudaStream_t stream) {
    auto params = decode_params(q, k, v, kv_lengths, output, softmax_lse, softmax_lse_accum, output_accum,
                                batch, query_length, heads, heads, kv_capacity, head_dim, num_splits,
                                q_batch_stride, q_row_stride, q_head_stride,
                                k_batch_stride, k_row_stride, k_head_stride,
                                q_batch_stride, q_row_stride, q_head_stride);
    if (head_dim == 128) {
        flash::launch_flash_decode_typed<flash::Traits128, false>(params, combine_counters, stream);
    } else {
        flash::launch_flash_decode_typed<flash::Traits256, false>(params, combine_counters, stream);
    }
}

// GQA decode over a [batch, kv_heads, capacity, 256] cache. causal: query row j of each head
// attends slots < kv_lengths[b] - query_length + j + 1 (the verify staircase); otherwise
// every row attends slots < kv_lengths[b].
extern "C" void launch_flash_decode_gqa(
    const void* q, const void* k, const void* v, const int* kv_lengths,
    void* output, float* softmax_lse, float* softmax_lse_accum, float* output_accum,
    int batch, int query_length, int heads, int kv_heads, int kv_capacity, int num_splits, bool causal,
    int64_t q_batch_stride, int64_t q_row_stride, int64_t q_head_stride,
    int64_t k_batch_stride, int64_t k_row_stride, int64_t k_head_stride,
    int64_t o_batch_stride, int64_t o_row_stride, int64_t o_head_stride,
    int* combine_counters, cudaStream_t stream) {
    auto params = decode_params(q, k, v, kv_lengths, output, softmax_lse, softmax_lse_accum, output_accum,
                                batch, query_length, heads, kv_heads, kv_capacity, 256, num_splits,
                                q_batch_stride, q_row_stride, q_head_stride,
                                k_batch_stride, k_row_stride, k_head_stride,
                                o_batch_stride, o_row_stride, o_head_stride);
    if (causal) {
        params.window_size_right = 0;  // row j sees cols < j + 1 + seqlen_k - seqlen_q
        flash::launch_flash_decode_typed<flash::Traits256, true>(params, combine_counters, stream);
    } else {
        flash::launch_flash_decode_typed<flash::Traits256, false>(params, combine_counters, stream);
    }
}

// Current-step rows: the prefix decode pass returned its output and log-sum-exp; row j's
// own token and the earlier rows of this step are the slots that prefix skips. Fold them in
// exactly as further softmax columns:
//   s_t = scale * q . k_t;  m = max(lse, max_t s_t)
//   o = (out * e^(lse - m) + sum_t v_t * e^(s_t - m)) / (e^(lse - m) + sum_t e^(s_t - m))
// over the step rows t <= j.
// One warp per (batch, row, head), eight dims per lane, fp32 math, bf16 out.
namespace step_merge {

constexpr int kMergeHeadDim = 256;
constexpr int kMergeWarps = 8;
constexpr int kMaxRows = 8;

__global__ void __launch_bounds__(kMergeWarps * 32) flash_decode_step_merge_kernel(
    const __nv_bfloat16* __restrict__ out, const float* __restrict__ lse,
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ merged,
    int total, int rows, int heads, int groups, float scale,
    int64_t q_bs, int64_t q_hs, int64_t q_rs,
    int64_t k_bs, int64_t k_hs, int64_t k_rs,
    int64_t v_bs, int64_t v_hs, int64_t v_rs,
    int64_t o_bs, int64_t o_rs,
    int64_t m_bs, int64_t m_rs) {
    const int lane = threadIdx.x & 31;
    const int idx = blockIdx.x * kMergeWarps + (threadIdx.x >> 5);
    if (idx >= total) return;
    const int h = idx % heads;
    const int j = (idx / heads) % rows;
    const int b = idx / (heads * rows);
    const int kh = h / groups;
    const int d0 = lane * 8;

    const uint4 qv = *reinterpret_cast<const uint4*>(q + b * q_bs + h * q_hs + j * q_rs + d0);
    const auto* q2 = reinterpret_cast<const __nv_bfloat162*>(&qv);

    float s[kMaxRows];
    const float l = lse[(b * heads + h) * rows + j];
    float m = l;
#pragma unroll
    for (int t = 0; t < kMaxRows; ++t) {
        s[t] = -INFINITY;
        if (t <= j) {
            const uint4 kv4 = *reinterpret_cast<const uint4*>(k + b * k_bs + kh * k_hs + t * k_rs + d0);
            const auto* k2 = reinterpret_cast<const __nv_bfloat162*>(&kv4);
            float dot = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 qf = __bfloat1622float2(q2[i]), kf = __bfloat1622float2(k2[i]);
                dot += qf.x * kf.x + qf.y * kf.y;
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
            s[t] = dot * scale;
            m = fmaxf(m, s[t]);
        }
    }

    const float w_flash = expf(l - m);
    float denom = w_flash;
    float acc[8];
    const uint4 ov = *reinterpret_cast<const uint4*>(out + b * o_bs + j * o_rs + h * kMergeHeadDim + d0);
    const auto* o2 = reinterpret_cast<const __nv_bfloat162*>(&ov);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 of = __bfloat1622float2(o2[i]);
        acc[2 * i] = of.x * w_flash;
        acc[2 * i + 1] = of.y * w_flash;
    }
#pragma unroll
    for (int t = 0; t < kMaxRows; ++t) {
        if (t <= j) {
            const float w = expf(s[t] - m);
            denom += w;
            const uint4 vv = *reinterpret_cast<const uint4*>(v + b * v_bs + kh * v_hs + t * v_rs + d0);
            const auto* v2 = reinterpret_cast<const __nv_bfloat162*>(&vv);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 vf = __bfloat1622float2(v2[i]);
                acc[2 * i] = fmaf(vf.x, w, acc[2 * i]);
                acc[2 * i + 1] = fmaf(vf.y, w, acc[2 * i + 1]);
            }
        }
    }
    const float inv = 1.f / denom;
    uint4 res;
    auto* r2 = reinterpret_cast<__nv_bfloat162*>(&res);
#pragma unroll
    for (int i = 0; i < 4; ++i)
        r2[i] = __floats2bfloat162_rn(acc[2 * i] * inv, acc[2 * i + 1] * inv);
    *reinterpret_cast<uint4*>(merged + b * m_bs + j * m_rs + h * kMergeHeadDim + d0) = res;
}

} // namespace step_merge

extern "C" void launch_flash_decode_step_merge(
    const void* out, const float* lse, const void* q, const void* k, const void* v, void* merged,
    int batch, int rows, int heads, int kv_heads,
    int64_t q_batch_stride, int64_t q_head_stride, int64_t q_row_stride,
    int64_t k_batch_stride, int64_t k_head_stride, int64_t k_row_stride,
    int64_t v_batch_stride, int64_t v_head_stride, int64_t v_row_stride,
    int64_t o_batch_stride, int64_t o_row_stride,
    int64_t m_batch_stride, int64_t m_row_stride,
    cudaStream_t stream) {
    using namespace step_merge;
    const int warps = batch * rows * heads;
    flash_decode_step_merge_kernel<<<(warps + kMergeWarps - 1) / kMergeWarps, kMergeWarps * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(out), lse,
        static_cast<const __nv_bfloat16*>(q), static_cast<const __nv_bfloat16*>(k), static_cast<const __nv_bfloat16*>(v),
        static_cast<__nv_bfloat16*>(merged),
        warps, rows, heads, heads / kv_heads, 1.0f / sqrtf(float(kMergeHeadDim)),
        q_batch_stride, q_head_stride, q_row_stride,
        k_batch_stride, k_head_stride, k_row_stride,
        v_batch_stride, v_head_stride, v_row_stride,
        o_batch_stride, o_row_stride, m_batch_stride, m_row_stride);
}
