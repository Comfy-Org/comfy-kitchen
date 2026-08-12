// SPDX-License-Identifier: BSD-3-Clause

#include <cuda_runtime.h>

#include "utils.cuh"
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

using Traits = Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass::bfloat16_t>;

template<bool Split>
__global__ void flash_decode_kernel(COMFY_FLASH_PARAM const Flash_fwd_params params) {
    COMFY_FLASH_BODY((compute_attn_splitkv<Traits, false, false, false, false, true, false, Split, false>(params));)
}

template<int LogMaxSplits>
__global__ void flash_decode_combine_kernel(COMFY_FLASH_PARAM const Flash_fwd_params params) {
    COMFY_FLASH_BODY((combine_attn_seqk_parallel<Traits, 4, LogMaxSplits, true>(params));)
}

void launch_flash_decode_typed(Flash_fwd_params& params, cudaStream_t stream) {
    constexpr size_t smem_size = Traits::kSmemSize;
    dim3 grid(1, params.num_splits > 1 ? params.num_splits : params.b, params.num_splits > 1 ? params.b * params.h : params.h);

    if (params.num_splits > 1) {
        auto kernel = &flash_decode_kernel<true>;
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        kernel<<<grid, Traits::kNThreads, smem_size, stream>>>(params);

        const dim3 combine_grid((params.b * params.h * params.seqlen_q + 3) / 4);
        if (params.num_splits <= 2) {
            flash_decode_combine_kernel<1><<<combine_grid, Traits::kNThreads, 0, stream>>>(params);
        } else if (params.num_splits <= 4) {
            flash_decode_combine_kernel<2><<<combine_grid, Traits::kNThreads, 0, stream>>>(params);
        } else if (params.num_splits <= 8) {
            flash_decode_combine_kernel<3><<<combine_grid, Traits::kNThreads, 0, stream>>>(params);
        } else if (params.num_splits <= 16) {
            flash_decode_combine_kernel<4><<<combine_grid, Traits::kNThreads, 0, stream>>>(params);
        } else {
            flash_decode_combine_kernel<5><<<combine_grid, Traits::kNThreads, 0, stream>>>(params);
        }
    } else {
        auto kernel = &flash_decode_kernel<false>;
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        kernel<<<grid, Traits::kNThreads, smem_size, stream>>>(params);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace flash

extern "C" void launch_flash_decode(
    const void* q, const void* k, const void* v, const int* kv_lengths,
    void* output, float* softmax_lse, float* softmax_lse_accum, float* output_accum,
    int batch, int query_length, int heads, int kv_capacity, int num_splits,
    int64_t q_batch_stride, int64_t q_row_stride, int64_t q_head_stride,
    int64_t k_batch_stride, int64_t k_row_stride, int64_t k_head_stride,
    cudaStream_t stream) {
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
    params.o_batch_stride = q_batch_stride;
    params.o_row_stride = q_row_stride;
    params.o_head_stride = q_head_stride;
    params.k_batch_stride = params.v_batch_stride = k_batch_stride;
    params.k_row_stride = params.v_row_stride = k_row_stride;
    params.k_head_stride = params.v_head_stride = k_head_stride;
    params.seqused_k = const_cast<int*>(kv_lengths);
    params.b = batch;
    params.h = params.h_k = heads;
    params.h_h_k_ratio = 1;
    params.seqlen_q = query_length;
    params.seqlen_k = kv_capacity;
    params.d = params.d_rounded = 128;
    params.seqlen_q_rounded = ((query_length + 127) / 128) * 128;
    params.total_q = batch * query_length;
    params.scale_softmax = 0.08838834764831845f;
    params.scale_softmax_log2 = 0.12751743082871335f;
    params.window_size_left = params.window_size_right = -1;
    params.num_splits = num_splits;
    params.is_bf16 = true;

    flash::launch_flash_decode_typed(params, stream);
}
