/*
* SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "dtype_dispatch.cuh"
#include "float_utils.cuh"
#include "utils.cuh"
#include <stdexcept>

namespace comfy {
namespace {
template <typename Input, int V, bool Masked>
__global__ void
quantize_mxfp8_kernel(const Input *input, uint8_t *output, uint8_t *scales,
                      uint64_t groups, uint64_t runtime_columns,
                      uint64_t original_rows, uint64_t original_columns) {
  constexpr int threads_per_group = 32 / V;
  const uint64_t columns = runtime_columns;
  const uint64_t chunk = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  {
    const uint64_t group = chunk / threads_per_group;
    const auto values = quantization::load_group<Input, V, Masked>(
        input, chunk, columns, original_rows, original_columns);
    float maximum = quantization::group_absmax(values);
    if constexpr (threads_per_group > 1) {
#pragma unroll
      for (int offset = threads_per_group / 2; offset; offset /= 2)
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset,
                                                  threads_per_group));
      maximum = __shfl_sync(0xffffffff, maximum, 0, threads_per_group);
    }
    const uint8_t scale = __nv_cvt_float_to_e8m0(
        __fdividef(maximum, 448.0f), __NV_SATFINITE, cudaRoundPosInf);
    const uint64_t row = group / (columns / 32), col = group % (columns / 32);
    if (threadIdx.x % threads_per_group == 0)
      scales[scale_factor_swizzled_offset(row, col, columns / 32)] = scale;
    const float encode = __uint_as_float(uint32_t(254 - scale) << 23);
    union {
      uint4 vectors[(V + 15) / 16];
      uint64_t word;
      uint16_t pairs[V / 2];
    } packed;
#pragma unroll
    for (int i = 0; i < V / 2; ++i) {
      const float a =
          fminf(fmaxf(float(values.values[2 * i]) * encode, -448.0f), 448.0f);
      const float b = fminf(
          fmaxf(float(values.values[2 * i + 1]) * encode, -448.0f), 448.0f);
      packed.pairs[i] = __nv_cvt_float2_to_fp8x2(make_float2(a, b),
                                                 __NV_SATFINITE, __NV_E4M3);
    }
    if constexpr (V == 8)
      reinterpret_cast<uint64_t *>(output)[chunk] = packed.word;
    else {
#pragma unroll
      for (int i = 0; i < V / 16; ++i)
        reinterpret_cast<uint4 *>(output + uint64_t(chunk) * V)[i] =
            packed.vectors[i];
    }
  }
}

template <typename Input, int V>
void launch_mxfp8_quantization(const void *input, void *output, void *scales,
                               int64_t rows, int64_t columns,
                               int64_t original_rows, int64_t original_columns,
                               int threads, cudaStream_t stream) {
  const int64_t groups = rows * columns / 32;
  const int64_t chunks = groups * (32 / V);
  threads = quantization::block_size(chunks, threads);
  const int blocks = static_cast<int>(chunks / threads);
  const bool masked = rows != original_rows || columns != original_columns ||
                      !optimized::aligned_to(input, 16);
#define LAUNCH(MASKED)                                                         \
  quantize_mxfp8_kernel<Input, V, MASKED><<<blocks, threads, 0, stream>>>(  \
      static_cast<const Input *>(input), static_cast<uint8_t *>(output),       \
      static_cast<uint8_t *>(scales), groups, columns, original_rows,          \
      original_columns)
  if (masked) {
    LAUNCH(true);
  } else {
    LAUNCH(false);
  }
#undef LAUNCH
}
__device__ float e8m0_to_f32(uint8_t e){
    if(e==0)return 0.0f;
    return exp2f(static_cast<float>(e)-127.0f);
}
template <typename OType>
__global__ void dequantize_mxfp8_kernel(
    const __nv_fp8_e4m3* __restrict__ input,
    const uint8_t* __restrict__ block_scales,  // E8M0 stored as uint8
    OType* __restrict__ output,
    const size_t num_cols,
    const size_t num_rows) {

    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t global_elem_idx = static_cast<size_t>(idx) * kMXFP8ValsPerThread;

    // Early exit for threads beyond data (reduces divergence)
    if (global_elem_idx >= num_rows * num_cols) return;

    const size_t group_number = global_elem_idx / kMXFP8BlockSize;
    const size_t groups_per_row = num_cols / kMXFP8BlockSize;
    const size_t group_in_row = group_number % groups_per_row;

    const size_t row_idx = group_number/groups_per_row;
    const size_t scale_idx=scale_factor_swizzled_offset(row_idx,group_in_row,groups_per_row);

    const float scale_value=e8m0_to_f32(block_scales[scale_idx]);

    for(unsigned int i=0;i<kMXFP8ValsPerThread;i++){
        const float input_value = static_cast<float>(input[global_elem_idx + i]);
        output[global_elem_idx+i]=static_cast<OType>(input_value*scale_value);
    }


}

} // namespace
} // namespace comfy

extern "C" void launch_quantize_mxfp8_kernel(const void *input, void *output,
                                             void *scales, int64_t rows,
                                             int64_t columns,
                                             int64_t original_rows,
                                             int64_t original_columns,
                                             int dtype, cudaStream_t stream) {
  if (!rows || !columns)
    return;
  if (rows % 32 || columns % 32)
    throw std::runtime_error("num_rows and num_cols must be divisible by 32 "
                             "for MXFP8 block quantization");
  const int threads = rows * columns <= 67108864 ? 128 : 256;
  DISPATCH_HALF_DTYPE(dtype, Input, [&] {
    comfy::launch_mxfp8_quantization<Input, 16>(
        input, output, scales, rows, columns, original_rows, original_columns,
        threads, stream);
  });
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_dequantize_mxfp8_kernel(
    const void* input,
    void* output,
    const void* block_scales,
    int64_t num_rows,
    int64_t num_cols,
    int output_dtype_code,
    cudaStream_t stream) {

        if (num_rows == 0 || num_cols == 0)
        return;

        // Each row must contain complete 32-value groups.
        if (num_cols % comfy::kMXFP8BlockSize != 0) {
            throw std::runtime_error("num_cols must be divisible by 32 for MXFP8 block dequantization");
    }
    const int64_t numel = num_rows * num_cols;

    // Each thread processes kMXFP8ValsPerThread values
    constexpr int threads_per_block = 128;
    const int64_t total_threads_needed = numel / comfy::kMXFP8ValsPerThread;
    const int blocks = static_cast<int>((total_threads_needed + threads_per_block - 1) / threads_per_block);

    DISPATCH_FP_DTYPE(output_dtype_code, OType, [&] {
        comfy::dequantize_mxfp8_kernel<OType>
        <<<blocks, threads_per_block, 0, stream>>>(
            static_cast<const __nv_fp8_e4m3*>(input),
            static_cast<const uint8_t*>(block_scales),
            static_cast<OType*>(output),
            num_cols,
            num_rows
        );
    });

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }

    }
