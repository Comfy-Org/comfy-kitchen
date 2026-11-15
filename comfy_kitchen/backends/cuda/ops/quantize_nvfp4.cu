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
#include <cfloat>
#include <stdexcept>

namespace comfy {
namespace {
// Identical group ownership and arithmetic on every GPU.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 1000
template <typename Input, int V, bool Masked, bool HiFirst>
__global__ void
quantize_nvfp4_kernel(const Input *input, const float *global_scale,
                      uint8_t *output, __nv_fp8_e4m3 *scales, uint64_t groups,
                      uint64_t runtime_columns, uint64_t original_rows,
                      uint64_t original_columns) {
  constexpr int threads_per_group = 16 / V;
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
    const float tensor_scale = global_scale[0];
    const __nv_fp8_e4m3 scale(
        fminf(__fdividef(maximum * (1.0f / 6.0f), tensor_scale), 448.0f));
    const float encode =
        fminf(__fdividef(1.0f, float(scale) * tensor_scale), FLT_MAX);
    if (threadIdx.x % threads_per_group == 0) {
      const uint64_t row = group / (columns / 16), col = group % (columns / 16);
      scales[scale_factor_swizzled_offset(row, col, columns / 16)] = scale;
    }
    union {
      uint64_t word;
      uint16_t short_word;
      __nv_fp4x2_storage_t pairs[V / 2];
    } packed;
#pragma unroll
    for (int i = 0; i < V / 2; ++i) {
      const float a = float(values.values[2 * i]) * encode;
      const float b = float(values.values[2 * i + 1]) * encode;
      packed.pairs[i] = __nv_cvt_float2_to_fp4x2(HiFirst ? make_float2(b, a)
                                                         : make_float2(a, b),
                                                 __NV_E2M1, cudaRoundNearest);
    }
    if constexpr (V == 16)
      reinterpret_cast<uint64_t *>(output)[chunk] = packed.word;
    else
      reinterpret_cast<uint16_t *>(output)[chunk] = packed.short_word;
  }
}
#endif

// Keep the existing four-value decode kernel on every GPU.
constexpr unsigned int kValsPerThread = 4;
template <
    typename IType=__nv_fp4x2_e2m1,
    typename OType,
    typename ScaleType=__nv_fp8_e4m3,
    bool HiFirst = true,
    bool AlignedInput = true>
__global__ void dequantize_nvfp4_kernel(
    const IType* const input,
    const float* global_scale,
    const ScaleType* const block_scales,
    OType* const output,
    const size_t num_cols,
    const size_t num_rows) {

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate which block this thread group is processing
    const size_t global_elem_idx = static_cast<size_t>(idx) * kValsPerThread;
    const size_t block_linear_idx = global_elem_idx / 16;
    
    // Convert linear block index to 2D coordinates
    const size_t num_blocks_per_row = num_cols / 16;
    const size_t row_idx = block_linear_idx / num_blocks_per_row;
    const size_t col_idx = block_linear_idx % num_blocks_per_row;
    
    // Read block scale from swizzled layout
    size_t scale_offset = scale_factor_swizzled_offset(
        row_idx, col_idx, num_blocks_per_row);
    ScaleType block_scale_fp8 = block_scales[scale_offset];
    
    // Compute decode scale: decode_scale = block_scale_fp8 * global_decode_scale
    const float global_decode_scale = global_scale[0];
    float decode_scale = static_cast<float>(block_scale_fp8) * global_decode_scale;
    
    // Read packed FP4 values
    // With kValsPerThread=4, we read 2 packed values (each IType contains 2 FP4 values)
    const IType* input_ptr = input + idx * (kValsPerThread / 2);
    
    // Unpack and dequantize
    OType vals_output[kValsPerThread];
    
    // __nv_cvt_fp4x2_to_halfraw2 returns: .x = low nibble value, .y = high nibble value.
    // HiFirst=true:  high nibble = even index (val0), low nibble = odd index (val1) → .y=val0, .x=val1
    // HiFirst=false: high nibble = odd index (val1), low nibble = even index (val0) → .x=val0, .y=val1
    if (kValsPerThread == 2) {
        __half2_raw unpacked_raw = __nv_cvt_fp4x2_to_halfraw2(
            *reinterpret_cast<const __nv_fp4x2_storage_t*>(input_ptr),
            __NV_E2M1);
        float2 unpacked_f2 = __half22float2(*reinterpret_cast<__half2*>(&unpacked_raw));
        if constexpr (HiFirst) {
            vals_output[0] = static_cast<OType>(unpacked_f2.y * decode_scale);
            vals_output[1] = static_cast<OType>(unpacked_f2.x * decode_scale);
        } else {
            vals_output[0] = static_cast<OType>(unpacked_f2.x * decode_scale);
            vals_output[1] = static_cast<OType>(unpacked_f2.y * decode_scale);
        }
    }
    else if (kValsPerThread == 4) {
        union {
            uint16_t u16;
            __nv_fp4x2_storage_t fp4x2[2];
        } packed;
        // Public contiguous views may begin at an odd byte offset.
        if constexpr (AlignedInput) {
            packed.u16 = *reinterpret_cast<const uint16_t*>(input_ptr);
        } else {
            packed.fp4x2[0] = reinterpret_cast<const __nv_fp4x2_storage_t*>(input_ptr)[0];
            packed.fp4x2[1] = reinterpret_cast<const __nv_fp4x2_storage_t*>(input_ptr)[1];
        }
        
        __half2_raw unpacked_raw0 = __nv_cvt_fp4x2_to_halfraw2(packed.fp4x2[0], __NV_E2M1);
        __half2_raw unpacked_raw1 = __nv_cvt_fp4x2_to_halfraw2(packed.fp4x2[1], __NV_E2M1);
        float2 unpacked_f2_0 = __half22float2(*reinterpret_cast<__half2*>(&unpacked_raw0));
        float2 unpacked_f2_1 = __half22float2(*reinterpret_cast<__half2*>(&unpacked_raw1));
        
        if constexpr (HiFirst) {
            vals_output[0] = static_cast<OType>(unpacked_f2_0.y * decode_scale);
            vals_output[1] = static_cast<OType>(unpacked_f2_0.x * decode_scale);
            vals_output[2] = static_cast<OType>(unpacked_f2_1.y * decode_scale);
            vals_output[3] = static_cast<OType>(unpacked_f2_1.x * decode_scale);
        } else {
            vals_output[0] = static_cast<OType>(unpacked_f2_0.x * decode_scale);
            vals_output[1] = static_cast<OType>(unpacked_f2_0.y * decode_scale);
            vals_output[2] = static_cast<OType>(unpacked_f2_1.x * decode_scale);
            vals_output[3] = static_cast<OType>(unpacked_f2_1.y * decode_scale);
        }
    }
    
    // Store output using vectorized writes
    // For 16-bit types: kValsPerThread=4 uses float2 (8 bytes), kValsPerThread=2 uses float (4 bytes)
    // For 32-bit types: kValsPerThread=4 uses float4 (16 bytes), kValsPerThread=2 uses float2 (8 bytes)
    OType* output_ptr = output + kValsPerThread * idx;
    if constexpr (sizeof(OType) == 2) {
        // FP16/BF16 output
        if (kValsPerThread == 2) {
            *reinterpret_cast<float*>(output_ptr) = *reinterpret_cast<float*>(vals_output);
        }
        else if (kValsPerThread == 4) {
            *reinterpret_cast<float2*>(output_ptr) = *reinterpret_cast<float2*>(vals_output);
        }
    } else {
        // FP32 output
        if (kValsPerThread == 2) {
            *reinterpret_cast<float2*>(output_ptr) = *reinterpret_cast<float2*>(vals_output);
        }
        else if (kValsPerThread == 4) {
            *reinterpret_cast<float4*>(output_ptr) = *reinterpret_cast<float4*>(vals_output);
        }
    }
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 1000
template <typename Input, int V>
void launch_nvfp4_quantization(const void *input, const void *global_scale,
                               void *output, void *scales, int64_t rows,
                               int64_t columns, int64_t original_rows,
                               int64_t original_columns, bool hi_first,
                               int threads, cudaStream_t stream) {
  const int64_t groups = rows * columns / 16;
  const int64_t chunks = groups * (16 / V);
  threads = quantization::block_size(chunks, threads);
  const int blocks = static_cast<int>(chunks / threads);
  const bool masked = rows != original_rows || columns != original_columns ||
                      !optimized::aligned_to(input, 16);
#define LAUNCH(MASKED, HI_FIRST)                                               \
  quantize_nvfp4_kernel<Input, V, MASKED, HI_FIRST>                         \
      <<<blocks, threads, 0, stream>>>(                                        \
          static_cast<const Input *>(input),                                   \
          static_cast<const float *>(global_scale),                            \
          static_cast<uint8_t *>(output),                                      \
          static_cast<__nv_fp8_e4m3 *>(scales), groups, columns,               \
          original_rows, original_columns)
  if (masked) {
    if (hi_first) {
      LAUNCH(true, true);
    } else {
      LAUNCH(true, false);
    }
  } else {
    if (hi_first) {
      LAUNCH(false, true);
    } else {
      LAUNCH(false, false);
    }
  }
#undef LAUNCH
}

template <typename Input>
void dispatch_nvfp4_quantization(const void *input, const void *global_scale,
                                 void *output, void *scales, int64_t rows,
                                 int64_t columns, int64_t original_rows,
                                 int64_t original_columns, bool hi_first,
                                 int threads, cudaStream_t stream) {
#define RUN(V)                                                                 \
  launch_nvfp4_quantization<Input, V>(                                  \
      input, global_scale, output, scales, rows, columns, original_rows,       \
      original_columns, hi_first, threads, stream)
  if (rows * columns < 4194304) {
    RUN(4);
  } else {
    RUN(16);
  }
#undef RUN
}
#endif

template <typename Output>
void dispatch_nvfp4_dequantization(const void *input, const void *global_scale,
                                   const void *scales, void *output,
                                   int64_t rows, int64_t columns, bool hi_first,
                                   int threads, cudaStream_t stream) {
  const int64_t chunks = rows * columns / kValsPerThread;
  threads = quantization::block_size(chunks, threads);
  const int blocks = static_cast<int>(chunks / threads);
  const bool aligned = optimized::aligned_to(input, alignof(uint16_t));
#define LAUNCH(HI_FIRST, ALIGNED)                                              \
  dequantize_nvfp4_kernel<__nv_fp4x2_e2m1, Output, __nv_fp8_e4m3, HI_FIRST,    \
                          ALIGNED><<<blocks, threads, 0, stream>>>(            \
      static_cast<const __nv_fp4x2_e2m1 *>(input),                             \
      static_cast<const float *>(global_scale),                                \
      static_cast<const __nv_fp8_e4m3 *>(scales),                              \
      static_cast<Output *>(output), columns, rows)
  if (aligned) {
    if (hi_first) {
      LAUNCH(true, true);
    } else {
      LAUNCH(false, true);
    }
  } else {
    if (hi_first) {
      LAUNCH(true, false);
    } else {
      LAUNCH(false, false);
    }
  }
#undef LAUNCH
}
} // namespace
} // namespace comfy

extern "C" {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 1000
void launch_quantize_nvfp4_kernel(const void *input, const void *global_scale,
                                  void *output, void *scales, int64_t rows,
                                  int64_t columns, int64_t original_rows,
                                  int64_t original_columns, float epsilon,
                                  int dtype, bool hi_first,
                                  cudaStream_t stream) {
  if (!rows || !columns)
    return;
  if (rows % 16 || columns % 16)
    throw std::runtime_error("num_rows and num_cols must be divisible by 16 "
                             "for FP4 block quantization");
  // Tensor-size tuning is identical on every GPU.
  const int threads = 128;
  DISPATCH_HALF_DTYPE(dtype, Input, [&] {
    comfy::dispatch_nvfp4_quantization<Input>(
        input, global_scale, output, scales, rows, columns, original_rows,
        original_columns, hi_first, threads, stream);
  });
  CUDA_CHECK(cudaGetLastError());
}
#endif

void launch_dequantize_nvfp4_kernel(const void *input, const void *global_scale,
                                    const void *scales, void *output,
                                    int64_t rows, int64_t columns, int dtype,
                                    bool hi_first, cudaStream_t stream) {
  if (!rows || !columns)
    return;
  if (rows % 16 || columns % 16)
    throw std::runtime_error("num_rows and num_cols must be divisible by 16 "
                             "for FP4 block dequantization");
  DISPATCH_FP_DTYPE(dtype, Output, [&] {
    comfy::dispatch_nvfp4_dequantization<Output>(input, global_scale, scales,
                                                 output, rows, columns,
                                                 hi_first, 128, stream);
  });
  CUDA_CHECK(cudaGetLastError());
}
} // extern "C"
