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
#ifndef COMFY_UTILS_CUH_
#define COMFY_UTILS_CUH_

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#if CUDA_VERSION >= 12080
#include <cuda_fp4.h>
#endif

#include <type_traits>
#include <mutex>
#include <stdexcept>

// MSVC compatibility layer for GCC builtins
#ifdef _MSC_VER
  #ifndef __builtin_assume
    #define __builtin_assume(x) __assume(x)
  #endif
  #ifndef __builtin_clz
    #include <intrin.h>
    inline int __builtin_clz(unsigned int x) {
      unsigned long leading_zero = 0;
      if (_BitScanReverse(&leading_zero, x)) {
        return 31 - leading_zero;
      }
      return 32; // undefined for x == 0
    }
  #endif
#endif

namespace comfy {

////////////////////////////////////////////////////////////////////////////////

constexpr int kThreadsPerWarp = 32;

////////////////////////////////////////////////////////////////////////////////
// NOTE: This file previously contained ATen-dependent type traits and macros.
// Those have been removed to eliminate all PyTorch C++ dependencies.
// The kernels that used these utilities are not compiled in pure DLPack mode.
////////////////////////////////////////////////////////////////////////////////

/* Use CUDA const memory to store scalar 1 and 0 for cublas usage
 */
__device__ __constant__ float one_device;
__device__ __constant__ float zero_device;

// Helper macro for CUDA error checking (replaces C10_CUDA_CHECK)
#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
  } while (0)

inline float* GetScalarOne() {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    float one = 1.0f;
    CUDA_CHECK(cudaMemcpyToSymbol(one_device, &one, sizeof(float)));
  });
  // return address by cudaGetSymbolAddress
  float* dev_ptr;
  CUDA_CHECK(cudaGetSymbolAddress((void**)&dev_ptr, one_device));
  return dev_ptr;
}

inline float* GetScalarZero() {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    float zero = 0.0f;
    CUDA_CHECK(cudaMemcpyToSymbol(zero_device, &zero, sizeof(float)));
  });
  // return address by cudaGetSymbolAddress
  float* dev_ptr;
  CUDA_CHECK(cudaGetSymbolAddress((void**)&dev_ptr, zero_device));
  return dev_ptr;
}

} // namespace comfy

#endif // COMFY_UTILS_CUH_
