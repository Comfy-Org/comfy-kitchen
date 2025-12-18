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
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <cassert>
#include <stdexcept>
#include <string>

#include "utils.cuh"
#include "../dtype_dispatch.cuh"

// Helper macro for cuBLAS error checking (replaces TORCH_CUDABLAS_CHECK)
#define CUBLAS_CHECK(call) \
  do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
      throw std::runtime_error(std::string("cuBLAS error: ") + std::to_string(status)); \
    } \
  } while (0)

namespace comfy {

namespace {

// Thread-local handle cache to avoid creating/destroying handles repeatedly
// This eliminates implicit device synchronization from cublasLtDestroy
thread_local cublasLtHandle_t cached_handle = nullptr;

cublasLtHandle_t get_cublas_lt_handle() {
  if (cached_handle == nullptr) {
    cublasStatus_t status = cublasLtCreate(&cached_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error(std::string("cuBLAS handle creation error: ") + std::to_string(status));
    }
  }
  return cached_handle;
}

// Optional: cleanup function to be called at module unload
// void cleanup_cublas_lt_handle() {
//   if (cached_handle != nullptr) {
//     cublasLtDestroy(cached_handle);  // Synchronization is OK during cleanup
//     cached_handle = nullptr;
//   }
// }

void cublas_gemm_blockwise_fp4_impl(
    const void* A_ptr,
    const void* A_decode_scale_ptr,
    const void* B_ptr,
    const void* B_decode_scale_ptr,
    void* D_ptr,
    const void* bias_ptr,
    int64_t A_rows,
    int64_t A_cols,
    int64_t B_rows,
    int64_t B_cols,
    int64_t D_rows,
    int64_t D_cols,
    int64_t bias_size,
    int D_dtype_code,
    bool transa,
    bool transb,
    bool grad,
    void* workspace_ptr,
    int64_t workspace_size,
    bool accumulate,
    int math_sm_count,
    const float* alpha_ptr,  // Host or device pointer depending on pointer mode
    cudaStream_t stream) {
  
  // Sanity checks
  // only TN layout is supported
  if (!(transa == true && transb == false)) {
    throw std::runtime_error("Only transa == true, transb == false is supported");
  }
  
  if (A_rows == 0 || A_cols == 0 || B_rows == 0 || B_cols == 0) {
    throw std::runtime_error("Tensor dimensions must be non-zero");
  }

  // m, k, n here are for cuBLAS column major layout
  // this is different with the M N K notation in torch, which is row major layout
  const int m = transa ? A_rows : A_cols;
  // Two fp4 values are packed into one fp8 value, so k is doubled
  const int k = (transa ? A_cols : A_rows) * 2;
  const int n = transb ? B_cols : B_rows;

  // Handle case where inputs are empty.
  if (m == 0 || n == 0 || k == 0) {
    // For wgrad [n, m] @ [m, k] = [n, k] with m = 0, we need to set D to 0.
    if (D_rows * D_cols != 0 && !accumulate) {
      cudaMemsetAsync(D_ptr, 0, D_rows * D_cols * sizeof(float), stream);
    }
    return;
  }
  
  // Verify dimensions match
  if (D_rows != B_rows || D_cols != A_rows) {
    throw std::runtime_error("D shape mismatch");
  }

  int lda = k, ldb = k, ldc = m, ldd = m;

  float* beta_ptr = accumulate ? GetScalarOne() : GetScalarZero();

  // Get cached handle - avoids creation/destruction overhead and synchronization
  cublasLtHandle_t ltHandle = get_cublas_lt_handle();

  // variable to store heuristic result
  int returnedResults = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};

  // Create operation descriptor
  cublasLtMatmulDesc_t operationDesc = nullptr;
  CUBLAS_CHECK(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

#if CUDA_VERSION >= 12090
  // Setup scaling for A and B
  cublasLtMatmulMatrixScale_t A_scale_mode, B_scale_mode;
  // Note: in cuBLAS term, tensor name A and B are swapped.
  A_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  B_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &A_scale_mode, sizeof(A_scale_mode)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &B_scale_mode, sizeof(B_scale_mode)));
#else
  throw std::runtime_error("NVFP4 cuBLAS GEMM requires CUDA 12.9 or later.");
#endif

  // setup transa and transb (for TN only, transa is true, transb is false)
  // Suppose in fwd pass, A is weight, B is input, D is output
  // transa true: A as weight tensor, with torch shape N x K is a transposed tensor
  // transb false: B as input tensor, with torch shape M x K is a non-transposed tensor
  const cublasOperation_t transa_type = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t transb_type = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa_type, sizeof(transa_type)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb_type, sizeof(transb_type)));

  // E2M1 FP4 format
  const cudaDataType_t Atype = CUDA_R_4F_E2M1;
  const cudaDataType_t Btype = CUDA_R_4F_E2M1;
  const cudaDataType_t Dtype = comfy::dtype_code_to_cuda_type(D_dtype_code);

  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc,
      CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
      &A_decode_scale_ptr,
      sizeof(A_decode_scale_ptr)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc,
      CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
      &B_decode_scale_ptr,
      sizeof(B_decode_scale_ptr)));

  // make sure alpha beta computation dtype remains fp32 by CUBLASLT_MATMUL_DESC_SCALE_TYPE
  cublasDataType_t scale_type = CUDA_R_32F;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));

  // Set pointer mode: alpha and beta must both be device pointers
  // beta comes from GetScalarOne/Zero which returns device pointers
  // alpha_ptr must also be a device pointer
  cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode, sizeof(pointer_mode)));

  // Setup mat layout descriptors
  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
      &Adesc, Atype, transa_type == CUBLAS_OP_N ? m : k, transa_type == CUBLAS_OP_N ? k : m, lda));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
      &Bdesc, Btype, transb_type == CUBLAS_OP_N ? k : n, transb_type == CUBLAS_OP_N ? n : k, ldb));

  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, Dtype, m, n, ldc));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Ddesc, Dtype, m, n, ldd));

  // setup epilogue attributes
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
  // If bias is provided, add it via cuBLASLt epilogue. Bias is expected to be length m (rows of
  // column-major D, which corresponds to output feature dimension N in row-major view).
  if (bias_ptr != nullptr && bias_size != 0) {
    if (!(bias_size == m)) {
      throw std::runtime_error("bias must have size matching m dimension");
    }
    epilogue = CUBLASLT_EPILOGUE_BIAS;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc,
        CUBLASLT_MATMUL_DESC_BIAS_POINTER,
        &bias_ptr,
        sizeof(bias_ptr)));
  }
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

  // setup preference attributes
  cublasLtMatmulPreference_t preference = nullptr;
  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));

  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      preference,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspace_size,
      sizeof(workspace_size)));

  // get heuristic result
  const auto status = cublasLtMatmulAlgoGetHeuristic(
      ltHandle,
      operationDesc,
      Adesc,
      Bdesc,
      Cdesc,
      Ddesc,
      preference,
      1,
      &heuristicResult,
      &returnedResults);

  if (status == CUBLAS_STATUS_NOT_SUPPORTED) {
    throw std::runtime_error("Unable to find suitable cuBLAS GEMM algorithm");
  }
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("cuBLAS error: ") + std::to_string(status));
  }
  
  if (returnedResults == 0) {
    throw std::runtime_error("Unable to find any suitable algorithms");
  }

  CUBLAS_CHECK(cublasLtMatmul(
      ltHandle,
      operationDesc,
      alpha_ptr,
      A_ptr,
      Adesc,
      B_ptr,
      Bdesc,
      beta_ptr,
      D_ptr,
      Cdesc,
      D_ptr,
      Ddesc,
      &heuristicResult.algo,
      workspace_ptr,
      workspace_size,
      stream));
  if (preference)
    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
  if (Ddesc)
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Ddesc));
  if (Cdesc)
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
  if (Bdesc)
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
  if (Adesc)
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
  if (operationDesc)
    CUBLAS_CHECK(cublasLtMatmulDescDestroy(operationDesc));

  // ltHandle is cached and reused - no destruction needed here
  // This avoids implicit device synchronization from cublasLtDestroy
}

} // anonymous namespace

} // namespace comfy

// C interface for DLPack bindings
extern "C" {

void launch_cublas_gemm_blockwise_fp4_kernel(
    const void* B_ptr,
    const void* B_decode_scale_ptr,
    const void* A_ptr,
    const void* A_decode_scale_ptr,
    void* D_ptr,
    const void* bias_ptr,
    int64_t M,
    int64_t N,
    int64_t K,
    const float* alpha_device_ptr,
    int out_dtype_code,  // 0=float32, 1=float16, 2=bfloat16
    void* workspace_ptr,
    bool accumulate,
    cudaStream_t stream) {
  
  // Note: cuBLAS uses column-major layout, but PyTorch uses row-major
  // So we swap A and B to match cuBLAS conventions
  // M, N, K are in row-major notation: (M, K) @ (K, N) = (M, N)
  // In cuBLAS column-major: B^T @ A^T = D^T
  
  comfy::cublas_gemm_blockwise_fp4_impl(
      B_ptr,                // weight data (was A_ptr=activation)
      B_decode_scale_ptr,   // weight scale
      A_ptr,                // activation data (was B_ptr=weight)
      A_decode_scale_ptr,   // activation scale
      D_ptr,
      bias_ptr,
      N,  // A_rows (weight rows in column-major)
      K / 2,  // A_cols (K is doubled for FP4 packing)
      M,  // B_rows (input rows in column-major)
      K / 2,  // B_cols
      M,  // D_rows (output rows)
      N,  // D_cols (output cols)
      bias_ptr ? N : 0,  // bias_size
      out_dtype_code,
      true,   // transa (transpose A)
      false,  // transb (don't transpose B)
      false,  // grad
      workspace_ptr,
      workspace_ptr ? 32 * 1024 * 1024 : 0,  // workspace_size (32MB default)
      accumulate,
      0,  // math_sm_count
      alpha_device_ptr,
      stream);
}

} // extern "C"
