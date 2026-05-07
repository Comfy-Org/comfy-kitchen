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
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <cassert>
#include <stdexcept>
#include <string>

#include "utils.cuh"
#include "../dtype_dispatch.cuh"
#include "../cublaslt_runtime.h"

// Helper macro for cuBLAS error checking
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
thread_local cublasLtHandle_t cached_int8_handle = nullptr;

cublasLtHandle_t get_cublas_lt_handle_int8() {
  auto& runtime = CublasLtRuntime::instance();
  if (!runtime.is_available()) {
    throw std::runtime_error("cuBLASLt not available: " + runtime.error_message());
  }
  
  if (cached_int8_handle == nullptr) {
    cublasStatus_t status = runtime.cublasLtCreate(&cached_int8_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error(std::string("cuBLAS handle creation error: ") + std::to_string(status));
    }
  }
  return cached_int8_handle;
}

void cublas_gemm_int8_impl(
    const int8_t* A_ptr,    // [M, K] row-major
    const int8_t* B_ptr,    // [K, N] row-major
    int32_t* C_ptr,         // [M, N] row-major output
    int64_t M,
    int64_t N,
    int64_t K,
    void* workspace_ptr,
    int64_t workspace_size,
    cudaStream_t stream) {
  
  auto& runtime = CublasLtRuntime::instance();
  if (!runtime.is_available()) {
    throw std::runtime_error("cuBLASLt not available: " + runtime.error_message());
  }

  if (M == 0 || N == 0 || K == 0) {
    return;
  }

  // cuBLAS uses column-major, so we compute C^T = B^T @ A^T
  // With row-major inputs: C = A @ B becomes C^T = B^T @ A^T in column-major
  // We set up the matrices such that cuBLAS computes what we want
  
  // For row-major A[M,K] @ B[K,N] = C[M,N]:
  // In cuBLAS column-major terms:
  // - A is viewed as [K,M] col-major (transposed)
  // - B is viewed as [N,K] col-major (transposed)
  // We compute C = B @ A in column-major, giving C[N,M] col-major = C[M,N] row-major
  
  int lda = K;  // Leading dimension of A in row-major = K
  int ldb = N;  // Leading dimension of B in row-major = N  
  int ldc = N;  // Leading dimension of C in row-major = N

  cublasLtHandle_t ltHandle = get_cublas_lt_handle_int8();

  // Create operation descriptor with INT8 compute type
  cublasLtMatmulDesc_t operationDesc = nullptr;
  CUBLAS_CHECK(runtime.cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I));

  // Set transpose operations for row-major
  // For C = A @ B in row-major:
  // cuBLAS sees C^T = B^T @ A^T, so we mark B as not transposed and A as transposed
  const cublasOperation_t transa = CUBLAS_OP_T;  // Transpose A (because col-major sees it flipped)
  const cublasOperation_t transb = CUBLAS_OP_N;  // Don't transpose B
  CUBLAS_CHECK(runtime.cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  CUBLAS_CHECK(runtime.cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

  // Matrix layouts: INT8 inputs, INT32 output
  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
  
  // A is [M, K] row-major, cuBLAS sees as [K, M] col-major (with transpose)
  CUBLAS_CHECK(runtime.cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, K, M, lda));
  // B is [K, N] row-major, cuBLAS sees as [N, K] col-major (no transpose)
  CUBLAS_CHECK(runtime.cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, K, N, ldb));
  // C is [M, N] row-major, cuBLAS sees as [N, M] col-major
  CUBLAS_CHECK(runtime.cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32I, N, M, ldc));

  // Alpha and beta scalars
  int32_t alpha = 1;
  int32_t beta = 0;

  // Preference and heuristic
  cublasLtMatmulPreference_t preference = nullptr;
  CUBLAS_CHECK(runtime.cublasLtMatmulPreferenceCreate(&preference));
  
  size_t ws_size = workspace_size > 0 ? workspace_size : 0;
  CUBLAS_CHECK(runtime.cublasLtMatmulPreferenceSetAttribute(
      preference,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &ws_size,
      sizeof(ws_size)));

  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  int returnedResults = 0;
  
  // For IMMA kernels, we compute B @ A (swapped order for row-major)
  const auto status = runtime.cublasLtMatmulAlgoGetHeuristic(
      ltHandle,
      operationDesc,
      Bdesc,  // First operand in cuBLAS terms
      Adesc,  // Second operand
      Cdesc,
      Cdesc,
      preference,
      1,
      &heuristicResult,
      &returnedResults);

  if (status == CUBLAS_STATUS_NOT_SUPPORTED || returnedResults == 0) {
    // Clean up and throw
    if (preference) runtime.cublasLtMatmulPreferenceDestroy(preference);
    if (Cdesc) runtime.cublasLtMatrixLayoutDestroy(Cdesc);
    if (Bdesc) runtime.cublasLtMatrixLayoutDestroy(Bdesc);
    if (Adesc) runtime.cublasLtMatrixLayoutDestroy(Adesc);
    if (operationDesc) runtime.cublasLtMatmulDescDestroy(operationDesc);
    throw std::runtime_error("INT8 GEMM not supported on this GPU (requires SM >= 7.5)");
  }
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("cuBLAS heuristic error: ") + std::to_string(status));
  }

  // Execute matmul: C = B @ A (in cuBLAS column-major terms)
  CUBLAS_CHECK(runtime.cublasLtMatmul(
      ltHandle,
      operationDesc,
      &alpha,
      B_ptr,  // First operand
      Bdesc,
      A_ptr,  // Second operand
      Adesc,
      &beta,
      C_ptr,
      Cdesc,
      C_ptr,
      Cdesc,
      &heuristicResult.algo,
      workspace_ptr,
      ws_size,
      stream));

  // Cleanup
  CUBLAS_CHECK(runtime.cublasLtMatmulPreferenceDestroy(preference));
  CUBLAS_CHECK(runtime.cublasLtMatrixLayoutDestroy(Cdesc));
  CUBLAS_CHECK(runtime.cublasLtMatrixLayoutDestroy(Bdesc));
  CUBLAS_CHECK(runtime.cublasLtMatrixLayoutDestroy(Adesc));
  CUBLAS_CHECK(runtime.cublasLtMatmulDescDestroy(operationDesc));
}

} // anonymous namespace

} // namespace comfy

// C interface for DLPack bindings
extern "C" {

void launch_cublas_gemm_int8_kernel(
    const void* A_ptr,
    const void* B_ptr,
    void* C_ptr,
    int64_t M,
    int64_t N,
    int64_t K,
    void* workspace_ptr,
    int64_t workspace_size,
    cudaStream_t stream) {
  
  comfy::cublas_gemm_int8_impl(
      static_cast<const int8_t*>(A_ptr),
      static_cast<const int8_t*>(B_ptr),
      static_cast<int32_t*>(C_ptr),
      M, N, K,
      workspace_ptr,
      workspace_size,
      stream);
}

} // extern "C"
