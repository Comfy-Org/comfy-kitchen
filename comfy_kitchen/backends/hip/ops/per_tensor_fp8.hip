// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Per-tensor FP8 quantize/dequantize. The OCP encode/decode lives in fp8_utils.h
// so this kernel and stochastic_round_fp8 cannot drift apart.
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "../fp8_utils.h"

namespace comfy::hip_backend {

constexpr int kThreads = 256;

template<typename T>
__forceinline__ __device__ void store_value(void* dst, int64_t idx, float value) {
    reinterpret_cast<T*>(dst)[idx] = static_cast<T>(value);
}

template<>
__forceinline__ __device__ void store_value<__half>(void* dst, int64_t idx, float value) {
    reinterpret_cast<__half*>(dst)[idx] = __float2half(value);
}

template<typename InputType>
__global__ void quantize_per_tensor_fp8_kernel(
    const InputType* input,
    const float* scale,
    uint8_t* output,
    int64_t numel,
    int output_dtype_code) {

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    const float scaled = to_float(input[idx]) / scale[0];
    output[idx] = pack_fp8(scaled, output_dtype_code);
}

template<typename OutputType>
__global__ void dequantize_per_tensor_fp8_kernel(
    const uint8_t* input,
    const float* scale,
    void* output,
    int64_t numel,
    int input_dtype_code) {

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    store_value<OutputType>(output, idx, unpack_fp8(input[idx], input_dtype_code) * scale[0]);
}

} // namespace comfy::hip_backend

extern "C" {

void launch_quantize_per_tensor_fp8_kernel(
    const void* input,
    const void* scale,
    void* output,
    int64_t numel,
    int input_dtype_code,
    int output_dtype_code,
    hipStream_t stream) {

    if (numel == 0) {
        return;  // a zero-block launch is hipErrorInvalidConfiguration
    }

    const int blocks = static_cast<int>((numel + comfy::hip_backend::kThreads - 1) / comfy::hip_backend::kThreads);
    if (input_dtype_code == 0) {
        comfy::hip_backend::quantize_per_tensor_fp8_kernel<float><<<blocks, comfy::hip_backend::kThreads, 0, stream>>>(static_cast<const float*>(input), static_cast<const float*>(scale), static_cast<uint8_t*>(output), numel, output_dtype_code);
    } else if (input_dtype_code == 1) {
        comfy::hip_backend::quantize_per_tensor_fp8_kernel<__half><<<blocks, comfy::hip_backend::kThreads, 0, stream>>>(static_cast<const __half*>(input), static_cast<const float*>(scale), static_cast<uint8_t*>(output), numel, output_dtype_code);
    } else if (input_dtype_code == 2) {
        comfy::hip_backend::quantize_per_tensor_fp8_kernel<__bf16><<<blocks, comfy::hip_backend::kThreads, 0, stream>>>(static_cast<const __bf16*>(input), static_cast<const float*>(scale), static_cast<uint8_t*>(output), numel, output_dtype_code);
    } else {
        throw std::runtime_error("Unsupported input dtype for quantize_per_tensor_fp8");
    }
}

void launch_dequantize_per_tensor_fp8_kernel(
    const void* input,
    const void* scale,
    void* output,
    int64_t numel,
    int input_dtype_code,
    int output_dtype_code,
    hipStream_t stream) {

    if (numel == 0) {
        return;  // a zero-block launch is hipErrorInvalidConfiguration
    }

    const int blocks = static_cast<int>((numel + comfy::hip_backend::kThreads - 1) / comfy::hip_backend::kThreads);
    if (output_dtype_code == 0) {
        comfy::hip_backend::dequantize_per_tensor_fp8_kernel<float><<<blocks, comfy::hip_backend::kThreads, 0, stream>>>(static_cast<const uint8_t*>(input), static_cast<const float*>(scale), output, numel, input_dtype_code);
    } else if (output_dtype_code == 1) {
        comfy::hip_backend::dequantize_per_tensor_fp8_kernel<__half><<<blocks, comfy::hip_backend::kThreads, 0, stream>>>(static_cast<const uint8_t*>(input), static_cast<const float*>(scale), output, numel, input_dtype_code);
    } else if (output_dtype_code == 2) {
        comfy::hip_backend::dequantize_per_tensor_fp8_kernel<__bf16><<<blocks, comfy::hip_backend::kThreads, 0, stream>>>(static_cast<const uint8_t*>(input), static_cast<const float*>(scale), output, numel, input_dtype_code);
    } else {
        throw std::runtime_error("Unsupported output dtype for dequantize_per_tensor_fp8");
    }
}

} // extern "C"
