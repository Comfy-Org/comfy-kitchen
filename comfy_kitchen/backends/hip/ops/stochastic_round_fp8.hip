// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Stochastic-rounding FP8 quantize. The OCP encode lives in fp8_utils.h so this
// kernel and per_tensor_fp8 cannot drift apart.
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "../fp8_utils.h"

namespace comfy::hip_backend {

constexpr int kMaxKernelThreads = 128;

template<typename InputType>
__global__ void stochastic_round_fp8_kernel(
    uint8_t* rng_and_dst,
    const InputType* src,
    int64_t size,
    int output_dtype_code) {

    const bool e4m3 = output_dtype_code == 5;
    const float fp8_max = e4m3 ? 448.0f : 57344.0f;
    const int exponent_bits = e4m3 ? 4 : 5;
    const int mantissa_bits = e4m3 ? 3 : 2;
    const int exponent_bias = e4m3 ? 7 : 15;
    const float mantissa_levels = static_cast<float>(1 << mantissa_bits);
    const float subnormal_mantissa_scale = exp2f(-exponent_bias + 1 - mantissa_bits);
    const float subnormal_value_scale = exp2f(-exponent_bias + 1);

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }

    // Match the CUDA implementation: reduce the source value through FP16 before
    // applying stochastic mantissa rounding.
    const float value = __half2float(__float2half(to_float(src[idx])));
    const float abs_value = fabsf(value);
    const float sign = value < 0.0f ? -1.0f : (value > 0.0f ? 1.0f : 0.0f);

    // log2f(0) is -inf, which -ffast-math is free to assume never happens. A NaN
    // has to leave here too: the fp8_clamp below would drop it for a finite bound
    // before pack_fp8 could encode it.
    if (fp8_is_nan(value)) {
        rng_and_dst[idx] = pack_fp8(value, output_dtype_code);
        return;
    }
    if (abs_value == 0.0f) {
        rng_and_dst[idx] = pack_fp8(0.0f, output_dtype_code);
        return;
    }

    float exponent = floorf(log2f(abs_value)) + exponent_bias;
    exponent = fp8_clamp(exponent, 0.0f, static_cast<float>((1 << exponent_bits) - 1));
    const bool normal = exponent != 0.0f;
    const float exponent_scale = exp2f(exponent - exponent_bias);

    const float normal_mantissa = (abs_value / exponent_scale - 1.0f) * mantissa_levels;
    const float subnormal_mantissa = abs_value / subnormal_mantissa_scale;
    const float random = static_cast<float>(rng_and_dst[idx]) * (1.0f / 256.0f);
    const float mantissa = floorf((normal ? normal_mantissa : subnormal_mantissa) + random) / mantissa_levels;
    const float rounded = sign * (normal ? exponent_scale * (1.0f + mantissa) : subnormal_value_scale * mantissa);

    rng_and_dst[idx] = pack_fp8(fp8_clamp(rounded, -fp8_max, fp8_max), output_dtype_code);
}

} // namespace comfy::hip_backend

extern "C" {

void launch_stochastic_round_fp8_kernel(
    void* rng_and_output,
    const void* input,
    int64_t numel,
    int rng_dtype_code,
    int input_dtype_code,
    int output_dtype_code,
    hipStream_t stream) {

    if (numel == 0) {
        return;
    }

    if (rng_dtype_code != 3) {
        throw std::runtime_error("stochastic_round_fp8 requires uint8 RNG storage");
    }

    const int blocks = static_cast<int>((numel + comfy::hip_backend::kMaxKernelThreads - 1) /
                                        comfy::hip_backend::kMaxKernelThreads);

    if (input_dtype_code == 0) {
        comfy::hip_backend::stochastic_round_fp8_kernel<float>
            <<<blocks, comfy::hip_backend::kMaxKernelThreads, 0, stream>>>(
                static_cast<uint8_t*>(rng_and_output),
                static_cast<const float*>(input),
                numel,
                output_dtype_code);
    } else if (input_dtype_code == 1) {
        comfy::hip_backend::stochastic_round_fp8_kernel<__half>
            <<<blocks, comfy::hip_backend::kMaxKernelThreads, 0, stream>>>(
                static_cast<uint8_t*>(rng_and_output),
                static_cast<const __half*>(input),
                numel,
                output_dtype_code);
    } else if (input_dtype_code == 2) {
        comfy::hip_backend::stochastic_round_fp8_kernel<hip_bfloat16>
            <<<blocks, comfy::hip_backend::kMaxKernelThreads, 0, stream>>>(
                static_cast<uint8_t*>(rng_and_output),
                static_cast<const hip_bfloat16*>(input),
                numel,
                output_dtype_code);
    } else {
        throw std::runtime_error("Unsupported input dtype for stochastic_round_fp8");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP kernel launch failed: ") + hipGetErrorString(err));
    }
}

} // extern "C"
