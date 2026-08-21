// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Stochastic-rounding FP8 quantize. The OCP encode lives in fp8_utils.h so this
// kernel and per_tensor_fp8 cannot drift apart.
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "../fp8_utils.h"

namespace comfy::hip_backend {

constexpr int kMaxKernelThreads = 128;

// Shared by the scalar and vectorized kernels so the two cannot drift onto
// different rounding. output_dtype_code is uniform across a launch, so the
// constants below hoist out of the vectorized body.
__forceinline__ __device__ uint8_t stochastic_round_one(
    float src_value,
    uint8_t rng_byte,
    int output_dtype_code) {

    const bool e4m3 = output_dtype_code == 5;
    const float fp8_max = e4m3 ? 448.0f : 57344.0f;
    const int exponent_bits = e4m3 ? 4 : 5;
    const int mantissa_bits = e4m3 ? 3 : 2;
    const int exponent_bias = e4m3 ? 7 : 15;
    const float mantissa_levels = static_cast<float>(1 << mantissa_bits);
    const float subnormal_mantissa_scale = exp2f(-exponent_bias + 1 - mantissa_bits);
    const float subnormal_value_scale = exp2f(-exponent_bias + 1);

    // Match the CUDA implementation: reduce the source value through FP16 before
    // applying stochastic mantissa rounding.
    const float value = __half2float(__float2half(src_value));
    const float abs_value = fabsf(value);
    const float sign = value < 0.0f ? -1.0f : (value > 0.0f ? 1.0f : 0.0f);

    // log2f(0) is -inf, which -ffast-math is free to assume never happens. A NaN
    // has to leave here too: the fp8_clamp below would drop it for a finite bound
    // before pack_fp8 could encode it.
    if (fp8_is_nan(value)) {
        return pack_fp8(value, output_dtype_code);
    }
    if (abs_value == 0.0f) {
        return pack_fp8(0.0f, output_dtype_code);
    }

    float exponent = floorf(log2f(abs_value)) + exponent_bias;
    exponent = fp8_clamp(exponent, 0.0f, static_cast<float>((1 << exponent_bits) - 1));
    const bool normal = exponent != 0.0f;
    const float exponent_scale = exp2f(exponent - exponent_bias);

    const float normal_mantissa = (abs_value / exponent_scale - 1.0f) * mantissa_levels;
    const float subnormal_mantissa = abs_value / subnormal_mantissa_scale;
    const float random = static_cast<float>(rng_byte) * (1.0f / 256.0f);
    const float mantissa = floorf((normal ? normal_mantissa : subnormal_mantissa) + random) / mantissa_levels;
    const float rounded = sign * (normal ? exponent_scale * (1.0f + mantissa) : subnormal_value_scale * mantissa);

    return pack_fp8(fp8_clamp(rounded, -fp8_max, fp8_max), output_dtype_code);
}

template<typename InputType>
__global__ void stochastic_round_fp8_kernel(
    uint8_t* rng_and_dst,
    const InputType* src,
    int64_t size,
    int output_dtype_code) {

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }
    rng_and_dst[idx] =
        stochastic_round_one(to_float(src[idx]), rng_and_dst[idx], output_dtype_code);
}

// Vectorized counterpart. The RNG bytes and the fp8 result share one buffer, so
// the chunk is read into registers once and written back after the whole run.
template<typename InputType>
__global__ void stochastic_round_fp8_vec_kernel(
    uint8_t* __restrict__ rng_and_dst,
    const InputType* __restrict__ src,
    int64_t size,
    int output_dtype_code) {

    const int64_t i0 = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * kVecElems;
    if (i0 >= size) {
        return;
    }

    if (i0 + kVecElems <= size) {
        const VecChunk<InputType> in = *reinterpret_cast<const VecChunk<InputType>*>(src + i0);
        VecChunk<uint8_t> rd = *reinterpret_cast<const VecChunk<uint8_t>*>(rng_and_dst + i0);
        #pragma unroll
        for (int i = 0; i < kVecElems; ++i) {
            rd.v[i] = stochastic_round_one(to_float(in.v[i]), rd.v[i], output_dtype_code);
        }
        *reinterpret_cast<VecChunk<uint8_t>*>(rng_and_dst + i0) = rd;
    } else {
        for (int64_t i = i0; i < size; ++i) {
            rng_and_dst[i] =
                stochastic_round_one(to_float(src[i]), rng_and_dst[i], output_dtype_code);
        }
    }
}

// A template rather than a macro so the two launches stay type-checked.
template <typename InputType>
static void launch_sr(void* rng_and_output, const void* input, int64_t numel,
                      int output_dtype_code, hipStream_t stream) {
    const bool vec = vec_aligned(rng_and_output) && vec_aligned(input);
    const int64_t per_thread = vec ? kVecElems : 1;
    const int blocks = static_cast<int>((numel + per_thread * kMaxKernelThreads - 1) /
                                        (per_thread * kMaxKernelThreads));

    auto* dst = static_cast<uint8_t*>(rng_and_output);
    const auto* src = static_cast<const InputType*>(input);
    if (vec) {
        stochastic_round_fp8_vec_kernel<InputType>
            <<<blocks, kMaxKernelThreads, 0, stream>>>(dst, src, numel, output_dtype_code);
    } else {
        stochastic_round_fp8_kernel<InputType>
            <<<blocks, kMaxKernelThreads, 0, stream>>>(dst, src, numel, output_dtype_code);
    }
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

    using namespace comfy::hip_backend;

    if (input_dtype_code == 0) {
        launch_sr<float>(rng_and_output, input, numel, output_dtype_code, stream);
    } else if (input_dtype_code == 1) {
        launch_sr<__half>(rng_and_output, input, numel, output_dtype_code, stream);
    } else if (input_dtype_code == 2) {
        launch_sr<__bf16>(rng_and_output, input, numel, output_dtype_code, stream);
    } else {
        throw std::runtime_error("Unsupported input dtype for stochastic_round_fp8");
    }
    // No hipGetLastError() here: every _C entry point calls check_hip_launch()
    // immediately after its launcher returns, and these launchers have no other
    // caller. Checking here too would just be a second, redundant read.
}

} // extern "C"
