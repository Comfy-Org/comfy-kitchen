// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Exact BF16 RMSNorm followed by the separately rounded BF16 gate product and
// residual add. One workgroup owns one row.
#include <cstdint>
#include <stdexcept>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "../rope_math.h"

namespace comfy::hip_backend {

typedef __bf16 RmsResidualBf16x4 __attribute__((ext_vector_type(4)));

extern "C" __device__ float __ocml_rsqrt_f32(float);

__forceinline__ __device__ float rms_residual_ieee_div_f32(
    float numerator, float denominator) {
    bool denominator_scale = false;
    bool scale = false;
    const float scaled_denominator = __builtin_amdgcn_div_scalef(
        numerator, denominator, false, &denominator_scale);
    const float scaled_numerator = __builtin_amdgcn_div_scalef(
        numerator, denominator, true, &scale);
    float reciprocal = __builtin_amdgcn_rcpf(scaled_denominator);
    const float reciprocal_error =
        fmaf(-scaled_denominator, reciprocal, 1.0f);
    reciprocal = fmaf(reciprocal_error, reciprocal, reciprocal);
    float quotient = scaled_numerator * reciprocal;
    float remainder =
        fmaf(-scaled_denominator, quotient, scaled_numerator);
    quotient = fmaf(remainder, reciprocal, quotient);
    remainder = fmaf(-scaled_denominator, quotient, scaled_numerator);
    quotient = __builtin_amdgcn_div_fmasf(
        remainder, reciprocal, quotient, scale);
    return __builtin_amdgcn_div_fixupf(
        quotient, denominator, numerator);
}

__forceinline__ __device__ __bf16 rms_gated_residual_value(
    __bf16 raw, __bf16 norm_weight, float rstd,
    __bf16 residual, __bf16 gate) {
    // PyTorch RMSNorm materializes gamma * (rstd * x) in BF16. Preserve the
    // two FP32 multiplies and the single visible BF16 rounding point.
    float normalized = rstd * static_cast<float>(raw);
    asm volatile("" : "+v"(normalized));
    normalized *= static_cast<float>(norm_weight);
    normalized = round_bf16(normalized);

    // The eager block exit is two more BF16 kernels: gate * normalized, then
    // residual + product. Each therefore has its own BF16 rounding point.
    const float product = round_bf16(
        normalized * static_cast<float>(gate));
    return static_cast<__bf16>(round_bf16(
        static_cast<float>(residual) + product));
}

template <int STATIC_WIDTH = 0>
__global__ __launch_bounds__(256) void rms_gated_residual_bf16_kernel(
    const __bf16* __restrict__ activation,
    const __bf16* __restrict__ norm_weight,
    const __bf16* __restrict__ residual,
    const __bf16* __restrict__ gate,
    __bf16* __restrict__ output, int width,
    float eps) {

    const int kWidth = STATIC_WIDTH ? STATIC_WIDTH : width;
    const int kVectors = kWidth / 4;
    __shared__ float wave_sums[8];
    extern __shared__ __bf16 row[];

    const int thread = threadIdx.x;
    const int lane = thread & 31;
    const int wave = thread >> 5;
    const int row_index = blockIdx.x;
    const int64_t row_offset = static_cast<int64_t>(row_index) * kWidth;

    const auto* input_vectors =
        reinterpret_cast<const RmsResidualBf16x4*>(activation + row_offset);
    auto* row_vectors = reinterpret_cast<RmsResidualBf16x4*>(row);

    float rstd;
    {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(on)
    #pragma clang fp reciprocal(off)
    float sum_sq = 0.0f;
    for (int vector = thread; vector < kVectors; vector += 256) {
        const RmsResidualBf16x4 values = input_vectors[vector];
        row_vectors[vector] = values;
        #pragma unroll
        for (int element = 0; element < 4; ++element) {
            const float value = static_cast<float>(values[element]);
            sum_sq += value * value;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down(sum_sq, offset, 32);
    }

    // Match PyTorch's eight-wave pairwise compute_stats() merge order.
    #pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        if (lane == 0 && wave >= offset && wave < 2 * offset) {
            wave_sums[wave - offset] = sum_sq;
        }
        __syncthreads();
        if (lane == 0 && wave < offset) {
            sum_sq += wave_sums[wave];
        }
        __syncthreads();
    }

    if (thread == 0) {
        const float mean_square = rms_residual_ieee_div_f32(
            sum_sq, static_cast<float>(kWidth));
        wave_sums[0] = __ocml_rsqrt_f32(mean_square + eps);
    }
    __syncthreads();
    rstd = wave_sums[0];
    }

    const auto* weight_vectors =
        reinterpret_cast<const RmsResidualBf16x4*>(norm_weight);
    const auto* residual_vectors =
        reinterpret_cast<const RmsResidualBf16x4*>(residual + row_offset);
    const auto* gate_vectors =
        reinterpret_cast<const RmsResidualBf16x4*>(gate);
    auto* output_vectors =
        reinterpret_cast<RmsResidualBf16x4*>(output + row_offset);

    for (int vector = thread; vector < kVectors; vector += 256) {
        const RmsResidualBf16x4 raw_values = row_vectors[vector];
        const RmsResidualBf16x4 weight_values = weight_vectors[vector];
        const RmsResidualBf16x4 residual_values = residual_vectors[vector];
        const RmsResidualBf16x4 gate_values = gate_vectors[vector];
        RmsResidualBf16x4 result;
        #pragma unroll
        for (int element = 0; element < 4; ++element) {
            result[element] = rms_gated_residual_value(
                raw_values[element], weight_values[element], rstd,
                residual_values[element], gate_values[element]);
        }
        output_vectors[vector] = result;
    }
}

}  // namespace comfy::hip_backend

extern "C" void launch_rms_gated_residual_bf16_kernel(
    const void* activation, const void* norm_weight, const void* residual,
    const void* gate, void* output, int rows, int width, float eps,
    hipStream_t stream) {

    using namespace comfy::hip_backend;
    if (rows < 0 || width <= 0 || width % 4 != 0) {
        throw std::runtime_error(
            "rms_gated_residual_bf16: requires non-negative rows and positive "
            "width divisible by 4");
    }
    if (rows == 0) return;
    const size_t shared_bytes = static_cast<size_t>(width) * sizeof(__bf16);
#define CK_LAUNCH_RMS_GATED(STATIC_WIDTH)                                      \
    rms_gated_residual_bf16_kernel<STATIC_WIDTH>                               \
        <<<rows, 256, shared_bytes, stream>>>(                                  \
            static_cast<const __bf16*>(activation),                            \
            static_cast<const __bf16*>(norm_weight),                           \
            static_cast<const __bf16*>(residual),                              \
            static_cast<const __bf16*>(gate), static_cast<__bf16*>(output),    \
            width, eps)
    if (width == 3840) {
        CK_LAUNCH_RMS_GATED(3840);
    } else {
        CK_LAUNCH_RMS_GATED(0);
    }
#undef CK_LAUNCH_RMS_GATED
}
