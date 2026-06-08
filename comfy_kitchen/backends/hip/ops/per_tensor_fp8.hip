#include <cmath>
#include <cstdint>
#include <stdexcept>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace comfy::hip_backend {

constexpr int kThreads = 256;

__forceinline__ __device__ float clampf(float v, float lo, float hi) {
    return fminf(hi, fmaxf(lo, v));
}

template<typename T>
__forceinline__ __device__ float to_float(T value) {
    return static_cast<float>(value);
}

template<>
__forceinline__ __device__ float to_float<__half>(__half value) {
    return __half2float(value);
}

template<>
__forceinline__ __device__ float to_float<__hip_bfloat16>(__hip_bfloat16 value) {
    return static_cast<float>(value);
}

template<typename T>
__forceinline__ __device__ void store_value(void* dst, int64_t idx, float value) {
    reinterpret_cast<T*>(dst)[idx] = static_cast<T>(value);
}

template<>
__forceinline__ __device__ void store_value<__half>(void* dst, int64_t idx, float value) {
    reinterpret_cast<__half*>(dst)[idx] = __float2half(value);
}

template<>
__forceinline__ __device__ void store_value<__hip_bfloat16>(void* dst, int64_t idx, float value) {
    reinterpret_cast<__hip_bfloat16*>(dst)[idx] = __float2bfloat16(value);
}

__forceinline__ __device__ uint8_t pack_fp8(float value, int dtype_code) {
    const bool e4m3 = dtype_code == 5;
    const int mantissa_bits = e4m3 ? 3 : 2;
    const int exponent_bias = e4m3 ? 7 : 15;
    const int mantissa_levels = 1 << mantissa_bits;
    const int max_exponent_field = e4m3 ? 15 : 30;
    const int max_mantissa_field = e4m3 ? 6 : (mantissa_levels - 1);
    const float fp8_max = e4m3 ? 448.0f : 57344.0f;

    const uint8_t sign_bit = signbit(value) ? 0x80 : 0x00;
    float abs_value = fabsf(value);
    if (isnan(abs_value)) {
        return e4m3 ? static_cast<uint8_t>(sign_bit | 0x7f) : static_cast<uint8_t>(sign_bit | 0x7e);
    }

    abs_value = clampf(abs_value, 0.0f, fp8_max);
    if (abs_value == 0.0f) {
        return sign_bit;
    }

    const float min_normal = exp2f(1 - exponent_bias);
    int exponent_field = 0;
    int mantissa_field = 0;
    if (abs_value < min_normal) {
        const float subnormal_scale = exp2f(1 - exponent_bias - mantissa_bits);
        mantissa_field = static_cast<int>(floorf(abs_value / subnormal_scale + 0.5f));
        if (mantissa_field >= mantissa_levels) {
            exponent_field = 1;
            mantissa_field = 0;
        }
    } else {
        exponent_field = static_cast<int>(floorf(log2f(abs_value))) + exponent_bias;
        const float exponent_scale = exp2f(exponent_field - exponent_bias);
        mantissa_field = static_cast<int>(floorf((abs_value / exponent_scale - 1.0f) * mantissa_levels + 0.5f));
        if (mantissa_field >= mantissa_levels) {
            mantissa_field = 0;
            exponent_field += 1;
        }
    }

    if (exponent_field > max_exponent_field ||
        (exponent_field == max_exponent_field && mantissa_field > max_mantissa_field)) {
        exponent_field = max_exponent_field;
        mantissa_field = max_mantissa_field;
    }

    return static_cast<uint8_t>(sign_bit | (exponent_field << mantissa_bits) | mantissa_field);
}

__forceinline__ __device__ float unpack_fp8(uint8_t value, int dtype_code = 5) {
    const bool e4m3 = dtype_code == 5;
    const int mantissa_bits = e4m3 ? 3 : 2;
    const int exponent_bits = e4m3 ? 4 : 5;
    const int exponent_bias = e4m3 ? 7 : 15;
    const int mantissa_mask = (1 << mantissa_bits) - 1;
    const int exponent_mask = (1 << exponent_bits) - 1;

    const float sign = (value & 0x80) ? -1.0f : 1.0f;
    const int exponent = (value >> mantissa_bits) & exponent_mask;
    const int mantissa = value & mantissa_mask;
    if (exponent == 0) {
        if (mantissa == 0) {
            return sign * 0.0f;
        }
        return sign * exp2f(1 - exponent_bias) * (static_cast<float>(mantissa) / (1 << mantissa_bits));
    }
    return sign * exp2f(exponent - exponent_bias) * (1.0f + static_cast<float>(mantissa) / (1 << mantissa_bits));
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

    const int blocks = static_cast<int>((numel + comfy::hip_backend::kThreads - 1) / comfy::hip_backend::kThreads);
    if (input_dtype_code == 0) {
        comfy::hip_backend::quantize_per_tensor_fp8_kernel<float><<<blocks, comfy::hip_backend::kThreads, 0, stream>>>(static_cast<const float*>(input), static_cast<const float*>(scale), static_cast<uint8_t*>(output), numel, output_dtype_code);
    } else if (input_dtype_code == 1) {
        comfy::hip_backend::quantize_per_tensor_fp8_kernel<__half><<<blocks, comfy::hip_backend::kThreads, 0, stream>>>(static_cast<const __half*>(input), static_cast<const float*>(scale), static_cast<uint8_t*>(output), numel, output_dtype_code);
    } else if (input_dtype_code == 2) {
        comfy::hip_backend::quantize_per_tensor_fp8_kernel<__hip_bfloat16><<<blocks, comfy::hip_backend::kThreads, 0, stream>>>(static_cast<const __hip_bfloat16*>(input), static_cast<const float*>(scale), static_cast<uint8_t*>(output), numel, output_dtype_code);
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

    const int blocks = static_cast<int>((numel + comfy::hip_backend::kThreads - 1) / comfy::hip_backend::kThreads);
    if (output_dtype_code == 0) {
        comfy::hip_backend::dequantize_per_tensor_fp8_kernel<float><<<blocks, comfy::hip_backend::kThreads, 0, stream>>>(static_cast<const uint8_t*>(input), static_cast<const float*>(scale), output, numel, input_dtype_code);
    } else if (output_dtype_code == 1) {
        comfy::hip_backend::dequantize_per_tensor_fp8_kernel<__half><<<blocks, comfy::hip_backend::kThreads, 0, stream>>>(static_cast<const uint8_t*>(input), static_cast<const float*>(scale), output, numel, input_dtype_code);
    } else if (output_dtype_code == 2) {
        comfy::hip_backend::dequantize_per_tensor_fp8_kernel<__hip_bfloat16><<<blocks, comfy::hip_backend::kThreads, 0, stream>>>(static_cast<const uint8_t*>(input), static_cast<const float*>(scale), output, numel, input_dtype_code);
    } else {
        throw std::runtime_error("Unsupported output dtype for dequantize_per_tensor_fp8");
    }
}

} // extern "C"
