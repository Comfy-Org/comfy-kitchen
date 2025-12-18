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

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace comfy {

// =============================================================================
// Dtype Code Constants
// These MUST match the Python DTYPE_CODE_TO_DTYPE in:
//   comfy_kitchen/backends/eager/quantization.py
// =============================================================================

// Standard floating point types
constexpr int DTYPE_CODE_FLOAT32 = 0;
constexpr int DTYPE_CODE_FLOAT16 = 1;
constexpr int DTYPE_CODE_BFLOAT16 = 2;

// FP8 types
constexpr int DTYPE_CODE_FLOAT8_E4M3FN = 5;
constexpr int DTYPE_CODE_FLOAT8_E5M2 = 6;

// =============================================================================
// Type Mappings
// =============================================================================

// Type mapping for input dtype codes
template<int DTypeCode> struct DTypeMap;

template<> struct DTypeMap<DTYPE_CODE_FLOAT32> { using type = float; };
template<> struct DTypeMap<DTYPE_CODE_FLOAT16> { using type = half; };
template<> struct DTypeMap<DTYPE_CODE_BFLOAT16> { using type = nv_bfloat16; };

// Type mapping for FP8 output dtype codes
template<int DTypeCode> struct FP8DTypeMap;

template<> struct FP8DTypeMap<DTYPE_CODE_FLOAT8_E4M3FN> { using type = __nv_fp8_e4m3; };
template<> struct FP8DTypeMap<DTYPE_CODE_FLOAT8_E5M2> { using type = __nv_fp8_e5m2; };

// =============================================================================
// Runtime Dtype Code to CUDA Type Conversion
// =============================================================================

inline cudaDataType_t dtype_code_to_cuda_type(int dtype_code) {
    switch (dtype_code) {
        case DTYPE_CODE_FLOAT32: return CUDA_R_32F;
        case DTYPE_CODE_FLOAT16: return CUDA_R_16F;
        case DTYPE_CODE_BFLOAT16: return CUDA_R_16BF;
        default:
            throw std::runtime_error(
                "Unsupported dtype code for CUDA type: " + std::to_string(dtype_code));
    }
}

} // namespace comfy

// Macro for dispatching based on a single input dtype code
// Usage: DISPATCH_INPUT_DTYPE(dtype_code, InputType, { /* code using InputType */ });
#define DISPATCH_INPUT_DTYPE(dtype_code, type_name, ...)                       \
    [&] {                                                                       \
        switch (dtype_code) {                                                   \
            case comfy::DTYPE_CODE_FLOAT32: {                                   \
                using type_name = float;                                        \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case comfy::DTYPE_CODE_FLOAT16: {                                   \
                using type_name = half;                                         \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case comfy::DTYPE_CODE_BFLOAT16: {                                  \
                using type_name = nv_bfloat16;                                  \
                return __VA_ARGS__();                                           \
            }                                                                   \
            default:                                                            \
                throw std::runtime_error(                                       \
                    "Unsupported input dtype code: " +                          \
                    std::to_string(dtype_code));                                \
        }                                                                       \
    }()

// Macro for dispatching based on FP8 output dtype code
// Usage: DISPATCH_FP8_OUTPUT_DTYPE(dtype_code, OutputType, { /* code using OutputType */ });
#define DISPATCH_FP8_OUTPUT_DTYPE(dtype_code, type_name, ...)                  \
    [&] {                                                                       \
        switch (dtype_code) {                                                   \
            case comfy::DTYPE_CODE_FLOAT8_E4M3FN: {                             \
                using type_name = __nv_fp8_e4m3;                                \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case comfy::DTYPE_CODE_FLOAT8_E5M2: {                               \
                using type_name = __nv_fp8_e5m2;                                \
                return __VA_ARGS__();                                           \
            }                                                                   \
            default:                                                            \
                throw std::runtime_error(                                       \
                    "Unsupported FP8 output dtype code: " +                     \
                    std::to_string(dtype_code));                                \
        }                                                                       \
    }()

// Macro for nested dispatch on both input and FP8 output dtypes
// Usage: DISPATCH_INPUT_FP8_OUTPUT_DTYPES(in_code, out_code, InputType, OutputType,
//                                          { /* code using InputType and OutputType */ });
#define DISPATCH_INPUT_FP8_OUTPUT_DTYPES(in_dtype_code, out_dtype_code,        \
                                         in_type_name, out_type_name, ...)      \
    DISPATCH_INPUT_DTYPE(in_dtype_code, in_type_name, [&] {                    \
        return DISPATCH_FP8_OUTPUT_DTYPE(out_dtype_code, out_type_name,        \
                                         __VA_ARGS__);                          \
    })

// Macro for dispatching only FP16/BF16 types (no FP32 support)
// Usage: DISPATCH_HALF_DTYPE(dtype_code, HalfType, { /* code using HalfType */ });
#define DISPATCH_HALF_DTYPE(dtype_code, type_name, ...)                        \
    [&] {                                                                       \
        switch (dtype_code) {                                                   \
            case comfy::DTYPE_CODE_FLOAT16: {                                   \
                using type_name = half;                                         \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case comfy::DTYPE_CODE_BFLOAT16: {                                  \
                using type_name = nv_bfloat16;                                  \
                return __VA_ARGS__();                                           \
            }                                                                   \
            default:                                                            \
                throw std::runtime_error(                                       \
                    "Unsupported dtype code (only FP16/BF16 supported): " +     \
                    std::to_string(dtype_code));                                \
        }                                                                       \
    }()

// Macro for dispatching FP32/FP16/BF16 types
// Usage: DISPATCH_FP_DTYPE(dtype_code, FPType, { /* code using FPType */ });
#define DISPATCH_FP_DTYPE(dtype_code, type_name, ...)                          \
    [&] {                                                                       \
        switch (dtype_code) {                                                   \
            case comfy::DTYPE_CODE_FLOAT32: {                                   \
                using type_name = float;                                        \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case comfy::DTYPE_CODE_FLOAT16: {                                   \
                using type_name = half;                                         \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case comfy::DTYPE_CODE_BFLOAT16: {                                  \
                using type_name = nv_bfloat16;                                  \
                return __VA_ARGS__();                                           \
            }                                                                   \
            default:                                                            \
                throw std::runtime_error(                                       \
                    "Unsupported dtype code (only FP32/FP16/BF16 supported): " +\
                    std::to_string(dtype_code));                                \
        }                                                                       \
    }()

// Macro for nested dispatch on FP8 input and FP16/BF16 output dtypes
// Usage: DISPATCH_FP8_INPUT_HALF_OUTPUT_DTYPES(in_code, out_code, InputType, OutputType,
//                                               { /* code using InputType and OutputType */ });
#define DISPATCH_FP8_INPUT_HALF_OUTPUT_DTYPES(in_dtype_code, out_dtype_code,   \
                                               in_type_name, out_type_name, ...)    \
    DISPATCH_FP8_OUTPUT_DTYPE(in_dtype_code, in_type_name, [&] {               \
        return DISPATCH_HALF_DTYPE(out_dtype_code, out_type_name,              \
                                   __VA_ARGS__);                                \
    })

// Macro for nested dispatch on FP8 input and FP32/FP16/BF16 output dtypes
// Usage: DISPATCH_FP8_INPUT_FP_OUTPUT_DTYPES(in_code, out_code, InputType, OutputType,
//                                            { /* code using InputType and OutputType */ });
#define DISPATCH_FP8_INPUT_FP_OUTPUT_DTYPES(in_dtype_code, out_dtype_code,     \
                                             in_type_name, out_type_name, ...) \
    DISPATCH_FP8_OUTPUT_DTYPE(in_dtype_code, in_type_name, [&] {               \
        return DISPATCH_FP_DTYPE(out_dtype_code, out_type_name,                \
                                 __VA_ARGS__);                                  \
    })

