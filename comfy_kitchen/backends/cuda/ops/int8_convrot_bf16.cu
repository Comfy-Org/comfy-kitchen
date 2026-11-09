/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "int8_convrot_bf16.cuh"
#include <cudaTypedefs.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace comfy {

// This translation unit uses architecture-specific images, independently of the
// portable fatbin. No CuTe DSL, Python compiler, or driver-library link needed.
bool try_quantize_int8_convrot_bf16_4096(const void *input, void *output, float *scales,
                                         int64_t rows, cudaStream_t stream) {
    if (rows <= 8 || rows > std::numeric_limits<int>::max() ||
        reinterpret_cast<uintptr_t>(input) % 16 || reinterpret_cast<uintptr_t>(output) % 16) {
        return false;
    }

    int device, major, minor;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) {
        throw std::runtime_error("ConvRot: cannot query the current CUDA device");
    }
    bool supported = false;
#ifdef COMFY_CONVROT_SM100
    supported |= major == 10 && minor == 0;
#endif
#ifdef COMFY_CONVROT_SM120
    supported |= major == 12 && minor == 0;
#endif
    if (!supported) {
        return false;
    }

    // Resolve once, without adding a libcuda dependency. Encoding is host-only
    // and uses the current call's pointer and dimensions, including in capture.
    static const auto encode = []() -> PFN_cuTensorMapEncodeTiled_v12000 {
        void *address = nullptr;
        cudaDriverEntryPointQueryResult status;
        if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &address, 12000,
                                             cudaEnableDefault, &status) != cudaSuccess ||
            status != cudaDriverEntryPointSuccess) {
            return nullptr;
        }
        return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(address);
    }();
    if (!encode) {
        return false;
    }

    CUtensorMap map;
    uint64_t dimensions[] = {64, 64, static_cast<uint64_t>(rows)};
    uint64_t strides[] = {128, 8192};
    uint32_t box[] = {64, 64, 1}, element_strides[] = {1, 1, 1};
    CUresult result = encode(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void *>(input),
                             dimensions, strides, box, element_strides,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                             CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error("ConvRot: tensor-map encoding failed (" +
                                 std::to_string(static_cast<int>(result)) + ")");
    }

    convrot_bf16::kernel<<<static_cast<unsigned int>(rows), 64, 0, stream>>>(
        map, static_cast<int8_t *>(output), scales);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA INT8 ConvRot BF16 quantization failed: ") +
                                 cudaGetErrorString(error));
    }
    return true;
}

} // namespace comfy
