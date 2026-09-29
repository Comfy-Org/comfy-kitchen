// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Exact BF16 RMSNorm + BF16 modulation + ConvRot + INT8
// rowwise quantization. The RMS statistic uses IEEE division and OCML rsqrt
// so it matches PyTorch under -ffast-math; the ConvRot pass is the same
// 512-thread schedule as the unfused producer.
#include <stdexcept>

#include "../hadamard.h"

extern "C" void launch_quantize_int8_convrot_rms_modulated_fused_stats_kernel(
    const void* x, const void* norm_weight, const void* modulation_scale,
    void* q, void* scales, int M, int K, int group_size, float eps,
    hipStream_t stream) {

    using namespace comfy::hip_backend;
    if (K != 3840 || group_size != 256) {
        throw std::runtime_error(
            "convrot fused rms modulated: requires BF16 K=3840 and group_size=256");
    }
    check_convrot_k(K, group_size, 2);
    if (!convrot_512x2_supported(K)) {
        throw std::runtime_error(
            "convrot fused rms modulated: requires wave32 and 512-thread workgroups");
    }
    if (M == 0) return;

    // <ACT, TILED_QOUT, SPLIT_SWIGLU, MODULATE, RMSNORM, PACKED_ELEMENT_SCHEDULE>
    convrot_quant_512x2_bf16_kernel<
        kActNone, false, false, true, true, true>
        <<<M, 512, static_cast<size_t>(K) * sizeof(__bf16), stream>>>(
            x, modulation_scale, 2, static_cast<int8_t*>(q),
            static_cast<float*>(scales), M, K, norm_weight, eps);
}
