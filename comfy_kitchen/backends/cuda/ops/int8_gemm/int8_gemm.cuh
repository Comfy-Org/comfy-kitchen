/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace comfy::int8_gemm {

struct Problem {
    const int8_t* a;
    const int8_t* b;
    const float* x_scale;
    const float* w_scale;
    const __nv_bfloat16* bias;
    __nv_bfloat16* out;
    int m, n, k;
};

bool launch_sm90(const Problem&, cudaStream_t);
bool launch_sm100(const Problem&, cudaStream_t);

}  // namespace comfy::int8_gemm
