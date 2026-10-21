// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// GEMM epilogues. Each holds device pointers only and is passed to the tile
// kernel by value. init() runs once per thread before the writeback loop, so
// scalar scales are loaded once rather than per output element.
#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace comfy::hip_backend {

// dtype codes match comfy_kitchen.backends.eager.quantization.DTYPE_TO_CODE
constexpr int kF32 = 0;
constexpr int kF16 = 1;
constexpr int kBF16 = 2;

__forceinline__ __device__ float load_scalar(const void* p, int code, int64_t i) {
    if (code == kF32) return static_cast<const float*>(p)[i];
    if (code == kF16) return __half2float(static_cast<const __half*>(p)[i]);
    return static_cast<float>(static_cast<const __bf16*>(p)[i]);
}

__forceinline__ __device__ void store_scalar(void* p, int code, int64_t i, float v) {
    if (code == kF32) {
        static_cast<float*>(p)[i] = v;
    } else if (code == kF16) {
        static_cast<__half*>(p)[i] = __float2half(v);
    } else {
        static_cast<__bf16*>(p)[i] = static_cast<__bf16>(v);
    }
}

// out = acc * (scale_a * scale_b) + bias[col]
struct EpiTensorwise {
    const float* scale_a;
    const float* scale_b;
    const void* bias;
    int bias_code;
    float alpha;

    __forceinline__ __device__ void init() { alpha = scale_a[0] * scale_b[0]; }

    __forceinline__ __device__ float operator()(int, int col, float acc) const {
        float v = acc * alpha;
        if (bias) v += load_scalar(bias, bias_code, col);
        return v;
    }
};

// out = acc * scale_a[row] * scale_b[col * scale_b_stride] + bias[col]
// scale_b_stride is 1 for a per-output-channel weight scale, 0 for a scalar.
struct EpiRowwise {
    const float* scale_a;
    const float* scale_b;
    int scale_b_stride;
    const void* bias;
    int bias_code;

    __forceinline__ __device__ void init() {}

    __forceinline__ __device__ float operator()(int row, int col, float acc) const {
        float v = acc * scale_a[row] * scale_b[col * scale_b_stride];
        if (bias) v += load_scalar(bias, bias_code, col);
        return v;
    }
};

// Bias-free specialization for INT8 GEMM. It uses the
// same left-to-right scale expression as EpiRowwise without the runtime bias
// pointer/dtype branches in every unrolled output element.
struct EpiRowwiseNoBias {
    const float* scale_a;
    const float* scale_b;

    __forceinline__ __device__ void init() {}

    __forceinline__ __device__ float operator()(int row, int col, float acc) const {
        return acc * scale_a[row] * scale_b[col];
    }
};

// Round the linear result as the separate GEMM store would, then apply addcmul.
// Each output element reads its own residual before overwriting it; no split-K.
template <typename OutT>
struct EpiRowwiseResidual : EpiRowwise {
    static constexpr bool kOutputMayAlias = true;
    const OutT* residual;
    const OutT* rscale;
    int residual_stride;

    __forceinline__ __device__ float finish(int row, int col, float value) const {
        const float linear = static_cast<float>(static_cast<OutT>(value));
        const float res = static_cast<float>(residual[static_cast<int64_t>(row) * residual_stride + col]);
        return rscale ? fmaf(static_cast<float>(rscale[col]), linear, res) : res + linear;
    }

    __forceinline__ __device__ float operator()(int row, int col, float acc) const {
        return finish(row, col, EpiRowwise::operator()(row, col, acc));
    }
};

// Unscaled fp16 operands: out = acc + bias[col], or with resid set
// out = resid[row * resid_stride + col] + rscale[col] * (acc + bias[col]).
// resid_stride 0 broadcasts a single [N] residual row. All operands are fp16.
// rscale == nullptr is a unit scale. resid may be the GEMM output (see GemmOutput).
struct EpiFp16 {
    static constexpr bool kOutputMayAlias = true;

    const __half* bias;
    const __half* rscale;
    const __half* resid;
    int resid_stride;

    __forceinline__ __device__ void init() {}

    __forceinline__ __device__ float operator()(int row, int col, float acc) const {
        float v = acc;
        if (bias) v += __half2float(bias[col]);
        if (resid) {
            v = __half2float(resid[static_cast<int64_t>(row) * resid_stride + col]) +
                (rscale ? __half2float(rscale[col]) * v : v);
        }
        return v;
    }
};

}  // namespace comfy::hip_backend
