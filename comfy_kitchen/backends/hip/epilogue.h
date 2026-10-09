// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// GEMM epilogues. Each holds device pointers only and is passed to the tile
// kernel by value. init() runs once per thread before the writeback loop, so
// scalar scales are loaded once rather than per output element.
//
// row_scale / col_scale / col_bias exist so the tile kernel can hoist the loads
// out of the writeback loop. It cannot do that itself: the loop's stores go
// through OutT*, which the compiler must assume may alias the scale / bias /
// residual pointers, so every load in it is re-issued per output element. One
// lane writes TM*8*TN outputs, so a kernel whose MMA side already measures ~80%
// of this part's int8 ceiling was paying 2-3 dependent loads per store on top.
// Each policy returns exactly what its operator() used to load, so the arithmetic
// is unchanged -- see EpiRowwise below for the one place where "unchanged" has to
// be read as "equal to within one output ULP" rather than bit-identical.
// COMFY_EPI_UNHOISTED keeps the 3-argument spelling for the kernels that write one
// element per call and gain nothing from hoisting.
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

#define COMFY_EPI_UNHOISTED(Epi)                                                        \
    __forceinline__ __device__ float operator()(int row, int col, float acc) const {     \
        return (*this)(row, col, acc, row_scale(row), col_scale(col), col_bias(col));   \
    }

// out = acc * (scale_a * scale_b) + bias[col]
struct EpiTensorwise {
    const float* scale_a;
    const float* scale_b;
    const void* bias;
    int bias_code;
    float alpha;

    __forceinline__ __device__ void init() { alpha = scale_a[0] * scale_b[0]; }

    __forceinline__ __device__ float row_scale(int) const { return alpha; }
    __forceinline__ __device__ float col_scale(int) const { return 1.0f; }
    __forceinline__ __device__ float col_bias(int col) const {
        return bias ? load_scalar(bias, bias_code, col) : 0.0f;
    }

    __forceinline__ __device__ float operator()(int row, int col, float acc, float rs, float cs,
                                                float cb) const {
        float v = acc * rs * cs;
        if (bias) v += cb;
        return v;
    }
    COMFY_EPI_UNHOISTED(EpiTensorwise)
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

    __forceinline__ __device__ float row_scale(int row) const { return scale_a[row]; }
    __forceinline__ __device__ float col_scale(int col) const {
        return scale_b[col * scale_b_stride];
    }
    __forceinline__ __device__ float col_bias(int col) const {
        return bias ? load_scalar(bias, bias_code, col) : 0.0f;
    }

    // **Not bit-identical to the pre-hoist build, and that is worth stating plainly.**
    // Hoisting the operands changed which of the two multiplies clang pairs first, and
    // -ffast-math (CMakeLists.txt:206) allows that reassociation, so ~1e-5 of a large
    // GEMM's output elements land on the other side of the output-dtype rounding
    // boundary: measured 236/16777216 at 8192x2048x2048 and 948/67108864 at
    // 8192x8192x2048, every one of them exactly one bf16 ULP (max|d| 0.25, 0.125).
    // Against an fp32 reference **both** spellings sit at the bf16 rounding floor --
    // 0.5 ULP mean, 1.00 ULP max, 0.0000% vs 0.0003% of elements past 1 ULP -- so
    // neither is the more correct answer; it is which side of the boundary a handful of
    // elements fall on. The fp16 policy and the M<=8 GEMV path are bit-identical.
    //
    // Spelling the product as `t = acc * rs; v = t * cs` was tried to pin the order and
    // does not reach bit-exactness (it only shifts which elements move), so it is not
    // here: reproducing the old bits means reproducing clang's arbitrary choice under
    // -ffast-math, which is not a property worth claiming. Measure instead of asserting
    // -- benchmark_attn/kernel_bitexact.py (same-process A/B) and ulp_check.py (error
    // against fp32) are the two tools this was settled with.
    __forceinline__ __device__ float operator()(int row, int col, float acc, float rs, float cs,
                                                float cb) const {
        float v = acc * rs * cs;
        if (bias) v += cb;
        return v;
    }
    COMFY_EPI_UNHOISTED(EpiRowwise)
};

// Unscaled fp16 operands: out = acc + bias[col], or with resid set
// out = resid[row * resid_stride + col] + (rscale[col] * (acc + bias[col])).
// resid_stride 0 broadcasts a single [N] residual row. All operands are fp16.
//
// Only bias and rscale hoist; the residual is per (row, col), so it stays in the
// loop, but it is then the only load left there.
struct EpiFp16 {
    const __half* bias;
    const __half* rscale;
    const __half* resid;
    int resid_stride;

    __forceinline__ __device__ void init() {}

    __forceinline__ __device__ float row_scale(int) const { return 1.0f; }
    __forceinline__ __device__ float col_scale(int col) const {
        return rscale ? __half2float(rscale[col]) : 1.0f;
    }
    __forceinline__ __device__ float col_bias(int col) const {
        return bias ? __half2float(bias[col]) : 0.0f;
    }

    __forceinline__ __device__ float operator()(int row, int col, float acc, float rs, float cs,
                                                float cb) const {
        float v = acc * rs;
        if (bias) v += cb;
        if (resid) {
            v = __half2float(resid[static_cast<int64_t>(row) * resid_stride + col]) +
                (rscale ? cs * v : v);
        }
        return v;
    }
    COMFY_EPI_UNHOISTED(EpiFp16)
};

}  // namespace comfy::hip_backend