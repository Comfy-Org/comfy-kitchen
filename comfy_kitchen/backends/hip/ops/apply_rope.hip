// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// RoPE. x is (batch, dim1, dim2, head_dim); freqs_cis is
// (fb, fd1, fd2, head_dim/2, 2, 2), broadcasting over any of its leading dims.
//
// For pair k:
//   out[a] = f[k][0][0] * x[a] + f[k][0][1] * x[b]
//   out[b] = f[k][1][0] * x[a] + f[k][1][1] * x[b]
// with (a, b) = (2k, 2k+1) for the interleaved layout and (k, k + head_dim/2)
// for the split-half layout. See rope_math.h for the rounding contract.
#include <stdexcept>

#include "../hadamard.h"    // load_in / dtype codes
#include "../rope_math.h"   // rope_combine / rope_store
#include "tensor.h"

namespace comfy::hip_backend {

constexpr int kRopeThreads = 256;

using TensorArg4 = tensor::TensorArg<4>;
using TensorArg6 = tensor::TensorArg<6>;

template <typename T>
__global__ void apply_rope_kernel(
    TensorArg4 xq_arg, TensorArg4 xk_arg, TensorArg6 freqs_arg,
    TensorArg4 xq_out_arg, TensorArg4 xk_out_arg, bool has_k,
    bool split_half) {

    const void* xq = xq_arg.data;
    const void* xk = xk_arg.data;
    const void* freqs = freqs_arg.data;
    T* xq_out = static_cast<T*>(xq_out_arg.data);
    T* xk_out = static_cast<T*>(xk_out_arg.data);
    const int64_t dim1 = xq_arg.meta.sizes[1];
    const int64_t dim2 = xq_arg.meta.sizes[2];
    const int64_t head_dim = xq_arg.meta.sizes[3];
    const int x_code = static_cast<int>(xq_arg.meta.dtype);
    const int f_code = static_cast<int>(freqs_arg.meta.dtype);

    const int64_t n_pairs = head_dim / 2;
    const int64_t total = xq_arg.meta.sizes[0] * dim1 * dim2 * n_pairs;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int64_t k = idx % n_pairs;
    int64_t tmp = idx / n_pairs;
    const int64_t d2 = tmp % dim2;
    tmp /= dim2;
    const int64_t d1 = tmp % dim1;
    const int64_t b = tmp / dim1;

    // Inputs keep their own strides so a qkv slice is read in place; the outputs
    // are freshly allocated, so they carry their own set.
    const int64_t xoff = b * xq_arg.meta.strides[0] +
                         d1 * xq_arg.meta.strides[1] +
                         d2 * xq_arg.meta.strides[2];
    const int64_t ooff = b * xq_out_arg.meta.strides[0] +
                         d1 * xq_out_arg.meta.strides[1] +
                         d2 * xq_out_arg.meta.strides[2];
    const int64_t ia = split_half ? k : 2 * k;
    const int64_t ib = split_half ? k + n_pairs : 2 * k + 1;
    const int64_t ia_in = xoff + ia * xq_arg.meta.strides[3];
    const int64_t ib_in = xoff + ib * xq_arg.meta.strides[3];
    const int64_t ia_out = ooff + ia * xq_out_arg.meta.strides[3];
    const int64_t ib_out = ooff + ib * xq_out_arg.meta.strides[3];

    const int64_t foff =
        (freqs_arg.meta.sizes[0] == 1 ? 0 : b) * freqs_arg.meta.strides[0] +
        (freqs_arg.meta.sizes[1] == 1 ? 0 : d1) * freqs_arg.meta.strides[1] +
        (freqs_arg.meta.sizes[2] == 1 ? 0 : d2) * freqs_arg.meta.strides[2] +
        k * freqs_arg.meta.strides[3];

    const float f00 = load_in(freqs, foff, f_code);
    const float f01 = load_in(freqs, foff + freqs_arg.meta.strides[5], f_code);
    const float f10 = load_in(freqs, foff + freqs_arg.meta.strides[4], f_code);
    const float f11 = load_in(
        freqs, foff + freqs_arg.meta.strides[4] + freqs_arg.meta.strides[5],
        f_code);

    const float qa = load_in(xq, ia_in, x_code);
    const float qb = load_in(xq, ib_in, x_code);
    rope_store<T>(xq_out, ia_out, rope_combine(f00, qa, f01, qb, f_code, split_half));
    rope_store<T>(xq_out, ib_out, rope_combine(f10, qa, f11, qb, f_code, split_half));

    if (has_k) {
        const float ka = load_in(xk, ia_in, x_code);
        const float kb = load_in(xk, ib_in, x_code);
        rope_store<T>(xk_out, ia_out, rope_combine(f00, ka, f01, kb, f_code, split_half));
        rope_store<T>(xk_out, ib_out, rope_combine(f10, ka, f11, kb, f_code, split_half));
    }
}

}  // namespace comfy::hip_backend

extern "C" void launch_apply_rope_kernel(
    comfy::tensor::TensorArg<4> xq, comfy::tensor::TensorArg<4> xk,
    comfy::tensor::TensorArg<6> freqs, comfy::tensor::TensorArg<4> xq_out,
    comfy::tensor::TensorArg<4> xk_out, bool has_k, bool split_half,
    hipStream_t stream) {

    using namespace comfy::hip_backend;

    const int64_t total = xq.meta.sizes[0] * xq.meta.sizes[1] *
                          xq.meta.sizes[2] * (xq.meta.sizes[3] / 2);
    const int blocks = static_cast<int>((total + kRopeThreads - 1) / kRopeThreads);
    if (blocks == 0) return;

#define CK_ROPE_LAUNCH(T)                                                   \
    apply_rope_kernel<T><<<blocks, kRopeThreads, 0, stream>>>(              \
        xq, xk, freqs, xq_out, xk_out, has_k, split_half)

    if (xq.meta.dtype == comfy::tensor::DType::Float32) {
        CK_ROPE_LAUNCH(float);
    } else if (xq.meta.dtype == comfy::tensor::DType::Float16) {
        CK_ROPE_LAUNCH(__half);
    } else if (xq.meta.dtype == comfy::tensor::DType::BFloat16) {
        CK_ROPE_LAUNCH(__bf16);
    } else {
        throw std::runtime_error("apply_rope: unsupported input dtype");
    }
#undef CK_ROPE_LAUNCH
}
