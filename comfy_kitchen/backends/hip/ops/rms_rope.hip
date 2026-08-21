// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Fused RMSNorm + RoPE. Same operand layout as ops/apply_rope.hip. The norm
// reduces over head_dim, the last axis in both the BHND and BNHD layouts, so the
// kernel never has to know which of dim1/dim2 is the sequence.
//
// One block per row, as in ops/adaln.hip, but sized to one row rather than
// adaln's 256 threads: rows are short and numerous, so a wider block would idle.
// The row is re-read for the rotation rather than held in registers, which keeps
// head_dim unbounded.
//
// The normalized value is rounded to x's dtype before the rotation reads it back.
// That is the unfused contract, which every other backend matches.
#include <stdexcept>

#include "../hadamard.h"   // load_in / dtype codes
#include "../rope_math.h"  // rope_combine / rope_store / round_to
#include "tensor.h"

namespace comfy::hip_backend {

constexpr int kRmsRopeThreads = 64;

using TensorArg1 = tensor::TensorArg<1>;
using TensorArg4 = tensor::TensorArg<4>;
using TensorArg6 = tensor::TensorArg<6>;

// q_scale and k_scale need not share x's dtype, so they carry their own code.
// k is optional: rms_rope1 passes nullptr for it.
template <typename T>
__global__ __launch_bounds__(kRmsRopeThreads) void rms_rope_kernel(
    TensorArg4 q_arg, TensorArg4 k_arg, TensorArg6 freqs_arg,
    TensorArg1 q_scale_arg, TensorArg1 k_scale_arg, TensorArg4 q_out_arg,
    TensorArg4 k_out_arg, int rot_dim, float epsilon, bool has_k,
    bool split_half) {

    const void* q = q_arg.data;
    const void* k = k_arg.data;
    const void* freqs = freqs_arg.data;
    const void* q_scale = q_scale_arg.data;
    const void* k_scale = k_scale_arg.data;
    T* q_out = static_cast<T*>(q_out_arg.data);
    T* k_out = static_cast<T*>(k_out_arg.data);
    const int64_t dim1 = q_arg.meta.sizes[1];
    const int64_t dim2 = q_arg.meta.sizes[2];
    const int head_dim = static_cast<int>(q_arg.meta.sizes[3]);
    const int x_code = static_cast<int>(q_arg.meta.dtype);
    const int f_code = static_cast<int>(freqs_arg.meta.dtype);
    const int s_code = static_cast<int>(q_scale_arg.meta.dtype);

    __shared__ float red_q[kRmsRopeThreads];
    __shared__ float red_k[kRmsRopeThreads];

    const int t = threadIdx.x;
    const int64_t row = blockIdx.x;
    const int64_t d2 = row % dim2;
    const int64_t d1 = (row / dim2) % dim1;
    const int64_t b = row / (dim2 * dim1);

    // Inputs keep their own strides so a qkv slice is read in place; the outputs
    // are freshly allocated, so they carry their own set.
    const int64_t xoff =
        b * q_arg.meta.strides[0] + d1 * q_arg.meta.strides[1] +
        d2 * q_arg.meta.strides[2];
    const int64_t ooff =
        b * q_out_arg.meta.strides[0] + d1 * q_out_arg.meta.strides[1] +
        d2 * q_out_arg.meta.strides[2];
    float sq_q = 0.0f;
    float sq_k = 0.0f;
    for (int i = t; i < head_dim; i += kRmsRopeThreads) {
        const float v = load_in(q, xoff + i * q_arg.meta.strides[3], x_code);
        sq_q += v * v;
        if (has_k) {
            const float w = load_in(k, xoff + i * q_arg.meta.strides[3], x_code);
            sq_k += w * w;
        }
    }

    red_q[t] = sq_q;
    red_k[t] = sq_k;
    __syncthreads();
    for (int s = kRmsRopeThreads / 2; s > 0; s >>= 1) {
        if (t < s) {
            red_q[t] += red_q[t + s];
            red_k[t] += red_k[t + s];
        }
        __syncthreads();
    }

    const float inv_d = 1.0f / static_cast<float>(head_dim);
    const float rrms_q = rsqrtf(red_q[0] * inv_d + epsilon);
    const float rrms_k = has_k ? rsqrtf(red_k[0] * inv_d + epsilon) : 0.0f;

    const int64_t foff_row =
        (freqs_arg.meta.sizes[0] == 1 ? 0 : b) * freqs_arg.meta.strides[0] +
        (freqs_arg.meta.sizes[1] == 1 ? 0 : d1) * freqs_arg.meta.strides[1] +
        (freqs_arg.meta.sizes[2] == 1 ? 0 : d2) * freqs_arg.meta.strides[2];

    // Rotation covers the first rot_dim dims, in split-half pairs (p, p + rot_dim/2);
    // the reduction above always spans the full head_dim.
    const int n_pairs = rot_dim / 2;
    for (int p = t; p < n_pairs; p += kRmsRopeThreads) {
        const int64_t ia = split_half ? p : 2 * p;
        const int64_t ib = split_half ? p + n_pairs : 2 * p + 1;
        const int64_t ia_in = xoff + ia * q_arg.meta.strides[3];
        const int64_t ib_in = xoff + ib * q_arg.meta.strides[3];
        const int64_t ia_out = ooff + ia * q_out_arg.meta.strides[3];
        const int64_t ib_out = ooff + ib * q_out_arg.meta.strides[3];

        const int64_t foff =
            foff_row + static_cast<int64_t>(p) * freqs_arg.meta.strides[3];
        const float f00 = load_in(freqs, foff, f_code);
        const float f01 =
            load_in(freqs, foff + freqs_arg.meta.strides[5], f_code);
        const float f10 =
            load_in(freqs, foff + freqs_arg.meta.strides[4], f_code);
        const float f11 = load_in(
            freqs,
            foff + freqs_arg.meta.strides[4] + freqs_arg.meta.strides[5],
            f_code);

        const float sa = load_in(q_scale, ia, s_code);
        const float sb = load_in(q_scale, ib, s_code);
        const float qa = round_to<T>(load_in(q, ia_in, x_code) * rrms_q * sa);
        const float qb = round_to<T>(load_in(q, ib_in, x_code) * rrms_q * sb);
        rope_store<T>(q_out, ia_out, rope_combine(f00, qa, f01, qb, f_code, split_half));
        rope_store<T>(q_out, ib_out, rope_combine(f10, qa, f11, qb, f_code, split_half));

        if (has_k) {
            const float ta = load_in(k_scale, ia, s_code);
            const float tb = load_in(k_scale, ib, s_code);
            const float ka = round_to<T>(load_in(k, ia_in, x_code) * rrms_k * ta);
            const float kb = round_to<T>(load_in(k, ib_in, x_code) * rrms_k * tb);
            rope_store<T>(k_out, ia_out, rope_combine(f00, ka, f01, kb, f_code, split_half));
            rope_store<T>(k_out, ib_out, rope_combine(f10, ka, f11, kb, f_code, split_half));
        }
    }

    // Dims past rot_dim are normalized and scaled but never rotated. The store
    // rounds to T, which is the single rounding eager's rms_norm applies. Empty
    // when rot_dim == head_dim.
    for (int d = rot_dim + t; d < head_dim; d += kRmsRopeThreads) {
        const int64_t in = xoff + d * q_arg.meta.strides[3];
        const int64_t out = ooff + d * q_out_arg.meta.strides[3];
        rope_store<T>(q_out, out, load_in(q, in, x_code) * rrms_q * load_in(q_scale, d, s_code));
        if (has_k) {
            rope_store<T>(k_out, out, load_in(k, in, x_code) * rrms_k * load_in(k_scale, d, s_code));
        }
    }
}

}  // namespace comfy::hip_backend

extern "C" void launch_rms_rope_kernel(
    comfy::tensor::TensorArg<4> q, comfy::tensor::TensorArg<4> k,
    comfy::tensor::TensorArg<6> freqs, comfy::tensor::TensorArg<1> q_scale,
    comfy::tensor::TensorArg<1> k_scale, comfy::tensor::TensorArg<4> q_out,
    comfy::tensor::TensorArg<4> k_out, int64_t rot_dim, float epsilon,
    bool has_k, bool split_half, hipStream_t stream) {

    using namespace comfy::hip_backend;

    const int64_t head_dim = q.meta.sizes[3];
    const int x_code = static_cast<int>(q.meta.dtype);
    const int64_t rows = q.meta.sizes[0] * q.meta.sizes[1] * q.meta.sizes[2];
    // A zero-block launch is hipErrorInvalidConfiguration, and head_dim == 0 would
    // divide by zero in the reduction.
    if (rows == 0 || head_dim == 0) return;
    // One block per row, and gridDim.x is 32-bit.
    if (rows > 0xffffffffLL) {
        throw std::runtime_error("rms_rope: too many rows for a single grid");
    }
    // 0 rotates everything; anything else is a partial-rotary prefix.
    const int64_t rot = rot_dim > 0 ? rot_dim : head_dim;
    if (rot % 2 != 0 || rot > head_dim) {
        throw std::runtime_error("rms_rope: rot_dim must be even and <= head_dim");
    }

#define CK_RMS_ROPE_LAUNCH(T)                                                           \
    rms_rope_kernel<T><<<static_cast<unsigned int>(rows), kRmsRopeThreads, 0, stream>>>(\
        q, k, freqs, q_scale, k_scale, q_out, k_out, static_cast<int>(rot), epsilon,    \
        has_k, split_half)

    if (x_code == 0) {
        CK_RMS_ROPE_LAUNCH(float);
    } else if (x_code == 1) {
        CK_RMS_ROPE_LAUNCH(__half);
    } else if (x_code == 2) {
        CK_RMS_ROPE_LAUNCH(__bf16);
    } else {
        throw std::runtime_error("rms_rope: unsupported input dtype");
    }
#undef CK_RMS_ROPE_LAUNCH
}
