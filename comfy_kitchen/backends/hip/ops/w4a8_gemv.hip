// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Fused W4A8 / W6A8 GEMV for decode (M <= 8). The chunked path decodes the whole
// packed weight into an INT8 workspace and the GEMM reads it back, which at these row
// counts costs three times the traffic of the packed weight itself. Here one wave
// owns one output column, decodes its 16-column vectors in registers and dots
// them straight against the INT8 activation.
//
// The result is bit-identical to the chunked path: the same rounded int8 grid
// from w4a8_codec.h, and the same EpiRowwise applied to the same exact int32
// accumulator. Requires group_size >= 16 and a multiple of 16 so one 16-column
// vector never spans two groups.
#include <stdexcept>

#include "../epilogue.h"
#include "../mma.h"
#include "../w4a8_codec.h"

namespace comfy::hip_backend {

constexpr int kW4a8GemvMaxM = 8;  // sizes acc[] below and gates the launcher
constexpr int kW4a8GemvWaves = 8;

template <typename OutT, int BITS>
__global__ __launch_bounds__(kWave* kW4a8GemvWaves) void w4a8_codebook_gemv_kernel(
    const int8_t* __restrict__ xq,        // [M, K] rotated + quantized activation
    const int8_t* __restrict__ qw,        // [N, K * BITS / 8] packed codes
    const uint8_t* __restrict__ s_rel,    // [N, K/G] raw e4m3
    const float* __restrict__ codebook,   // 16 levels, or null for (code - 8)
    OutT* __restrict__ out,               // [M, N]
    int M, int N, int K, int group_size, EpiRowwise epi) {

    __shared__ float cb[16];
    if constexpr (BITS == 4) {
        if (threadIdx.x < 16) {
            cb[threadIdx.x] =
                codebook ? codebook[threadIdx.x] : (static_cast<float>(threadIdx.x) - 8.0f);
        }
        __syncthreads();
    }

    const int lane = threadIdx.x % kWave;
    const int n = blockIdx.x * kW4a8GemvWaves + static_cast<int>(threadIdx.x) / kWave;
    if (n >= N) {
        return;
    }

    const int k_half = K / 2;
    const int nvec = k_half / 8;  // 16-column vectors per weight row
    const int64_t row_bytes = static_cast<int64_t>(K) * BITS / 8;
    const int8_t* __restrict__ wrow = qw + static_cast<int64_t>(n) * row_bytes;
    const uint8_t* __restrict__ srow = s_rel + static_cast<int64_t>(n) * (K / group_size);

    // One pass over the weight serves every activation row, which is the whole
    // point at M <= 8. The row index stays a constant so acc[] keeps its registers
    // instead of spilling to scratch.
    int acc[kW4a8GemvMaxM];
#pragma unroll
    for (int m = 0; m < kW4a8GemvMaxM; ++m) {
        acc[m] = 0;
    }

    // Two 16-column vectors per step: one global_load_dwordx4 rather than two
    // dwordx2, which is what the decode ALU below needs to stay fed. That load is
    // only naturally aligned when both the row base (steps of K * BITS / 8 bytes)
    // and the half-row are multiples of 16 bytes, which is also when the pair loop
    // divides the row evenly. Anything else walks one vector per lane instead.
    if ((k_half & 15) == 0 && (row_bytes & 15) == 0) {
        for (int v = lane; v < nvec / 2; v += kWave) {
            const uint4 packed = *reinterpret_cast<const uint4*>(wrow + v * 16);
            uint2 hi = make_uint2(0u, 0u);
            if constexpr (BITS == 6) {
                hi = *reinterpret_cast<const uint2*>(wrow + k_half + v * 8);
            }
            const int k0 = v * 32;
            const float s0 = load_group_scale<uint8_t>(srow[k0 / group_size]);
            const float s1 = load_group_scale<uint8_t>(srow[(k0 + 16) / group_size]);
            const uint4 w0 = dequant16_to_int8<BITS>(make_uint2(packed.x, packed.y), hi.x, cb,
                                                     s0, s0, s0, s0, group_size);
            const uint4 w1 = dequant16_to_int8<BITS>(make_uint2(packed.z, packed.w), hi.y, cb,
                                                     s1, s1, s1, s1, group_size);
            const int kw = k0 >> 2;
#pragma unroll
            for (int m = 0; m < kW4a8GemvMaxM; ++m) {
                if (m < M) {
                    const int* __restrict__ x4 =
                        reinterpret_cast<const int*>(xq + static_cast<int64_t>(m) * K);
                    acc[m] = dot4_i8(x4[kw + 0], w0.x, acc[m]);
                    acc[m] = dot4_i8(x4[kw + 1], w0.y, acc[m]);
                    acc[m] = dot4_i8(x4[kw + 2], w0.z, acc[m]);
                    acc[m] = dot4_i8(x4[kw + 3], w0.w, acc[m]);
                    acc[m] = dot4_i8(x4[kw + 4], w1.x, acc[m]);
                    acc[m] = dot4_i8(x4[kw + 5], w1.y, acc[m]);
                    acc[m] = dot4_i8(x4[kw + 6], w1.z, acc[m]);
                    acc[m] = dot4_i8(x4[kw + 7], w1.w, acc[m]);
                }
            }
        }
    } else {
        for (int v = lane; v < nvec; v += kWave) {
            const uint2 packed = *reinterpret_cast<const uint2*>(wrow + v * 8);
            unsigned hi = 0u;
            if constexpr (BITS == 6) {
                hi = *reinterpret_cast<const unsigned*>(wrow + k_half + v * 4);
            }
            const int k0 = v * 16;
            const float s = load_group_scale<uint8_t>(srow[k0 / group_size]);
            const uint4 w = dequant16_to_int8<BITS>(packed, hi, cb, s, s, s, s, group_size);
            const int kw = k0 >> 2;
#pragma unroll
            for (int m = 0; m < kW4a8GemvMaxM; ++m) {
                if (m < M) {
                    const int* __restrict__ x4 =
                        reinterpret_cast<const int*>(xq + static_cast<int64_t>(m) * K);
                    acc[m] = dot4_i8(x4[kw + 0], w.x, acc[m]);
                    acc[m] = dot4_i8(x4[kw + 1], w.y, acc[m]);
                    acc[m] = dot4_i8(x4[kw + 2], w.z, acc[m]);
                    acc[m] = dot4_i8(x4[kw + 3], w.w, acc[m]);
                }
            }
        }
    }

    epi.init();
#pragma unroll
    for (int m = 0; m < kW4a8GemvMaxM; ++m) {
        if (m < M) {
            const int total = wave_reduce_sum(acc[m]);
            if (lane == 0) {
                out[static_cast<int64_t>(m) * N + n] =
                    static_cast<OutT>(epi(m, n, static_cast<float>(total)));
            }
        }
    }
}

}  // namespace comfy::hip_backend

// Returns false when the shape is outside the GEMV's envelope, so the caller can
// fall through to the chunked path rather than having to predict it.
extern "C" bool launch_w4a8_codebook_gemv_kernel(
    const void* xq, const void* qw, const void* s_rel, const void* codebook,
    const void* s_channel, const void* xs, const void* bias, int bias_code, void* out,
    int out_code, int M, int N, int K, int group_size, int bits, hipStream_t stream) {

    using namespace comfy::hip_backend;
    if (M < 1 || M > kW4a8GemvMaxM || N <= 0) {
        return false;
    }
    // A 16-column vector reads one group scale, and the dot4 operands are whole
    // dwords of the activation row.
    if (K % 16 != 0 || group_size < 16 || group_size % 16 != 0 || K % group_size != 0) {
        return false;
    }
    // 6-bit rows are 3K/4 bytes; K % 32 keeps them and the high plane 8-byte aligned.
    if ((bits != 4 && bits != 6) || (bits == 6 && (K % 32 != 0 || codebook))) {
        return false;
    }

    // Same epilogue object the INT8 GEMM builds, so the two paths round alike.
    EpiRowwise epi{static_cast<const float*>(xs), static_cast<const float*>(s_channel), 1, bias,
                   bias_code};
    const dim3 block(kWave * kW4a8GemvWaves);
    const dim3 grid(static_cast<unsigned int>((N + kW4a8GemvWaves - 1) / kW4a8GemvWaves));

#define CK_W4A8_GEMV_LAUNCH_BITS(OutT, BITS)                                                   \
    w4a8_codebook_gemv_kernel<OutT, BITS><<<grid, block, 0, stream>>>(                          \
        static_cast<const int8_t*>(xq), static_cast<const int8_t*>(qw),                         \
        static_cast<const uint8_t*>(s_rel), static_cast<const float*>(codebook),                \
        static_cast<OutT*>(out), M, N, K, group_size, epi)
#define CK_W4A8_GEMV_LAUNCH(OutT)                                                              \
    do {                                                                                       \
        if (bits == 6) {                                                                       \
            CK_W4A8_GEMV_LAUNCH_BITS(OutT, 6);                                                 \
        } else {                                                                               \
            CK_W4A8_GEMV_LAUNCH_BITS(OutT, 4);                                                 \
        }                                                                                      \
    } while (0)

    if (out_code == kBF16) {
        CK_W4A8_GEMV_LAUNCH(__bf16);
    } else if (out_code == kF16) {
        CK_W4A8_GEMV_LAUNCH(__half);
    } else if (out_code == kF32) {
        CK_W4A8_GEMV_LAUNCH(float);
    } else {
        throw std::runtime_error("w4a8_codebook_gemv: unsupported output dtype");
    }
#undef CK_W4A8_GEMV_LAUNCH
#undef CK_W4A8_GEMV_LAUNCH_BITS
    return true;
}
