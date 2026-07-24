// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// ConvRot W4A4 on gfx12 WMMA (v_wmma_i32_16x16x32_iu4).
//
// out = (Aq[M, K/2] @ Wq[N, K/2]^T) * x_scale[row] * w_scale[col] + bias
//
// Both operands are signed int4 packed two per byte (low nibble = even index),
// which is the byte layout the iu4 A-fragment consumes directly. The activation
// rotation is fused into the quantizer.
#include <stdexcept>
#include <string>

#include "../epilogue.h"
#include "../gemm_wmma.h"
#include "../hadamard.h"

namespace comfy::hip_backend {

__global__ void unpack_int4_kernel(const int8_t* __restrict__ q, int8_t* __restrict__ out,
                                   int64_t nbytes) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= nbytes) return;
    const int8_t b = q[i];
    // sign-extend each nibble from 4 bits
    const int lo = static_cast<int>(b & 0xF);
    const int hi = static_cast<int>((b >> 4) & 0xF);
    out[2 * i] = static_cast<int8_t>(lo > 7 ? lo - 16 : lo);
    out[2 * i + 1] = static_cast<int8_t>(hi > 7 ? hi - 16 : hi);
}

template <typename OutT>
static void launch_w4a4(const uint8_t* A, const uint8_t* B, OutT* C, int M, int N, int K,
                        EpiRowwise epi, hipStream_t stream) {
    const int kbytes = K / 2;
    if (M >= 512 && N >= 128 && K > N) {
        constexpr int BM = 256, BN = 128, BKB = 64;
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_wmma_kernel<MmaInt4, EpiRowwise, OutT, BM, BN, BKB, 4, 2, 4, 4>
            <<<grid, 256, 0, stream>>>(A, B, C, M, N, kbytes, epi);
    } else if (M >= 96 && N >= 96) {
        constexpr int BM = 128, BN = 128, BKB = 64;
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_wmma_kernel<MmaInt4, EpiRowwise, OutT, BM, BN, BKB, 4, 2, 2, 4>
            <<<grid, 256, 0, stream>>>(A, B, C, M, N, kbytes, epi);
    } else {
        constexpr int BM = 64, BN = 64, BKB = 64;
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_wmma_kernel<MmaInt4, EpiRowwise, OutT, BM, BN, BKB, 2, 2, 2, 2>
            <<<grid, 128, 0, stream>>>(A, B, C, M, N, kbytes, epi);
    }
}

}  // namespace comfy::hip_backend

extern "C" void launch_convrot_w4a4_gemm_kernel(
    const void* a, const void* b, void* c,
    const void* x_scale, const void* w_scale, const void* bias, int bias_code,
    int M, int N, int K, int out_code, hipStream_t stream) {

    using namespace comfy::hip_backend;

    if (M == 0 || N == 0) {
        return;  // a zero-dimension grid is hipErrorInvalidConfiguration
    }
    // The tile loader reads a row 16 bytes at a time and int4 packs two per byte,
    // so the row length K/2 has to be a whole number of 16-byte chunks. An empty
    // GEMM does no reads, so this only applies once there is real work above.
    if (K % 32 != 0) {
        throw std::runtime_error("convrot_w4a4_linear: K=" + std::to_string(K) +
                                 " must be a multiple of 32");
    }

    EpiRowwise epi{static_cast<const float*>(x_scale), static_cast<const float*>(w_scale), 1,
                   bias, bias_code};

    const uint8_t* A = static_cast<const uint8_t*>(a);
    const uint8_t* B = static_cast<const uint8_t*>(b);

    if (out_code == kBF16) {
        launch_w4a4<__bf16>(A, B, static_cast<__bf16*>(c), M, N, K, epi, stream);
    } else if (out_code == kF16) {
        launch_w4a4<__half>(A, B, static_cast<__half*>(c), M, N, K, epi, stream);
    } else if (out_code == kF32) {
        launch_w4a4<float>(A, B, static_cast<float*>(c), M, N, K, epi, stream);
    } else {
        throw std::runtime_error("convrot_w4a4_linear: unsupported output dtype");
    }
}

// Fused Hadamard rotation + rowwise int4 quantize + nibble pack. Used for both
// the activation and the weight: both rotate along K and quantize per row.
extern "C" void launch_convrot_quant_int4_kernel(
    const void* x, int in_code, void* qout, void* scales,
    int M, int K, int group_size, hipStream_t stream) {

    using namespace comfy::hip_backend;
    check_convrot_group_size(group_size);
    check_convrot_k(K, group_size);
    if (M == 0) {
        return;  // a zero-block launch is hipErrorInvalidConfiguration
    }
    const size_t shmem = static_cast<size_t>(K) * sizeof(__bf16);
    convrot_quant_kernel<true><<<M, 256, shmem, stream>>>(
        x, in_code, static_cast<int8_t*>(qout), static_cast<float*>(scales), M, K, group_size);
}

// The largest K the fused rotation can stage in LDS, for the dispatch wrappers.
extern "C" int convrot_max_k_host() { return comfy::hip_backend::convrot_max_k(); }

extern "C" void launch_unpack_int4_kernel(
    const void* q, void* out, int64_t nbytes, hipStream_t stream) {

    using namespace comfy::hip_backend;
    if (nbytes == 0) {
        return;  // a zero-block launch is hipErrorInvalidConfiguration
    }
    const int threads = 256;
    const int blocks = static_cast<int>((nbytes + threads - 1) / threads);
    unpack_int4_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const int8_t*>(q), static_cast<int8_t*>(out), nbytes);
}
