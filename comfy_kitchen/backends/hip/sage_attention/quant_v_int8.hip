// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Fused V to signed INT8 with a per-channel scale, transposed to
// [B * H * D, padded_N] so the attention kernel reads a V fragment as a
// contiguous run of keys along one channel row.
//
// The CUDA backend permutes each group of 16 keys here to match what ldmatrix
// hands its MMA. RDNA needs no permutation: the probability fragment and the V
// fragment both index keys in natural order (see int8_attn.h), so keys are
// written where they belong and the tail is zero filled.
//
// One block per (b, h, d_tile) of 8 channels. Pass 1 reduces the per-channel
// absmax along N with 4x unrolled vector loads, pass 2 quantizes.
#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

#include "sage_common.h"

namespace comfy::hip_backend::sage {
namespace {

constexpr int kDTile = 8;

// Optimized producer: one scale per (head, 64-key tile, channel), with
// the quantized bytes written directly as [head, tile, D, 64].  A synchronous
// BF16 LDS tile keeps both the physical-NHD input loads and tile-major output
// stores vectorized; this kernel family has no global-to-LDS async/direct path.
template <typename T>
__global__ __launch_bounds__(256) void gfx12_tile_channel_v_quant_kernel(
    const T* __restrict__ v, int8_t* __restrict__ output,
    float* __restrict__ descale, int n_ctx, int tiles,
    int64_t stride_h, int64_t stride_n) {
    constexpr int kTileN = 64;
    constexpr int kD = 128;
    constexpr int kStageStride = kD + 8;
    __shared__ __align__(16) T staged[kTileN * kStageStride];
    __shared__ float multiplier[kD];

    const int tid = threadIdx.x;
    const int tile = blockIdx.x;
    const int head = blockIdx.y;
    const int n0 = tile * kTileN;

    // 64*128 BF16 values = 1024 aligned 16-byte chunks, four per thread.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int chunk = tid + i * 256;
        const int row = chunk >> 4;
        const int d8 = (chunk & 15) << 3;
        uint4 value = make_uint4(0, 0, 0, 0);
        if (n0 + row < n_ctx) {
            value = *reinterpret_cast<const uint4*>(
                v + static_cast<int64_t>(head) * stride_h +
                static_cast<int64_t>(n0 + row) * stride_n + d8);
        }
        *reinterpret_cast<uint4*>(staged + row * kStageStride + d8) = value;
    }
    __syncthreads();

    if (tid < kD) {
        float maximum = 0.0f;
#pragma unroll
        for (int row = 0; row < kTileN; ++row) {
            maximum = fmaxf(maximum, fabsf(to_float(
                staged[row * kStageStride + tid])));
        }
        if (maximum == 0.0f) maximum = 1.0e-8f;
        descale[(static_cast<int64_t>(head) * tiles + tile) * kD + tid] =
            maximum * (1.0f / 127.0f);
        multiplier[tid] = __builtin_amdgcn_rcpf(maximum) * 127.0f;
    }
    __syncthreads();

    // 128 rows of 64 output bytes = 512 uint4 chunks, two per thread.
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int chunk = tid + i * 256;
        const int d = chunk >> 2;
        const int n16 = (chunk & 3) << 4;
        uint4 packed = make_uint4(0, 0, 0, 0);
        uint32_t* words = reinterpret_cast<uint32_t*>(&packed);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const uint32_t code = static_cast<uint8_t>(float_to_int8_rn(
                to_float(staged[(n16 + j) * kStageStride + d]) *
                multiplier[d]));
            words[j >> 2] |= code << (8 * (j & 3));
        }
        *reinterpret_cast<uint4*>(
            output + static_cast<int64_t>(head) * tiles * kD * kTileN +
            static_cast<int64_t>(tile) * kD * kTileN + d * kTileN + n16) = packed;
    }
}

template <typename T>
void launch_gfx12_tile_channel_v_quant(
    const T* v, int8_t* output, float* descale,
    int num_heads, int n_ctx, int tiles,
    int64_t stride_h, int64_t stride_n,
    hipStream_t stream) {
    gfx12_tile_channel_v_quant_kernel<T><<<dim3(tiles, num_heads), 256, 0, stream>>>(
        v, output, descale, n_ctx, tiles, stride_h, stride_n);
}

// Eight adjacent channels: 32 bytes for fp32, 16 for the 16-bit types.
template <typename T>
__forceinline__ __device__ void load8(const T* p, float* out) {
    if constexpr (std::is_same_v<T, float>) {
        const float4 lo = *reinterpret_cast<const float4*>(p);
        const float4 hi = *reinterpret_cast<const float4*>(p + 4);
        out[0] = lo.x;
        out[1] = lo.y;
        out[2] = lo.z;
        out[3] = lo.w;
        out[4] = hi.x;
        out[5] = hi.y;
        out[6] = hi.z;
        out[7] = hi.w;
    } else {
        const uint4 raw = *reinterpret_cast<const uint4*>(p);
        const T* vals = reinterpret_cast<const T*>(&raw);
#pragma unroll
        for (int i = 0; i < kDTile; ++i) out[i] = to_float(vals[i]);
    }
}

template <typename T, int Threads>
__global__ __launch_bounds__(Threads) void quant_v_int8_kernel(
    const T* __restrict__ v, int8_t* __restrict__ out, float* __restrict__ scale_out, int N,
    int padded_N, int H, int D, int64_t stride_b, int64_t stride_h, int64_t stride_n) {

    constexpr int kWaves = Threads / 32;

    const int d_tiles = D / kDTile;
    const int d_tile = blockIdx.x % d_tiles;
    const int bh = blockIdx.x / d_tiles;
    const int h = bh % H;
    const int b = bh / H;
    const int d0 = d_tile * kDTile;

    const T* base = v + static_cast<int64_t>(b) * stride_b + static_cast<int64_t>(h) * stride_h +
                    d0;

    float mx[kDTile];
#pragma unroll
    for (int i = 0; i < kDTile; ++i) mx[i] = 0.0f;

    // Four independent loads per iteration so the memory system has something to
    // overlap; this block is the only one touching these channels.
    int n = threadIdx.x;
    const int n_body = N - 3 * Threads;
    for (; n < n_body; n += 4 * Threads) {
        float t0[kDTile], t1[kDTile], t2[kDTile], t3[kDTile];
        load8<T>(base + static_cast<int64_t>(n) * stride_n, t0);
        load8<T>(base + static_cast<int64_t>(n + Threads) * stride_n, t1);
        load8<T>(base + static_cast<int64_t>(n + 2 * Threads) * stride_n, t2);
        load8<T>(base + static_cast<int64_t>(n + 3 * Threads) * stride_n, t3);
#pragma unroll
        for (int i = 0; i < kDTile; ++i) {
            mx[i] = fmaxf(mx[i], fmaxf(fmaxf(fabsf(t0[i]), fabsf(t1[i])),
                                       fmaxf(fabsf(t2[i]), fabsf(t3[i]))));
        }
    }
    for (; n < N; n += Threads) {
        float t[kDTile];
        load8<T>(base + static_cast<int64_t>(n) * stride_n, t);
#pragma unroll
        for (int i = 0; i < kDTile; ++i) mx[i] = fmaxf(mx[i], fabsf(t[i]));
    }

    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;

#pragma unroll
    for (int i = 0; i < kDTile; ++i) mx[i] = row_reduce_fmax<32>(mx[i]);

    __shared__ float wave_max[kDTile][kWaves];
    __shared__ float inv_scale_shared[kDTile];
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < kDTile; ++i) wave_max[i][wave] = mx[i];
    }
    __syncthreads();

    if (threadIdx.x < kDTile) {
        float val = 0.0f;
#pragma unroll
        for (int w = 0; w < kWaves; ++w) val = fmaxf(val, wave_max[threadIdx.x][w]);
        const float sc = fmaxf(val * (1.0f / kInt8Max), 1e-12f);
        scale_out[(static_cast<int64_t>(b) * H + h) * D + d0 + threadIdx.x] = sc;
        inv_scale_shared[threadIdx.x] = 1.0f / sc;
    }
    __syncthreads();

    float inv_scale[kDTile];
#pragma unroll
    for (int i = 0; i < kDTile; ++i) inv_scale[i] = inv_scale_shared[i];

    const int64_t out_row = (static_cast<int64_t>(b) * H + h) * D + d0;

    // Descending so the tail of pass 1 is still in L2 when pass 2 starts.
    for (int src = N - 1 - static_cast<int>(threadIdx.x); src >= 0; src -= Threads) {
        float t[kDTile];
        load8<T>(base + static_cast<int64_t>(src) * stride_n, t);
#pragma unroll
        for (int i = 0; i < kDTile; ++i) {
            out[(out_row + i) * padded_N + src] = float_to_int8_rn(t[i] * inv_scale[i]);
        }
    }

    // The attention kernel reads whole CTA_K tiles, so the padding has to be a
    // real zero rather than whatever the allocator left behind.
    for (int src = N + static_cast<int>(threadIdx.x); src < padded_N; src += Threads) {
#pragma unroll
        for (int i = 0; i < kDTile; ++i) out[(out_row + i) * padded_N + src] = 0;
    }
}

template <typename T>
void launch_typed(const void* v, void* out, void* scale, int B, int H, int N, int D, int padded_N,
                  int64_t stride_b, int64_t stride_h, int64_t stride_n, hipStream_t stream) {
    const int blocks = B * H * (D / kDTile);
    if (N <= 256) {
        quant_v_int8_kernel<T, 128><<<blocks, 128, 0, stream>>>(
            static_cast<const T*>(v), static_cast<int8_t*>(out), static_cast<float*>(scale), N,
            padded_N, H, D, stride_b, stride_h, stride_n);
    } else {
        quant_v_int8_kernel<T, 512><<<blocks, 512, 0, stream>>>(
            static_cast<const T*>(v), static_cast<int8_t*>(out), static_cast<float*>(scale), N,
            padded_N, H, D, stride_b, stride_h, stride_n);
    }
}

}  // namespace
}  // namespace comfy::hip_backend::sage

using namespace comfy::hip_backend::sage;

extern "C" void launch_gfx12_tile_channel_v_quant(
    const void* v, void* output, void* descale, int num_heads,
    int n_ctx, int tiles,
    int64_t stride_h, int64_t stride_n, int input_dtype_code,
    hipStream_t stream) {
    if (num_heads <= 0 || n_ctx <= 0) {
        throw std::runtime_error(
            "gfx12 tile-channel V quantization requires positive heads and length");
    }
    if (tiles != (n_ctx + 63) / 64) {
        throw std::runtime_error(
            "gfx12 tile-channel V quantization received an invalid tile count");
    }
    const auto stride_ok = [](int64_t stride, int extent) {
        return extent < 2 ||
            (stride > 0 && (static_cast<size_t>(stride) * sizeof(__half)) % 16 == 0);
    };
    if (reinterpret_cast<uintptr_t>(v) % 16 != 0 ||
        !stride_ok(stride_h, num_heads) || !stride_ok(stride_n, n_ctx)) {
        throw std::runtime_error(
            "gfx12 V base pointer and H/N strides must be 16-byte aligned");
    }
    if (input_dtype_code == 1) {
        launch_gfx12_tile_channel_v_quant(
            static_cast<const __half*>(v), static_cast<int8_t*>(output),
            static_cast<float*>(descale), num_heads, n_ctx, tiles,
            stride_h, stride_n, stream);
    } else if (input_dtype_code == 2) {
        launch_gfx12_tile_channel_v_quant(
            static_cast<const __bf16*>(v), static_cast<int8_t*>(output),
            static_cast<float*>(descale), num_heads, n_ctx, tiles,
            stride_h, stride_n, stream);
    } else {
        throw std::runtime_error(
            "gfx12 tile-channel V quantization supports float16 and bfloat16");
    }
}

extern "C" void launch_sage_quant_v_int8(const void* v, void* out, void* scale, int B, int H,
                                         int N, int D, int padded_N, int64_t stride_b,
                                         int64_t stride_h, int64_t stride_n, int input_dtype_code,
                                         hipStream_t stream) {
    // The kernel loads V 16 bytes at a time, so the base pointer and every row
    // offset it can form have to be 16-byte aligned. An extent of one forms no
    // offset, and PyTorch rewrites such a stride to 1 on the way through DLPack
    // (ATen/DLConvertor.cpp, gh-83069), so policing it would reject a tensor the
    // caller has no way to hand over differently.
    const size_t element_size = input_dtype_code == 0 ? sizeof(float) : sizeof(__half);
    const auto stride_ok = [element_size](int64_t stride, int extent) {
        return extent < 2 || (static_cast<size_t>(stride) * element_size) % 16 == 0;
    };
    if (reinterpret_cast<uintptr_t>(v) % 16 != 0 || !stride_ok(stride_b, B) ||
        !stride_ok(stride_h, H) || !stride_ok(stride_n, N)) {
        throw std::runtime_error(
            "sage int8 attention: V base pointer and B/H/N strides must be 16-byte aligned");
    }
    if (D <= 0 || D % kDTile != 0) {
        throw std::runtime_error(
            "sage int8 attention: head_dim must be a positive multiple of 8");
    }
    // Rows are padded_N wide and the quantize pass writes columns up to N, so a
    // short pad would run every row past its end.
    if (N < 0 || padded_N < N) {
        throw std::runtime_error(
            "sage int8 attention: padded key length must be at least the key length");
    }

    switch (input_dtype_code) {
        case 0:
            launch_typed<float>(v, out, scale, B, H, N, D, padded_N, stride_b, stride_h, stride_n,
                                stream);
            break;
        case 1:
            launch_typed<__half>(v, out, scale, B, H, N, D, padded_N, stride_b, stride_h, stride_n,
                                 stream);
            break;
        case 2:
            launch_typed<__bf16>(v, out, scale, B, H, N, D, padded_N, stride_b, stride_h, stride_n,
                                 stream);
            break;
        default:
            throw std::runtime_error(
                "sage int8 attention: input dtype must be fp32, fp16 or bf16");
    }

    const hipError_t error = hipGetLastError();
    if (error != hipSuccess) {
        throw std::runtime_error(std::string("sage int8 V quantize kernel launch failed: ") +
                                 hipGetErrorString(error));
    }
}
