// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// WMMA intrinsics and fragment addressing for RDNA4 (gfx12), wave32.
//
// Fragment layout for the w32 gfx12 WMMA instructions:
//   A (M x K, row-major): lane l, element e -> A[l % 16][kbase + e]
//   B (N x K, row-major): lane l, element e -> B[l % 16][kbase + e]
//   D (M x N):            lane l, element e -> D[e + 8 * (l / 16)][l % 16]
// with kbase = (K_STEP / 2) * (l / 16).
//
// K_STEP is 16 for fp8/iu8/bf16 (8 elements per lane) and 32 for iu4 (16
// nibbles per lane). fp8, iu8 and iu4 all hold 8 bytes per lane, so their
// operand fragments are a v2i regardless of element type, which is what lets one
// byte-addressed tile kernel serve all three. bf16 is the exception: 8 elements
// of 2 bytes is a 16-byte v8bf fragment, and only wmma_bf16 below consumes it.
#pragma once

#include <hip/hip_runtime.h>

namespace comfy::hip_backend {

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef __bf16 v8bf __attribute__((ext_vector_type(8)));

constexpr int kWave = 32;

// Operand bytes for one lane of a 16x16xK tile whose rows are `stride` bytes
// apart in LDS. `row` is the lane's row (A) or column (B); `kbyte` is the byte
// offset of the K-step within the row.
__forceinline__ __device__ v2i load_frag_b64(const void* lds, int row, int kbyte, int stride) {
    const char* p = static_cast<const char*>(lds) + row * stride + kbyte;
    return *reinterpret_cast<const v2i*>(p);
}

__forceinline__ __device__ int frag_row(int lane) { return lane % 16; }
__forceinline__ __device__ int frag_khalf(int lane) { return lane / 16; }

// Row of accumulator element `e` for this lane; the column is lane % 16.
__forceinline__ __device__ int acc_row(int lane, int e) { return e + 8 * (lane / 16); }
__forceinline__ __device__ int acc_col(int lane) { return lane % 16; }

__forceinline__ __device__ v8f wmma_fp8(v2i a, v2i b, v8f acc) {
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, acc);
}

__forceinline__ __device__ v8f wmma_bf8(v2i a, v2i b, v8f acc) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(a, b, acc);
}

__forceinline__ __device__ v8i wmma_iu8(v2i a, v2i b, v8i acc) {
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, acc, false);
}

// Operand signedness is an immediate operand of the instruction, so an unsigned
// A operand requires a separate entry point rather than a runtime flag.
__forceinline__ __device__ v8i wmma_iu8_ua(v2i a, v2i b, v8i acc) {
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a, true, b, acc, false);
}

__forceinline__ __device__ v8i wmma_iu4(v2i a, v2i b, v8i acc) {
    return __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, a, true, b, acc, false);
}

__forceinline__ __device__ v8i wmma_iu4_ua(v2i a, v2i b, v8i acc) {
    return __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false, a, true, b, acc, false);
}

__forceinline__ __device__ v8f wmma_bf16(v8bf a, v8bf b, v8f acc) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, acc);
}

}  // namespace comfy::hip_backend
