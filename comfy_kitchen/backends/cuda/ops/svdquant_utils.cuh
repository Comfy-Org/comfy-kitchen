// SPDX-License-Identifier: Apache-2.0
//
// Shared device helpers for SVDQuant W4A4 kernels.
//
// All code gated on __CUDA_ARCH__ >= 800. On older arches (sm_75) the
// callers compile out the kernel body and installation is still valid;
// the op simply raises at runtime.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

namespace comfy::svdquant {

// Kitchen's group size for W4A4 int4 quantization.
constexpr int kGroupSize = 64;
// Symmetric int4 range [-7, 7] (we skip -8 to match nunchaku's absmax/7 scheme).
constexpr int kInt4Max = 7;
// Default padding multiple for M (activation batch).
constexpr int kMPad = 256;

// ---------------------------------------------------------------------------
// int4 packing: two signed int4s per byte, row-major.
//   byte & 0x0F  = q[n, 2k]    (low nibble,  sign-extended from 4 bits)
//   byte >> 4    = q[n, 2k+1]  (high nibble, sign-extended from 4 bits)
// ---------------------------------------------------------------------------
__forceinline__ __device__ int8_t pack_int4_pair(int lo, int hi) {
    uint32_t packed = (static_cast<uint32_t>(lo) & 0x0F) |
                      ((static_cast<uint32_t>(hi) & 0x0F) << 4);
    return static_cast<int8_t>(packed);
}

__forceinline__ __device__ void unpack_int4_pair(int8_t packed, int &lo, int &hi) {
    int lo4 = packed & 0x0F;
    int hi4 = (packed >> 4) & 0x0F;
    lo = (lo4 >= 8) ? lo4 - 16 : lo4;
    hi = (hi4 >= 8) ? hi4 - 16 : hi4;
}

// ---------------------------------------------------------------------------
// Warp-level reductions. Uses full-warp mask 0xffffffff.
// ---------------------------------------------------------------------------

// absmax across the lower `width` lanes (power of 2).
__forceinline__ __device__ float warp_absmax(float v, int width = 32) {
    v = fabsf(v);
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffffu, v, offset);
        v = fmaxf(v, other);
    }
    return v;
}

// sum across the lower `width` lanes.
__forceinline__ __device__ float warp_sum(float v, int width = 32) {
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

// ---------------------------------------------------------------------------
// Arch guard sentinel for kernel bodies.
// Caller pattern:
//   #if __CUDA_ARCH__ >= 800
//     ... real kernel body ...
//   #else
//     comfy::svdquant::trap_pre_sm80();
//   #endif
// ---------------------------------------------------------------------------
__forceinline__ __device__ void trap_pre_sm80() {
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        printf("[svdquant_w4a4] int4 MMA requires sm_80 or newer.\n");
    }
    __trap();
}

// ---------------------------------------------------------------------------
// mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32
//
// Per-warp fragment shapes:
//   A : 16 rows × 64 cols int4 → 4 × uint32 per lane
//       lane i at (groupID=i/4, tid=i%4):
//         a[0]: row=groupID,   cols=tid*16 + [0..7]   (8 int4 in one uint32)
//         a[1]: row=groupID+8, cols=tid*16 + [0..7]
//         a[2]: row=groupID,   cols=tid*16 + [8..15]
//         a[3]: row=groupID+8, cols=tid*16 + [8..15]
//   B : 64 rows × 8 cols int4 col-major → 2 × uint32 per lane
//       b[0]: n=tid,    k=(groupID*8) + [0..7]
//       b[1]: n=tid+4,  k=(groupID*8) + [0..7]
//   D : 16 rows × 8 cols int32 → 4 × int32 per lane
//       d[0]: row=groupID,   col=tid*2 + 0
//       d[1]: row=groupID,   col=tid*2 + 1
//       d[2]: row=groupID+8, col=tid*2 + 0
//       d[3]: row=groupID+8, col=tid*2 + 1
//
// sm_80+ gates the inline asm at compile time; callers on older archs must
// not dispatch to kernels that use this.
// ---------------------------------------------------------------------------
__forceinline__ __device__ void mma_m16n8k64_s4s4s32(
    const uint32_t a[4], const uint32_t b[2], const int32_t c[4], int32_t d[4])
{
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
#else
    (void)a; (void)b; (void)c;
    d[0] = d[1] = d[2] = d[3] = 0;
#endif
}

// Unsigned-activation variant: A fragment treated as u4 [0,15], B as s4 [-8,7].
// Used for layers with act_unsigned=True (e.g., post-GELU fc2 inputs shifted to
// be non-negative). Bit patterns of A are interpreted as unsigned instead of
// signed, doubling the effective quantization range ([0,15] vs [-7,7]).
__forceinline__ __device__ void mma_m16n8k64_u4s4s32(
    const uint32_t a[4], const uint32_t b[2], const int32_t c[4], int32_t d[4])
{
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
#else
    (void)a; (void)b; (void)c;
    d[0] = d[1] = d[2] = d[3] = 0;
#endif
}

// ---------------------------------------------------------------------------
// bf16/fp16 tensor-core helpers for LoRA-up epilogues:
//   mma.sync.aligned.m16n8k16.row.col.f32.{bf16|f16}.{bf16|f16}.f32
//
// Shared-memory fragments follow the same lane mapping as the int4 output
// fragment used by mma_m16n8k64_* above:
//   d[0]: row=lane/4,   col=(lane%4)*2
//   d[1]: row=lane/4,   col=(lane%4)*2+1
//   d[2]: row=lane/4+8, col=(lane%4)*2
//   d[3]: row=lane/4+8, col=(lane%4)*2+1
// ---------------------------------------------------------------------------
__forceinline__ __device__ uint32_t cvta_smem_u32(const void* ptr) {
    uint32_t s;
#if __CUDA_ARCH__ >= 800
    asm("{ .reg .u64 ll; cvta.to.shared.u64 ll, %1; cvt.u32.u64 %0, ll; }"
        : "=r"(s) : "l"(ptr));
#else
    s = 0;
    (void)ptr;
#endif
    return s;
}

__forceinline__ __device__ void ldmatrix_x2(uint32_t (&dst)[2], uint32_t addr) {
#if __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
                 "{%0, %1}, [%2];\n"
                 : "=r"(dst[0]), "=r"(dst[1])
                 : "r"(addr));
#else
    dst[0] = dst[1] = 0;
    (void)addr;
#endif
}

__forceinline__ __device__ void ldmatrix_x4(uint32_t (&dst)[4], uint32_t addr) {
#if __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                 "{%0, %1, %2, %3}, [%4];\n"
                 : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
                 : "r"(addr));
#else
    dst[0] = dst[1] = dst[2] = dst[3] = 0;
    (void)addr;
#endif
}

template<typename T>
__device__ __forceinline__ void mma_m16n8k16_f32(
    float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]);

template<>
__device__ __forceinline__ void mma_m16n8k16_f32<__nv_bfloat16>(
    float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2])
{
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#else
    (void)a; (void)b;
#endif
}

template<>
__device__ __forceinline__ void mma_m16n8k16_f32<__half>(
    float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2])
{
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#else
    (void)a; (void)b;
#endif
}

// ---------------------------------------------------------------------------
// cp.async.cg.shared.global — 16-byte async copy from GMEM to SMEM.
// "cg" = "cache-global": bypass L1, go through L2 for smaller footprint.
// ---------------------------------------------------------------------------
__forceinline__ __device__ void cp_async_16b(void* smem_ptr, const void* gmem_ptr) {
#if __CUDA_ARCH__ >= 800
    uint32_t smem_int = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        : : "r"(smem_int), "l"(gmem_ptr));
#else
    *reinterpret_cast<uint4*>(smem_ptr) = *reinterpret_cast<const uint4*>(gmem_ptr);
#endif
}

__forceinline__ __device__ void cp_async_commit_group() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template<int N>
__forceinline__ __device__ void cp_async_wait_group() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

} // namespace comfy::svdquant
