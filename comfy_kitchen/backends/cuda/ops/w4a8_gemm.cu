// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// W4A8 weight dequant: grouped int4 -> int8 for the tuned int8-GEMM path.
//
// The AsymW4A8Int8Layout dequantizes int4 weights to the "grouped int8"
// representation (per-group scale folded in, per-channel scale left for the int8
// GEMM epilogue) and runs comfy's tuned int8 CUTLASS GEMM -- so this file only
// needs the memory-bound int4->int8 dequant kernel (fp32 and fp8/e4m3 group
// scales, optional Lloyd-Max codebook). The matmul itself is cutlass_gemm_int8.

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdint>

// Grouped int4 -> int8 dequant for the int8-GEMM W4A8 path: out[n,k] =
// round((q_u[n,k]-8) * s_rel[n, k/G]), q_u packed uint4 (even col=low nibble).
// s_rel = per-group scale / per-channel scale (so the int8 range is used). The
// per-channel scale is applied later in the int8 GEMM epilogue. Memory-bound.
namespace {
// Per-group scale may be stored fp32 or fp8 (e4m3). fp8 halves the scale
// metadata (g16 fp32 -> 0.75 B/elem; fp8 -> ~0.56, ~half int8) at a tiny
// quality cost (still beats NVFP4). uint8_t storage == e4m3 raw bits.
template <typename ScaleT> __device__ __forceinline__ float load_scale(ScaleT v);
template <> __device__ __forceinline__ float load_scale<float>(float v) { return v; }
template <> __device__ __forceinline__ float load_scale<uint8_t>(uint8_t v) {
    return __half2float(__nv_cvt_fp8_to_halfraw(v, __NV_E4M3));
}

// Each thread: 8 packed bytes (uint2) -> 16 int8 (uint4 store). The 16 output
// cols may span multiple groups when G<16 (finer groups = better int4 quality),
// so the scale is (re)loaded per output pair from its own group. G must be even
// and either divide 16 or be a multiple of 16 (so groups stay pair-aligned).
// If codebook != nullptr, the 4-bit code indexes a shared 16-entry non-uniform
// codebook (Lloyd-Max on the rotated-Gaussian weight) instead of the uniform
// level (q-8); same storage/speed, ~14% lower weight error at coarse groups.
template <typename ScaleT>
__global__ void dequant_int4_grouped_to_int8_kernel(
    const int8_t* __restrict__ qw,   // (N, K/2) packed uint4
    const ScaleT* __restrict__ s_rel,// (N, K/G) fp32 or e4m3 raw
    const float*  __restrict__ codebook, // 16 floats or nullptr
    int8_t*       __restrict__ out,  // (N, K)
    long n_vec, int Khalf, int K, int G)
{
    __shared__ float cb[16];
    if (codebook && threadIdx.x < 16) cb[threadIdx.x] = codebook[threadIdx.x];
    if (codebook) __syncthreads();
    long v = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_vec) return;                       // n_vec = N*Khalf/8
    const int vec_per_row = Khalf / 8;
    const int n = v / vec_per_row;
    const int hv = v % vec_per_row;               // which uint2 in the row
    const int kh = hv * 8;                        // packed byte offset
    const int k0 = kh * 2;                         // output col base (16 wide)
    const int nG = K / G;
    const long srow = (long)n * nG;
    const uint2 pk = *reinterpret_cast<const uint2*>(&qw[(long)n * Khalf + kh]);
    const unsigned words[2] = {pk.x, pk.y};
    char4 o4[4];
    #pragma unroll
    for (int w = 0; w < 2; ++w) {
        const unsigned bb = words[w];
        #pragma unroll
        for (int bi = 0; bi < 4; ++bi) {
            const int oo = w * 4 + bi;             // 0..7 -> cols oo*2, oo*2+1
            // q0/q1 are consecutive cols (same group since G>=2 even).
            const float s = load_scale<ScaleT>(s_rel[srow + (k0 + oo * 2) / G]);
            const unsigned byte = (bb >> (bi * 8)) & 0xFF;
            const unsigned c0 = byte & 0xF, c1 = (byte >> 4) & 0xF;
            const float v0 = codebook ? cb[c0] : (static_cast<float>(c0) - 8.0f);
            const float v1 = codebook ? cb[c1] : (static_cast<float>(c1) - 8.0f);
            reinterpret_cast<int8_t*>(&o4[oo / 2])[(oo % 2) * 2]     =
                static_cast<int8_t>(max(-127, min(127, __float2int_rn(v0 * s))));
            reinterpret_cast<int8_t*>(&o4[oo / 2])[(oo % 2) * 2 + 1] =
                static_cast<int8_t>(max(-127, min(127, __float2int_rn(v1 * s))));
        }
    }
    *reinterpret_cast<uint4*>(&out[(long)n * K + k0]) = *reinterpret_cast<uint4*>(o4);
}
}  // namespace

// codebook: 16 floats (non-uniform levels) or nullptr for uniform (q-8).
extern "C" void launch_dequant_int4_grouped_to_int8(
    const void* qw, const void* s_rel, const void* codebook, void* out,
    int64_t N, int64_t K, int64_t G, cudaStream_t stream)
{
    const int Khalf = K / 2;
    const long n_vec = (long)N * Khalf / 8;
    const int block = 256;
    const long grid = (n_vec + block - 1) / block;
    dequant_int4_grouped_to_int8_kernel<float><<<grid, block, 0, stream>>>(
        static_cast<const int8_t*>(qw), static_cast<const float*>(s_rel),
        static_cast<const float*>(codebook),
        static_cast<int8_t*>(out), n_vec, Khalf, static_cast<int>(K), static_cast<int>(G));
}

// fp8 (e4m3) per-group scale variant; s_rel passed as raw uint8 bits.
extern "C" void launch_dequant_int4_grouped_to_int8_e4m3(
    const void* qw, const void* s_rel, const void* codebook, void* out,
    int64_t N, int64_t K, int64_t G, cudaStream_t stream)
{
    const int Khalf = K / 2;
    const long n_vec = (long)N * Khalf / 8;
    const int block = 256;
    const long grid = (n_vec + block - 1) / block;
    dequant_int4_grouped_to_int8_kernel<uint8_t><<<grid, block, 0, stream>>>(
        static_cast<const int8_t*>(qw), static_cast<const uint8_t*>(s_rel),
        static_cast<const float*>(codebook),
        static_cast<int8_t*>(out), n_vec, Khalf, static_cast<int>(K), static_cast<int>(G));
}

