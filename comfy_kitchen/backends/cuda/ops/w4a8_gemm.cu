// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// W4A8 weight dequant: grouped int4 -> int8 for the tuned int8-GEMM path.
//
// AsymW4A8Int8Layout dequantizes int4 weights to "grouped int8" (per-group scale
// folded in, per-channel scale left for the int8 GEMM epilogue), then runs comfy's
// tuned int8 CUTLASS GEMM. So this file is just the memory-bound int4->int8 dequant
// kernel (fp32/fp8-e4m3 group scales, optional codebook); the matmul is cutlass_gemm_int8.

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdint>

// Grouped int4 -> int8 dequant for the int8-GEMM W4A8 path: out[n,k] =
// round((q_u[n,k]-8) * s_rel[n, k/G]), q_u packed uint4 (even col=low nibble).
// s_rel = per-group scale / per-channel scale (so the int8 range is used). The
// per-channel scale is applied later in the int8 GEMM epilogue. Memory-bound.
namespace {
// Per-group scale is fp32 or fp8 (e4m3). fp8 halves the scale metadata at a tiny
// quality cost. uint8_t storage == e4m3 raw bits.
template <typename ScaleT> __device__ __forceinline__ float load_scale(ScaleT v);
template <> __device__ __forceinline__ float load_scale<float>(float v) { return v; }
template <> __device__ __forceinline__ float load_scale<uint8_t>(uint8_t v) {
    return __half2float(__nv_cvt_fp8_to_halfraw(v, __NV_E4M3));
}

// Each thread: 8 packed bytes (uint2) -> 16 int8 (uint4 store). The 16 output
// cols may span multiple groups when G<16 (finer groups = better int4 quality),
// so the scale is (re)loaded per output pair from its own group. Only 4 group
// scales (sc0..sc3) are loaded, so a 16-col vec may span at most 4 groups: G must
// be in {4, 8, 16} or a multiple of 16 (G<4 would span >4 groups and mis-scale).
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
    // The 16-col vec spans 1 group (G>=16, the common case), 2 (G=8), or 4 (G=4).
    // Load+decode each distinct group scale ONCE instead of per output pair.
    const int base_g = k0 / G;
    float sc0 = load_scale<ScaleT>(s_rel[srow + base_g]);
    float sc1 = sc0, sc2 = sc0, sc3 = sc0;
    if (G < 16) {
        sc1 = load_scale<ScaleT>(s_rel[srow + base_g + 1]);
        if (G < 8) {  // G == 4
            sc2 = load_scale<ScaleT>(s_rel[srow + base_g + 2]);
            sc3 = load_scale<ScaleT>(s_rel[srow + base_g + 3]);
        }
    }
    char4 o4[4];
    #pragma unroll
    for (int w = 0; w < 2; ++w) {
        const unsigned bb = words[w];
        #pragma unroll
        for (int bi = 0; bi < 4; ++bi) {
            const int oo = w * 4 + bi;             // 0..7 -> cols oo*2, oo*2+1
            const int lg = (G >= 16) ? 0 : ((oo * 2) / G);  // local group in the vec
            const float s = (lg == 0) ? sc0 : (lg == 1 ? sc1 : (lg == 2 ? sc2 : sc3));
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

// Fused-quality W4A8: dequant int4 -> int8 in column chunks (codebook + per-group
// s_rel) feeding the tuned STRIDED int8 GEMM, so each int8 weight chunk stays
// L2-resident instead of the full [N,K] round-tripping global (the convrot_w4a4
// chunking trick, run at our group-16 codebook quality). Returns false if the
// strided GEMM rejects a chunk config -> caller falls back to the 2-pass path.
extern "C" bool launch_cutlass_int8_dequant_strided(
    const void* A, const void* B, const void* xs, const void* ws, const void* bias,
    void* D, int64_t M, int64_t N, int64_t K, int64_t output_stride, int out_dtype_code,
    cudaStream_t stream);

extern "C" bool launch_w4a8_codebook_gemm_chunked(
    const void* xq,        // [M, K] int8 activation
    const void* weight,    // [N, K/2] packed uint4
    const void* s_rel,     // [N, K/G] fp8 (e4m3) per-group scale
    const void* codebook,  // [16] fp32 or nullptr
    const void* s_channel, // [N] fp32 per-channel scale
    const void* xs,        // [M] fp32 per-row activation scale
    const void* bias,      // [N] fp32 or nullptr
    void* workspace,       // [chunk_cols, K] int8 scratch (preallocated, reused)
    void* out,             // [M, N] output (out_dtype)
    int64_t M, int64_t N, int64_t K, int64_t G, int64_t chunk_cols,
    int out_dtype_code, cudaStream_t stream)
{
    // A non-positive chunk stride never advances n0 -> would loop forever; a non-positive
    // K/G would divide by zero below. Bail so the caller uses the 2-pass path.
    if (chunk_cols <= 0 || K <= 0 || G <= 0) return false;
    const int64_t Khalf = K / 2, KG = K / G, osz = (out_dtype_code == 0) ? 4 : 2;
    for (int64_t n0 = 0; n0 < N; n0 += chunk_cols) {
        const int64_t cols = (chunk_cols < N - n0) ? chunk_cols : (N - n0);
        launch_dequant_int4_grouped_to_int8_e4m3(
            static_cast<const int8_t*>(weight) + n0 * Khalf,
            static_cast<const uint8_t*>(s_rel) + n0 * KG,
            codebook, workspace, cols, K, G, stream);
        const void* bias_chunk = bias ? static_cast<const float*>(bias) + n0 : nullptr;
        void* out_chunk = static_cast<char*>(out) + n0 * osz;
        if (!launch_cutlass_int8_dequant_strided(
                xq, workspace, xs, static_cast<const float*>(s_channel) + n0, bias_chunk,
                out_chunk, M, cols, K, N /*output_stride*/, out_dtype_code, stream))
            return false;
    }
    return true;
}

