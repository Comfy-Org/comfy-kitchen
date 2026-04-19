// SPDX-License-Identifier: Apache-2.0
//
// Kitchen CUDA SVDQuant W4A4 int4 GEMM with per-group dequant.
//
// Tile layout:
//   CTA  = 4 warps × 32 threads = 128 threads, warp grid 2×2
//   CTA covers 32 M × 128 N output (each warp computes 16 M × 64 N)
//   grid = (ceil(N/128), ceil(M/32))
//
// Per K iteration (BLOCK_K = kGroupSize = 64):
//   A (32 M × 32 B/group) and B (128 N × 32 B/group) are CTA-cooperatively
//   loaded into shmem via cp.async, triple-buffered (kStages=3).
//   Each warp issues kNUnroll = 8 MMAs covering its 16M × 64N output tile.
//
// Dequant: int32 MMA output → fp32 scalar multiply-adds with per-group
// ascale × wscale. fp32 accumulator (not fp16) for numerical robustness —
// in production Qwen-Image-Edit, scale products (ascale × wscale) can reach
// ~100 and per-MMA d_reg values can reach ~3000, so the per-term product
// overflows fp16's ±65504 range mid-sampling and silently propagates NaN
// (see ops/scaled_mm_svdquant_w4a4.cu accumulator declaration).
//
// act_unsigned: when true, A fragments are interpreted by u4.s4 MMA instead
// of s4.s4 (enables +1 bit of activation precision for layers whose input
// is known non-negative, e.g., post-GELU fc2 with nunchaku's +0.171875 shift
// — caller applies the shift at the layer level; this kernel only picks the
// MMA variant).
//
// LoRA-up + bias are applied externally in the Python wrapper via cuBLAS bf16
// matmul + addmm_ (faster than in-kernel fusion for small R={16..256}; see
// comfy_kitchen/backends/cuda/__init__.py::scaled_mm_svdquant_w4a4).
#include "svdquant_utils.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {

using comfy::svdquant::kGroupSize;
using comfy::svdquant::mma_m16n8k64_s4s4s32;
using comfy::svdquant::mma_m16n8k64_u4s4s32;
using comfy::svdquant::cp_async_16b;
using comfy::svdquant::cp_async_commit_group;
using comfy::svdquant::cp_async_wait_group;

constexpr int kStages = 3;  // opt10: 3-stage pipeline (smem 10→15 KB, same occupancy)

constexpr int kMUnroll  = 1;               // MMA_M tiles per warp (each 16 M)
constexpr int kWarpM    = kMUnroll * 16;   // 16 M rows per warp
constexpr int kNUnroll  = 8;               // MMAs per K iter per warp (N dim)
constexpr int kWarpN    = kNUnroll * 8;    // 64 N cols per warp
constexpr int kWarpsM   = 2;               // warps stacked in M
constexpr int kWarpsN   = 2;               // warps stacked in N
constexpr int kNumWarps = kWarpsM * kWarpsN;      // 8
constexpr int kBlockM   = kWarpM * kWarpsM;       // 64 M per CTA
constexpr int kBlockN   = kWarpN * kWarpsN;       // 128 N per CTA
constexpr int kBlockKBytes = kGroupSize / 2;      // 32 bytes of int4 per K group

template<typename OutType>
__device__ __forceinline__ OutType fp32_to_out(float v);

template<>
__device__ __forceinline__ __nv_bfloat16 fp32_to_out<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template<>
__device__ __forceinline__ __half fp32_to_out<__half>(float v) {
    return __float2half(v);
}

template<typename OutType>
__device__ __forceinline__ float load_scale(const OutType* p) {
    if constexpr (std::is_same_v<OutType, __nv_bfloat16>) return __bfloat162float(*p);
    else return __half2float(*p);
}

template<typename OutType, bool kActUnsigned>
__global__ void svdquant_scaled_mm_w4a4_kernel(
    const int8_t* __restrict__ act,          // (M, K/2)
    const int8_t* __restrict__ wgt,          // (N, K/2)
    const OutType* __restrict__ ascales,     // (K/G, M)
    const OutType* __restrict__ wscales,     // (K/G, N)
    const float*   __restrict__ lora_act_in, // unused; epilogue is in Python
    const OutType* __restrict__ lora_up,     // unused
    const OutType* __restrict__ bias,        // unused
    OutType* __restrict__ out,               // (M, N)
    int M, int N, int K, int R)
{
    (void)lora_act_in; (void)lora_up; (void)bias; (void)R;

    // CTA coordinates
    const int cta_m = blockIdx.y * kBlockM;
    const int cta_n = blockIdx.x * kBlockN;

    // Warp layout within CTA
    const int warp_id   = threadIdx.x >> 5;          // 0..kNumWarps-1
    const int lane      = threadIdx.x & 31;
    const int warp_m    = warp_id & (kWarpsM - 1);   // 0..kWarpsM-1
    const int warp_n    = warp_id / kWarpsM;         // 0..kWarpsN-1

    const int groupID      = lane >> 2;   // 0..7
    const int tid_in_group = lane & 3;    // 0..3

    // This warp's output M/N base
    const int warp_m_base = cta_m + warp_m * kWarpM;
    const int warp_n_base = cta_n + warp_n * kWarpN;

    // Per-warp accumulator: kMUnroll M-tiles × kNUnroll N-chunks × 4 fp32.
    // We accumulate in fp32 (not fp16) because in production Qwen-Image-Edit,
    // per-group scale products ascale*wscale can reach ~100 and single-MMA d_reg
    // values can reach ~3000, pushing the per-term contribution well past fp16's
    // ±65504 range. fp16 accumulation overflow -> inf -> NaN propagation causes
    // black-image mid-sampling failures that single-layer random parity tests miss.
    // nunchaku works with fp16 because their calibration bounds scales tighter;
    // kitchen's conservative choice is fp32 for end-to-end robustness.
    float out_f[kMUnroll][kNUnroll][4];
    #pragma unroll
    for (int mi = 0; mi < kMUnroll; ++mi) {
        #pragma unroll
        for (int c = 0; c < kNUnroll; ++c) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) out_f[mi][c][i] = 0.f;
        }
    }

    const int K_half      = K / 2;
    const int num_groups  = K / kGroupSize;

    // Shared memory: double-buffered A and B tiles.
    // B stage: kBlockN rows × kBlockKBytes bytes = 128 * 32 = 4 KB
    // A stage: kBlockM rows × kBlockKBytes bytes =  32 * 32 = 1 KB
    __shared__ alignas(16) int8_t smem_B[kStages][kBlockN * kBlockKBytes];
    __shared__ alignas(16) int8_t smem_A[kStages][kBlockM * kBlockKBytes];

    // ---------- Helper: issue async cp for B tile at group `g` into stage ----------
    auto issue_B_load = [&](int g, int stage) {
        if (g >= num_groups) return;
        // 128 threads × (kBlockN/64 sweeps) × 16B per thread = 4 KB per stage
        // We need kBlockN * kBlockKBytes = 4096 bytes. 128 threads × 16 B = 2 KB per sweep → 2 sweeps.
        const int thread_idx = threadIdx.x;
        #pragma unroll
        for (int sweep = 0; sweep < 2; ++sweep) {
            const int t = thread_idx + sweep * (kNumWarps * 32);
            // each t loads one 16-byte chunk: n_row = t/2, half = t%2
            if (t < kBlockN * 2) {
                const int n_row = t >> 1;
                const int half  = t & 1;
                const int n_global = cta_n + n_row;
                int8_t* dst = &smem_B[stage][n_row * kBlockKBytes + half * 16];
                if (n_global < N) {
                    const int8_t* src = wgt + n_global * K_half + g * kBlockKBytes + half * 16;
                    cp_async_16b(dst, src);
                } else {
                    // Out-of-bounds rows: pad with zeros (use regular 16-byte zero store).
                    reinterpret_cast<uint4*>(dst)[0] = {0, 0, 0, 0};
                }
            }
        }
    };

    // ---------- Helper: issue async cp for A tile at group `g` into stage ----------
    auto issue_A_load = [&](int g, int stage) {
        if (g >= num_groups) return;
        const int t = threadIdx.x;
        if (t < kBlockM * 2) {
            const int m_row = t >> 1;
            const int half  = t & 1;
            const int m_global = cta_m + m_row;
            int8_t* dst = &smem_A[stage][m_row * kBlockKBytes + half * 16];
            if (m_global < M) {
                const int8_t* src = act + m_global * K_half + g * kBlockKBytes + half * 16;
                cp_async_16b(dst, src);
            } else {
                reinterpret_cast<uint4*>(dst)[0] = {0, 0, 0, 0};
            }
        }
    };

    // ---------- Prime the pipeline: launch first (kStages-1) loads ----------
    #pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        issue_A_load(s, s);
        issue_B_load(s, s);
        cp_async_commit_group();
    }

    for (int g = 0; g < num_groups; ++g) {
        // Start next iteration's load (stage = (g + kStages - 1) % kStages)
        const int next_g = g + kStages - 1;
        if (next_g < num_groups) {
            const int next_stage = (g + kStages - 1) % kStages;
            issue_A_load(next_g, next_stage);
            issue_B_load(next_g, next_stage);
        }
        cp_async_commit_group();

        // Wait for the load corresponding to current g (stages ahead of current)
        cp_async_wait_group<kStages - 1>();
        __syncthreads();

        const int cur_stage = g % kStages;

        // ---------- A loads from shmem for each of kMUnroll M-tiles ----------
        uint32_t a_reg[kMUnroll][4];
        float as_row0_arr[kMUnroll], as_row1_arr[kMUnroll];
        #pragma unroll
        for (int mi = 0; mi < kMUnroll; ++mi) {
            const int m_tile_base = warp_m_base + mi * 16;
            const int row0_m = m_tile_base + groupID;
            const int row1_m = m_tile_base + groupID + 8;
            const int row0_local = warp_m * kWarpM + mi * 16 + groupID;
            const int row1_local = warp_m * kWarpM + mi * 16 + groupID + 8;
            a_reg[mi][0] = a_reg[mi][1] = a_reg[mi][2] = a_reg[mi][3] = 0;
            if (row0_m < M) {
                const int8_t* rb = &smem_A[cur_stage][row0_local * kBlockKBytes];
                a_reg[mi][0] = *reinterpret_cast<const uint32_t*>(rb + tid_in_group * 8);
                a_reg[mi][2] = *reinterpret_cast<const uint32_t*>(rb + tid_in_group * 8 + 4);
            }
            if (row1_m < M) {
                const int8_t* rb = &smem_A[cur_stage][row1_local * kBlockKBytes];
                a_reg[mi][1] = *reinterpret_cast<const uint32_t*>(rb + tid_in_group * 8);
                a_reg[mi][3] = *reinterpret_cast<const uint32_t*>(rb + tid_in_group * 8 + 4);
            }
            as_row0_arr[mi] = (row0_m < M) ? load_scale<OutType>(&ascales[g * M + row0_m]) : 0.f;
            as_row1_arr[mi] = (row1_m < M) ? load_scale<OutType>(&ascales[g * M + row1_m]) : 0.f;
        }

        // Pre-load this K-iter's wscales into per-lane registers, hoisted out of
        // the inner MMA loop so the compiler can schedule the loads ahead of MMAs.
        const OutType* ws_base_g = &wscales[g * N];
        float ws_regs[kNUnroll][2];
        #pragma unroll
        for (int cc = 0; cc < kNUnroll; ++cc) {
            const int col0 = warp_n_base + cc * 8 + tid_in_group * 2 + 0;
            const int col1 = warp_n_base + cc * 8 + tid_in_group * 2 + 1;
            ws_regs[cc][0] = (col0 < N) ? load_scale<OutType>(&ws_base_g[col0]) : 0.f;
            ws_regs[cc][1] = (col1 < N) ? load_scale<OutType>(&ws_base_g[col1]) : 0.f;
        }

        #pragma unroll
        for (int c = 0; c < kNUnroll; ++c) {
            const int b_col_local = (warp_n * kWarpN) + c * 8 + groupID;
            const int b_col_global = cta_n + b_col_local;

            uint32_t b_reg[2] = {0, 0};
            if (b_col_local < kBlockN && b_col_global < N) {
                const int8_t* row_base = &smem_B[cur_stage][b_col_local * kBlockKBytes];
                b_reg[0] = *reinterpret_cast<const uint32_t*>(row_base + tid_in_group * 8);
                b_reg[1] = *reinterpret_cast<const uint32_t*>(row_base + tid_in_group * 8 + 4);
            }

            const float ws_col0 = ws_regs[c][0];
            const float ws_col1 = ws_regs[c][1];

            // Reuse b_reg for each M-tile
            #pragma unroll
            for (int mi = 0; mi < kMUnroll; ++mi) {
                int32_t c_reg[4] = {0, 0, 0, 0};
                int32_t d_reg[4];
                if constexpr (kActUnsigned) {
                    mma_m16n8k64_u4s4s32(a_reg[mi], b_reg, c_reg, d_reg);
                } else {
                    mma_m16n8k64_s4s4s32(a_reg[mi], b_reg, c_reg, d_reg);
                }

                // fp32 dequant: out_f += cvt(d) * ascale * wscale. Slower than
                // the fp16 hfma2 variant but avoids the overflow on large
                // scale products (see accumulator declaration for rationale).
                out_f[mi][c][0] += static_cast<float>(d_reg[0]) * as_row0_arr[mi] * ws_col0;
                out_f[mi][c][1] += static_cast<float>(d_reg[1]) * as_row0_arr[mi] * ws_col1;
                out_f[mi][c][2] += static_cast<float>(d_reg[2]) * as_row1_arr[mi] * ws_col0;
                out_f[mi][c][3] += static_cast<float>(d_reg[3]) * as_row1_arr[mi] * ws_col1;
            }
        }
    }
    cp_async_wait_group<0>();  // drain pipeline

    // ---------- Write output ----------
    #pragma unroll
    for (int mi = 0; mi < kMUnroll; ++mi) {
        const int m_tile_base = warp_m_base + mi * 16;
        const int row0_m = m_tile_base + groupID;
        const int row1_m = m_tile_base + groupID + 8;
        #pragma unroll
        for (int c = 0; c < kNUnroll; ++c) {
            const int n_chunk_base = warp_n_base + c * 8;
            const int col0 = n_chunk_base + tid_in_group * 2 + 0;
            const int col1 = n_chunk_base + tid_in_group * 2 + 1;

            if (row0_m < M && col0 < N) out[row0_m * N + col0] = fp32_to_out<OutType>(out_f[mi][c][0]);
            if (row0_m < M && col1 < N) out[row0_m * N + col1] = fp32_to_out<OutType>(out_f[mi][c][1]);
            if (row1_m < M && col0 < N) out[row1_m * N + col0] = fp32_to_out<OutType>(out_f[mi][c][2]);
            if (row1_m < M && col1 < N) out[row1_m * N + col1] = fp32_to_out<OutType>(out_f[mi][c][3]);
        }
    }
}

} // anonymous namespace

extern "C" {

void launch_svdquant_scaled_mm_w4a4_kernel(
    const void* act,
    const void* wgt,
    const void* ascales,
    const void* wscales,
    const void* lora_act_in,
    const void* lora_up,
    const void* bias,
    void* out,
    int M,
    int N,
    int K,
    int R,
    int act_unsigned,
    int out_dtype_code,
    cudaStream_t stream)
{
    if (K % comfy::svdquant::kGroupSize != 0) return;

    const dim3 grid((N + kBlockN - 1) / kBlockN, (M + kBlockM - 1) / kBlockM);
    const dim3 block(kNumWarps * 32);  // 128 threads

    #define LAUNCH_GEMM(OutType, Unsigned)                                                  \
        svdquant_scaled_mm_w4a4_kernel<OutType, Unsigned><<<grid, block, 0, stream>>>(      \
            reinterpret_cast<const int8_t*>(act),                                           \
            reinterpret_cast<const int8_t*>(wgt),                                           \
            reinterpret_cast<const OutType*>(ascales),                                      \
            reinterpret_cast<const OutType*>(wscales),                                      \
            reinterpret_cast<const float*>(lora_act_in),                                    \
            reinterpret_cast<const OutType*>(lora_up),                                      \
            reinterpret_cast<const OutType*>(bias),                                         \
            reinterpret_cast<OutType*>(out),                                                \
            M, N, K, R)

    if (out_dtype_code == 2 /* bf16 */) {
        if (act_unsigned) LAUNCH_GEMM(__nv_bfloat16, true);
        else              LAUNCH_GEMM(__nv_bfloat16, false);
    } else if (out_dtype_code == 1 /* fp16 */) {
        if (act_unsigned) LAUNCH_GEMM(__half, true);
        else              LAUNCH_GEMM(__half, false);
    }
    #undef LAUNCH_GEMM
}

} // extern "C"
