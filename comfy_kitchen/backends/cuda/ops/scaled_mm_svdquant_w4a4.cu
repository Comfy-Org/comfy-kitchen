// SPDX-License-Identifier: Apache-2.0
//
// Kitchen CUDA SVDQuant W4A4 int4 GEMM with per-group dequant.
//
// Tile layout:
//   CTA  = 8 warps × 32 threads = 256 threads, warp grid 2×4
//   CTA covers 32 M × 128 N output (each warp computes 16 M × 32 N)
//   grid = (ceil(N/128), ceil(M/32))
//
// Per K iteration (BLOCK_K = kGroupSize = 64):
//   A (32 M × 32 B/group) and, by default, B (128 N × 32 B/group) are
//   CTA-cooperatively loaded into shmem via cp.async, triple-buffered
//   (kStages=3).
//   Each warp issues kNUnroll = 4 MMAs covering its 16M × 32N output tile.
//
// Experimental shared-scale path:
//   COMFY_KITCHEN_SVDQUANT_SHARED_SCALE=1 stages each CTA's 128 wscales for the
//   current K group once via cp.async, then all warps read scales from shmem.
//   This approximates nunchaku's packed scale reuse without requiring a new
//   checkpoint scale layout or warp shuffle broadcast in the hot dequant loop.
//
// Weight storage:
//   natural:
//     wgt     (N, K/2)
//     wscales (K/64, N)
//   kitchen_tile_packed_w4a4:
//     wgt     (N/128, K/64, 32, 128), where the last axis is
//             4 interleaved N rows × 32 packed-K bytes.
//     wscales (N/128, K/64, 128)
//
// The tile-packed path only changes global-memory addressing. The shmem layout
// remains row-major (128 N rows × 32 packed-K bytes), so the MMA code below is
// shared between the two layouts.
//
// Dequant: int32 MMA output → fp32 scalar multiply-adds with per-group
// ascale × wscale by default. fp32 accumulator (not fp16) for numerical robustness —
// in production Qwen-Image-Edit, scale products (ascale × wscale) can reach
// ~100 and per-MMA d_reg values can reach ~3000, so the per-term product
// overflows fp16's ±65504 range mid-sampling and silently propagates NaN
// (see ops/scaled_mm_svdquant_w4a4.cu accumulator declaration).
//
// Experimental fast path: COMFY_KITCHEN_SVDQUANT_FAST_ACCUM=1 dispatches a
// packed half2/bfloat162 accumulator that mirrors nunchaku's USE_FP32_ACCUM=false
// path. It is intentionally opt-in until real-model stability is verified.
//
// act_unsigned: when true, A fragments are interpreted by u4.s4 MMA instead
// of s4.s4 (enables +1 bit of activation precision for layers whose input
// is known non-negative, e.g., post-GELU fc2 with nunchaku's +0.171875 shift
// — caller applies the shift at the layer level; this kernel only picks the
// MMA variant).
//
// Bias is applied in this kernel at writeback. Optional LoRA-up epilogue
// fusion uses warp-level bf16/fp16 m16n8k16 MMA over 16-rank slices and adds
// the low-rank correction before the final global store.
#include "svdquant_utils.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

namespace {

using comfy::svdquant::kGroupSize;
using comfy::svdquant::mma_m16n8k64_s4s4s32;
using comfy::svdquant::mma_m16n8k64_u4s4s32;
using comfy::svdquant::cvta_smem_u32;
using comfy::svdquant::ldmatrix_x2;
using comfy::svdquant::ldmatrix_x4;
using comfy::svdquant::mma_m16n8k16_f32;
using comfy::svdquant::cp_async_16b;
using comfy::svdquant::cp_async_commit_group;
using comfy::svdquant::cp_async_wait_group;

constexpr int kStages = 3;  // 3-stage cp.async hides GMEM/scale latency better than 2-stage on sm_120.

constexpr int kMUnroll  = 1;               // MMA_M tiles per warp (each 16 M)
constexpr int kWarpM    = kMUnroll * 16;   // 16 M rows per warp
constexpr int kNUnroll  = 4;               // MMAs per K iter per warp (N dim)
constexpr int kWarpN    = kNUnroll * 8;    // 32 N cols per warp
constexpr int kWarpsM   = 2;               // warps stacked in M
constexpr int kWarpsN   = 4;               // warps stacked in N
constexpr int kNumWarps = kWarpsM * kWarpsN;      // 8
constexpr int kBlockM   = kWarpM * kWarpsM;       // 32 M per CTA
constexpr int kBlockN   = kWarpN * kWarpsN;       // 128 N per CTA
constexpr int kBlockKBytes = kGroupSize / 2;      // 32 bytes of int4 per K group
constexpr int kTileInterleave = 4;
constexpr int kLoraRankTile = 16;
constexpr int kThreadsPerBlock = kNumWarps * 32;
constexpr int kBLoadChunks = kBlockN * 2;  // two 16-byte chunks per N row.
constexpr int kBLoadSweeps = (kBLoadChunks + kThreadsPerBlock - 1) / kThreadsPerBlock;
constexpr int kScaleChunkBytes = 16;

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

template<typename OutType>
__device__ __forceinline__ float2 load_scale2(const OutType* p);

template<>
__device__ __forceinline__ float2 load_scale2<__nv_bfloat16>(const __nv_bfloat16* p) {
    return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p));
}

template<>
__device__ __forceinline__ float2 load_scale2<__half>(const __half* p) {
    return __half22float2(*reinterpret_cast<const __half2*>(p));
}

template<typename OutType>
struct Vec2Traits;

template<>
struct Vec2Traits<__half> {
    using Pair = __half2;

    __device__ __forceinline__ static Pair zero() {
        return __floats2half2_rn(0.f, 0.f);
    }

    __device__ __forceinline__ static Pair from_floats(float x, float y) {
        return __floats2half2_rn(x, y);
    }

    __device__ __forceinline__ static Pair from_ints(int x, int y) {
        return __floats2half2_rn(__int2float_rn(x), __int2float_rn(y));
    }

    __device__ __forceinline__ static Pair mul(Pair a, Pair b) {
        return __hmul2(a, b);
    }

    __device__ __forceinline__ static Pair fma(Pair a, Pair b, Pair c) {
        return __hfma2(a, b, c);
    }

    __device__ __forceinline__ static __half low(Pair v) {
        return __low2half(v);
    }

    __device__ __forceinline__ static __half high(Pair v) {
        return __high2half(v);
    }
};

template<>
struct Vec2Traits<__nv_bfloat16> {
    using Pair = __nv_bfloat162;

    __device__ __forceinline__ static uint32_t to_bits(Pair v) {
        const __nv_bfloat162_raw raw = static_cast<__nv_bfloat162_raw>(v);
        return static_cast<uint32_t>(raw.x) | (static_cast<uint32_t>(raw.y) << 16);
    }

    __device__ __forceinline__ static Pair from_bits(uint32_t v) {
        __nv_bfloat162_raw raw;
        raw.x = static_cast<unsigned short>(v & 0xffffu);
        raw.y = static_cast<unsigned short>(v >> 16);
        return Pair(raw);
    }

    __device__ __forceinline__ static Pair zero() {
        return __floats2bfloat162_rn(0.f, 0.f);
    }

    __device__ __forceinline__ static Pair from_floats(float x, float y) {
        return __floats2bfloat162_rn(x, y);
    }

    __device__ __forceinline__ static Pair from_ints(int x, int y) {
        return __floats2bfloat162_rn(__int2float_rn(x), __int2float_rn(y));
    }

    __device__ __forceinline__ static Pair mul(Pair a, Pair b) {
        return __hmul2(a, b);
    }

    __device__ __forceinline__ static Pair fma(Pair a, Pair b, Pair c) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        uint32_t out;
        asm("{ fma.rn.bf16x2 %0, %1, %2, %3; }\n"
            : "=r"(out)
            : "r"(to_bits(a)), "r"(to_bits(b)), "r"(to_bits(c)));
        return from_bits(out);
#else
        const float2 af = __bfloat1622float2(a);
        const float2 bf = __bfloat1622float2(b);
        const float2 cf = __bfloat1622float2(c);
        return __floats2bfloat162_rn(af.x * bf.x + cf.x, af.y * bf.y + cf.y);
#endif
    }

    __device__ __forceinline__ static __nv_bfloat16 low(Pair v) {
        return __low2bfloat16(v);
    }

    __device__ __forceinline__ static __nv_bfloat16 high(Pair v) {
        return __high2bfloat16(v);
    }
};

template<
    typename OutType,
    bool kActUnsigned,
    bool kTilePacked,
    bool kFastAccum,
    bool kSharedScale,
    bool kFuseLora,
    bool kRankOne,
    int kMUnrollT = kMUnroll,
    int kWarpsMT = kWarpsM,
    int kStagesT = kStages>
__global__ void svdquant_scaled_mm_w4a4_kernel(
    const int8_t* __restrict__ act,          // (M, K/2)
    const int8_t* __restrict__ wgt,          // (N, K/2)
    const OutType* __restrict__ ascales,     // (K/G, M)
    const OutType* __restrict__ wscales,     // (K/G, N)
    const OutType* __restrict__ lora_act_in, // (M, R) when kFuseLora
    const OutType* __restrict__ lora_up,     // (N, R) or tile-packed (N/128, R, 128)
    const OutType* __restrict__ bias,        // optional (N,) flat bias
    OutType* __restrict__ out,               // (M, N)
    int M, int N, int K, int R)
{
    if constexpr (!kFuseLora) {
        (void)lora_act_in; (void)lora_up; (void)R;
    }

    // Tile geometry / pipeline depth may be widened per instantiation (rank-one
    // uses a taller CTA to cut B-tile rereads); these shadow the file-scope
    // defaults, and every derived constant is recomputed from them.
    constexpr int kMUnroll = kMUnrollT;
    constexpr int kWarpM   = kMUnroll * 16;
    constexpr int kWarpsM  = kWarpsMT;
    constexpr int kBlockM  = kWarpM * kWarpsM;
    constexpr int kNumWarps = kWarpsM * kWarpsN;
    constexpr int kThreadsPerBlock = kNumWarps * 32;
    constexpr int kStages = kStagesT;
    constexpr int kBLoadSweeps = (kBlockN * 2 + kThreadsPerBlock - 1) / kThreadsPerBlock;
    constexpr int kALoadSweeps = (kBlockM * 2 + kThreadsPerBlock - 1) / kThreadsPerBlock;
    // Rank-one loads fragments with ldmatrix; a 48-byte smem row stride staggers the
    // 8 row addresses of each ldmatrix phase across distinct bank groups (32-byte
    // stride would alias rows r and r+4). Non-rank-one keeps the packed 32B stride.
    constexpr int kSmemRowBytes = kRankOne ? (kBlockKBytes + 16) : kBlockKBytes;
    static_assert((kWarpsM & (kWarpsM - 1)) == 0, "kWarpsM must be a power of two");

    // CTA coordinates
    const int cta_m = blockIdx.y * kBlockM;
    const int cta_n = blockIdx.x * kBlockN;
    const int cta_n_tile = blockIdx.x;

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

    // Per-warp accumulator: kMUnroll M-tiles × kNUnroll N-chunks.
    // We accumulate in fp32 (not fp16) because in production Qwen-Image-Edit,
    // per-group scale products ascale*wscale can reach ~100 and single-MMA d_reg
    // values can reach ~3000, pushing the per-term contribution well past fp16's
    // ±65504 range. fp16 accumulation overflow -> inf -> NaN propagation causes
    // black-image mid-sampling failures that single-layer random parity tests miss.
    // nunchaku works with fp16 because their calibration bounds scales tighter;
    // kitchen's conservative choice is fp32 for end-to-end robustness.
    using Vec2 = Vec2Traits<OutType>;
    using Pair = typename Vec2::Pair;
    float out_f[kMUnroll][kNUnroll][4];
    Pair out_h[kMUnroll][kNUnroll][2];
    // kRankOne: per-row x per-col scales factor out of the K loop entirely, so the
    // MMA carries raw int32 accumulators across ALL K groups (|acc| <= 49*K << 2^31)
    // and dequantization collapses into the epilogue — no per-group scale loads, no
    // per-group fp conversion (the QServe W4A4 main-loop critique does not apply).
    int32_t acc_i[kMUnroll][kNUnroll][4];
    if constexpr (kRankOne) {
        #pragma unroll
        for (int mi = 0; mi < kMUnroll; ++mi) {
            #pragma unroll
            for (int c = 0; c < kNUnroll; ++c) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) acc_i[mi][c][i] = 0;
            }
        }
    }
    if constexpr (kFastAccum) {
        #pragma unroll
        for (int mi = 0; mi < kMUnroll; ++mi) {
            #pragma unroll
            for (int c = 0; c < kNUnroll; ++c) {
                out_h[mi][c][0] = Vec2::zero();
                out_h[mi][c][1] = Vec2::zero();
                if constexpr (kFuseLora) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) out_f[mi][c][i] = 0.f;
                }
            }
        }
    } else {
        #pragma unroll
        for (int mi = 0; mi < kMUnroll; ++mi) {
            #pragma unroll
            for (int c = 0; c < kNUnroll; ++c) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) out_f[mi][c][i] = 0.f;
            }
        }
    }

    const int K_half      = K / 2;
    const int num_groups  = K / kGroupSize;

    // Shared memory: triple-buffered A and B tiles.
    // B stage: kBlockN rows × kBlockKBytes bytes = 128 * 32 = 4 KB
    // A stage: kBlockM rows × kBlockKBytes bytes =  32 * 32 = 1 KB
    __shared__ alignas(16) int8_t smem_B[kStages][kBlockN * kSmemRowBytes];
    __shared__ alignas(16) int8_t smem_A[kStages][kBlockM * kSmemRowBytes];
    __shared__ alignas(16) int8_t smem_WS[kSharedScale ? kStages : 1][kBlockN * sizeof(OutType)];
    __shared__ alignas(16) OutType smem_LoraA[kFuseLora ? (kBlockM * kLoraRankTile) : 1];
    __shared__ alignas(16) OutType smem_LoraB[kFuseLora ? (kBlockN * kLoraRankTile) : 1];

    // ---------- Helper: issue async cp for B tile at group `g` into stage ----------
    auto issue_B_load = [&](int g, int stage) {
        if (g >= num_groups) return;
        const int thread_idx = threadIdx.x;
        #pragma unroll
        for (int sweep = 0; sweep < kBLoadSweeps; ++sweep) {
            const int t = thread_idx + sweep * kThreadsPerBlock;
            // each t loads one 16-byte chunk: n_row = t/2, half = t%2
            if (t < kBlockN * 2) {
                const int n_row = t >> 1;
                const int half  = t & 1;
                const int n_global = cta_n + n_row;
                int8_t* dst = &smem_B[stage][n_row * kSmemRowBytes + half * 16];
                if (n_global < N) {
                    const int8_t* src;
                    if constexpr (kTilePacked) {
                        const int n_quad = n_row >> 2;
                        const int n_lane = n_row & (kTileInterleave - 1);
                        src = wgt + (cta_n_tile * num_groups + g) *
                              (kBlockN * kBlockKBytes) +
                              n_quad * (kTileInterleave * kBlockKBytes) +
                              n_lane * kBlockKBytes + half * 16;
                    } else {
                        src = wgt + n_global * K_half + g * kBlockKBytes + half * 16;
                    }
                    cp_async_16b(dst, src);
                } else {
                    // Out-of-bounds rows: pad with zeros (use regular 16-byte zero store).
                    reinterpret_cast<uint4*>(dst)[0] = {0, 0, 0, 0};
                }
            }
        }
    };

    auto load_B_fragment = [&](int stage, int c, uint32_t (&b_reg)[2]) {
        if constexpr (kRankOne) {
            // ldmatrix.x2: lanes 0..15 address the 8 n-rows x two 16B k-chunks of
            // this N-chunk's tile (staging zero-pads OOB rows, so no guard needed).
            const int r = lane & 7;
            const int h = (lane >> 3) & 1;
            const int row_local = (warp_n * kWarpN) + c * 8 + r;
            const int8_t* addr = &smem_B[stage][row_local * kSmemRowBytes + h * 16];
            ldmatrix_x2(b_reg, cvta_smem_u32(addr));
            return;
        }
        const int b_col_local = (warp_n * kWarpN) + c * 8 + groupID;
        const int b_col_global = cta_n + b_col_local;
        b_reg[0] = b_reg[1] = 0;
        if (b_col_local < kBlockN && b_col_global < N) {
            const int byte0 = tid_in_group * 8;
            const int8_t* row_base = &smem_B[stage][b_col_local * kSmemRowBytes];
            b_reg[0] = *reinterpret_cast<const uint32_t*>(row_base + byte0);
            b_reg[1] = *reinterpret_cast<const uint32_t*>(row_base + byte0 + 4);
        }
    };

    // ---------- Helper: issue async cp for A tile at group `g` into stage ----------
    auto issue_A_load = [&](int g, int stage) {
        if (g >= num_groups) return;
        #pragma unroll
        for (int sweep = 0; sweep < kALoadSweeps; ++sweep) {
            const int t = threadIdx.x + sweep * kThreadsPerBlock;
            if (t < kBlockM * 2) {
                const int m_row = t >> 1;
                const int half  = t & 1;
                const int m_global = cta_m + m_row;
                int8_t* dst = &smem_A[stage][m_row * kSmemRowBytes + half * 16];
                if (m_global < M) {
                    const int8_t* src = act + m_global * K_half + g * kBlockKBytes + half * 16;
                    cp_async_16b(dst, src);
                } else {
                    reinterpret_cast<uint4*>(dst)[0] = {0, 0, 0, 0};
                }
            }
        }
    };

    auto issue_WS_load = [&](int g, int stage) {
        if constexpr (!kSharedScale) {
            (void)g;
            (void)stage;
            return;
        }
        if (g >= num_groups) return;
        constexpr int kScaleElemsPerChunk = kScaleChunkBytes / sizeof(OutType);
        constexpr int kScaleLoadChunks = kBlockN / kScaleElemsPerChunk;
        const int t = threadIdx.x;
        if (t < kScaleLoadChunks) {
            const int n0 = t * kScaleElemsPerChunk;
            int8_t* dst = &smem_WS[stage][t * kScaleChunkBytes];
            const OutType* src;
            if constexpr (kTilePacked) {
                src = &wscales[(cta_n_tile * num_groups + g) * kBlockN + n0];
            } else {
                src = &wscales[g * N + cta_n + n0];
            }
            if (cta_n + n0 + kScaleElemsPerChunk - 1 < N) {
                cp_async_16b(dst, src);
            } else {
                OutType* dst_vals = reinterpret_cast<OutType*>(dst);
                #pragma unroll
                for (int i = 0; i < kScaleElemsPerChunk; ++i) {
                    dst_vals[i] = (cta_n + n0 + i < N) ? src[i] : OutType{};
                }
            }
        }
    };

    // ---------- Prime the pipeline: launch first (kStages-1) loads ----------
    #pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        issue_A_load(s, s);
        issue_B_load(s, s);
        issue_WS_load(s, s);
        cp_async_commit_group();
    }

    for (int g = 0; g < num_groups; ++g) {
        // Start next iteration's load (stage = (g + kStages - 1) % kStages)
        const int next_g = g + kStages - 1;
        if (next_g < num_groups) {
            const int next_stage = (g + kStages - 1) % kStages;
            issue_A_load(next_g, next_stage);
            issue_B_load(next_g, next_stage);
            issue_WS_load(next_g, next_stage);
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
            if constexpr (kRankOne) {
                // ldmatrix.x4 over the 16-row x 32B A tile: matrices land as
                // {r0k0, r1k0, r0k1, r1k1} — the exact s4 MMA A-fragment order.
                // Both operands use the same ldmatrix k-permutation, so the MMA
                // dot product is unchanged; staging zero-pads rows >= M.
                const int row_local = warp_m * kWarpM + mi * 16 + (lane & 15);
                const int8_t* addr =
                    &smem_A[cur_stage][row_local * kSmemRowBytes + ((lane >> 4) & 1) * 16];
                ldmatrix_x4(a_reg[mi], cvta_smem_u32(addr));
                continue;
            }
            const int m_tile_base = warp_m_base + mi * 16;
            const int row0_m = m_tile_base + groupID;
            const int row1_m = m_tile_base + groupID + 8;
            const int row0_local = warp_m * kWarpM + mi * 16 + groupID;
            const int row1_local = warp_m * kWarpM + mi * 16 + groupID + 8;
            a_reg[mi][0] = a_reg[mi][1] = a_reg[mi][2] = a_reg[mi][3] = 0;
            if (row0_m < M) {
                const int8_t* rb = &smem_A[cur_stage][row0_local * kSmemRowBytes];
                a_reg[mi][0] = *reinterpret_cast<const uint32_t*>(rb + tid_in_group * 8);
                a_reg[mi][2] = *reinterpret_cast<const uint32_t*>(rb + tid_in_group * 8 + 4);
            }
            if (row1_m < M) {
                const int8_t* rb = &smem_A[cur_stage][row1_local * kSmemRowBytes];
                a_reg[mi][1] = *reinterpret_cast<const uint32_t*>(rb + tid_in_group * 8);
                a_reg[mi][3] = *reinterpret_cast<const uint32_t*>(rb + tid_in_group * 8 + 4);
            }
            if constexpr (!kRankOne) {
                as_row0_arr[mi] = (row0_m < M) ? load_scale<OutType>(&ascales[g * M + row0_m]) : 0.f;
                as_row1_arr[mi] = (row1_m < M) ? load_scale<OutType>(&ascales[g * M + row1_m]) : 0.f;
            }
        }

        // Pre-load this K-iter's wscales into per-lane registers, hoisted out of
        // the inner MMA loop so the compiler can schedule the loads ahead of MMAs.
        float ws_regs[kNUnroll][2];
        #pragma unroll
        for (int cc = 0; cc < kNUnroll && !kRankOne; ++cc) {
            const int col0 = warp_n_base + cc * 8 + tid_in_group * 2 + 0;
            const int col1 = warp_n_base + cc * 8 + tid_in_group * 2 + 1;
            if constexpr (kSharedScale) {
                const int n0 = col0 - cta_n;
                const OutType* ws_base_g = reinterpret_cast<const OutType*>(&smem_WS[cur_stage][0]);
                if (col1 < N) {
                    const float2 ws = load_scale2<OutType>(&ws_base_g[n0]);
                    ws_regs[cc][0] = ws.x;
                    ws_regs[cc][1] = ws.y;
                } else {
                    ws_regs[cc][0] = (col0 < N) ? load_scale<OutType>(&ws_base_g[n0]) : 0.f;
                    ws_regs[cc][1] = 0.f;
                }
            } else {
                if constexpr (kTilePacked) {
                    const int n0 = col0 - cta_n;
                    const OutType* ws_base_g = &wscales[(cta_n_tile * num_groups + g) * kBlockN];
                    if (col1 < N) {
                        const float2 ws = load_scale2<OutType>(&ws_base_g[n0]);
                        ws_regs[cc][0] = ws.x;
                        ws_regs[cc][1] = ws.y;
                    } else {
                        ws_regs[cc][0] = (col0 < N) ? load_scale<OutType>(&ws_base_g[n0]) : 0.f;
                        ws_regs[cc][1] = 0.f;
                    }
                } else {
                    const OutType* ws_base_g = &wscales[g * N];
                    if (col1 < N) {
                        const float2 ws = load_scale2<OutType>(&ws_base_g[col0]);
                        ws_regs[cc][0] = ws.x;
                        ws_regs[cc][1] = ws.y;
                    } else {
                        ws_regs[cc][0] = (col0 < N) ? load_scale<OutType>(&ws_base_g[col0]) : 0.f;
                        ws_regs[cc][1] = 0.f;
                    }
                }
            }
        }

        #pragma unroll
        for (int c = 0; c < kNUnroll; ++c) {
            uint32_t b_reg[2];
            load_B_fragment(cur_stage, c, b_reg);

            const float ws_col0 = ws_regs[c][0];
            const float ws_col1 = ws_regs[c][1];

            // Reuse b_reg for each M-tile
            #pragma unroll
            for (int mi = 0; mi < kMUnroll; ++mi) {
                if constexpr (kRankOne) {
                    // Accumulate raw int32 across the whole K loop; dequant in epilogue.
                    if constexpr (kActUnsigned) {
                        mma_m16n8k64_u4s4s32(a_reg[mi], b_reg, acc_i[mi][c], acc_i[mi][c]);
                    } else {
                        mma_m16n8k64_s4s4s32(a_reg[mi], b_reg, acc_i[mi][c], acc_i[mi][c]);
                    }
                    continue;
                }
                int32_t c_reg[4] = {0, 0, 0, 0};
                int32_t d_reg[4];
                if constexpr (kActUnsigned) {
                    mma_m16n8k64_u4s4s32(a_reg[mi], b_reg, c_reg, d_reg);
                } else {
                    mma_m16n8k64_s4s4s32(a_reg[mi], b_reg, c_reg, d_reg);
                }

                if constexpr (kFastAccum) {
                    // nunchaku-like fast dequant: convert each row's two output
                    // columns to a packed 16-bit pair and accumulate with h*fma2.
                    const Pair ws_pair = Vec2::from_floats(ws_col0, ws_col1);
                    const Pair row0_scale = Vec2::mul(
                        Vec2::from_floats(as_row0_arr[mi], as_row0_arr[mi]), ws_pair);
                    const Pair row1_scale = Vec2::mul(
                        Vec2::from_floats(as_row1_arr[mi], as_row1_arr[mi]), ws_pair);
                    out_h[mi][c][0] = Vec2::fma(
                        Vec2::from_ints(d_reg[0], d_reg[1]), row0_scale, out_h[mi][c][0]);
                    out_h[mi][c][1] = Vec2::fma(
                        Vec2::from_ints(d_reg[2], d_reg[3]), row1_scale, out_h[mi][c][1]);
                } else {
                    // fp32 dequant: out_f += cvt(d) * ascale * wscale. Slower than
                    // the packed hfma2/bf162 variant but preserves the robust path.
                    out_f[mi][c][0] += static_cast<float>(d_reg[0]) * as_row0_arr[mi] * ws_col0;
                    out_f[mi][c][1] += static_cast<float>(d_reg[1]) * as_row0_arr[mi] * ws_col1;
                    out_f[mi][c][2] += static_cast<float>(d_reg[2]) * as_row1_arr[mi] * ws_col0;
                    out_f[mi][c][3] += static_cast<float>(d_reg[3]) * as_row1_arr[mi] * ws_col1;
                }
            }
        }
    }
    cp_async_wait_group<0>();  // drain pipeline

    if constexpr (kRankOne) {
        // Rank-1 dequant: one per-row and one per-col scale (read from the first
        // group row of the (possibly 1-group) ascales/wscales buffers).
        #pragma unroll
        for (int mi = 0; mi < kMUnroll; ++mi) {
            const int m_tile_base = warp_m_base + mi * 16;
            const int row0_m = m_tile_base + groupID;
            const int row1_m = m_tile_base + groupID + 8;
            const float as0 = (row0_m < M) ? load_scale<OutType>(&ascales[row0_m]) : 0.f;
            const float as1 = (row1_m < M) ? load_scale<OutType>(&ascales[row1_m]) : 0.f;
            #pragma unroll
            for (int c = 0; c < kNUnroll; ++c) {
                const int col0 = warp_n_base + c * 8 + tid_in_group * 2 + 0;
                const int col1 = col0 + 1;
                const float ws0 = (col0 < N) ? load_scale<OutType>(&wscales[col0]) : 0.f;
                const float ws1 = (col1 < N) ? load_scale<OutType>(&wscales[col1]) : 0.f;
                out_f[mi][c][0] = static_cast<float>(acc_i[mi][c][0]) * as0 * ws0;
                out_f[mi][c][1] = static_cast<float>(acc_i[mi][c][1]) * as0 * ws1;
                out_f[mi][c][2] = static_cast<float>(acc_i[mi][c][2]) * as1 * ws0;
                out_f[mi][c][3] = static_cast<float>(acc_i[mi][c][3]) * as1 * ws1;
            }
        }
    }

    if constexpr (kFuseLora) {
        // LoRA-up epilogue: out += lora_act_in @ lora_up.T.
        // Shared staging keeps natural/tile-packed checkpoint layouts out of
        // the MMA fragment code. Each warp computes its existing 16M x 32N
        // output tile as four m16n8k16 instructions per rank slice.
        for (int r0 = 0; r0 < R; r0 += kLoraRankTile) {
            for (int idx = threadIdx.x; idx < kBlockM * kLoraRankTile; idx += kThreadsPerBlock) {
                const int m_local = idx / kLoraRankTile;
                const int r_local = idx - m_local * kLoraRankTile;
                const int m_global = cta_m + m_local;
                const int r = r0 + r_local;
                smem_LoraA[idx] = (m_global < M && r < R)
                    ? lora_act_in[m_global * R + r]
                    : OutType{};
            }

            for (int idx = threadIdx.x; idx < kBlockN * kLoraRankTile; idx += kThreadsPerBlock) {
                const int n_local = idx / kLoraRankTile;
                const int r_local = idx - n_local * kLoraRankTile;
                const int n_global = cta_n + n_local;
                const int r = r0 + r_local;
                OutType v{};
                if (n_global < N && r < R) {
                    if constexpr (kTilePacked) {
                        v = lora_up[(cta_n_tile * R + r) * kBlockN + n_local];
                    } else {
                        v = lora_up[n_global * R + r];
                    }
                }
                smem_LoraB[idx] = v;
            }
            __syncthreads();

            uint32_t a_frag[4];
            const OutType* a_addr = &smem_LoraA[
                (warp_m * kWarpM + (lane & 15)) * kLoraRankTile + ((lane >> 4) * 8)];
            ldmatrix_x4(a_frag, cvta_smem_u32(a_addr));

            #pragma unroll
            for (int b_pair = 0; b_pair < kNUnroll / 2; ++b_pair) {
                const int n_off = warp_n * kWarpN + b_pair * 16;
                uint32_t b_frag4[4];
                const OutType* b_addr = &smem_LoraB[
                    (n_off + (lane & 15)) * kLoraRankTile + ((lane >> 4) * 8)];
                ldmatrix_x4(b_frag4, cvta_smem_u32(b_addr));

                {
                    const uint32_t b0[2] = {b_frag4[0], b_frag4[2]};
                    if constexpr (kFastAccum) {
                        float tmp[4] = {0.f, 0.f, 0.f, 0.f};
                        mma_m16n8k16_f32<OutType>(tmp, a_frag, b0);
                        out_f[0][b_pair * 2 + 0][0] += tmp[0];
                        out_f[0][b_pair * 2 + 0][1] += tmp[1];
                        out_f[0][b_pair * 2 + 0][2] += tmp[2];
                        out_f[0][b_pair * 2 + 0][3] += tmp[3];
                    } else {
                        mma_m16n8k16_f32<OutType>(out_f[0][b_pair * 2 + 0], a_frag, b0);
                    }
                }
                {
                    const uint32_t b1[2] = {b_frag4[1], b_frag4[3]};
                    if constexpr (kFastAccum) {
                        float tmp[4] = {0.f, 0.f, 0.f, 0.f};
                        mma_m16n8k16_f32<OutType>(tmp, a_frag, b1);
                        out_f[0][b_pair * 2 + 1][0] += tmp[0];
                        out_f[0][b_pair * 2 + 1][1] += tmp[1];
                        out_f[0][b_pair * 2 + 1][2] += tmp[2];
                        out_f[0][b_pair * 2 + 1][3] += tmp[3];
                    } else {
                        mma_m16n8k16_f32<OutType>(out_f[0][b_pair * 2 + 1], a_frag, b1);
                    }
                }
            }
            __syncthreads();
        }
    }

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

            if constexpr (kFastAccum) {
                const Pair row0 = out_h[mi][c][0];
                const Pair row1 = out_h[mi][c][1];
                if (row0_m < M && col0 < N) {
                    const OutType base = Vec2::low(row0);
                    float v = load_scale<OutType>(&base) + (kFuseLora ? out_f[mi][c][0] : 0.f) +
                              (bias ? load_scale<OutType>(&bias[col0]) : 0.f);
                    out[row0_m * N + col0] = fp32_to_out<OutType>(v);
                }
                if (row0_m < M && col1 < N) {
                    const OutType base = Vec2::high(row0);
                    float v = load_scale<OutType>(&base) + (kFuseLora ? out_f[mi][c][1] : 0.f) +
                              (bias ? load_scale<OutType>(&bias[col1]) : 0.f);
                    out[row0_m * N + col1] = fp32_to_out<OutType>(v);
                }
                if (row1_m < M && col0 < N) {
                    const OutType base = Vec2::low(row1);
                    float v = load_scale<OutType>(&base) + (kFuseLora ? out_f[mi][c][2] : 0.f) +
                              (bias ? load_scale<OutType>(&bias[col0]) : 0.f);
                    out[row1_m * N + col0] = fp32_to_out<OutType>(v);
                }
                if (row1_m < M && col1 < N) {
                    const OutType base = Vec2::high(row1);
                    float v = load_scale<OutType>(&base) + (kFuseLora ? out_f[mi][c][3] : 0.f) +
                              (bias ? load_scale<OutType>(&bias[col1]) : 0.f);
                    out[row1_m * N + col1] = fp32_to_out<OutType>(v);
                }
            } else {
                const float bias0 = (bias && col0 < N) ? load_scale<OutType>(&bias[col0]) : 0.f;
                const float bias1 = (bias && col1 < N) ? load_scale<OutType>(&bias[col1]) : 0.f;
                if (row0_m < M && col0 < N) out[row0_m * N + col0] = fp32_to_out<OutType>(out_f[mi][c][0] + bias0);
                if (row0_m < M && col1 < N) out[row0_m * N + col1] = fp32_to_out<OutType>(out_f[mi][c][1] + bias1);
                if (row1_m < M && col0 < N) out[row1_m * N + col0] = fp32_to_out<OutType>(out_f[mi][c][2] + bias0);
                if (row1_m < M && col1 < N) out[row1_m * N + col1] = fp32_to_out<OutType>(out_f[mi][c][3] + bias1);
            }
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
    int tile_packed,
    int fast_accum,
    int shared_scale,
    int fuse_lora,
    int rank_one,
    cudaStream_t stream)
{
    if (K % comfy::svdquant::kGroupSize != 0) return;

    const dim3 grid((N + kBlockN - 1) / kBlockN, (M + kBlockM - 1) / kBlockM);
    const dim3 block(kNumWarps * 32);

    if (rank_one) {
        // int4_tensorwise path: natural layout, signed, robust accumulator, no LoRA.
        // Taller M tiles cut B-tile rereads (the freed per-group scale registers /
        // extra warps pay for them). Config selectable for tuning via
        // COMFY_KITCHEN_INT4_R1_CONFIG (0: 128M/8w/3st, 1: 128M/8w/4st,
        // 2: 64M/8w/3st, 3: 96M/8w/3st); default 0. Register-file trade: kMUnroll=4
        // costs 128 regs -> 2 CTAs/SM (33% occupancy); smaller M-tiles raise
        // occupancy at the price of more (L2-resident) B rereads.
        int r1_config = 0;
        if (const char* cfg = std::getenv("COMFY_KITCHEN_INT4_R1_CONFIG")) {
            r1_config = std::atoi(cfg);
        }
        #define LAUNCH_GEMM_R1(OutType, MU, WM, ST)                                         \
            do {                                                                            \
                constexpr int kBlockM_R1 = (MU) * 16 * (WM);                                \
                const dim3 grid_r1((N + kBlockN - 1) / kBlockN,                             \
                                   (M + kBlockM_R1 - 1) / kBlockM_R1);                      \
                const dim3 block_r1((WM) * kWarpsN * 32);                                   \
                svdquant_scaled_mm_w4a4_kernel<OutType, false, false, false, false, false,  \
                                               true, (MU), (WM), (ST)>                      \
                    <<<grid_r1, block_r1, 0, stream>>>(                                     \
                    reinterpret_cast<const int8_t*>(act),                                   \
                    reinterpret_cast<const int8_t*>(wgt),                                   \
                    reinterpret_cast<const OutType*>(ascales),                              \
                    reinterpret_cast<const OutType*>(wscales),                              \
                    reinterpret_cast<const OutType*>(lora_act_in),                          \
                    reinterpret_cast<const OutType*>(lora_up),                              \
                    reinterpret_cast<const OutType*>(bias),                                 \
                    reinterpret_cast<OutType*>(out),                                        \
                    M, N, K, R);                                                            \
            } while (0)
        #define DISPATCH_R1(OutType)                                                        \
            do {                                                                            \
                switch (r1_config) {                                                        \
                    case 1:  LAUNCH_GEMM_R1(OutType, 4, 2, 4); break;                       \
                    case 2:  LAUNCH_GEMM_R1(OutType, 2, 2, 3); break;                       \
                    case 3:  LAUNCH_GEMM_R1(OutType, 3, 2, 3); break;                       \
                    default: LAUNCH_GEMM_R1(OutType, 4, 2, 3); break;                       \
                }                                                                           \
            } while (0)
        if (out_dtype_code == 2 /* bf16 */) {
            DISPATCH_R1(__nv_bfloat16);
        } else if (out_dtype_code == 1 /* fp16 */) {
            DISPATCH_R1(__half);
        }
        #undef DISPATCH_R1
        #undef LAUNCH_GEMM_R1
        return;
    }

    #define LAUNCH_GEMM(OutType, Unsigned, TilePacked, FastAccum, SharedScale, FuseLora) \
        svdquant_scaled_mm_w4a4_kernel<                                                     \
            OutType, Unsigned, TilePacked, FastAccum, SharedScale, FuseLora, false>         \
            <<<grid, block, 0, stream>>>(                                                   \
            reinterpret_cast<const int8_t*>(act),                                           \
            reinterpret_cast<const int8_t*>(wgt),                                           \
            reinterpret_cast<const OutType*>(ascales),                                      \
            reinterpret_cast<const OutType*>(wscales),                                      \
            reinterpret_cast<const OutType*>(lora_act_in),                                  \
            reinterpret_cast<const OutType*>(lora_up),                                      \
            reinterpret_cast<const OutType*>(bias),                                         \
            reinterpret_cast<OutType*>(out),                                                \
            M, N, K, R)

    #define DISPATCH_SHARED_SCALE(OutType, Unsigned, TilePacked, FastAccum)                  \
        do {                                                                                 \
            if (shared_scale) {                                                              \
                if (fuse_lora) LAUNCH_GEMM(OutType, Unsigned, TilePacked, FastAccum, true, true);  \
                else           LAUNCH_GEMM(OutType, Unsigned, TilePacked, FastAccum, true, false); \
            } else {                                                                         \
                if (fuse_lora) LAUNCH_GEMM(OutType, Unsigned, TilePacked, FastAccum, false, true); \
                else           LAUNCH_GEMM(OutType, Unsigned, TilePacked, FastAccum, false, false);\
            }                                                                                \
        } while (0)

    if (out_dtype_code == 2 /* bf16 */) {
        if (tile_packed) {
            if (act_unsigned) {
                if (fast_accum) DISPATCH_SHARED_SCALE(__nv_bfloat16, true, true, true);
                else            DISPATCH_SHARED_SCALE(__nv_bfloat16, true, true, false);
            } else {
                if (fast_accum) DISPATCH_SHARED_SCALE(__nv_bfloat16, false, true, true);
                else            DISPATCH_SHARED_SCALE(__nv_bfloat16, false, true, false);
            }
        } else {
            if (act_unsigned) {
                if (fast_accum) DISPATCH_SHARED_SCALE(__nv_bfloat16, true, false, true);
                else            DISPATCH_SHARED_SCALE(__nv_bfloat16, true, false, false);
            } else {
                if (fast_accum) DISPATCH_SHARED_SCALE(__nv_bfloat16, false, false, true);
                else            DISPATCH_SHARED_SCALE(__nv_bfloat16, false, false, false);
            }
        }
    } else if (out_dtype_code == 1 /* fp16 */) {
        if (tile_packed) {
            if (act_unsigned) {
                if (fast_accum) DISPATCH_SHARED_SCALE(__half, true, true, true);
                else            DISPATCH_SHARED_SCALE(__half, true, true, false);
            } else {
                if (fast_accum) DISPATCH_SHARED_SCALE(__half, false, true, true);
                else            DISPATCH_SHARED_SCALE(__half, false, true, false);
            }
        } else {
            if (act_unsigned) {
                if (fast_accum) DISPATCH_SHARED_SCALE(__half, true, false, true);
                else            DISPATCH_SHARED_SCALE(__half, true, false, false);
            } else {
                if (fast_accum) DISPATCH_SHARED_SCALE(__half, false, false, true);
                else            DISPATCH_SHARED_SCALE(__half, false, false, false);
            }
        }
    }
    #undef DISPATCH_SHARED_SCALE
    #undef LAUNCH_GEMM
}

} // extern "C"
