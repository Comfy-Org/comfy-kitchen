// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// INT8 GEMM on WMMA (v_wmma_i32_16x16x16_iu8), gfx11 and gfx12. See mma.h.
// C[M, N] = (A[M, K] @ B[N, K]^T) * scale_a[row] * scale_b[col] + bias
#include <stdexcept>
#include <string>

#include "../device_arch.h"
#include "../epilogue.h"
#include "../gemm_gfx10.h"
#include "../gemm_wmma.h"
#include "../launchers.h"

namespace comfy::hip_backend {

struct PackedInt8B {
    const uint8_t* data;
    int kbytes;

    __forceinline__ __device__ uint4 chunk(int row, int kbyte) const {
        const int64_t offset = (static_cast<int64_t>(row / 256) * (kbytes / 64) + kbyte / 64) * 256 * 64
                               + (row % 256) * 64 + kbyte % 64;
        return *reinterpret_cast<const uint4*>(data + offset);
    }
};

// Small-M path. A row-major column uses a whole wave. With tiled B, each
// half-wave reduces a column and the other half reads the neighbouring column.
template <bool Tiled, typename OutT, typename Epi>
__global__ __launch_bounds__(256) void gemv_int8_kernel(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ B, OutT* __restrict__ C,
    int M, int N, int K, int ldc, Epi epi) {

    const int col = blockIdx.x * blockDim.y + threadIdx.y;
    const int row = blockIdx.y;
    if (col >= N || row >= M) return;

    constexpr int lanes = Tiled ? 16 : kWave;
    int sum = 0;
    for (int k = threadIdx.x * 16; k < K; k += lanes * 16) {
        uint4 av, bv;
        if constexpr (Tiled) {
            // M <= 8, so every row belongs to the first 128-row activation tile.
            const int64_t offset = static_cast<int64_t>(k / 128) * 128 * 128 + row * 128 + k % 128;
            av = *reinterpret_cast<const uint4*>(A + offset);
            bv = PackedInt8B{B, K}.chunk(col, k);
        } else {
            av = *reinterpret_cast<const uint4*>(A + static_cast<int64_t>(row) * K + k);
            bv = *reinterpret_cast<const uint4*>(B + static_cast<int64_t>(col) * K + k);
        }
        const int8_t* ab = reinterpret_cast<const int8_t*>(&av);
        const int8_t* bb = reinterpret_cast<const int8_t*>(&bv);
        #pragma unroll
        for (int i = 0; i < 16; ++i) sum += ab[i] * bb[i];
    }
    #pragma unroll
    for (int offset = lanes / 2; offset > 0; offset /= 2)
        sum += __shfl_xor(sum, offset, lanes);

    if (threadIdx.x == 0) {
        epi.init();
        C[static_cast<int64_t>(row) * ldc + col] =
            static_cast<OutT>(epi(row, col, static_cast<float>(sum)));
    }
}

template <typename OutT>
static void launch_int8(const int8_t* A, const int8_t* B, OutT* C, int M, int N, int K, int ldc,
                        EpiRowwise epi, hipStream_t stream) {
    const uint8_t* Au = reinterpret_cast<const uint8_t*>(A);
    const uint8_t* Bu = reinterpret_cast<const uint8_t*>(B);

    if (M <= 8) {
        dim3 block(kWave, 8);
        dim3 grid((N + 7) / 8, M);
        gemv_int8_kernel<false><<<grid, block, 0, stream>>>(Au, Bu, C, M, N, K, ldc, epi);
        return;
    }

    if (comfy_is_gfx10()) {
        // RDNA2: no matrix cores, use VALU v_dot4_i32_i8.
        //
        // One block per tile, never a persistent walk. The persistent variant
        // this replaces was 1.14x-1.80x slower on every shape measured on the
        // 6-CU gfx1035 (256^3 .. 4096x4096x8192), because a fixed grid of
        // min(tiles, wgps * blocks_per_cu) gives the hardware's own block
        // scheduler no tiles to overlap: consecutive tiles run back to back on
        // one CU with a barrier between them. Letting the scheduler stream the
        // grid is both simpler and faster. See the standalone benchmark notes B5.
        // Shared with gemm_int8_valu_kernel so the grid cannot drift from the
        // tile the kernel assumes. See kInt8BM in gemm_gfx10.h.
        constexpr int BM = kInt8BM;
        constexpr int BN = kInt8BN;
        dim3 block(kInt8Threads);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_int8_valu_kernel<OutT><<<grid, block, 0, stream>>>(
            A, B, epi, C, M, N, K, ldc);
    } else {
        // 8-bit operand, and on gfx11 the fragment is a whole 128-bit K-step
        // (mma.h's MmaInt8 : LdsPad16), so the heuristic may pick the wider thread
        // tile. Stated here rather than read off the policy type because the
        // tile choice is host code and only a device pass resolves MmaInt8 to an
        // architecture -- see GemmPolicy in gemm_wmma.h. Harmless on gfx12, whose
        // int8 fragment is 8 bytes: that whole branch is inside
        // `comfy_small_igpu()`.
        launch_gemm_wmma<MmaInt8, EpiRowwise, OutT>(Au, Bu, C, M, N, K, ldc, epi, stream,
                                                    GemmPolicy{false, true});
    }
}

}  // namespace comfy::hip_backend

// scale_b_stride is 1 for a per-output-channel weight scale, 0 for a scalar one.
extern "C" void launch_int8_gemm_kernel(
    const void* a, const void* b, void* c,
    const void* scale_a, const void* scale_b, int scale_b_stride,
    const void* bias, int bias_code,
    int M, int N, int K, int ldc, int out_code, hipStream_t stream) {

    using namespace comfy::hip_backend;

    if (M == 0 || N == 0) {
        return;  // a zero-dimension grid is hipErrorInvalidConfiguration
    }
    if (ldc < N) {
        throw std::runtime_error("int8_gemm: ldc=" + std::to_string(ldc) +
                                 " is narrower than N=" + std::to_string(N));
    }
    // Both the tile loader and the small-M GEMV read a row 16 bytes at a time; an
    // unaligned K would take the last chunk past the end of the row. An empty GEMM
    // does no reads, so this only applies once there is real work above.
    if (K % 16 != 0) {
        throw std::runtime_error("int8_gemm: K=" + std::to_string(K) +
                                 " must be a multiple of 16");
    }

    EpiRowwise epi{static_cast<const float*>(scale_a), static_cast<const float*>(scale_b),
                   scale_b_stride, bias, bias_code};

    const int8_t* A = static_cast<const int8_t*>(a);
    const int8_t* B = static_cast<const int8_t*>(b);

    if (out_code == kBF16) {
        launch_int8<__bf16>(A, B, static_cast<__bf16*>(c), M, N, K, ldc, epi, stream);
    } else if (out_code == kF16) {
        launch_int8<__half>(A, B, static_cast<__half*>(c), M, N, K, ldc, epi, stream);
    } else if (out_code == kF32) {
        launch_int8<float>(A, B, static_cast<float*>(c), M, N, K, ldc, epi, stream);
    } else {
        throw std::runtime_error("int8_gemm: unsupported output dtype");
    }
}

// INT4 on VALU, the int4 counterpart of the above. a and b are signed int4
// packed two per byte (low nibble = even k), so a row is K / 2 bytes wide and a
// 16-byte LDS read covers 32 k-values -- hence K must be a multiple of 32.
//
// The launcher lives here rather than in gemm_gfx10.h because that header is
// included by ops/gemm_fp16.hip as well, and a plain extern "C" definition would
// be emitted into both objects and collide at link time.
extern "C" void launch_int4_gemm_kernel(
    const void* a, const void* b, void* c,
    const void* scale_a, const void* scale_b, int scale_b_stride,
    const void* bias, int bias_code,
    int M, int N, int K, int ldc, int out_code, hipStream_t stream) {

    using namespace comfy::hip_backend;

    if (M == 0 || N == 0) {
        return;
    }
    // v_dot8_i32_i4 exists on gfx10 and nowhere else: the stub in gemm_gfx10.h's
    // !__GFX10__ arm is a trap. This binding is exposed on every target, so a
    // caller on an RDNA3/4 part would get that trap rather than an answer --
    // refuse here instead. convrot_w4a4.hip checks comfy_is_gfx10() before
    // calling, so the only way to land here is a direct call on the wrong device.
    if (!comfy_is_gfx10()) {
        throw std::runtime_error("int4_gemm: v_dot8_i32_i4 is gfx10 only");
    }
    if (ldc < N) {
        throw std::runtime_error("int4_gemm: ldc=" + std::to_string(ldc) +
                                 " is narrower than N=" + std::to_string(N));
    }
    // The tile reads a row 16 bytes at a time and each read is 32 k-values, so
    // the row has to be a whole number of reads. An empty GEMM does no reads, so
    // this only applies once there is real work above.
    if (K % 32 != 0) {
        throw std::runtime_error("int4_gemm: K=" + std::to_string(K) +
                                 " must be a multiple of 32");
    }

    EpiRowwise epi{static_cast<const float*>(scale_a), static_cast<const float*>(scale_b),
                   scale_b_stride, bias, bias_code};

    const int8_t* A = static_cast<const int8_t*>(a);
    const int8_t* B = static_cast<const int8_t*>(b);

    // One block per tile, the same grid launch_int8 settled on.
    dim3 block(kInt8Threads);
    dim3 grid((N + kInt8BN - 1) / kInt8BN, (M + kInt8BM - 1) / kInt8BM);

    if (out_code == kBF16) {
        gemm_int4_valu_kernel<__bf16><<<grid, block, 0, stream>>>(A, B, epi,
                                                                  static_cast<__bf16*>(c), M, N, K, ldc);
    } else if (out_code == kF16) {
        gemm_int4_valu_kernel<__half><<<grid, block, 0, stream>>>(A, B, epi,
                                                                 static_cast<__half*>(c), M, N, K, ldc);
    } else if (out_code == kF32) {
        gemm_int4_valu_kernel<float><<<grid, block, 0, stream>>>(A, B, epi,
                                                                 static_cast<float*>(c), M, N, K, ldc);
    } else {
        throw std::runtime_error("int4_gemm: unsupported output dtype");
    }
}

namespace comfy::hip_backend {

// Two 128-row activation tiles share one weight tile. Each half of the block
// retains the single-M accumulator shape and K order; only B staging is shared.
// The producer still writes 128x128 tiles, including a padded final M tile.
template <typename Epi, typename BSrc = const uint8_t*>
__global__ __launch_bounds__(512) void gemm_int8_dual_m_tiled_kernel(
    const uint8_t* __restrict__ A, typename GemmOperandA<BSrc>::type B,
    __bf16* __restrict__ C, int M, int N, int K, int ldc, Epi epi) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    constexpr int BM = 128, BN = 128, BK = 128, stride = BK + MmaInt8::kLdsPad;
    constexpr int TM = 2, TN = 4, steps = BK / MmaInt8::kStepBytes;
    __shared__ __align__(16) uint8_t As[2][BM * stride];
    __shared__ __align__(16) uint8_t Bs[BN * stride];

    const int lane = threadIdx.x % kWave;
    const int warp = threadIdx.x / kWave;
    const int half = warp / 8;
    const int wm = (warp % 8) / 2;
    const int wn = warp % 2;

    // Two block-rows retain the four-M-tile B locality of the single-M kernel.
    const int bid = blockIdx.y * gridDim.x + blockIdx.x;
    const int per_group = 2 * gridDim.x;
    const int group = bid / per_group;
    const int within = bid % per_group;
    const int group_rows = min(2, static_cast<int>(gridDim.y) - group * 2);
    const int pair = group * 2 + within % group_rows;
    const int n0 = (within / group_rows) * BN;
    const int m_pair = pair * 2 * BM;
    const int m0 = m_pair + half * BM;
    const bool second_tile = m_pair + BM < M;

    MmaInt8::Acc acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = MmaInt8::zero();

    TileStager<BM, BK, 512, MmaInt8::kLdsPad, true> sa0, sa1;
    TileStager<BN, BK, 512, MmaInt8::kLdsPad> sb;
    const int64_t tile0 = static_cast<int64_t>(pair * 2) * (K / BK);
    const int64_t tile1 = tile0 + K / BK;
    sa0.load_contiguous(A + tile0 * BM * BK);
    if (second_tile) sa1.load_contiguous(A + tile1 * BM * BK);
    sb.load(B, n0, N, 0, K);
    sa0.store(As[0]);
    if (second_tile) sa1.store(As[1]);
    sb.store(Bs);
    __syncthreads();

    const int row = frag_row(lane);
    for (int kb = 0; kb < K; kb += BK) {
        const int next = kb + BK;
        const bool has_next = next < K;
        if (has_next) {
            sa0.load_contiguous(A + (tile0 + next / BK) * BM * BK);
            if (second_tile)
                sa1.load_contiguous(A + (tile1 + next / BK) * BM * BK);
            sb.load(B, n0, N, next, K);
        }
        // An absent second tile must participate in barriers, but not read LDS.
        if (m0 < M) {
            MmaInt8::Frag af[2][TM], bf[2][TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                af[0][i] = MmaInt8::load(As[half], wm * TM * 16 + i * 16 + row,
                                        0, stride, lane);
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                bf[0][j] = MmaInt8::load(Bs, wn * TN * 16 + j * 16 + row,
                                        0, stride, lane);
            #pragma unroll
            for (int kk = 0; kk < steps; ++kk) {
                const int cur = kk & 1, nxt = cur ^ 1;
                if (kk + 1 < steps) {
                    const int kbyte = (kk + 1) * MmaInt8::kStepBytes;
                    #pragma unroll
                    for (int i = 0; i < TM; ++i)
                        af[nxt][i] = MmaInt8::load(
                            As[half], wm * TM * 16 + i * 16 + row, kbyte, stride, lane);
                    #pragma unroll
                    for (int j = 0; j < TN; ++j)
                        bf[nxt][j] = MmaInt8::load(
                            Bs, wn * TN * 16 + j * 16 + row, kbyte, stride, lane);
                }
                #pragma unroll
                for (int i = 0; i < TM; ++i)
                    #pragma unroll
                    for (int j = 0; j < TN; ++j)
                        acc[i][j] = MmaInt8::mma(af[cur][i], bf[cur][j], acc[i][j]);
            }
        }
        if (has_next) {
            __syncthreads();
            sa0.store(As[0]);
            if (second_tile) sa1.store(As[1]);
            sb.store(Bs);
            __syncthreads();
        }
    }

    // Keep channel operands in registers across output rows. Preserve the
    // producer's (acc * row_scale) * channel_scale, followed by a separate add.
    float col_scale[TN], col_bias[TN];
    const int col_lane = acc_col(lane);
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int col = n0 + wn * TN * 16 + j * 16 + col_lane;
        if constexpr (std::is_same_v<Epi, EpiRowwiseNoBias>) {
            col_scale[j] = col < N ? epi.scale_b[col] : 0.0f;
        } else {
            col_scale[j] = col < N ? epi.scale_b[col * epi.scale_b_stride] : 0.0f;
            col_bias[j] = col < N && epi.bias
                ? load_scalar(epi.bias, epi.bias_code, col) : 0.0f;
        }
    }
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int r = m0 + wm * TM * 16 + i * 16 + acc_row(lane, e);
            if (r >= M) continue;
            const float row_scale = epi.scale_a[r];
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int col = n0 + wn * TN * 16 + j * 16 + col_lane;
                if (col >= N) continue;
                float value = MmaInt8::get(acc[i][j], e) * row_scale;
                value *= col_scale[j];
                if constexpr (!std::is_same_v<Epi, EpiRowwiseNoBias>)
                    value += col_bias[j];
                C[static_cast<int64_t>(r) * ldc + col] = static_cast<__bf16>(value);
            }
        }
    }
}

// Eight lanes reduce one output column's 128-byte K slice. Adjacent columns
// occupy adjacent slices in packed B; every weight load serves all M rows.
template <int ROWS, typename OutT, typename Epi>
__global__ __launch_bounds__(256) void gemv_int8_b_tiled_kernel(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
    OutT* __restrict__ C, int M, int N, int K, int ldc, Epi epi) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    constexpr int CHUNK = 4096;
    __shared__ __align__(16) int8_t As[ROWS * CHUNK];
    const int lane = threadIdx.x % 32;
    const int col = blockIdx.x * 32 + (threadIdx.x / 32) * 4 + lane / 8;
    const int k16 = (lane % 8) * 16;
    int acc[ROWS] = {};
    for (int k0 = 0; k0 < K; k0 += CHUNK) {
        const int kc = min(CHUNK, K - k0);
        __syncthreads();
        for (int i = threadIdx.x * 16; i < M * kc; i += 256 * 16) {
            const int m = i / kc, k = k0 + i % kc;
            const int64_t src = static_cast<int64_t>(k / 128) * 128 * 128 + m * 128 + k % 128;
            *reinterpret_cast<uint4*>(As + m * CHUNK + i % kc) =
                *reinterpret_cast<const uint4*>(A + src);
        }
        __syncthreads();
        for (int k = k16; k < kc; k += 128) {
            const uint4 bv = PackedInt8B{B, K}.chunk(col, k0 + k);
            #pragma unroll
            for (int m = 0; m < ROWS; ++m) {
                if (m < M) {
                    const uint4 av = *reinterpret_cast<const uint4*>(As + m * CHUNK + k);
                    acc[m] = dot4_i8(av.x, bv.x, acc[m]);
                    acc[m] = dot4_i8(av.y, bv.y, acc[m]);
                    acc[m] = dot4_i8(av.z, bv.z, acc[m]);
                    acc[m] = dot4_i8(av.w, bv.w, acc[m]);
                }
            }
        }
    }
    epi.init();
    #pragma unroll
    for (int m = 0; m < ROWS; ++m) {
        if (m < M) {
            int sum = acc[m];
            sum += __shfl_xor(sum, 1);
            sum += __shfl_xor(sum, 2);
            sum += __shfl_xor(sum, 4);
            if (lane % 8 == 0)
                C[static_cast<int64_t>(m) * ldc + col] = static_cast<OutT>(epi(m, col, static_cast<float>(sum)));
        }
    }
}

// gfx12 large-M kernel: one 32-wave block per WGP, 256x256x64 block tile, 32x64
// wave tiles. Same tiled-A input and bf16 NoBias epilogue as the dual-M kernel.
//
// Why it is faster than the dual-M kernel on gfx12 (R9700 sweep, H3 qkv shape):
// a wave's four B fragments serve two A fragments (0.75 LDS fragment loads per
// WMMA instead of 1), the whole WGP shares one 256x256 output tile so each K
// byte is staged once per 256 rows, and the tile stores are issued against an
// explicit s_wait_loadcnt so the next K-block's global loads really stay in
// flight behind the LDS store of the current one (the compiler otherwise drains
// loadcnt at the store). 1590 ops/CU/clk vs 1433 (peak 2048), bit-exact.
//
// Pipeline. K-block t computes from LDS buffer t & 1 in four 16-byte k-steps of
// 8 WMMAs each; the fragments of step kk + 3 are loaded at step kk, so the ones
// of the next tile follow the barrier at step 1. Step 0 stores tile t + 1 (held
// in registers since the previous K-block) into the other buffer and refills the
// registers with tile t + 2. The stores of t + 1 may go anywhere in K-block t
// because buffer (t + 1) & 1 was last read before K-block t - 1's barrier.
//
// Staging. Thread tid stages one 16-byte chunk per operand per K-block: row
// tid / 4, k-chunk tid % 4. LDS rows are 64 bytes; chunks are XOR-swizzled by
// (row / 4) % 4 and their two 8-byte halves swapped on rows with bit 1 set, which
// together make both the b64 stores and the fragment loads bank-conflict free.
// The fragment side undoes this with per-lane constants (fsw, fhalf): fragment
// base rows are multiples of 16, so a lane's swizzle terms never change.
//
// The last K-block stores a stale tile into the dead buffer and its lookahead
// reads it; the refill re-reads the last K-block. All harmless, and it keeps the
// loop branch-free so the compiler's counter tracking stays straight-line.
template <int V>
using IC = std::integral_constant<int, V>;

// R9700, ROCm 10.1 M/N/K sweep against the dual-M kernel: ahead by 7-10% on the
// model rows (M 1024-41713, N >= 5376), the gain growing with M and N and
// shrinking with K. One block of 32 waves fills a WGP, so a grid under a few
// rounds of the device pays its tail in full: grids of 40-80 blocks lost 2-6%
// and 16 blocks 18%, while 160+ blocks never lost. Below 1024 rows the single-M
// kernel's 128-row tiles waste less of the last tile.
constexpr int kBigTiledMinM = 1024;
constexpr int kBigTiledMinRounds = 5;

// BT: B is resident in this kernel's own tile order, [N / 256][K / 64][256][64],
// so a block's K-block is one contiguous 16 KiB run and thread tid stages its
// bytes tid * 16 .. + 16. Requires N % 256 == 0 (no column clamp).
// Epi is EpiRowwise or EpiRowwiseNoBias; the channel scale and bias are read
// once per column and held in registers across the tile's rows.
template <bool BT, typename Epi, typename OutT>
__global__ __launch_bounds__(1024, 8) void gemm_int8_big_tiled_kernel(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
    OutT* __restrict__ C, int M, int N, int K, int ldc, Epi epi) {
#if defined(COMFY_MMA_GFX12)
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    constexpr bool kBias = !std::is_same_v<Epi, EpiRowwiseNoBias>;
    constexpr int BM = 256, BN = 256, BK = 64;
    constexpr int TM = 2, TN = 4, steps = BK / MmaInt8::kStepBytes;
    constexpr int WARPS_N = BN / (TN * 16);
    constexpr int kBufBytes = BM * BK, kBOff = 2 * kBufBytes;
    __shared__ __align__(16) uint8_t smem[4 * kBufBytes];  // A0 A1 B0 B1
    auto As = [&](int buf) { return smem + buf * kBufBytes; };
    auto Bs = [&](int buf) { return smem + kBOff + buf * kBufBytes; };

    const int lane = threadIdx.x % kWave;
    const int warp = threadIdx.x / kWave;
    const int wm = warp / WARPS_N;
    const int wn = warp % WARPS_N;

    // Supertile walk: 4 M-tiles x all N-tiles per supertile, N fastest, so the
    // blocks resident together share B tiles in L2. The row count only moves
    // DRAM traffic, not ops/clock: 8 rows cut it a third where B outruns the
    // MALL (N 21504) and raised it by half where B fits (N 5376, K 7168), a
    // wash in TOPS either way.
    constexpr int SP = 4;
    const int bid = blockIdx.y * gridDim.x + blockIdx.x;
    const int ntiles = gridDim.x, mtiles = gridDim.y;
    const int rows_full = SP * ntiles;
    const int srow = bid / rows_full;
    const int in_row = bid % rows_full;
    const int sp_rows = min(SP, mtiles - srow * SP);
    const int mt = srow * SP + in_row % sp_rows;
    const int n0 = (in_row / sp_rows) * BN;
    const int m0 = mt * BM;
    const int64_t tile0 = static_cast<int64_t>(mt * 2) * (K / 128);
    const bool second_tile = m0 + 128 < M;
    // A wave whose 32 rows all lie past M still stages tiles and joins barriers
    // but issues no MMAs; its accumulators are never stored. Wave-uniform.
    const bool live = m0 + wm * TM * 16 < M;

    MmaInt8::Acc acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = MmaInt8::zero();

    // Rows past M and columns past N are clamped onto valid memory instead of
    // zeroed: an output element depends only on its own A row and B row, and the
    // epilogue never stores them. Unconditional loads keep the loop branch-free.
    const int sr = threadIdx.x / 4, skc = threadIdx.x % 4;
    const uint8_t* a_src =
        A + (tile0 + (sr < 128 || !second_tile ? 0 : K / 128)) * (128 * 128) + (sr % 128) * 128 + skc * 16;
    // Tiled B: tile (n0 / BN, kb / BK) starts at (n0 / BN) * K * BN + kb * BN.
    const uint8_t* b_src = BT
        ? B + static_cast<int64_t>(n0 / BN) * K * BN + threadIdx.x * 16
        : B + static_cast<int64_t>(min(n0 + sr, N - 1)) * K + skc * 16;
    const int s_off = sr * BK + ((skc ^ ((sr >> 2) & 3)) << 4);
    const uint32_t lds_lo = (sr >> 1) & 1 ? 8u : 0u, lds_hi = 8u - lds_lo;
    uint4 ra, rb;
    // The loads are inline asm so the compiler cannot wait for them early; the
    // store consumes their registers inside the same asm as the s_wait_loadcnt.
    auto gload = [](const uint8_t* p) {
        uint4 v;
        asm volatile("global_load_b128 %0, %1, off" : "=v"(v) : "v"(p) : "memory");
        return v;
    };
    auto issue_a = [&](int kb) { ra = gload(a_src + (kb / 128) * (128 * 128) + (kb % 128)); };
    auto issue_b = [&](int kb) { rb = gload(b_src + (BT ? kb * BN : kb)); };
    // CNT = number of younger loads that may still be in flight.
    auto gstore = [lds_lo, lds_hi](auto cnt, uint8_t* dst, const uint4& v) {
        const uint32_t base = static_cast<uint32_t>(reinterpret_cast<size_t>(dst));
        const uint32_t a_lo = base + lds_lo, a_hi = base + lds_hi;
        const uint2 lo = make_uint2(v.x, v.y), hi = make_uint2(v.z, v.w);
        asm volatile("s_wait_loadcnt %4\n\tds_store_b64 %0, %2\n\tds_store_b64 %1, %3"
                     :
                     : "v"(a_lo), "v"(a_hi), "v"(lo), "v"(hi), "i"(decltype(cnt)::value)
                     : "memory");
    };
    auto store_a = [&](auto cnt, int buf) { gstore(cnt, As(buf) + s_off, ra); };
    auto store_b = [&](auto cnt, int buf) { gstore(cnt, Bs(buf) + s_off, rb); };
    auto lds_barrier = [] {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
    };

    const int row = frag_row(lane);
    const int fsw = (row >> 2) & 3;
    const int fhalf = 8 * (((row >> 1) & 1) ^ (lane / 16));
    MmaInt8::Frag af[steps][TM], bf[steps][TN];
    auto load_step = [&](auto bufc, auto kkc) {
        constexpr int buf = decltype(bufc)::value, kk = decltype(kkc)::value;
        const int k_off = ((kk ^ fsw) << 4) + fhalf;
        #pragma unroll
        for (int i = 0; i < TM; ++i)
            af[kk][i] = load_frag_b64(As(buf), wm * TM * 16 + i * 16 + row, k_off, BK);
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            bf[kk][j] = load_frag_b64(Bs(buf), wn * TN * 16 + j * 16 + row, k_off, BK);
    };
    auto mma_step = [&](int kk) {
        #pragma unroll
        for (int i = 0; i < TM; ++i)
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                acc[i][j] = MmaInt8::mma(af[kk][i], bf[kk][j], acc[i][j]);
    };

    const int nk = K / BK;  // >= 2: K % 128 == 0
    // B rows lie K bytes apart, so when K is a multiple of 4 KiB every row of a
    // step shares its low address bits, and with the grid in lockstep so does
    // every resident block: all of them queue on one L2 slice (gfx1200, K 16384:
    // 61 TOPS against 121 at K 16640). Stagger those blocks by 256 bytes.
    // Other row strides and packed B already spread the loads across slices;
    // walking K in order lets neighbouring blocks reuse the same tiles.
    const int rot = !BT && K % 4096 == 0 ? (bid % 16) * (256 / BK) % nk : 0;
    auto kb_at = [&](int t) { const int s = t + rot; return (s < nk ? s : s - nk) * BK; };
    issue_a(kb_at(0)); issue_b(kb_at(0));
    store_a(IC<1>{}, 0); store_b(IC<0>{}, 0);
    issue_a(kb_at(1)); issue_b(kb_at(1));
    lds_barrier();
    if (live) {
        load_step(IC<0>{}, IC<0>{});
        load_step(IC<0>{}, IC<1>{});
        load_step(IC<0>{}, IC<2>{});
    }

    // Compiled twice so the live path keeps its exact instruction stream; the
    // dead path only stages tiles and joins barriers.
    auto kblock = [&](int t, auto bufc, auto livec) {
        constexpr int buf = decltype(bufc)::value;
        constexpr bool compute = decltype(livec)::value;
        constexpr auto nb = IC<buf ^ 1>{};
        const int kb_issue = kb_at(min(t + 2, nk - 1));
        // The scheduler would otherwise drift WMMAs across the barrier and the
        // loads; pin the step structure.
        __builtin_amdgcn_sched_barrier(0);
        store_a(IC<1>{}, buf ^ 1);  // tile t + 1; its B load is the one younger load
        issue_a(kb_issue);
        store_b(IC<1>{}, buf ^ 1);  // ... and now the A just issued is
        issue_b(kb_issue);
        if constexpr (compute) { load_step(bufc, IC<3>{}); mma_step(0); }
        __builtin_amdgcn_sched_barrier(0);
        lds_barrier();
        if constexpr (compute) { load_step(nb, IC<0>{}); mma_step(1); }
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (compute) { load_step(nb, IC<1>{}); mma_step(2); }
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (compute) { load_step(nb, IC<2>{}); mma_step(3); }
        __builtin_amdgcn_sched_barrier(0);
    };
    auto kloop = [&](auto livec) {
        for (int t = 0; t < nk; t += 2) {
            kblock(t, IC<0>{}, livec);
            if (t + 1 < nk) kblock(t + 1, IC<1>{}, livec);
        }
    };
    if (live) kloop(std::true_type{}); else kloop(std::false_type{});
    // The stale tail refills are still in flight and the compiler believes their
    // registers are free: drain them while they are pinned as asm inputs.
    asm volatile("s_wait_loadcnt 0x0" ::"v"(ra.x), "v"(ra.y), "v"(ra.z), "v"(ra.w),
                 "v"(rb.x), "v"(rb.y), "v"(rb.z), "v"(rb.w) : "memory");

    float col_scale[TN], col_bias[TN];
    const int col_lane = acc_col(lane);
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int col = n0 + wn * TN * 16 + j * 16 + col_lane;
        if constexpr (kBias) {
            col_scale[j] = col < N ? epi.scale_b[col * epi.scale_b_stride] : 0.0f;
            col_bias[j] = col < N && epi.bias ? load_scalar(epi.bias, epi.bias_code, col) : 0.0f;
        } else {
            col_scale[j] = col < N ? epi.scale_b[col] : 0.0f;
            col_bias[j] = 0.0f;
        }
    }
    if constexpr (std::is_same_v<OutT, __bf16>) {
        if (m0 + BM <= M && n0 + BN <= N) {
            // Full tile: every result is in bounds, so store bf16 pairs. Element e of
            // lane l is row e + 8 * (l / 16), column l % 16 (acc_row/acc_col): lanes
            // l, l ^ 1 hold the two columns of a pair. For rows (2p, 2p + 1) the even
            // lane sends its row-(2p + 1) value and the odd lane its row-2p value
            // across the pair (DPP quad_perm [1,0,3,2]); the even lane then owns the
            // pair of row 2p and the odd lane that of row 2p + 1. Rounding is RNE on
            // the bits; an int32 accumulator times finite scales is never NaN.
            const int odd = lane & 1;
            const int row0 = m0 + wm * TM * 16 + 8 * (lane / 16) + odd;
            float row_scale[TM][8];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int e = 0; e < 8; ++e) row_scale[i][e] = epi.scale_a[row0 - odd + i * 16 + e];
            auto rne = [](float f) {
                const uint32_t u = __float_as_uint(f);
                return u + 0x7fffu + ((u >> 16) & 1u);
            };
            __bf16* cp = C + static_cast<int64_t>(row0) * ldc + n0 + wn * TN * 16 + (lane & 14);
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    __bf16* rowp = cp + static_cast<int64_t>(i * 16 + 2 * p) * ldc;
                    #pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        float f0 = MmaInt8::get(acc[i][j], 2 * p) * row_scale[i][2 * p] * col_scale[j];
                        float f1 = MmaInt8::get(acc[i][j], 2 * p + 1) * row_scale[i][2 * p + 1] * col_scale[j];
                        if constexpr (kBias) { f0 += col_bias[j]; f1 += col_bias[j]; }
                        const uint32_t v0 = rne(f0), v1 = rne(f1);
                        const uint32_t send = odd ? v0 : v1;
                        const uint32_t recv = static_cast<uint32_t>(
                            __builtin_amdgcn_update_dpp(0, static_cast<int>(send), 0xB1, 0xF, 0xF, true));
                        const uint32_t lo = odd ? recv : v0, hi = odd ? v1 : recv;
                        *reinterpret_cast<uint32_t*>(rowp + j * 16) = __builtin_amdgcn_perm(hi, lo, 0x07060302u);
                    }
                }
            }
            return;
        }
    }
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int r = m0 + wm * TM * 16 + i * 16 + acc_row(lane, e);
            if (r >= M) continue;
            const float row_scale = epi.scale_a[r];
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int col = n0 + wn * TN * 16 + j * 16 + col_lane;
                if (col >= N) continue;
                float value = MmaInt8::get(acc[i][j], e) * row_scale;
                value *= col_scale[j];
                if constexpr (kBias) value += col_bias[j];
                C[static_cast<int64_t>(r) * ldc + col] = static_cast<OutT>(value);
            }
        }
    }
#else
    // gfx11 has neither this WMMA layout nor s_wait_loadcnt; never dispatched there.
    __builtin_trap();
#endif
}

}  // namespace comfy::hip_backend

template <typename OutT, typename Epi>
static void launch_int8_tiled_a(
    const uint8_t* A, const uint8_t* B, OutT* C,
    int M, int N, int K, int ldc, Epi epi, bool b_tiled, hipStream_t stream) {
    using namespace comfy::hip_backend;
    constexpr int BM = 128, BN = 128, BKB = 128;
    if (b_tiled) {
        if (M <= 8 && N <= 1024 && K > 1024) {
            // Narrow weights have few column tiles. For long K, split rows across blocks.
            gemv_int8_kernel<true><<<dim3(N / 8, M), dim3(16, 8), 0, stream>>>(
                A, B, C, M, N, K, ldc, epi);
        } else if (M <= 4) {
            dim3 grid(N / 32);
            if (M == 1)
                gemv_int8_b_tiled_kernel<1><<<grid, 256, 0, stream>>>(A, B, C, M, N, K, ldc, epi);
            else if (M <= 2)
                gemv_int8_b_tiled_kernel<2><<<grid, 256, 0, stream>>>(A, B, C, M, N, K, ldc, epi);
            else
                gemv_int8_b_tiled_kernel<4><<<grid, 256, 0, stream>>>(A, B, C, M, N, K, ldc, epi);
        } else {
            dim3 grid(N / 256, (M + 255) / 256);
            if constexpr (std::is_same_v<OutT, __bf16>) {
                const int blocks = grid.x * grid.y, wgps = device_wgp_count();
                if (M >= 256 && (blocks <= wgps / 2 || (M >= 512 && blocks <= 2 * wgps))) {
                    dim3 small_grid(N / BN, (M + 2 * BM - 1) / (2 * BM));
                    gemm_int8_dual_m_tiled_kernel<Epi, PackedInt8B>
                        <<<small_grid, 512, 0, stream>>>(A, PackedInt8B{B, K}, C, M, N, K, ldc, epi);
                    return;
                }
            }
            gemm_int8_big_tiled_kernel<true, Epi><<<grid, 1024, 0, stream>>>(A, B, C, M, N, K, ldc, epi);
        }
        return;
    }
    if constexpr (std::is_same_v<OutT, __bf16>) {
        dim3 grid((N + 255) / 256, (M + 255) / 256);
        if (use_nonduplicated_wmma_schedule() && M >= kBigTiledMinM &&
            grid.x * grid.y >= kBigTiledMinRounds * device_wgp_count()) {
            gemm_int8_big_tiled_kernel<false, Epi><<<grid, 1024, 0, stream>>>(
                A, B, C, M, N, K, ldc, epi);
            return;
        }
    }
    if constexpr (std::is_same_v<OutT, __bf16>) {
        // ROCm 10.1 gfx12 sweep: sharing B wins the model-sized rows; retain
        // single-M below this one cut to avoid underfilled dual-M blocks.
        if (use_nonduplicated_wmma_schedule() && M >= 1024) {
            dim3 grid((N + BN - 1) / BN, (M + 2 * BM - 1) / (2 * BM));
            gemm_int8_dual_m_tiled_kernel<<<grid, 512, 0, stream>>>(
                A, B, C, M, N, K, ldc, epi);
            return;
        }
    }
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_wmma_kernel<MmaInt8, Epi, OutT, BM, BN, BKB,
                     4, 2, 2, 4, const uint8_t*, true, true>
        <<<grid, 256, 0, stream>>>(A, B, C, M, N, K, ldc, epi);
}

extern "C" void launch_int8_gemm_a_tiled128_kernel(
    const void* a, const void* b, void* c,
    const void* scale_a, const void* scale_b, int scale_b_stride,
    const void* bias, int bias_code,
    int M, int N, int K, int ldc, int out_code, int b_tiled,
    hipStream_t stream) {

    using namespace comfy::hip_backend;
    if (M == 0) return;
    if (N <= 0 || K <= 0 || K % 128 != 0 || ldc < N) {
        throw std::runtime_error(
            "tiled-A int8 GEMM: requires positive N/K, K divisible by 128 and ldc>=N");
    }
    if (b_tiled && N % 256 != 0) {
        throw std::runtime_error(
            "tiled-A int8 GEMM: a tiled B requires N divisible by 256");
    }
    EpiRowwise epi{
        static_cast<const float*>(scale_a), static_cast<const float*>(scale_b),
        scale_b_stride, bias, bias_code};
    const uint8_t* A = static_cast<const uint8_t*>(a);
    const uint8_t* B = static_cast<const uint8_t*>(b);
    if (out_code == kBF16) {
        if (epi.bias == nullptr && epi.scale_b_stride == 1) {
            EpiRowwiseNoBias no_bias{epi.scale_a, epi.scale_b};
            launch_int8_tiled_a(
                A, B, static_cast<__bf16*>(c), M, N, K, ldc, no_bias,
                b_tiled != 0, stream);
        } else {
            launch_int8_tiled_a(
                A, B, static_cast<__bf16*>(c), M, N, K, ldc, epi,
                b_tiled != 0, stream);
        }
    } else if (out_code == kF16) {
        launch_int8_tiled_a(
            A, B, static_cast<__half*>(c), M, N, K, ldc, epi,
                b_tiled != 0, stream);
    } else if (out_code == kF32) {
        launch_int8_tiled_a(
            A, B, static_cast<float*>(c), M, N, K, ldc, epi,
                b_tiled != 0, stream);
    } else {
        throw std::runtime_error("tiled-A int8 GEMM: unsupported output dtype");
    }
}
