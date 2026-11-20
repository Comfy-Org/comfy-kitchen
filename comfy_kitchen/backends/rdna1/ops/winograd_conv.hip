// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Winograd F(2x2, 3x3) convolution on packed fp16 FMAs, for [T, 3, 3] filters at stride 1
// and no dilation over H and W. A 2x2 tile of outputs takes 16 products per input channel
// instead of 36: with d the 4x4 input patch under the tile and g the filter,
//   Y = At [ (G g Gt) . (Bt d B) ] A
// where . is elementwise. The implicit GEMM (conv3d_fp16.hip) spends 9 products an output;
// MIOpen's fp32 Winograd kernels were 1.4-2.7x faster than it on gfx1010.
//
// Everything but the filter transform is fused into one kernel, so no transformed input or
// product buffer exists (as a GEMM over them, their traffic cost more than the products
// saved under 512 channels). A block owns 64 tiles by 32 output channels: with 32 tiles,
// re-reading the 16 components of the transformed filter for every block bound the kernel. x is a scaled
// fp16 copy blocked by 16 channels, [N, C / 16, D, H, W, 16], so the 16 channels of a stage
// lie 32 bytes a position and a patch row is one contiguous run. Per stage the block loads
// the input rows under its tiles into LDS with coalesced 16-byte loads (gathering each
// tile's patch from global memory instead touched a cache line per lane and cost more than
// the products), transforms the patches into LDS, copies the matching 16 KiB of
// transformed filter there, and accumulates 16 x 4 products per K word and thread with
// v_pk_fma_f16; the fp16 chains fold into fp32 every 32 products, as in SimtF16Packed. The
// threads then apply the output transform, the scale and the bias and write NCDHW.
//
// A depth of T filter taps is T times the input channels: the transforms are spatial only.
#include "../epilogue.h"

#include <algorithm>

namespace comfy::hip_backend {

typedef _Float16 wino_half2 __attribute__((ext_vector_type(2)));

constexpr int kWinoWords = 8;       // K words (16 channels) per stage
constexpr int kWinoTiles = 64;      // tiles per block
constexpr int kWinoThreads = 512;
constexpr int kWinoCols = 32;       // output channels per block
constexpr int kWinoFoldStages = 4;  // 32 products per fp16 chain between folds
constexpr int kWinoStageWords = kWinoWords * 16 * kWinoCols;  // transformed filter per stage and block
// Input vectors (8 channels) a block stages, two a position: its 64 tiles lie 8, 16 or 32 to
// a row (kWinoMinTws..kWinoMaxTws), at most 6 rows of 66 positions. A row of 64 or of 4
// would stage 520 positions, past what the 64 KiB of LDS leaves.
constexpr int kWinoMinTws = 3, kWinoMaxTws = 5;
constexpr int kWinoRawVecs = 6 * 66 * 2;
constexpr int kWinoRawPer = (kWinoRawVecs + kWinoThreads - 1) / kWinoThreads;
constexpr int kWinoBPer = kWinoStageWords / 4 / kWinoThreads;

struct WinoDims {
    int N, D, H, W;  // input extents
    int C;           // input channels, a multiple of 16
    int K, T;        // output channels, depth taps
    int Z, P, Q;     // output extents
    int sd, pd, ph, pw;
    int tws;         // log2 of the tiles a block takes along W; the rest of its 64 stack along H
};

// U = G g Gt along one axis: G = [1 0 0; .5 .5 .5; .5 -.5 .5; 0 0 1]
__forceinline__ __device__ void wino_filter_axis(float g0, float g1, float g2, float u[4]) {
    const float s = 0.5f * (g0 + g2), h = 0.5f * g1;
    u[0] = g0, u[1] = s + h, u[2] = s - h, u[3] = g2;
}

// The filter [K, C, T, 3, 3] of dtype `code` as the main kernel's LDS image: for each block
// of 32 output channels and each stage, word (w * 16 + i) * 32 + k holds component i of
// channels (2w, 2w + 1) of the stage for output channel k, as an fp16 pair. Channels past C
// (the copy's zero padding) and output channels past K are zeros. grid = (stages, K blocks).
__global__ __launch_bounds__(256) void wino_filter_kernel(
    const void* __restrict__ w, int code, uint32_t* __restrict__ u, int C, int Cp, int K, int T) {
    __shared__ __align__(16) uint32_t Ls[kWinoStageWords];
    const int tid = threadIdx.x, kw = tid % kWinoWords, kl = tid / kWinoWords;
    const int k = blockIdx.y * kWinoCols + kl;
    float uv[2][16];
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        const int slot = blockIdx.x * 16 + 2 * kw + e;
        const int t = slot / Cp, c = slot - t * Cp;
        float g[9];
#pragma unroll
        for (int i = 0; i < 9; ++i) {
            g[i] = (k < K && c < C)
                       ? load_scalar(w, code, ((static_cast<int64_t>(k) * C + c) * T + t) * 9 + i) : 0.0f;
        }
        float rows[3][4];
#pragma unroll
        for (int r = 0; r < 3; ++r) wino_filter_axis(g[r * 3], g[r * 3 + 1], g[r * 3 + 2], rows[r]);
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            float col[4];
            wino_filter_axis(rows[0][s], rows[1][s], rows[2][s], col);
#pragma unroll
            for (int a = 0; a < 4; ++a) uv[e][a * 4 + s] = col[a];
        }
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const wino_half2 h = {static_cast<_Float16>(uv[0][i]), static_cast<_Float16>(uv[1][i])};
        Ls[(kw * 16 + i) * kWinoCols + kl] = __builtin_bit_cast(uint32_t, h);
    }
    __syncthreads();
    uint32_t* dst = u + (static_cast<int64_t>(blockIdx.y) * gridDim.x + blockIdx.x) * kWinoStageWords;
#pragma unroll
    for (int i = 0; i < kWinoStageWords / 4 / 256; ++i) {
        const int c = tid + i * 256;
        *reinterpret_cast<uint4*>(dst + c * 4) = *reinterpret_cast<const uint4*>(Ls + c * 4);
    }
}

// A quad lane's 32-bit value from quad lane perm[lane % 4], perm packed two bits a lane.
template <int kPerm>
__forceinline__ __device__ uint32_t wino_quad(uint32_t v) {
    return static_cast<uint32_t>(__builtin_amdgcn_mov_dpp(static_cast<int>(v), kPerm, 0xf, 0xf, false));
}

// x is the scaled channel-blocked fp16 copy, u the transformed filter, bias and out of dtype `code`,
// out [N, K, Z, P, Q] contiguous. out = acc * acc_scale[0] + bias.
// grid = (N * Z * tile blocks, K blocks).
//
// The product threads each own one transform component of 8 tiles by 8 output channels:
// per K word two 4-tile and two 4-channel LDS vectors for 64 FMAs, the GEMM's 8x8 ratio.
// Owning all 16 components of fewer tiles instead took 8 vectors per 64 FMAs and ran a
// third as fast, LDS-bound. Lane bits 0-1 and wave bit 0 pick the tile group, lane bit 2
// and wave bit 1 the channel group, and lane bits 3-4 with wave bits 2-3 the component, so
// each 8-lane group of a read shares its component and its vectors fall in distinct banks.
// The components meet through LDS for the output transform.
// The transform threads: lane r + 4 * t of wave v reads row r of the patch of tile
// 8 * (v % 8) + t from the staged rows, 8 channels (half v / 8 of the stage) at each of its 4
// columns, applies Bt along the row, and takes Bt down the rows from its quad with two DPP
// moves a word.
__global__ __launch_bounds__(kWinoThreads) void wino_conv_kernel(
    const uint4* __restrict__ x, const uint32_t* __restrict__ u, const void* __restrict__ bias,
    void* __restrict__ out, int code, const float* __restrict__ acc_scale, int stages, WinoDims g) {
    // As and Bs in one array, which the output transform reuses whole
    constexpr int kAWords = kWinoWords * 16 * kWinoTiles;
    __shared__ __align__(16) uint32_t Lds[kAWords + kWinoStageWords];
    uint32_t* const As = Lds;
    uint32_t* const Bs = Lds + kAWords;
    __shared__ __align__(16) uint4 Rs[kWinoRawVecs];  // the stage's input rows, 2 vectors a position

    const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
    const int quad = lane % 4;  // the staging thread's patch row

    const int th = (g.P + 1) / 2, tw = (g.Q + 1) / 2;
    const int tw_blk = 1 << g.tws, th_blk = kWinoTiles >> g.tws;
    const int gx = (tw + tw_blk - 1) / tw_blk, gy = (th + th_blk - 1) / th_blk;
    int rem = blockIdx.x;
    const int bx = rem % gx;
    rem /= gx;
    const int by = rem % gy;
    rem /= gy;
    const int z = rem % g.Z, n = rem / g.Z;
    const auto tile_at = [&](int t, int& ta, int& tb) {
        ta = by * th_blk + (t >> g.tws), tb = bx * tw_blk + (t & (tw_blk - 1));
        return ta < th && tb < tw;
    };

    // the input rows under the block's tiles: from (pi0, qi0), rows x cols positions
    const int rows = 2 * th_blk + 2, cols = 2 * tw_blk + 2;
    const int pi0 = 2 * by * th_blk - g.ph, qi0 = 2 * bx * tw_blk - g.pw;
    const int raw = rows * cols * 2;
    // transform role
    const int st = 8 * (wave % 8) + lane / 4, sh = wave / 8;
    const int sr = 2 * (st >> g.tws) + quad, sc = 2 * (st & (tw_blk - 1));
    const int cstages = g.C / 16;  // stages per depth tap
    // Bt down the rows: row a is X + sign * Y, X from quad lane (0, 1, 2, 1)[a], Y from (2, 2, 1, 3)[a]
    const _Float16 sgn = quad == 1 ? 1.0f16 : -1.0f16;
    const wino_half2 sign = {sgn, sgn};

    const uint32_t* ub = u + static_cast<int64_t>(blockIdx.y) * stages * kWinoStageWords;

    // Two stages in flight: with one block per CU, one stage of math did not cover the loads.
    uint4 ra0[kWinoRawPer], rb0[kWinoBPer], ra1[kWinoRawPer], rb1[kWinoBPer];
    auto load = [&](int s, uint4 (&ra)[kWinoRawPer], uint4 (&rb)[kWinoBPer]) {
        const int t = s / cstages;
        const int zi = z * g.sd - g.pd + t;
        const bool z_in = static_cast<unsigned>(zi) < static_cast<unsigned>(g.D);
        const int64_t plane = ((static_cast<int64_t>(n) * cstages + (s - t * cstages)) * g.D + zi) * g.H;
#pragma unroll
        for (int i = 0; i < kWinoRawPer; ++i) {
            const int e = tid + i * kWinoThreads, row = e / (2 * cols), col = e / 2 - row * cols;
            const int pi = pi0 + row, qi = qi0 + col;
            const bool in = e < raw && z_in && static_cast<unsigned>(pi) < static_cast<unsigned>(g.H) &&
                            static_cast<unsigned>(qi) < static_cast<unsigned>(g.W);
            ra[i] = in ? x[((plane + pi) * g.W + qi) * 2 + e % 2] : make_uint4(0, 0, 0, 0);
        }
        const uint32_t* us = ub + static_cast<int64_t>(s) * kWinoStageWords;
#pragma unroll
        for (int i = 0; i < kWinoBPer; ++i) rb[i] = *reinterpret_cast<const uint4*>(us + (tid + i * kWinoThreads) * 4);
    };
    // As: component i = 4a + b of K word w of tile t at [(w * 16 + i) * 64 + (t ^ 8a)]. The XOR
    // puts the four rows a quad writes in distinct banks (4-way conflicted without it) and
    // keeps every aligned run of 8 tiles contiguous for the product reads.
    auto store = [&](const uint4 (&ra)[kWinoRawPer], const uint4 (&rb)[kWinoBPer]) {
#pragma unroll
        for (int i = 0; i < kWinoRawPer; ++i) {
            const int e = tid + i * kWinoThreads;
            if (e < raw) Rs[e] = ra[i];
        }
#pragma unroll
        for (int i = 0; i < kWinoBPer; ++i) *reinterpret_cast<uint4*>(Bs + (tid + i * kWinoThreads) * 4) = rb[i];
        __syncthreads();
        uint4 p[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) p[q] = Rs[(sr * cols + sc + q) * 2 + sh];
        const uint32_t d[4][4] = {{p[0].x, p[0].y, p[0].z, p[0].w}, {p[1].x, p[1].y, p[1].z, p[1].w},
                                  {p[2].x, p[2].y, p[2].z, p[2].w}, {p[3].x, p[3].y, p[3].z, p[3].w}};
#pragma unroll
        for (int v = 0; v < 4; ++v) {  // the four K words of this thread's 8 channels
            wino_half2 c[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) c[q] = __builtin_bit_cast(wino_half2, d[q][v]);
            const wino_half2 e[4] = {c[0] - c[2], c[1] + c[2], c[2] - c[1], c[1] - c[3]};
            const int w = 4 * sh + v;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const uint32_t eb = __builtin_bit_cast(uint32_t, e[b]);
                const wino_half2 X = __builtin_bit_cast(wino_half2, wino_quad<0x64>(eb));  // (0, 1, 2, 1)
                const wino_half2 Y = __builtin_bit_cast(wino_half2, wino_quad<0xDA>(eb));  // (2, 2, 1, 3)
                As[(w * 16 + quad * 4 + b) * kWinoTiles + (st ^ quad << 3)] =
                    __builtin_bit_cast(uint32_t, __builtin_elementwise_fma(Y, sign, X));
            }
        }
    };

    // product role
    const int tg = (lane & 3) | (wave & 1) << 2, kg = (lane >> 2 & 1) | (wave >> 1 & 1) << 1;
    const int comp = (lane >> 3) | (wave >> 2) << 2;
    float acc[8][8];  // [tile][channel]
    wino_half2 pack[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = 0.0f, pack[i][j] = {};

    auto math = [&]() {
#pragma unroll 1
        for (int w = 0; w < kWinoWords; ++w) {
            const uint32_t* arow = As + (w * 16 + comp) * kWinoTiles + (tg ^ comp >> 2) * 8;
            const uint32_t* brow = Bs + (w * 16 + comp) * kWinoCols + kg * 8;
            const uint4 a0 = *reinterpret_cast<const uint4*>(arow), a1 = *reinterpret_cast<const uint4*>(arow + 4);
            const uint4 b0 = *reinterpret_cast<const uint4*>(brow), b1 = *reinterpret_cast<const uint4*>(brow + 4);
            const uint32_t aw[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const uint32_t bw[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    pack[i][j] = __builtin_elementwise_fma(__builtin_bit_cast(wino_half2, aw[i]),
                                                           __builtin_bit_cast(wino_half2, bw[j]), pack[i][j]);
                }
        }
    };
    auto fold = [&](int s) {
        if ((s + 1) % kWinoFoldStages == 0 || s + 1 == stages) {
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    acc[i][j] += static_cast<float>(pack[i][j].x) + static_cast<float>(pack[i][j].y);
                    pack[i][j] = {};
                }
        }
    };

    load(0, ra0, rb0);
    if (stages > 1) load(1, ra1, rb1);
    store(ra0, rb0);
    __syncthreads();
    for (int s = 0; s < stages; s += 2) {
        if (s + 2 < stages) load(s + 2, ra0, rb0);
        math();
        fold(s);
        if (s + 1 == stages) break;
        __syncthreads();  // every thread is done reading this stage
        store(ra1, rb1);
        __syncthreads();
        if (s + 3 < stages) load(s + 3, ra1, rb1);
        math();
        fold(s + 1);
        if (s + 2 == stages) break;
        __syncthreads();
        store(ra0, rb0);
        __syncthreads();
    }

    // The output transform, 8 output channels of the block at a time: every thread parks its
    // component of those channels in LDS (As and Bs, now free), then takes 4 (tile,
    // channel) pairs, reads their 16 components and applies At = [1 1 1 0; 0 1 -1 -1] along
    // both axes. Ys[(pair * 17 + component)]: the odd stride spreads a pair's reads over banks.
    float* Ys = reinterpret_cast<float*>(Lds);
    static_assert(sizeof(Lds) >= kWinoTiles * 8 * 17 * sizeof(float), "the transform's LDS");
    const float scale = acc_scale[0];
    const int64_t pq = static_cast<int64_t>(g.P) * g.Q;
#pragma unroll
    for (int round = 0; round < 4; ++round) {
        __syncthreads();  // the products' LDS reads, or the previous round's
        // this thread's channels kg * 8 + 2 * round, + 1 are the round's 2kg, 2kg + 1
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int j = 0; j < 2; ++j)
                Ys[((tg * 8 + i) * 8 + kg * 2 + j) * 17 + comp] = acc[i][2 * round + j];
        __syncthreads();
        const int t = tid / 8, c = tid % 8;  // one tile, one of the round's channels
        const int k = blockIdx.y * kWinoCols + (c / 2) * 8 + 2 * round + c % 2;
        int ta, tb;
        if (!tile_at(t, ta, tb) || k >= g.K) continue;
        float m[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) m[i] = Ys[(t * 8 + c) * 17 + i];
        float y[2][4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            y[0][q] = m[q] + m[4 + q] + m[8 + q];
            y[1][q] = m[4 + q] - m[8 + q] - m[12 + q];
        }
        const float b = bias ? load_scalar(bias, code, k) : 0.0f;
        const int64_t base = ((static_cast<int64_t>(n) * g.K + k) * g.Z + z) * pq;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            if (2 * ta + r >= g.P) continue;
            const int64_t o = base + static_cast<int64_t>(2 * ta + r) * g.Q + 2 * tb;
            store_scalar(out, code, o, (y[r][0] + y[r][1] + y[r][2]) * scale + b);
            if (2 * tb + 1 < g.Q) store_scalar(out, code, o + 1, (y[r][1] - y[r][2] - y[r][3]) * scale + b);
        }
    }
}

}  // namespace comfy::hip_backend

// conv3d of the scaled NDHWC fp16 x [N, D, H, W, Cp] (Cp a multiple of 16, channels past C
// zero) with w [K, C, T, 3, 3] of dtype w_code, stride (sd, 1, 1), zero padding, into the
// contiguous out [N, K, Z, P, Q] of dtype out_code with bias in that dtype or null. u is
// scratch for the transformed filter: ceil(K / 32) * (T * Cp / 16) * 4096 32-bit words.
// Returns false, having launched nothing, for a shape the kernel does not index.
extern "C" bool launch_fp16_winograd_conv_kernel(
    const void* x, const void* w, int w_code, const void* bias, void* out, int out_code, void* u,
    int N, int D, int H, int W, int Cp, int C, int K, int T, int Z, int P, int Q, int sd, int pd,
    int ph, int pw, const void* acc_scale, hipStream_t stream) {
    using namespace comfy::hip_backend;
    if (Cp % 16 != 0 || C > Cp || sd <= 0) {
        return false;
    }
    if (static_cast<int64_t>(N) * Z * P * Q == 0 || K == 0) {
        return true;
    }
    const int th = (P + 1) / 2, tw = (Q + 1) / 2;
    int tws = 0;
    while (tws < kWinoMaxTws && (1 << tws) < tw) ++tws;
    tws = std::max(tws, kWinoMinTws);
    const int tw_blk = 1 << tws, th_blk = kWinoTiles >> tws;
    const int64_t blocks = static_cast<int64_t>(N) * Z * ((th + th_blk - 1) / th_blk) *
                           ((tw + tw_blk - 1) / tw_blk);
    const int64_t stages = static_cast<int64_t>(T) * Cp / 16;
    const int kblocks = (K + kWinoCols - 1) / kWinoCols;
    if (blocks > INT_MAX || stages > 65535 || kblocks > 65535) {
        return false;
    }
    auto* ub = static_cast<uint32_t*>(u);
    hipLaunchKernelGGL(wino_filter_kernel, dim3(static_cast<unsigned>(stages), kblocks), dim3(256), 0,
                       stream, w, w_code, ub, C, Cp, K, T);
    const WinoDims g{N, D, H, W, Cp, K, T, Z, P, Q, sd, pd, ph, pw, tws};
    hipLaunchKernelGGL(wino_conv_kernel, dim3(static_cast<unsigned>(blocks), kblocks), dim3(kWinoThreads), 0,
                       stream, static_cast<const uint4*>(x), ub, bias, out, out_code,
                       static_cast<const float*>(acc_scale), static_cast<int>(stages), g);
    return true;
}
