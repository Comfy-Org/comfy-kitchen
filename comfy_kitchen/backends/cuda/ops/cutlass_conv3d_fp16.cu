/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// fp16-accumulate NDHWC conv3d (CUTLASS implicit GEMM) with bias and an optional
// residual in the epilogue. Zero padding only: callers pre-pad (group_norm_pad3d.cu).
// Same opt-in numerics as cutlass_gemm_fp16.cu; a missing residual is a stride-0 zero vector.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

struct Conv3dDims {
    int N, D, H, W, C;      // input NDHWC
    int K, T, R, S;         // filter KTRSC
    int Z, P, Q;            // output NZPQK
    int sd, sh, sw;         // strides
};

#ifdef COMFY_HAVE_CUTLASS

#include "cutlass/cutlass.h"
#include "cutlass/functional.h"
#include "cutlass/conv/kernel/default_conv3d_fprop_with_broadcast.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/epilogue/thread/linear_combination_residual_block.h"

namespace {

using half_t = cutlass::half_t;
using Layout = cutlass::layout::TensorNDHWC;

// Z = identity(acc + bias) + residual, computed in fp32, rounded once to fp16.
template <typename Acc>
using EpilogueOpT = cutlass::epilogue::thread::LinearCombinationResidualBlock<
    half_t, Acc, float, half_t, 8,
    cutlass::epilogue::thread::Identity, cutlass::plus, cutlass::epilogue::thread::Identity>;

// Acc = half_t is the fp16-accumulate kernel; Acc = float serves reductions
// deeper than kMaxFp16Depth, where one fp16 accumulator per output costs
// ~6 dB on the encoder's round trip.
template <int TBM, int TBN, int WM, int WN, int Stages, typename Acc>
struct Conv3dFp16 {
    static constexpr int kM = TBM, kN = TBN;
    using EpilogueOp = EpilogueOpT<Acc>;
    using Kernel = typename cutlass::conv::kernel::DefaultConv3dFpropWithBroadcast<
        half_t, Layout, half_t, Layout, half_t, Layout, Acc,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<TBM, TBN, 32>, cutlass::gemm::GemmShape<WM, WN, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>, EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages,
        cutlass::arch::OpMultiplyAdd, cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
    using Op = cutlass::conv::device::ImplicitGemmConvolution<Kernel>;

    static bool run(const half_t* x, const half_t* w, const half_t* bias,
                    const half_t* resid, bool resid_full, half_t* out,
                    const Conv3dDims& d, cudaStream_t stream) {
        const cutlass::Tensor5DCoord in_size(d.N, d.D, d.H, d.W, d.C);
        const cutlass::Tensor5DCoord filter_size(d.K, d.T, d.R, d.S, d.C);
        const cutlass::Tensor5DCoord out_size(d.N, d.Z, d.P, d.Q, d.K);
        cutlass::conv::Conv3dProblemSize problem(
            in_size, filter_size,
            cutlass::make_Coord(0, 0, 0), cutlass::make_Coord(d.sd, d.sh, d.sw), cutlass::make_Coord(1, 1, 1),
            out_size, cutlass::conv::Mode::kCrossCorrelation, 1, 1);
        const Layout lx = Layout::packed(in_size);
        const Layout lw = Layout::packed(filter_size);
        const Layout lo = Layout::packed(out_size);
        const Layout lr = resid_full ? lo : Layout(0, 0, 0, 0);  // every output row reads resid[0:K]

        // TensorRef holds non-const pointers; the kernel only reads x/w/resid.
        typename Op::Arguments args(
            problem,
            typename Kernel::TensorRefA(const_cast<half_t*>(x), lx),
            typename Kernel::TensorRefB(const_cast<half_t*>(w), lw),
            typename Kernel::TensorRefC(const_cast<half_t*>(resid), lr),
            typename Kernel::TensorRefC(out, lo),
            typename EpilogueOp::Params(1.0f, 1.0f), cutlass::conv::SplitKMode::kSerial,
            const_cast<half_t*>(bias), nullptr, /*ldr=*/0, /*ldt=*/d.K);

        Op op;
        if (op.can_implement(args) != cutlass::Status::kSuccess) return false;
        if (Op::get_workspace_size(args) != 0) return false;
        if (op.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) return false;
        return op.run(stream) == cutlass::Status::kSuccess;
    }
};

// 0/1: fp16-accumulate, largest tile that still fills the GPU. 2: fp32-accumulate
// 64x64 tiles for the deep-K low-resolution stages, where tile fit beats cuDNN
// 2-4x and fp16 accumulation would cost ~6 dB.
using Conv0 = Conv3dFp16<128, 256, 64, 64, 3, half_t>;
using Conv1 = Conv3dFp16<128, 128, 64, 64, 4, half_t>;
using Conv2 = Conv3dFp16<64, 64, 32, 32, 4, float>;
constexpr int kConvConfigCount = 3;
constexpr int64_t kMaxFp16Depth = 8192;   // the decoder's verified depth
constexpr int64_t kMinTiles = 128;
// conv_out (K=48) has 20 threadblocks at 64x64 and still beats cuDNN 2x
constexpr int64_t kMinTilesSmall = 16;

template <typename Cfg>
int64_t conv_tiles(int64_t m, int k) {
    return ((m + Cfg::kM - 1) / Cfg::kM) * ((k + Cfg::kN - 1) / Cfg::kN);
}

int select_conv_config(int64_t m, int k, int64_t depth) {
    if (depth > kMaxFp16Depth) {
        const bool small_launch = conv_tiles<Conv1>(m, k) < kMinTiles;
        return small_launch && conv_tiles<Conv2>(m, k) >= kMinTilesSmall ? 2 : -1;
    }
    if (k >= 256 && conv_tiles<Conv0>(m, k) >= kMinTiles) return 0;
    if (conv_tiles<Conv1>(m, k) >= kMinTiles) return 1;
    return -1;  // cuDNN
}

}  // namespace

// config: -1 selects by shape; 0..2 force a config (benchmarking/tests).
extern "C" bool launch_cutlass_fp16_conv3d(
    const void* x, const void* w, const void* bias, const void* resid, bool resid_full, void* out,
    int N, int D, int H, int W, int C, int K, int T, int R, int S, int Z, int P, int Q,
    int sd, int sh, int sw, int config, cudaStream_t stream) {
    const Conv3dDims d{N, D, H, W, C, K, T, R, S, Z, P, Q, sd, sh, sw};
    if (d.C % 8 != 0 || d.K % 8 != 0 || config >= kConvConfigCount) return false;
    const int64_t m = static_cast<int64_t>(d.N) * d.Z * d.P * d.Q;
    if (config < 0) config = select_conv_config(m, d.K, static_cast<int64_t>(d.C) * d.T * d.R * d.S);
    if (config < 0) return false;
    const auto xp = static_cast<const half_t*>(x);
    const auto wp = static_cast<const half_t*>(w);
    const auto bp = static_cast<const half_t*>(bias);
    const auto rp = static_cast<const half_t*>(resid);
    const auto op = static_cast<half_t*>(out);
    switch (config) {
        case 0: return Conv0::run(xp, wp, bp, rp, resid_full, op, d, stream);
        case 1: return Conv1::run(xp, wp, bp, rp, resid_full, op, d, stream);
        default: return Conv2::run(xp, wp, bp, rp, resid_full, op, d, stream);
    }
}

#else  // !COMFY_HAVE_CUTLASS -- stub; caller falls back to torch's conv.

extern "C" bool launch_cutlass_fp16_conv3d(
    const void*, const void*, const void*, const void*, bool, void*,
    int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t) {
    return false;
}

#endif
