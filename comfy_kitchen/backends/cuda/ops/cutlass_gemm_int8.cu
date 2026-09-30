/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * INT8 GEMM with a FUSED dequant epilogue via CUTLASS (EVT):
 *   D[m,n] = (sum_k A[m,k]*B[n,k]) * x_scale[m] * w_scale[n] + bias[n]   -> out dtype
 * bias (and the residual rscale) are read in the OUTPUT dtype and converted
 * to float in-register, so callers never cast them.
 *
 * Replaces cuBLAS-GEMM(int32) + separate dequant with one near-peak kernel.
 * Multiple tile configs are instantiated and selected with a shape heuristic
 * fitted from sustained Ada and Blackwell benchmarks.
 * Falls back to cuBLAS when CUTLASS is unavailable or no config can run.
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include <cstdlib>

#ifdef COMFY_HAVE_CUTLASS

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"

#include "cutlass_gemm_common.cuh"

namespace {
using namespace cute;
using comfy_cutlass::ThreadblockSwizzleLeanStreamK;

template <typename ThreadMap, bool Scalar>
struct WeightScaleBroadcast;

template <typename ThreadMap>
struct WeightScaleBroadcast<ThreadMap, false> {
    using Type = cutlass::epilogue::threadblock::VisitorRowBroadcast<
        ThreadMap, float, cute::Stride<_0, _1, int32_t>>;

    static typename Type::Arguments arguments(const float* scale, int n) {
        return {scale, 0.f, {_0{}, _1{}, n}};
    }
};

template <typename ThreadMap>
struct WeightScaleBroadcast<ThreadMap, true> {
    using Type = cutlass::epilogue::threadblock::VisitorScalarBroadcast<float>;

    static typename Type::Arguments arguments(const float* scale, int) {
        typename Type::Arguments result{};
        result.scalar_ptrs[0] = scale;
        return result;
    }
};

// One fused int8 GEMM, parameterized on output type AND tile/warp/stage config.
// bias is read in ElementOutput (nullptr broadcasts 0).
template <typename ElementOutput, int TBM, int TBN, int TBK, int WM, int WN, int WK, int NumStages,
          typename ArchTag = cutlass::arch::Sm80,
          bool ScalarWeightScale = false, int AlignmentAB = 16,
          typename ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
struct FusedInt8Gemm {
    using ElementA = int8_t; using ElementB = int8_t;
    using ElementC = ElementOutput;
    using ElementAcc = int32_t; using ElementCompute = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;   // B[N,K] row == [K,N] col
    using LayoutC = cutlass::layout::RowMajor;
    static constexpr int AlignA = AlignmentAB, AlignB = AlignmentAB;
    static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
    using TB   = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
    using Warp = cutlass::gemm::GemmShape<WM, WN, WK>;
    using Inst = cutlass::gemm::GemmShape<16, 8, 32>;
    static constexpr int EVTStages = 1;

    using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, Warp, ElementC, AlignC, EVTStages>;
    using Accum  = cutlass::epilogue::threadblock::VisitorAccFetch;
    using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<ThreadMap, ElementCompute, cute::Stride<_1, _0, int32_t>>;
    using WScale = typename WeightScaleBroadcast<ThreadMap, ScalarWeightScale>::Type;
    using Bias   = cutlass::epilogue::threadblock::VisitorRowBroadcast<ThreadMap, ElementOutput, cute::Stride<_0, _1, int32_t>>;
    using Mul0 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT0 = cutlass::epilogue::threadblock::Sm80EVT<Mul0, Accum, XScale>;
    using Mul1 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT1 = cutlass::epilogue::threadblock::Sm80EVT<Mul1, EVT0, WScale>;
    using Add2 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::plus, ElementOutput, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT2 = cutlass::epilogue::threadblock::Sm80EVT<Add2, EVT1, Bias>;
    using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<ThreadMap, ElementOutput, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, _1, int64_t>>;
    using EVTD = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT2>;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignA,
        ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignB,
        ElementC, LayoutC, AlignC,
        ElementAcc, ElementCompute,
        cutlass::arch::OpClassTensorOp, ArchTag,
        TB, Warp, Inst, EVTD,
        ThreadblockSwizzle,
        NumStages, cutlass::arch::OpMultiplyAddSaturate, EVTStages>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    static bool run_strided(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                            const ElementOutput* bias, ElementOutput* D, int M, int N, int K,
                            int output_stride, cudaStream_t stream) {
        const auto weight_scale_args = WeightScaleBroadcast<ThreadMap, ScalarWeightScale>::arguments(ws, N);
        typename EVTD::Arguments cb{
            { {  { {}, {const_cast<float*>(xs), 0.f, {_1{}, _0{}, M}}, {} },
                 weight_scale_args, {} },
              {const_cast<ElementOutput*>(bias), ElementOutput(0), {_0{}, _1{}, N}}, {} },
            {D, {output_stride, _1{}, M * output_stride}} };
        return comfy_cutlass::launch_universal<Gemm>(
            A, B, cb, M, N, K, stream);
    }
};

// FusedInt8Gemm with the pre-norm block's addcmul in the epilogue:
//   D = residual + rscale[n] * (acc * xs * ws + bias); rscale and residual in the output dtype.
template <typename ElementOutput, int TBM, int TBN, int TBK, int WM, int WN, int WK, int NumStages,
          typename ArchTag = cutlass::arch::Sm80,
          bool ScalarWeightScale = false, int AlignmentAB = 16,
          typename ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
struct FusedInt8GemmResidual {
    using ElementA = int8_t; using ElementB = int8_t;
    using ElementC = ElementOutput;
    using ElementAcc = int32_t; using ElementCompute = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    static constexpr int AlignA = AlignmentAB, AlignB = AlignmentAB;
    static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
    using TB   = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
    using Warp = cutlass::gemm::GemmShape<WM, WN, WK>;
    using Inst = cutlass::gemm::GemmShape<16, 8, 32>;
    static constexpr int EVTStages = 1;

    using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, Warp, ElementC, AlignC, EVTStages>;
    using Accum  = cutlass::epilogue::threadblock::VisitorAccFetch;
    using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<ThreadMap, ElementCompute, cute::Stride<_1, _0, int32_t>>;
    using WScale = typename WeightScaleBroadcast<ThreadMap, ScalarWeightScale>::Type;
    using Bias   = cutlass::epilogue::threadblock::VisitorRowBroadcast<ThreadMap, ElementOutput, cute::Stride<_0, _1, int32_t>>;
    using RScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<ThreadMap, ElementOutput, cute::Stride<_0, _1, int32_t>>;
    using Resid  = cutlass::epilogue::threadblock::VisitorAuxLoad<ThreadMap, ElementOutput, cute::Stride<int64_t, _1, int64_t>>;
    using Mul0 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT0 = cutlass::epilogue::threadblock::Sm80EVT<Mul0, Accum, XScale>;
    using Mul1 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT1 = cutlass::epilogue::threadblock::Sm80EVT<Mul1, EVT0, WScale>;
    // bias add rounds to ElementOutput like the plain kernel; only the addcmul stays fp32
    using Add2 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::plus, ElementOutput, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT2 = cutlass::epilogue::threadblock::Sm80EVT<Add2, EVT1, Bias>;
    using Mul3 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT3 = cutlass::epilogue::threadblock::Sm80EVT<Mul3, EVT2, RScale>;
    using Add4 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::plus, ElementOutput, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT4 = cutlass::epilogue::threadblock::Sm80EVT<Add4, EVT3, Resid>;
    using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<ThreadMap, ElementOutput, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, _1, int64_t>>;
    using EVTD = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT4>;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignA,
        ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignB,
        ElementC, LayoutC, AlignC,
        ElementAcc, ElementCompute,
        cutlass::arch::OpClassTensorOp, ArchTag,
        TB, Warp, Inst, EVTD,
        ThreadblockSwizzle,
        NumStages, cutlass::arch::OpMultiplyAddSaturate, EVTStages>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    static bool run(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                    const ElementOutput* bias, const ElementOutput* rscale, const ElementOutput* resid,
                    ElementOutput* D, int M, int N, int K, cudaStream_t stream) {
        const auto weight_scale_args = WeightScaleBroadcast<ThreadMap, ScalarWeightScale>::arguments(ws, N);
        typename EVTD::Arguments cb{
            { { { {  { {}, {const_cast<float*>(xs), 0.f, {_1{}, _0{}, M}}, {} },
                     weight_scale_args, {} },
                  {const_cast<ElementOutput*>(bias), ElementOutput(0), {_0{}, _1{}, N}}, {} },
                {const_cast<ElementOutput*>(rscale), ElementOutput(0), {_0{}, _1{}, N}}, {} },
              {const_cast<ElementOutput*>(resid), ElementOutput(0), {int64_t(N), _1{}, int64_t(M) * N}}, {} },
            {D, {int64_t(N), _1{}, int64_t(M) * N}} };
        return comfy_cutlass::launch_universal<Gemm>(
            A, B, cb, M, N, K, stream);
    }
};

namespace {

// Parse an exact small non-negative integer from an env value; -1 when unset,
// empty, or anything but digits (rejects "1junk" / " 1" / "1,2").
int parse_forced_config_env() {
    const char* v = std::getenv("COMFY_KITCHEN_FORCE_CUTLASS_INT8_CONFIG");
    if (v == nullptr || *v == '\0') return -1;
    int i = 0;
    for (const char* p = v; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9' || i > 999) return -1;
        i = i * 10 + (*p - '0');
    }
    return (i >= 0 && i <= 13) ? i : -1;
}

// Whether the given CUDA device is sm86 (GA102 consumer Ampere, e.g. RTX 3090 /
// A6000 / A40). Cached per device: a multi-GPU box may mix arches, so the
// first query must not decide for all of them.
bool device_is_sm86() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    static bool cached[16] = {false};
    static bool known[16] = {false};
    if (dev < 0 || dev >= 16) {
        cudaDeviceProp props;
        return cudaGetDeviceProperties(&props, dev) == cudaSuccess
            && props.major == 8 && props.minor == 6;
    }
    if (!known[dev]) {
        cudaDeviceProp props;
        cached[dev] = cudaGetDeviceProperties(&props, dev) == cudaSuccess
            && props.major == 8 && props.minor == 6;
        known[dev] = true;
    }
    return cached[dev];
}

}  // namespace

int select_fused_int8_config(int m, int n, int k) {
    // COMFY_KITCHEN_FORCE_CUTLASS_INT8_CONFIG=<int> forces a config index for
    // benchmarking. Has no effect when unset or not an exact 0-13 integer.
    static const int kForceConfig = parse_forced_config_env();
    if (kForceConfig >= 0) return kForceConfig;

    if (k % 16 != 0) return 9;

    // sm86 is the combination (84 SMs, 6MB L2, GDDR6) that makes StreamK lose
    // at large M, and the Ada/Blackwell thresholds below pick the wrong tile.
    if (device_is_sm86()) {
        // sm86 heuristic refit from the 2026-09-04 14-cfg sweep on A6000
        // (300W PL, 84 SMs, 6 MB L2, 768 GB/s GDDR6) against 30 shapes.
        // The sweep times the no-bias bf16 epilogue only; the bias, residual,
        // fp16 and fp32 variants share these picks unmeasured. Shapes below:
        // spanning LTX 2.5 (M=274..25900) and MiniMax H3 (M=53730..80666);
        // see int8_autotune_sweep.py and a6000_int8_cfg_table.json. Margins
        // in parentheses are (runner_up / best_ms). Avg regret vs all 38
        // merged sweep shapes (Sep 4 + Aug 29 mid-M series): < 0.4%; worst
        // remaining misses are sub-6% same-config thermal drift on the
        // 300W PL card (cfg0<->cfg13 coin-flip bands).
        //
        // Sweep winners used to fit this table:
        //     M        N      K    best  runner-up / margin
        //     274    2048   2048    cfg7  cfg2  (1.011x)
        //     274    8192   2048    cfg3  cfg2  (1.151x)
        //     274    2048   8192    cfg2  cfg7  (1.049x)
        //    1024    2048   2048    cfg12 cfg1  (1.003x)
        //    1024    4096   4096    cfg12 cfg13 (1.121x)
        //    1024    8192   2048    cfg12 cfg1  (1.018x)
        //    1024   16384   4096    cfg12 cfg0  (1.025x)
        //    1024    2048   8192    cfg12 cfg13 (1.042x)
        //    1024    4096  16384    cfg12 cfg0  (1.115x)
        //    1797    2048   2048    cfg1  cfg3  (1.090x)
        //    1797    6144   2048    cfg1  cfg12 (1.010x)
        //    1797   16384   2048    cfg0  cfg1  (1.045x)
        //    1797    2048   8192    cfg1  cfg13 (1.129x)
        //   25900    2048   4096    cfg13 cfg0  (1.039x)
        //   25900    4096   2048    cfg13 cfg0  (1.003x)
        //   25900    4096   4096    cfg13 cfg0  (1.043x)
        //   25900    4096  16384    cfg0  cfg13 (1.063x)
        //   25900   16384   4096    cfg0  cfg3  (1.173x)
        //   53730    5376   7168    cfg13 cfg0  (1.020x)
        //   53730    5376  14336    cfg13 cfg0  (1.034x)
        //   53730   21504   5376    cfg0  cfg3  (1.204x)
        //   53730   28672   5376    cfg0  cfg3  (1.195x)
        //   74977    5376   7168    cfg0  cfg13 (1.001x; within noise)
        //   74977    5376  14336    cfg13 cfg0  (1.001x; within noise)
        //   74977   21504   5376    cfg0  cfg3  (1.193x)
        //   74977   28672   5376    cfg0  cfg3  (1.196x)
        //   80666    5376   7168    cfg0  cfg13 (1.107x)
        //   80666    5376  14336    cfg0  cfg13 (1.134x)
        //   80666   21504   5376    cfg0  cfg3  (1.194x)
        //   80666   28672   5376    cfg0  cfg3  (1.194x)
        //
        // Patterns used to fit (avg regret vs the 30-point sweep: 0.004%):
        //   * Tiny M (m <= 512): cfg7 for K<=4096, cfg2 for taller K, cfg3
        //     for the wide-N low-K corner. (An earlier fit picked cfg9 here;
        //     the Sep 4 sweep has cfg7 ahead by 13% at 274x2048x2048.)
        //   * M in (512, 1024]: cfg12 (StreamK 128x128) wins every measured
        //     shape — all six 1024-row winners are cfg12. The previous
        //     K<=2048 cfg0/cfg1 split was fitted on Aug-29 margins that
        //     flipped in this sweep.
        //   * M in (1024, 2048]: K<=2048 -> cfg1 (cfg0 at n>=16384); tall K
        //     -> cfg0 for n>=8192, cfg12 for medium N (4096..8192, measured
        //     2048x4096x4096), cfg1 for narrow N. The previous rule sent
        //     narrow-N tall-K here to cfg12 — measured +62% wrong at
        //     1797x2048x8192.
        //   * M > 2048, wide N (n > 8192): cfg0 dominates. Note "wide"
        //     starts at 8192, not 4096: n=5376 at mid-M wants cfg13
        //     (53730x5376x7168/14336 are cfg13 wins).
        //   * M > 2048, narrow N: tall-K (m>=16384 & k>=16384) boosts cfg0
        //     (measured 25900x4096x16384, +6.3%); otherwise StreamK 128x256
        //     (cfg13) through the 25900 / 53730 / 74977 bands, cfg0 from
        //     80666. The 74977x5376 winners were cfg0<->cfg13 ties (0.1%) on
        //     Sep 4 and cfg13 by 3-6% on 2026-09-26; 80666x5376 was cfg0 by
        //     11-13% on Sep 4 and a tie on 2026-09-26 — power-cap dependent.
        //   * 2026-09-26 revalidation (300 W cap hit in 99% of samples, SM clock
        //     1135-1660 MHz): rule matches the winner on 34/38 shapes, avg
        //     regret 1.0% on both an sm_80-cubin and an sm86-native build; the
        //     two builds agree within noise on every shape.

        // --- Tiny-M band ------------------------------------------------
        if (m <= 512) {
            if (n >= 8192 && k <= 4096) return 3;   // small-M, wide-N, low-K
            return k <= 4096 ? 7 : 2;               // small-M generic
        }

        // --- Small-M band (M <= 2048) -------------------------------------
        if (m <= 1024) {
            // 1024x2048x2048: plain 128x128 (cfg1) beats StreamK cfg12 by 13-15% in the
            // 2026-09-26 revalidation (sm_80-cubin and sm86-native builds agree; K is
            // too short for StreamK's split to pay). It was a 0.3% tie on Sep 4.
            if (n <= 2048 && k <= 2048) return 1;
            return 12;                     // every other measured 1024-row shape: cfg12
        }
        if (m <= 2048) {
            if (k <= 2048) return n >= 16384 ? 0 : 1;  // 1797x16384x2048 vs 1797x2048x2048
            // tall K: genuinely wide N -> cfg0; medium N (4096..8192) ->
            // cfg12 (2048x4096x4096, Aug-29 sweep +1.6%, re-validated +16.6%);
            // narrow N -> cfg1 (1797x2048x8192, +62% over cfg12)
            if (n >= 8192) return 0;
            return n >= 4096 ? 12 : 1;
        }

        // --- Mid/large-M (M > 2048) ---------------------------------------
        if (n > 8192) return 0;            // wide-N: cfg0 dominates for m>2048
        // (n=5376 at mid-M wants cfg13 — wide starts at 8192, not 4096)

        // narrow-N:
        //  - tall-K at m>=16384 boosts cfg0 (25900x4096x16384, +6.3%)
        //  - otherwise StreamK 128x256 (cfg13) through m=65536, cfg0 beyond
        if (m >= 16384 && k >= 16384) return 0;
        // 2026-09-26 revalidation: the 74977x5376 band (K=7168/14336) is cfg13 by 3-6%
        // in both builds (a 0.1% tie on Sep 4), so StreamK runs through ~78k rows;
        // 80666x5376 stays cfg0 (cfg0 by 11-13% on Sep 4, a tie today).
        if (m <= 78000) return 13;
        return 0;
    }

    const int64_t mn = int64_t(m) * n;
    if (n <= 24832) {
        if (mn <= 1477632) {
            if (mn <= 259072) return k <= 7296 ? 6 : 12;
            return int64_t(n) * k <= 16252928 ? 2 : 12;
        }
        if (mn <= 4193792) {
            return int64_t(n) * k <= 5275648 ? 1 : 12;
        }
        return int64_t(m) * k <= int64_t(n) * 5675 ? 0 : 13;
    }
    if (int64_t(n) * k <= int64_t(m) * 11096) return 0;
    return mn <= 108003328 ? 0 : 13;
}

// The tree ignores wave quantization: at decoder-tile M (~2k rows) its 128x256
// pick can leave a 2.1-wave grid where 128x128 wins despite ~8% lower per-tile
// throughput. Between those two, take the smaller wave-rounding x tile-cost.
int device_sm_count() {
    static int counts[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) return 1;
    if (counts[dev] == 0) {
        cudaDeviceGetAttribute(&counts[dev], cudaDevAttrMultiProcessorCount, dev);
        if (counts[dev] <= 0) counts[dev] = 1;
    }
    return counts[dev];
}

int wave_guard(int m, int n, int selected) {
    if (selected != 0 && selected != 1) return selected;
    // The sm86 branch of select_fused_int8_config is a table of measured winners
    // (wave quantization already priced in), so the estimate below must not
    // re-decide it: on the 38 swept shapes it agrees everywhere except
    // 1024x2048x2048, where it turns the measured cfg1 pick into cfg0 (+20%).
    if (device_is_sm86()) return selected;
    const int sms = device_sm_count();
    auto cost = [&](int64_t tile_n, double per_tile) {
        const double waves = double(((m + 127) / 128) * ((n + tile_n - 1) / tile_n)) / sms;
        return std::ceil(waves) / waves * per_tile;
    };
    return cost(256, 1.0) <= cost(128, 1.08) ? 0 : 1;
}

template <typename Launch>
bool launch_fused_int8_heuristic(int m, int n, int k, Launch launch) {
    // COMFY_KITCHEN_FORCE_CUTLASS_INT8_CONFIG wins outright: neither the wave guard
    // nor the fallback list may substitute another tile, so a forced config is
    // exactly what runs (or fails) and benchmarks stay honest.
    static const int kForcedConfig = parse_forced_config_env();
    const int selected = kForcedConfig >= 0
        ? kForcedConfig
        : wave_guard(m, n, select_fused_int8_config(m, n, k));
    if (launch(selected)) return true;
    if (kForcedConfig >= 0) return false;

    static constexpr int aligned_fallbacks[] = {2, 12, 0, 13, 1, 6, 8, 7, 3, 4, 5};
    static constexpr int low_alignment_fallbacks[] = {9, 10, 11};
    if (k % 16 == 0) {
        for (int config : aligned_fallbacks) {
            if (config != selected && launch(config)) return true;
        }
    } else {
        for (int config : low_alignment_fallbacks) {
            if (config != selected && launch(config)) return true;
        }
    }
    return false;
}

// The tile table; select_fused_int8_config and the fallback orders index it, and
// every epilogue variant's runner table is instantiated from it.
using comfy_cutlass::TileConfig;
using comfy_cutlass::ConfigList;
using FusedInt8Configs = ConfigList<
    TileConfig<128, 256,  64, 64, 64,  64, 3>,                                       // 0
    TileConfig<128, 128,  64, 64, 64,  64, 4>,                                       // 1
    TileConfig< 64, 128,  64, 32, 64,  64, 4>,                                       // 2
    TileConfig< 64, 256,  64, 32, 64,  64, 3>,                                       // 3
    TileConfig< 32, 256,  64, 32, 64,  64, 4>,                                       // 4
    TileConfig< 32, 128,  64, 32, 64,  64, 4>,                                       // 5
    TileConfig< 16, 128,  64, 16, 64,  64, 4>,                                       // 6
    TileConfig< 64, 128, 128, 32, 64, 128, 3>,                                       // 7
    TileConfig<128,  64, 128, 64, 32, 128, 3>,                                       // 8
    TileConfig< 64, 128,  64, 32, 64,  64, 4, 8>,                                    // 9  (K % 8 alignment)
    TileConfig< 32, 128,  64, 32, 64,  64, 4, 8>,                                    // 10
    TileConfig< 16, 128,  64, 16, 64,  64, 4, 8>,                                    // 11
    TileConfig<128, 128,  64, 64, 64,  64, 4, 16, ThreadblockSwizzleLeanStreamK>,    // 12 (stream-K)
    TileConfig<128, 256,  64, 64, 64,  64, 3, 16, ThreadblockSwizzleLeanStreamK>>;   // 13
constexpr int kFusedConfigCount = FusedInt8Configs::size;

template <typename OutT, typename C>
using PlainGemm = FusedInt8Gemm<OutT, C::TBM, C::TBN, C::TBK, C::WM, C::WN, C::WK, C::NumStages,
                                cutlass::arch::Sm80, false, C::AlignmentAB,
                                typename C::ThreadblockSwizzle>;
template <typename OutT, typename C>
using ResidualGemm = FusedInt8GemmResidual<OutT, C::TBM, C::TBN, C::TBK, C::WM, C::WN, C::WK,
                                           C::NumStages, cutlass::arch::Sm80, false,
                                           C::AlignmentAB, typename C::ThreadblockSwizzle>;

// One run_strided table (stride == N is the plain case) serves every entry
// point; bias may be nullptr (the RowBroadcast visitor broadcasts 0).
template <typename OutT>
using FusedFn = bool (*)(const int8_t*, const int8_t*, const float*, const float*,
                         const OutT*, OutT*, int, int, int, int, cudaStream_t);
template <typename OutT>
using FusedResidualFn = bool (*)(const int8_t*, const int8_t*, const float*, const float*,
                                 const OutT*, const OutT*, const OutT*, OutT*, int, int, int,
                                 cudaStream_t);

template <typename OutT, typename... Cs>
const FusedFn<OutT>* make_fused_runners(ConfigList<Cs...>) {
    static const FusedFn<OutT> runners[sizeof...(Cs)] = {&PlainGemm<OutT, Cs>::run_strided...};
    return runners;
}

template <typename OutT, typename... Cs>
const FusedResidualFn<OutT>* make_fused_residual_runners(ConfigList<Cs...>) {
    static const FusedResidualFn<OutT> runners[sizeof...(Cs)] = {&ResidualGemm<OutT, Cs>::run...};
    return runners;
}

template <typename OutT>
const FusedFn<OutT>* fused_runners() {
    return make_fused_runners<OutT>(FusedInt8Configs{});
}

template <typename OutT>
bool dispatch_fused_strided(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                            const OutT* bias, OutT* D, int M, int N, int K, int output_stride,
                            cudaStream_t stream) {
    const FusedFn<OutT>* runners = fused_runners<OutT>();
    return launch_fused_int8_heuristic(M, N, K, [&](int config) {
        return runners[config](A, B, xs, ws, bias, D, M, N, K, output_stride, stream);
    });
}

template <typename OutT>
bool dispatch_fused(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                    const OutT* bias, OutT* D, int M, int N, int K, cudaStream_t stream) {
    return dispatch_fused_strided<OutT>(A, B, xs, ws, bias, D, M, N, K, N, stream);
}

template <typename OutT>
bool dispatch_fused_config(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                           OutT* D, int M, int N, int K, int config, cudaStream_t stream) {
    if (config < 0 || config >= kFusedConfigCount) return false;
    return fused_runners<OutT>()[config](A, B, xs, ws, nullptr, D, M, N, K, N, stream);
}

template <typename OutT>
bool dispatch_fused_residual(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                             const OutT* bias, const OutT* rscale, const OutT* resid,
                             OutT* D, int M, int N, int K, cudaStream_t stream) {
    const FusedResidualFn<OutT>* runners = make_fused_residual_runners<OutT>(FusedInt8Configs{});
    return launch_fused_int8_heuristic(M, N, K, [&](int config) {
        return runners[config](A, B, xs, ws, bias, rscale, resid, D, M, N, K, stream);
    });
}

}  // namespace

extern "C" {
bool launch_cutlass_int8_dequant_residual(
    const void* A, const void* B, const void* xs, const void* ws, const void* bias,
    const void* rscale, const void* resid, void* D, int64_t M, int64_t N, int64_t K,
    int out_dtype_code, cudaStream_t stream)
{
    if (M == 0 || N == 0) return true;
    // Declining K == 0 keeps the caller's eager residual+bias fallback correct.
    if (K == 0) return false;
    // bias may be nullptr: the RowBroadcast visitor broadcasts null_default(0).
    if (rscale == nullptr || resid == nullptr) return false;
    const int8_t* a = static_cast<const int8_t*>(A);
    const int8_t* b = static_cast<const int8_t*>(B);
    const float* x = static_cast<const float*>(xs);
    const float* w = static_cast<const float*>(ws);
    switch (out_dtype_code) {
        case 0: return dispatch_fused_residual<float>(
            a, b, x, w, static_cast<const float*>(bias), static_cast<const float*>(rscale), static_cast<const float*>(resid), static_cast<float*>(D), M, N, K, stream);
        case 1: return dispatch_fused_residual<cutlass::half_t>(
            a, b, x, w, static_cast<const cutlass::half_t*>(bias), static_cast<const cutlass::half_t*>(rscale), static_cast<const cutlass::half_t*>(resid), static_cast<cutlass::half_t*>(D), M, N, K, stream);
        case 2: return dispatch_fused_residual<cutlass::bfloat16_t>(
            a, b, x, w, static_cast<const cutlass::bfloat16_t*>(bias), static_cast<const cutlass::bfloat16_t*>(rscale), static_cast<const cutlass::bfloat16_t*>(resid), static_cast<cutlass::bfloat16_t*>(D), M, N, K, stream);
        default: return false;
    }
}

bool launch_cutlass_int8_dequant(
    const void* A, const void* B, const void* xs, const void* ws, const void* bias,
    void* D, int64_t M, int64_t N, int64_t K, int out_dtype_code, cudaStream_t stream)
{
    if (M == 0 || N == 0 || K == 0) return true;
    const int8_t* a = static_cast<const int8_t*>(A);
    const int8_t* b = static_cast<const int8_t*>(B);
    const float* x = static_cast<const float*>(xs);
    const float* w = static_cast<const float*>(ws);
    switch (out_dtype_code) {
        case 0: return dispatch_fused<float>(a, b, x, w, static_cast<const float*>(bias), static_cast<float*>(D), M, N, K, stream);
        case 1: return dispatch_fused<cutlass::half_t>(a, b, x, w, static_cast<const cutlass::half_t*>(bias), static_cast<cutlass::half_t*>(D), M, N, K, stream);
        case 2: return dispatch_fused<cutlass::bfloat16_t>(a, b, x, w, static_cast<const cutlass::bfloat16_t*>(bias), static_cast<cutlass::bfloat16_t*>(D), M, N, K, stream);
        default: return false;
    }
}

bool launch_cutlass_int8_dequant_strided(
    const void* A, const void* B, const void* xs, const void* ws, const void* bias,
    void* D, int64_t M, int64_t N, int64_t K, int64_t output_stride, int out_dtype_code,
    cudaStream_t stream)
{
    if (M == 0 || N == 0 || K == 0) return true;
    if (output_stride < N) return false;
    const int8_t* a = static_cast<const int8_t*>(A);
    const int8_t* b = static_cast<const int8_t*>(B);
    const float* x = static_cast<const float*>(xs);
    const float* w = static_cast<const float*>(ws);
    switch (out_dtype_code) {
        case 0: return dispatch_fused_strided<float>(a, b, x, w, static_cast<const float*>(bias), static_cast<float*>(D), M, N, K, output_stride, stream);
        case 1: return dispatch_fused_strided<cutlass::half_t>(a, b, x, w, static_cast<const cutlass::half_t*>(bias), static_cast<cutlass::half_t*>(D), M, N, K, output_stride, stream);
        case 2: return dispatch_fused_strided<cutlass::bfloat16_t>(a, b, x, w, static_cast<const cutlass::bfloat16_t*>(bias), static_cast<cutlass::bfloat16_t*>(D), M, N, K, output_stride, stream);
        default: return false;
    }
}

bool launch_cutlass_int8_dequant_config(
    const void* A, const void* B, const void* xs, const void* ws, void* D,
    int64_t M, int64_t N, int64_t K, int out_dtype_code, int config,
    cudaStream_t stream)
{
    if (M == 0 || N == 0 || K == 0) return true;
    const int8_t* a = static_cast<const int8_t*>(A);
    const int8_t* b = static_cast<const int8_t*>(B);
    const float* x = static_cast<const float*>(xs);
    const float* w = static_cast<const float*>(ws);
    switch (out_dtype_code) {
        case 2: return dispatch_fused_config<cutlass::bfloat16_t>(
            a, b, x, w, static_cast<cutlass::bfloat16_t*>(D), M, N, K, config, stream);
        default: return false;
    }
}
}  // extern "C"

#else  // !COMFY_HAVE_CUTLASS -- stub; caller falls back to cuBLAS + separate dequant.

extern "C" bool launch_cutlass_int8_dequant(
    const void*, const void*, const void*, const void*, const void*,
    void*, int64_t, int64_t, int64_t, int, cudaStream_t) {
    return false;
}

extern "C" bool launch_cutlass_int8_dequant_strided(
    const void*, const void*, const void*, const void*, const void*,
    void*, int64_t, int64_t, int64_t, int64_t, int, cudaStream_t) {
    return false;
}

extern "C" bool launch_cutlass_int8_dequant_config(
    const void*, const void*, const void*, const void*, void*,
    int64_t, int64_t, int64_t, int, int, cudaStream_t) {
    return false;
}

extern "C" bool launch_cutlass_int8_dequant_residual(
    const void*, const void*, const void*, const void*, const void*,
    const void*, const void*, void*, int64_t, int64_t, int64_t, int,
    cudaStream_t) {
    return false;
}

#endif
