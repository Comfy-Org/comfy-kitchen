// SPDX-License-Identifier: Apache-2.0
//
// Plain ConvRot W4A4 helpers: rowwise signed int4 quantization and int4 MMA
// with row/column dequant scales.
#include "dtype_dispatch.cuh"
#include "svdquant_utils.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef COMFY_HAVE_CUTLASS
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include <map>
#include <mutex>
#include <tuple>
#endif

extern "C" void launch_cublas_gemm_int8_kernel(
    const void* A_ptr,
    const void* B_ptr,
    void* C_ptr,
    int64_t M,
    int64_t N,
    int64_t K,
    void* workspace_ptr,
    int64_t workspace_size,
    cudaStream_t stream);

extern "C" bool launch_cutlass_int8_dequant_strided(
    const void* A,
    const void* B,
    const void* xs,
    const void* ws,
    const void* bias,
    void* D,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t output_stride,
    int out_dtype_code,
    cudaStream_t stream);

namespace {

using comfy::svdquant::cp_async_16b;
using comfy::svdquant::cp_async_commit_group;
using comfy::svdquant::cp_async_wait_group;
using comfy::svdquant::kGroupSize;
using comfy::svdquant::kInt4Max;
using comfy::svdquant::mma_m16n8k64_s4s4s32;
using comfy::svdquant::pack_int4_pair;

constexpr int kThreadsPerWarp = 32;
constexpr int kConvRotGroup = 256;

__device__ __forceinline__ uint32_t pcg_hash(uint32_t x) {
    const uint32_t state = x * 747796405u + 2891336453u;
    const uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ __forceinline__ float stochastic_rng_value(int64_t idx, uint64_t seed) {
    const uint64_t key = static_cast<uint64_t>(idx) + seed;
    const uint32_t folded = static_cast<uint32_t>(key) ^ static_cast<uint32_t>(key >> 32);
    return static_cast<float>(pcg_hash(folded) >> 8) * 0x1.0p-24f;
}

template<bool STOCHASTIC>
__device__ __forceinline__ int quantize_int4_value(float x, float inv_scale, uint64_t seed, int64_t idx) {
    const float scaled = x * inv_scale;
    int q;
    if constexpr (STOCHASTIC) {
        q = static_cast<int>(floorf(scaled + stochastic_rng_value(idx, seed)));
    } else {
        q = __float2int_rn(scaled);
    }
    return max(-kInt4Max, min(kInt4Max, q));
}

template<typename T>
__device__ __forceinline__ float finite_max_for_dtype();
template<> __device__ __forceinline__ float finite_max_for_dtype<float>() { return FLT_MAX; }
template<> __device__ __forceinline__ float finite_max_for_dtype<__half>() { return 65504.0f; }
template<> __device__ __forceinline__ float finite_max_for_dtype<__nv_bfloat16>() { return 3.38953139e38f; }

template<typename T>
__device__ __forceinline__ float finite_absmax_for_int4_scale(float abs_max) {
    return fminf(abs_max, finite_max_for_dtype<T>());
}

inline int fp_dtype_size_bytes(int dtype_code) {
    switch (dtype_code) {
        case 0: return 4;
        case 1:
        case 2: return 2;
        default: throw std::runtime_error("unsupported floating point dtype code");
    }
}

__device__ __forceinline__ int unpack_int4_nibble(uint32_t v) {
    return static_cast<int>((v & 0x0fu) ^ 0x08u) - 8;
}

#ifdef COMFY_HAVE_CUTLASS
template <typename ElementOutput, int TBM, int TBN, int TBK, int WM, int WN, int WK, int NumStages,
          typename ArchTag = cutlass::arch::Sm89>
struct FusedInt4Gemm {
    using ElementA = cutlass::int4b_t;
    using ElementB = cutlass::int4b_t;
    using ElementC = ElementOutput;
    using ElementAcc = int32_t;
    using ElementCompute = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    static constexpr int AlignA = 32;
    static constexpr int AlignB = 32;
    static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
    using TB = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
    using Warp = cutlass::gemm::GemmShape<WM, WN, WK>;
    using Inst = cutlass::gemm::GemmShape<16, 8, 64>;
    static constexpr int EVTStages = 1;

    using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, Warp, ElementC, AlignC, EVTStages>;
    using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
    using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<ThreadMap, ElementCompute, cute::Stride<cute::_1, cute::_0, int32_t>>;
    using WScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<ThreadMap, ElementCompute, cute::Stride<cute::_0, cute::_1, int32_t>>;
    using Bias = cutlass::epilogue::threadblock::VisitorRowBroadcast<ThreadMap, ElementCompute, cute::Stride<cute::_0, cute::_1, int32_t>>;
    using Mul0 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT0 = cutlass::epilogue::threadblock::Sm80EVT<Mul0, Accum, XScale>;
    using Mul1 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT1 = cutlass::epilogue::threadblock::Sm80EVT<Mul1, EVT0, WScale>;
    using Add2 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::plus, ElementOutput, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT2 = cutlass::epilogue::threadblock::Sm80EVT<Add2, EVT1, Bias>;
    using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<ThreadMap, ElementOutput, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, cute::_1, int64_t>>;
    using EVTD = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT2>;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignA,
        ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignB,
        ElementC, LayoutC, AlignC,
        ElementAcc, ElementCompute,
        cutlass::arch::OpClassTensorOp, ArchTag,
        TB, Warp, Inst, EVTD,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        NumStages, cutlass::arch::OpMultiplyAddSaturate, EVTStages>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    static bool run(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                    const float* bias, ElementOutput* D, int M, int N, int K, cudaStream_t stream) {
        cutlass::gemm::GemmCoord problem(M, N, K);
        typename EVTD::Arguments cb{
            { {  { {}, {const_cast<float*>(xs), 0.f, {cute::_1{}, cute::_0{}, M}}, {} },
                 {const_cast<float*>(ws), 0.f, {cute::_0{}, cute::_1{}, N}}, {} },
              {const_cast<float*>(bias), 0.f, {cute::_0{}, cute::_1{}, N}}, {} },
            {D, {N, cute::_1{}, M * N}} };
        typename Gemm::Arguments args(
            cutlass::gemm::GemmUniversalMode::kGemm, problem, 1, cb,
            reinterpret_cast<cutlass::int4b_t*>(const_cast<int8_t*>(A)),
            reinterpret_cast<cutlass::int4b_t*>(const_cast<int8_t*>(B)),
            nullptr, nullptr,
            (int64_t)M * K, (int64_t)N * K, 0, 0, K, K, 0, 0);

        Gemm gemm;
        if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
        if (Gemm::get_workspace_size(args) != 0) return false;
        if (gemm.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) return false;
        return gemm(stream) == cutlass::Status::kSuccess;
    }
};

template <typename ElementOutput, int TBM, int TBN, int TBK, int WM, int WN, int WK, int NumStages,
          typename ArchTag = cutlass::arch::Sm89>
struct FusedInt4GemmNoBias {
    using ElementA = cutlass::int4b_t;
    using ElementB = cutlass::int4b_t;
    using ElementC = ElementOutput;
    using ElementAcc = int32_t;
    using ElementCompute = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    static constexpr int AlignA = 32;
    static constexpr int AlignB = 32;
    static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
    using TB = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
    using Warp = cutlass::gemm::GemmShape<WM, WN, WK>;
    using Inst = cutlass::gemm::GemmShape<16, 8, 64>;
    static constexpr int EVTStages = 1;

    using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, Warp, ElementC, AlignC, EVTStages>;
    using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
    using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<ThreadMap, ElementCompute, cute::Stride<cute::_1, cute::_0, int32_t>>;
    using WScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<ThreadMap, ElementCompute, cute::Stride<cute::_0, cute::_1, int32_t>>;
    using Mul0 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT0 = cutlass::epilogue::threadblock::Sm80EVT<Mul0, Accum, XScale>;
    using Mul1 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementOutput, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT1 = cutlass::epilogue::threadblock::Sm80EVT<Mul1, EVT0, WScale>;
    using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<ThreadMap, ElementOutput, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, cute::_1, int64_t>>;
    using EVTD = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT1>;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignA,
        ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignB,
        ElementC, LayoutC, AlignC,
        ElementAcc, ElementCompute,
        cutlass::arch::OpClassTensorOp, ArchTag,
        TB, Warp, Inst, EVTD,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        NumStages, cutlass::arch::OpMultiplyAddSaturate, EVTStages>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    static bool run(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                    ElementOutput* D, int M, int N, int K, cudaStream_t stream) {
        cutlass::gemm::GemmCoord problem(M, N, K);
        typename EVTD::Arguments cb{
            { { {}, {const_cast<float*>(xs), 0.f, {cute::_1{}, cute::_0{}, M}}, {} },
              {const_cast<float*>(ws), 0.f, {cute::_0{}, cute::_1{}, N}}, {} },
            {D, {N, cute::_1{}, M * N}} };
        typename Gemm::Arguments args(
            cutlass::gemm::GemmUniversalMode::kGemm, problem, 1, cb,
            reinterpret_cast<cutlass::int4b_t*>(const_cast<int8_t*>(A)),
            reinterpret_cast<cutlass::int4b_t*>(const_cast<int8_t*>(B)),
            nullptr, nullptr,
            (int64_t)M * K, (int64_t)N * K, 0, 0, K, K, 0, 0);

        Gemm gemm;
        if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
        if (Gemm::get_workspace_size(args) != 0) return false;
        if (gemm.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) return false;
        return gemm(stream) == cutlass::Status::kSuccess;
    }
};

template <typename OutT>
bool dispatch_fused_int4(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                         const float* bias, OutT* D, int M, int N, int K, cudaStream_t stream) {
    using Fn = bool (*)(const int8_t*, const int8_t*, const float*, const float*, const float*, OutT*, int, int, int, cudaStream_t);
    static const Fn runners[] = {
        &FusedInt4Gemm<OutT, 128, 256, 128, 64, 64, 128, 3>::run,
        &FusedInt4Gemm<OutT, 128, 256, 256, 64, 64, 256, 3>::run,
        &FusedInt4Gemm<OutT, 128, 512, 256, 64, 128, 256, 3>::run,
        &FusedInt4Gemm<OutT, 256, 256, 256, 64, 128, 256, 3>::run,
        &FusedInt4Gemm<OutT, 256, 128, 256, 64, 64, 256, 3>::run,
        &FusedInt4Gemm<OutT, 128, 128, 256, 64, 64, 256, 3>::run,
        &FusedInt4Gemm<OutT,  64, 256, 256, 64, 64, 256, 3>::run,
    };
    constexpr int NC = sizeof(runners) / sizeof(runners[0]);

    if (M == 4608 && N == 3072 && K == 15360) {
        return runners[0](A, B, xs, ws, bias, D, M, N, K, stream);
    }

    static std::mutex mtx;
    static std::map<std::tuple<int, int, int>, int> cache;
    const std::tuple<int, int, int> key{M, N, K};

    static thread_local int last_m = -1;
    static thread_local int last_n = -1;
    static thread_local int last_k = -1;
    static thread_local int last_best = -2;
    if (M == last_m && N == last_n && K == last_k) {
        if (last_best < 0) return false;
        if (runners[last_best](A, B, xs, ws, bias, D, M, N, K, stream)) return true;
        return runners[last_best](A, B, xs, ws, bias, D, M, N, K, stream);
    }

    int best;
    {
        std::lock_guard<std::mutex> lk(mtx);
        auto it = cache.find(key);
        best = (it != cache.end()) ? it->second : -2;
    }
    if (best == -2) {
        best = -1;
        float best_ms = 1e30f;
        cudaEvent_t s, e;
        cudaEventCreate(&s);
        cudaEventCreate(&e);
        for (int i = 0; i < NC; ++i) {
            if (!runners[i](A, B, xs, ws, bias, D, M, N, K, stream)) continue;
            cudaStreamSynchronize(stream);
            for (int r = 0; r < 8; ++r) {
                runners[i](A, B, xs, ws, bias, D, M, N, K, stream);
            }
            cudaStreamSynchronize(stream);
            cudaEventRecord(s, stream);
            for (int r = 0; r < 32; ++r) {
                runners[i](A, B, xs, ws, bias, D, M, N, K, stream);
            }
            cudaEventRecord(e, stream);
            cudaEventSynchronize(e);
            float ms = 0.f;
            cudaEventElapsedTime(&ms, s, e);
            if (ms < best_ms) {
                best_ms = ms;
                best = i;
            }
        }
        cudaEventDestroy(s);
        cudaEventDestroy(e);
        std::lock_guard<std::mutex> lk(mtx);
        cache[key] = best;
    }
    last_m = M;
    last_n = N;
    last_k = K;
    last_best = best;
    if (best < 0) return false;
    if (runners[best](A, B, xs, ws, bias, D, M, N, K, stream)) return true;
    return runners[best](A, B, xs, ws, bias, D, M, N, K, stream);
}

template <typename OutT>
bool dispatch_fused_int4_no_bias(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                                 OutT* D, int M, int N, int K, cudaStream_t stream) {
    using Fn = bool (*)(const int8_t*, const int8_t*, const float*, const float*, OutT*, int, int, int, cudaStream_t);
    static const Fn runners[] = {
        &FusedInt4GemmNoBias<OutT, 128, 256, 128, 64, 64, 128, 3>::run,
        &FusedInt4GemmNoBias<OutT, 128, 256, 256, 64, 64, 256, 3>::run,
        &FusedInt4GemmNoBias<OutT, 128, 512, 256, 64, 128, 256, 3>::run,
        &FusedInt4GemmNoBias<OutT, 256, 256, 256, 64, 128, 256, 3>::run,
        &FusedInt4GemmNoBias<OutT, 256, 128, 256, 64, 64, 256, 3>::run,
        &FusedInt4GemmNoBias<OutT, 128, 128, 256, 64, 64, 256, 3>::run,
        &FusedInt4GemmNoBias<OutT,  64, 256, 256, 64, 64, 256, 3>::run,
    };
    constexpr int NC = sizeof(runners) / sizeof(runners[0]);

    if (M == 4608 && N == 3072 && K == 15360) {
        return runners[0](A, B, xs, ws, D, M, N, K, stream);
    }

    static std::mutex mtx;
    static std::map<std::tuple<int, int, int>, int> cache;
    const std::tuple<int, int, int> key{M, N, K};

    static thread_local int last_m = -1;
    static thread_local int last_n = -1;
    static thread_local int last_k = -1;
    static thread_local int last_best = -2;
    if (M == last_m && N == last_n && K == last_k) {
        if (last_best < 0) return false;
        if (runners[last_best](A, B, xs, ws, D, M, N, K, stream)) return true;
        return runners[last_best](A, B, xs, ws, D, M, N, K, stream);
    }

    int best;
    {
        std::lock_guard<std::mutex> lk(mtx);
        auto it = cache.find(key);
        best = (it != cache.end()) ? it->second : -2;
    }
    if (best == -2) {
        best = -1;
        float best_ms = 1e30f;
        cudaEvent_t s, e;
        cudaEventCreate(&s);
        cudaEventCreate(&e);
        for (int i = 0; i < NC; ++i) {
            if (!runners[i](A, B, xs, ws, D, M, N, K, stream)) continue;
            cudaStreamSynchronize(stream);
            for (int r = 0; r < 8; ++r) {
                runners[i](A, B, xs, ws, D, M, N, K, stream);
            }
            cudaStreamSynchronize(stream);
            cudaEventRecord(s, stream);
            for (int r = 0; r < 32; ++r) {
                runners[i](A, B, xs, ws, D, M, N, K, stream);
            }
            cudaEventRecord(e, stream);
            cudaEventSynchronize(e);
            float ms = 0.f;
            cudaEventElapsedTime(&ms, s, e);
            if (ms < best_ms) {
                best_ms = ms;
                best = i;
            }
        }
        cudaEventDestroy(s);
        cudaEventDestroy(e);
        std::lock_guard<std::mutex> lk(mtx);
        cache[key] = best;
    }
    last_m = M;
    last_n = N;
    last_k = K;
    last_best = best;
    if (best < 0) return false;
    if (runners[best](A, B, xs, ws, D, M, N, K, stream)) return true;
    return runners[best](A, B, xs, ws, D, M, N, K, stream);
}

#endif

template<typename T>
__device__ __forceinline__ float to_float(T v);

template<>
__device__ __forceinline__ float to_float<float>(float v) { return v; }

template<>
__device__ __forceinline__ float to_float<__half>(__half v) { return __half2float(v); }

template<>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template<typename T>
__device__ __forceinline__ T from_float(float v);

template<>
__device__ __forceinline__ float from_float<float>(float v) { return v; }

template<>
__device__ __forceinline__ __half from_float<__half>(float v) { return __float2half(v); }

template<>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

template<int NUM_WARPS>
__device__ __forceinline__ float block_reduce_max(float v, float* warp_smem, float* block_smem) {
    const int lane = threadIdx.x & (kThreadsPerWarp - 1);
    const int wid = threadIdx.x >> 5;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    if (lane == 0) {
        warp_smem[wid] = v;
    }
    __syncthreads();
    if (wid == 0) {
        float total = lane < NUM_WARPS ? warp_smem[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            total = fmaxf(total, __shfl_down_sync(0xffffffffu, total, offset));
        }
        if (lane == 0) {
            *block_smem = total;
        }
    }
    __syncthreads();
    return *block_smem;
}

template<typename T>
__device__ __forceinline__ void convrot_fht_stage64_typed(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int lane,
    int stride)
{
    const int base = (lane % stride) + (lane / stride) * (4 * stride);
    const float x0 = to_float(src[base]);
    const float x1 = to_float(src[base + stride]);
    const float x2 = to_float(src[base + 2 * stride]);
    const float x3 = to_float(src[base + 3 * stride]);
    dst[base] = from_float<T>(0.5f * (x0 + x1 + x2 - x3));
    dst[base + stride] = from_float<T>(0.5f * (x0 + x1 - x2 + x3));
    dst[base + 2 * stride] = from_float<T>(0.5f * (x0 - x1 + x2 + x3));
    dst[base + 3 * stride] = from_float<T>(0.5f * (-x0 + x1 + x2 + x3));
}

template<typename T>
__device__ __forceinline__ void convrot_fht_stage64_store_typed(
    const T* __restrict__ src,
    T* __restrict__ output,
    int lane,
    int stride)
{
    const int base = (lane % stride) + (lane / stride) * (4 * stride);
    const float x0 = to_float(src[base]);
    const float x1 = to_float(src[base + stride]);
    const float x2 = to_float(src[base + 2 * stride]);
    const float x3 = to_float(src[base + 3 * stride]);
    output[base] = from_float<T>(0.5f * (x0 + x1 + x2 - x3));
    output[base + stride] = from_float<T>(0.5f * (x0 + x1 - x2 + x3));
    output[base + 2 * stride] = from_float<T>(0.5f * (x0 - x1 + x2 + x3));
    output[base + 3 * stride] = from_float<T>(0.5f * (-x0 + x1 + x2 + x3));
}

template<typename T>
__device__ __forceinline__ float convrot_fht_stage64_store_absmax_typed(
    const T* __restrict__ src,
    T* __restrict__ output,
    int lane,
    int stride)
{
    const int base = (lane % stride) + (lane / stride) * (4 * stride);
    const float x0 = to_float(src[base]);
    const float x1 = to_float(src[base + stride]);
    const float x2 = to_float(src[base + 2 * stride]);
    const float x3 = to_float(src[base + 3 * stride]);
    const T y0_t = from_float<T>(0.5f * (x0 + x1 + x2 - x3));
    const T y1_t = from_float<T>(0.5f * (x0 + x1 - x2 + x3));
    const T y2_t = from_float<T>(0.5f * (x0 - x1 + x2 + x3));
    const T y3_t = from_float<T>(0.5f * (-x0 + x1 + x2 + x3));
    output[base] = y0_t;
    output[base + stride] = y1_t;
    output[base + 2 * stride] = y2_t;
    output[base + 3 * stride] = y3_t;
    const float y0 = to_float(y0_t);
    const float y1 = to_float(y1_t);
    const float y2 = to_float(y2_t);
    const float y3 = to_float(y3_t);
    return fmaxf(fmaxf(fabsf(y0), fabsf(y1)), fmaxf(fabsf(y2), fabsf(y3)));
}

template<int S>
__device__ __forceinline__ void convrot_fht_stage64_float(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int lane)
{
    const int base = (lane % S) + (lane / S) * (4 * S);
    const float x0 = src[base];
    const float x1 = src[base + S];
    const float x2 = src[base + 2 * S];
    const float x3 = src[base + 3 * S];
    dst[base] = 0.5f * (x0 + x1 + x2 - x3);
    dst[base + S] = 0.5f * (x0 + x1 - x2 + x3);
    dst[base + 2 * S] = 0.5f * (x0 - x1 + x2 + x3);
    dst[base + 3 * S] = 0.5f * (-x0 + x1 + x2 + x3);
}

template<int S>
__device__ __forceinline__ float convrot_fht_stage64_store_absmax_float(
    const float* __restrict__ src,
    float* __restrict__ output,
    int lane)
{
    const int base = (lane % S) + (lane / S) * (4 * S);
    const float x0 = src[base];
    const float x1 = src[base + S];
    const float x2 = src[base + 2 * S];
    const float x3 = src[base + 3 * S];
    const float y0 = 0.5f * (x0 + x1 + x2 - x3);
    const float y1 = 0.5f * (x0 + x1 - x2 + x3);
    const float y2 = 0.5f * (x0 - x1 + x2 + x3);
    const float y3 = 0.5f * (-x0 + x1 + x2 + x3);
    output[base] = y0;
    output[base + S] = y1;
    output[base + 2 * S] = y2;
    output[base + 3 * S] = y3;
    return fmaxf(fmaxf(fabsf(y0), fabsf(y1)), fmaxf(fabsf(y2), fabsf(y3)));
}

__device__ __forceinline__ __half half_fht_scale(__half v) {
    return __hmul(v, __float2half(0.5f));
}

__device__ __forceinline__ __nv_bfloat16 bfloat16_fht_scale(__nv_bfloat16 v) {
    return __hmul(v, __float2bfloat16(0.5f));
}

template<typename T>
__device__ __forceinline__ void convrot_h4_store_typed(
    T x0,
    T x1,
    T x2,
    T x3,
    T* __restrict__ output,
    int base)
{
    const float f0 = to_float(x0);
    const float f1 = to_float(x1);
    const float f2 = to_float(x2);
    const float f3 = to_float(x3);
    output[base] = from_float<T>(0.5f * (f0 + f1 + f2 - f3));
    output[base + 1] = from_float<T>(0.5f * (f0 + f1 - f2 + f3));
    output[base + 2] = from_float<T>(0.5f * (f0 - f1 + f2 + f3));
    output[base + 3] = from_float<T>(0.5f * (-f0 + f1 + f2 + f3));
}

__device__ __forceinline__ void convrot_h4_store_typed(
    __half x0,
    __half x1,
    __half x2,
    __half x3,
    __half* __restrict__ output,
    int base)
{
    const __half x01 = __hadd(x0, x1);
    const __half x0m1 = __hsub(x0, x1);
    const __half x23 = __hadd(x2, x3);
    const __half x2m3 = __hsub(x2, x3);
    output[base] = half_fht_scale(__hadd(x01, x2m3));
    output[base + 1] = half_fht_scale(__hsub(x01, x2m3));
    output[base + 2] = half_fht_scale(__hadd(x0m1, x23));
    output[base + 3] = half_fht_scale(__hsub(x23, x0m1));
}

__device__ __forceinline__ void convrot_h4_store_typed(
    __nv_bfloat16 x0,
    __nv_bfloat16 x1,
    __nv_bfloat16 x2,
    __nv_bfloat16 x3,
    __nv_bfloat16* __restrict__ output,
    int base)
{
    const __nv_bfloat16 x01 = __hadd(x0, x1);
    const __nv_bfloat16 x0m1 = __hsub(x0, x1);
    const __nv_bfloat16 x23 = __hadd(x2, x3);
    const __nv_bfloat16 x2m3 = __hsub(x2, x3);
    output[base] = bfloat16_fht_scale(__hadd(x01, x2m3));
    output[base + 1] = bfloat16_fht_scale(__hsub(x01, x2m3));
    output[base + 2] = bfloat16_fht_scale(__hadd(x0m1, x23));
    output[base + 3] = bfloat16_fht_scale(__hsub(x23, x0m1));
}

template<typename T>
__device__ __forceinline__ void dequant_int4_h4_store_typed(
    int q0,
    int q1,
    int q2,
    int q3,
    float scale,
    T* __restrict__ output,
    int base)
{
    const float first_stage_scale = 0.5f * scale;
    output[base] = from_float<T>(static_cast<float>(q0 + q1 + q2 - q3) * first_stage_scale);
    output[base + 1] = from_float<T>(static_cast<float>(q0 + q1 - q2 + q3) * first_stage_scale);
    output[base + 2] = from_float<T>(static_cast<float>(q0 - q1 + q2 + q3) * first_stage_scale);
    output[base + 3] = from_float<T>(static_cast<float>(-q0 + q1 + q2 + q3) * first_stage_scale);
}

__device__ __forceinline__ void dequant_int4_h4_store_typed(
    int q0,
    int q1,
    int q2,
    int q3,
    float scale,
    __half* __restrict__ output,
    int base)
{
    const __half2 s = __float2half2_rn(0.5f * scale);
    const __half2 y01 = __hmul2(
        __floats2half2_rn(
            static_cast<float>(q0 + q1 + q2 - q3),
            static_cast<float>(q0 + q1 - q2 + q3)),
        s);
    const __half2 y23 = __hmul2(
        __floats2half2_rn(
            static_cast<float>(q0 - q1 + q2 + q3),
            static_cast<float>(-q0 + q1 + q2 + q3)),
        s);
    *reinterpret_cast<__half2*>(output + base) = y01;
    *reinterpret_cast<__half2*>(output + base + 2) = y23;
}

__device__ __forceinline__ void dequant_int4_h4_store_typed(
    int q0,
    int q1,
    int q2,
    int q3,
    float scale,
    __nv_bfloat16* __restrict__ output,
    int base)
{
    const __nv_bfloat162 s = __float2bfloat162_rn(0.5f * scale);
    const __nv_bfloat162 y01 = __hmul2(
        __floats2bfloat162_rn(
            static_cast<float>(q0 + q1 + q2 - q3),
            static_cast<float>(q0 + q1 - q2 + q3)),
        s);
    const __nv_bfloat162 y23 = __hmul2(
        __floats2bfloat162_rn(
            static_cast<float>(q0 - q1 + q2 + q3),
            static_cast<float>(-q0 + q1 + q2 + q3)),
        s);
    *reinterpret_cast<__nv_bfloat162*>(output + base) = y01;
    *reinterpret_cast<__nv_bfloat162*>(output + base + 2) = y23;
}

__device__ __forceinline__ void convrot_fht_stage64_typed(
    const __half* __restrict__ src,
    __half* __restrict__ dst,
    int lane,
    int stride)
{
    const int base = (lane % stride) + (lane / stride) * (4 * stride);
    const __half x0 = src[base];
    const __half x1 = src[base + stride];
    const __half x2 = src[base + 2 * stride];
    const __half x3 = src[base + 3 * stride];
    const __half x01 = __hadd(x0, x1);
    const __half x0m1 = __hsub(x0, x1);
    const __half x23 = __hadd(x2, x3);
    const __half x2m3 = __hsub(x2, x3);
    dst[base] = half_fht_scale(__hadd(x01, x2m3));
    dst[base + stride] = half_fht_scale(__hsub(x01, x2m3));
    dst[base + 2 * stride] = half_fht_scale(__hadd(x0m1, x23));
    dst[base + 3 * stride] = half_fht_scale(__hsub(x23, x0m1));
}

__device__ __forceinline__ void convrot_fht_stage64_typed(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int lane,
    int stride)
{
    const int base = (lane % stride) + (lane / stride) * (4 * stride);
    const __nv_bfloat16 x0 = src[base];
    const __nv_bfloat16 x1 = src[base + stride];
    const __nv_bfloat16 x2 = src[base + 2 * stride];
    const __nv_bfloat16 x3 = src[base + 3 * stride];
    const __nv_bfloat16 x01 = __hadd(x0, x1);
    const __nv_bfloat16 x0m1 = __hsub(x0, x1);
    const __nv_bfloat16 x23 = __hadd(x2, x3);
    const __nv_bfloat16 x2m3 = __hsub(x2, x3);
    dst[base] = bfloat16_fht_scale(__hadd(x01, x2m3));
    dst[base + stride] = bfloat16_fht_scale(__hsub(x01, x2m3));
    dst[base + 2 * stride] = bfloat16_fht_scale(__hadd(x0m1, x23));
    dst[base + 3 * stride] = bfloat16_fht_scale(__hsub(x23, x0m1));
}

template<typename T>
__device__ __forceinline__ void convrot_fht_stage64_vec2_typed(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int lane,
    int stride)
{
    convrot_fht_stage64_typed(src, dst, lane, stride);
}

__device__ __forceinline__ void convrot_fht_stage64_vec2_typed(
    const __half* __restrict__ src,
    __half* __restrict__ dst,
    int lane,
    int stride)
{
    if (lane & 1) {
        return;
    }
    const int base = (lane % stride) + (lane / stride) * (4 * stride);
    const __half2 x0 = *reinterpret_cast<const __half2*>(src + base);
    const __half2 x1 = *reinterpret_cast<const __half2*>(src + base + stride);
    const __half2 x2 = *reinterpret_cast<const __half2*>(src + base + 2 * stride);
    const __half2 x3 = *reinterpret_cast<const __half2*>(src + base + 3 * stride);
    const __half2 x01 = __hadd2(x0, x1);
    const __half2 x0m1 = __hsub2(x0, x1);
    const __half2 x23 = __hadd2(x2, x3);
    const __half2 x2m3 = __hsub2(x2, x3);
    const __half2 s = __float2half2_rn(0.5f);
    *reinterpret_cast<__half2*>(dst + base) = __hmul2(__hadd2(x01, x2m3), s);
    *reinterpret_cast<__half2*>(dst + base + stride) = __hmul2(__hsub2(x01, x2m3), s);
    *reinterpret_cast<__half2*>(dst + base + 2 * stride) = __hmul2(__hadd2(x0m1, x23), s);
    *reinterpret_cast<__half2*>(dst + base + 3 * stride) = __hmul2(__hsub2(x23, x0m1), s);
}

__device__ __forceinline__ void convrot_fht_stage64_vec2_typed(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int lane,
    int stride)
{
    if (lane & 1) {
        return;
    }
    const int base = (lane % stride) + (lane / stride) * (4 * stride);
    const __nv_bfloat162 x0 = *reinterpret_cast<const __nv_bfloat162*>(src + base);
    const __nv_bfloat162 x1 = *reinterpret_cast<const __nv_bfloat162*>(src + base + stride);
    const __nv_bfloat162 x2 = *reinterpret_cast<const __nv_bfloat162*>(src + base + 2 * stride);
    const __nv_bfloat162 x3 = *reinterpret_cast<const __nv_bfloat162*>(src + base + 3 * stride);
    const __nv_bfloat162 x01 = __hadd2(x0, x1);
    const __nv_bfloat162 x0m1 = __hsub2(x0, x1);
    const __nv_bfloat162 x23 = __hadd2(x2, x3);
    const __nv_bfloat162 x2m3 = __hsub2(x2, x3);
    const __nv_bfloat162 s = __float2bfloat162_rn(0.5f);
    *reinterpret_cast<__nv_bfloat162*>(dst + base) = __hmul2(__hadd2(x01, x2m3), s);
    *reinterpret_cast<__nv_bfloat162*>(dst + base + stride) = __hmul2(__hsub2(x01, x2m3), s);
    *reinterpret_cast<__nv_bfloat162*>(dst + base + 2 * stride) = __hmul2(__hadd2(x0m1, x23), s);
    *reinterpret_cast<__nv_bfloat162*>(dst + base + 3 * stride) = __hmul2(__hsub2(x23, x0m1), s);
}

__device__ __forceinline__ void convrot_fht_stage64_store_typed(
    const __half* __restrict__ src,
    __half* __restrict__ output,
    int lane,
    int stride)
{
    convrot_fht_stage64_typed(src, output, lane, stride);
}

__device__ __forceinline__ void convrot_fht_stage64_store_typed(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ output,
    int lane,
    int stride)
{
    convrot_fht_stage64_typed(src, output, lane, stride);
}

template<typename T>
__device__ __forceinline__ void convrot_fht_stage64_store_final_typed(
    const T* __restrict__ src,
    T* __restrict__ output,
    int lane)
{
    convrot_fht_stage64_store_typed(src, output, lane, 64);
}

__device__ __forceinline__ void convrot_fht_stage64_store_final_typed(
    const __half* __restrict__ src,
    __half* __restrict__ output,
    int lane)
{
    if (lane >= 32) {
        return;
    }
    const int col = lane * 2;
    const __half2 x0 = *reinterpret_cast<const __half2*>(src + col);
    const __half2 x1 = *reinterpret_cast<const __half2*>(src + col + 64);
    const __half2 x2 = *reinterpret_cast<const __half2*>(src + col + 128);
    const __half2 x3 = *reinterpret_cast<const __half2*>(src + col + 192);
    const __half2 x01 = __hadd2(x0, x1);
    const __half2 x0m1 = __hsub2(x0, x1);
    const __half2 x23 = __hadd2(x2, x3);
    const __half2 x2m3 = __hsub2(x2, x3);
    const __half2 s = __float2half2_rn(0.5f);
    *reinterpret_cast<__half2*>(output + col) = __hmul2(__hadd2(x01, x2m3), s);
    *reinterpret_cast<__half2*>(output + col + 64) = __hmul2(__hsub2(x01, x2m3), s);
    *reinterpret_cast<__half2*>(output + col + 128) = __hmul2(__hadd2(x0m1, x23), s);
    *reinterpret_cast<__half2*>(output + col + 192) = __hmul2(__hsub2(x23, x0m1), s);
}

__device__ __forceinline__ void convrot_fht_stage64_store_final_typed(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ output,
    int lane)
{
    if (lane >= 32) {
        return;
    }
    const int col = lane * 2;
    const __nv_bfloat162 x0 = *reinterpret_cast<const __nv_bfloat162*>(src + col);
    const __nv_bfloat162 x1 = *reinterpret_cast<const __nv_bfloat162*>(src + col + 64);
    const __nv_bfloat162 x2 = *reinterpret_cast<const __nv_bfloat162*>(src + col + 128);
    const __nv_bfloat162 x3 = *reinterpret_cast<const __nv_bfloat162*>(src + col + 192);
    const __nv_bfloat162 x01 = __hadd2(x0, x1);
    const __nv_bfloat162 x0m1 = __hsub2(x0, x1);
    const __nv_bfloat162 x23 = __hadd2(x2, x3);
    const __nv_bfloat162 x2m3 = __hsub2(x2, x3);
    const __nv_bfloat162 s = __float2bfloat162_rn(0.5f);
    *reinterpret_cast<__nv_bfloat162*>(output + col) = __hmul2(__hadd2(x01, x2m3), s);
    *reinterpret_cast<__nv_bfloat162*>(output + col + 64) = __hmul2(__hsub2(x01, x2m3), s);
    *reinterpret_cast<__nv_bfloat162*>(output + col + 128) = __hmul2(__hadd2(x0m1, x23), s);
    *reinterpret_cast<__nv_bfloat162*>(output + col + 192) = __hmul2(__hsub2(x23, x0m1), s);
}

__device__ __forceinline__ float convrot_fht_stage64_store_absmax_typed(
    const __half* __restrict__ src,
    __half* __restrict__ output,
    int lane,
    int stride)
{
    if (stride == 64) {
        if (lane >= 32) {
            return 0.0f;
        }
        const int col = lane * 2;
        const __half2 x0 = *reinterpret_cast<const __half2*>(src + col);
        const __half2 x1 = *reinterpret_cast<const __half2*>(src + col + 64);
        const __half2 x2 = *reinterpret_cast<const __half2*>(src + col + 128);
        const __half2 x3 = *reinterpret_cast<const __half2*>(src + col + 192);
        const __half2 x01 = __hadd2(x0, x1);
        const __half2 x0m1 = __hsub2(x0, x1);
        const __half2 x23 = __hadd2(x2, x3);
        const __half2 x2m3 = __hsub2(x2, x3);
        const __half2 s = __float2half2_rn(0.5f);
        const __half2 y0 = __hmul2(__hadd2(x01, x2m3), s);
        const __half2 y1 = __hmul2(__hsub2(x01, x2m3), s);
        const __half2 y2 = __hmul2(__hadd2(x0m1, x23), s);
        const __half2 y3 = __hmul2(__hsub2(x23, x0m1), s);
        *reinterpret_cast<__half2*>(output + col) = y0;
        *reinterpret_cast<__half2*>(output + col + 64) = y1;
        *reinterpret_cast<__half2*>(output + col + 128) = y2;
        *reinterpret_cast<__half2*>(output + col + 192) = y3;
        const float2 f0 = __half22float2(y0);
        const float2 f1 = __half22float2(y1);
        const float2 f2 = __half22float2(y2);
        const float2 f3 = __half22float2(y3);
        float v = fmaxf(fabsf(f0.x), fabsf(f0.y));
        v = fmaxf(v, fmaxf(fabsf(f1.x), fabsf(f1.y)));
        v = fmaxf(v, fmaxf(fabsf(f2.x), fabsf(f2.y)));
        v = fmaxf(v, fmaxf(fabsf(f3.x), fabsf(f3.y)));
        return v;
    }
    const int base = (lane % stride) + (lane / stride) * (4 * stride);
    const __half x0 = src[base];
    const __half x1 = src[base + stride];
    const __half x2 = src[base + 2 * stride];
    const __half x3 = src[base + 3 * stride];
    const __half x01 = __hadd(x0, x1);
    const __half x0m1 = __hsub(x0, x1);
    const __half x23 = __hadd(x2, x3);
    const __half x2m3 = __hsub(x2, x3);
    const __half y0 = half_fht_scale(__hadd(x01, x2m3));
    const __half y1 = half_fht_scale(__hsub(x01, x2m3));
    const __half y2 = half_fht_scale(__hadd(x0m1, x23));
    const __half y3 = half_fht_scale(__hsub(x23, x0m1));
    output[base] = y0;
    output[base + stride] = y1;
    output[base + 2 * stride] = y2;
    output[base + 3 * stride] = y3;
    const float f0 = fabsf(__half2float(y0));
    const float f1 = fabsf(__half2float(y1));
    const float f2 = fabsf(__half2float(y2));
    const float f3 = fabsf(__half2float(y3));
    return fmaxf(fmaxf(f0, f1), fmaxf(f2, f3));
}

__device__ __forceinline__ float convrot_fht_stage64_store_absmax_typed(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ output,
    int lane,
    int stride)
{
    if (stride == 64) {
        if (lane >= 32) {
            return 0.0f;
        }
        const int col = lane * 2;
        const __nv_bfloat162 x0 = *reinterpret_cast<const __nv_bfloat162*>(src + col);
        const __nv_bfloat162 x1 = *reinterpret_cast<const __nv_bfloat162*>(src + col + 64);
        const __nv_bfloat162 x2 = *reinterpret_cast<const __nv_bfloat162*>(src + col + 128);
        const __nv_bfloat162 x3 = *reinterpret_cast<const __nv_bfloat162*>(src + col + 192);
        const __nv_bfloat162 x01 = __hadd2(x0, x1);
        const __nv_bfloat162 x0m1 = __hsub2(x0, x1);
        const __nv_bfloat162 x23 = __hadd2(x2, x3);
        const __nv_bfloat162 x2m3 = __hsub2(x2, x3);
        const __nv_bfloat162 s = __float2bfloat162_rn(0.5f);
        const __nv_bfloat162 y0 = __hmul2(__hadd2(x01, x2m3), s);
        const __nv_bfloat162 y1 = __hmul2(__hsub2(x01, x2m3), s);
        const __nv_bfloat162 y2 = __hmul2(__hadd2(x0m1, x23), s);
        const __nv_bfloat162 y3 = __hmul2(__hsub2(x23, x0m1), s);
        *reinterpret_cast<__nv_bfloat162*>(output + col) = y0;
        *reinterpret_cast<__nv_bfloat162*>(output + col + 64) = y1;
        *reinterpret_cast<__nv_bfloat162*>(output + col + 128) = y2;
        *reinterpret_cast<__nv_bfloat162*>(output + col + 192) = y3;
        const float2 f0 = __bfloat1622float2(y0);
        const float2 f1 = __bfloat1622float2(y1);
        const float2 f2 = __bfloat1622float2(y2);
        const float2 f3 = __bfloat1622float2(y3);
        float v = fmaxf(fabsf(f0.x), fabsf(f0.y));
        v = fmaxf(v, fmaxf(fabsf(f1.x), fabsf(f1.y)));
        v = fmaxf(v, fmaxf(fabsf(f2.x), fabsf(f2.y)));
        v = fmaxf(v, fmaxf(fabsf(f3.x), fabsf(f3.y)));
        return v;
    }
    const int base = (lane % stride) + (lane / stride) * (4 * stride);
    const __nv_bfloat16 x0 = src[base];
    const __nv_bfloat16 x1 = src[base + stride];
    const __nv_bfloat16 x2 = src[base + 2 * stride];
    const __nv_bfloat16 x3 = src[base + 3 * stride];
    const __nv_bfloat16 x01 = __hadd(x0, x1);
    const __nv_bfloat16 x0m1 = __hsub(x0, x1);
    const __nv_bfloat16 x23 = __hadd(x2, x3);
    const __nv_bfloat16 x2m3 = __hsub(x2, x3);
    const __nv_bfloat16 y0 = bfloat16_fht_scale(__hadd(x01, x2m3));
    const __nv_bfloat16 y1 = bfloat16_fht_scale(__hsub(x01, x2m3));
    const __nv_bfloat16 y2 = bfloat16_fht_scale(__hadd(x0m1, x23));
    const __nv_bfloat16 y3 = bfloat16_fht_scale(__hsub(x23, x0m1));
    output[base] = y0;
    output[base + stride] = y1;
    output[base + 2 * stride] = y2;
    output[base + 3 * stride] = y3;
    const float f0 = fabsf(__bfloat162float(y0));
    const float f1 = fabsf(__bfloat162float(y1));
    const float f2 = fabsf(__bfloat162float(y2));
    const float f3 = fabsf(__bfloat162float(y3));
    return fmaxf(fmaxf(f0, f1), fmaxf(f2, f3));
}

template<bool STOCHASTIC>
__device__ __forceinline__ uint8_t quantize_int4_pair_byte(
    float x0,
    float x1,
    float inv_scale,
    uint64_t seed,
    int64_t idx0)
{
    const int q0 = quantize_int4_value<STOCHASTIC>(x0, inv_scale, seed, idx0);
    const int q1 = quantize_int4_value<STOCHASTIC>(x1, inv_scale, seed, idx0 + 1);
    return static_cast<uint8_t>(pack_int4_pair(q0, q1));
}

template<typename T, bool STOCHASTIC>
__device__ __forceinline__ uint32_t quantize_int4_pack4_word(
    const T* __restrict__ row_buf,
    int col,
    float inv_scale,
    uint64_t seed,
    int64_t row_offset)
{
    const uint32_t b0 = quantize_int4_pair_byte<STOCHASTIC>(to_float(row_buf[col]), to_float(row_buf[col + 1]), inv_scale, seed, row_offset + col);
    const uint32_t b1 = quantize_int4_pair_byte<STOCHASTIC>(to_float(row_buf[col + 2]), to_float(row_buf[col + 3]), inv_scale, seed, row_offset + col + 2);
    const uint32_t b2 = quantize_int4_pair_byte<STOCHASTIC>(to_float(row_buf[col + 4]), to_float(row_buf[col + 5]), inv_scale, seed, row_offset + col + 4);
    const uint32_t b3 = quantize_int4_pair_byte<STOCHASTIC>(to_float(row_buf[col + 6]), to_float(row_buf[col + 7]), inv_scale, seed, row_offset + col + 6);
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

template<typename T, bool STOCHASTIC>
__device__ __forceinline__ uint32_t quantize_int4_values4_int8_word(
    const T* __restrict__ row_buf,
    int col,
    float inv_scale,
    uint64_t seed,
    int64_t row_offset)
{
    const uint32_t q0 = static_cast<uint8_t>(
        quantize_int4_value<STOCHASTIC>(to_float(row_buf[col]), inv_scale, seed, row_offset + col));
    const uint32_t q1 = static_cast<uint8_t>(
        quantize_int4_value<STOCHASTIC>(to_float(row_buf[col + 1]), inv_scale, seed, row_offset + col + 1));
    const uint32_t q2 = static_cast<uint8_t>(
        quantize_int4_value<STOCHASTIC>(to_float(row_buf[col + 2]), inv_scale, seed, row_offset + col + 2));
    const uint32_t q3 = static_cast<uint8_t>(
        quantize_int4_value<STOCHASTIC>(to_float(row_buf[col + 3]), inv_scale, seed, row_offset + col + 3));
    return q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
}

template<typename InType, int BLOCK_THREADS, bool STOCHASTIC>
__global__ void quantize_int4_rowwise_kernel(
    const InType* __restrict__ x,
    int8_t* __restrict__ q,
    float* __restrict__ scales,
    int K,
    uint64_t seed)
{
    constexpr int kWarps = BLOCK_THREADS / kThreadsPerWarp;
    __shared__ float warp_smem[kWarps];
    __shared__ float block_smem;

    const int row = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;
    const int64_t row_offset = static_cast<int64_t>(row) * K;

    float absmax = 0.0f;
    for (int col = tid; col < K; col += BLOCK_THREADS) {
        absmax = fmaxf(absmax, fabsf(to_float(x[row_offset + col])));
    }
    absmax = block_reduce_max<kWarps>(absmax, warp_smem, &block_smem);
    const float scale = fmaxf(
        finite_absmax_for_int4_scale<InType>(absmax) * (1.0f / static_cast<float>(kInt4Max)),
        1.0e-10f);
    if (tid == 0) {
        scales[row] = scale;
    }
    const float inv_scale = 1.0f / scale;

    const int K_half = K / 2;
    for (int packed_col = tid; packed_col < K_half; packed_col += BLOCK_THREADS) {
        const int col0 = packed_col * 2;
        const int col1 = col0 + 1;
        const int q0 = quantize_int4_value<STOCHASTIC>(to_float(x[row_offset + col0]), inv_scale, seed, row_offset + col0);
        const int q1 = quantize_int4_value<STOCHASTIC>(to_float(x[row_offset + col1]), inv_scale, seed, row_offset + col1);
        q[static_cast<int64_t>(row) * K_half + packed_col] = pack_int4_pair(q0, q1);
    }
}

template<typename InType, int BLOCK_THREADS, bool PACK4, bool STOCHASTIC, bool OUTPUT_INT8 = false>
__global__ void quantize_int4_rowwise_convrot64_kernel(
    const InType* __restrict__ x,
    int8_t* __restrict__ q,
    float* __restrict__ scales,
    int K,
    uint64_t seed)
{
    constexpr int kGroupThreads = 64;
    constexpr int kGroupsInFlight = BLOCK_THREADS / kGroupThreads;
    constexpr int kWarps = BLOCK_THREADS / kThreadsPerWarp;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    InType* row_buf = reinterpret_cast<InType*>(smem_raw);
    InType* tmp = row_buf + K;

    __shared__ float warp_smem[kWarps];
    __shared__ float block_smem;

    const int row = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;
    const int sub = tid / kGroupThreads;
    const int lane = tid % kGroupThreads;
    const int64_t row_offset = static_cast<int64_t>(row) * K;
    const int n_groups = K / kConvRotGroup;

    float abs_max = 0.0f;

    if constexpr (PACK4) {
        static_assert(BLOCK_THREADS == 640 || BLOCK_THREADS == 768 || BLOCK_THREADS == 960, "PACK4 path is specialized for K=15360");
        #pragma unroll
        for (int it = 0; it < 60 / kGroupsInFlight; ++it) {
            const int group = it * kGroupsInFlight + sub;
            const int base = lane * 4;
            const int group_col = group * kConvRotGroup;
            const int64_t x_offset = row_offset + group_col + base;
            InType* group_buf = row_buf + group_col;
            InType* buf0 = tmp + sub * kConvRotGroup;

            convrot_h4_store_typed(
                x[x_offset],
                x[x_offset + 1],
                x[x_offset + 2],
                x[x_offset + 3],
                group_buf,
                base);
            __syncwarp();

            convrot_fht_stage64_vec2_typed(group_buf, buf0, lane, 4);
            __syncwarp();
            convrot_fht_stage64_vec2_typed(buf0, group_buf, lane, 16);
            __syncthreads();

            abs_max = fmaxf(
                abs_max,
                convrot_fht_stage64_store_absmax_typed(group_buf, group_buf, lane, 64));
        }
    } else {
        InType* buf0 = tmp + sub * (2 * kConvRotGroup);
        InType* buf1 = buf0 + kConvRotGroup;
        const int iters = (n_groups + kGroupsInFlight - 1) / kGroupsInFlight;
        for (int it = 0; it < iters; ++it) {
            const int group = it * kGroupsInFlight + sub;
            const bool active = group < n_groups;
            const int base = lane * 4;
            const int group_col = group * kConvRotGroup;
            const int64_t x_offset = row_offset + group_col + base;

            const InType x0 = active ? x[x_offset] : from_float<InType>(0.0f);
            const InType x1 = active ? x[x_offset + 1] : from_float<InType>(0.0f);
            const InType x2 = active ? x[x_offset + 2] : from_float<InType>(0.0f);
            const InType x3 = active ? x[x_offset + 3] : from_float<InType>(0.0f);
            convrot_h4_store_typed(x0, x1, x2, x3, buf1, base);
            __syncwarp();

            convrot_fht_stage64_vec2_typed(buf1, buf0, lane, 4);
            __syncwarp();
            convrot_fht_stage64_vec2_typed(buf0, buf1, lane, 16);
            __syncthreads();

            if (active) {
                abs_max = fmaxf(
                    abs_max,
                    convrot_fht_stage64_store_absmax_typed(buf1, row_buf + group_col, lane, 64));
            }
            __syncthreads();
        }
    }

    abs_max = block_reduce_max<kWarps>(abs_max, warp_smem, &block_smem);
    const float scale = fmaxf(
        finite_absmax_for_int4_scale<InType>(abs_max) * (1.0f / static_cast<float>(kInt4Max)),
        1.0e-10f);
    if (tid == 0) {
        scales[row] = scale;
    }
    const float inv_scale = 1.0f / scale;

    if constexpr (OUTPUT_INT8) {
        int8_t* q_row = q + static_cast<int64_t>(row) * K;
        uint32_t* q_words = reinterpret_cast<uint32_t*>(q_row);
        const int word_count = K / 4;
        for (int word = tid; word < word_count; word += BLOCK_THREADS) {
            q_words[word] = quantize_int4_values4_int8_word<InType, STOCHASTIC>(
                row_buf, word * 4, inv_scale, seed, row_offset);
        }
    } else if constexpr (PACK4) {
        const int K_half = K / 2;
        int8_t* q_row = q + static_cast<int64_t>(row) * K_half;
        uint32_t* q_words = reinterpret_cast<uint32_t*>(q_row);
        constexpr int kWordCount = 15360 / 8;
        for (int word = tid; word < kWordCount; word += BLOCK_THREADS) {
            q_words[word] = quantize_int4_pack4_word<InType, STOCHASTIC>(row_buf, word * 8, inv_scale, seed, row_offset);
        }
    } else {
        const int K_half = K / 2;
        int8_t* q_row = q + static_cast<int64_t>(row) * K_half;
        uint32_t* q_words = reinterpret_cast<uint32_t*>(q_row);
        const int word_count = K / 8;
        for (int word = tid; word < word_count; word += BLOCK_THREADS) {
            q_words[word] = quantize_int4_pack4_word<InType, STOCHASTIC>(row_buf, word * 8, inv_scale, seed, row_offset);
        }
    }
}

template<typename InType, int BLOCK_THREADS, bool STOCHASTIC>
__global__ void quantize_int4_rowwise_convrot64_to_int8_float_kernel(
    const InType* __restrict__ x,
    int8_t* __restrict__ q,
    float* __restrict__ scales,
    int K,
    uint64_t seed)
{
    constexpr int kGroupThreads = 64;
    constexpr int kGroupsInFlight = BLOCK_THREADS / kGroupThreads;
    constexpr int kWarps = BLOCK_THREADS / kThreadsPerWarp;

    extern __shared__ float smem[];
    float* row_buf = smem;
    float* tmp = row_buf + K;

    __shared__ float warp_smem[kWarps];
    __shared__ float block_smem;

    const int row = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;
    const int sub = tid / kGroupThreads;
    const int lane = tid % kGroupThreads;
    const int64_t row_offset = static_cast<int64_t>(row) * K;
    const int n_groups = K / kConvRotGroup;

    float* buf0 = tmp + sub * (2 * kConvRotGroup);
    float* buf1 = buf0 + kConvRotGroup;
    float abs_max = 0.0f;

    const int iters = (n_groups + kGroupsInFlight - 1) / kGroupsInFlight;
    for (int it = 0; it < iters; ++it) {
        const int group = it * kGroupsInFlight + sub;
        const bool active = group < n_groups;
        const int base = lane * 4;
        const int group_col = group * kConvRotGroup;
        const int64_t x_offset = row_offset + group_col + base;

        const float x0 = active ? to_float(x[x_offset]) : 0.0f;
        const float x1 = active ? to_float(x[x_offset + 1]) : 0.0f;
        const float x2 = active ? to_float(x[x_offset + 2]) : 0.0f;
        const float x3 = active ? to_float(x[x_offset + 3]) : 0.0f;
        buf1[base] = 0.5f * (x0 + x1 + x2 - x3);
        buf1[base + 1] = 0.5f * (x0 + x1 - x2 + x3);
        buf1[base + 2] = 0.5f * (x0 - x1 + x2 + x3);
        buf1[base + 3] = 0.5f * (-x0 + x1 + x2 + x3);
        __syncthreads();

        convrot_fht_stage64_float<4>(buf1, buf0, lane);
        __syncthreads();
        convrot_fht_stage64_float<16>(buf0, buf1, lane);
        __syncthreads();

        if (active) {
            abs_max = fmaxf(
                abs_max,
                convrot_fht_stage64_store_absmax_float<64>(buf1, row_buf + group_col, lane));
        }
        __syncthreads();
    }

    abs_max = block_reduce_max<kWarps>(abs_max, warp_smem, &block_smem);
    const float scale = fmaxf(
        finite_absmax_for_int4_scale<InType>(abs_max) * (1.0f / static_cast<float>(kInt4Max)),
        1.0e-10f);
    if (tid == 0) {
        scales[row] = scale;
    }
    const float inv_scale = 1.0f / scale;

    for (int col = tid; col < K; col += BLOCK_THREADS) {
        const int64_t idx = row_offset + col;
        const float scaled = row_buf[col] * inv_scale;
        float quantized;
        if constexpr (STOCHASTIC) {
            quantized = floorf(scaled + stochastic_rng_value(idx, seed));
        } else {
            quantized = nearbyintf(scaled);
        }
        quantized = fminf(static_cast<float>(kInt4Max), fmaxf(static_cast<float>(-kInt4Max), quantized));
        q[idx] = static_cast<int8_t>(quantized);
    }
}

template<int GROUP_SIZE, typename InType, int BLOCK_THREADS, bool STOCHASTIC>
__global__ void quantize_int4_rowwise_convrot_small_kernel(
    const InType* __restrict__ x,
    int8_t* __restrict__ q,
    float* __restrict__ scales,
    int K,
    uint64_t seed)
{
    static_assert(GROUP_SIZE == 16 || GROUP_SIZE == 64, "small ConvRot int4 kernel supports group sizes 16 and 64");
    constexpr int kGroupThreads = GROUP_SIZE / 4;
    constexpr int kGroupsInFlight = BLOCK_THREADS / kGroupThreads;
    constexpr int kWarps = BLOCK_THREADS / kThreadsPerWarp;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    InType* row_buf = reinterpret_cast<InType*>(smem_raw);
    InType* tmp = row_buf + K;

    __shared__ float warp_smem[kWarps];
    __shared__ float block_smem;

    const int row = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;
    const int sub = tid / kGroupThreads;
    const int lane = tid % kGroupThreads;
    const int64_t row_offset = static_cast<int64_t>(row) * K;
    const int n_groups = K / GROUP_SIZE;
    const int iters = (n_groups + kGroupsInFlight - 1) / kGroupsInFlight;

    InType* buf0 = tmp + sub * (2 * GROUP_SIZE);
    InType* buf1 = buf0 + GROUP_SIZE;
    float abs_max = 0.0f;

    for (int it = 0; it < iters; ++it) {
        const int group = it * kGroupsInFlight + sub;
        const bool active = group < n_groups;
        const int base = lane * 4;
        const int group_col = group * GROUP_SIZE;
        const int64_t x_offset = row_offset + group_col + base;

        const InType zero = from_float<InType>(0.0f);
        const InType x0 = active ? x[x_offset] : zero;
        const InType x1 = active ? x[x_offset + 1] : zero;
        const InType x2 = active ? x[x_offset + 2] : zero;
        const InType x3 = active ? x[x_offset + 3] : zero;
        convrot_h4_store_typed(x0, x1, x2, x3, buf1, base);
        __syncthreads();

        if constexpr (GROUP_SIZE == 16) {
            if (active) {
                abs_max = fmaxf(
                    abs_max,
                    convrot_fht_stage64_store_absmax_typed(buf1, row_buf + group_col, lane, 4));
            }
        } else {
            convrot_fht_stage64_typed(buf1, buf0, lane, 4);
            __syncthreads();
            if (active) {
                abs_max = fmaxf(
                    abs_max,
                    convrot_fht_stage64_store_absmax_typed(buf0, row_buf + group_col, lane, 16));
            }
        }
        __syncthreads();
    }

    abs_max = block_reduce_max<kWarps>(abs_max, warp_smem, &block_smem);
    const float scale = fmaxf(
        finite_absmax_for_int4_scale<InType>(abs_max) * (1.0f / static_cast<float>(kInt4Max)),
        1.0e-10f);
    if (tid == 0) {
        scales[row] = scale;
    }
    const float inv_scale = 1.0f / scale;

    const int K_half = K / 2;
    int8_t* q_row = q + static_cast<int64_t>(row) * K_half;
    uint32_t* q_words = reinterpret_cast<uint32_t*>(q_row);
    const int word_count = K / 8;
    for (int word = tid; word < word_count; word += BLOCK_THREADS) {
        q_words[word] = quantize_int4_pack4_word<InType, STOCHASTIC>(row_buf, word * 8, inv_scale, seed, row_offset);
    }
}

template<int GROUP_SIZE, int GROUPS_PER_BLOCK, typename OutputType, bool CHECK_BOUNDS, bool SCALE_PER_ROW>
__global__ void dequantize_int4_convrot_small_kernel(
    const int8_t* __restrict__ q,
    const float* __restrict__ scales,
    OutputType* __restrict__ output,
    int K,
    int scale_size)
{
    static_assert(GROUP_SIZE == 16 || GROUP_SIZE == 64, "small ConvRot int4 dequant kernel supports group sizes 16 and 64");
    constexpr int kGroupThreads = GROUP_SIZE / 4;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    OutputType* smem = reinterpret_cast<OutputType*>(smem_raw);

    const int sub = threadIdx.x / kGroupThreads;
    const int lane = threadIdx.x % kGroupThreads;
    const int group = static_cast<int>(blockIdx.y) * GROUPS_PER_BLOCK + sub;
    const int row = static_cast<int>(blockIdx.x);
    const bool active = !CHECK_BOUNDS || group < K / GROUP_SIZE;
    const int64_t row_offset = static_cast<int64_t>(row) * K;
    const int group_col = group * GROUP_SIZE;
    const float scale = scales[SCALE_PER_ROW ? row : 0];

    OutputType* buf0 = smem + sub * (2 * GROUP_SIZE);
    OutputType* buf1 = buf0 + GROUP_SIZE;

    const int base = lane * 4;
    const int64_t q_offset = (row_offset + group_col + base) >> 1;
    uint32_t packed0 = 0;
    uint32_t packed1 = 0;
    if (active) {
        packed0 = static_cast<uint8_t>(q[q_offset]);
        packed1 = static_cast<uint8_t>(q[q_offset + 1]);
    }
    dequant_int4_h4_store_typed(
        unpack_int4_nibble(packed0),
        unpack_int4_nibble(packed0 >> 4),
        unpack_int4_nibble(packed1),
        unpack_int4_nibble(packed1 >> 4),
        scale,
        buf1,
        base);
    __syncthreads();

    if constexpr (GROUP_SIZE == 16) {
        if (active) {
            convrot_fht_stage64_store_typed(buf1, output + row_offset + group_col, lane, 4);
        }
    } else {
        convrot_fht_stage64_typed(buf1, buf0, lane, 4);
        __syncthreads();
        if (active) {
            convrot_fht_stage64_store_typed(buf0, output + row_offset + group_col, lane, 16);
        }
    }
}

template<int GROUPS_PER_BLOCK, typename OutputType, bool CHECK_BOUNDS, bool SCALE_PER_ROW>
__global__ void dequantize_int4_convrot64_kernel(
    const int8_t* __restrict__ q,
    const float* __restrict__ scales,
    OutputType* __restrict__ output,
    int K,
    int scale_size)
{
    constexpr int kGroupThreads = 64;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    OutputType* smem = reinterpret_cast<OutputType*>(smem_raw);

    const int sub = threadIdx.x / kGroupThreads;
    const int lane = threadIdx.x % kGroupThreads;
    const int group = static_cast<int>(blockIdx.y) * GROUPS_PER_BLOCK + sub;
    const int row = static_cast<int>(blockIdx.x);
    const bool active = !CHECK_BOUNDS || group < K / kConvRotGroup;
    const int64_t row_offset = static_cast<int64_t>(row) * K;
    const int group_col = group * kConvRotGroup;
    const int group_col_half = group_col >> 1;
    const float scale = scales[SCALE_PER_ROW ? row : 0];

    OutputType* buf0 = smem + sub * (2 * kConvRotGroup);
    OutputType* buf1 = buf0 + kConvRotGroup;

    const int base = lane * 4;
    const int64_t q_offset = (static_cast<int64_t>(row) * (K >> 1)) + group_col_half + (base >> 1);
    uint32_t packed = 0;
    if constexpr (CHECK_BOUNDS) {
        if (active) {
            packed = static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(q + q_offset));
        }
    } else {
        packed = static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(q + q_offset));
    }
    const int q0 = unpack_int4_nibble(packed);
    const int q1 = unpack_int4_nibble(packed >> 4);
    const int q2 = unpack_int4_nibble(packed >> 8);
    const int q3 = unpack_int4_nibble(packed >> 12);
    dequant_int4_h4_store_typed(q0, q1, q2, q3, scale, buf1, base);
    __syncwarp();

    convrot_fht_stage64_vec2_typed(buf1, buf0, lane, 4);
    __syncwarp();
    convrot_fht_stage64_vec2_typed(buf0, buf1, lane, 16);
    __syncthreads();

    if constexpr (CHECK_BOUNDS) {
        if (active) {
            convrot_fht_stage64_store_final_typed(buf1, output + row_offset + group_col, lane);
        }
    } else {
        convrot_fht_stage64_store_final_typed(buf1, output + row_offset + group_col, lane);
    }
}

template<int GROUPS_PER_BLOCK, typename OutputType, bool CHECK_BOUNDS, bool SCALE_PER_ROW>
__global__ void dequantize_int4_convrot64_warp32_kernel(
    const int8_t* __restrict__ q,
    const float* __restrict__ scales,
    OutputType* __restrict__ output,
    int K,
    int scale_size)
{
    constexpr int kGroupThreads = 32;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    OutputType* smem = reinterpret_cast<OutputType*>(smem_raw);

    const int sub = threadIdx.x / kGroupThreads;
    const int lane = threadIdx.x & (kGroupThreads - 1);
    const int group = static_cast<int>(blockIdx.y) * GROUPS_PER_BLOCK + sub;
    const int row = static_cast<int>(blockIdx.x);
    const bool active = !CHECK_BOUNDS || group < K / kConvRotGroup;
    const int64_t row_offset = static_cast<int64_t>(row) * K;
    const int group_col = group * kConvRotGroup;
    const int group_col_half = group_col >> 1;
    const float scale = scales[SCALE_PER_ROW ? row : 0];

    OutputType* buf0 = smem + sub * (2 * kConvRotGroup);
    OutputType* buf1 = buf0 + kConvRotGroup;

    const int base = lane * 8;
    const int64_t q_offset = (static_cast<int64_t>(row) * (K >> 1)) + group_col_half + (base >> 1);
    uint32_t packed = 0;
    if constexpr (CHECK_BOUNDS) {
        if (active) {
            packed = *reinterpret_cast<const uint32_t*>(q + q_offset);
        }
    } else {
        packed = *reinterpret_cast<const uint32_t*>(q + q_offset);
    }

    dequant_int4_h4_store_typed(
        unpack_int4_nibble(packed),
        unpack_int4_nibble(packed >> 4),
        unpack_int4_nibble(packed >> 8),
        unpack_int4_nibble(packed >> 12),
        scale,
        buf1,
        base);
    dequant_int4_h4_store_typed(
        unpack_int4_nibble(packed >> 16),
        unpack_int4_nibble(packed >> 20),
        unpack_int4_nibble(packed >> 24),
        unpack_int4_nibble(packed >> 28),
        scale,
        buf1,
        base + 4);
    __syncwarp();

    const int pair_lane = lane * 2;
    convrot_fht_stage64_vec2_typed(buf1, buf0, pair_lane, 4);
    __syncwarp();
    convrot_fht_stage64_vec2_typed(buf0, buf1, pair_lane, 16);
    __syncwarp();

    if constexpr (CHECK_BOUNDS) {
        if (active) {
            convrot_fht_stage64_store_final_typed(buf1, output + row_offset + group_col, lane);
        }
    } else {
        convrot_fht_stage64_store_final_typed(buf1, output + row_offset + group_col, lane);
    }
}

constexpr int kStages = 3;
constexpr int kMUnroll = 1;
constexpr int kWarpM = kMUnroll * 16;
constexpr int kNUnroll = 8;
constexpr int kWarpN = kNUnroll * 8;
constexpr int kWarpsM = 2;
constexpr int kWarpsN = 4;
constexpr int kNumWarps = kWarpsM * kWarpsN;
constexpr int kBlockM = kWarpM * kWarpsM;
constexpr int kBlockN = kWarpN * kWarpsN;
constexpr int kBlockKBytes = kGroupSize / 2;
constexpr int kThreadsPerBlock = kNumWarps * 32;
constexpr int kBLoadChunks = kBlockN * 2;
constexpr int kBLoadSweeps = (kBLoadChunks + kThreadsPerBlock - 1) / kThreadsPerBlock;

template<typename OutType, typename BiasType>
__global__ void int4_linear_kernel(
    const int8_t* __restrict__ act,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scales,
    const float* __restrict__ weight_scales,
    const BiasType* __restrict__ bias,
    OutType* __restrict__ out,
    int M,
    int N,
    int K,
    bool has_bias)
{
    const int cta_m = blockIdx.y * kBlockM;
    const int cta_n = blockIdx.x * kBlockN;
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_m = warp_id & (kWarpsM - 1);
    const int warp_n = warp_id / kWarpsM;
    const int groupID = lane >> 2;
    const int tid_in_group = lane & 3;
    const int warp_m_base = cta_m + warp_m * kWarpM;
    const int warp_n_base = cta_n + warp_n * kWarpN;

    int32_t accum[kMUnroll][kNUnroll][4];
    #pragma unroll
    for (int mi = 0; mi < kMUnroll; ++mi) {
        #pragma unroll
        for (int c = 0; c < kNUnroll; ++c) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                accum[mi][c][i] = 0;
            }
        }
    }

    const int K_half = K / 2;
    const int num_groups = K / kGroupSize;

    __shared__ alignas(16) int8_t smem_B[kStages][kBlockN * kBlockKBytes];
    __shared__ alignas(16) int8_t smem_A[kStages][kBlockM * kBlockKBytes];

    auto issue_B_load = [&](int g, int stage) {
        if (g >= num_groups) return;
        const int thread_idx = threadIdx.x;
        #pragma unroll
        for (int sweep = 0; sweep < kBLoadSweeps; ++sweep) {
            const int t = thread_idx + sweep * kThreadsPerBlock;
            if (t < kBlockN * 2) {
                const int n_row = t >> 1;
                const int half = t & 1;
                const int n_global = cta_n + n_row;
                int8_t* dst = &smem_B[stage][n_row * kBlockKBytes + half * 16];
                if (n_global < N) {
                    const int8_t* src = weight + n_global * K_half + g * kBlockKBytes + half * 16;
                    cp_async_16b(dst, src);
                } else {
                    reinterpret_cast<uint4*>(dst)[0] = {0, 0, 0, 0};
                }
            }
        }
    };

    auto issue_A_load = [&](int g, int stage) {
        if (g >= num_groups) return;
        const int t = threadIdx.x;
        if (t < kBlockM * 2) {
            const int m_row = t >> 1;
            const int half = t & 1;
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

    auto load_B_fragment = [&](int stage, int c, uint32_t (&b_reg)[2]) {
        const int b_col_local = (warp_n * kWarpN) + c * 8 + groupID;
        const int b_col_global = cta_n + b_col_local;
        b_reg[0] = b_reg[1] = 0;
        if (b_col_local < kBlockN && b_col_global < N) {
            const int byte0 = tid_in_group * 8;
            const int8_t* row_base = &smem_B[stage][b_col_local * kBlockKBytes];
            b_reg[0] = *reinterpret_cast<const uint32_t*>(row_base + byte0);
            b_reg[1] = *reinterpret_cast<const uint32_t*>(row_base + byte0 + 4);
        }
    };

    #pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        issue_A_load(s, s);
        issue_B_load(s, s);
        cp_async_commit_group();
    }

    for (int g = 0; g < num_groups; ++g) {
        const int next_g = g + kStages - 1;
        if (next_g < num_groups) {
            const int next_stage = (g + kStages - 1) % kStages;
            issue_A_load(next_g, next_stage);
            issue_B_load(next_g, next_stage);
        }
        cp_async_commit_group();
        cp_async_wait_group<kStages - 1>();
        __syncthreads();

        const int cur_stage = g % kStages;
        uint32_t a_reg[kMUnroll][4];
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
        }

        #pragma unroll
        for (int c = 0; c < kNUnroll; ++c) {
            uint32_t b_reg[2];
            load_B_fragment(cur_stage, c, b_reg);
            #pragma unroll
            for (int mi = 0; mi < kMUnroll; ++mi) {
                int32_t zero[4] = {0, 0, 0, 0};
                int32_t d[4];
                mma_m16n8k64_s4s4s32(a_reg[mi], b_reg, zero, d);
                accum[mi][c][0] += d[0];
                accum[mi][c][1] += d[1];
                accum[mi][c][2] += d[2];
                accum[mi][c][3] += d[3];
            }
        }
    }
    cp_async_wait_group<0>();

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
            if (row0_m < M && col0 < N) {
                float v = static_cast<float>(accum[mi][c][0]) * x_scales[row0_m] * weight_scales[col0];
                if (has_bias) v += to_float(bias[col0]);
                out[row0_m * N + col0] = from_float<OutType>(v);
            }
            if (row0_m < M && col1 < N) {
                float v = static_cast<float>(accum[mi][c][1]) * x_scales[row0_m] * weight_scales[col1];
                if (has_bias) v += to_float(bias[col1]);
                out[row0_m * N + col1] = from_float<OutType>(v);
            }
            if (row1_m < M && col0 < N) {
                float v = static_cast<float>(accum[mi][c][2]) * x_scales[row1_m] * weight_scales[col0];
                if (has_bias) v += to_float(bias[col0]);
                out[row1_m * N + col0] = from_float<OutType>(v);
            }
            if (row1_m < M && col1 < N) {
                float v = static_cast<float>(accum[mi][c][3]) * x_scales[row1_m] * weight_scales[col1];
                if (has_bias) v += to_float(bias[col1]);
                out[row1_m * N + col1] = from_float<OutType>(v);
            }
        }
    }
}

__device__ __forceinline__ int pack_int4_weight4_as_int8_word(uint32_t packed01, uint32_t packed23) {
    const uint32_t b0 = static_cast<uint8_t>(static_cast<int8_t>(unpack_int4_nibble(packed01)));
    const uint32_t b1 = static_cast<uint8_t>(static_cast<int8_t>(unpack_int4_nibble(packed01 >> 4)));
    const uint32_t b2 = static_cast<uint8_t>(static_cast<int8_t>(unpack_int4_nibble(packed23)));
    const uint32_t b3 = static_cast<uint8_t>(static_cast<int8_t>(unpack_int4_nibble(packed23 >> 4)));
    return static_cast<int>(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
}

template<int WARPS_PER_BLOCK, typename OutputType, typename BiasType>
__global__ void int4_weight_int8_act_gemv_dequant_warp_kernel(
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scales,
    const float* __restrict__ weight_scales,
    const BiasType* __restrict__ bias,
    OutputType* __restrict__ output,
    int M,
    int N,
    int K,
    int weight_scale_size,
    bool has_bias)
{
    const int lane = threadIdx.x & (kThreadsPerWarp - 1);
    const int warp = threadIdx.x >> 5;
    const int n = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp;
    const int m = static_cast<int>(blockIdx.y);
    if (n >= N) {
        return;
    }

    const int K4 = K >> 2;
    const int K_half = K >> 1;
    const int* __restrict__ x4 = reinterpret_cast<const int*>(x + static_cast<int64_t>(m) * K);
    const int8_t* __restrict__ w_row = weight + static_cast<int64_t>(n) * K_half;

    int acc = 0;
    for (int k4 = lane; k4 < K4; k4 += kThreadsPerWarp) {
        const uint32_t packed01 = static_cast<uint8_t>(w_row[k4 * 2]);
        const uint32_t packed23 = static_cast<uint8_t>(w_row[k4 * 2 + 1]);
        acc = __dp4a(x4[k4], pack_int4_weight4_as_int8_word(packed01, packed23), acc);
    }

    #pragma unroll
    for (int offset = kThreadsPerWarp / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    if (lane == 0) {
        const float weight_scale = weight_scales[weight_scale_size == 1 ? 0 : n];
        float value = static_cast<float>(acc) * x_scales[m] * weight_scale;
        if (has_bias) {
            value += to_float(bias[n]);
        }
        output[static_cast<int64_t>(m) * N + n] = from_float<OutputType>(value);
    }
}

template<typename OutputType, typename BiasType>
__global__ void dequantize_int4_weight_int8_act_chunk_kernel(
    const int32_t* __restrict__ input,
    const float* __restrict__ x_scales,
    const float* __restrict__ weight_scales,
    const BiasType* __restrict__ bias,
    OutputType* __restrict__ output,
    int64_t total,
    int output_stride,
    int chunk_cols,
    int col_offset,
    int weight_scale_size,
    bool has_bias)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int chunk_col = static_cast<int>(idx % chunk_cols);
    const int row = static_cast<int>(idx / chunk_cols);
    const int col = col_offset + chunk_col;
    const float weight_scale = weight_scales[weight_scale_size == 1 ? 0 : col];
    float value = static_cast<float>(input[idx]) * x_scales[row] * weight_scale;
    if (has_bias) {
        value += to_float(bias[col]);
    }
    output[static_cast<int64_t>(row) * output_stride + col] = from_float<OutputType>(value);
}

__device__ __forceinline__ uint64_t unpack_int4_u32_to_int8_u64(uint32_t packed) {
    uint64_t out = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t byte = (packed >> (i * 8)) & 0xffu;
        const uint32_t low = static_cast<uint8_t>(static_cast<int8_t>(unpack_int4_nibble(byte)));
        const uint32_t high = static_cast<uint8_t>(static_cast<int8_t>(unpack_int4_nibble(byte >> 4)));
        out |= static_cast<uint64_t>(low | (high << 8)) << (i * 16);
    }
    return out;
}

__global__ void unpack_int4_to_int8_vec8_kernel(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ output,
    int64_t total_packed)
{
    const int64_t base = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
    if (base + 7 < total_packed) {
        const uint2 packed = *reinterpret_cast<const uint2*>(input + base);
        uint64_t* __restrict__ out64 = reinterpret_cast<uint64_t*>(output + base * 2);
        out64[0] = unpack_int4_u32_to_int8_u64(packed.x);
        out64[1] = unpack_int4_u32_to_int8_u64(packed.y);
        return;
    }

    for (int64_t idx = base; idx < total_packed; ++idx) {
        const uint32_t packed = static_cast<uint8_t>(input[idx]);
        output[idx * 2] = static_cast<int8_t>(unpack_int4_nibble(packed));
        output[idx * 2 + 1] = static_cast<int8_t>(unpack_int4_nibble(packed >> 4));
    }
}

} // namespace

extern "C" {

void launch_quantize_int4_rowwise_kernel(
    const void* input,
    void* output,
    void* scales,
    int64_t M,
    int64_t K,
    int input_dtype_code,
    bool stochastic,
    uint64_t seed,
    cudaStream_t stream)
{
    if (K % comfy::svdquant::kGroupSize != 0) return;
    constexpr int kThreads = 256;
    const dim3 grid(static_cast<unsigned int>(M));
    const dim3 block(kThreads);
    if (input_dtype_code == 2) {
        if (stochastic) {
            quantize_int4_rowwise_kernel<__nv_bfloat16, kThreads, true>
                <<<grid, block, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(input),
                                             reinterpret_cast<int8_t*>(output),
                                             reinterpret_cast<float*>(scales),
                                             static_cast<int>(K),
                                             seed);
        } else {
            quantize_int4_rowwise_kernel<__nv_bfloat16, kThreads, false>
                <<<grid, block, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(input),
                                             reinterpret_cast<int8_t*>(output),
                                             reinterpret_cast<float*>(scales),
                                             static_cast<int>(K),
                                             seed);
        }
    } else if (input_dtype_code == 1) {
        if (stochastic) {
            quantize_int4_rowwise_kernel<__half, kThreads, true>
                <<<grid, block, 0, stream>>>(reinterpret_cast<const __half*>(input),
                                             reinterpret_cast<int8_t*>(output),
                                             reinterpret_cast<float*>(scales),
                                             static_cast<int>(K),
                                             seed);
        } else {
            quantize_int4_rowwise_kernel<__half, kThreads, false>
                <<<grid, block, 0, stream>>>(reinterpret_cast<const __half*>(input),
                                             reinterpret_cast<int8_t*>(output),
                                             reinterpret_cast<float*>(scales),
                                             static_cast<int>(K),
                                             seed);
        }
    } else if (input_dtype_code == 0) {
        if (stochastic) {
            quantize_int4_rowwise_kernel<float, kThreads, true>
                <<<grid, block, 0, stream>>>(reinterpret_cast<const float*>(input),
                                             reinterpret_cast<int8_t*>(output),
                                             reinterpret_cast<float*>(scales),
                                             static_cast<int>(K),
                                             seed);
        } else {
            quantize_int4_rowwise_kernel<float, kThreads, false>
                <<<grid, block, 0, stream>>>(reinterpret_cast<const float*>(input),
                                             reinterpret_cast<int8_t*>(output),
                                             reinterpret_cast<float*>(scales),
                                             static_cast<int>(K),
                                             seed);
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA INT4 rowwise quantization failed: ") + cudaGetErrorString(err));
    }
}

void launch_quantize_int4_rowwise_convrot64_kernel(
    const void* input,
    void* output,
    void* scales,
    int64_t M,
    int64_t K,
    int group_size,
    int input_dtype_code,
    bool stochastic,
    uint64_t seed,
    cudaStream_t stream)
{
    if ((group_size != 16 && group_size != 64 && group_size != kConvRotGroup) || K % group_size != 0) return;
    constexpr int block_threads_multi = 1024;
    constexpr int block_threads_15360 = 640;
    constexpr int block_threads_single = 512;
    constexpr int block_threads_small = 256;

    auto launch = [&](auto kernel, auto typed_input, int block_threads, int scratch_buffers) {
        using TypedInputPtr = decltype(typed_input);
        using TypedInput = std::remove_cv_t<std::remove_pointer_t<TypedInputPtr>>;
        const int groups_in_flight = block_threads / 64;
        const size_t smem_bytes =
            (static_cast<size_t>(K) + groups_in_flight * scratch_buffers * kConvRotGroup) * sizeof(TypedInput);
        cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));
        kernel<<<static_cast<unsigned int>(M), block_threads, smem_bytes, stream>>>(
            typed_input,
            reinterpret_cast<int8_t*>(output),
            reinterpret_cast<float*>(scales),
            static_cast<int>(K),
            seed);
    };
    auto launch_small = [&](auto kernel, auto typed_input, int small_group_size, int block_threads_small_group) {
        using TypedInputPtr = decltype(typed_input);
        using TypedInput = std::remove_cv_t<std::remove_pointer_t<TypedInputPtr>>;
        const int groups_in_flight = block_threads_small_group / (small_group_size / 4);
        const size_t smem_bytes =
            (static_cast<size_t>(K) + groups_in_flight * 2 * small_group_size) * sizeof(TypedInput);
        cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));
        kernel<<<static_cast<unsigned int>(M), block_threads_small_group, smem_bytes, stream>>>(
            typed_input,
            reinterpret_cast<int8_t*>(output),
            reinterpret_cast<float*>(scales),
            static_cast<int>(K),
            seed);
    };
    auto check_launch = [] {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("CUDA INT4 rowwise convrot quantization failed: ") + cudaGetErrorString(err));
        }
    };
    if (group_size == 16) {
        const bool use_wide_small_group = K > 4096;
        if (input_dtype_code == 2) {
            auto typed_input = reinterpret_cast<const __nv_bfloat16*>(input);
            if (use_wide_small_group) {
                if (stochastic) {
                    launch_small(quantize_int4_rowwise_convrot_small_kernel<16, __nv_bfloat16, 512, true>, typed_input, 16, 512);
                } else {
                    launch_small(quantize_int4_rowwise_convrot_small_kernel<16, __nv_bfloat16, 512, false>, typed_input, 16, 512);
                }
            } else if (stochastic) {
                launch_small(quantize_int4_rowwise_convrot_small_kernel<16, __nv_bfloat16, 128, true>, typed_input, 16, 128);
            } else {
                launch_small(quantize_int4_rowwise_convrot_small_kernel<16, __nv_bfloat16, 128, false>, typed_input, 16, 128);
            }
        } else if (input_dtype_code == 1) {
            auto typed_input = reinterpret_cast<const __half*>(input);
            if (use_wide_small_group) {
                if (stochastic) {
                    launch_small(quantize_int4_rowwise_convrot_small_kernel<16, __half, 512, true>, typed_input, 16, 512);
                } else {
                    launch_small(quantize_int4_rowwise_convrot_small_kernel<16, __half, 512, false>, typed_input, 16, 512);
                }
            } else if (stochastic) {
                launch_small(quantize_int4_rowwise_convrot_small_kernel<16, __half, 128, true>, typed_input, 16, 128);
            } else {
                launch_small(quantize_int4_rowwise_convrot_small_kernel<16, __half, 128, false>, typed_input, 16, 128);
            }
        } else if (input_dtype_code == 0) {
            auto typed_input = reinterpret_cast<const float*>(input);
            if (use_wide_small_group) {
                if (stochastic) {
                    launch_small(quantize_int4_rowwise_convrot_small_kernel<16, float, 512, true>, typed_input, 16, 512);
                } else {
                    launch_small(quantize_int4_rowwise_convrot_small_kernel<16, float, 512, false>, typed_input, 16, 512);
                }
            } else if (stochastic) {
                launch_small(quantize_int4_rowwise_convrot_small_kernel<16, float, 128, true>, typed_input, 16, 128);
            } else {
                launch_small(quantize_int4_rowwise_convrot_small_kernel<16, float, 128, false>, typed_input, 16, 128);
            }
        }
        check_launch();
        return;
    }
    if (group_size == 64) {
        const bool use_wide_small_group = K > 4096;
        if (input_dtype_code == 2) {
            auto typed_input = reinterpret_cast<const __nv_bfloat16*>(input);
            if (use_wide_small_group) {
                if (stochastic) {
                    launch_small(quantize_int4_rowwise_convrot_small_kernel<64, __nv_bfloat16, 512, true>, typed_input, 64, 512);
                } else {
                    launch_small(quantize_int4_rowwise_convrot_small_kernel<64, __nv_bfloat16, 512, false>, typed_input, 64, 512);
                }
            } else if (stochastic) {
                launch_small(quantize_int4_rowwise_convrot_small_kernel<64, __nv_bfloat16, 128, true>, typed_input, 64, 128);
            } else {
                launch_small(quantize_int4_rowwise_convrot_small_kernel<64, __nv_bfloat16, 128, false>, typed_input, 64, 128);
            }
        } else if (input_dtype_code == 1) {
            auto typed_input = reinterpret_cast<const __half*>(input);
            if (use_wide_small_group) {
                if (stochastic) {
                    launch_small(quantize_int4_rowwise_convrot_small_kernel<64, __half, 512, true>, typed_input, 64, 512);
                } else {
                    launch_small(quantize_int4_rowwise_convrot_small_kernel<64, __half, 512, false>, typed_input, 64, 512);
                }
            } else if (stochastic) {
                launch_small(quantize_int4_rowwise_convrot_small_kernel<64, __half, 128, true>, typed_input, 64, 128);
            } else {
                launch_small(quantize_int4_rowwise_convrot_small_kernel<64, __half, 128, false>, typed_input, 64, 128);
            }
        } else if (input_dtype_code == 0) {
            auto typed_input = reinterpret_cast<const float*>(input);
            if (use_wide_small_group) {
                if (stochastic) {
                    launch_small(quantize_int4_rowwise_convrot_small_kernel<64, float, 512, true>, typed_input, 64, 512);
                } else {
                    launch_small(quantize_int4_rowwise_convrot_small_kernel<64, float, 512, false>, typed_input, 64, 512);
                }
            } else if (stochastic) {
                launch_small(quantize_int4_rowwise_convrot_small_kernel<64, float, 128, true>, typed_input, 64, 128);
            } else {
                launch_small(quantize_int4_rowwise_convrot_small_kernel<64, float, 128, false>, typed_input, 64, 128);
            }
        }
        check_launch();
        return;
    }
    if (input_dtype_code == 2) {
        auto typed_input = reinterpret_cast<const __nv_bfloat16*>(input);
        if (M != 1 && K <= 4096) {
            if (stochastic) {
                launch(quantize_int4_rowwise_convrot64_kernel<__nv_bfloat16, block_threads_small, false, true>, typed_input, block_threads_small, 2);
            } else {
                launch(quantize_int4_rowwise_convrot64_kernel<__nv_bfloat16, block_threads_small, false, false>, typed_input, block_threads_small, 2);
            }
        } else if (M == 1) {
            if (stochastic) {
                launch(quantize_int4_rowwise_convrot64_kernel<__nv_bfloat16, block_threads_single, false, true>, typed_input, block_threads_single, 2);
            } else {
                launch(quantize_int4_rowwise_convrot64_kernel<__nv_bfloat16, block_threads_single, false, false>, typed_input, block_threads_single, 2);
            }
        } else if (K == 15360) {
            if (stochastic) {
                launch(quantize_int4_rowwise_convrot64_kernel<__nv_bfloat16, block_threads_15360, true, true>, typed_input, block_threads_15360, 1);
            } else {
                launch(quantize_int4_rowwise_convrot64_kernel<__nv_bfloat16, block_threads_15360, true, false>, typed_input, block_threads_15360, 1);
            }
        } else {
            if (stochastic) {
                launch(quantize_int4_rowwise_convrot64_kernel<__nv_bfloat16, block_threads_multi, false, true>, typed_input, block_threads_multi, 2);
            } else {
                launch(quantize_int4_rowwise_convrot64_kernel<__nv_bfloat16, block_threads_multi, false, false>, typed_input, block_threads_multi, 2);
            }
        }
    } else if (input_dtype_code == 1) {
        auto typed_input = reinterpret_cast<const __half*>(input);
        if (M != 1 && K <= 4096) {
            if (stochastic) {
                launch(quantize_int4_rowwise_convrot64_kernel<__half, block_threads_small, false, true>, typed_input, block_threads_small, 2);
            } else {
                launch(quantize_int4_rowwise_convrot64_kernel<__half, block_threads_small, false, false>, typed_input, block_threads_small, 2);
            }
        } else if (M == 1) {
            if (stochastic) {
                launch(quantize_int4_rowwise_convrot64_kernel<__half, block_threads_single, false, true>, typed_input, block_threads_single, 2);
            } else {
                launch(quantize_int4_rowwise_convrot64_kernel<__half, block_threads_single, false, false>, typed_input, block_threads_single, 2);
            }
        } else if (K == 15360) {
            if (stochastic) {
                launch(quantize_int4_rowwise_convrot64_kernel<__half, block_threads_15360, true, true>, typed_input, block_threads_15360, 1);
            } else {
                launch(quantize_int4_rowwise_convrot64_kernel<__half, block_threads_15360, true, false>, typed_input, block_threads_15360, 1);
            }
        } else {
            if (stochastic) {
                launch(quantize_int4_rowwise_convrot64_kernel<__half, block_threads_multi, false, true>, typed_input, block_threads_multi, 2);
            } else {
                launch(quantize_int4_rowwise_convrot64_kernel<__half, block_threads_multi, false, false>, typed_input, block_threads_multi, 2);
            }
        }
    } else if (input_dtype_code == 0) {
        auto typed_input = reinterpret_cast<const float*>(input);
        if (M != 1 && K <= 4096) {
            if (stochastic) {
                launch(quantize_int4_rowwise_convrot64_kernel<float, block_threads_small, false, true>, typed_input, block_threads_small, 2);
            } else {
                launch(quantize_int4_rowwise_convrot64_kernel<float, block_threads_small, false, false>, typed_input, block_threads_small, 2);
            }
        } else if (M == 1) {
            if (stochastic) {
                launch(quantize_int4_rowwise_convrot64_kernel<float, block_threads_single, false, true>, typed_input, block_threads_single, 2);
            } else {
                launch(quantize_int4_rowwise_convrot64_kernel<float, block_threads_single, false, false>, typed_input, block_threads_single, 2);
            }
        } else if (K == 15360) {
            if (stochastic) {
                launch(quantize_int4_rowwise_convrot64_kernel<float, block_threads_15360, true, true>, typed_input, block_threads_15360, 1);
            } else {
                launch(quantize_int4_rowwise_convrot64_kernel<float, block_threads_15360, true, false>, typed_input, block_threads_15360, 1);
            }
        } else {
            if (stochastic) {
                launch(quantize_int4_rowwise_convrot64_kernel<float, block_threads_multi, false, true>, typed_input, block_threads_multi, 2);
            } else {
                launch(quantize_int4_rowwise_convrot64_kernel<float, block_threads_multi, false, false>, typed_input, block_threads_multi, 2);
            }
        }
    }

    check_launch();
}

void launch_quantize_int4_rowwise_convrot64_to_int8_kernel(
    const void* input,
    void* output,
    void* scales,
    int64_t M,
    int64_t K,
    int group_size,
    int input_dtype_code,
    bool stochastic,
    uint64_t seed,
    cudaStream_t stream)
{
    if (group_size != kConvRotGroup || K % kConvRotGroup != 0) return;
    constexpr int block_threads_multi = 1024;
    constexpr int block_threads_15360 = 640;
    constexpr int block_threads_single = 512;

    auto launch = [&](auto kernel, auto typed_input, int block_threads, int scratch_buffers) {
        using TypedInputPtr = decltype(typed_input);
        using TypedInput = std::remove_cv_t<std::remove_pointer_t<TypedInputPtr>>;
        const int groups_in_flight = block_threads / 64;
        const size_t smem_bytes =
            (static_cast<size_t>(K) + groups_in_flight * scratch_buffers * kConvRotGroup) * sizeof(TypedInput);
        cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));
        kernel<<<static_cast<unsigned int>(M), block_threads, smem_bytes, stream>>>(
            typed_input,
            reinterpret_cast<int8_t*>(output),
            reinterpret_cast<float*>(scales),
            static_cast<int>(K),
            seed);
    };

    auto launch_float = [&](auto kernel, auto typed_input, int block_threads) {
        const int groups_in_flight = block_threads / 64;
        const size_t smem_bytes =
            (static_cast<size_t>(K) + groups_in_flight * 2 * kConvRotGroup) * sizeof(float);
        cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));
        kernel<<<static_cast<unsigned int>(M), block_threads, smem_bytes, stream>>>(
            typed_input,
            reinterpret_cast<int8_t*>(output),
            reinterpret_cast<float*>(scales),
            static_cast<int>(K),
            seed);
    };

    if (input_dtype_code == 2) {
        auto typed_input = reinterpret_cast<const __nv_bfloat16*>(input);
        if (K <= 4096) {
            if (stochastic) {
                launch_float(
                    quantize_int4_rowwise_convrot64_to_int8_float_kernel<__nv_bfloat16, block_threads_multi, true>,
                    typed_input,
                    block_threads_multi);
            } else {
                launch_float(
                    quantize_int4_rowwise_convrot64_to_int8_float_kernel<__nv_bfloat16, block_threads_multi, false>,
                    typed_input,
                    block_threads_multi);
            }
        } else if (M == 1) {
            if (stochastic) {
                launch(quantize_int4_rowwise_convrot64_kernel<__nv_bfloat16, block_threads_single, false, true, true>, typed_input, block_threads_single, 2);
            } else {
                launch(quantize_int4_rowwise_convrot64_kernel<__nv_bfloat16, block_threads_single, false, false, true>, typed_input, block_threads_single, 2);
            }
        } else if (K == 15360) {
            if (stochastic) {
                launch(quantize_int4_rowwise_convrot64_kernel<__nv_bfloat16, block_threads_15360, true, true, true>, typed_input, block_threads_15360, 1);
            } else {
                launch(quantize_int4_rowwise_convrot64_kernel<__nv_bfloat16, block_threads_15360, true, false, true>, typed_input, block_threads_15360, 1);
            }
        } else {
            if (stochastic) {
                launch(quantize_int4_rowwise_convrot64_kernel<__nv_bfloat16, block_threads_multi, false, true, true>, typed_input, block_threads_multi, 2);
            } else {
                launch(quantize_int4_rowwise_convrot64_kernel<__nv_bfloat16, block_threads_multi, false, false, true>, typed_input, block_threads_multi, 2);
            }
        }
    } else if (input_dtype_code == 1) {
        auto typed_input = reinterpret_cast<const __half*>(input);
        if (K <= 4096) {
            if (stochastic) {
                launch_float(
                    quantize_int4_rowwise_convrot64_to_int8_float_kernel<__half, block_threads_multi, true>,
                    typed_input,
                    block_threads_multi);
            } else {
                launch_float(
                    quantize_int4_rowwise_convrot64_to_int8_float_kernel<__half, block_threads_multi, false>,
                    typed_input,
                    block_threads_multi);
            }
        } else if (M == 1) {
            if (stochastic) {
                launch(quantize_int4_rowwise_convrot64_kernel<__half, block_threads_single, false, true, true>, typed_input, block_threads_single, 2);
            } else {
                launch(quantize_int4_rowwise_convrot64_kernel<__half, block_threads_single, false, false, true>, typed_input, block_threads_single, 2);
            }
        } else if (K == 15360) {
            if (stochastic) {
                launch(quantize_int4_rowwise_convrot64_kernel<__half, block_threads_15360, true, true, true>, typed_input, block_threads_15360, 1);
            } else {
                launch(quantize_int4_rowwise_convrot64_kernel<__half, block_threads_15360, true, false, true>, typed_input, block_threads_15360, 1);
            }
        } else {
            if (stochastic) {
                launch(quantize_int4_rowwise_convrot64_kernel<__half, block_threads_multi, false, true, true>, typed_input, block_threads_multi, 2);
            } else {
                launch(quantize_int4_rowwise_convrot64_kernel<__half, block_threads_multi, false, false, true>, typed_input, block_threads_multi, 2);
            }
        }
    } else if (input_dtype_code == 0) {
        auto typed_input = reinterpret_cast<const float*>(input);
        if (K <= 4096) {
            if (stochastic) {
                launch_float(
                    quantize_int4_rowwise_convrot64_to_int8_float_kernel<float, block_threads_multi, true>,
                    typed_input,
                    block_threads_multi);
            } else {
                launch_float(
                    quantize_int4_rowwise_convrot64_to_int8_float_kernel<float, block_threads_multi, false>,
                    typed_input,
                    block_threads_multi);
            }
        } else if (M == 1) {
            if (stochastic) {
                launch(quantize_int4_rowwise_convrot64_kernel<float, block_threads_single, false, true, true>, typed_input, block_threads_single, 2);
            } else {
                launch(quantize_int4_rowwise_convrot64_kernel<float, block_threads_single, false, false, true>, typed_input, block_threads_single, 2);
            }
        } else if (K == 15360) {
            if (stochastic) {
                launch(quantize_int4_rowwise_convrot64_kernel<float, block_threads_15360, true, true, true>, typed_input, block_threads_15360, 1);
            } else {
                launch(quantize_int4_rowwise_convrot64_kernel<float, block_threads_15360, true, false, true>, typed_input, block_threads_15360, 1);
            }
        } else {
            if (stochastic) {
                launch(quantize_int4_rowwise_convrot64_kernel<float, block_threads_multi, false, true, true>, typed_input, block_threads_multi, 2);
            } else {
                launch(quantize_int4_rowwise_convrot64_kernel<float, block_threads_multi, false, false, true>, typed_input, block_threads_multi, 2);
            }
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA INT4 rowwise convrot INT8-output quantization failed: ") + cudaGetErrorString(err));
    }
}

void launch_dequantize_int4_convrot64_kernel(
    const void* input,
    const void* scales,
    void* output,
    int64_t num_rows,
    int64_t num_cols,
    int64_t scale_size,
    int group_size,
    int output_dtype_code,
    cudaStream_t stream)
{
    if (num_rows == 0 || num_cols == 0) return;
    if ((group_size != 16 && group_size != 64 && group_size != kConvRotGroup) || num_cols % group_size != 0) return;

    auto launch_small_groups = [&](auto group_tag, auto groups_tag) {
        constexpr int small_group_size = decltype(group_tag)::value;
        constexpr int groups_per_block = decltype(groups_tag)::value;
        constexpr int block_threads = groups_per_block * (small_group_size / 4);
        const int group_blocks =
            static_cast<int>((num_cols / small_group_size + groups_per_block - 1) / groups_per_block);
        dim3 grid(static_cast<unsigned int>(num_rows), static_cast<unsigned int>(group_blocks));
        const bool check_bounds = ((num_cols / small_group_size) % groups_per_block) != 0;
        const bool scale_per_row = scale_size != 1;

        if (output_dtype_code == 2) {
            const size_t smem_bytes = groups_per_block * 2 * small_group_size * sizeof(__nv_bfloat16);
            if (check_bounds) {
                if (scale_per_row) {
                    dequantize_int4_convrot_small_kernel<small_group_size, groups_per_block, __nv_bfloat16, true, true>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<__nv_bfloat16*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                } else {
                    dequantize_int4_convrot_small_kernel<small_group_size, groups_per_block, __nv_bfloat16, true, false>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<__nv_bfloat16*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                }
            } else {
                if (scale_per_row) {
                    dequantize_int4_convrot_small_kernel<small_group_size, groups_per_block, __nv_bfloat16, false, true>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<__nv_bfloat16*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                } else {
                    dequantize_int4_convrot_small_kernel<small_group_size, groups_per_block, __nv_bfloat16, false, false>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<__nv_bfloat16*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                }
            }
        } else if (output_dtype_code == 1) {
            const size_t smem_bytes = groups_per_block * 2 * small_group_size * sizeof(__half);
            if (check_bounds) {
                if (scale_per_row) {
                    dequantize_int4_convrot_small_kernel<small_group_size, groups_per_block, __half, true, true>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<__half*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                } else {
                    dequantize_int4_convrot_small_kernel<small_group_size, groups_per_block, __half, true, false>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<__half*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                }
            } else {
                if (scale_per_row) {
                    dequantize_int4_convrot_small_kernel<small_group_size, groups_per_block, __half, false, true>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<__half*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                } else {
                    dequantize_int4_convrot_small_kernel<small_group_size, groups_per_block, __half, false, false>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<__half*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                }
            }
        } else {
            const size_t smem_bytes = groups_per_block * 2 * small_group_size * sizeof(float);
            if (check_bounds) {
                if (scale_per_row) {
                    dequantize_int4_convrot_small_kernel<small_group_size, groups_per_block, float, true, true>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<float*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                } else {
                    dequantize_int4_convrot_small_kernel<small_group_size, groups_per_block, float, true, false>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<float*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                }
            } else {
                if (scale_per_row) {
                    dequantize_int4_convrot_small_kernel<small_group_size, groups_per_block, float, false, true>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<float*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                } else {
                    dequantize_int4_convrot_small_kernel<small_group_size, groups_per_block, float, false, false>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<float*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                }
            }
        }
    };

    if (group_size == 16) {
        launch_small_groups(std::integral_constant<int, 16>{}, std::integral_constant<int, 32>{});
        return;
    }
    if (group_size == 64) {
        launch_small_groups(std::integral_constant<int, 64>{}, std::integral_constant<int, 8>{});
        return;
    }

    auto launch_groups = [&](auto groups_tag) {
        constexpr int groups_per_block = decltype(groups_tag)::value;
        constexpr int block_threads = groups_per_block * 64;
        const int group_blocks =
            static_cast<int>((num_cols / kConvRotGroup + groups_per_block - 1) / groups_per_block);
        dim3 grid(static_cast<unsigned int>(num_rows), static_cast<unsigned int>(group_blocks));
        const bool check_bounds = ((num_cols / kConvRotGroup) % groups_per_block) != 0;
        const bool scale_per_row = scale_size != 1;

        if (output_dtype_code == 2) {
            const size_t smem_bytes = groups_per_block * 2 * kConvRotGroup * sizeof(__nv_bfloat16);
            if (check_bounds) {
                if (scale_per_row) {
                    dequantize_int4_convrot64_kernel<groups_per_block, __nv_bfloat16, true, true>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<__nv_bfloat16*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                } else {
                    dequantize_int4_convrot64_kernel<groups_per_block, __nv_bfloat16, true, false>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<__nv_bfloat16*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                }
            } else {
                if (scale_per_row) {
                    dequantize_int4_convrot64_kernel<groups_per_block, __nv_bfloat16, false, true>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<__nv_bfloat16*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                } else {
                    dequantize_int4_convrot64_kernel<groups_per_block, __nv_bfloat16, false, false>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<__nv_bfloat16*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                }
            }
        } else if (output_dtype_code == 1) {
            const size_t smem_bytes = groups_per_block * 2 * kConvRotGroup * sizeof(__half);
            if (check_bounds) {
                if (scale_per_row) {
                    dequantize_int4_convrot64_kernel<groups_per_block, __half, true, true>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<__half*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                } else {
                    dequantize_int4_convrot64_kernel<groups_per_block, __half, true, false>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<__half*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                }
            } else {
                if (scale_per_row) {
                    dequantize_int4_convrot64_kernel<groups_per_block, __half, false, true>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<__half*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                } else {
                    dequantize_int4_convrot64_kernel<groups_per_block, __half, false, false>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<__half*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                }
            }
        } else {
            const size_t smem_bytes = groups_per_block * 2 * kConvRotGroup * sizeof(float);
            if (check_bounds) {
                if (scale_per_row) {
                    dequantize_int4_convrot64_kernel<groups_per_block, float, true, true>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<float*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                } else {
                    dequantize_int4_convrot64_kernel<groups_per_block, float, true, false>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<float*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                }
            } else {
                if (scale_per_row) {
                    dequantize_int4_convrot64_kernel<groups_per_block, float, false, true>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<float*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                } else {
                    dequantize_int4_convrot64_kernel<groups_per_block, float, false, false>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<float*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                }
            }
        }
    };

    auto launch_groups_warp32 = [&](auto groups_tag) {
        constexpr int groups_per_block = decltype(groups_tag)::value;
        constexpr int block_threads = groups_per_block * 32;
        const int group_blocks =
            static_cast<int>((num_cols / kConvRotGroup + groups_per_block - 1) / groups_per_block);
        dim3 grid(static_cast<unsigned int>(num_rows), static_cast<unsigned int>(group_blocks));
        const bool check_bounds = ((num_cols / kConvRotGroup) % groups_per_block) != 0;
        const bool scale_per_row = scale_size != 1;

        if (output_dtype_code == 2) {
            const size_t smem_bytes = groups_per_block * 2 * kConvRotGroup * sizeof(__nv_bfloat16);
            if (check_bounds) {
                if (scale_per_row) {
                    dequantize_int4_convrot64_warp32_kernel<groups_per_block, __nv_bfloat16, true, true>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<__nv_bfloat16*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                } else {
                    dequantize_int4_convrot64_warp32_kernel<groups_per_block, __nv_bfloat16, true, false>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<__nv_bfloat16*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                }
            } else {
                if (scale_per_row) {
                    dequantize_int4_convrot64_warp32_kernel<groups_per_block, __nv_bfloat16, false, true>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<__nv_bfloat16*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                } else {
                    dequantize_int4_convrot64_warp32_kernel<groups_per_block, __nv_bfloat16, false, false>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<__nv_bfloat16*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                }
            }
        } else {
            const size_t smem_bytes = groups_per_block * 2 * kConvRotGroup * sizeof(__half);
            if (check_bounds) {
                if (scale_per_row) {
                    dequantize_int4_convrot64_warp32_kernel<groups_per_block, __half, true, true>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<__half*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                } else {
                    dequantize_int4_convrot64_warp32_kernel<groups_per_block, __half, true, false>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<__half*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                }
            } else {
                if (scale_per_row) {
                    dequantize_int4_convrot64_warp32_kernel<groups_per_block, __half, false, true>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<__half*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                } else {
                    dequantize_int4_convrot64_warp32_kernel<groups_per_block, __half, false, false>
                        <<<grid, block_threads, smem_bytes, stream>>>(
                            reinterpret_cast<const int8_t*>(input),
                            reinterpret_cast<const float*>(scales),
                            reinterpret_cast<__half*>(output),
                            static_cast<int>(num_cols),
                            static_cast<int>(scale_size));
                }
            }
        }
    };

    if (output_dtype_code == 1 || output_dtype_code == 2) {
        if (num_cols < 1024) {
            launch_groups_warp32(std::integral_constant<int, 1>{});
        } else if (num_cols < 4096) {
            launch_groups_warp32(std::integral_constant<int, 4>{});
        } else {
            launch_groups_warp32(std::integral_constant<int, 4>{});
        }
    } else {
        if (num_cols < 1024) {
            launch_groups(std::integral_constant<int, 1>{});
        } else if (num_cols < 4096) {
            launch_groups(std::integral_constant<int, 2>{});
        } else if (num_cols < 8192) {
            launch_groups(std::integral_constant<int, 4>{});
        } else {
            launch_groups(std::integral_constant<int, 5>{});
        }
    }
}

void launch_int4_linear_kernel(
    const void* act,
    const void* weight,
    const void* x_scales,
    const void* weight_scales,
    const void* bias,
    void* output,
    int64_t M,
    int64_t N,
    int64_t K,
    bool has_bias,
    int output_dtype_code,
    int bias_dtype_code,
    cudaStream_t stream)
{
    if (K % comfy::svdquant::kGroupSize != 0) return;
    const dim3 grid(static_cast<unsigned int>((N + kBlockN - 1) / kBlockN),
                    static_cast<unsigned int>((M + kBlockM - 1) / kBlockM));
    const dim3 block(kThreadsPerBlock);

#define DISPATCH_OUT_BIAS(OutType, BiasType)                                              \
    int4_linear_kernel<OutType, BiasType><<<grid, block, 0, stream>>>(                    \
        reinterpret_cast<const int8_t*>(act),                                             \
        reinterpret_cast<const int8_t*>(weight),                                          \
        reinterpret_cast<const float*>(x_scales),                                         \
        reinterpret_cast<const float*>(weight_scales),                                    \
        reinterpret_cast<const BiasType*>(bias),                                          \
        reinterpret_cast<OutType*>(output),                                               \
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), has_bias)

#define DISPATCH_BIAS(OutType)                                                            \
    do {                                                                                  \
        if (bias_dtype_code == 2) DISPATCH_OUT_BIAS(OutType, __nv_bfloat16);              \
        else if (bias_dtype_code == 1) DISPATCH_OUT_BIAS(OutType, __half);                \
        else DISPATCH_OUT_BIAS(OutType, float);                                           \
    } while (0)

    if (output_dtype_code == 2) {
        DISPATCH_BIAS(__nv_bfloat16);
    } else if (output_dtype_code == 1) {
        DISPATCH_BIAS(__half);
    } else {
        DISPATCH_BIAS(float);
    }

#undef DISPATCH_BIAS
#undef DISPATCH_OUT_BIAS
}

void launch_unpack_int4_to_int8_kernel(
    const void* input,
    void* output,
    int64_t rows,
    int64_t K_half,
    cudaStream_t stream)
{
    if (rows == 0 || K_half == 0) return;
    const int64_t total_packed = rows * K_half;
    constexpr int block_threads = 256;
    const int64_t blocks = ((total_packed + 7) / 8 + block_threads - 1) / block_threads;
    unpack_int4_to_int8_vec8_kernel<<<static_cast<unsigned int>(blocks), block_threads, 0, stream>>>(
        reinterpret_cast<const int8_t*>(input),
        reinterpret_cast<int8_t*>(output),
        total_packed);
}

void launch_int4_weight_int8_act_gemv_dequant_kernel(
    const void* input,
    const void* weight,
    const void* x_scales,
    const void* weight_scales,
    const void* bias,
    void* output,
    int64_t num_rows,
    int64_t num_cols,
    int64_t K,
    int64_t weight_scale_size,
    bool has_bias,
    int output_dtype_code,
    int bias_dtype_code,
    cudaStream_t stream)
{
    if (num_cols == 0 || K == 0) return;
    if ((K & 3) != 0) {
        throw std::runtime_error("int4_weight_int8_act_gemv_dequant requires K divisible by 4");
    }
    if (num_cols > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        num_rows > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        K > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("int4_weight_int8_act_gemv_dequant only supports M,N,K <= INT_MAX");
    }
    if (weight_scale_size != 1 && weight_scale_size != num_cols) {
        throw std::runtime_error("INT4 packed GEMV weight scale must be scalar or per-output-channel");
    }

    constexpr int kWarpsPerBlock = 8;
    const dim3 grid(
        static_cast<unsigned int>((num_cols + kWarpsPerBlock - 1) / kWarpsPerBlock),
        static_cast<unsigned int>(num_rows));

    DISPATCH_FP_DTYPE(output_dtype_code, OutputType, [&] {
        if (!has_bias) {
            int4_weight_int8_act_gemv_dequant_warp_kernel<kWarpsPerBlock, OutputType, float>
                <<<grid, kWarpsPerBlock * kThreadsPerWarp, 0, stream>>>(
                    static_cast<const int8_t*>(input),
                    static_cast<const int8_t*>(weight),
                    static_cast<const float*>(x_scales),
                    static_cast<const float*>(weight_scales),
                    nullptr,
                    static_cast<OutputType*>(output),
                    static_cast<int>(num_rows),
                    static_cast<int>(num_cols),
                    static_cast<int>(K),
                    static_cast<int>(weight_scale_size),
                    false);
            return;
        }

        DISPATCH_FP_DTYPE(bias_dtype_code, BiasType, [&] {
            int4_weight_int8_act_gemv_dequant_warp_kernel<kWarpsPerBlock, OutputType, BiasType>
                <<<grid, kWarpsPerBlock * kThreadsPerWarp, 0, stream>>>(
                    static_cast<const int8_t*>(input),
                    static_cast<const int8_t*>(weight),
                    static_cast<const float*>(x_scales),
                    static_cast<const float*>(weight_scales),
                    static_cast<const BiasType*>(bias),
                    static_cast<OutputType*>(output),
                    static_cast<int>(num_rows),
                    static_cast<int>(num_cols),
                    static_cast<int>(K),
                    static_cast<int>(weight_scale_size),
                    true);
        });
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA packed INT4 weight GEMV failed: ") + cudaGetErrorString(err));
    }
}

void launch_int4_weight_int8_act_gemm_dequant_chunked_kernel(
    const void* input,
    const void* weight,
    const void* x_scales,
    const void* weight_scales,
    const void* bias,
    void* output,
    void* weight_workspace,
    void* acc_workspace,
    void* cublas_workspace,
    int64_t cublas_workspace_size,
    int64_t num_rows,
    int64_t num_cols,
    int64_t K,
    int64_t weight_scale_size,
    int64_t chunk_cols,
    bool allow_sm80_cutlass,
    bool has_bias,
    int output_dtype_code,
    int bias_dtype_code,
    cudaStream_t stream)
{
    if (num_rows == 0 || num_cols == 0 || K == 0) return;
    if ((K & 1) != 0) {
        throw std::runtime_error("int4_weight_int8_act_gemm_dequant_chunked requires K divisible by 2");
    }
    if (chunk_cols <= 0) {
        throw std::runtime_error("int4_weight_int8_act_gemm_dequant_chunked requires positive chunk_cols");
    }
    if (num_cols > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        num_rows > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        K > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        chunk_cols > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("int4_weight_int8_act_gemm_dequant_chunked only supports M,N,K,chunk <= INT_MAX");
    }
    if (weight_scale_size != 1 && weight_scale_size != num_cols) {
        throw std::runtime_error("INT4 chunked GEMM weight scale must be scalar or per-output-channel");
    }

    const int64_t K_half = K >> 1;
    constexpr int unpack_threads = 256;
    constexpr int dequant_threads = 256;

    for (int64_t n0 = 0; n0 < num_cols; n0 += chunk_cols) {
        const int64_t cols = std::min(chunk_cols, num_cols - n0);
        const int64_t total_packed = cols * K_half;
        const int64_t unpack_blocks = ((total_packed + 7) / 8 + unpack_threads - 1) / unpack_threads;
        unpack_int4_to_int8_vec8_kernel<<<static_cast<unsigned int>(unpack_blocks), unpack_threads, 0, stream>>>(
            static_cast<const int8_t*>(weight) + n0 * K_half,
            static_cast<int8_t*>(weight_workspace),
            total_packed);

        const float* chunk_weight_scales = static_cast<const float*>(weight_scales) + (weight_scale_size == 1 ? 0 : n0);
        const void* chunk_bias = has_bias ? static_cast<const char*>(bias) + n0 * fp_dtype_size_bytes(bias_dtype_code) : nullptr;
        void* chunk_output = static_cast<char*>(output) + n0 * fp_dtype_size_bytes(output_dtype_code);
        const bool used_cutlass = (
            allow_sm80_cutlass
            && num_rows >= 1024
            && (cols >= 4096 || (num_cols == 2560 && cols == 2560))
            && weight_scale_size != 1
            && (!has_bias || bias_dtype_code == 0)
            && launch_cutlass_int8_dequant_strided(
                input,
                weight_workspace,
                x_scales,
                chunk_weight_scales,
                chunk_bias,
                chunk_output,
                num_rows,
                cols,
                K,
                num_cols,
                output_dtype_code,
                stream));
        if (used_cutlass) {
            continue;
        }

        launch_cublas_gemm_int8_kernel(
            input,
            weight_workspace,
            acc_workspace,
            num_rows,
            cols,
            K,
            cublas_workspace,
            cublas_workspace_size,
            stream);

        const int64_t total = num_rows * cols;
        const int64_t dequant_blocks = (total + dequant_threads - 1) / dequant_threads;
        DISPATCH_FP_DTYPE(output_dtype_code, OutputType, [&] {
            if (!has_bias) {
                dequantize_int4_weight_int8_act_chunk_kernel<OutputType, float>
                    <<<static_cast<unsigned int>(dequant_blocks), dequant_threads, 0, stream>>>(
                        static_cast<const int32_t*>(acc_workspace),
                        static_cast<const float*>(x_scales),
                        static_cast<const float*>(weight_scales),
                        nullptr,
                        static_cast<OutputType*>(output),
                        total,
                        static_cast<int>(num_cols),
                        static_cast<int>(cols),
                        static_cast<int>(n0),
                        static_cast<int>(weight_scale_size),
                        false);
                return;
            }

            DISPATCH_FP_DTYPE(bias_dtype_code, BiasType, [&] {
                dequantize_int4_weight_int8_act_chunk_kernel<OutputType, BiasType>
                    <<<static_cast<unsigned int>(dequant_blocks), dequant_threads, 0, stream>>>(
                        static_cast<const int32_t*>(acc_workspace),
                        static_cast<const float*>(x_scales),
                        static_cast<const float*>(weight_scales),
                        static_cast<const BiasType*>(bias),
                        static_cast<OutputType*>(output),
                        total,
                        static_cast<int>(num_cols),
                        static_cast<int>(cols),
                        static_cast<int>(n0),
                        static_cast<int>(weight_scale_size),
                        true);
            });
        });
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA chunked INT4 weight INT8 GEMM failed: ") + cudaGetErrorString(err));
    }
}

bool launch_cutlass_int4_dequant(
    const void* A,
    const void* B,
    const void* xs,
    const void* ws,
    const void* bias,
    void* D,
    int64_t M,
    int64_t N,
    int64_t K,
    int out_dtype_code,
    cudaStream_t stream)
{
#ifdef COMFY_HAVE_CUTLASS
    if (M == 0 || N == 0 || K == 0) return true;
    if (K % comfy::svdquant::kGroupSize != 0) return false;
    const int8_t* a = static_cast<const int8_t*>(A);
    const int8_t* b = static_cast<const int8_t*>(B);
    const float* x = static_cast<const float*>(xs);
    const float* w = static_cast<const float*>(ws);
    const float* bs = static_cast<const float*>(bias);
    if (bs == nullptr) {
        switch (out_dtype_code) {
            case 0: return dispatch_fused_int4_no_bias<float>(a, b, x, w, static_cast<float*>(D), M, N, K, stream);
            case 1: return dispatch_fused_int4_no_bias<cutlass::half_t>(a, b, x, w, static_cast<cutlass::half_t*>(D), M, N, K, stream);
            case 2: return dispatch_fused_int4_no_bias<cutlass::bfloat16_t>(a, b, x, w, static_cast<cutlass::bfloat16_t*>(D), M, N, K, stream);
            default: return false;
        }
    }
    switch (out_dtype_code) {
        case 0: return dispatch_fused_int4<float>(a, b, x, w, bs, static_cast<float*>(D), M, N, K, stream);
        case 1: return dispatch_fused_int4<cutlass::half_t>(a, b, x, w, bs, static_cast<cutlass::half_t*>(D), M, N, K, stream);
        case 2: return dispatch_fused_int4<cutlass::bfloat16_t>(a, b, x, w, bs, static_cast<cutlass::bfloat16_t*>(D), M, N, K, stream);
        default: return false;
    }
#else
    return false;
#endif
}

} // extern "C"
