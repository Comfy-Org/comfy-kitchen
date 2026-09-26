// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// INT8 quantization kernels for the WMMA int8 GEMM.
#include <stdexcept>

#include "../hadamard.h"
#include "../launchers.h"  // comfy_small_igpu

namespace comfy::hip_backend {

constexpr float kInt8Max = 127.0f;

__forceinline__ __device__ int8_t quant_round(float v, float inv) {
    // rintf is round-half-to-even, matching torch.round.
    int q = static_cast<int>(rintf(v * inv));
    q = q < -128 ? -128 : (q > 127 ? 127 : q);
    return static_cast<int8_t>(q);
}

// 16-byte element loads for the rowwise quantizer: 4 floats or 8 16-bit values
// per thread per chunk, so a warp requests a full 512B run of a row at once
// instead of 64B with the old scalar stride. The staging keeps the values in
// registers, so the row is still read exactly once when it fits the budget.
template <int CODE>
__forceinline__ __device__ void load_vec16(const uint4 v, float* out) {
    if constexpr (CODE == 0) {
        const float* f = reinterpret_cast<const float*>(&v);
        out[0] = f[0];
        out[1] = f[1];
        out[2] = f[2];
        out[3] = f[3];
    } else if constexpr (CODE == 1) {
        const __half* h = reinterpret_cast<const __half*>(&v);
        #pragma unroll
        for (int i = 0; i < 8; ++i) out[i] = __half2float(h[i]);
    } else {
        const __bf16* b = reinterpret_cast<const __bf16*>(&v);
        #pragma unroll
        for (int i = 0; i < 8; ++i) out[i] = static_cast<float>(b[i]);
    }
}

__forceinline__ __device__ void load_vec16_rt(const uint4 v, float* out, int code) {
    if (code == 0)
        load_vec16<0>(v, out);
    else if (code == 1)
        load_vec16<1>(v, out);
    else
        load_vec16<2>(v, out);
}

// Packs kVec quantized int8 values into one aligned store (8 bytes for 16-bit
// inputs, 4 for f32). WIDTH: 0/1 selects the store width at runtime.
__forceinline__ __device__ void quant_store_vec_rt(const float* vals, int8_t* q, float inv,
                                                   int width) {
    if (width == 4) {
        int32_t p = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            p |= static_cast<int32_t>(static_cast<uint8_t>(quant_round(vals[i], inv))) << (8 * i);
        *reinterpret_cast<int32_t*>(q) = p;
    } else {
        int64_t p = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            p |= static_cast<int64_t>(static_cast<uint8_t>(quant_round(vals[i], inv))) << (8 * i);
        *reinterpret_cast<int64_t*>(q) = p;
    }
}

// One block per row: block-reduce the row absmax, then quantize the row.
// gfx1103 iGPU path (selected in launch_quantize_int8_rowwise_kernel): reads x
// once when the row fits the per-thread register budget (K <= 12288),
// which on a bandwidth-limited iGPU saves a full second global read of x; the
// two-pass fallback covers arbitrarily long rows. Loads are 16-byte chunks
// (8 bf16/fp16 or 4 fp32 per thread), so a warp covers 512 bytes of a row per
// request. The staged values use unrolled loops with statically-known indices
// so they stay in registers (dynamic indexing would spill them to local
// memory, which on an iGPU is a round trip through DRAM). The K % 16 == 0
// contract of the callers keeps every row 16-byte aligned; the scalar path
// below covers anything else.
constexpr int kQuantMaxStaged = 48;  // floats per thread, K / 256 <= 48

__global__ __launch_bounds__(256) void quantize_int8_rowwise_kernel_staged(
    const void* __restrict__ x, int in_code, int8_t* __restrict__ q,
    float* __restrict__ scales, int M, int K) {

    __shared__ float red[8];  // one slot per warp
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const int w = t / 32;
    const int lane = t % 32;
    const int64_t base = static_cast<int64_t>(row) * K;

    const bool staged = ((K + 255) / 256 <= kQuantMaxStaged);
    const bool is_f32 = (in_code == 0);
    const bool vec = is_f32 ? (K % 4 == 0) : (K % 8 == 0);
    const bool packed = staged && vec;

    float vals[kQuantMaxStaged];  // register-resident when staged
    float lmax = 0.0f;

    if (!vec) {
        for (int j = t; j < K; j += 256)
            lmax = fmaxf(lmax, fabsf(load_in(x, base + j, in_code)));
    } else if (is_f32) {
        // One 16-byte chunk (4 floats) per lane per block-wide step of 1024
        // elements. The staged rows keep c statically bounded so vals[] stays in
        // registers (runtime c would spill it to local memory); a row longer than
        // the staged window reduces every chunk with runtime c so the absmax scale
        // still spans the whole row before the two-pass output below quantizes it.
        if (staged) {
            #pragma unroll
            for (int c = 0; c < kQuantMaxStaged / 4; ++c) {
                const int j = t * 4 + c * 1024;
                if (j < K) {
                    const uint4 v = *reinterpret_cast<const uint4*>(
                        static_cast<const char*>(x) + (base + j) * 4);
                    float tmp[4];
                    load_vec16_rt(v, tmp, 0);
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        lmax = fmaxf(lmax, fabsf(tmp[i]));
                        vals[c * 4 + i] = tmp[i];
                    }
                }
            }
        } else {
            for (int j = t * 4; j < K; j += 1024) {
                const uint4 v = *reinterpret_cast<const uint4*>(
                    static_cast<const char*>(x) + (base + j) * 4);
                float tmp[4];
                load_vec16_rt(v, tmp, 0);
                #pragma unroll
                for (int i = 0; i < 4; ++i) lmax = fmaxf(lmax, fabsf(tmp[i]));
            }
        }
    } else {
        if (staged) {
            #pragma unroll
            for (int c = 0; c < kQuantMaxStaged / 8; ++c) {
                const int j = t * 8 + c * 2048;
                if (j < K) {
                    const uint4 v = *reinterpret_cast<const uint4*>(
                        static_cast<const char*>(x) + (base + j) * 2);
                    float tmp[8];
                    load_vec16_rt(v, tmp, in_code);
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        lmax = fmaxf(lmax, fabsf(tmp[i]));
                        vals[c * 8 + i] = tmp[i];
                    }
                }
            }
        } else {
            for (int j = t * 8; j < K; j += 2048) {
                const uint4 v = *reinterpret_cast<const uint4*>(
                    static_cast<const char*>(x) + (base + j) * 2);
                float tmp[8];
                load_vec16_rt(v, tmp, in_code);
                #pragma unroll
                for (int i = 0; i < 8; ++i) lmax = fmaxf(lmax, fabsf(tmp[i]));
            }
        }
    }

    // Warp-shuffle block reduction: two barriers instead of one per halving
    // step of the full block.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) lmax = fmaxf(lmax, __shfl_xor(lmax, off, 32));
    if (lane == 0) red[w] = lmax;
    __syncthreads();
    if (w == 0) {
        const float tmax = (lane < 8) ? red[lane] : 0.0f;
        float v = tmax;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor(v, off, 32));
        if (lane == 0) red[0] = v;
    }
    __syncthreads();

    const float scale = fmaxf(red[0] / kInt8Max, 1e-30f);
    if (t == 0) scales[row] = scale;
    const float inv = 1.0f / scale;

    // Packed (staged and vectorizable) rows write out of the staged registers.
    // Short non-vectorizable rows (K not a whole chunk) leave vals unfilled, so
    // they fall through to the scalar second pass below. packed implies staged,
    // so c stays statically bounded and vals[] stays in registers.
    if (packed) {
        if (is_f32) {
            #pragma unroll
            for (int c = 0; c < kQuantMaxStaged / 4; ++c) {
                const int j = t * 4 + c * 1024;
                if (j < K) quant_store_vec_rt(vals + c * 4, q + base + j, inv, 4);
            }
        } else {
            #pragma unroll
            for (int c = 0; c < kQuantMaxStaged / 8; ++c) {
                const int j = t * 8 + c * 2048;
                if (j < K) quant_store_vec_rt(vals + c * 8, q + base + j, inv, 8);
            }
        }
        return;
    }

    // Two-pass fallback for rows longer than the register budget.
    if (!vec) {
        for (int j = t; j < K; j += 256)
            q[base + j] = quant_round(load_in(x, base + j, in_code), inv);
    } else if (is_f32) {
        for (int j = t * 4; j < K; j += 1024) {
            const uint4 v = *reinterpret_cast<const uint4*>(
                static_cast<const char*>(x) + (base + j) * 4);
            float tmp[4];
            load_vec16_rt(v, tmp, 0);
            quant_store_vec_rt(tmp, q + base + j, inv, 4);
        }
    } else {
        for (int j = t * 8; j < K; j += 2048) {
            const uint4 v = *reinterpret_cast<const uint4*>(
                static_cast<const char*>(x) + (base + j) * 2);
            float tmp[8];
            load_vec16_rt(v, tmp, in_code);
            quant_store_vec_rt(tmp, q + base + j, inv, 8);
        }
    }
}

// Upstream implementation, kept for every architecture other than the gfx1103
// iGPU: the register-staged rewrite above trades a second global read of x for
// register pressure and was tuned against the 6-WGP 780M, where the re-read is
// the more expensive side of the trade. This is the scalar two-pass original:
// block-reduce the row absmax through LDS, then quantize with one strided pass.
__global__ __launch_bounds__(256) void quantize_int8_rowwise_kernel(
    const void* __restrict__ x, int in_code, int8_t* __restrict__ q,
    float* __restrict__ scales, int M, int K) {

    __shared__ float red[256];
    const int row = blockIdx.x;
    const int t = threadIdx.x;

    float lmax = 0.0f;
    for (int j = t; j < K; j += 256)
        lmax = fmaxf(lmax, fabsf(load_in(x, static_cast<int64_t>(row) * K + j, in_code)));

    red[t] = lmax;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (t < s) red[t] = fmaxf(red[t], red[t + s]);
        __syncthreads();
    }

    const float scale = fmaxf(red[0] / kInt8Max, 1e-30f);
    if (t == 0) scales[row] = scale;
    const float inv = 1.0f / scale;

    for (int j = t; j < K; j += 256) {
        const float v = load_in(x, static_cast<int64_t>(row) * K + j, in_code);
        q[static_cast<int64_t>(row) * K + j] = quant_round(v, inv);
    }
}

__global__ void absmax_kernel(const void* __restrict__ x, int in_code,
                              unsigned int* __restrict__ out, int64_t numel) {
    __shared__ float red[256];
    const int t = threadIdx.x;

    float lmax = 0.0f;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + t; i < numel;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
        lmax = fmaxf(lmax, fabsf(load_in(x, i, in_code)));

    red[t] = lmax;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (t < s) red[t] = fmaxf(red[t], red[t + s]);
        __syncthreads();
    }
    // Non-negative floats compare identically to their bit patterns as unsigned.
    if (t == 0) atomicMax(out, __float_as_uint(red[0]));
}

__global__ void scale_from_absmax_kernel(const unsigned int* __restrict__ absmax,
                                         float* __restrict__ scale) {
    if (threadIdx.x == 0) *scale = fmaxf(__uint_as_float(*absmax) / kInt8Max, 1e-30f);
}

__global__ void quantize_int8_tensorwise_kernel(
    const void* __restrict__ x, int in_code, int8_t* __restrict__ q,
    const float* __restrict__ scale, int64_t numel) {

    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    q[i] = quant_round(load_in(x, i, in_code), 1.0f / scale[0]);
}

template <typename T>
__forceinline__ __device__ void store_dequant(T* output, int64_t i, float value) {
    output[i] = static_cast<T>(value);
}

template <>
__forceinline__ __device__ void store_dequant<__half>(
    __half* output, int64_t i, float value) {
    output[i] = __float2half(value);
}

template <typename OutputT>
__global__ void dequantize_int8_simple_kernel(
    const int8_t* __restrict__ q, const float* __restrict__ scales,
    OutputT* __restrict__ output, int64_t numel, int64_t inner_dim,
    int scale_mode) {

    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < numel; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t scale_idx = 0;
        if (scale_mode == 1) {
            scale_idx = i;
        } else if (scale_mode == 2) {
            scale_idx = i / inner_dim;
        }
        store_dequant(output, i, static_cast<float>(q[i]) * scales[scale_idx]);
    }
}

// One block per row, rotating up to 256 values at a time. Unlike the fused
// quantizer, this does not stage the whole row, so K is not limited by the
// device's per-block LDS capacity.
template <typename OutputT>
__global__ __launch_bounds__(256) void dequantize_int8_convrot_weight_kernel(
    const int8_t* __restrict__ q, const float* __restrict__ scales,
    OutputT* __restrict__ output, int K, int scale_size, int G) {

    const float h4[4][4] = {
        {1, 1, 1, -1},
        {1, 1, -1, 1},
        {1, -1, 1, 1},
        {-1, 1, 1, 1},
    };
    __shared__ float values[256];

    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const int groups_per_pass = 256 / G;
    const int local_group = t / G;
    const int element = t % G;
    const int local_base = local_group * G;
    const int group_count = K / G;
    const float scale = scales[scale_size == 1 ? 0 : row];
    const float norm = rsqrtf(static_cast<float>(G));

    int stage_count = 0;
    while ((1 << (2 * stage_count)) < G) {
        ++stage_count;
    }

    for (int group_base = 0; group_base < group_count;
         group_base += groups_per_pass) {
        const int group = group_base + local_group;
        const bool active = group < group_count;
        const int64_t col = static_cast<int64_t>(group) * G + element;
        values[t] = active
            ? static_cast<float>(q[static_cast<int64_t>(row) * K + col]) * scale
            : 0.0f;
        __syncthreads();

        for (int stage = 0; stage < stage_count; ++stage) {
            const int stride = 1 << (2 * stage);
            const int digit = (element / stride) & 3;
            const int base = local_base + element - digit * stride;
            const float v0 = values[base];
            const float v1 = values[base + stride];
            const float v2 = values[base + 2 * stride];
            const float v3 = values[base + 3 * stride];
            const float transformed =
                h4[digit][0] * v0 + h4[digit][1] * v1 +
                h4[digit][2] * v2 + h4[digit][3] * v3;
            __syncthreads();
            values[t] = transformed;
            __syncthreads();
        }

        if (active) {
            store_dequant(
                output, static_cast<int64_t>(row) * K + col, values[t] * norm);
        }
        __syncthreads();
    }
}

}  // namespace comfy::hip_backend

namespace comfy::hip_backend {

template <int ACT>
struct LaunchConvrotQuantGlobalManaged {
    const void* x;
    int in_dtype;
    int8_t* qout;
    float* scaleout;
    int M;
    int K;
    hipStream_t stream;
    void* spill_rotated;
    void* spill_partials;

    template <typename RowT>
    void operator()() const {
        launch_convrot_quant_global<RowT, ACT>(
            x, in_dtype, qout, scaleout, M, K,
            static_cast<RowT*>(spill_rotated),
            static_cast<float*>(spill_partials), stream);
    }
};

template <int ACT>
void launch_convrot_quant_global_managed(
    const void* x, int in_dtype, int8_t* qout, float* scaleout, int M, int K, hipStream_t stream,
    void* spill_rotated, void* spill_partials) {
    if (spill_rotated == nullptr || spill_partials == nullptr) {
        throw std::runtime_error("convrot global spill requires caller-provided workspace buffers");
    }

    dispatch_convrot_row_type(
        in_dtype,
        LaunchConvrotQuantGlobalManaged<ACT>{
            x,
            in_dtype,
            qout,
            scaleout,
            M,
            K,
            stream,
            spill_rotated,
            spill_partials});
}

template void launch_convrot_quant_global_managed<kActNone>(
    const void*, int, int8_t*, float*, int, int, hipStream_t, void*, void*);
template void launch_convrot_quant_global_managed<kActGeluTanh>(
    const void*, int, int8_t*, float*, int, int, hipStream_t, void*, void*);
template void launch_convrot_quant_global_managed<kActSwiGLU>(
    const void*, int, int8_t*, float*, int, int, hipStream_t, void*, void*);

}  // namespace comfy::hip_backend

extern "C" void launch_quantize_int8_rowwise_kernel(
    const void* x, int in_code, void* q, void* scales, int M, int K, hipStream_t stream) {

    using namespace comfy::hip_backend;
    if (M == 0) {
        return;  // a zero-block launch is hipErrorInvalidConfiguration
    }
    // Architecture dispatch: the staged kernel is the gfx1103 iGPU tuning; every
    // other GPU runs the upstream scalar kernel above.
    if (comfy_small_igpu()) {
        quantize_int8_rowwise_kernel_staged<<<M, 256, 0, stream>>>(
            x, in_code, static_cast<int8_t*>(q), static_cast<float*>(scales), M, K);
    } else {
        quantize_int8_rowwise_kernel<<<M, 256, 0, stream>>>(
            x, in_code, static_cast<int8_t*>(q), static_cast<float*>(scales), M, K);
    }
}

// Fused ConvRot rotation + rowwise int8 quantize. act_code: 0 none, 1 gelu tanh,
// 2 swiglu, 3 rms_norm (G=256 only, weight in the input dtype). K is the activated
// (written) width; swiglu reads a row twice as wide.
extern "C" void launch_quantize_int8_convrot_kernel(
    const void* x, int in_code, void* q, void* scales, void* spill_rotated, void* spill_partials,
    int M, int K, int group_size, int act_code, const void* act_weight, float act_eps,
    hipStream_t stream) {

    using namespace comfy::hip_backend;
    check_convrot_group_size(group_size);
    check_convrot_k(K, group_size, in_code, /*int8_quant=*/true);
    check_convrot_act(act_code);
    if (act_code == kActRmsNorm && group_size != 256) {
        throw std::runtime_error("convrot: rms_norm activation requires group_size 256");
    }
    if (M == 0) {
        return;  // a zero-block launch is hipErrorInvalidConfiguration
    }
    if (act_code == kActRmsNorm) {
        launch_convrot_quant_rms_norm(
            x, in_code, static_cast<int8_t*>(q), static_cast<float*>(scales),
            M, K, act_weight, act_eps, stream);
    } else if (act_code == kActGeluTanh) {
        launch_convrot_quant<false, kActGeluTanh>(
            x, in_code, static_cast<int8_t*>(q), static_cast<float*>(scales),
            M, K, group_size, stream, spill_rotated, spill_partials);
    } else if (act_code == kActSwiGLU) {
        launch_convrot_quant<false, kActSwiGLU>(
            x, in_code, static_cast<int8_t*>(q), static_cast<float*>(scales),
            M, K, group_size, stream, spill_rotated, spill_partials);
    } else {
        launch_convrot_quant<false, kActNone>(
            x, in_code, static_cast<int8_t*>(q), static_cast<float*>(scales),
            M, K, group_size, stream, spill_rotated, spill_partials);
    }
}

// Returns 1 when INT8 G=256 must use the global spill path for (M, K, in_dtype).
extern "C" int convrot_int8_needs_spill_host(int M, int K, int in_code) {
    using namespace comfy::hip_backend;
    return convrot_int8_needs_spill_buffers(M, K, in_code) ? 1 : 0;
}

// scratch is a 4-byte device buffer, pre-zeroed by the caller.
extern "C" void launch_quantize_int8_tensorwise_kernel(
    const void* x, int in_code, void* q, void* scale, void* scratch,
    int64_t numel, hipStream_t stream) {

    using namespace comfy::hip_backend;
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    const int rblocks = blocks < 1024 ? blocks : 1024;

    // An empty input has no absmax and nothing to quantize, but the scale is still
    // an output, so only the zero-block launches are skipped. scratch is zeroed,
    // which the scale kernel clamps.
    if (numel > 0) {
        absmax_kernel<<<rblocks, threads, 0, stream>>>(
            x, in_code, static_cast<unsigned int*>(scratch), numel);
    }
    scale_from_absmax_kernel<<<1, 1, 0, stream>>>(
        static_cast<const unsigned int*>(scratch), static_cast<float*>(scale));
    if (numel > 0) {
        quantize_int8_tensorwise_kernel<<<blocks, threads, 0, stream>>>(
            x, in_code, static_cast<int8_t*>(q), static_cast<const float*>(scale), numel);
    }
}

extern "C" void launch_dequantize_int8_simple_kernel(
    const void* q, const void* scales, void* output, int64_t numel,
    int64_t inner_dim, int scale_mode, int output_dtype_code, hipStream_t stream) {

    using namespace comfy::hip_backend;
    if (numel == 0) {
        return;
    }
    if (inner_dim <= 0) {
        throw std::runtime_error("dequantize_int8_simple: inner_dim must be positive");
    }
    if (scale_mode < 0 || scale_mode > 2) {
        throw std::runtime_error("dequantize_int8_simple: invalid scale mode");
    }

    constexpr int threads = 256;
    const int64_t required_blocks = (numel + threads - 1) / threads;
    const int blocks = static_cast<int>(required_blocks < 4096 ? required_blocks : 4096);

#define CK_LAUNCH_DEQUANT_SIMPLE(OutputT)                                                \
    dequantize_int8_simple_kernel<OutputT><<<blocks, threads, 0, stream>>>(              \
        static_cast<const int8_t*>(q), static_cast<const float*>(scales),                \
        static_cast<OutputT*>(output), numel, inner_dim, scale_mode)

    if (output_dtype_code == 0) {
        CK_LAUNCH_DEQUANT_SIMPLE(float);
    } else if (output_dtype_code == 1) {
        CK_LAUNCH_DEQUANT_SIMPLE(__half);
    } else if (output_dtype_code == 2) {
        CK_LAUNCH_DEQUANT_SIMPLE(__bf16);
    } else {
        throw std::runtime_error("dequantize_int8_simple: unsupported output dtype code");
    }
#undef CK_LAUNCH_DEQUANT_SIMPLE
}

extern "C" void launch_dequantize_int8_convrot_weight_kernel(
    const void* q, const void* scales, void* output, int M, int K,
    int scale_size, int group_size, int output_dtype_code, hipStream_t stream) {

    using namespace comfy::hip_backend;
    check_convrot_group_size(group_size);
    if (K % group_size != 0) {
        throw std::runtime_error(
            "dequantize_int8_convrot_weight: K=" + std::to_string(K) +
            " is not divisible by group_size=" + std::to_string(group_size));
    }
    if (scale_size != 1 && scale_size != M) {
        throw std::runtime_error(
            "dequantize_int8_convrot_weight: scale must be scalar or per-row");
    }
    if (M == 0 || K == 0) {
        return;
    }

#define CK_LAUNCH_DEQUANT_CONVROT(OutputT)                                               \
    dequantize_int8_convrot_weight_kernel<OutputT><<<M, 256, 0, stream>>>(               \
        static_cast<const int8_t*>(q), static_cast<const float*>(scales),                \
        static_cast<OutputT*>(output), K, scale_size, group_size)

    if (output_dtype_code == 0) {
        CK_LAUNCH_DEQUANT_CONVROT(float);
    } else if (output_dtype_code == 1) {
        CK_LAUNCH_DEQUANT_CONVROT(__half);
    } else if (output_dtype_code == 2) {
        CK_LAUNCH_DEQUANT_CONVROT(__bf16);
    } else {
        throw std::runtime_error(
            "dequantize_int8_convrot_weight: unsupported output dtype code");
    }
#undef CK_LAUNCH_DEQUANT_CONVROT
}
