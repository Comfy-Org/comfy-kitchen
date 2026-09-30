#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor_struct.h>

#include "torch_version.h"
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
#include <torch/csrc/stable/tensor_inl.h>
#endif

#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#else
#error "attn_gfx103x.cu is only intended for ROCm/HIP."
#endif

#include "reduction_utils.cuh"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using torch::stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;

namespace {

constexpr int kNHD = 0;
constexpr int kHND = 1;
constexpr float kLog2e = 1.4426950408889634f;

constexpr int MIN_BLK_Q = 32;
constexpr int MIN_BLK_K = 16;

#include "mma_gfx10.h"

// Every operand the attention kernel takes, gathered so the launcher can stay a
// template over the tile shape and the mask representation instead of a wall of
// positional arguments. The pointer fields are the raw device pointers; the
// stride fields are in elements, as the kernels index them.
struct AttnLaunch {
    const int8_t* q;
    const int8_t* k;
    const __half* v;
    void* out;
    const float* q_scale;
    const float* k_scale;
    const void* q_fp;
    int q_src_bf16;
    float sm_scale_log2e;
    int64_t batch, qo_len, kv_len, q_heads, kv_heads;
    int64_t q_stride_b, q_stride_n, q_stride_h;
    int64_t q_fp_stride_b, q_fp_stride_n, q_fp_stride_h;
    int64_t k_stride_b, k_stride_n, k_stride_h;
    int64_t v_stride_b, v_stride_n, v_stride_h;
    int64_t o_stride_b, o_stride_n, o_stride_h;
    int64_t qs_stride_b, qs_stride_h, ks_stride_b, ks_stride_h;
    // Mask. `mask` is null when mode is kNone; the strides are only meaningful
    // for kRaw, every prepared form being contiguous (see gfx10_mask_row).
    const void* mask;
    int mask_mode, mask_dtype;
    int64_t mask_sb, mask_sh, mask_sq, mask_sk, mask_batch, mask_heads, mask_q_tiles;
    int64_t mask_k_tiles;
    // The key mask's real row width, which the producer pads past the exact
    // tiles*65 so its vector loads stay aligned.
    int64_t mask_k_row_pitch;
};

// The launch context: the operands plus the geometry of the grid they run on.
// Bundled so the mask-mode dispatcher below can take one argument instead of six
// and stay readable at the call sites.
struct AttnLaunchCtx {
    AttnLaunch ops;
    dim3 grid;
    dim3 block;
    hipStream_t stream;
    int mask_mode;
    int mask_dtype;
};

// Dispatches one kernel instantiation. MASK and MASK_DTYPE are template
// parameters rather than arguments to the kernel for the same reason HD and TM
// are: a runtime mode would put a branch and a live register on the critical
// path of every key in the inner loop, where the register budget is already the
// binding constraint on this architecture.
template <int HD, bool CAUSAL, int BN, int TM, typename ODT, bool INQ, int MASK, int MASK_DTYPE>
void launch_attn(const AttnLaunchCtx& c) {
    const AttnLaunch& a = c.ops;
    // The in-kernel-Q path reads Q from the fp16/bf16 source; otherwise it reads
    // the packed int8 the prepass wrote.
    const int64_t qi_b = INQ ? a.q_fp_stride_b : a.q_stride_b;
    const int64_t qi_n = INQ ? a.q_fp_stride_n : a.q_stride_n;
    const int64_t qi_h = INQ ? a.q_fp_stride_h : a.q_stride_h;
    sageattn_gfx10::attn_kernel_i8q_f16pv_tiled_pv<HD, CAUSAL, BN, 128, ODT, TM, INQ, MASK,
                                                   MASK_DTYPE>
        <<<c.grid, c.block, 0, c.stream>>>(
        a.q, a.k, a.v, static_cast<ODT*>(a.out), a.q_scale, a.k_scale, a.batch, a.qo_len,
        a.kv_len, a.q_heads, a.kv_heads, qi_b, qi_n, qi_h, a.k_stride_b, a.k_stride_n,
        a.k_stride_h, a.v_stride_b, a.v_stride_n, a.v_stride_h, a.o_stride_b, a.o_stride_n,
        a.o_stride_h, a.qs_stride_b, a.qs_stride_h, a.ks_stride_b, a.ks_stride_h, 0, a.q_fp,
        a.q_src_bf16, a.sm_scale_log2e, a.mask, a.mask_sb, a.mask_sh, a.mask_sq, a.mask_sk,
        a.mask_batch, a.mask_heads, a.mask_q_tiles, a.mask_k_tiles, a.mask_k_row_pitch);
}

// Resolves the runtime mask mode into the template argument the kernel takes.
// The dtype is a second dimension of the same dispatch: kRaw is the only mode
// that cares, and a Boolean mask and a float mask compile to different reads.
template <int HD, bool CAUSAL, int BN, int TM, typename ODT, bool INQ>
void launch_attn_mask(const AttnLaunchCtx& c) {
    const AttnLaunch& a = c.ops;
    switch (c.mask_mode) {
        case sageattn_gfx10::MaskMode::kNone:
            launch_attn<HD, CAUSAL, BN, TM, ODT, INQ, sageattn_gfx10::MaskMode::kNone, 0>(c);
            return;
        case sageattn_gfx10::MaskMode::kPreparedKey:
            launch_attn<HD, CAUSAL, BN, TM, ODT, INQ, sageattn_gfx10::MaskMode::kPreparedKey, 0>(c);
            return;
        case sageattn_gfx10::MaskMode::kPreparedDense:
            launch_attn<HD, CAUSAL, BN, TM, ODT, INQ, sageattn_gfx10::MaskMode::kPreparedDense, 0>(c);
            return;
        case sageattn_gfx10::MaskMode::kPreparedDenseBool:
            launch_attn<HD, CAUSAL, BN, TM, ODT, INQ,
                        sageattn_gfx10::MaskMode::kPreparedDenseBool, 0>(c);
            return;
        case sageattn_gfx10::MaskMode::kPreparedDenseBF16:
            launch_attn<HD, CAUSAL, BN, TM, ODT, INQ,
                        sageattn_gfx10::MaskMode::kPreparedDenseBF16, 0>(c);
            return;
        case sageattn_gfx10::MaskMode::kPreparedDenseF16:
            launch_attn<HD, CAUSAL, BN, TM, ODT, INQ,
                        sageattn_gfx10::MaskMode::kPreparedDenseF16, 0>(c);
            return;
        case sageattn_gfx10::MaskMode::kRaw:
            break;
        default:
            throw std::runtime_error("attn_gfx103x: unknown attn_mask mode");
    }
    // kRaw: the dtype selects which load the inner loop uses. 0 fp32, 1 fp16,
    // 2 bf16, 3 bool -- the same codes the main HIP backend's mask helpers use.
    switch (c.mask_dtype) {
        case 3:
            launch_attn<HD, CAUSAL, BN, TM, ODT, INQ, sageattn_gfx10::MaskMode::kRaw, 3>(c);
            return;
        case 1:
            launch_attn<HD, CAUSAL, BN, TM, ODT, INQ, sageattn_gfx10::MaskMode::kRaw, 1>(c);
            return;
        case 2:
            launch_attn<HD, CAUSAL, BN, TM, ODT, INQ, sageattn_gfx10::MaskMode::kRaw, 2>(c);
            return;
        default:
            launch_attn<HD, CAUSAL, BN, TM, ODT, INQ, sageattn_gfx10::MaskMode::kRaw, 0>(c);
            return;
    }
}

// Resolves the runtime tile shape onto the template arguments the kernel takes.
// Every level is a template parameter for the same reason: the mask lookup and
// the tile geometry sit in the innermost loop, where a runtime branch would cost
// a register and an issue slot per key. Only the mask mode, its dtype, and the
// tuning overrides stay runtime, and those are decided once per launch.
template <bool CAUSAL>
void launch_attn_cf(const AttnLaunchCtx& c, int64_t head_dim, bool inq, bool out_bf, int v10_bn,
                    int tm) {
    if (head_dim == 64) {
        // D64 runs a 32-key tile: BN=16 halves the V tile and measured slower.
        if (out_bf) {
            if (inq) {
                if (tm == 2)      { launch_attn_mask<64, CAUSAL, 32, 2, __hip_bfloat16, true>(c); }
                else if (tm == 4) { launch_attn_mask<64, CAUSAL, 32, 4, __hip_bfloat16, true>(c); }
                else              { launch_attn_mask<64, CAUSAL, 32, 8, __hip_bfloat16, true>(c); }
            } else {
                if (tm == 2)      { launch_attn_mask<64, CAUSAL, 32, 2, __hip_bfloat16, false>(c); }
                else if (tm == 4) { launch_attn_mask<64, CAUSAL, 32, 4, __hip_bfloat16, false>(c); }
                else              { launch_attn_mask<64, CAUSAL, 32, 8, __hip_bfloat16, false>(c); }
            }
        } else {
            if (inq) {
                if (tm == 2)      { launch_attn_mask<64, CAUSAL, 32, 2, __half, true>(c); }
                else if (tm == 4) { launch_attn_mask<64, CAUSAL, 32, 4, __half, true>(c); }
                else              { launch_attn_mask<64, CAUSAL, 32, 8, __half, true>(c); }
            } else {
                if (tm == 2)      { launch_attn_mask<64, CAUSAL, 32, 2, __half, false>(c); }
                else if (tm == 4) { launch_attn_mask<64, CAUSAL, 32, 4, __half, false>(c); }
                else              { launch_attn_mask<64, CAUSAL, 32, 8, __half, false>(c); }
            }
        }
        return;
    }
    // D128 tunes its key tile between 16 and 32.
    if (v10_bn == 32) {
        if (out_bf) {
            if (inq) {
                if (tm == 2)      { launch_attn_mask<128, CAUSAL, 32, 2, __hip_bfloat16, true>(c); }
                else if (tm == 4) { launch_attn_mask<128, CAUSAL, 32, 4, __hip_bfloat16, true>(c); }
                else              { launch_attn_mask<128, CAUSAL, 32, 8, __hip_bfloat16, true>(c); }
            } else {
                if (tm == 2)      { launch_attn_mask<128, CAUSAL, 32, 2, __hip_bfloat16, false>(c); }
                else if (tm == 4) { launch_attn_mask<128, CAUSAL, 32, 4, __hip_bfloat16, false>(c); }
                else              { launch_attn_mask<128, CAUSAL, 32, 8, __hip_bfloat16, false>(c); }
            }
        } else {
            if (inq) {
                if (tm == 2)      { launch_attn_mask<128, CAUSAL, 32, 2, __half, true>(c); }
                else if (tm == 4) { launch_attn_mask<128, CAUSAL, 32, 4, __half, true>(c); }
                else              { launch_attn_mask<128, CAUSAL, 32, 8, __half, true>(c); }
            } else {
                if (tm == 2)      { launch_attn_mask<128, CAUSAL, 32, 2, __half, false>(c); }
                else if (tm == 4) { launch_attn_mask<128, CAUSAL, 32, 4, __half, false>(c); }
                else              { launch_attn_mask<128, CAUSAL, 32, 8, __half, false>(c); }
            }
        }
        return;
    }
    if (out_bf) {
        if (inq) {
            if (tm == 2)      { launch_attn_mask<128, CAUSAL, 16, 2, __hip_bfloat16, true>(c); }
            else if (tm == 4) { launch_attn_mask<128, CAUSAL, 16, 4, __hip_bfloat16, true>(c); }
            else              { launch_attn_mask<128, CAUSAL, 16, 8, __hip_bfloat16, true>(c); }
        } else {
            if (tm == 2)      { launch_attn_mask<128, CAUSAL, 16, 2, __hip_bfloat16, false>(c); }
            else if (tm == 4) { launch_attn_mask<128, CAUSAL, 16, 4, __hip_bfloat16, false>(c); }
            else              { launch_attn_mask<128, CAUSAL, 16, 8, __hip_bfloat16, false>(c); }
        }
    } else {
        if (inq) {
            if (tm == 2)      { launch_attn_mask<128, CAUSAL, 16, 2, __half, true>(c); }
            else if (tm == 4) { launch_attn_mask<128, CAUSAL, 16, 4, __half, true>(c); }
            else              { launch_attn_mask<128, CAUSAL, 16, 8, __half, true>(c); }
        } else {
            if (tm == 2)      { launch_attn_mask<128, CAUSAL, 16, 2, __half, false>(c); }
            else if (tm == 4) { launch_attn_mask<128, CAUSAL, 16, 4, __half, false>(c); }
            else              { launch_attn_mask<128, CAUSAL, 16, 8, __half, false>(c); }
        }
    }
}

Tensor new_empty_like(const Tensor& like, std::initializer_list<int64_t> sizes, ScalarType dtype) {
    return torch::stable::new_empty(like, std::vector<int64_t>(sizes), std::make_optional(dtype));
}

hipStream_t current_hip_stream(const Tensor& tensor) {
    int32_t device_index = tensor.get_device_index();
    void* stream = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &stream));
    return reinterpret_cast<hipStream_t>(stream);
}

__device__ __forceinline__ float to_float(const __half v) { return __half2float(v); }
__device__ __forceinline__ float to_float(const __hip_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ __half from_float_f16(float v) { return __float2half_rn(v); }
__device__ __forceinline__ __hip_bfloat16 from_float_bf16(float v) { return __float2bfloat16(v); }

__device__ __forceinline__ int8_t float_to_int8(float x) {
    x += (x >= 0.0f) ? 0.5f : -0.5f;
    int32_t rounded;
    asm volatile("v_cvt_i32_f32 %[dst], %[src]" : [dst] "=v"(rounded) : [src] "v"(x));
    rounded = rounded > 127 ? 127 : rounded;
    rounded = rounded < -128 ? -128 : rounded;
    return static_cast<int8_t>(rounded);
}

template <typename T>
__global__ void mean_hnd_kernel(
    const T* __restrict__ input,
    T* __restrict__ mean_out,
    const int64_t seq_len,
    const int64_t heads,
    const int64_t head_dim,
    const int64_t in_stride_b,
    const int64_t in_stride_n,
    const int64_t in_stride_h) {
    constexpr int TileD = 16;
    constexpr int Threads = 256;
    __shared__ float partial_sum[16][256];

    const int tid = threadIdx.x;
    const int64_t d_base = static_cast<int64_t>(blockIdx.x) * TileD;
    const int64_t h = blockIdx.y;
    const int64_t b = blockIdx.z;
    if (d_base + TileD > head_dim) return;

    float acc[16];
#pragma unroll
    for (int c = 0; c < 16; ++c) acc[c] = 0.0f;
    for (int64_t s = tid; s < seq_len; s += Threads) {
        const int64_t offset = b * in_stride_b + s * in_stride_n + h * in_stride_h + d_base;
        const uint4* p4 = reinterpret_cast<const uint4*>(input + offset);
        const T* v = reinterpret_cast<const T*>(p4);
#pragma unroll
        for (int c = 0; c < 16; ++c) acc[c] += to_float(v[c]);
    }
#pragma unroll
    for (int c = 0; c < 16; ++c) partial_sum[c][tid] = acc[c];
    __syncthreads();
    if (tid < TileD) {
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < Threads; ++i) sum += partial_sum[tid][i];
        const int64_t mean_d = d_base + tid;
        const float value = sum / static_cast<float>(seq_len);
        if constexpr (std::is_same<T, __half>::value) {
            mean_out[(b * heads + h) * head_dim + mean_d] = from_float_f16(value);
        } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
            mean_out[(b * heads + h) * head_dim + mean_d] = from_float_bf16(value);
        } else {
            mean_out[(b * heads + h) * head_dim + mean_d] = value;
        }
    }
}

template <typename T, int HeadDim, int BLK, int MIN_BLK>
__global__ void quant_qk_int8_hnd_kernel(
    const T* __restrict__ input,
    int8_t* __restrict__ output,
    const T* __restrict__ key_mean,
    float* __restrict__ scale_out,
    const int64_t batch,
    const int64_t heads,
    const int64_t seq_len,
    const int scale_groups,
    const int is_q,
    const float sm_scale_log2e,
    const int64_t in_stride_b,
    const int64_t in_stride_n,
    const int64_t in_stride_h,
    const int64_t groups_per_block) {
    constexpr int Threads = 256;
    constexpr int RATIO = BLK / MIN_BLK;
    constexpr int PackElems = 8;
    __shared__ float shared_amax[RATIO];
    __shared__ uint4 shared_data[(BLK * HeadDim) / 8];

    const int head = blockIdx.y;
    const int b = blockIdx.z;
    if (b >= batch || head >= heads) return;
    const int total_blocks = static_cast<int>((seq_len + BLK - 1) / BLK);
    const float pass1_scale = is_q ? sm_scale_log2e : 1.0f;
    const float extra_scale = is_q ? sm_scale_log2e : 1.0f;

    for (int gi = 0; gi < groups_per_block; ++gi) {
        const int blk = blockIdx.x * groups_per_block + gi;
        if (blk >= total_blocks) break;
        const int64_t base_row = static_cast<int64_t>(blk) * BLK;
        const int tid = threadIdx.x;
        constexpr int Packs = (BLK * HeadDim) / 8;

        float local_amax[RATIO];
#pragma unroll
        for (int r = 0; r < RATIO; ++r) local_amax[r] = 1e-7f;
        for (int p = tid; p < Packs / 2; p += Threads) {
            const int pack = p * 2;
            const int elem_base = pack * PackElems;
            const int row = elem_base / HeadDim;
            const int d = elem_base - row * HeadDim;
            const int64_t seq = base_row + row;
            if (seq < seq_len) {
                const int64_t in_off = static_cast<int64_t>(b) * in_stride_b + seq * in_stride_n + head * in_stride_h + d;
                const uint4 raw0 = *reinterpret_cast<const uint4*>(input + in_off);
                const uint4 raw1 = *reinterpret_cast<const uint4*>(input + in_off + 8);
                shared_data[pack] = raw0;
                shared_data[pack + 1] = raw1;
                const int r = row / MIN_BLK;
                const T* v0 = reinterpret_cast<const T*>(&raw0);
                const T* v1 = reinterpret_cast<const T*>(&raw1);
                float am = local_amax[r];
                if (!is_q && key_mean != nullptr) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        float v = to_float(v0[i]) - to_float(key_mean[(b * heads + head) * HeadDim + d + i]);
                        am = fmaxf(am, fabsf(v * pass1_scale));
                    }
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        float v = to_float(v1[i]) - to_float(key_mean[(b * heads + head) * HeadDim + d + 8 + i]);
                        am = fmaxf(am, fabsf(v * pass1_scale));
                    }
                } else {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        float v = to_float(v0[i]);
                        am = fmaxf(am, fabsf(v * pass1_scale));
                    }
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        float v = to_float(v1[i]);
                        am = fmaxf(am, fabsf(v * pass1_scale));
                    }
                }
                local_amax[r] = am;
            } else {
                shared_data[pack] = make_uint4(0, 0, 0, 0);
                shared_data[pack + 1] = make_uint4(0, 0, 0, 0);
            }
        }
        for (int r = 0; r < RATIO; ++r) {
            const float block_amax = vllm::blockReduceMax(local_amax[r]);
            if (tid == 0) {
                shared_amax[r] = block_amax;
                if (base_row + static_cast<int64_t>(r) * MIN_BLK < seq_len) {
                    scale_out[(static_cast<int64_t>(b) * heads + head) * scale_groups + blk * RATIO + r] =
                        block_amax / 127.0f;
                }
            }
        }
        __syncthreads();
        float inv_scale[RATIO];
#pragma unroll
        for (int r = 0; r < RATIO; ++r) inv_scale[r] = 127.0f / shared_amax[r];

        for (int p = tid; p < Packs / 2; p += Threads) {
            const int pack = p * 2;
            const int elem_base = pack * PackElems;
            const int row = elem_base / HeadDim;
            const int d = elem_base - row * HeadDim;
            const int64_t seq = base_row + row;
            if (seq < seq_len) {
                const int r = row / MIN_BLK;
                const int64_t out_off = (static_cast<int64_t>(b) * heads + head) * seq_len * HeadDim + seq * HeadDim + d;
                const uint4 raw0 = shared_data[pack];
                const uint4 raw1 = shared_data[pack + 1];
                const T* values = reinterpret_cast<const T*>(&raw0);
                const T* values1 = reinterpret_cast<const T*>(&raw1);
                char4 out0, out1, out2, out3;
                float v0 = to_float(values[0]), v1 = to_float(values[1]);
                float v2 = to_float(values[2]), v3 = to_float(values[3]);
                float v4 = to_float(values[4]), v5 = to_float(values[5]);
                float v6 = to_float(values[6]), v7 = to_float(values[7]);
                float w0 = to_float(values1[0]), w1 = to_float(values1[1]);
                float w2 = to_float(values1[2]), w3 = to_float(values1[3]);
                float w4 = to_float(values1[4]), w5 = to_float(values1[5]);
                float w6 = to_float(values1[6]), w7 = to_float(values1[7]);
                if (!is_q && key_mean != nullptr) {
                    const int64_t mean_base = (b * heads + head) * HeadDim + d;
                    v0 -= to_float(key_mean[mean_base + 0]);
                    v1 -= to_float(key_mean[mean_base + 1]);
                    v2 -= to_float(key_mean[mean_base + 2]);
                    v3 -= to_float(key_mean[mean_base + 3]);
                    v4 -= to_float(key_mean[mean_base + 4]);
                    v5 -= to_float(key_mean[mean_base + 5]);
                    v6 -= to_float(key_mean[mean_base + 6]);
                    v7 -= to_float(key_mean[mean_base + 7]);
                    w0 -= to_float(key_mean[mean_base + 8]);
                    w1 -= to_float(key_mean[mean_base + 9]);
                    w2 -= to_float(key_mean[mean_base + 10]);
                    w3 -= to_float(key_mean[mean_base + 11]);
                    w4 -= to_float(key_mean[mean_base + 12]);
                    w5 -= to_float(key_mean[mean_base + 13]);
                    w6 -= to_float(key_mean[mean_base + 14]);
                    w7 -= to_float(key_mean[mean_base + 15]);
                }
                const float iscale = inv_scale[r] * extra_scale;
                out0.x = float_to_int8(v0 * iscale);
                out0.y = float_to_int8(v1 * iscale);
                out0.z = float_to_int8(v2 * iscale);
                out0.w = float_to_int8(v3 * iscale);
                out1.x = float_to_int8(v4 * iscale);
                out1.y = float_to_int8(v5 * iscale);
                out1.z = float_to_int8(v6 * iscale);
                out1.w = float_to_int8(v7 * iscale);
                out2.x = float_to_int8(w0 * iscale);
                out2.y = float_to_int8(w1 * iscale);
                out2.z = float_to_int8(w2 * iscale);
                out2.w = float_to_int8(w3 * iscale);
                out3.x = float_to_int8(w4 * iscale);
                out3.y = float_to_int8(w5 * iscale);
                out3.z = float_to_int8(w6 * iscale);
                out3.w = float_to_int8(w7 * iscale);
                *reinterpret_cast<char4*>(output + out_off) = out0;
                *reinterpret_cast<char4*>(output + out_off + 4) = out1;
                *reinterpret_cast<char4*>(output + out_off + 8) = out2;
                *reinterpret_cast<char4*>(output + out_off + 12) = out3;
            }
        }
        __syncthreads();
    }
}

}

Tensor mean_seq_gfx103x(Tensor input, int64_t tensor_layout) {
    const int64_t batch = input.size(0);
    const int64_t heads = (tensor_layout == kHND) ? input.size(1) : input.size(2);
    const int64_t seq_len = (tensor_layout == kHND) ? input.size(2) : input.size(1);
    const int64_t head_dim = input.size(3);

    const int64_t in_stride_b = input.stride(0);
    const int64_t in_stride_n = (tensor_layout == kHND) ? input.stride(2) : input.stride(1);
    const int64_t in_stride_h = (tensor_layout == kHND) ? input.stride(1) : input.stride(2);

    Tensor output = new_empty_like(input, {batch, heads, head_dim}, input.scalar_type());
    const hipStream_t stream = current_hip_stream(input);
    dim3 block(256);
    dim3 grid((head_dim + 15) / 16, heads, batch);

    if (input.scalar_type() == ScalarType::Half) {
        mean_hnd_kernel<__half><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(output.data_ptr()),
            seq_len, heads, head_dim,
            in_stride_b, in_stride_n, in_stride_h);
    } else if (input.scalar_type() == ScalarType::BFloat16) {
        mean_hnd_kernel<__hip_bfloat16><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __hip_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__hip_bfloat16*>(output.data_ptr()),
            seq_len, heads, head_dim,
            in_stride_b, in_stride_n, in_stride_h);
    } else {
        mean_hnd_kernel<float><<<grid, block, 0, stream>>>(
            reinterpret_cast<const float*>(input.data_ptr()),
            reinterpret_cast<float*>(output.data_ptr()),
            seq_len, heads, head_dim,
            in_stride_b, in_stride_n, in_stride_h);
    }
    return output;
}

std::tuple<Tensor, Tensor, Tensor, Tensor> quant_qk_int8_gfx103x(
    Tensor query, Tensor key, Tensor key_mean,
    int64_t tensor_layout, double sm_scale, int64_t skip_q) {

    const int64_t batch = query.size(0);
    const int64_t q_heads = (tensor_layout == kHND) ? query.size(1) : query.size(2);
    const int64_t kv_heads = (tensor_layout == kHND) ? key.size(1) : key.size(2);
    const int64_t q_len = (tensor_layout == kHND) ? query.size(2) : query.size(1);
    const int64_t kv_len = (tensor_layout == kHND) ? key.size(2) : key.size(1);
    const int64_t head_dim = query.size(3);

    Tensor q_int8 = new_empty_like(query, {batch, q_heads, q_len, head_dim}, ScalarType::Char);
    Tensor k_int8 = new_empty_like(key, {batch, kv_heads, kv_len, head_dim}, ScalarType::Char);

    const int q_groups = (q_len + MIN_BLK_Q - 1) / MIN_BLK_Q;
    const int k_groups = (kv_len + MIN_BLK_K - 1) / MIN_BLK_K;

    Tensor q_scale = new_empty_like(query, {batch, q_heads, q_groups}, ScalarType::Float);
    Tensor k_scale = new_empty_like(key, {batch, kv_heads, k_groups}, ScalarType::Float);
    if (skip_q) {

        q_int8 = new_empty_like(query, {0}, ScalarType::Char);
        q_scale = new_empty_like(query, {0}, ScalarType::Float);
    }

    const hipStream_t stream = current_hip_stream(query);
    const float sm_scale_log2e = static_cast<float>(sm_scale) * kLog2e;
    const bool has_mean = key_mean.numel() > 0;

    const int64_t q_sb = query.stride(0);
    const int64_t q_sn = (tensor_layout == kHND) ? query.stride(2) : query.stride(1);
    const int64_t q_sh = (tensor_layout == kHND) ? query.stride(1) : query.stride(2);
    const int64_t k_sb = key.stride(0);
    const int64_t k_sn = (tensor_layout == kHND) ? key.stride(2) : key.stride(1);
    const int64_t k_sh = (tensor_layout == kHND) ? key.stride(1) : key.stride(2);

    dim3 block(256);

    const int blk_sel = getenv("SAGEATTN_QUANT_BLK") ? atoi(getenv("SAGEATTN_QUANT_BLK")) : 1;
    constexpr int BLK_Q64 = 128;
    constexpr int BLK_K64 = 64;
    constexpr int BLK_Q128 = 64;
    constexpr int BLK_K128 = 32;
    int blk_q, blk_k;
    if (blk_sel == 128) { blk_q = BLK_Q64; blk_k = BLK_K64; }
    else if (blk_sel == 64) { blk_q = BLK_Q128; blk_k = BLK_K128; }
    else if (blk_sel == 0) { blk_q = MIN_BLK_Q; blk_k = MIN_BLK_K; }
    else if (head_dim == 64) { blk_q = BLK_Q64; blk_k = BLK_K64; }
    else { blk_q = MIN_BLK_Q; blk_k = MIN_BLK_K; }
    const int q_blocks = (q_len + blk_q - 1) / blk_q;
    const int k_blocks = (kv_len + blk_k - 1) / blk_k;

    int q_gpb = getenv("SAGEATTN_QUANT_GPB") ? atoi(getenv("SAGEATTN_QUANT_GPB")) : 1;
    if (q_gpb < 1) q_gpb = 1;
    dim3 grid_q((q_blocks + q_gpb - 1) / q_gpb, q_heads, batch);
    dim3 grid_k((k_blocks + q_gpb - 1) / q_gpb, kv_heads, batch);

    #define LAUNCH_QUANT(HD, T, BQ, BK) \
        do { \
            if (!skip_q) { \
                quant_qk_int8_hnd_kernel<T, HD, BQ, MIN_BLK_Q><<<grid_q, block, 0, stream>>>( \
                    reinterpret_cast<const T*>(query.data_ptr()), \
                    reinterpret_cast<int8_t*>(q_int8.data_ptr()), \
                    nullptr, \
                    reinterpret_cast<float*>(q_scale.data_ptr()), \
                    batch, q_heads, q_len, q_groups, 1, sm_scale_log2e, \
                    q_sb, q_sn, q_sh, q_gpb); \
            } \
            quant_qk_int8_hnd_kernel<T, HD, BK, MIN_BLK_K><<<grid_k, block, 0, stream>>>( \
                reinterpret_cast<const T*>(key.data_ptr()), \
                reinterpret_cast<int8_t*>(k_int8.data_ptr()), \
                has_mean ? reinterpret_cast<const T*>(key_mean.data_ptr()) : nullptr, \
                reinterpret_cast<float*>(k_scale.data_ptr()), \
                batch, kv_heads, kv_len, k_groups, 0, 1.0f, \
                k_sb, k_sn, k_sh, q_gpb); \
        } while(0)

    #define LAUNCH_QUANT_DISPATCH(HD, T) \
        do { \
            if (blk_q == BLK_Q64) { LAUNCH_QUANT(HD, T, 128, 64); } \
            else if (blk_q == BLK_Q128) { LAUNCH_QUANT(HD, T, 64, 32); } \
            else { LAUNCH_QUANT(HD, T, MIN_BLK_Q, MIN_BLK_K); } \
        } while(0)

    if (query.scalar_type() == ScalarType::Half) {
        if (head_dim == 64) { LAUNCH_QUANT_DISPATCH(64, __half); }
        else { LAUNCH_QUANT_DISPATCH(128, __half); }
    } else {
        if (head_dim == 64) { LAUNCH_QUANT_DISPATCH(64, __hip_bfloat16); }
        else { LAUNCH_QUANT_DISPATCH(128, __hip_bfloat16); }
    }
    #undef LAUNCH_QUANT_DISPATCH
    #undef LAUNCH_QUANT
    return {q_int8, q_scale, k_int8, k_scale};
}

Tensor qk_int8_sv_bf16_attn_gfx103x_t(
    Tensor query, Tensor key, Tensor value, Tensor output,
    Tensor q_scale, Tensor k_scale, Tensor v_scale,
    int64_t tensor_layout, int64_t is_causal, double sm_scale, Tensor q_fp,
    int64_t mask_mode, int64_t mask_dtype, Tensor mask) {
    (void)v_scale;

    const Tensor& qgeo = (query.dim() < 4) ? q_fp : query;
    const int64_t batch = qgeo.size(0);
    const int64_t q_heads = qgeo.size(1);
    const int64_t kv_heads = key.size(1);
    const int64_t qo_len = qgeo.size(2);
    const int64_t kv_len = key.size(2);
    const int64_t head_dim = qgeo.size(3);
    const int64_t q_stride_b = (query.dim() >= 4) ? query.stride(0) : 0;
    const int64_t q_stride_n = (query.dim() >= 4) ? query.stride(2) : 0;
    const int64_t q_stride_h = (query.dim() >= 4) ? query.stride(1) : 0;
    const int64_t k_stride_b = key.stride(0);
    const int64_t k_stride_n = key.stride(2);
    const int64_t k_stride_h = key.stride(1);

    const int64_t v_stride_b = value.stride(0);
    const int64_t v_stride_n = value.stride(1);
    const int64_t v_stride_h = value.stride(2);
    const int64_t o_stride_b = output.stride(0);
    const int64_t o_stride_n = (tensor_layout == kHND) ? output.stride(2) : output.stride(1);
    const int64_t o_stride_h = (tensor_layout == kHND) ? output.stride(1) : output.stride(2);
    const int64_t qs_stride_b = (q_scale.numel() > 0) ? q_scale.stride(0) : 0,
      qs_stride_h = (q_scale.numel() > 0) ? q_scale.stride(1) : 0;
    const int64_t ks_stride_b = k_scale.stride(0), ks_stride_h = k_scale.stride(1);
    const hipStream_t stream = current_hip_stream(query);

    // Mask descriptor. Raw masks are indexed by the caller's strides; every
    // prepared form is contiguous with a pitch the tile counts fix, so the
    // kernel indexes those arithmetically (see gfx10_mask_row).
    const int gmask = static_cast<int>(mask_mode);
    const bool masked = gmask != 0;
    int64_t mask_sb = 0, mask_sh = 0, mask_sq = 0, mask_sk = 0;
    int64_t mask_batch = 1, mask_heads = 1, mask_q_tiles = 0, mask_k_tiles = 0;
    int64_t mask_k_row_pitch = 0;
    if (masked) {
        if (gmask == sageattn_gfx10::MaskMode::kRaw) {
            if (mask.dim() != 4) {
                throw std::runtime_error("attn_gfx103x: a raw attn_mask must be [B,H,Q,K]");
            }
            mask_sb = mask.stride(0);
            mask_sh = mask.stride(1);
            mask_sq = mask.stride(2);
            mask_sk = mask.stride(3);
            // The compact extents tell the kernel which batch/head indices the
            // mask actually covers; a stride-0 axis is one it does not. The
            // strides are passed through unchanged -- they already encode the
            // broadcast, and rewriting them would break the in-bounds
            // arithmetic the producer's layout depends on.
            mask_batch = (mask_sb == 0) ? 1 : mask.size(0);
            mask_heads = (mask_sh == 0) ? 1 : mask.size(1);
        } else if (gmask == sageattn_gfx10::MaskMode::kPreparedKey) {
            // [B, H, tiles * 65]: the row pitch is the buffer's own width, and
            // the kernel derives the tile count from kv_len. There are no query
            // or key tile extents to read -- that is why this branch does not
            // touch size(2)/size(3), which a 3D tensor does not have.
            if (mask.dim() != 3) {
                throw std::runtime_error("attn_gfx103x: a prepared key mask must be [B,H,W]");
            }
            if (!mask.is_contiguous()) {
                throw std::runtime_error("attn_gfx103x: a prepared attn_mask must be contiguous");
            }
            mask_batch = mask.size(0);
            mask_heads = mask.size(1);
            mask_k_tiles = (kv_len + sageattn_gfx10::kGfx10TileK - 1) / sageattn_gfx10::kGfx10TileK;
            // The producer rounds the row up to a multiple of four so its own
            // vector loads stay aligned; the kernel has to stride by that width,
            // not by the exact tiles*65.
            const int64_t exact = mask_k_tiles * (sageattn_gfx10::kGfx10TileK + 1);
            mask_k_row_pitch = (exact + 3) / 4 * 4;
        } else {
            // Dense: [B, H, q_tiles, k_tiles, slots].
            if (mask.dim() != 5) {
                throw std::runtime_error("attn_gfx103x: a prepared dense mask must be 5D");
            }
            if (!mask.is_contiguous()) {
                throw std::runtime_error("attn_gfx103x: a prepared attn_mask must be contiguous");
            }
            mask_batch = mask.size(0);
            mask_heads = mask.size(1);
            mask_q_tiles = mask.size(2);
            mask_k_tiles = mask.size(3);
            // The buffer has to cover every (query, key) the kernel will ask
            // for, or the lookup runs off the end of the allocation.
            const int64_t need_q_tiles = (qo_len + sageattn_gfx10::kGfx10TileQ - 1) /
                                         sageattn_gfx10::kGfx10TileQ;
            const int64_t need_k_tiles = (kv_len + sageattn_gfx10::kGfx10TileK - 1) /
                                         sageattn_gfx10::kGfx10TileK;
            if (mask_q_tiles < need_q_tiles || mask_k_tiles < need_k_tiles) {
                throw std::runtime_error(
                    "attn_gfx103x: prepared dense mask covers " +
                    std::to_string(mask_q_tiles) + "x" + std::to_string(mask_k_tiles) +
                    " tiles but the call needs " + std::to_string(need_q_tiles) + "x" +
                    std::to_string(need_k_tiles));
            }
        }
    }

    const int v10_inq_mode = getenv("SAGEATTN_V10_INQ") ? atoi(getenv("SAGEATTN_V10_INQ")) : 2;
    const bool v10_inq_wanted = (v10_inq_mode == 1) || (v10_inq_mode == 2 && kv_len <= 1024);
    // The in-kernel-Q path is only real when an fp16/bf16 Q source came in; the
    // prepass path hands over an empty tensor to say "Q is already packed".
    const bool inq = v10_inq_wanted && q_fp.numel() > 0;
    const int q_fp_bf16 = (q_fp.scalar_type() == ScalarType::BFloat16) ? 1 : 0;

    const bool out_bf = (output.scalar_type() == ScalarType::BFloat16);

    AttnLaunch a{};
    a.q = reinterpret_cast<const int8_t*>(inq ? q_fp.data_ptr() : query.data_ptr());
    a.k = reinterpret_cast<const int8_t*>(key.data_ptr());
    a.v = reinterpret_cast<const __half*>(value.data_ptr());
    a.out = output.data_ptr();
    a.q_scale = reinterpret_cast<const float*>(q_scale.data_ptr());
    a.k_scale = reinterpret_cast<const float*>(k_scale.data_ptr());
    // The kernel always takes a Q pointer; on the prepass path it is the packed
    // int8, on the in-kernel path the fp16/bf16 source it quantizes itself.
    a.q_fp = inq ? reinterpret_cast<const void*>(q_fp.data_ptr()) : a.q;
    a.q_src_bf16 = inq ? q_fp_bf16 : 0;
    a.sm_scale_log2e = inq ? static_cast<float>(sm_scale) * kLog2e : 0.0f;
    a.batch = batch;
    a.qo_len = qo_len;
    a.kv_len = kv_len;
    a.q_heads = q_heads;
    a.kv_heads = kv_heads;
    a.q_stride_b = q_stride_b;
    a.q_stride_n = q_stride_n;
    a.q_stride_h = q_stride_h;
    // q_fp is the empty placeholder on the packed-Q path, so its strides are
    // only read when there is a real 4D source behind it.
    if (inq) {
        a.q_fp_stride_b = q_fp.stride(0);
        a.q_fp_stride_n = q_fp.stride(2);
        a.q_fp_stride_h = q_fp.stride(1);
    }
    a.k_stride_b = k_stride_b;
    a.k_stride_n = k_stride_n;
    a.k_stride_h = k_stride_h;
    a.v_stride_b = v_stride_b;
    a.v_stride_n = v_stride_n;
    a.v_stride_h = v_stride_h;
    a.o_stride_b = o_stride_b;
    a.o_stride_n = o_stride_n;
    a.o_stride_h = o_stride_h;
    a.qs_stride_b = qs_stride_b;
    a.qs_stride_h = qs_stride_h;
    a.ks_stride_b = ks_stride_b;
    a.ks_stride_h = ks_stride_h;
    a.mask = masked ? mask.data_ptr() : nullptr;
    a.mask_mode = gmask;
    a.mask_dtype = static_cast<int>(mask_dtype);
    a.mask_sb = mask_sb;
    a.mask_sh = mask_sh;
    a.mask_sq = mask_sq;
    a.mask_sk = mask_sk;
    a.mask_batch = mask_batch;
    a.mask_heads = mask_heads;
    a.mask_q_tiles = mask_q_tiles;
    a.mask_k_tiles = mask_k_tiles;
    a.mask_k_row_pitch = mask_k_row_pitch;

    AttnLaunchCtx ctx{};
    ctx.ops = a;
    ctx.block = dim3(128);
    ctx.grid = dim3(static_cast<unsigned>((qo_len + 127) / 128),
                    static_cast<unsigned>(q_heads), static_cast<unsigned>(batch));
    ctx.stream = stream;
    ctx.mask_mode = gmask;
    ctx.mask_dtype = static_cast<int>(mask_dtype);

    // Tile shape. The env overrides are the ones the ported kernel shipped with,
    // so a tuning experiment stays an environment variable and not a rebuild.
    int v10_bn, v10_tm;
    if (head_dim == 64) {
        const int bn_env = getenv("SAGEATTN_BN_D64") ? atoi(getenv("SAGEATTN_BN_D64")) : 32;
        v10_bn = (bn_env == 64) ? 64 : 32;
        v10_tm = getenv("SAGEATTN_TM_D64") ? atoi(getenv("SAGEATTN_TM_D64"))
              : (getenv("SAGEATTN_TM") ? atoi(getenv("SAGEATTN_TM")) : 8);
    } else {
        const int bn_env = getenv("SAGEATTN_BN_D128") ? atoi(getenv("SAGEATTN_BN_D128"))
                           : (getenv("SAGEATTN_BN") ? atoi(getenv("SAGEATTN_BN")) : 16);
        v10_bn = (bn_env == 32) ? 32 : 16;
        v10_tm = getenv("SAGEATTN_TM_D128") ? atoi(getenv("SAGEATTN_TM_D128"))
              : (getenv("SAGEATTN_TM") ? atoi(getenv("SAGEATTN_TM")) : 8);
    }
    const int tm = (v10_tm == 2) ? 2 : ((v10_tm == 4) ? 4 : 8);

    // One launcher covers every combination the old macro set spelled out. The
    // tile shape (head dim, causal, key tile, rows per thread, in-kernel Q
    // quantization) and the mask representation are all template arguments, so a
    // masked launch costs nothing the unmasked one did not already pay: the
    // lookup sits behind a constant, and the MaskMode::kNone instantiation
    // compiles to exactly the code it did before masks existed.
    //
    // Only the mask mode and its dtype are left as runtime switches, and they
    // cost one branch per launch. Everything else is a template argument because
    // a runtime tile choice would put a branch and a live register on the inner
    // loop's critical path, where the register budget is already the binding
    // constraint on this architecture.
    const bool causal = is_causal != 0;
    if (causal) { launch_attn_cf<true>(ctx, head_dim, inq, out_bf, v10_bn, tm); }
    else        { launch_attn_cf<false>(ctx, head_dim, inq, out_bf, v10_bn, tm); }
    return output;

    // is_causal selects the triangular schedule. comfy_kitchen's int8 attention
    // has no causal mode, but the ported op exposes the flag, so the
    // instantiation is kept rather than dropped.
    if (causal) { launch_attn_cf<true>(ctx, head_dim, inq, out_bf, v10_bn, tm); }
    else        { launch_attn_cf<false>(ctx, head_dim, inq, out_bf, v10_bn, tm); }
    #undef CK_DISPATCH_OUT128
    #undef CK_DISPATCH_OUT64
    #undef CK_DISPATCH_BN128
    #undef CK_DISPATCH_TM
    return output;
}