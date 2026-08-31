/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The exact 64Q x 64KV branch used by Sol when no approximate tail needs to be
// resumed. Q/K/V use Kitchen's existing pure-INT8 Sage carrier; the route is
// Sol's absolute uint16 block list plus one int32 count per query block.

#include "qk_int_sv_i8_cuda.cuh"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace {

constexpr int CTA_Q = 64;
constexpr int CTA_K = 64;
constexpr int WARP_Q = 16;
constexpr int WARP_K = 64;
constexpr int HEAD_DIM = 128;

template <typename DTypeOut>
void launch_impl(
    int8_t* q, int8_t* k, int8_t* v, DTypeOut* out,
    float* q_scale, float* k_scale, float* v_scale,
    const uint16_t* block_lut, const int32_t* valid_block_num,
    int batch, int sequence, int heads, int padded_sequence,
    int q_scale_stride_b, int q_scale_stride_h, float scale,
    cudaStream_t stream)
{
    const size_t smem = std::max(
        static_cast<size_t>((CTA_Q + CTA_K + CTA_K) * HEAD_DIM * sizeof(int8_t)),
        static_cast<size_t>(CTA_Q * HEAD_DIM * sizeof(half)));

    auto kernel = qk_int_sv_i8_attn_kernel<
        CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM,
        DataType::kInt8, QuantGranularity::kPerThread,
        QuantGranularity::kPerThread, float, false, DTypeOut,
        ComputeUnit::kCudaCore, MaskMode::kNone, false, true, false, false, true, true>;

    cudaError_t error = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    if (error != cudaSuccess) {
        throw std::runtime_error(
            "sol_attn failed to request " + std::to_string(smem) +
            " bytes for the Sage exact branch: " + cudaGetErrorString(error));
    }

    const int q_tiles = (sequence + CTA_Q - 1) / CTA_Q;
    dim3 grid(q_tiles, heads, batch);
    dim3 block(32, CTA_Q / WARP_Q);
    kernel<<<grid, block, smem, stream>>>(
        q, k, v, out, nullptr, q_scale, k_scale, v_scale, nullptr, nullptr,
        0, 0, 0, 0, 0, sequence, sequence, 1,
        heads * sequence * HEAD_DIM, HEAD_DIM, sequence * HEAD_DIM,
        heads * sequence * HEAD_DIM, HEAD_DIM, sequence * HEAD_DIM,
        heads * HEAD_DIM * padded_sequence, HEAD_DIM * padded_sequence,
        padded_sequence,
        sequence * heads * HEAD_DIM, heads * HEAD_DIM, HEAD_DIM, scale,
        q_scale_stride_b, q_scale_stride_h,
        block_lut, valid_block_num, q_tiles);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("Sol Sage exact kernel launch failed: ") +
            cudaGetErrorString(error));
    }
}

}  // namespace

void launch_sage_attn_sparse64_kernel(
    const void* q, const void* k, const void* v, void* out,
    const void* q_scale, const void* k_scale, const void* v_scale,
    const void* block_lut, const void* valid_block_num,
    int batch, int sequence, int heads, int padded_sequence,
    int q_scale_stride_b, int q_scale_stride_h, float scale,
    cudaStream_t stream)
{
    auto q_ptr = const_cast<int8_t*>(static_cast<const int8_t*>(q));
    auto k_ptr = const_cast<int8_t*>(static_cast<const int8_t*>(k));
    auto v_ptr = const_cast<int8_t*>(static_cast<const int8_t*>(v));
    auto qs_ptr = const_cast<float*>(static_cast<const float*>(q_scale));
    auto ks_ptr = const_cast<float*>(static_cast<const float*>(k_scale));
    auto vs_ptr = const_cast<float*>(static_cast<const float*>(v_scale));

    launch_impl(
        q_ptr, k_ptr, v_ptr, static_cast<nv_bfloat16*>(out),
        qs_ptr, ks_ptr, vs_ptr,
        static_cast<const uint16_t*>(block_lut),
        static_cast<const int32_t*>(valid_block_num),
        batch, sequence, heads, padded_sequence,
        q_scale_stride_b, q_scale_stride_h, scale, stream);
}
