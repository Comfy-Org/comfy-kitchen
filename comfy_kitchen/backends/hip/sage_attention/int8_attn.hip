// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Mask preparation for the INT8 attention kernels, and their extern "C" entry points.
// The kernels themselves are in int8_attn.h, instantiated in their own files.
#include "../gemm_simt.h"
#include "int8_attn.h"

namespace comfy::hip_backend::sage::int8_attn {
namespace {

// Preprocess once per key row. Biases are in the base-two softmax domain;
// nonfinite inputs and padding become -inf. The descriptor is -inf for an
// empty tile and zero otherwise, so attention can skip empty tiles uniformly.
__global__ void prepare_key_mask_kernel(
    const void* mask, float* packed, int kv_len, int64_t stride_b, int64_t stride_h,
    int64_t stride_k, int dtype_code, int packed_width) {
    constexpr int tile_k = 64;
    __shared__ int wave_valid[2];
    const int lane = threadIdx.x;
    const int key = blockIdx.x * tile_k + lane;
    const int tiles = gridDim.x;
    const int64_t row = static_cast<int64_t>(blockIdx.z) * stride_b +
                        static_cast<int64_t>(blockIdx.y) * stride_h;
    float bias = 0.0f;
    const bool keep = key < kv_len && mask_keep(mask, row + key * stride_k, dtype_code, bias);
    bias = keep ? bias * kLog2e : __uint_as_float(0xff800000u);
    float* out = packed + (static_cast<int64_t>(blockIdx.z) * gridDim.y + blockIdx.y) * packed_width;
    out[key] = bias;
    const int any_valid = __any(keep);
    if ((lane & 31) == 0) wave_valid[lane / 32] = any_valid;
    __syncthreads();
    if (lane == 0) {
        out[tiles * tile_k + blockIdx.x] =
            (wave_valid[0] | wave_valid[1]) ? 0.0f : __uint_as_float(0xff800000u);
    }
}

__global__ void prepare_dense_mask_kernel(
    const void* mask, float* packed, int heads, int qo_len, int kv_len,
    int64_t stride_b, int64_t stride_h, int64_t stride_q, int64_t stride_k, int dtype_code) {
    const int head = blockIdx.z % heads;
    const int batch = blockIdx.z / heads;
    const int64_t out_tile =
        ((static_cast<int64_t>(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * 1024;
    const int first = threadIdx.x * 4;
    const int first_query = blockIdx.y * 16 + (first / 8) % 16;
    const int first_key = blockIdx.x * 64 + (first / 256) * 16 + ((first / 128) % 2) * 8 + first % 8;
    if (dtype_code == 0 && stride_k == 1 && first_query < qo_len && first_key + 4 <= kv_len) {
        uint4 bits;
        const int64_t offset = batch * stride_b + head * stride_h + first_query * stride_q + first_key;
        __builtin_memcpy(&bits, static_cast<const uint32_t*>(mask) + offset, sizeof(bits));
        const uint32_t raw[4] = {bits.x, bits.y, bits.z, bits.w};
        float values[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            values[e] = (raw[e] & 0x7f800000u) == 0x7f800000u
                            ? __uint_as_float(0xff800000u) : __uint_as_float(raw[e]) * kLog2e;
        }
        *reinterpret_cast<float4*>(packed + out_tile + first) =
            float4{values[0], values[1], values[2], values[3]};
        return;
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int index = threadIdx.x * 4 + e;
        const int query = blockIdx.y * 16 + (index / 8) % 16;
        const int key = blockIdx.x * 64 + (index / 256) * 16 + ((index / 128) % 2) * 8 + index % 8;
        float bias = 0.0f;
        const bool keep = query < qo_len && key < kv_len &&
                          mask_keep(mask, batch * stride_b + head * stride_h +
                                    query * stride_q + key * stride_k, dtype_code, bias);
        packed[out_tile + index] = keep ? bias * kLog2e : __uint_as_float(0xff800000u);
    }
}

// Native 16-bit inputs retain their exact finite bits. Biases stay in natural
// units, avoiding overflow or an extra rounding from pre-multiplying by log2(e).
template <typename PackedT>
__global__ void prepare_dense_mask_16_kernel(
    const void* mask, PackedT* packed, int heads, int qo_len, int kv_len,
    int64_t stride_b, int64_t stride_h, int64_t stride_q, int64_t stride_k, int dtype_code) {
    const int head = blockIdx.z % heads;
    const int batch = blockIdx.z / heads;
    const int64_t out_tile =
        ((static_cast<int64_t>(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * 1024;
    const int first = threadIdx.x * 4;
    const int first_query = blockIdx.y * 16 + (first / 8) % 16;
    const int first_key = blockIdx.x * 64 + (first / 256) * 16 + ((first / 128) % 2) * 8 + first % 8;
    constexpr bool bf16_output = std::is_same_v<PackedT, __bf16>;
    if (dtype_code == (bf16_output ? 2 : 1)) {
        constexpr uint16_t exponent = bf16_output ? 0x7f80u : 0x7c00u;
        uint16_t values[4];
        const int64_t offset = batch * stride_b + head * stride_h + first_query * stride_q + first_key * stride_k;
        if (stride_k == 1 && first_query < qo_len && first_key + 4 <= kv_len) {
            __builtin_memcpy(values, static_cast<const uint16_t*>(mask) + offset, sizeof(values));
        } else {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                values[e] = first_query < qo_len && first_key + e < kv_len
                    ? static_cast<const uint16_t*>(mask)[offset + e * stride_k]
                    : exponent | 0x8000u;
            }
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            if ((values[e] & exponent) == exponent) values[e] = exponent | 0x8000u;
        }
        uint2 bits;
        __builtin_memcpy(&bits, values, sizeof(bits));
        *reinterpret_cast<uint2*>(packed + out_tile + first) = bits;
        return;
    }
    if (dtype_code == 0 && stride_k == 1 && first_query < qo_len && first_key + 4 <= kv_len) {
        uint4 bits;
        const int64_t offset = batch * stride_b + head * stride_h + first_query * stride_q + first_key;
        __builtin_memcpy(&bits, static_cast<const uint32_t*>(mask) + offset, sizeof(bits));
        const uint32_t raw[4] = {bits.x, bits.y, bits.z, bits.w};
        PackedT values[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            values[e] = (raw[e] & 0x7f800000u) == 0x7f800000u
                            ? __uint_as_float(0xff800000u) : __uint_as_float(raw[e]);
        }
        uint2 out_bits;
        __builtin_memcpy(&out_bits, values, sizeof(out_bits));
        *reinterpret_cast<uint2*>(packed + out_tile + first) = out_bits;
        return;
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int index = threadIdx.x * 4 + e;
        const int query = blockIdx.y * 16 + (index / 8) % 16;
        const int key = blockIdx.x * 64 + (index / 256) * 16 + ((index / 128) % 2) * 8 + index % 8;
        float bias = 0.0f;
        const bool keep = query < qo_len && key < kv_len &&
                          mask_keep(mask, batch * stride_b + head * stride_h +
                                    query * stride_q + key * stride_k, dtype_code, bias);
        packed[out_tile + index] = keep ? bias : __uint_as_float(0xff800000u);
    }
}

// One word per gfx12 accumulator lane holds its 32 Boolean keys across a
// 16-query, 64-key tile. Packing once also removes repeated byte-mask reads.
template <int TilesPerBlock>
__global__ void prepare_dense_bool_bits_kernel(
    const void* mask, uint32_t* packed, int heads, int qo_len, int kv_len,
    int64_t stride_b, int64_t stride_h, int64_t stride_q, int64_t stride_k) {
    __shared__ uint32_t row_words[32 * TilesPerBlock];
    const int tid = threadIdx.x;
    const int head = blockIdx.z % heads;
    const int batch = blockIdx.z / heads;
    const int tiles = (kv_len + 63) / 64;
#pragma unroll
    for (int base = 0; base < 1024 * TilesPerBlock; base += 256) {
        const int index = base + tid;
        const int query = blockIdx.y * 16 + index / (64 * TilesPerBlock);
        const int key = blockIdx.x * (64 * TilesPerBlock) + index % (64 * TilesPerBlock);
        const int64_t offset = batch * stride_b + head * stride_h + query * stride_q + key * stride_k;
        const bool keep = query < qo_len && key < kv_len && static_cast<const uint8_t*>(mask)[offset] != 0;
        const uint32_t votes = static_cast<uint32_t>(__ballot(keep));
        if ((tid & 31) == 0) row_words[index / 32] = votes;
    }
    __syncthreads();
    for (int out_index = tid; out_index < 32 * TilesPerBlock; out_index += 256) {
        const int sub = out_index / 32;
        const int row = out_index % 16;
        const uint32_t first = row_words[row * (2 * TilesPerBlock) + sub * 2];
        const uint32_t second = row_words[row * (2 * TilesPerBlock) + sub * 2 + 1];
        const uint32_t bits = __builtin_amdgcn_perm(second, first, (out_index % 32) < 16 ? 0x06040200u : 0x07050301u);
        const int key_tile = blockIdx.x * TilesPerBlock + sub;
        if (key_tile < tiles) {
            const int64_t tile = (static_cast<int64_t>(blockIdx.z) * gridDim.y + blockIdx.y) * tiles + key_tile;
            packed[tile * 32 + out_index % 32] = bits;
        }
    }
}

}  // namespace
}  // namespace comfy::hip_backend::sage::int8_attn

using namespace comfy::hip_backend::sage::int8_attn;

extern "C" void launch_sage_prepare_key_mask(
    const void* mask, float* packed, int batch, int heads, int kv_len, int64_t stride_b,
    int64_t stride_h, int64_t stride_k, int dtype_code, int packed_width, hipStream_t stream) {
    const dim3 grid(div_ceil_host(kv_len, 64), heads, batch);
    prepare_key_mask_kernel<<<grid, 64, 0, stream>>>(
        mask, packed, kv_len, stride_b, stride_h, stride_k, dtype_code, packed_width);
}

extern "C" void launch_sage_prepare_dense_mask(
    const void* mask, void* packed, int batch, int heads, int qo_len, int kv_len,
    int64_t stride_b, int64_t stride_h, int64_t stride_q, int64_t stride_k,
    int dtype_code, int output_code, hipStream_t stream) {
    const dim3 grid(div_ceil_host(kv_len, 64), div_ceil_host(qo_len, 16), batch * heads);
    if (output_code == 8) {
        prepare_dense_mask_16_kernel<<<grid, 256, 0, stream>>>(
            mask, static_cast<__bf16*>(packed), heads, qo_len, kv_len,
            stride_b, stride_h, stride_q, stride_k, dtype_code);
    } else if (output_code == 9) {
        prepare_dense_mask_16_kernel<<<grid, 256, 0, stream>>>(
            mask, static_cast<__half*>(packed), heads, qo_len, kv_len,
            stride_b, stride_h, stride_q, stride_k, dtype_code);
    } else if (output_code == 7) {
        if (kv_len <= 2048) {
            prepare_dense_bool_bits_kernel<1><<<grid, 256, 0, stream>>>(
                mask, static_cast<uint32_t*>(packed), heads, qo_len, kv_len,
                stride_b, stride_h, stride_q, stride_k);
        } else {
            // Amortize ballot/CTA overhead while preserving the same tile layout.
            const dim3 bits_grid(div_ceil_host(kv_len, 1024), div_ceil_host(qo_len, 16), batch * heads);
            prepare_dense_bool_bits_kernel<16><<<bits_grid, 256, 0, stream>>>(
                mask, static_cast<uint32_t*>(packed), heads, qo_len, kv_len,
                stride_b, stride_h, stride_q, stride_k);
        }
    } else {
        prepare_dense_mask_kernel<<<grid, 256, 0, stream>>>(
            mask, static_cast<float*>(packed), heads, qo_len, kv_len,
            stride_b, stride_h, stride_q, stride_k, dtype_code);
    }
}

extern "C" void launch_sage_int8_attn(
    const void* q, const void* k, const void* v, void* o, const void* q_scale,
    const void* k_scale, const void* v_scale, const void* mask, int64_t mask_stride_b,
    int64_t mask_stride_h, int64_t mask_stride_q, int64_t mask_stride_k, int mask_dtype_code,
    int cta_k, int batch, int qo_len, int kv_len, int qo_len_padded, int num_qo_heads,
    int num_kv_heads, int head_dim, int k_groups_per_head, int64_t q_stride_b, int64_t q_stride_h,
    int64_t k_stride_b, int64_t k_stride_h, int64_t v_stride_b, int64_t v_stride_h,
    int64_t v_stride_d, int64_t o_stride_b, int64_t o_stride_h, int64_t o_stride_n,
    float sm_scale, int output_dtype_code, hipStream_t stream) {

    // Q, K and V are read as 16-byte fragments and O is stored 8 elements at a
    // time; every offset the kernel forms is a multiple of those widths, so the
    // base addresses are the whole of the alignment requirement. The V quantizer
    // launcher checks the same thing.
    if (reinterpret_cast<uintptr_t>(q) % 16 != 0 || reinterpret_cast<uintptr_t>(k) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(v) % 16 != 0 || reinterpret_cast<uintptr_t>(o) % 16 != 0) {
        throw std::runtime_error(
            "sage int8 attention: q, k, v and o base pointers must be 16-byte aligned");
    }
    (void)cta_k;  // always 64-key tiles, see launch_int8_attn

    const Int8AttnLaunch a{
        .q = static_cast<const int8_t*>(q), .k = static_cast<const int8_t*>(k),
        .v = static_cast<const int8_t*>(v), .o = o,
        .q_scale = static_cast<const float*>(q_scale), .k_scale = static_cast<const float*>(k_scale),
        .v_scale = static_cast<const float*>(v_scale), .mask = mask,
        .mask_stride_b = mask_stride_b, .mask_stride_h = mask_stride_h,
        .mask_stride_q = mask_stride_q, .mask_stride_k = mask_stride_k,
        .mask_dtype_code = mask_dtype_code, .batch = batch, .qo_len = qo_len, .kv_len = kv_len,
        .qo_len_padded = qo_len_padded, .num_qo_heads = num_qo_heads,
        .k_groups_per_head = k_groups_per_head, .num_kv_groups = num_qo_heads / num_kv_heads,
        .q_stride_b = q_stride_b, .q_stride_h = q_stride_h, .k_stride_b = k_stride_b,
        .k_stride_h = k_stride_h, .v_stride_b = v_stride_b, .v_stride_h = v_stride_h,
        .v_stride_d = v_stride_d, .o_stride_b = o_stride_b, .o_stride_h = o_stride_h,
        .o_stride_n = o_stride_n, .sm_scale = sm_scale, .stream = stream};

    // The fp16 FMA kernel does not support masks, D256, or a negative scale (it
    // takes the tile maximum before scaling).
    if (mask == nullptr && (head_dim == 64 || head_dim == 128) && sm_scale >= 0.0f &&
        kv_len > 0 && comfy::hip_backend::simt_gemm_target() == comfy::hip_backend::SimtTarget::kGfx1010) {
        launch_int8_attn_fp16(a, head_dim, output_dtype_code, nullptr, 0, 128);
        return;
    }

    const bool half_out = output_dtype_code == 1;
    switch (head_dim) {
        case 64: half_out ? launch_int8_attn<64, __half>(a) : launch_int8_attn<64, __bf16>(a); break;
        case 128: half_out ? launch_int8_attn<128, __half>(a) : launch_int8_attn<128, __bf16>(a); break;
        case 256: half_out ? launch_int8_attn<256, __half>(a) : launch_int8_attn<256, __bf16>(a); break;
        default:
            throw std::runtime_error("sage int8 attention: unsupported head_dim " +
                                     std::to_string(head_dim));
    }
}

// Block-sparse attention over the packed sage layouts: query block b of head h attends
// to the 64-key tiles lut[batch, h, b, :topk]. gfx1010 only, where the fp16 FMA kernel
// serves it; block_q (64 or 128) is the query block the lut was built for.
extern "C" void launch_sage_int8_block_sparse_attn(
    const void* q, const void* k, const void* v, void* o, const void* q_scale,
    const void* k_scale, const void* v_scale, const void* lut, int topk, int block_q, int batch,
    int qo_len, int kv_len, int qo_len_padded, int num_qo_heads, int num_kv_heads, int head_dim,
    int k_groups_per_head, int64_t q_stride_b, int64_t q_stride_h, int64_t k_stride_b,
    int64_t k_stride_h, int64_t v_stride_b, int64_t v_stride_h, int64_t v_stride_d,
    int64_t o_stride_b, int64_t o_stride_h, int64_t o_stride_n, float sm_scale,
    int output_dtype_code, hipStream_t stream) {

    if (comfy::hip_backend::simt_gemm_target() != comfy::hip_backend::SimtTarget::kGfx1010) {
        throw std::runtime_error("sage block-sparse attention: requires gfx1010");
    }
    if ((head_dim != 64 && head_dim != 128) || (block_q != 64 && block_q != 128) ||
        sm_scale < 0.0f || kv_len <= 0) {
        throw std::runtime_error(
            "sage block-sparse attention: needs head_dim 64 or 128, block_q 64 or 128, a "
            "nonnegative scale and keys");
    }
    const Int8AttnLaunch a{
        .q = static_cast<const int8_t*>(q), .k = static_cast<const int8_t*>(k),
        .v = static_cast<const int8_t*>(v), .o = o,
        .q_scale = static_cast<const float*>(q_scale), .k_scale = static_cast<const float*>(k_scale),
        .v_scale = static_cast<const float*>(v_scale), .mask = nullptr,
        .mask_stride_b = 0, .mask_stride_h = 0, .mask_stride_q = 0, .mask_stride_k = 0,
        .mask_dtype_code = 0, .batch = batch, .qo_len = qo_len, .kv_len = kv_len,
        .qo_len_padded = qo_len_padded, .num_qo_heads = num_qo_heads,
        .k_groups_per_head = k_groups_per_head, .num_kv_groups = num_qo_heads / num_kv_heads,
        .q_stride_b = q_stride_b, .q_stride_h = q_stride_h, .k_stride_b = k_stride_b,
        .k_stride_h = k_stride_h, .v_stride_b = v_stride_b, .v_stride_h = v_stride_h,
        .v_stride_d = v_stride_d, .o_stride_b = o_stride_b, .o_stride_h = o_stride_h,
        .o_stride_n = o_stride_n, .sm_scale = sm_scale, .stream = stream};
    launch_int8_attn_fp16(a, head_dim, output_dtype_code, static_cast<const int*>(lut), topk,
                          block_q);
}
