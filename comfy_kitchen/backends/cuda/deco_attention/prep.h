// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

namespace comfy_deco {

// Byte offsets in a per-invocation, 256-byte aligned CUDA workspace.
struct PrepLayout {
    size_t q4, q_sf, score_scale, v4_scale, img, split;
    size_t ut, side, chosen_scales, maxima, bytes;
};

PrepLayout prep_layout(int sequence, int heads, int sms);

// BF16 Q/K/V each have D=128 contiguous channels. All six strides are in
// bytes. Rows contains max(64, sequence / 256) sample indices on the device.
// Rotation is the signed, unnormalized 128x128 BF16 Hadamard matrix.
cudaError_t prepare(const void* q, const void* k, const void* v, const int32_t* rows,
                    const void* rotation, void* workspace, int sequence, int heads, int sms,
                    int64_t q_token_bytes, int64_t q_head_bytes,
                    int64_t k_token_bytes, int64_t k_head_bytes,
                    int64_t v_token_bytes, int64_t v_head_bytes, cudaStream_t stream);

// Draw rows in priority order from zero and positive tags, ignoring negative
// tags. order is the fixed permutation of [0, sequence); tag_stride is in
// elements. scratch uses one uint32 per 256 tokens and can alias the Q4
// workspace before prepare() is called on the same stream.
cudaError_t sample_tags(const void* tags, int dtype_bits, bool is_unsigned, int64_t tag_stride,
                        const int32_t* order, int32_t* rows, void* scratch, int sequence,
                        cudaStream_t stream);

} // namespace comfy_deco
