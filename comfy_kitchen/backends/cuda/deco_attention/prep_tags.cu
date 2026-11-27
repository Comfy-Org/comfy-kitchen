// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "prep.h"
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>

namespace comfy_deco {
namespace {
constexpr int block_size = 256;

// A block of 256 tags cannot overflow either half of its packed count.
template<class Tag>
__device__ unsigned classify(const Tag* tags, int64_t stride, int row, int sequence) {
    if (static_cast<unsigned>(row) >= static_cast<unsigned>(sequence)) return 0;
    const Tag tag = tags[static_cast<int64_t>(row) * stride];
    return (tag == 0) | (static_cast<unsigned>(tag > 0) << 16);
}

template<class Tag>
__global__ void count_tags(const Tag* tags, int64_t stride, const int32_t* order,
                           unsigned* counts, int sequence) {
    using Reduce = cub::BlockReduce<unsigned, block_size>;
    __shared__ typename Reduce::TempStorage tmp;
    const int i = blockIdx.x * block_size + threadIdx.x;
    const int row = i < sequence ? order[i] : -1;
    const unsigned count = Reduce(tmp).Sum(classify(tags, stride, row, sequence));
    if (threadIdx.x == 0) counts[blockIdx.x] = count;
}

template<class Tag>
__global__ void draw_tags(const Tag* tags, int64_t stride, const int32_t* order,
                          const unsigned* counts, int32_t* rows, int sequence, int size) {
    using Scan = cub::BlockScan<unsigned, block_size>;
    __shared__ typename Scan::TempStorage tmp;
    __shared__ int before_video, before_other, take_video, take_other;
    if (threadIdx.x < 32) {
        // Separate the packed block counts before adding across the sequence.
        int videos = 0, others = 0, prefix_video = 0, prefix_other = 0;
        for (int b = threadIdx.x; b < gridDim.x; b += 32) {
            const unsigned value = counts[b];
            const int video = value & 0xffff, other = value >> 16;
            videos += video;
            others += other;
            if (b < blockIdx.x) {
                prefix_video += video;
                prefix_other += other;
            }
        }
        videos = __reduce_add_sync(0xffffffff, videos);
        others = __reduce_add_sync(0xffffffff, others);
        prefix_video = __reduce_add_sync(0xffffffff, prefix_video);
        prefix_other = __reduce_add_sync(0xffffffff, prefix_other);
        if (threadIdx.x == 0) {
            before_video = prefix_video;
            before_other = prefix_other;
            take_video = min(videos, size - min(others, size / 2));
            take_other = min(others, size - take_video);
        }
    }
    __syncthreads();
    const int drawn = take_video + take_other;
    const int i = blockIdx.x * block_size + threadIdx.x;
    if (drawn == 0) {
        if (i < size) rows[i] = 0;
        return;
    }
    if (before_video >= take_video && before_other >= take_other) return;

    const int row = i < sequence ? order[i] : -1;
    const unsigned flag = classify(tags, stride, row, sequence);
    unsigned prefix;
    Scan(tmp).ExclusiveSum(flag, prefix);
    const bool video = flag == 1;
    const int rank = video ? before_video + int(prefix & 0xffff) : before_other + int(prefix >> 16);
    if (flag && rank < (video ? take_video : take_other)) {
        // Repeat a short draw in the same order until all basis slots are filled.
        for (int slot = (video ? 0 : take_video) + rank; slot < size; slot += drawn)
            rows[slot] = row;
    }
}

template<class Tag>
cudaError_t sample(const void* tags, int64_t stride, const int32_t* order, int32_t* rows,
                   unsigned* counts, int sequence, cudaStream_t stream) {
    const auto* typed = static_cast<const Tag*>(tags);
    const int blocks = 1 + (sequence - 1) / block_size;
    const int size = max(64, sequence / 256);
    count_tags<<<blocks, block_size, 0, stream>>>(typed, stride, order, counts, sequence);
    const auto err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    draw_tags<<<blocks, block_size, 0, stream>>>(typed, stride, order, counts, rows, sequence, size);
    return cudaGetLastError();
}
} // namespace

cudaError_t sample_tags(const void* tags, int dtype_bits, bool is_unsigned, int64_t tag_stride,
                        const int32_t* order, int32_t* rows, void* scratch, int sequence,
                        cudaStream_t stream) {
    if (sequence < 64 || tag_stride < 0) return cudaErrorInvalidValue;
    auto* counts = static_cast<unsigned*>(scratch);
    if (dtype_bits == 8 && is_unsigned)
        return sample<uint8_t>(tags, tag_stride, order, rows, counts, sequence, stream);
    if (is_unsigned) return cudaErrorInvalidValue;
    switch (dtype_bits) {
    case 8:  return sample<int8_t>(tags, tag_stride, order, rows, counts, sequence, stream);
    case 16: return sample<int16_t>(tags, tag_stride, order, rows, counts, sequence, stream);
    case 32: return sample<int32_t>(tags, tag_stride, order, rows, counts, sequence, stream);
    case 64: return sample<int64_t>(tags, tag_stride, order, rows, counts, sequence, stream);
    default: return cudaErrorInvalidValue;
    }
}
} // namespace comfy_deco
