// SPDX-License-Identifier: Apache-2.0
// Shared donor H3 routing kernels: identical FP16 composite sort keys, stable
// phase packing and anchor correction on Ada and Blackwell. The only physical
// lowering difference is the existing compile-time implicit FP16 prefix flag.
#pragma once
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int kThreads = 256;
constexpr int kProbabilityBits = 16;
constexpr int kMaxSegments = 1 << 16;
constexpr int64_t kMaxGridX = 2147483647LL;
constexpr uint8_t kSkip = 0;
constexpr uint8_t kLow = 1;
constexpr uint8_t kMiddle = 2;
constexpr uint8_t kHigh = 3;

struct PhaseCounts {
  int middle;
  int high;
  int low;
};
struct AddPhaseCounts {
  __device__ __forceinline__ PhaseCounts operator()(
      const PhaseCounts& lhs, const PhaseCounts& rhs) const {
    return {lhs.middle + rhs.middle, lhs.high + rhs.high, lhs.low + rhs.low};
  }
};
using RowReduce = cub::BlockReduce<PhaseCounts, kThreads>;
using RowScan = cub::BlockScan<PhaseCounts, kThreads>;
using AnchorReduce = cub::BlockReduce<int, kThreads>;
using AnchorScan = cub::BlockScan<int, kThreads>;
union RowTempStorage {
  typename RowReduce::TempStorage reduce;
  typename RowScan::TempStorage scan;
};
union AnchorTempStorage {
  typename AnchorReduce::TempStorage reduce;
  typename AnchorScan::TempStorage scan;
};
inline int required_unsigned_bits(int value_count) {
  int bits = 0;
  for (int value = value_count - 1; value > 0; value >>= 1) {
    ++bits;
  }
  return bits;
}

__global__ void initialize_composite_sort_kernel(
    const half* __restrict__ probability,
    uint32_t* __restrict__ composite_keys,
    int* __restrict__ initial_ids,
    int total_items,
    int segment_items) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total_items) {
    return;
  }
  const int segment = static_cast<int>(index) / segment_items;
  const int flat_id = static_cast<int>(index) - segment * segment_items;
  const uint16_t probability_bits =
      __half_as_ushort(probability[index]) & uint16_t{0x7fff};
  const uint32_t descending_probability =
      uint32_t{0xffff} - static_cast<uint32_t>(probability_bits);
  composite_keys[index] =
      (static_cast<uint32_t>(segment) << kProbabilityBits) |
      descending_probability;
  initial_ids[index] = flat_id;
}

__global__ void scatter_precision_kernel(
    const int* __restrict__ sorted_ids,
    uint8_t* __restrict__ precision_map,
    int total_items,
    int segment_items,
    int high_end,
    int middle_end,
    int keep) {
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total_items) {
    return;
  }
  const int rank = static_cast<int>(linear) % segment_items;
  const int segment = static_cast<int>(linear) / segment_items;
  uint8_t code = kSkip;
  if (rank < high_end) {
    code = kHigh;
  } else if (rank < middle_end) {
    code = kMiddle;
  } else if (rank < keep) {
    code = kLow;
  }
  precision_map[segment * segment_items + sorted_ids[linear]] = code;
}

__global__ __launch_bounds__(kThreads) void apply_anchor_budget_kernel(
    const int* __restrict__ sorted_ids,
    const bool* __restrict__ anchors,
    const int* __restrict__ anchor_ids,
    uint8_t* __restrict__ precision_map,
    int anchor_count,
    int segment_items,
    int low_begin,
    int keep,
    uint8_t low_code) {
  const int segment = static_cast<int>(blockIdx.x);
  const int segment_offset = segment * segment_items;
  __shared__ AnchorTempStorage temp;
  __shared__ int missing_total;
  __shared__ int evicted;
  int local_missing = 0;
  for (int id_index = threadIdx.x; id_index < anchor_count;
       id_index += blockDim.x) {
    const int id = anchor_ids[id_index];
    const int index = segment_offset + id;
    const bool missing = precision_map[index] == kSkip;
    if (missing) {
      precision_map[index] = low_code;
      ++local_missing;
    }
  }
  const int total = AnchorReduce(temp.reduce).Sum(local_missing);
  __syncthreads();
  if (threadIdx.x == 0) {
    missing_total = total;
    evicted = 0;
  }
  __syncthreads();
  if (missing_total == 0) {
    return;
  }
  for (int tile = 0; tile < keep - low_begin; tile += kThreads) {
    const int rank = keep - 1 - tile - threadIdx.x;
    const int id = rank >= low_begin ? sorted_ids[segment_offset + rank] : 0;
    const int candidate = rank >= low_begin && !anchors[id];
    int prefix = 0;
    int tile_total = 0;
    AnchorScan(temp.scan).ExclusiveSum(candidate, prefix, tile_total);
    __syncthreads();
    if (candidate && evicted + prefix < missing_total) {
      precision_map[segment_offset + id] = kSkip;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      evicted += tile_total;
    }
    __syncthreads();
    if (evicted >= missing_total) {
      break;
    }
  }
}

__global__ __launch_bounds__(kThreads) void pack_active_rows_kernel(
    const uint8_t* __restrict__ precision_map,
    int* __restrict__ packed_ids,
    int* __restrict__ low_counts,
    int* __restrict__ middle_counts,
    int* __restrict__ high_counts,
    int row_count,
    int columns) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= row_count) {
    return;
  }
  const int row_offset = row * columns;
  __shared__ RowTempStorage temp;
  __shared__ int totals[3];
  __shared__ int running[3];
  PhaseCounts local{0, 0, 0};
  for (int column = threadIdx.x; column < columns; column += blockDim.x) {
    const uint8_t code = precision_map[row_offset + column];
    local.middle += code == kMiddle;
    local.high += code == kHigh;
    local.low += code == kLow;
    packed_ids[row_offset + column] = 0;
  }
  const PhaseCounts total = RowReduce(temp.reduce).Reduce(local, AddPhaseCounts{});
  __syncthreads();
  if (threadIdx.x == 0) {
    totals[0] = total.middle;
    totals[1] = total.high;
    totals[2] = total.low;
    running[0] = running[1] = running[2] = 0;
    low_counts[row] = total.low;
    middle_counts[row] = total.middle;
    high_counts[row] = total.high;
  }
  __syncthreads();
  for (int tile = 0; tile < columns; tile += kThreads) {
    const int column = tile + threadIdx.x;
    const uint8_t code = column < columns ? precision_map[row_offset + column] : kSkip;
    const bool valid = column < columns;
    const PhaseCounts is_phase{
        valid && code == kMiddle, valid && code == kHigh, valid && code == kLow};
    PhaseCounts prefix;
    PhaseCounts tile_counts;
    RowScan(temp.scan).ExclusiveScan(
        is_phase, prefix, PhaseCounts{0, 0, 0}, AddPhaseCounts{}, tile_counts);
    __syncthreads();
    if (is_phase.low) {
      packed_ids[row_offset + running[2] + prefix.low] = column;
    } else if (is_phase.middle) {
      packed_ids[row_offset + totals[2] + running[0] + prefix.middle] = column;
    } else if (is_phase.high) {
      packed_ids[row_offset + totals[2] + totals[0] + running[1] + prefix.high] = column;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      running[0] += tile_counts.middle;
      running[1] += tile_counts.high;
      running[2] += tile_counts.low;
    }
    __syncthreads();
  }
}

template <bool ImplicitHighPrefix>
__global__ __launch_bounds__(kThreads) void materialize_physical_route_kernel(
    const int* __restrict__ logical_ids,
    const int* __restrict__ low_counts,
    const int* __restrict__ middle_counts,
    const int* __restrict__ high_counts,
    int* __restrict__ physical_ids,
    int* __restrict__ physical_low_counts,
    int* __restrict__ physical_middle_counts,
    int* __restrict__ physical_high_counts,
    int row_count,
    int logical_columns,
    int physical_columns,
    int factor,
    int prefix_blocks,
    int prefix_phase,
    bool prefix_first,
    bool has_high) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= row_count) {
    return;
  }
  const int low = low_counts[row];
  const int middle = middle_counts[row];
  const int high = has_high ? high_counts[row] : 0;
  const int active = low + middle + high;
  const bool implicit_high_prefix = ImplicitHighPrefix && prefix_phase == 2;
  const int prefix_position = prefix_first
      ? 0 : (prefix_phase == 0 ? 0 : (prefix_phase == 1 ? low * factor : active * factor));
  for (int column = threadIdx.x; column < physical_columns; column += blockDim.x) {
    physical_ids[row * physical_columns + column] = 0;
  }
  __syncthreads();
  for (int position = threadIdx.x; position < active; position += blockDim.x) {
    int destination = position * factor;
    if (!implicit_high_prefix && destination >= prefix_position) {
      destination += prefix_blocks;
    }
    const int first =
        prefix_blocks + logical_ids[row * logical_columns + position] * factor;
#pragma unroll
    for (int stage = 0; stage < 2; ++stage) {
      if (stage < factor) {
        physical_ids[row * physical_columns + destination + stage] = first + stage;
      }
    }
  }
  if (!implicit_high_prefix) {
    for (int prefix = threadIdx.x; prefix < prefix_blocks; prefix += blockDim.x) {
      physical_ids[row * physical_columns + prefix_position + prefix] = prefix;
    }
  }
  if (threadIdx.x == 0) {
    physical_low_counts[row] = low * factor + (prefix_phase == 0 ? prefix_blocks : 0);
    physical_middle_counts[row] = middle * factor + (prefix_phase == 1 ? prefix_blocks : 0);
    if (has_high) {
      physical_high_counts[row] = high * factor + (prefix_phase == 2 ? prefix_blocks : 0);
    }
  }
}
} // namespace
