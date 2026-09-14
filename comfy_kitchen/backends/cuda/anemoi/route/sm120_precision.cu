/* Native global route for the SM120 MiniMax-H3 production path. */
#include "route_precision_launch.h"
#include "common.cuh"
#include <cub/device/device_radix_sort.cuh>
#include <limits>

namespace {
// Preserve the donor's int32 device indexing; check before multiplying or
// narrowing. Leave headroom for the final kThreads increment in int loops.
constexpr int64_t kIntMax = std::numeric_limits<int>::max();
constexpr int64_t kLoopMax = kIntMax - kThreads;

struct RouteGeometry {
  int segment_items;
  int total_items;
  int row_count;
  unsigned int blocks;
  int key_bits;
};

bool route_geometry(int64_t segments, int64_t rows, RouteGeometry* geometry) {
  if (segments <= 0 || segments > kMaxSegments || rows <= 0 ||
      rows > kLoopMax || rows > kIntMax / rows) {
    return false;
  }
  const int64_t segment_items = rows * rows;
  if (segment_items > kLoopMax || segments > kIntMax / segment_items) {
    return false;
  }
  const int64_t total_items = segments * segment_items;
  const int64_t row_count = segments * rows;
  const int64_t blocks = (total_items + kThreads - 1) / kThreads;
  if (blocks > kMaxGridX || row_count > kMaxGridX || row_count > kIntMax ||
      segments > kMaxGridX) {
    return false;
  }
  *geometry = {static_cast<int>(segment_items), static_cast<int>(total_items),
               static_cast<int>(row_count), static_cast<unsigned int>(blocks),
               kProbabilityBits + required_unsigned_bits(static_cast<int>(segments))};
  return true;
}

cudaError_t query_workspace(const RouteGeometry& geometry, size_t* bytes,
                            cudaStream_t stream) {
  cub::DoubleBuffer<uint32_t> keys(nullptr, nullptr);
  cub::DoubleBuffer<int> values(nullptr, nullptr);
  *bytes = 0;
  return cub::DeviceRadixSort::SortPairs(
      nullptr, *bytes, keys, values, geometry.total_items, 0,
      geometry.key_bits, stream);
}
}  // namespace

cudaError_t sm120_h3_route_precision_workspace_size(
    int64_t segments, int64_t rows, size_t* workspace_bytes, cudaStream_t stream) {
  RouteGeometry geometry;
  if (!workspace_bytes || !route_geometry(segments, rows, &geometry)) {
    return cudaErrorInvalidValue;
  }
  return query_workspace(geometry, workspace_bytes, stream);
}

cudaError_t sm120_h3_route_precision(
    const void* probability, const bool* anchors, const int* anchor_ids,
    int64_t segments, int64_t rows, int64_t n16, int64_t n8, int64_t n4,
    int64_t anchor_count, uint8_t* precision_map, uint32_t* input_keys,
    uint32_t* sorted_keys, int* sorted_ids, int* block_ids, int* low_counts,
    int* middle_counts, int* high_counts, void* workspace,
    size_t workspace_bytes, cudaStream_t stream) {
  RouteGeometry geometry;
  if (!route_geometry(segments, rows, &geometry) || !probability ||
      !precision_map || !input_keys || !sorted_keys || !sorted_ids ||
      !block_ids || !low_counts || !middle_counts || !high_counts ||
      n16 < 0 || n16 > kIntMax || n8 < 0 || n8 > kIntMax ||
      n4 < 0 || n4 > kIntMax || anchor_count < 0 || anchor_count > kIntMax) {
    return cudaErrorInvalidValue;
  }
  const int64_t keep_value = n16 + n8 + n4;
  const int64_t lowest_count = n4 ? n4 : (n8 ? n8 : n16);
  if (keep_value <= 0 || keep_value > geometry.segment_items ||
      (anchors && (anchor_count > lowest_count || (anchor_count && !anchor_ids))) ||
      (!anchors && (anchor_ids || anchor_count))) {
    return cudaErrorInvalidValue;
  }
  size_t required_bytes = 0;
  cudaError_t status = query_workspace(geometry, &required_bytes, stream);
  if (status != cudaSuccess) return status;
  if (!workspace || workspace_bytes < required_bytes) return cudaErrorInvalidValue;

  initialize_composite_sort_kernel<<<geometry.blocks, kThreads, 0, stream>>>(
      static_cast<const half*>(probability), input_keys, block_ids,
      geometry.total_items, geometry.segment_items);
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;

  cub::DoubleBuffer<uint32_t> sort_keys(input_keys, sorted_keys);
  cub::DoubleBuffer<int> sort_values(block_ids, sorted_ids);
  status = cub::DeviceRadixSort::SortPairs(
      workspace, workspace_bytes, sort_keys, sort_values, geometry.total_items,
      0, geometry.key_bits, stream);
  if (status != cudaSuccess) return status;

  const int keep = static_cast<int>(keep_value);
  scatter_precision_kernel<<<geometry.blocks, kThreads, 0, stream>>>(
      sort_values.Current(), precision_map, geometry.total_items,
      geometry.segment_items, static_cast<int>(n16), static_cast<int>(n16 + n8), keep);
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  if (anchors && anchor_count) {
    const int low_begin = static_cast<int>(n4 ? n16 + n8 : (n8 ? n16 : 0));
    const uint8_t low_code = n4 ? kLow : (n8 ? kMiddle : kHigh);
    apply_anchor_budget_kernel<<<static_cast<unsigned int>(segments), kThreads, 0, stream>>>(
        sort_values.Current(), anchors, anchor_ids, precision_map,
        static_cast<int>(anchor_count), geometry.segment_items, low_begin, keep, low_code);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
  }
  pack_active_rows_kernel<<<static_cast<unsigned int>(geometry.row_count), kThreads, 0, stream>>>(
      precision_map, block_ids, low_counts, middle_counts, high_counts,
      geometry.row_count, static_cast<int>(rows));
  return cudaGetLastError();
}

cudaError_t sm120_h3_materialize_route(
    const int* logical_ids, const int* low_counts, const int* middle_counts,
    const int* high_counts, int64_t row_count, int64_t logical_columns,
    int64_t query_block_size, int64_t prefix_blocks, int64_t prefix_phase,
    bool prefix_first, bool has_high, int* physical_ids,
    int* physical_low_counts, int* physical_middle_counts,
    int* physical_high_counts, cudaStream_t stream) {
  if (!logical_ids || !low_counts || !middle_counts || !physical_ids ||
      !physical_low_counts || !physical_middle_counts ||
      (has_high && (!high_counts || !physical_high_counts)) ||
      (query_block_size != 64 && query_block_size != 128) ||
      prefix_blocks < 0 || prefix_blocks > kLoopMax ||
      prefix_phase < 0 || prefix_phase > 2 || (!has_high && prefix_phase == 2) ||
      row_count <= 0 || row_count > kMaxGridX || row_count > kIntMax ||
      logical_columns <= 0 || logical_columns > kLoopMax) {
    return cudaErrorInvalidValue;
  }
  const int factor = static_cast<int>(query_block_size / 64);
  const int64_t physical_columns = logical_columns * factor + prefix_blocks;
  // Both kernels use signed int row offsets, not int64 or size_t offsets.
  if (physical_columns > kLoopMax || row_count > kIntMax / logical_columns ||
      row_count > kIntMax / physical_columns) {
    return cudaErrorInvalidValue;
  }
  materialize_physical_route_kernel<false><<<static_cast<unsigned int>(row_count), kThreads, 0, stream>>>(
      logical_ids, low_counts, middle_counts, high_counts, physical_ids,
      physical_low_counts, physical_middle_counts, physical_high_counts,
      static_cast<int>(row_count), static_cast<int>(logical_columns),
      static_cast<int>(physical_columns), factor, static_cast<int>(prefix_blocks),
      static_cast<int>(prefix_phase), prefix_first, has_high);
  return cudaGetLastError();
}
