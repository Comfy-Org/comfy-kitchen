/* Native global route for the SM89 H3 production path. */
#include "common.cuh"
#include <cub/device/device_radix_sort.cuh>
#include <limits>
#include <tuple>
#include "assembly_route_draft_native.cuh"
namespace anemoi_native {
int64_t sm89_route_workspace_bytes(int64_t batch, int64_t heads, int64_t rows,
                                   int device, cudaStream_t stream);
}

namespace {
inline int64_t checked_positive_product(
    int64_t lhs, int64_t rhs, const char* description) {
  ANEMOI_CHECK(lhs > 0 && rhs > 0, description, " factors must be positive");
  ANEMOI_CHECK(
      lhs <= std::numeric_limits<int64_t>::max() / rhs,
      description, " exceeds int64 range");
  return lhs * rhs;
}
inline int require_nonnegative_int32(int64_t value, const char* name) {
  ANEMOI_CHECK(value >= 0, name, " must be nonnegative");
  ANEMOI_CHECK(
      value <= std::numeric_limits<int>::max(),
      name, " exceeds int32 range");
  return static_cast<int>(value);
}

std::tuple<anemoi_native::Tensor, anemoi_native::Tensor, anemoi_native::Tensor, anemoi_native::Tensor>
route_precision_impl(
    anemoi_native::Tensor probability,
    int64_t n16_value,
    int64_t n8_value,
    int64_t n4_value,
    std::optional<anemoi_native::Tensor> anchors,
    std::optional<anemoi_native::Tensor> anchor_ids,
    int64_t anchor_count_value) {
  ANEMOI_CHECK(
      probability.is_cuda() &&
          probability.is_contiguous() &&
          probability.scalar_type() == anemoi_native::ScalarType::Half,
      "probability must be contiguous CUDA FP16");
  ANEMOI_CHECK(
      probability.dim() == 4 && probability.size(0) > 0 &&
          probability.size(1) > 0 && probability.size(2) > 0 &&
          probability.size(2) == probability.size(3),
      "probability must be square [B,H,R,R]");
  const int n16 = require_nonnegative_int32(n16_value, "n16");
  const int n8 = require_nonnegative_int32(n8_value, "n8");
  const int n4 = require_nonnegative_int32(n4_value, "n4");
  const int anchor_count =
      require_nonnegative_int32(anchor_count_value, "anchor_count");
  ANEMOI_CHECK(
      n16_value <= std::numeric_limits<int64_t>::max() - n8_value &&
          n16_value + n8_value <=
              std::numeric_limits<int64_t>::max() - n4_value,
      "n16+n8+n4 exceeds int64 range");
  const int64_t keep_value = n16_value + n8_value + n4_value;
  ANEMOI_CHECK(keep_value > 0, "retained count must be positive");

  const int64_t rows = probability.size(2);
  const int lowest_count = n4 ? n4 : (n8 ? n8 : n16);
  if (anchors.has_value()) {
    ANEMOI_CHECK(
        anchors->is_cuda() && anchors->is_contiguous() &&
            anchors->scalar_type() == anemoi_native::ScalarType::Bool &&
            anchors->device() == probability.device() && anchors->dim() == 2 &&
            anchors->size(0) == rows && anchors->size(1) == rows,
        "anchors must be contiguous CUDA bool [R,R] on the probability device");
    ANEMOI_CHECK(
        anchor_ids.has_value() && anchor_ids->is_cuda() && anchor_ids->is_contiguous() &&
            anchor_ids->scalar_type() == anemoi_native::ScalarType::Int &&
            anchor_ids->device() == probability.device() &&
            anchor_ids->dim() == 1 && anchor_ids->numel() == anchor_count,
        "anchor_ids must be contiguous CUDA int32 [anchor_count]");
    ANEMOI_CHECK(
        anchor_count <= lowest_count,
        "anchor_count exceeds the configured lowest-precision budget");
  } else {
    ANEMOI_CHECK(
        !anchor_ids.has_value() && anchor_count == 0,
        "anchor_ids must be absent and anchor_count zero when anchors are disabled");
  }
  const int64_t segment_items_value =
      checked_positive_product(rows, rows, "route items per head");
  const int64_t segments_value = checked_positive_product(
      probability.size(0), probability.size(1), "route segments");
  const int64_t total_items_value = checked_positive_product(
      segments_value, segment_items_value, "total route items");
  const int64_t row_count_value =
      checked_positive_product(segments_value, rows, "route rows");
  ANEMOI_CHECK(keep_value <= segment_items_value, "retained count exceeds R*R");
  ANEMOI_CHECK(
      segment_items_value <= std::numeric_limits<int>::max() &&
          total_items_value <= std::numeric_limits<int>::max() &&
          row_count_value <= std::numeric_limits<int>::max(),
      "route geometry exceeds CUB int32 range");
  ANEMOI_CHECK(segments_value <= kMaxSegments, "B*H exceeds global route key");

  assembly_route_draft_native::DeviceGuard device_guard(probability.device());
  const auto device_properties = assembly_route_draft_native::device_properties(probability.device());
  const cudaDeviceProp* properties = &device_properties;
  ANEMOI_CHECK(
      anemoi_native::ada_serves_device(properties),
      "native H3 route serves SM89 to SM119");

  const int segment_items = static_cast<int>(segment_items_value);
  const int segments = static_cast<int>(segments_value);
  const int total_items = static_cast<int>(total_items_value);
  const int row_count = static_cast<int>(row_count_value);
  const int keep = static_cast<int>(keep_value);
  auto int_options = probability.options().dtype(anemoi_native::ScalarType::Int);
  auto byte_options = probability.options().dtype(anemoi_native::ScalarType::Byte);
  auto precision_map = anemoi_native::empty(assembly_route_draft_native::shape(probability), byte_options);
  auto block_ids = anemoi_native::empty(assembly_route_draft_native::shape(probability), int_options);
  auto sorted_ids = anemoi_native::empty_like(block_ids);
  auto input_keys = anemoi_native::empty(assembly_route_draft_native::shape(probability), int_options);
  auto sorted_keys = anemoi_native::empty_like(input_keys);
  auto low_counts = anemoi_native::empty(
      {probability.size(0), probability.size(1), rows}, int_options);
  auto middle_counts = anemoi_native::empty_like(low_counts);
  auto high_counts = anemoi_native::empty_like(low_counts);
  const cudaStream_t stream =
      anemoi_native::cuda::getCurrentCUDAStream();

  const int64_t blocks = (total_items_value + kThreads - 1) / kThreads;
  ANEMOI_CHECK(blocks <= kMaxGridX, "route grid.x exceeds CUDA limit");
  const int key_bits = kProbabilityBits + required_unsigned_bits(segments);
  // Same authoritative query used by the native workspace recipe; claim the exact caller
  // workspace before enqueueing any route kernel. No capacity heuristic.
  const int64_t required_workspace = anemoi_native::sm89_route_workspace_bytes(
      probability.size(0), probability.size(1), rows, probability.device(), stream);
  auto workspace = anemoi_native::empty({required_workspace}, byte_options);
  size_t workspace_bytes = static_cast<size_t>(required_workspace);
  initialize_composite_sort_kernel<<<
      static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(
      reinterpret_cast<const half*>(probability.data_ptr<half>()),
      reinterpret_cast<uint32_t*>(input_keys.data_ptr<int>()),
      block_ids.data_ptr<int>(),
      total_items,
      segment_items);
  ANEMOI_CUDA_KERNEL_LAUNCH_CHECK();
  cub::DoubleBuffer<uint32_t> sort_keys(
      reinterpret_cast<uint32_t*>(input_keys.data_ptr<int>()),
      reinterpret_cast<uint32_t*>(sorted_keys.data_ptr<int>()));
  cub::DoubleBuffer<int> sort_values(
      block_ids.data_ptr<int>(), sorted_ids.data_ptr<int>());
  assembly_route_draft_native::cuda_check(cub::DeviceRadixSort::SortPairs(
      workspace.data_ptr<uint8_t>(),
      workspace_bytes,
      sort_keys,
      sort_values,
      total_items,
      0,
      key_bits,
      stream));

  scatter_precision_kernel<<<
      static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(
      sort_values.Current(),
      precision_map.data_ptr<uint8_t>(),
      total_items,
      segment_items,
      n16,
      n16 + n8,
      keep);
  ANEMOI_CUDA_KERNEL_LAUNCH_CHECK();
  if (anchors.has_value()) {
    const int low_begin = n4 ? n16 + n8 : (n8 ? n16 : 0);
    const uint8_t low_code = n4 ? kLow : (n8 ? kMiddle : kHigh);
    apply_anchor_budget_kernel<<<
        static_cast<unsigned int>(segments), kThreads, 0, stream>>>(
        sort_values.Current(),
        anchors->data_ptr<bool>(),
        anchor_ids->data_ptr<int>(),
        precision_map.data_ptr<uint8_t>(),
        anchor_count,
        segment_items,
        low_begin,
        keep,
        low_code);
    ANEMOI_CUDA_KERNEL_LAUNCH_CHECK();
  }
  ANEMOI_CHECK(row_count_value <= kMaxGridX, "row-pack grid.x exceeds CUDA limit");
  pack_active_rows_kernel<<<
      static_cast<unsigned int>(row_count_value), kThreads, 0, stream>>>(
      precision_map.data_ptr<uint8_t>(),
      block_ids.data_ptr<int>(),
      low_counts.data_ptr<int>(),
      middle_counts.data_ptr<int>(),
      high_counts.data_ptr<int>(),
      row_count,
      static_cast<int>(rows));
  ANEMOI_CUDA_KERNEL_LAUNCH_CHECK();
  return {block_ids, low_counts, middle_counts, high_counts};
}

template <bool ImplicitHighPrefix>
std::tuple<anemoi_native::Tensor, anemoi_native::Tensor, anemoi_native::Tensor, anemoi_native::Tensor>
materialize_route_impl(
    anemoi_native::Tensor logical_ids,
    anemoi_native::Tensor low_counts,
    anemoi_native::Tensor middle_counts,
    anemoi_native::Tensor high_counts,
    int64_t query_block_size_value,
    int64_t prefix_blocks_value,
    int64_t prefix_phase_value,
    bool prefix_first,
    bool has_high) {
  ANEMOI_CHECK(
      logical_ids.is_cuda() &&
          logical_ids.is_contiguous() &&
          logical_ids.scalar_type() == anemoi_native::ScalarType::Int &&
          logical_ids.dim() == 4,
      "logical_ids must be contiguous CUDA int32 [B,H,R,C]");
  const auto check_counts = [&](const anemoi_native::Tensor& counts, const char* name) {
    ANEMOI_CHECK(
        counts.is_cuda() && counts.is_contiguous() &&
            counts.scalar_type() == anemoi_native::ScalarType::Int &&
            counts.device() == logical_ids.device() && counts.dim() == 3 &&
            assembly_route_draft_native::shape(counts) == std::vector<int64_t>({logical_ids.size(0), logical_ids.size(1), logical_ids.size(2)}),
        name, " must be contiguous CUDA int32 [B,H,R]");
  };
  check_counts(low_counts, "low_counts");
  check_counts(middle_counts, "middle_counts");
  check_counts(high_counts, "high_counts");
  ANEMOI_CHECK(
      query_block_size_value == 64 || query_block_size_value == 128,
      "query_block_size must be 64 or 128");
  const int prefix_blocks =
      require_nonnegative_int32(prefix_blocks_value, "prefix_blocks");
  const int prefix_phase =
      require_nonnegative_int32(prefix_phase_value, "prefix_phase");
  ANEMOI_CHECK(prefix_phase < 3, "prefix_phase must be 0, 1, or 2");
  ANEMOI_CHECK(
      has_high || prefix_phase != 2,
      "FP16 prefix requires active high counts");

  const int factor = static_cast<int>(query_block_size_value / 64);
  ANEMOI_CHECK(logical_ids.size(3) > 0 &&
                   logical_ids.size(3) <= std::numeric_limits<int>::max(),
               "logical route columns must fit positive int32");
  const int logical_columns = static_cast<int>(logical_ids.size(3));
  const int64_t physical_columns_value =
      checked_positive_product(logical_columns, factor, "physical route columns") +
      prefix_blocks;
  ANEMOI_CHECK(
      physical_columns_value <= std::numeric_limits<int>::max(),
      "physical route columns exceed int32 range");
  const int64_t row_count_value = logical_ids.numel() / logical_columns;
  ANEMOI_CHECK(
      row_count_value > 0 && row_count_value <= kMaxGridX &&
          row_count_value <= std::numeric_limits<int>::max(),
      "physical route row count exceeds CUDA range");

  assembly_route_draft_native::DeviceGuard device_guard(logical_ids.device());
  auto physical_sizes = assembly_route_draft_native::shape(logical_ids);
  physical_sizes.back() = physical_columns_value;
  auto physical_ids = anemoi_native::empty(physical_sizes, logical_ids.options());
  auto physical_low_counts = anemoi_native::empty_like(low_counts);
  auto physical_middle_counts = anemoi_native::empty_like(middle_counts);
  auto physical_high_counts = has_high
      ? anemoi_native::empty_like(high_counts)
      : anemoi_native::empty({0}, high_counts.options());
  const cudaStream_t stream =
      anemoi_native::cuda::getCurrentCUDAStream();
  materialize_physical_route_kernel<ImplicitHighPrefix><<<
      static_cast<unsigned int>(row_count_value), kThreads, 0, stream>>>(
      logical_ids.data_ptr<int>(),
      low_counts.data_ptr<int>(),
      middle_counts.data_ptr<int>(),
      high_counts.data_ptr<int>(),
      physical_ids.data_ptr<int>(),
      physical_low_counts.data_ptr<int>(),
      physical_middle_counts.data_ptr<int>(),
      has_high ? physical_high_counts.data_ptr<int>() : nullptr,
      static_cast<int>(row_count_value),
      logical_columns,
      static_cast<int>(physical_columns_value),
      factor,
      prefix_blocks,
      prefix_phase,
      prefix_first,
      has_high);
  ANEMOI_CUDA_KERNEL_LAUNCH_CHECK();
  return {
      physical_ids,
      physical_low_counts,
      physical_middle_counts,
      physical_high_counts};
}
}  // namespace

std::tuple<anemoi_native::Tensor, anemoi_native::Tensor, anemoi_native::Tensor, anemoi_native::Tensor>
sm89_h3_route_precision(
    anemoi_native::Tensor probability, int64_t n16, int64_t n8, int64_t n4,
    std::optional<anemoi_native::Tensor> anchors,
    std::optional<anemoi_native::Tensor> anchor_ids, int64_t anchor_count) {
  return route_precision_impl(probability, n16, n8, n4, anchors, anchor_ids,
                              anchor_count);
}

std::tuple<anemoi_native::Tensor, anemoi_native::Tensor, anemoi_native::Tensor, anemoi_native::Tensor>
sm89_h3_materialize_route(
    anemoi_native::Tensor logical_ids, anemoi_native::Tensor low_counts,
    anemoi_native::Tensor middle_counts, anemoi_native::Tensor high_counts,
    int64_t query_block_size, int64_t prefix_blocks, int64_t prefix_phase,
    bool prefix_first, bool has_high) {
  assembly_route_draft_native::DeviceGuard device_guard(logical_ids.device());
  const auto properties = assembly_route_draft_native::device_properties(logical_ids.device());
  ANEMOI_CHECK(anemoi_native::ada_serves_device(&properties),
               "native H3 materialization serves SM89 to SM119");
  return materialize_route_impl<true>(logical_ids, low_counts, middle_counts,
      high_counts, query_block_size, prefix_blocks, prefix_phase, prefix_first, has_high);
}

int64_t anemoi_native::sm89_route_workspace_bytes(
    int64_t batch, int64_t heads, int64_t rows, int device, cudaStream_t stream) {
  // CUB's authoritative null-storage query performs no device computation,
  // allocation, copies, or input-pointer dereferences. Device metadata may be
  // queried by CUB to select its dispatch policy.
  const int64_t segments = checked_positive_product(batch, heads, "route segments");
  const int64_t segment_items = checked_positive_product(rows, rows, "route items per head");
  const int64_t items = checked_positive_product(segments, segment_items, "total route items");
  ANEMOI_CHECK(segments <= kMaxSegments && items <= std::numeric_limits<int>::max(),
               "route geometry exceeds CUB int32 range");
  anemoi_native::cuda::CUDAGuard guard(device);
  anemoi_native::require_ada_gpu();
  cub::DoubleBuffer<uint32_t> keys(nullptr, nullptr);
  cub::DoubleBuffer<int> values(nullptr, nullptr);
  size_t bytes = 0;
  ANEMOI_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      nullptr, bytes, keys, values, static_cast<int>(items), 0,
      kProbabilityBits + required_unsigned_bits(static_cast<int>(segments)), stream));
  ANEMOI_CHECK(bytes <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
               "route workspace exceeds tensor range");
  return static_cast<int64_t>(bytes);
}
