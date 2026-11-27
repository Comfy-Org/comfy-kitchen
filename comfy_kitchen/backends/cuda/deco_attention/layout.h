// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>

namespace comfy_deco::layout {

// Each 128-token tile holds 3 packed K fragments and 2 packed V fragments.
// Q uses the same three fragments stored contiguously within each token/head.
constexpr int fragment_bytes = 4096;
constexpr int scale_fragment_bytes = 512;
constexpr int query_row_bytes = 96;
constexpr int query_scale_bytes = 3 * scale_fragment_bytes;
constexpr int key_scale_offset = 3 * fragment_bytes;
constexpr int value_offset = key_scale_offset + query_scale_bytes;
constexpr int value_scale_offset = value_offset + 2 * fragment_bytes;
constexpr int mean_offset = value_scale_offset + 2 * scale_fragment_bytes;
constexpr int mean_bytes = 4096;
constexpr int tile_bytes = mean_offset + mean_bytes;
constexpr int scale_side_bytes = 832;

// Each split pair can publish two records, each containing 64 FP32
// accumulators, two sums and two anchors per thread, packed as 17 uint4s.
constexpr int threads = 256;
constexpr int partial_vectors = 17;
constexpr std::size_t split_pair_bytes = 2 * threads * partial_vectors * 16;

} // namespace comfy_deco::layout
