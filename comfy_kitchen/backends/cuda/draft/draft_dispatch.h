// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "include/native_tensor.h"
#include <array>
#include <cstddef>
#include <cmath>

namespace comfy::draft {
using Tensor = draft_native::Tensor;
using DType = draft_native::ScalarType;

// Geometry and algorithm parameters only; no model or framework state.
struct Config {
  int device = 0;
  int architecture = 89;
  int batch = 1;
  int tokens = 0;
  int heads = 0;
  int dim = 128;
  int query_block = 64;
  int prefix_tokens = 0;
  int video_blocks = 0;
  int anchor_count = 0;
  DType input_type = DType::Half;
  double sparsity = 0.8;
  // NVFP4, INT8/E4M3, MXFP8, FP16.
  std::array<double, 4> ratios{0.0, 1.0, 0.0, 0.0};
  // -1 auto; otherwise the precision index above.
  int prefix_kv = -1;
  int prefix_query = -1;
  // 0 mean, 1 K-tail R1, 2 K-tail R2.
  int draft_proxy = 0;
  bool jensen = false;
  double maxpool_weight = 0.0;
  bool anchors = false;
  bool smooth_k = false;
  std::array<float, 3> global_scales{1.0f, 1.0f, 1.0f};
};

struct Inputs {
  // Q/K/V are logical BHSD views of public BTHD tensors.
  Tensor q, k, v;
  Tensor indices, slot_valid, counts, inverse;
  Tensor anchors, anchor_ids;
  // Original-dtype dense SDPA prefix, or typed empty when native INT8 is used.
  Tensor prefix_output;
  // Caller-owned contiguous BTHD output; never allocated by native code.
  Tensor output;
};

struct Region {
  size_t offset = 0;
  Tensor metadata;
};

// One aligned workspace. The same native recipe determines every offset during
// query and launch; no Python-owned list of intermediate tensors is exposed.
struct Workspace {
  size_t bytes = 0;
  Region add(draft_native::IntArrayRef shape, DType dtype, int device);
};
Tensor view(const Region &region, void *workspace);
Tensor empty_view(DType dtype, int device);
Tensor bthd_to_bhsd(Tensor tensor);
int resolved_prefix_query(const Config &config);
int resolved_prefix_kv(const Config &config);
void validate(const Config &config);

// The Ada (SM89) mma.sync kernel set serves every capability in [89, 120).
// SM120 and newer Blackwell-capability devices use the SM120a kernels.
inline bool ada_serves(int architecture) {
  return architecture >= 89 && architecture < 120;
}

// Hamilton allocation with FP16/middle/NVFP4 tie order, shared by both GPUs.
inline std::array<int64_t, 3> route_budgets(const Config &c) {
  const int64_t items = int64_t(c.video_blocks) * c.video_blocks;
  const int64_t retained = std::min(items, std::max<int64_t>(
      1, int64_t(std::floor((1.0 - c.sparsity) * items + 0.5))));
  const std::array<double, 3> quota{retained * c.ratios[3],
      retained * (c.ratios[1] + c.ratios[2]), retained * c.ratios[0]};
  std::array<int64_t, 3> counts{};
  for (int i = 0; i < 3; ++i) counts[i] = int64_t(std::floor(quota[i]));
  std::array<int, 3> order{0, 1, 2};
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return quota[a] - counts[a] > quota[b] - counts[b];
  });
  const int64_t remaining = retained - counts[0] - counts[1] - counts[2];
  // Preserve the reference's order[:remaining], including its negative-slice
  // behavior for accepted ratios whose sum is slightly greater than one.
  const int awards = int(std::clamp<int64_t>(remaining < 0 ? 3 + remaining : remaining, 0, 3));
  for (int i = 0; i < awards; ++i) ++counts[order[i]];
  return counts;
}

// Native shared peripheral kernels (implemented in draft_preprocess.cu).
void valid_key_counts(const Config &, const Tensor &logical_counts,
                      const Tensor &physical_counts, cudaStream_t);
void moments(const Tensor &packed, const Tensor &counts, int query_block,
             const Tensor &second, cudaStream_t);
void pool(const Tensor &packed, const Tensor &counts, int query_block,
          const Tensor &mean, const Tensor &maximum, cudaStream_t);
void fill_scales(const Config &, const Tensor &q_scale, const Tensor &k_scale,
                 const Tensor &v_scale, cudaStream_t);
void pack_h3(const Config &, const Inputs &, const Tensor &q_packed,
             const Tensor &k_packed, const Tensor &v_packed, cudaStream_t);
void assemble_h3(const Config &, const Inputs &, const Tensor &prefix_bhsd,
                 const Tensor &video_bhsd, const Tensor &low_counts,
                 const Tensor &middle_counts, const Tensor &high_counts,
                 cudaStream_t);

// FP32 matmul + scaling + softmax matches the previous Python mean/Jensen math.
// All scratch is supplied from the caller's one workspace.
size_t draft_workspace_bytes(int batch, int heads, int blocks, int dim, bool jensen);
void draft_probability(const Tensor &q_pool, const Tensor &k_pool,
                       const Tensor &q_second, const Tensor &k_second,
                       const Tensor &probability, void *scratch,
                       size_t scratch_bytes, cudaStream_t);

// Architecture specializations behind one native dispatch boundary. These are
// private C++ entry points, not separate Python APIs.
size_t workspace_sm89(const Config &);
void execute_sm89(const Config &, const Inputs &, void *, size_t, cudaStream_t);
size_t workspace_sm120(const Config &);
void execute_sm120(const Config &, const Inputs &, void *, size_t, cudaStream_t);

} // namespace comfy::draft
