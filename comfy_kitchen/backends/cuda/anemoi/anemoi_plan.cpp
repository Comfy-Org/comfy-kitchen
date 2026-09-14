// SPDX-License-Identifier: Apache-2.0
#include "anemoi_dispatch.h"
#include "include/api.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace comfy::anemoi {
using anemoi_native::check;

Region Workspace::add(anemoi_native::IntArrayRef shape, DType dtype, int device) {
  Tensor t;
  t.shape = std::move(shape);
  t.metadata = {dtype, device};
  t.present = true;
  t.steps.resize(t.shape.size());
  int64_t stride = 1;
  for (int i = int(t.shape.size()) - 1; i >= 0; --i) {
    check(t.shape[i] >= 0, "negative workspace extent");
    t.steps[i] = stride;
    check(stride <= INT64_MAX / std::max<int64_t>(1, t.shape[i]),
          "workspace shape overflow");
    stride *= t.shape[i];
  }
  check(t.numel() <= INT32_MAX, "workspace tensor exceeds native indexing range");
  size_t length = size_t(t.numel()) * anemoi_native::itemsize(dtype);
  check(bytes <= SIZE_MAX - 255, "workspace offset overflow");
  size_t offset = (bytes + 255) & ~size_t(255);
  check(length <= SIZE_MAX - offset, "workspace size overflow");
  bytes = offset + length;
  return {offset, std::move(t)};
}

Tensor view(const Region &region, void *workspace) {
  auto t = region.metadata;
  t.pointer = workspace ? static_cast<char *>(workspace) + region.offset : nullptr;
  return t;
}
Tensor empty_view(DType dtype, int device) {
  Workspace w;
  return w.add({0}, dtype, device).metadata;
}
Tensor bthd_to_bhsd(Tensor t) {
  check(t.dim() == 4, "expected BTHD tensor");
  std::swap(t.shape[1], t.shape[2]);
  std::swap(t.steps[1], t.steps[2]);
  return t;
}

int resolved_prefix_query(const Config &c) {
  if (c.prefix_query >= 0)
    return c.prefix_query;
  return c.architecture == 120 && c.ratios[1] > 0 ? 1 : 3;
}
int resolved_prefix_kv(const Config &c) {
  if (c.prefix_kv >= 0)
    return c.prefix_kv;
  if (ada_serves(c.architecture) || c.ratios[3] > 0)
    return 3;
  for (int i = 0; i < 4; ++i)
    if (c.ratios[i] > 0)
      return i;
  throw std::invalid_argument("no active attention precision");
}

void validate(const Config &c) {
  check(ada_serves(c.architecture) || c.architecture == 120,
        "Anemoi requires SM89 or newer (SM120+ uses the Blackwell kernels)");
  check(c.device >= 0 && c.batch == 1 && c.tokens > 0 && c.heads > 0,
        "native Anemoi requires batch one and positive tensor dimensions");
  check(c.dim == 128 || c.dim == 64, "unsupported head dimension");
  check(c.query_block == 64 || c.query_block == 128, "query block must be 64 or 128");
  check(c.input_type == DType::Half || c.input_type == DType::BFloat16,
        "Anemoi input must be FP16 or BF16");
  check(c.prefix_tokens >= 0 && c.prefix_tokens < c.tokens && c.video_blocks > 0,
        "invalid prefix/video geometry");
  int64_t pairs = int64_t(c.video_blocks) * c.video_blocks;
  check(pairs <= INT32_MAX && pairs <= INT32_MAX / c.heads,
        "DraftMap exceeds native indexing range");
  check(c.anchor_count >= 0 && c.anchor_count <= pairs && (c.anchors || !c.anchor_count),
        "invalid anchor count");
  check(std::isfinite(c.sparsity) && c.sparsity >= 0 && c.sparsity < 1,
        "sparsity must be in [0, 1)");
  double sum = 0;
  for (auto r : c.ratios) {
    check(std::isfinite(r) && r >= 0, "precision ratios must be finite and nonnegative");
    sum += r;
  }
  check(std::abs(sum - 1) <= 1e-6, "precision ratios must sum to one");
  check(!(c.ratios[1] && c.ratios[2]), "INT8 and MXFP8 cannot share the middle phase");
  check(c.prefix_kv >= -1 && c.prefix_kv <= 3 && c.prefix_query >= -1 && c.prefix_query <= 3,
        "invalid prefix precision");
  if (ada_serves(c.architecture)) {
    check(!c.ratios[0] && !c.ratios[2], "the SM89 kernel set supports INT8/E4M3 and FP16 only");
    check((c.prefix_kv == -1 || c.prefix_kv == 1 || c.prefix_kv == 3) &&
              (c.prefix_query == -1 || c.prefix_query == 1 || c.prefix_query == 3),
          "SM89-set prefix precision must be auto, INT8 or FP16");
  } else {
    check(!c.smooth_k, "smooth_k is available on the SM89 kernel set only");
  }
  int prefix = resolved_prefix_kv(c);
  check(!c.prefix_tokens || !((prefix == 1 && c.ratios[2]) || (prefix == 2 && c.ratios[1])),
        "prefix and video cannot require both INT8 and MXFP8");
  check(c.draft_proxy >= 0 && c.draft_proxy <= 2, "invalid draft proxy");
  check(!c.draft_proxy || (c.architecture == 120 && c.query_block == 64 && !c.jensen),
        "K-tail requires SM120 Q64 without Jensen");
  check(std::isfinite(c.maxpool_weight) && c.maxpool_weight >= 0 && c.maxpool_weight <= 1,
        "maxpool weight must be in [0, 1]");
  check(!c.maxpool_weight || (!c.jensen && !c.draft_proxy),
        "maxpool cannot be combined with Jensen or K-tail");
  for (auto s : c.global_scales)
    check(std::isfinite(s) && s > 0, "NVFP4 scales must be finite and positive");
}

namespace {
struct Outputs {
  anemoi_native::ExecutionContext context;
  anemoi_native::ExecutionContext *previous;
  Outputs(int device, cudaStream_t stream, std::vector<Tensor> buffers)
      : previous(anemoi_native::active_context) {
    context.device = device;
    context.stream = stream;
    context.buffers = std::move(buffers);
    anemoi_native::active_context = &context;
  }
  ~Outputs() { anemoi_native::active_context = previous; }
};
}

void pack_h3(const Config &c, const Inputs &in, const Tensor &q,
             const Tensor &k, const Tensor &v, cudaStream_t stream) {
  Outputs outputs(c.device, stream, {q, k, v});
  pack_h3_k64_qkv_fp16(in.q, in.k, in.v, in.indices, in.slot_valid, c.prefix_tokens);
}
void assemble_h3(const Config &c, const Inputs &in, const Tensor &prefix,
                 const Tensor &video, const Tensor &low, const Tensor &middle,
                 const Tensor &high, cudaStream_t stream) {
  Outputs outputs(c.device, stream, {in.output});
  assemble_h3_k64_output(prefix, video, in.inverse, c.input_type,
                        low, middle, high, c.query_block);
}
} // namespace comfy::anemoi
