// SPDX-License-Identifier: Apache-2.0
#include "draft_dispatch.h"
#include <cuda.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <stdexcept>
#include <nanobind/stl/optional.h>

namespace nb = nanobind;
namespace {
using namespace comfy::draft;
using draft_native::check;
using Array = nb::ndarray<nb::device::cuda>;

struct Plan {
  Config config;
  size_t workspace_bytes;
  bool native_prefix;
};

// CMake fills COMFY_DRAFT_SM89_ARCHS with the comma-separated capability
// numbers the Ada kernel set was built for (for example "89,90,100"); a device
// is served only when its exact number is present, so JIT of generic PTX to
// unlisted architectures stays unavailable.
#if defined(COMFY_DRAFT_SM89) && !defined(COMFY_DRAFT_SM89_ARCHS)
#define COMFY_DRAFT_SM89_ARCHS "89"
#endif

bool compiled(int architecture) {
#ifdef COMFY_DRAFT_SM89_ARCHS
  for (const char *p = COMFY_DRAFT_SM89_ARCHS; *p;) {
    int value = 0;
    while (*p >= '0' && *p <= '9')
      value = value * 10 + (*p++ - '0');
    if (value == architecture)
      return true;
    while (*p && *p != ',')
      ++p;
    if (*p == ',')
      ++p;
  }
#endif
#ifdef COMFY_DRAFT_SM120
  if (architecture == 120) return true;
#endif
  return false;
}
size_t workspace_bytes(const Config &c) {
#ifdef COMFY_DRAFT_SM89
  if (ada_serves(c.architecture)) return workspace_sm89(c);
#endif
#ifdef COMFY_DRAFT_SM120
  if (c.architecture == 120) return workspace_sm120(c);
#endif
  throw std::invalid_argument("Draft architecture was not compiled");
}

DType scalar(nb::dlpack::dtype d) {
  check(d.lanes == 1, "vector dtypes unsupported");
  if (d.code == 2 && d.bits == 16) return DType::Half;
  if (d.code == 4 && d.bits == 16) return DType::BFloat16;
  if (d.code == 0 && d.bits == 32) return DType::Int;
  if (d.code == 0 && d.bits == 64) return DType::Long;
  if (d.code == 1 && d.bits == 8) return DType::Byte;
  if (d.code == 6 && d.bits == 8) return DType::Bool;
  throw std::invalid_argument("unsupported Draft boundary dtype");
}
Tensor tensor(const Array &a, const Config &c, const char *what, bool indexed = true) {
  Tensor t;
  t.pointer = a.data();
  t.present = true;
  t.metadata = {scalar(a.dtype()), a.device_id()};
  check(a.device_id() == c.device && a.ndim() <= 4, what, ": tensor device/rank mismatch");
  size_t span = 0;
  for (size_t i = 0; i < a.ndim(); ++i) {
    check(a.shape(i) <= size_t(indexed ? INT32_MAX : INT64_MAX) && a.stride(i) >= 0,
          what, ": invalid tensor shape/stride");
    t.shape.push_back(a.shape(i));
    t.steps.push_back(a.stride(i));
    size_t extent = a.shape(i) ? a.shape(i) - 1 : 0;
    check(!extent || size_t(a.stride(i)) <= (SIZE_MAX - span) / extent,
          what, ": tensor span overflow");
    span += extent * size_t(a.stride(i));
  }
  check(!indexed || t.numel() <= INT32_MAX, what, ": tensor exceeds native indexing range");
  if (t.numel()) {
    size_t width = draft_native::itemsize(t.scalar_type());
    check(span < SIZE_MAX / width, what, ": tensor byte span overflow");
    cudaPointerAttributes attributes{};
    draft_native::cuda_check(cudaPointerGetAttributes(&attributes, t.pointer));
    check(attributes.type == cudaMemoryTypeDevice && attributes.device == c.device,
          what, ": tensor does not reference device memory on the selected GPU");
    // No allocation-capacity probe here: cuMemGetAddressRange reports only the
    // mapped chunk under VMM allocators (expandable_segments), which can be far
    // smaller than the tensor, so it cannot bound a view. PyTorch already
    // guarantees that a tensor's elements stay inside its storage.
  }
  return t;
}
void expect(const Tensor &t, draft_native::IntArrayRef shape, DType dtype,
            bool contiguous = true) {
  check(t.shape == shape && t.scalar_type() == dtype && (!contiguous || t.is_contiguous()),
        "Draft tensor shape, dtype or layout does not match plan");
}

Plan plan(int device, int batch, int tokens, int heads, int dim, int block,
          int prefix, int video_blocks, int anchor_count, bool bfloat16,
          double sparsity, std::array<double, 4> ratios, int prefix_kv,
          int prefix_query, int draft_proxy, bool jensen, double maxpool_weight,
          bool anchors, bool smooth_k, std::array<float, 3> global_scales) {
  draft_native::cuda::CUDAGuard guard(device);
  const cudaDeviceProp &properties = *draft_native::cuda::deviceProperties(device);
  Config c;
  c.device = device;
  c.architecture = properties.major * 10 + properties.minor;
  c.batch = batch; c.tokens = tokens; c.heads = heads; c.dim = dim;
  c.query_block = block; c.prefix_tokens = prefix; c.video_blocks = video_blocks;
  c.anchor_count = anchor_count;
  c.input_type = bfloat16 ? DType::BFloat16 : DType::Half;
  c.sparsity = sparsity; c.ratios = ratios;
  c.prefix_kv = prefix_kv; c.prefix_query = prefix_query;
  c.draft_proxy = draft_proxy; c.jensen = jensen; c.maxpool_weight = maxpool_weight;
  c.anchors = anchors; c.smooth_k = smooth_k; c.global_scales = global_scales;
  validate(c);
  check(compiled(c.architecture), "Draft architecture was not compiled");
  bool native_prefix = prefix && resolved_prefix_query(c) == 1 &&
                       (ada_serves(c.architecture) || c.ratios[1] > 0);
  return {c, workspace_bytes(c), native_prefix};
}

void launch(const Plan &p, Array q, Array k, Array v, Array indices, Array valid,
            Array counts, Array inverse, std::optional<Array> anchors,
            std::optional<Array> anchor_ids, Array prefix_output, Array output,
            Array workspace, uintptr_t stream_value) {
  const auto &c = p.config;
  draft_native::cuda::CUDAGuard guard(c.device);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_value);
  if (stream) {
    int stream_device;
    draft_native::cuda_check(cudaStreamGetDevice(stream, &stream_device));
    check(stream_device == c.device, "Draft stream belongs to another device");
  }
  Inputs in;
  in.q = tensor(q, c, "q"); in.k = tensor(k, c, "k"); in.v = tensor(v, c, "v");
  for (auto *t : {&in.q, &in.k, &in.v}) {
    expect(*t, {c.batch, c.tokens, c.heads, c.dim}, c.input_type, false);
    check(uintptr_t(t->pointer) % 16 == 0, "Q/K/V must be 16-byte aligned");
    Tensor bhsd = bthd_to_bhsd(*t);
    check(t->is_contiguous() || bhsd.is_contiguous(),
          "Q/K/V storage must be contiguous BTHD or BHSD");
    *t = std::move(bhsd);
  }
  in.indices = tensor(indices, c, "indices"); in.slot_valid = tensor(valid, c, "slot_valid");
  in.counts = tensor(counts, c, "counts"); in.inverse = tensor(inverse, c, "inverse");
  expect(in.indices, {int64_t(c.video_blocks) * c.query_block}, DType::Long);
  expect(in.slot_valid, in.indices.shape, DType::Bool);
  expect(in.counts, {c.video_blocks}, DType::Int);
  expect(in.inverse, {c.tokens - c.prefix_tokens}, DType::Long);
  check(bool(anchors) == c.anchors && bool(anchor_ids) == c.anchors,
        "anchor tensors do not match plan");
  if (c.anchors) {
    in.anchors = tensor(*anchors, c, "anchors"); in.anchor_ids = tensor(*anchor_ids, c, "anchor_ids");
    expect(in.anchors, {c.video_blocks, c.video_blocks}, DType::Bool);
    expect(in.anchor_ids, {c.anchor_count}, DType::Int);
  }
  in.prefix_output = tensor(prefix_output, c, "prefix_output");
  if (!p.native_prefix)
    expect(in.prefix_output, {c.batch, c.heads, c.prefix_tokens, c.dim}, c.input_type, false);
  else
    check(!in.prefix_output.numel(), "native prefix must not receive dense prefix output");
  in.output = tensor(output, c, "output");
  expect(in.output, {c.batch, c.tokens, c.heads, c.dim}, c.input_type);
  Tensor storage = tensor(workspace, c, "workspace", false);
  check(storage.scalar_type() == DType::Byte && storage.dim() == 1 &&
            storage.is_contiguous() && size_t(storage.numel()) >= p.workspace_bytes &&
            uintptr_t(storage.pointer) % 256 == 0, "invalid Draft workspace");
  // Inputs may alias each other (self-attention); writable storage must not.
  auto overlap = [](const Tensor &a, const Tensor &b) {
    if (!a.numel() || !b.numel()) return false;
    auto end = [](const Tensor &t) {
      size_t span = 1;
      for (size_t i = 0; i < t.shape.size(); ++i)
        span += size_t(t.shape[i] - 1) * size_t(t.steps[i]);
      return uintptr_t(t.pointer) + span * draft_native::itemsize(t.scalar_type());
    };
    return uintptr_t(a.pointer) < end(b) && uintptr_t(b.pointer) < end(a);
  };
  for (const Tensor *t : {&in.q, &in.k, &in.v, &in.indices, &in.slot_valid,
                          &in.counts, &in.inverse, &in.anchors, &in.anchor_ids,
                          &in.prefix_output}) {
    check(!overlap(*t, storage) && !overlap(*t, in.output), "Draft writable storage aliases input");
  }
  check(!overlap(storage, in.output), "output aliases workspace");
  nb::gil_scoped_release release;
#ifdef COMFY_DRAFT_SM89
  if (ada_serves(c.architecture)) {
    execute_sm89(c, in, storage.pointer, p.workspace_bytes, stream);
    return;
  }
#endif
#ifdef COMFY_DRAFT_SM120
  if (c.architecture == 120) {
    execute_sm120(c, in, storage.pointer, p.workspace_bytes, stream);
    return;
  }
#endif
  throw std::invalid_argument("Draft architecture was not compiled");
}
} // namespace

void register_draft(nb::module_ &m) {
  nb::class_<Plan>(m, "_DraftPlan")
      .def_ro("workspace_bytes", &Plan::workspace_bytes)
      .def_ro("native_prefix", &Plan::native_prefix);
  m.def("draft_supports_arch", &compiled);
  m.def("draft_plan", &plan, nb::arg("device"), nb::arg("batch"), nb::arg("tokens"),
        nb::arg("heads"), nb::arg("dim"), nb::arg("query_block"), nb::arg("prefix_tokens"),
        nb::arg("video_blocks"), nb::arg("anchor_count"), nb::arg("bfloat16"),
        nb::arg("sparsity"), nb::arg("ratios"), nb::arg("prefix_kv"),
        nb::arg("prefix_query"), nb::arg("draft_proxy"), nb::arg("jensen"),
        nb::arg("maxpool_weight"), nb::arg("anchors"), nb::arg("smooth_k"), nb::arg("global_scales"));
  m.def("draft", &launch, nb::arg("plan"), nb::arg("q"), nb::arg("k"), nb::arg("v"),
        nb::arg("indices"), nb::arg("slot_valid"), nb::arg("counts"), nb::arg("inverse"),
        nb::arg("anchors").none(), nb::arg("anchor_ids").none(), nb::arg("prefix_output"),
        nb::arg("output"), nb::arg("workspace"), nb::arg("stream"));
}
